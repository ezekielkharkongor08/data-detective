"""Fairness check: compare favorable outcome rates across sensitive groups.

Design rules
------------
- **User-specified columns only.** The check never picks a sensitive column
  on its own. A short column-name hint list is used only inside the skip note
  to suggest candidates; hinted columns are not analysed.
- **Binary targets only.** Fairness rates need a clear favorable outcome.
  Multiclass and continuous targets are skipped with one informational note.
- **No model training.** The check compares observed outcome rates. It does
  not train a classifier or use cross-validation.
- **Every skip is explicit.** Missing inputs, ambiguous favorable values,
  groups below ``check_config.bias.min_group_size``, and non-significant differences all return
  an informational note instead of silently doing nothing.

Statistics
----------
For each sensitive column, eligible groups (at least ``check_config.bias.min_group_size``
non-missing rows) are compared:

- **Favorable rate**: share of the group whose target equals the favorable
  value.
- **Rate difference**: reference group rate minus the worst-off group rate,
  in percentage points.
- **Disparate impact (DI)**: worst-off rate divided by the reference rate.
  The reference is the group with the highest favorable rate.
- **Chi-square p-value**: computed with ``scipy.stats.chi2_contingency`` on
  the group-by-outcome contingency table. A finding needs ``p < check_config.bias.p_value_limit``
  so a small-sample fluke is not reported as a real disparity.

Severity is based on the disparate impact ratio:

- ``DI < 0.4`` -> CRITICAL
- ``DI < 0.6`` -> HIGH
- ``DI < 0.8`` -> MEDIUM
- ``DI >= 0.8`` -> no finding
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from scipy.stats import chi2_contingency

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import info_note, register_check
from src.tools.target_type import detect_target_type

# Fairness thresholds live in the shared config (check_config.bias).
# Generic column-name hints. These are *hints only*: the check never selects
# a sensitive column from this list. It uses the list to suggest columns in
# the "no sensitive columns specified" note.
SENSITIVE_NAME_HINTS = (
    "sex",
    "gender",
    "race",
    "ethnic",
    "age",
    "religion",
    "marital",
    "disabilit",
    "orientation",
    "nationalit",
    "country",
    "veteran",
    "pregnan",
)


def _skip(message: str) -> list[Finding]:
    """Return a single informational note with the fairness check name."""
    return [info_note("bias", message)]


def _hint_candidates(df: pd.DataFrame, target_column: Optional[str]) -> list[str]:
    """Return column names that *look* sensitive, for the skip note only.

    Matching is token-based so ``language`` does not match ``age``.
    """
    candidates: list[str] = []
    for column in df.columns:
        if column == target_column:
            continue
        tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", str(column).lower())
            if token
        ]
        if any(
            token.startswith(hint)
            for token in tokens
            for hint in SENSITIVE_NAME_HINTS
        ):
            candidates.append(str(column))
    return candidates


def _favorable_mask(series: pd.Series, favorable_value: object) -> pd.Series:
    """Return a boolean mask for rows whose target is the favorable outcome."""
    if pd.api.types.is_bool_dtype(series):
        if isinstance(favorable_value, str):
            favorable = favorable_value.strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
                "t",
            }
        else:
            favorable = bool(favorable_value)
        return series.eq(favorable).fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        try:
            favorable_number = float(favorable_value)
            return series.eq(favorable_number).fillna(False).astype(bool)
        except (TypeError, ValueError):
            pass

    return (
        series.astype(str)
        .str.strip()
        .eq(str(favorable_value).strip())
        .fillna(False)
        .astype(bool)
    )


def _resolve_favorable_value(
    target: pd.Series,
    favorable_value: object | None,
) -> tuple[object | None, bool, str | None]:
    """Decide which target value counts as the favorable outcome.

    Returns ``(value, inferred, error)``.

    - A user-provided value is validated against the target values.
    - A boolean target defaults to ``True`` and a 0/1 numeric target to ``1``.
    - Anything else is ambiguous, so the check skips with a message instead of
      guessing semantics (for example, whether "Yes" churn is favorable).
    """
    if favorable_value is not None and str(favorable_value).strip() != "":
        mask = _favorable_mask(target.dropna(), favorable_value)
        if not bool(mask.any()):
            return (
                None,
                False,
                f"Skipped: favorable_value '{favorable_value}' does not "
                f"appear in target '{target.name}'.",
            )
        return favorable_value, False, None

    if pd.api.types.is_bool_dtype(target):
        return True, True, None

    if pd.api.types.is_numeric_dtype(target):
        unique_values = set(target.dropna().unique().tolist())
        if unique_values == {0, 1}:
            return 1, True, None

    values = sorted({str(value) for value in target.dropna().unique().tolist()})[:5]
    return (
        None,
        False,
        "Skipped: no favorable_value was provided and the target values are "
        f"ambiguous ({', '.join(values)}). Pass favorable_value (for example "
        "'Yes' or '>50K') so fairness rates have a clear positive outcome.",
    )


def _group_stats(
    df: pd.DataFrame,
    sensitive_column: str,
    target_column: str,
    favorable_value: object,
) -> tuple[list[dict], int, int]:
    """Compute per-group favorable counts and rates.

    Returns ``(eligible_stats, dropped_rows, dropped_small_groups)``.
    Groups smaller than ``check_config.bias.min_group_size`` are left out.
    """
    clean = df[[sensitive_column, target_column]].dropna()
    dropped_rows = len(df) - len(clean)
    if clean.empty:
        return [], dropped_rows, 0

    clean = clean.assign(
        _favorable=_favorable_mask(clean[target_column], favorable_value)
    )
    grouped = clean.groupby(sensitive_column, observed=True)["_favorable"].agg(
        ["count", "sum"]
    )

    stats: list[dict] = []
    for key, row in grouped.iterrows():
        n_rows = int(row["count"])
        if n_rows < check_config.bias.min_group_size:
            continue
        favorable_count = int(row["sum"])
        stats.append(
            {
                "group": str(key),
                "n": n_rows,
                "favorable": favorable_count,
                "rate": favorable_count / n_rows,
            }
        )

    dropped_small_groups = len(grouped) - len(stats)
    return stats, dropped_rows, dropped_small_groups


def _check_one_column(
    df: pd.DataFrame,
    sensitive_column: str,
    target_column: str,
    favorable_value: object,
    inferred_favorable: bool,
) -> list[Finding]:
    """Run the fairness comparison for one sensitive column."""
    try:
        stats, dropped_rows, dropped_small_groups = _group_stats(
            df, sensitive_column, target_column, favorable_value
        )
    except Exception as exc:  # noqa: BLE001 - odd column types must not crash
        return _skip(
            f"Skipped '{sensitive_column}': could not group the data "
            f"({type(exc).__name__}: {exc})."
        )

    if len(stats) < 2:
        return _skip(
            f"Skipped '{sensitive_column}': need at least two groups with at "
            f"least {check_config.bias.min_group_size} non-missing rows; found {len(stats)}."
        )

    # Reference = highest favorable rate; worst-off = lowest. Ties are broken
    # by group name so repeated runs produce the same report.
    ordered = sorted(stats, key=lambda item: (-item["rate"], item["group"]))
    reference = ordered[0]
    worst = ordered[-1]

    if reference["rate"] <= 0:
        return _skip(
            f"Skipped '{sensitive_column}': all eligible groups have a 0% "
            "favorable rate, so disparate impact is undefined."
        )

    disparate_impact = worst["rate"] / reference["rate"]
    rate_difference = (reference["rate"] - worst["rate"]) * 100

    # No material disparity: nothing to report.
    if disparate_impact >= check_config.bias.di_medium:
        return []

    contingency = [[item["favorable"], item["n"] - item["favorable"]] for item in ordered]
    try:
        _, p_value, _, expected = chi2_contingency(contingency)
    except Exception as exc:  # noqa: BLE001 - scipy validates the table
        return _skip(
            f"Skipped '{sensitive_column}': could not compute the chi-square "
            f"p-value ({type(exc).__name__}: {exc})."
        )

    # A visible difference in a small sample is not enough evidence.
    if p_value >= check_config.bias.p_value_limit:
        return [
            info_note(
                "bias",
                f"Not flagged: observed an apparent disparity for "
                f"'{sensitive_column}' (disparate impact "
                f"{disparate_impact:.3f}, rate difference "
                f"{rate_difference:.2f} percentage points), but the "
                f"chi-square p-value {p_value:.3e} is not below "
                f"{check_config.bias.p_value_limit}. With this sample size the difference is "
                "not statistically significant.",
            )
        ]

    if disparate_impact < check_config.bias.di_critical:
        severity = Severity.CRITICAL
    elif disparate_impact < check_config.bias.di_high:
        severity = Severity.HIGH
    else:
        severity = Severity.MEDIUM

    group_text = "; ".join(
        f"{item['group']}: {item['rate'] * 100:.2f}% (n={item['n']})"
        for item in ordered
    )

    notes: list[str] = []
    if dropped_rows:
        notes.append(
            f"Dropped {dropped_rows} row(s) with missing target or "
            "sensitive values."
        )
    if dropped_small_groups:
        notes.append(
            f"Dropped {dropped_small_groups} group(s) smaller than "
            f"{check_config.bias.min_group_size}."
        )
    if inferred_favorable:
        notes.append(
            "Favorable outcome was inferred because no favorable_value was "
            "provided."
        )
    if (expected < 5).any():
        notes.append(
            "Some expected cell counts are below 5; the p-value is "
            "approximate."
        )
    note_text = (" " + " ".join(notes)) if notes else ""

    evidence = (
        f"Fairness check for sensitive column '{sensitive_column}' "
        f"(target '{target_column}', favorable outcome '{favorable_value}'): "
        f"group rates: {group_text}. "
        f"Reference group (highest rate): '{reference['group']}' at "
        f"{reference['rate'] * 100:.2f}%; worst-off group: "
        f"'{worst['group']}' at {worst['rate'] * 100:.2f}%. "
        f"Rate difference: {rate_difference:.2f} percentage points. "
        f"Disparate impact ratio: {disparate_impact:.3f}. "
        f"Chi-square p-value: {p_value:.3e} (threshold {check_config.bias.p_value_limit}). "
        f"Minimum group size: {check_config.bias.min_group_size}."
        f"{note_text}"
    )

    return [
        Finding(
            check_name="bias",
            severity=severity,
            evidence=evidence,
            affected_columns=[str(sensitive_column)],
            confidence=0.9,
        )
    ]


@register_check(
    name="bias",
    required=("csv_path", "target_column"),
    optional=("sensitive_columns", "favorable_value"),
    target_types=("binary", "boolean"),
    severity_key="fairness",
    description=(
        "Fairness check: compares favorable outcome rates across "
        "user-specified sensitive groups."
    ),
    # Fairness must never invent sensitive columns in the fallback.
    fallback=False,
    param_types=(
        ("sensitive_columns", list[str]),
        ("favorable_value", str),
    ),
)
def bias_check(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    sensitive_columns: Optional[list[str]] = None,
    favorable_value: object | None = None,
) -> list[Finding]:
    """Compare favorable outcome rates across user-specified sensitive groups.

    Args:
        df: The pre-loaded DataFrame.
        target_column: Binary or boolean target column.
        sensitive_columns: Sensitive columns to compare. Required; the check
            never selects columns on its own.
        favorable_value: Value of the target that counts as a favorable
            outcome. Optional for boolean targets and 0/1 numeric targets,
            where it is inferred.

    Returns:
        Fairness findings, or informational notes for every skip.
    """
    if not target_column:
        return _skip(
            "Skipped: no target_column was provided; fairness needs a binary "
            "target and a favorable outcome."
        )

    if target_column not in df.columns:
        return _skip(
            f"Skipped: target column '{target_column}' was not found in the "
            "CSV."
        )

    if df.empty:
        return _skip("Skipped: the dataset is empty.")

    target = df[target_column]
    target_type = detect_target_type(target)
    if target_type not in ("binary", "boolean"):
        return _skip(
            f"Skipped: fairness supports binary or boolean targets only; "
            f"target '{target_column}' is {target_type}."
        )

    if not sensitive_columns:
        hints = _hint_candidates(df, target_column)
        hint_text = (
            f" Name-based hints (not analysed): {', '.join(hints)}."
            if hints
            else " No name-based hints were found."
        )
        return _skip(
            "Skipped: no sensitive columns were specified. Fairness only runs "
            "on user-specified columns."
            f"{hint_text} Pass sensitive_columns via check_params['bias'] or "
            "--sensitive."
        )

    if isinstance(sensitive_columns, str):
        sensitive_columns = [sensitive_columns]

    requested: list[str] = []
    for column in sensitive_columns:
        if column not in requested:
            requested.append(column)

    findings: list[Finding] = []
    valid = [column for column in requested if column in df.columns]
    missing = [column for column in requested if column not in df.columns]
    if missing:
        findings.append(
            info_note(
                "bias",
                "Skipped: sensitive column(s) not found: "
                f"{', '.join(str(column) for column in missing)}.",
            )
        )

    if not valid:
        return findings or _skip("Skipped: no valid sensitive columns were provided.")

    favorable, inferred, error = _resolve_favorable_value(target, favorable_value)
    if error:
        findings.append(info_note("bias", error))
        return findings

    for column in valid:
        if column == target_column:
            findings.append(
                info_note(
                    "bias",
                    f"Skipped '{column}': the sensitive column is the target "
                    "column.",
                )
            )
            continue
        findings.extend(
            _check_one_column(
                df,
                column,
                target_column,
                favorable,
                inferred,
            )
        )

    return findings


__all__ = [
    "bias_check",
    "SENSITIVE_NAME_HINTS",
]
