"""Outlier check for numeric columns.

Evaluates numeric columns using the Interquartile Range (IQR) method.
Flags columns where outliers make up a significant percentage of rows.

Returns a list of Finding objects (empty means no severe outliers found).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import info_note, register_check
from src.tools.target_type import is_classification_target

# Fewer distinct values than this make IQR outlier detection meaningless:
# the quartiles collapse and nearly every value would be flagged.
MIN_DISTINCT_VALUES = 3


@register_check(
    name="outliers",
    required=("csv_path",),
    severity_key="outliers",
    description="Check numeric columns for excessive outliers using IQR.",
    always=True,
    accepted_semantic=frozenset({"numeric_measurement", "numeric_discrete"}),
)
def outlier_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect numeric columns with excessive outliers using the IQR method.

    Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are outliers; columns with
    more than the configured share get MEDIUM/HIGH. The registry's semantic
    gating (numeric_measurement/numeric_discrete) pre-filters columns here.
    """
    findings: list[Finding] = []

    # Class labels (binary, boolean, multiclass) have no meaningful
    # outliers, so only the target column itself is excluded; eligible
    # feature columns are still analyzed. Continuous targets are analyzed
    # like any other column.
    target_is_classification = False
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        if len(target_series) > 0:
            target_is_classification = is_classification_target(target_series)

    degenerate_columns: list[str] = []

    for col in df.columns:
        series = df[col]

        # Type guard for direct callers; the registry gates this check to
        # numeric semantics normally.
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if col == target_column and target_is_classification:
            continue

        clean = series.dropna()

        if len(clean) < 4:
            # Too few rows for a meaningful IQR.
            continue

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        # Degenerate IQR: the bounds collapse to a single value (or there
        # are too few distinct values), so every other value would be
        # flagged as an outlier. Skip the column and report it in a note.
        if iqr == 0 or clean.nunique() < MIN_DISTINCT_VALUES:
            degenerate_columns.append(col)
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
        outlier_pct = float(outlier_count / len(clean) * 100)

        if outlier_pct > check_config.outliers.high_outlier_pct_threshold:
            severity = Severity.HIGH
        elif outlier_pct > check_config.outliers.outlier_pct_threshold:
            severity = Severity.MEDIUM
        else:
            continue

        evidence = (
            f"Column '{col}' has {outlier_count} outlier(s) "
            f"({outlier_pct:.2f}% of {len(clean)} non-null rows) "
            f"outside [{lower_bound:.4f}, {upper_bound:.4f}] "
            f"(IQR method)."
        )
        if col == target_column:
            evidence += (
                " This is the target column: review these rows before "
                "removing them; consider a log transform or a robust loss."
            )

        findings.append(
            Finding(
                check_name="outliers",
                severity=severity,
                evidence=evidence,
                affected_columns=[col],
                confidence=0.9,
            )
        )

    if degenerate_columns:
        findings.insert(
            0,
            info_note(
                "outliers",
                (
                    f"Skipped {len(degenerate_columns)} column(s) in the "
                    f"outliers check: too few distinct values for IQR "
                    f"outlier detection: {_cap_column_names(degenerate_columns)}."
                ),
            ),
        )

    return findings


def _cap_column_names(cols: list[str], limit: int = 6) -> str:
    """Cap a column list with ``and N more`` when it exceeds the limit."""
    if len(cols) <= limit:
        return ", ".join(cols)
    return f"{', '.join(cols[:limit])} and {len(cols) - limit} more"


__all__ = ["outlier_check"]
