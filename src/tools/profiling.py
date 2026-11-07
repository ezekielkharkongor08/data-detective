"""Dataset profiling.

A plain pandas function — no LLM, no LangGraph. Loads a CSV and returns a
compact summary: column names, dtypes, missing counts, duplicates, basic
stats, and a few sample rows, plus a per-column ``semantic`` classification
so checks never run on identifiers, constants, or integer-coded categoricals.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.check_config import check_config
from src.tools.csv_reader import read_csv_robust
from src.tools.missing_markers import count_missing
from src.tools.target_type import detect_target_type

# Stable set of semantic type names. The CLI override is validated against it.
SEMANTIC_TYPES = (
    "numeric_measurement",
    "numeric_discrete",
    "binary",
    "categorical",
    "ordinal_candidate",
    "datetime",
    "identifier",
    "text",
    "constant",
    "unknown",
)

# Name hints are tie-breakers only (rule 6). They never create high confidence
# alone and never override a value-based rule. Ordinal hints deliberately do
# NOT include "num".
ORDINAL_NAME_HINTS = ("level", "grade", "stage", "rank", "rating", "tier", "class")
IDENTIFIER_NAME_HINTS = ("id", "uuid", "key", "index")
COUNT_NAME_HINTS = ("count", "qty", "quantity", "total", "frequency", "freq", "size", "records")

BOOLEAN_SPELLINGS = {"yes", "no", "y", "n", "true", "false", "t", "f", "0", "1"}
DATE_SEPARATORS = ("-", "/", ".")

# Missing-value markers are treated as unparseable (never as numbers/dates).
MISSING_SPELLINGS = frozenset({"?", "NA", "N/A", "null", "NULL", "", " ", "-", "--"})


def _semantic(type_: str, confidence: float, ambiguous: bool, reasons: list[str]) -> dict[str, Any]:
    """Build one ``semantic`` sub-dict (at most 2 reasons)."""
    return {
        "type": type_,
        "confidence": round(float(confidence), 2),
        "ambiguous": bool(ambiguous),
        "reasons": list(reasons)[:2],
    }


def _conf(
    type_: str,
    base: float,
    ambiguous: bool,
    reasons: list[str],
    n_eff: int,
    min_rows: int,
    parse_ratio: float | None = None,
) -> dict[str, Any]:
    """Confidence = per-rule base, capped by sample size, reduced by parse ratio.

    - base: the rule's intrinsic confidence (documented per rule).
    - sample cap: min(1.0, n_eff / min_rows) so small samples are never
      over-confident.
    - parse ratio: multiplied in when below 0.99 (rule 7).
    """
    sample_cap = min(1.0, n_eff / float(min_rows)) if min_rows else 1.0
    confidence = base * sample_cap
    if parse_ratio is not None and parse_ratio < 0.99:
        confidence *= parse_ratio
    confidence = max(0.0, min(1.0, confidence))
    return _semantic(type_, confidence, ambiguous, reasons)


def _sample(series: pd.Series, size: int) -> pd.Series:
    """Deterministic sample (fixed seed) used by expensive parse/pairwise tests."""
    clean = series.dropna()
    if len(clean) == 0 or len(clean) <= size:
        return clean
    return clean.sample(n=size, random_state=42)


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """Parse a series as numbers after stripping commas, spaces, currency
    symbols and percent signs. Missing markers become NaN."""
    cleaned = (
        series.astype(str).str.strip()
        .str.replace(r"[,]", "", regex=True)
        .str.replace(r"[\$€£¥]", "", regex=True)
        .str.replace("%", "", regex=False)
        .replace(list(MISSING_SPELLINGS), pd.NA)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _numeric_parse_ratio(series: pd.Series, config: Any) -> float:
    """Fraction of sampled values that parse as numbers after stripping
    commas, spaces, currency symbols and percent signs."""
    sample = _sample(series, config.SAMPLE_SIZE)
    if len(sample) == 0:
        return 0.0
    parsed = _to_numeric_clean(sample)
    return float(parsed.notna().mean())


def _date_parse_ratio(series: pd.Series, config: Any) -> tuple[float, bool]:
    """(parse ratio, any date separator present) for a string column.

    Plain integers are never parsed as dates (rule 3).
    """
    sample = _sample(series, config.SAMPLE_SIZE).astype(str).str.strip()
    if len(sample) == 0:
        return 0.0, False
    sep_pattern = "|".join(map(re.escape, DATE_SEPARATORS))
    has_sep = bool(sample.str.contains(sep_pattern).any())
    if not has_sep:
        return 0.0, False
    # Suppress pandas' "Could not infer format" warning: we only care about
    # the parse ratio, and errors="coerce" already handles unparseable values.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*")
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()), True


def _is_textish(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or series.dtype == object


def _name_hint(name: str, hints: tuple[str, ...]) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]
    return any(t in hints for t in tokens)


def _is_row_identifier(vals: np.ndarray) -> bool:
    """True when values are exactly 1..n with every row unique (row numbers)."""
    if len(vals) == 0:
        return False
    u = np.unique(vals)
    return len(u) == len(vals) and np.array_equal(u, np.arange(1, len(u) + 1))


def _same_digit_count(vals: np.ndarray, min_digits: int = 4) -> bool:
    """All values share the same digit count of at least ``min_digits``."""
    if len(vals) == 0:
        return False
    digits = np.array([len(str(abs(int(v)))) for v in vals])
    return len(np.unique(digits)) == 1 and int(digits[0]) >= min_digits


def _zeros_right_skew(vals: np.ndarray) -> bool:
    """Lots of zeros plus a right-skewed distribution (count-like data)."""
    if len(vals) == 0:
        return False
    zero_share = float(np.mean(vals == 0))
    median = float(np.median(vals))
    if median <= 0:
        return False
    return zero_share > 0.2 and float(np.mean(vals)) > median * 1.5


def _count_evidence(name: str, vals: np.ndarray) -> bool:
    """A column looks like a count when its name says so or its distribution
    does (many zeros, right skew)."""
    return _name_hint(name, COUNT_NAME_HINTS) or _zeros_right_skew(vals)


def _short_tokens(sample: pd.Series) -> bool:
    """Identifier-like text: short tokens with no internal whitespace."""
    if len(sample) == 0:
        return False
    s = sample.astype(str).str.strip()
    return bool((s.str.len() <= 20).all() and (~s.str.contains(r"\s")).all())


def _aligned_sample(s: pd.Series, o: pd.Series, size: int) -> tuple[pd.Series, pd.Series]:
    """Deterministic positional sample of two series sharing an index."""
    common = s.index.intersection(o.index)
    n = len(common)
    if n == 0:
        return s.iloc[:0], o.iloc[:0]
    if n <= size:
        sel = common
    else:
        pos = np.random.RandomState(42).choice(n, size=size, replace=False)
        sel = common[pos]
    return s.loc[sel], o.loc[sel]


def _is_one_to_one(s_vals: pd.Series, o_vals: pd.Series) -> bool:
    """True when each s value maps to exactly one o value and vice versa."""
    mapping: dict[Any, Any] = {}
    for sv, ov in zip(s_vals.tolist(), o_vals.tolist()):
        if sv in mapping:
            if mapping[sv] != ov:
                return False
        else:
            mapping[sv] = ov
    return len(set(mapping.values())) == len(mapping)


def _find_encoded_duplicate(series: pd.Series, column_series: dict[str, pd.Series], config: Any) -> str | None:
    """Return the name of a text column this int column is a 1:1 encoding of,
    or None. The pairwise scan is capped at PAIRWISE_CAP candidates per side."""
    s_clean = series.dropna()
    nunique = int(s_clean.nunique())
    if nunique < 3:
        return None
    candidates = 0
    for other_name, other in column_series.items():
        if other_name == series.name:
            continue
        if candidates >= config.PAIRWISE_CAP:
            break
        if not _is_textish(other):
            continue
        o_clean = other.dropna()
        if int(o_clean.nunique()) != nunique:
            continue
        candidates += 1
        s_sub, o_sub = _aligned_sample(s_clean, o_clean, config.SAMPLE_SIZE)
        if len(s_sub) < 3:
            continue
        if _is_one_to_one(s_sub, o_sub):
            return other_name
    return None


def _classify_string(
    s_clean: pd.Series, name: str, config: Any, n_eff: int, nunique: int
) -> dict[str, Any]:
    uniq_ratio = nunique / n_eff
    if uniq_ratio >= config.IDENTIFIER_UNIQUE_RATIO:
        if _short_tokens(s_clean):
            return _conf("identifier", 0.9, False, ["near-unique short tokens"], n_eff, config.MIN_ROWS)
        return _conf("text", 0.85, False, ["near-unique long values"], n_eff, config.MIN_ROWS)
    if uniq_ratio <= config.CARDINALITY_RATIO or nunique <= config.DISCRETE_ABS_CAP:
        return _conf("categorical", 0.9, False, ["low cardinality"], n_eff, config.MIN_ROWS)
    return _conf("text", 0.9, False, ["high cardinality text"], n_eff, config.MIN_ROWS)


def _classify_numeric(
    series: pd.Series,
    s_clean: pd.Series,
    name: str,
    column_series: dict[str, pd.Series],
    config: Any,
    n_eff: int,
    nunique: int,
    parse_ratio: float,
) -> dict[str, Any]:
    is_float = pd.api.types.is_float_dtype(series)

    # float: numeric_measurement, never an identifier (rule 5).
    if is_float:
        return _conf("numeric_measurement", 0.95, False, ["float dtype"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)

    num = _to_numeric_clean(s_clean).dropna()
    if len(num) == 0:
        return _conf("unknown", 0.5, True, ["numeric path but no parseable values"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)
    whole_mask = np.modf(num.values.astype(float))[0] == 0
    all_whole = bool(np.all(whole_mask))
    # A column with any fractional value (prices, rates) is a measurement,
    # not an integer code, so skip the integer-like sub-rules.
    if not all_whole:
        return _conf("numeric_measurement", 0.9, False, ["numeric values"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)
    ints = num[whole_mask].astype(np.int64)

    # 1. contiguous 1..n covering every row, or identifier name hint -> identifier.
    if _is_row_identifier(ints.values) or _name_hint(name, IDENTIFIER_NAME_HINTS):
        return _conf("identifier", 0.9, False, ["sequential row identifier or id-like name"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)

    # 2. all values share a 4+ digit count -> codes / phones / years -> ambiguous.
    if _same_digit_count(ints.values, min_digits=4):
        return _conf(
            "unknown",
            0.7,
            True,
            [f"all values share a {len(str(abs(int(ints.iloc[0]))))}-digit code pattern"],
            n_eff,
            config.MIN_ROWS,
            parse_ratio=parse_ratio,
        )

    # 3. near-unique -> unknown + ambiguous.
    uniq_ratio = nunique / n_eff
    if uniq_ratio >= config.IDENTIFIER_UNIQUE_RATIO:
        return _conf("unknown", 0.75, True, [f"near-unique values ({uniq_ratio:.2f})"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)

    # 4. one-to-one encoding of a text column -> categorical.
    paired = _find_encoded_duplicate(series, column_series, config) if column_series else None
    if paired is not None:
        return _conf("categorical", 0.8, False, [f"one-to-one encoding of {paired}"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)

    # 5. small distinct set: count evidence -> numeric_discrete, else ambiguous.
    if nunique <= config.DISCRETE_MAX_UNIQUE:
        if _count_evidence(name, ints.values):
            return _conf("numeric_discrete", 0.85, False, ["small-count integer"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)
        return _conf(
            "unknown",
            0.6,
            True,
            [f"small integer range ({nunique} levels) with no count evidence; use --semantic-override"],
            n_eff,
            config.MIN_ROWS,
            parse_ratio=parse_ratio,
        )

    # 6. otherwise numeric_measurement.
    return _conf("numeric_measurement", 0.9, False, ["numeric measurement column"], n_eff, config.MIN_ROWS, parse_ratio=parse_ratio)


def _classify_column(
    series: pd.Series,
    col_info: dict[str, Any],
    column_series: dict[str, pd.Series],
    config: Any,
) -> dict[str, Any]:
    """Classify one column into a semantic type. Never raises; unclear cases
    become ``unknown`` with ``ambiguous=true`` (rule 5)."""
    try:
        return _classify_column_unwrapped(series, col_info, column_series, config)
    except Exception:  # noqa: BLE001 - a single column must never crash profiling
        return _semantic("unknown", 0.0, True, ["classification failed; treated as unknown"])


def _classify_column_unwrapped(
    series: pd.Series,
    col_info: dict[str, Any],
    column_series: dict[str, pd.Series],
    config: Any,
) -> dict[str, Any]:
    name = series.name if series.name is not None else ""
    s_clean = series.dropna()
    n_eff = int(len(s_clean))
    nunique = int(s_clean.nunique())
    # Ratios are computed on n_eff (non-missing count), never on row count.

    # Rule 1: all missing -> unknown ambiguous; one distinct value -> constant.
    if n_eff == 0:
        return _conf("unknown", 1.0, True, ["all values missing"], n_eff, config.MIN_ROWS)
    if nunique <= 1:
        return _conf("constant", 1.0, False, ["one distinct non-missing value"], n_eff, config.MIN_ROWS)

    low_sample = n_eff < config.MIN_ROWS
    is_bool = pd.api.types.is_bool_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_float = pd.api.types.is_float_dtype(series)
    is_textish = _is_textish(series)

    # Rule 2: small sample -> dtype / exact-two-values only, mark ambiguous.
    if low_sample:
        if is_bool:
            return _conf("binary", 0.8, True, ["boolean dtype, small sample"], n_eff, config.MIN_ROWS)
        if nunique == 2:
            return _conf("binary", 0.7, True, ["exactly two distinct values, small sample"], n_eff, config.MIN_ROWS)
        if is_datetime:
            return _conf("datetime", 0.8, True, ["datetime dtype, small sample"], n_eff, config.MIN_ROWS)
        if is_numeric:
            return _conf("numeric_measurement", 0.6, True, ["numeric dtype, small sample"], n_eff, config.MIN_ROWS)
        return _conf("unknown", 0.5, True, ["small sample, could not classify"], n_eff, config.MIN_ROWS)

    # Rule 3: datetime.
    if is_datetime:
        return _conf("datetime", 0.95, False, ["datetime dtype"], n_eff, config.MIN_ROWS)
    if is_textish:
        ratio, has_sep = _date_parse_ratio(s_clean, config)
        if ratio >= config.DATE_PARSE_RATIO and has_sep:
            return _conf("datetime", 0.85, False, [f"date-string parse ratio {ratio:.2f}"], n_eff, config.MIN_ROWS, parse_ratio=ratio)

    # Rule 4: binary = exactly two distinct non-missing values (any spelling).
    if nunique == 2:
        return _conf("binary", 0.9, False, ["exactly two distinct non-missing values"], n_eff, config.MIN_ROWS)

    # Rule 5: numeric vs string path.
    if is_numeric:
        numeric = True
        parse_ratio = 1.0
    elif is_textish:
        parse_ratio = _numeric_parse_ratio(s_clean, config)
        numeric = parse_ratio >= config.NUMERIC_PARSE_RATIO
    else:
        numeric = False
        parse_ratio = 0.0

    if numeric:
        return _classify_numeric(
            series, s_clean, name, column_series, config, n_eff, nunique, parse_ratio
        )
    return _classify_string(s_clean, name, config, n_eff, nunique)


def classify_column(
    series: pd.Series,
    col_info: dict[str, Any] | None = None,
    column_series: dict[str, pd.Series] | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Public wrapper around :func:`_classify_column` for direct testing.

    ``config`` defaults to :data:`src.check_config.check_config.semantic`.
    """
    if config is None:
        config = check_config.semantic
    if column_series is None:
        column_series = {}
    if col_info is None:
        col_info = {}
    return _classify_column(series, col_info, column_series, config)


def classify_columns(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]],
    target_column: str | None = None,
    overrides: dict[str, str] | None = None,
    config: Any = None,
) -> list[dict[str, Any]]:
    """Second profiling pass: classify every column in ``columns_info``.

    Pure and deterministic: reuses values already in each column info dict,
    never reloads the CSV, and never raises. ``overrides`` (name -> type) win
    over every rule; unknown override columns produce a warning.
    """
    if config is None:
        config = check_config.semantic
    column_series = {name: df[name] for name in df.columns}
    results: list[dict[str, Any]] = []
    for col_info in columns_info:
        name = col_info.get("name")
        series = df[name] if name in df.columns else pd.Series(dtype=object)
        result = _classify_column(series, col_info, column_series, config)
        results.append(result)

    if overrides:
        name_to_result = {col_info.get("name"): result for col_info, result in zip(columns_info, results)}
        for name, new_type in overrides.items():
            if new_type not in SEMANTIC_TYPES:
                warnings.warn(f"Invalid semantic override type {new_type!r} for column {name!r}; skipped.")
                continue
            if name not in df.columns:
                warnings.warn(f"Semantic override column {name!r} not found in dataset; skipped.")
                continue
            result = name_to_result.get(name)
            if result is not None:
                result["type"] = new_type
                result["confidence"] = 1.0
                result["ambiguous"] = False
                result["reasons"] = [f"user override via --semantic-override: {new_type}"]

    return results


def get_semantic_lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{column_name: semantic_dict}`` from a profile.

    Missing ``semantic`` keys yield an empty dict so callers can safely use
    ``.get()`` on profiles built before classification existed.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for col in profile.get("columns", []) or []:
        lookup[col.get("name")] = col.get("semantic", {}) or {}
    return lookup


def _profile_column(series: pd.Series) -> dict[str, Any]:
    """Build the per-column info dict for a single Series.

    Kept separate from ``profile_dataset`` so it can be tested directly
    without a CSV round-trip (CSVs don't preserve boolean dtypes).

    ``missing`` counts true NaN values AND common text markers such as ``?``,
    so the profile never claims a column is complete when it only looks clean.
    ``text_markers`` keeps the marker share visible on its own.
    """
    nan_count, marker_count = count_missing(series)
    total_missing = nan_count + marker_count
    n_rows = len(series)

    col_info: dict[str, Any] = {
        "name": series.name,
        "dtype": str(series.dtype),
        "missing": total_missing,
        "missing_pct": (
            round(total_missing / n_rows * 100, 2) if n_rows else 0.0
        ),
        "text_markers": marker_count,
    }

    if pd.api.types.is_bool_dtype(series):
        clean_bool = series.dropna()
        bool_info = {
            "type_category": "boolean",
            "true_count": int(clean_bool.sum()),
            "true_pct": round(
                float((clean_bool.mean() if not clean_bool.empty else 0) * 100),
                2,
            ),
        }
        col_info.update(bool_info)

    elif pd.api.types.is_numeric_dtype(series):
        col_info["stats"] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
        }

    else:
        col_info["unique"] = int(series.nunique(dropna=True))
        col_info["sample_values"] = (
            series.dropna().astype(str).head(3).tolist()
        )

    return col_info


def profile_dataset(
    csv_path: str | Path,
    target_column: str | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compact profile of a CSV dataset.

    Reports the target's ``target_type`` when given, and applies semantic
    ``overrides`` (rule 0: overrides win over every classifier rule).
    """
    df = read_csv_robust(csv_path)

    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    columns: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        columns.append(_profile_column(series))

    # Second pass: compact semantic classification per column.
    # Reuses values already computed by _profile_column; does not reload the CSV.
    semantic = classify_columns(
        df, columns, target_column=target_column, overrides=overrides
    )
    for col_info, sem in zip(columns, semantic):
        col_info["semantic"] = sem

    profile["columns"] = columns

    # Target type is a hint for the check registry. An absent or unknown
    # target is reported as "unknown"; individual checks decide what to do.
    if target_column and target_column in df.columns:
        profile["target_type"] = detect_target_type(df[target_column])
    else:
        profile["target_type"] = "unknown"

    # First few rows as records (compact sample for the LLM)
    profile["sample_rows"] = df.head(3).to_dict(orient="records")

    return profile


__all__ = [
    "profile_dataset",
    "_profile_column",
    "classify_column",
    "classify_columns",
    "get_semantic_lookup",
    "SEMANTIC_TYPES",
]
