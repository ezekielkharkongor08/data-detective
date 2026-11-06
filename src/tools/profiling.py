"""Dataset profiling.

A plain pandas function with no LLM and no LangGraph. It loads a CSV from disk
and returns a compact summary: column names, dtypes, missing counts, dupliactes, basic numeric stats, and a few sample rows.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.tools.csv_reader import read_csv_robust
from src.tools.missing_markers import count_missing
from src.tools.target_type import detect_target_type


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
) -> dict[str, Any]:
    """Build a compact profile of a CSV dataset.

    Args:
        csv_path: Path to the CSV file.
        target_column: Optional target column. When it exists, the profile
            reports its ``target_type`` so checks can decide whether they
            apply (for example, fairness only supports classification
            targets).

    Returns:
        A JSON-friendly profile dict.
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


__all__ = ["profile_dataset", "_profile_column"]
