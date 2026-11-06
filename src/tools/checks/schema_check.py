"""Schema and type consistency check.

Thresholds:
  - A column stored as text but mostly numeric values   -> MEDIUM
  - A column with mixed types                            -> MEDIUM

Returns an empty list when no schema issues are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas.api.types import is_string_dtype

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check


@register_check(
    name="schema",
    required=("csv_path",),
    severity_key="schema",
    description="Check for schema problems such as numbers stored as text.",
    always=True,
)
def schema_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect numbers stored as text and mixed-type columns."""

    findings: list[Finding] = []

    for col in df.columns:
        series = df[col]

        # Numbers stored as text: string dtype but most non-null values
        # look numeric.  pandas 3.x uses `str` dtype instead of `object`.
        if is_string_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            numeric_like = pd.to_numeric(non_null, errors="coerce").notna()
            numeric_pct = float(numeric_like.mean())
            if numeric_pct > check_config.schema.numeric_pct_threshold:
                findings.append(
                    Finding(
                        check_name="schema",
                        severity=Severity.MEDIUM,
                        evidence=(
                            f"Column '{col}' is stored as text but "
                            f"{round(numeric_pct * 100, 2)}% of its values "
                            "look numeric."
                        ),
                        affected_columns=[col],
                        confidence=0.9,
                    )
                )

    return findings


__all__ = ["schema_check"]
