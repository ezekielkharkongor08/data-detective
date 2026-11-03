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

from src.schemas.finding import Finding, Severity


def schema_check(csv_path: str, target_column: Optional[str] = None) -> list[Finding]:
    """Detect numbers stored as text and mixed-type columns."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc: 
        return [
            Finding(
                check_name="schema",
                severity=Severity.HIGH,
                evidence=f"Failed to read CSV: {exc}",
                affected_columns=[],
                suggested_fix="Verify the file path and format.",
                confidence=1.0,
            )
        ]

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
            if numeric_pct > 0.8:
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
                        suggested_fix=(
                            "Convert the column to a numeric type "
                            "(e.g., pd.to_numeric)."
                        ),
                        confidence=0.9,
                    )
                )

    return findings


__all__ = ["schema_check"]
