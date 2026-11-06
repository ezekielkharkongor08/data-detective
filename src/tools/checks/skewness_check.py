"""Skewness check for numeric columns.

Evaluates numeric columns using pandas .skew(). Flags columns with high
skew magnitude that may indicate problematic distributions for modeling.

Returns a list of Finding objects (empty means no skewness issues found).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check


@register_check(
    name="skewness",
    required=("csv_path",),
    severity_key="skewness",
    description="Check numeric columns for high skewness.",
    always=True,
)
def skewness_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect numeric columns with high skew magnitude.

    Flags columns where the absolute skewness exceeds configured thresholds:
    - |skew| > skew_threshold    -> MEDIUM
    - |skew| > high_skew_threshold -> HIGH

    Non-numeric columns are skipped.
    """
    findings: list[Finding] = []

    for col in df.columns:
        series = df[col]

        # Skip non-numeric columns safely.
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # Drop NaN before computing skew.
        clean = series.dropna()
        if len(clean) < 3:
            # Not enough data for a meaningful skew calculation.
            continue

        skew = float(clean.skew())
        abs_skew = abs(skew)

        if abs_skew > check_config.skewness.high_skew_threshold:
            severity = Severity.HIGH
        elif abs_skew > check_config.skewness.skew_threshold:
            severity = Severity.MEDIUM
        else:
            continue  # No skewness issue flagged

        direction = "positive" if skew > 0 else "negative"

        findings.append(
            Finding(
                check_name="skewness",
                severity=severity,
                evidence=(
                    f"Column '{col}' has {direction} skew ({skew:.4f}). "
                    f"Absolute skew {abs_skew:.4f} exceeds threshold."
                ),
                affected_columns=[col],
                confidence=0.9,
            )
        )

    return findings


__all__ = ["skewness_check"]
