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
from src.tools.target_type import detect_target_type


@register_check(
    name="skewness",
    required=("csv_path",),
    severity_key="skewness",
    description="Check numeric columns for high skewness.",
    always=True,
    accepted_semantic=frozenset({"numeric_measurement", "numeric_discrete"}),
)
def skewness_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect numeric columns with high absolute skew.

    Columns over the configured skew thresholds get MEDIUM/HIGH; the
    registry's semantic gating (numeric_measurement/numeric_discrete)
    pre-filters columns here.
    """
    findings: list[Finding] = []

    # Skip for classification targets — redundant with the class_imbalance
    # check; skewness only matters for regression targets.
    if target_column and target_column in df.columns:
        target = df[target_column].dropna()
        if len(target) > 0:
            target_type = detect_target_type(target)
            if target_type in ("binary", "multiclass"):
                return []

    for col in df.columns:
        series = df[col]

        # Type guard for direct callers; the registry gates this check to
        # numeric semantics normally.
        if not pd.api.types.is_numeric_dtype(series):
            continue

        clean = series.dropna()

        if len(clean) < 3:
            # Too few rows for a meaningful skew value.
            continue

        skew = float(clean.skew())
        abs_skew = abs(skew)

        if abs_skew > check_config.skewness.high_skew_threshold:
            severity = Severity.HIGH
        elif abs_skew > check_config.skewness.skew_threshold:
            severity = Severity.MEDIUM
        else:
            continue

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
