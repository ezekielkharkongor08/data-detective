"""Class imbalance check.

Thresholds:
  - Minority class share < 10%                         -> HIGH
  - Minority class share < 30%                         -> MEDIUM

Requires ``target_column`` to be set in state. Returns an empty list when
the target is balanced or when no target column is configured.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.schemas.finding import Finding, Severity


def imbalance_check(
    csv_path: str, target_column: Optional[str] = None
) -> list[Finding]:
    """Detect a skewed target class distribution."""
    if not target_column:
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return [
            Finding(
                check_name="class_imbalance",
                severity=Severity.HIGH,
                evidence=f"Failed to read CSV: {exc}",
                affected_columns=[],
                suggested_fix="Verify the file path and format.",
                confidence=1.0,
            )
        ]

    if target_column not in df.columns:
        return [
            Finding(
                check_name="class_imbalance",
                severity=Severity.MEDIUM,
                evidence=(
                    f"Target column '{target_column}' not found in CSV."
                ),
                affected_columns=[],
                suggested_fix=(
                    "Provide the correct target column name."
                ),
                confidence=1.0,
            )
        ]

    target = df[target_column].dropna()
    if len(target) == 0:
        return []

    value_counts = target.value_counts(normalize=True)
    minority_share = float(value_counts.min())

    if minority_share < 0.10:
        severity = Severity.HIGH
    elif minority_share < 0.30:
        severity = Severity.MEDIUM
    else:
        return []

    return [
        Finding(
            check_name="class_imbalance",
            severity=severity,
            evidence=(
                f"Target '{target_column}' is imbalanced: minority class "
                f"is {round(minority_share * 100, 2)}% of rows."
            ),
            affected_columns=[target_column],
            suggested_fix=(
                "Consider class weights, resampling, or stratified "
                "sampling during training."
            ),
            confidence=1.0,
        )
    ]


__all__ = ["imbalance_check"]
