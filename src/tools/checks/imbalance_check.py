"""Class imbalance check.

Thresholds:
  - Minority class share < imbalance.high_threshold  -> HIGH
  - Minority class share < imbalance.medium_threshold -> MEDIUM

Requires ``target_column`` to be set in state. Returns an empty list when
the target is balanced or when no target column is configured.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check
from src.tools.target_type import detect_target_type


@register_check(
    name="class_imbalance",
    required=("csv_path", "target_column"),
    severity_key="class_imbalance",
    description="Check the class balance of a classification target column.",
    target_hint=True,
    tool_name="run_imbalance_check",
)
def imbalance_check(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> list[Finding]:
    """Detect a skewed target class distribution."""
    if not target_column:
        return []

    if target_column not in df.columns:
        return [
            Finding(
                check_name="class_imbalance",
                severity=Severity.MEDIUM,
                evidence=(
                    f"Target column '{target_column}' not found in CSV."
                ),
                affected_columns=[],
                confidence=1.0,
            )
        ]

    target = df[target_column].dropna()
    if len(target) == 0:
        return []

    target_type = detect_target_type(target)
    if target_type == "continuous":
        return []

    value_counts = target.value_counts(normalize=True)
    minority_share = float(value_counts.min())

    if minority_share < check_config.imbalance.high_threshold:
        severity = Severity.HIGH
    elif minority_share < check_config.imbalance.medium_threshold:
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
            confidence=1.0,
        )
    ]


__all__ = ["imbalance_check"]
