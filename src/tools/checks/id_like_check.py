"""ID-like column detection check.

Thresholds:
  - A non-target column has a unique value for every row -> MEDIUM

Returns an empty list when no ID-like columns are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check


@register_check(
    name="id_like",
    required=("csv_path",),
    severity_key="id_like",
    description="Check for ID-like columns that are unique per row.",
    target_hint=True,
)
def id_like_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect columns that are unique per row (likely identifiers)."""

    n_rows = len(df)
    if n_rows == 0:
        return []

    findings: list[Finding] = []

    for col in df.columns:
        if col == target_column:
            continue
        unique_count = int(df[col].nunique(dropna=True))
        if unique_count == n_rows:
            findings.append(
                Finding(
                    check_name="id_like",
                    severity=Severity.MEDIUM,
                    evidence=(
                        f"Column '{col}' has a unique value for every "
                        f"row ({n_rows} rows). It may be an identifier "
                        "and should not be used as a feature."
                    ),
                    affected_columns=[col],
                    confidence=0.85,
                )
            )

    return findings


__all__ = ["id_like_check"]
