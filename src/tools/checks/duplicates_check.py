"""Duplicate row detection check.

Thresholds:
  - Any duplicate rows exist                          -> MEDIUM
  - More than duplicates.high_pct  % of rows are dup  -> HIGH

Returns an empty list when no duplicates are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check


@register_check(
    name="duplicates",
    required=("csv_path",),
    severity_key="duplicates",
    description="Check a CSV for duplicate rows.",
    always=True,
)
def duplicates_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect repeated rows in a CSV file."""

    n_rows = len(df)
    if n_rows == 0:
        return []

    n_dupes = int(df.duplicated().sum())
    if n_dupes == 0:
        return []

    dupe_pct = round(n_dupes / n_rows * 100, 2)
    severity = Severity.HIGH if dupe_pct > check_config.duplicates.high_pct else Severity.MEDIUM

    return [
        Finding(
            check_name="duplicates",
            severity=severity,
            evidence=(
                f"Found {n_dupes} duplicate row(s) out of {n_rows} "
                f"({dupe_pct}% of rows)."
            ),
            affected_columns=[],
            confidence=1.0,
        )
    ]


__all__ = ["duplicates_check"]
