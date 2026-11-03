"""Duplicate row detection check.

Thresholds:
  - Any duplicate rows exist                          -> MEDIUM
  - More than 10% of rows are duplicates              -> HIGH

Returns an empty list when no duplicates are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.schemas.finding import Finding, Severity


def duplicates_check(csv_path: str, target_column: Optional[str] = None) -> list[Finding]:
    """Detect repeated rows in a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc: 
        return [
            Finding(
                check_name="duplicates",
                severity=Severity.HIGH,
                evidence=f"Failed to read CSV: {exc}",
                affected_columns=[],
                suggested_fix="Verify the file path and format.",
                confidence=1.0,
            )
        ]

    n_rows = len(df)
    if n_rows == 0:
        return []

    n_dupes = int(df.duplicated().sum())
    if n_dupes == 0:
        return []

    dupe_pct = round(n_dupes / n_rows * 100, 2)
    severity = Severity.HIGH if dupe_pct > 10 else Severity.MEDIUM

    return [
        Finding(
            check_name="duplicates",
            severity=severity,
            evidence=(
                f"Found {n_dupes} duplicate row(s) out of {n_rows} "
                f"({dupe_pct}% of rows)."
            ),
            affected_columns=[],
            suggested_fix=(
                "Remove exact duplicate rows or investigate why they were "
                "recorded multiple times."
            ),
            confidence=1.0,
        )
    ]


__all__ = ["duplicates_check"]
