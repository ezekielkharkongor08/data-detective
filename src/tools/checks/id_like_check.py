"""ID-like column detection check.

Thresholds:
  - A non-target column has a unique value for every row -> MEDIUM

Returns an empty list when no ID-like columns are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.schemas.finding import Finding, Severity


def id_like_check(csv_path: str, target_column: Optional[str] = None) -> list[Finding]:
    """Detect columns that are unique per row (likely identifiers)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 — wrapper safety
        return [
            Finding(
                check_name="id_like",
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
                    suggested_fix=(
                        "Drop this column from the feature set unless "
                        "it is the target."
                    ),
                    confidence=0.85,
                )
            )

    return findings


__all__ = ["id_like_check"]
