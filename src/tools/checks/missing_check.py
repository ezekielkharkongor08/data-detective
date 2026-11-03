"""Missing value detection check.

Thresholds:
  - Any column has > 0% missing                        -> LOW
  - Any column has > 20% missing                        -> HIGH
  - Any column has > 50% missing                        -> CRITICAL

Flags common text missing markers.

Returns an empty list when no missing values are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.schemas.finding import Finding, Severity
from src.tools.missing_markers import count_missing


def missing_check(csv_path: str, target_column: Optional[str] = None) -> list[Finding]:
    """Detect missing values."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc: 
        return [
            Finding(
                check_name="missing_values",
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

        nan_count, marker_count = count_missing(series)
        total_missing = nan_count + marker_count
        if total_missing == 0:
            continue

        missing_pct = round(total_missing / len(df) * 100, 2)

        if missing_pct > 50:
            severity = Severity.CRITICAL
        elif missing_pct > 20:
            severity = Severity.HIGH
        else:
            severity = Severity.LOW

        marker_detail = (
            f" including {marker_count} text marker(s)"
            if marker_count > 0
            else ""
        )
        findings.append(
            Finding(
                check_name="missing_values",
                severity=severity,
                evidence=(
                    f"Column '{col}' has {total_missing} missing value(s) "
                    f"({missing_pct}% of rows){marker_detail}."
                ),
                affected_columns=[col],
                suggested_fix=(
                    "Impute missing values, drop affected rows, or investigate "
                    "the source of the gaps."
                ),
                confidence=1.0,
            )
        )

    return findings


__all__ = ["missing_check"]
