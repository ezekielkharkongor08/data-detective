"""Missing value detection check.

Thresholds:
  - Any column has > 0% missing                        -> LOW
  - Any column has > missing.high_pct  missing         -> HIGH
  - Any column has > missing.critical_pct missing      -> CRITICAL

Flags common text missing markers.

Returns an empty list when no missing values are found.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check
from src.tools.missing_markers import count_missing


@register_check(
    name="missing_values",
    required=("csv_path",),
    severity_key="missing_values",
    description="Check a CSV for missing values (NaN, blanks, and text markers).",
    always=True,
    tool_name="run_missing_check",
)
def missing_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect missing values."""

    findings: list[Finding] = []

    for col in df.columns:
        series = df[col]

        nan_count, marker_count = count_missing(series)
        total_missing = nan_count + marker_count
        if total_missing == 0:
            continue

        missing_pct = round(total_missing / len(df) * 100, 2)

        if missing_pct > check_config.missing.critical_pct:
            severity = Severity.CRITICAL
        elif missing_pct > check_config.missing.high_pct:
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
                confidence=1.0,
            )
        )

    return findings


__all__ = ["missing_check"]
