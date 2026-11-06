"""Outlier check for numeric columns.

Evaluates numeric columns using the Interquartile Range (IQR) method.
Flags columns where outliers make up a significant percentage of rows.

Returns a list of Finding objects (empty means no severe outliers found).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check


@register_check(
    name="outliers",
    required=("csv_path",),
    severity_key="outliers",
    description="Check numeric columns for excessive outliers using IQR.",
    always=True,
)
def outlier_check(df: pd.DataFrame, target_column: Optional[str] = None) -> list[Finding]:
    """Detect numeric columns with excessive outliers using the IQR method.

    For each numeric column, computes Q1, Q3, and IQR. Any value outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is considered an outlier.

    Flags columns where outliers exceed configured thresholds:
    - outlier_pct > outlier_pct_threshold     -> MEDIUM
    - outlier_pct > high_outlier_pct_threshold -> HIGH

    Non-numeric columns are skipped.
    """
    findings: list[Finding] = []

    for col in df.columns:
        series = df[col]

        # Skip non-numeric columns safely.
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # Drop NaN before computing IQR.
        clean = series.dropna()
        if len(clean) < 4:
            # Not enough data for a meaningful IQR calculation.
            continue

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
        outlier_pct = float(outlier_count / len(clean) * 100)

        if outlier_pct > check_config.outliers.high_outlier_pct_threshold:
            severity = Severity.HIGH
        elif outlier_pct > check_config.outliers.outlier_pct_threshold:
            severity = Severity.MEDIUM
        else:
            continue  # No outlier issue flagged

        findings.append(
            Finding(
                check_name="outliers",
                severity=severity,
                evidence=(
                    f"Column '{col}' has {outlier_count} outlier(s) "
                    f"({outlier_pct:.2f}% of {len(clean)} non-null rows) "
                    f"outside [{lower_bound:.4f}, {upper_bound:.4f}] "
                    f"(IQR method)."
                ),
                affected_columns=[col],
                confidence=0.9,
            )
        )

    return findings


__all__ = ["outlier_check"]
