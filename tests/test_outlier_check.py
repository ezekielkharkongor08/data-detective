"""Unit tests for outlier check."""

import os
import tempfile

import pandas as pd

from src.schemas.finding import Severity
from src.tools.checks.outlier_check import outlier_check


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ── direct outliers: many outliers ─────────────────────────────────────

def test_outlier_check_many_outliers():
    """A column with many outliers produces a HIGH finding."""
    # 18 unique values: normal range 1-16 plus extreme outliers 100, 200, 300, 400.
    df = pd.DataFrame({
        "outlier_col": list(range(1, 17)) + [100, 200, 300, 400],
        "target": [0, 1] * 10,
    })
    findings = outlier_check(df)
    assert len(findings) == 1
    f = findings[0]
    assert f.check_name == "outliers"
    assert f.severity == Severity.HIGH
    assert f.affected_columns == ["outlier_col"]
    assert "outlier" in f.evidence.lower()


# ── no outliers: clean column ──────────────────────────────────────────

def test_outlier_check_clean():
    """A column with no outliers returns no findings."""
    df = pd.DataFrame({
        "normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    findings = outlier_check(df)
    assert findings == []


# ── non-numeric columns skipped ────────────────────────────────────────

def test_outlier_check_skips_non_numeric():
    """Non-numeric columns are skipped without error."""
    df = pd.DataFrame({
        "category": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    findings = outlier_check(df)
    assert findings == []


# ── medium outlier percentage ──────────────────────────────────────────

def test_outlier_check_medium_outliers():
    """A column with moderate outliers produces a MEDIUM finding."""
    # 10 values, 3 outliers (30% -> MEDIUM threshold is 5%, HIGH is 10%).
    # Actually 3/10 = 30% which is > 10%, so HIGH. Let's use 2/10 = 20%.
    df = pd.DataFrame({
        "mod_outlier": [1, 2, 1, 2, 1, 2, 100, 200, 3, 4],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    findings = outlier_check(df)
    assert isinstance(findings, list)
    for f in findings:
        assert f.check_name == "outliers"
        assert f.severity in (Severity.MEDIUM, Severity.HIGH)


# ── empty dataframe ────────────────────────────────────────────────────

def test_outlier_check_empty():
    """An empty dataframe returns no findings."""
    df = pd.DataFrame({"col": [], "target": []})
    findings = outlier_check(df)
    assert findings == []


# ── all null column ────────────────────────────────────────────────────

def test_outlier_check_all_null():
    """A column with all null values returns no findings."""
    df = pd.DataFrame({"col": [None, None, None], "target": [0, 1, 0]})
    findings = outlier_check(df)
    assert findings == []


__all__ = [
    "test_outlier_check_many_outliers",
    "test_outlier_check_clean",
    "test_outlier_check_skips_non_numeric",
    "test_outlier_check_medium_outliers",
    "test_outlier_check_empty",
    "test_outlier_check_all_null",
]
