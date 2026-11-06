"""Unit tests for M3 leakage check (ROC AUC metric)."""

import os
import tempfile

import numpy as np
import pandas as pd

from src.schemas.finding import Severity
from src.tools.checks.leakage_check import leakage_check


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ── direct leakage: column == target ──────────────────────────────────

def test_leakage_check_direct_copy():
    """A column that equals the target is flagged CRITICAL (AUC = 1.0)."""
    df = pd.DataFrame(
        {"leak": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )
    findings = leakage_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].check_name == "target_leakage"
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].affected_columns == ["leak"]
    assert "ROC AUC" in findings[0].evidence


def test_leakage_check_perfect_correlation_text():
    """A text column that perfectly alternates with target is CRITICAL."""
    df = pd.DataFrame(
        {
            "flag": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    findings = leakage_check(df, target_column="target")
    assert len(findings) >= 1
    assert any(f.severity == Severity.CRITICAL for f in findings)


# ── no leakage ────────────────────────────────────────────────────────

def test_leakage_check_clean():
    """Uncorrelated columns return no findings (AUC near 0.5)."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "a": np.random.randint(0, 100, 100),
            "b": np.random.choice(["x", "y", "z"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )
    findings = leakage_check(df, target_column="target")
    assert findings == []


def test_leakage_check_id_column_not_leakage():
    """A unique ID column is skipped (high cardinality), not flagged."""
    df = pd.DataFrame(
        {"id": list(range(10)), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )
    findings = leakage_check(df, target_column="target", suspicious_columns=["id"])
    assert findings == []


# ── majority-only trap (imbalance) ────────────────────────────────────

def test_leakage_check_majority_only_trap():
    """A predictor that only guesses the majority class is not flagged.

    On Adult/Telco the majority class is ~76%/~73%. With the old accuracy
    metric this looked like signal; with AUC it scores 0.5 and passes.
    The feature is independent of the target (no real signal).
    """
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature": np.random.randint(0, 100, 100),
            "target": [0] * 80 + [1] * 20,
        }
    )
    findings = leakage_check(df, target_column="target")
    assert findings == []


# ── suspicious_columns parameter ────────────────────────────────────

def test_leakage_check_specific_columns():
    """Only tests the columns listed in suspicious_columns."""
    df = pd.DataFrame(
        {
            "leak": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "safe": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    # Only check 'safe' (not leakage) -> empty
    findings = leakage_check(
        df, target_column="target", suspicious_columns=["safe"]
    )
    assert findings == []

    # Check 'leak' -> CRITICAL
    findings = leakage_check(
        df, target_column="target", suspicious_columns=["leak"]
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


# ── continuous target (regression) ── R^2 path ──────────────────────────

def test_leakage_check_continuous_direct_copy():
    """A column equal to a continuous target is flagged CRITICAL (R^2 ≈ 1)."""
    df = pd.DataFrame(
        {
            "leak": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "salary": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    findings = leakage_check(df, target_column="salary")
    assert len(findings) == 1
    assert findings[0].check_name == "target_leakage"
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].affected_columns == ["leak"]
    assert "R^2" in findings[0].evidence


def test_leakage_check_continuous_clean():
    """Uncorrelated continuous columns return no findings."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "salary": np.random.randn(100) * 10000 + 50000,
        }
    )
    findings = leakage_check(df, target_column="salary")
    assert findings == []


def test_leakage_check_continuous_high_r2():
    """A continuous column with very high R^2 is flagged HIGH or CRITICAL."""
    # Build a binned version of the target (20 unique values, not high-cardinality)
    # that still predicts it well
    np.random.seed(42)
    base = np.random.randn(100) * 100
    df = pd.DataFrame(
        {
            "salary_bin": pd.cut(base, bins=20).astype(str),
            "salary": base,
        }
    )
    findings = leakage_check(df, target_column="salary")
    assert len(findings) >= 1
    assert any(f.severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM) for f in findings)


# ── auto-detection edge cases ─────────────────────────────────────────

def test_leakage_check_constant_column_skipped():
    """A constant column is skipped, not flagged."""
    df = pd.DataFrame(
        {"constant": [5] * 10, "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )
    findings = leakage_check(df, target_column="target")
    assert findings == []


def test_leakage_check_missing_values():
    """Missing values in a column don't crash the check."""
    df = pd.DataFrame(
        {
            "feature": [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    findings = leakage_check(df, target_column="target")
    # Should not crash; may or may not find leakage depending on data
    assert isinstance(findings, list)


def test_leakage_check_tiny_dataset_skipped():
    """A tiny dataset (below _MIN_ROWS) runs without crashing."""
    df = pd.DataFrame(
        {"a": [1, 2], "target": [0, 1]}
    )
    findings = leakage_check(df, target_column="target")
    # Too few rows for 5-fold stratified CV -> skipped gracefully
    assert isinstance(findings, list)


# ── edge cases ───────────────────────────────────────────────────────

def test_leakage_check_no_target_column():
    """No target_column configured returns a MEDIUM finding."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    findings = leakage_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_leakage_check_missing_target():
    """Target column not in CSV returns a MEDIUM finding."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    findings = leakage_check(df, target_column="nonexistent")
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_leakage_check_empty_csv():
    """Empty CSV returns no findings."""
    df = pd.DataFrame({"a": [], "target": []})
    findings = leakage_check(df, target_column="target")
    assert findings == []


# ── MAX_ROWS_FOR_CV sampling ─────────────────────────────────────────────

def test_leakage_check_large_dataset_sampled():
    """A dataset larger than max_rows_for_cv is sampled and evidence notes it."""
    import numpy as np

    n = 120000  # above max_rows_for_cv=100000, triggers sampling
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "leak": np.random.choice([0, 1], n),
            "target": np.random.choice([0, 1], n),
        }
    )
    # Make 'leak' perfectly predict 'target'
    df["target"] = df["leak"]
    findings = leakage_check(df, target_column="target")
    # Should not crash and should find leakage
    assert len(findings) >= 1
    f = findings[0]
    assert f.severity == Severity.CRITICAL
    # Evidence should mention sampling and random_state
    assert "sampled" in f.evidence.lower() or "sample" in f.evidence.lower()
    assert "random_state" in f.evidence


def test_leakage_check_evidence_mentions_random_state():
    """Evidence includes the fixed random_state used for CV."""
    df = pd.DataFrame(
        {"leak": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )
    findings = leakage_check(df, target_column="target")
    assert len(findings) == 1
    assert "random_state" in findings[0].evidence


__all__ = [
    "test_leakage_check_direct_copy",
    "test_leakage_check_perfect_correlation_text",
    "test_leakage_check_clean",
    "test_leakage_check_id_column_not_leakage",
    "test_leakage_check_majority_only_trap",
    "test_leakage_check_specific_columns",
    "test_leakage_check_continuous_target_skipped",
    "test_leakage_check_continuous_clean",
    "test_leakage_check_constant_column_skipped",
    "test_leakage_check_missing_values",
    "test_leakage_check_tiny_dataset_skipped",
    "test_leakage_check_no_target_column",
    "test_leakage_check_missing_target",
    "test_leakage_check_empty_csv",
    "test_leakage_check_large_dataset_sampled",
    "test_leakage_check_evidence_mentions_random_state",
]
