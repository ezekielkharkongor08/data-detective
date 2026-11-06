"""Tests for the multi-dataset harness (eval/run_harness.py)."""

import os
import tempfile

import pytest

from eval.run_harness import _audit_dataset, _check_findings_valid, _find_duplicates, _finding_key
from src.schemas.finding import Severity


# ── _finding_key ──────────────────────────────────────────────────────

def test_finding_key_is_column_order_insensitive():
    """Finding key treats affected_columns as a set (order-insensitive)."""
    f1 = {"check_name": "bias", "affected_columns": ["sex", "race"], "evidence": "disparate impact"}
    f2 = {"check_name": "bias", "affected_columns": ["race", "sex"], "evidence": "disparate impact"}
    assert _finding_key(f1) == _finding_key(f2)


def test_finding_key_distinguishes_different_findings():
    """Different checks or evidence produce different keys."""
    f1 = {"check_name": "bias", "affected_columns": ["sex"], "evidence": "disparate impact 0.358"}
    f2 = {"check_name": "bias", "affected_columns": ["sex"], "evidence": "disparate impact 0.500"}
    f3 = {"check_name": "missing_values", "affected_columns": ["sex"], "evidence": "disparate impact 0.358"}
    assert _finding_key(f1) != _finding_key(f2)
    assert _finding_key(f1) != _finding_key(f3)


# ── _find_duplicates ──────────────────────────────────────────────────

def test_find_duplicates_empty():
    """No findings means no duplicates."""
    assert _find_duplicates([]) == {}


def test_find_duplicates_none():
    """Unique findings produce no duplicates."""
    findings = [
        {"check_name": "bias", "affected_columns": ["sex"], "evidence": "DI 0.358"},
        {"check_name": "bias", "affected_columns": ["race"], "evidence": "DI 0.500"},
        {"check_name": "missing_values", "affected_columns": ["age"], "evidence": "10 missing"},
    ]
    assert _find_duplicates(findings) == {}


def test_find_duplicates_detects():
    """Identical findings (same key) are flagged as duplicates."""
    findings = [
        {"check_name": "bias", "affected_columns": ["sex"], "evidence": "DI 0.358"},
        {"check_name": "bias", "affected_columns": ["sex"], "evidence": "DI 0.358"},
        {"check_name": "bias", "affected_columns": ["sex"], "evidence": "DI 0.358"},
    ]
    dupes = _find_duplicates(findings)
    assert len(dupes) == 1
    key = ("bias", ("sex",), "DI 0.358")
    assert key in dupes
    assert dupes[key] == 3


# ── _check_findings_valid ─────────────────────────────────────────────

def test_check_findings_valid_empty():
    """No findings means no errors."""
    assert _check_findings_valid([]) == []


def test_check_findings_valid_all_valid():
    """Valid findings produce no errors."""
    findings = [
        {
            "check_name": "bias",
            "severity": Severity.CRITICAL.value,
            "evidence": "Disparate impact 0.358 for sex groups",
            "affected_columns": ["sex"],
            "confidence": 0.9,
        },
        {
            "check_name": "missing_values",
            "severity": Severity.LOW.value,
            "evidence": "Column 'age' has 5 missing values",
            "affected_columns": ["age"],
            "confidence": 1.0,
        },
    ]
    assert _check_findings_valid(findings) == []


def test_check_findings_valid_catches_invalid():
    """Invalid findings are reported."""
    findings = [
        {
            "check_name": "",  # missing check_name
            "severity": Severity.CRITICAL.value,
            "evidence": "test",
            "affected_columns": [],
            "confidence": 1.0,
        },
        {
            "check_name": "bias",
            "severity": "invalid-severity",  # invalid severity
            "evidence": "test",
            "affected_columns": [],
            "confidence": 1.0,
        },
        {
            "check_name": "bias",
            "severity": Severity.HIGH.value,
            "evidence": "",  # missing evidence
            "affected_columns": [],
            "confidence": 1.0,
        },
        {
            "check_name": "bias",
            "severity": Severity.MEDIUM.value,
            "evidence": "test",
            "affected_columns": "not-a-list",  # affected_columns not a list
            "confidence": 1.0,
        },
        {
            "check_name": "bias",
            "severity": Severity.LOW.value,
            "evidence": "test",
            "affected_columns": [],
            "confidence": 1.5,  # confidence out of range
        },
    ]
    errors = _check_findings_valid(findings)
    assert len(errors) == 5


# ── _audit_dataset integration test ───────────────────────────────────

def _write_csv(df):
    import pandas as pd
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def test_audit_dataset_small_csv_passes():
    """A small clean CSV passes all harness invariants."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 25, 30, 35, 40],
            "city": ["NY", "LA", "NY", "LA", "NY", "LA", "NY", "LA"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = _write_csv(df)

    result = _audit_dataset(
        {
            "name": "test_small",
            "path": csv_path,
            "target": "target",
            "sensitive": [],
            "favorable": None,
        }
    )

    assert result["passed"], f"Audit failed: {result['errors']}"
    assert result["findings_count"] >= 0


def test_audit_dataset_dirty_csv_passes():
    """A dirty CSV with real findings passes all harness invariants."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 1],
            "target": [0, 0, 1, 1, 0],
            "dup_col": ["x", "x", "y", "z", "x"],
        }
    )
    csv_path = _write_csv(df)

    result = _audit_dataset(
        {
            "name": "test_dirty",
            "path": csv_path,
            "target": "target",
            "sensitive": [],
            "favorable": None,
        }
    )

    assert result["passed"], f"Audit failed: {result['errors']}"
    assert result["findings_count"] > 0, "Expected findings from dirty CSV"
    check_names = result["check_names"]
    assert len(check_names) >= 2, f"Expected multiple check types, got {check_names}"


def test_audit_dataset_with_sensitive_columns_runs_bias():
    """A dataset with sensitive columns runs the bias check."""
    import pandas as pd

    # Create data with a strong gender disparity
    df = pd.DataFrame(
        {
            "gender": ["M"] * 50 + ["F"] * 50,
            "target": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
        }
    )
    csv_path = _write_csv(df)

    result = _audit_dataset(
        {
            "name": "test_bias",
            "path": csv_path,
            "target": "target",
            "sensitive": ["gender"],
            "favorable": "1",
        }
    )

    assert result["passed"], f"Audit failed: {result['errors']}"
    assert "bias" in result["check_names"], f"Expected bias check, got {result['check_names']}"


def test_audit_dataset_without_sensitive_skips_bias():
    """A dataset without sensitive columns skips bias with an INFO note."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "gender": ["M"] * 50 + ["F"] * 50,
            "target": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
        }
    )
    csv_path = _write_csv(df)

    result = _audit_dataset(
        {
            "name": "test_no_sensitive",
            "path": csv_path,
            "target": "target",
            "sensitive": [],
            "favorable": None,
        }
    )

    assert result["passed"], f"Audit failed: {result['errors']}"

    # Bias check runs but returns an INFO skip note (not a CRITICAL/HIGH/MEDIUM finding)
    from src.tools.checks.bias_check import bias_check

    findings = bias_check(csv_path, target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert "no sensitive columns" in findings[0].evidence.lower()


__all__ = []
