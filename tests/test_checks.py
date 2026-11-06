"""Unit tests for M2 check functions."""

import os
import tempfile

import pandas as pd

from src.schemas.finding import Finding, Severity
from src.tools.checks.duplicates_check import duplicates_check
from src.tools.checks.id_like_check import id_like_check
from src.tools.checks.imbalance_check import imbalance_check
from src.tools.checks.missing_check import missing_check
from src.tools.checks.schema_check import schema_check
from src.check_config import check_config


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ── duplicates_check ──────────────────────────────────────────────────

def test_duplicates_check_dirty():
    """Duplicate rows produce a HIGH finding (>10% dupes)."""
    df = pd.DataFrame({"a": [1, 2, 1, 3], "b": ["x", "y", "x", "z"]})
    findings = duplicates_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].check_name == "duplicates"
    assert findings[0].severity == Severity.HIGH
    assert "1 duplicate" in findings[0].evidence


def test_duplicates_check_clean():
    """No duplicates returns an empty list."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert duplicates_check(_write_csv(df)) == []


# ── missing_check ─────────────────────────────────────────────────────

def test_missing_check_with_markers():
    """NaN and '?' markers are both counted as missing."""
    df = pd.DataFrame({"a": [1, None, "?", 4], "b": ["x", "y", "z", "w"]})
    findings = missing_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].check_name == "missing_values"
    assert findings[0].affected_columns == ["a"]
    # 2 missing out of 4 = 50% -> HIGH (>20%, not >50%)
    assert findings[0].severity == Severity.HIGH


def test_missing_check_clean():
    """No missing values returns an empty list."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert missing_check(_write_csv(df)) == []


def test_missing_check_critical():
    """>50% missing is CRITICAL."""
    df = pd.DataFrame({"a": [1, None, None, None, None]})
    findings = missing_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


# ── schema_check ──────────────────────────────────────────────────────

def test_schema_check_text_as_number():
    """A text column that is mostly numeric is flagged MEDIUM."""
    # Force object dtype with mixed numeric/text values
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "hello"]})
    findings = schema_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].check_name == "schema"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].affected_columns == ["a"]


def test_schema_check_clean():
    """Clean numeric columns return an empty list."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert schema_check(_write_csv(df)) == []


# ── imbalance_check ───────────────────────────────────────────────────

def test_imbalance_check_skewed():
    """90/10 split -> MEDIUM (minority < 30%, not < 10%)."""
    df = pd.DataFrame({"target": [0] * 90 + [1] * 10})
    findings = imbalance_check(_write_csv(df), target_column="target")
    assert len(findings) == 1
    assert findings[0].check_name == "class_imbalance"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].affected_columns == ["target"]


def test_imbalance_check_extreme():
    """99/1 split -> HIGH (minority < 10%)."""
    df = pd.DataFrame({"target": [0] * 99 + [1] * 1})
    findings = imbalance_check(_write_csv(df), target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_imbalance_check_balanced():
    """50/50 split returns an empty list."""
    df = pd.DataFrame({"target": [0] * 50 + [1] * 50})
    assert imbalance_check(_write_csv(df), target_column="target") == []


def test_imbalance_check_no_target_column():
    """No target_column configured returns an empty list."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert imbalance_check(_write_csv(df)) == []


def test_imbalance_check_missing_target():
    """Target column not in CSV returns a MEDIUM finding."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    findings = imbalance_check(_write_csv(df), target_column="nonexistent")
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


# ── id_like_check ─────────────────────────────────────────────────────

def test_id_like_check_dirty():
    """A unique-per-row column is flagged MEDIUM."""
    df = pd.DataFrame({"id": [1, 2, 3, 4], "val": ["a", "a", "b", "b"]})
    findings = id_like_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].check_name == "id_like"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].affected_columns == ["id"]


def test_id_like_check_clean():
    """No unique-per-row columns returns an empty list."""
    df = pd.DataFrame({"val": ["a", "a", "b", "b"]})
    assert id_like_check(_write_csv(df)) == []


def test_id_like_check_skips_target():
    """The target column is skipped even if unique; other ID-like columns
    are still flagged."""
    df = pd.DataFrame({"id": [1, 2, 3, 4], "target": ["a", "b", "c", "d"]})
    findings = id_like_check(_write_csv(df), target_column="target")
    # id is unique and NOT the target -> flagged
    assert len(findings) == 1
    assert findings[0].affected_columns == ["id"]
    # target is unique but IS the target -> not flagged
    assert "target" not in findings[0].affected_columns


# ── wrapper safety ────────────────────────────────────────────────────

def test_check_wrapper_handles_exception():
    """A check that raises returns an error finding instead of crashing."""

    def bad_check(csv_path, target_column=None):
        raise RuntimeError("boom")

    from src.graph.nodes import _wrap_check

    result = _wrap_check(bad_check, {"csv_path": "x.csv", "target_column": None})
    findings = result["findings"]
    assert len(findings) == 1
    assert Severity(findings[0]["severity"]) == Severity.HIGH
    assert "boom" in findings[0]["evidence"]
    assert findings[0]["confidence"] == 0.0


# ── Finding sorting ───────────────────────────────────────────────────

def test_finding_sorting_highest_first():
    """Findings sort by severity, highest first."""
    f_low = Finding(check_name="a", severity=Severity.LOW, evidence="x")
    f_crit = Finding(check_name="b", severity=Severity.CRITICAL, evidence="y")
    f_med = Finding(check_name="c", severity=Severity.MEDIUM, evidence="z")
    sorted_list = sorted([f_low, f_crit, f_med])
    assert [f.severity for f in sorted_list] == [
        Severity.CRITICAL,
        Severity.MEDIUM,
        Severity.LOW,
    ]


# ── threshold boundary tests (aligned with check_config) ──────────────────

def test_duplicates_at_threshold_boundary():
    """Exactly check_config.duplicates.high_pct % dupes stays MEDIUM (not > threshold)."""
    df = pd.DataFrame({"a": list(range(10))})
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)  # 1/10 = 10% dupes
    findings = duplicates_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_duplicates_above_threshold_is_high():
    """More than check_config.duplicates.high_pct % dupes -> HIGH."""
    df = pd.DataFrame({"a": list(range(5))})
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)  # 2/5 = 40% dupes
    findings = duplicates_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_missing_at_high_boundary_is_low():
    """Exactly check_config.missing.high_pct % missing -> LOW (not > threshold)."""
    df = pd.DataFrame({"a": [1, None, 3, 4, 5], "b": list("xyzwv")})  # 1/5 = 20%
    findings = missing_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.LOW


def test_missing_above_high_is_high():
    """More than check_config.missing.high_pct % missing -> HIGH."""
    df = pd.DataFrame({"a": [1, None, None, 3, 4], "b": list("xyzwv")})  # 2/5 = 40%
    findings = missing_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_missing_at_critical_boundary_is_high():
    """Exactly check_config.missing.critical_pct % missing -> HIGH (not > threshold)."""
    df = pd.DataFrame({"a": [None] * 5 + [1] * 5, "b": list(range(10))})  # 5/10 = 50%
    findings = missing_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_schema_at_threshold_boundary_is_clean():
    """Exactly check_config.schema.numeric_pct_threshold stays clean (not > threshold)."""
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "hello"]})  # 4/5 = 80%
    findings = schema_check(_write_csv(df))
    assert findings == []


def test_schema_above_threshold_is_medium():
    """More than check_config.schema.numeric_pct_threshold -> MEDIUM."""
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "hello"]})  # 5/6 ~ 83%
    findings = schema_check(_write_csv(df))
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_imbalance_at_high_threshold_is_medium():
    """Exactly check_config.imbalance.high_threshold minority -> MEDIUM (not < threshold)."""
    df = pd.DataFrame({"target": [0] * 90 + [1] * 10})  # 10% minority
    findings = imbalance_check(_write_csv(df), target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_imbalance_at_medium_threshold_is_clean():
    """Exactly check_config.imbalance.medium_threshold minority -> clean (not < threshold)."""
    df = pd.DataFrame({"target": [0] * 70 + [1] * 30})  # 30% minority
    findings = imbalance_check(_write_csv(df), target_column="target")
    assert findings == []


# ── odd column robustness ─────────────────────────────────────────────────

def test_checks_handle_empty_column_name():
    """Checks don't crash on a column with an empty name."""
    df = pd.DataFrame({0: [1, 2, 3], "": [4, 5, 6], "target": [0, 1, 0]})
    csv = _write_csv(df)
    # All checks should return without crashing.
    for check_fn in (
        duplicates_check,
        id_like_check,
        schema_check,
        missing_check,
    ):
        findings = check_fn(csv)
        assert isinstance(findings, list)


def test_checks_handle_duplicate_column_names():
    """Checks don't crash on duplicate column names."""
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "target"])
    csv = _write_csv(df)
    for check_fn in (
        duplicates_check,
        id_like_check,
        schema_check,
        missing_check,
    ):
        findings = check_fn(csv)
        assert isinstance(findings, list)


def test_checks_handle_all_null_column():
    """Checks don't crash on an all-null column."""
    df = pd.DataFrame({"a": [None, None, None], "target": [0, 1, 0]})
    csv = _write_csv(df)
    for check_fn in (
        duplicates_check,
        id_like_check,
        schema_check,
        missing_check,
    ):
        findings = check_fn(csv)
        assert isinstance(findings, list)


def test_checks_handle_special_char_column_names():
    """Checks don't crash on columns with special characters in names."""
    df = pd.DataFrame({"col with spaces": [1, 2, 3], "col-dot.name": [4, 5, 6], "target": [0, 1, 0]})
    csv = _write_csv(df)
    for check_fn in (
        duplicates_check,
        id_like_check,
        schema_check,
        missing_check,
    ):
        findings = check_fn(csv)
        assert isinstance(findings, list)


__all__ = [
    "test_duplicates_check_dirty",
    "test_duplicates_check_clean",
    "test_duplicates_at_threshold_boundary",
    "test_duplicates_above_threshold_is_high",
    "test_missing_check_with_markers",
    "test_missing_check_clean",
    "test_missing_check_critical",
    "test_missing_at_high_boundary_is_low",
    "test_missing_above_high_is_high",
    "test_missing_at_critical_boundary_is_high",
    "test_schema_check_text_as_number",
    "test_schema_check_clean",
    "test_schema_at_threshold_boundary_is_clean",
    "test_schema_above_threshold_is_medium",
    "test_imbalance_check_skewed",
    "test_imbalance_check_extreme",
    "test_imbalance_check_balanced",
    "test_imbalance_check_no_target_column",
    "test_imbalance_check_missing_target",
    "test_imbalance_at_high_threshold_is_medium",
    "test_imbalance_at_medium_threshold_is_clean",
    "test_id_like_check_dirty",
    "test_id_like_check_clean",
    "test_id_like_check_skips_target",
    "test_check_wrapper_handles_exception",
    "test_finding_sorting_highest_first",
    "test_checks_handle_empty_column_name",
    "test_checks_handle_duplicate_column_names",
    "test_checks_handle_all_null_column",
    "test_checks_handle_special_char_column_names",
]
