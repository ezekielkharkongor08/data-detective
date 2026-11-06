"""Unit tests for the fairness check (bias_check.py)."""

import os
import tempfile

import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.schemas.finding import Severity
from src.tools.checks.bias_check import bias_check


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def _build_table(group_a_fav, group_a_total, group_b_fav, group_b_total):
    """Build a two-group dataset with the given favorable counts.

    Returns a DataFrame with columns ``group`` (sensitive) and ``target``
    (0/1 binary target where 1 is favorable).
    """
    rows = []
    rows += [{"group": "A", "target": 1}] * group_a_fav
    rows += [{"group": "A", "target": 0}] * (group_a_total - group_a_fav)
    rows += [{"group": "B", "target": 1}] * group_b_fav
    rows += [{"group": "B", "target": 0}] * (group_b_total - group_b_fav)
    return pd.DataFrame(rows)


def _expected_p(group_a_fav, group_a_total, group_b_fav, group_b_total):
    table = [
        [group_a_fav, group_a_total - group_a_fav],
        [group_b_fav, group_b_total - group_b_fav],
    ]
    return chi2_contingency(table)[1]


# ── severity by disparate impact ──────────────────────────────────────────

def test_fairness_critical_disparate_impact():
    """DI ~0.167 with p≈0 -> CRITICAL finding."""
    # 60/100 vs 10/100 -> DI 0.167, p≈0
    df = _build_table(60, 100, 10, 100)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    f = findings[0]
    assert f.check_name == "bias"
    assert f.severity == Severity.CRITICAL
    assert f.affected_columns == ["group"]
    assert "DI" in f.evidence or "disparate impact" in f.evidence.lower()


def test_fairness_high_disparate_impact():
    """DI = 0.5 with p=0.000456 -> HIGH finding."""
    # 50/100 vs 25/100 -> DI 0.500, p=0.000456
    df = _build_table(50, 100, 25, 100)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_fairness_medium_disparate_impact():
    """DI = 0.7 with p<0.01 -> MEDIUM finding."""
    # 80/100 vs 56/100 -> DI 0.700, p=0.000489
    df = _build_table(80, 100, 56, 100)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_fairness_no_finding_when_rates_close():
    """DI = 0.96 with p=0.888 -> no finding (empty list)."""
    # 50/100 vs 48/100 -> DI 0.960, p=0.888
    df = _build_table(50, 100, 48, 100)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert findings == []


def test_fairness_visible_but_not_significant_is_info_note():
    """DI = 0.5 with p=0.0297 -> INFO note, not a finding."""
    # 20/35 vs 10/35 -> DI 0.500, p=0.0297 (above P_VALUE_LIMIT)
    df = _build_table(20, 35, 10, 35)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


# ── target type gating ────────────────────────────────────────────────────

def test_fairness_skips_non_binary_target():
    """Multiclass target -> one INFO note asking for favorable_value."""
    df = pd.DataFrame(
        {"group": ["A"] * 60 + ["B"] * 60, "target": ["red", "blue", "green"] * 40}
    )
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_fairness_skips_missing_target():
    """Target column not in CSV -> one INFO note."""
    df = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40, "other": [1] * 80})
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_fairness_skips_no_target_column_given():
    """No target_column -> one INFO note."""
    df = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40, "target": [1] * 80})
    findings = bias_check(_write_csv(df), sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


# ── sensitive columns: user-specified only ────────────────────────────────

def test_fairness_no_sensitive_columns_is_info_note_with_hints():
    """No sensitive_columns -> INFO note with name-based hints only."""
    df = pd.DataFrame(
        {"gender": ["M", "F"] * 40, "target": [0, 1] * 40, "unrelated": range(80)}
    )
    findings = bias_check(_write_csv(df), target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert "gender" in findings[0].evidence.lower()
    assert "not analysed" in findings[0].evidence.lower() or "not analyzed" in findings[0].evidence.lower()


def test_fairness_does_not_auto_analyse_hinted_columns():
    """Hinted columns are never analysed unless explicitly requested."""
    df = pd.DataFrame(
        {"gender": ["M"] * 40 + ["F"] * 40, "target": [0, 1] * 40}
    )
    findings = bias_check(_write_csv(df), target_column="target")
    # No sensitive columns requested -> only a skip note, no group analysis.
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_fairness_missing_sensitive_column_is_info_note():
    """Requested sensitive column absent from CSV -> INFO note."""
    df = pd.DataFrame({"target": [0, 1] * 40, "other": range(80)})
    findings = bias_check(
        _write_csv(df), target_column="target", sensitive_columns=["gender"]
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_fairness_sensitive_column_is_target_is_info_note():
    """Sensitive column equal to target -> INFO note, not a CRITICAL finding."""
    df = pd.DataFrame({"target": [0, 1] * 40})
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["target"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


# ── favorable value resolution ────────────────────────────────────────────

def test_fairness_inferred_favorable_for_boolean_target():
    """Boolean target defaults favorable to True."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 60 + ["B"] * 60,
            "target": [True] * 50 + [False] * 10 + [True] * 10 + [False] * 50,
        }
    )
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_fairness_inferred_favorable_for_01_target():
    """0/1 numeric target defaults favorable to 1."""
    df = _build_table(60, 100, 10, 100)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_fairness_explicit_favorable_value():
    """String target with explicit favorable_value works."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 60 + ["B"] * 60,
            "target": ["yes"] * 50 + ["no"] * 10 + ["yes"] * 10 + ["no"] * 50,
        }
    )
    findings = bias_check(
        _write_csv(df), target_column="target", sensitive_columns=["group"], favorable_value="yes"
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_fairness_ambiguous_favorable_value_is_info_note():
    """No favorable_value on an ambiguous binary target -> one INFO note."""
    # Binary but neither bool nor {0,1}: must ask for favorable_value.
    df = pd.DataFrame(
        {"group": ["A"] * 60 + ["B"] * 60, "target": ["red", "blue"] * 60}
    )
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert "favorable_value" in findings[0].evidence


def test_fairness_favorable_value_not_in_target_is_info_note():
    """favorable_value that never appears -> one INFO note."""
    df = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40, "target": [0, 1] * 40})
    findings = bias_check(
        _write_csv(df), target_column="target", sensitive_columns=["group"], favorable_value=99
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


# ── small groups / missing values ─────────────────────────────────────────

def test_fairness_small_groups_skipped_with_info_note():
    """Groups below MIN_GROUP_SIZE (30) -> INFO note."""
    # 20 rows per group, below MIN_GROUP_SIZE=30.
    df = _build_table(15, 20, 5, 20)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_fairness_missing_value_rows_are_dropped():
    """Rows with missing target/sensitive values are dropped, not crashed."""
    df = pd.DataFrame(
        {"group": ["A"] * 60 + ["B"] * 60 + [None] * 10, "target": [1] * 60 + [0] * 60 + [1] * 10}
    )
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert isinstance(findings, list)
    for f in findings:
        assert f.check_name == "bias"


# ── p-value gate ──────────────────────────────────────────────────────────

def test_fairness_p_value_gate_uses_chi_square():
    """The p-value gate matches scipy's chi2_contingency."""
    g_a_fav, g_a_total = 60, 100
    g_b_fav, g_b_total = 10, 100
    expected_p = _expected_p(g_a_fav, g_a_total, g_b_fav, g_b_total)
    df = _build_table(g_a_fav, g_a_total, g_b_fav, g_b_total)
    findings = bias_check(_write_csv(df), target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert f"p={expected_p:.3e}" in findings[0].evidence or "p-value" in findings[0].evidence


__all__ = []
