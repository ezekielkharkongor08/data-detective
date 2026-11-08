"""Unit tests for all registered check functions.

Every check now accepts ``df: pd.DataFrame`` directly (the CSV is loaded
centrally by ``run_check`` / the graph, not by each check).
"""

import os
import tempfile

import pandas as pd
import pytest

from src.schemas.finding import Finding, Severity
from src.check_config import check_config
from src.tools.checks.bias_check import bias_check
from src.tools.checks.duplicates_check import duplicates_check
from src.tools.checks.id_like_check import id_like_check
from src.tools.checks.imbalance_check import imbalance_check
from src.tools.checks.leakage_check import leakage_check
from src.tools.checks.missing_check import missing_check
from src.tools.checks.outlier_check import outlier_check
from src.tools.checks.schema_check import schema_check
from src.tools.checks.skewness_check import skewness_check
from src.tools.checks.registry import (
    get_check,
    run_check,
    _skip_note_message,
    _cap_names,
)
from src.tools.profiling import profile_dataset
from src.tools.target_type import is_classification_target

# ── shared dirty dataset ────────────────────────────────────────────────

DIRTY_DF = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 4],
        "target": [0, 0, 0, 1, 1],
        "dup": ["x", "x", "y", "z", "x"],
        "leak": [0, 1, 0, 1, 0],
        "num": ["1", "2", "3", "4", "5"],
        "missing": [1, None, 3, None, 5],
    }
)
# Add one fully duplicated row (row 0 repeated) so duplicates_check fires.
DIRTY_DF = pd.concat([DIRTY_DF, DIRTY_DF.iloc[:1]], ignore_index=True)


# ── duplicates_check ────────────────────────────────────────────────────

def test_duplicates_check_dirty():
    findings = duplicates_check(DIRTY_DF)
    assert len(findings) == 1
    assert findings[0].check_name == "duplicates"
    assert findings[0].severity == Severity.HIGH
    assert "1 duplicate" in findings[0].evidence


def test_duplicates_check_clean():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert duplicates_check(df) == []


def test_duplicates_at_threshold_boundary():
    df = pd.DataFrame({"a": list(range(10))})
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)  # 1/10 = 10% dupes
    findings = duplicates_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


# ── missing_check ───────────────────────────────────────────────────────

def test_missing_check_with_markers():
    df = pd.DataFrame({"a": [1, None, "?", 4], "b": ["x", "y", "z", "w"]})
    findings = missing_check(df)
    assert len(findings) == 1
    assert findings[0].check_name == "missing_values"
    assert findings[0].affected_columns == ["a"]
    assert findings[0].severity == Severity.HIGH


def test_missing_check_clean():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert missing_check(df) == []


def test_missing_check_critical():
    df = pd.DataFrame({"a": [1, None, None, None, None]})
    findings = missing_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_missing_at_high_boundary_is_low():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5], "b": list("xyzwv")})  # 1/5 = 20%
    findings = missing_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.LOW


def test_missing_above_high_is_high():
    df = pd.DataFrame({"a": [1, None, None, 3, 4], "b": list("xyzwv")})  # 2/5 = 40%
    findings = missing_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_missing_at_critical_boundary_is_high():
    df = pd.DataFrame({"a": [None] * 5 + [1] * 5, "b": list(range(10))})  # 5/10 = 50%
    findings = missing_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


# ── schema_check ────────────────────────────────────────────────────────

def test_schema_check_text_as_number():
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "hello"]})
    findings = schema_check(df)
    assert len(findings) == 1
    assert findings[0].check_name == "schema"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].affected_columns == ["a"]


def test_schema_check_clean():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert schema_check(df) == []


def test_schema_at_threshold_boundary_is_clean():
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "hello"]})  # 4/5 = 80%
    findings = schema_check(df)
    assert findings == []


def test_schema_above_threshold_is_medium():
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "hello"]})  # 5/6 ~ 83%
    findings = schema_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


# ── imbalance_check ─────────────────────────────────────────────────────

def test_imbalance_check_skewed():
    df = pd.DataFrame({"target": [0] * 90 + [1] * 10})
    findings = imbalance_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].check_name == "class_imbalance"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].affected_columns == ["target"]


def test_imbalance_check_extreme():
    df = pd.DataFrame({"target": [0] * 99 + [1] * 1})
    findings = imbalance_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_imbalance_check_balanced():
    df = pd.DataFrame({"target": [0] * 50 + [1] * 50})
    assert imbalance_check(df, target_column="target") == []


def test_imbalance_check_no_target_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert imbalance_check(df) == []


def test_imbalance_check_missing_target():
    df = pd.DataFrame({"a": [1, 2, 3]})
    findings = imbalance_check(df, target_column="nonexistent")
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_imbalance_at_high_threshold_is_medium():
    df = pd.DataFrame({"target": [0] * 90 + [1] * 10})  # 10% minority
    findings = imbalance_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_imbalance_at_medium_threshold_is_clean():
    df = pd.DataFrame({"target": [0] * 70 + [1] * 30})  # 30% minority
    findings = imbalance_check(df, target_column="target")
    assert findings == []


# ── id_like_check ───────────────────────────────────────────────────────

def test_id_like_check_dirty():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "val": ["a", "a", "b", "b"]})
    findings = id_like_check(df)
    assert len(findings) == 1
    assert findings[0].check_name == "id_like"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].affected_columns == ["id"]


def test_id_like_check_clean():
    df = pd.DataFrame({"val": ["a", "a", "b", "b"]})
    assert id_like_check(df) == []


def test_id_like_check_skips_target():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "target": ["a", "b", "c", "d"]})
    findings = id_like_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].affected_columns == ["id"]
    assert "target" not in findings[0].affected_columns


# ── skewness_check ──────────────────────────────────────────────────────

def test_skewness_check_high_skew():
    df = pd.DataFrame({"skewed": list(range(1, 16)) + [1000], "normal": list(range(1, 17))})
    findings = skewness_check(df)
    assert len(findings) == 1
    assert findings[0].check_name == "skewness"
    assert findings[0].severity == Severity.HIGH
    assert findings[0].affected_columns == ["skewed"]


def test_skewness_check_symmetric():
    df = pd.DataFrame({"normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    assert skewness_check(df) == []


def test_skewness_check_skips_non_numeric():
    df = pd.DataFrame({"category": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]})
    assert skewness_check(df) == []


# ── outlier_check ───────────────────────────────────────────────────────

def test_outlier_check_many_outliers():
    df = pd.DataFrame({"outlier_col": list(range(1, 17)) + [100, 200, 300, 400]})
    findings = outlier_check(df)
    assert len(findings) == 1
    assert findings[0].check_name == "outliers"
    assert findings[0].severity == Severity.HIGH
    assert findings[0].affected_columns == ["outlier_col"]


def test_outlier_check_clean():
    df = pd.DataFrame({"normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    assert outlier_check(df) == []


def test_outlier_check_skips_non_numeric():
    df = pd.DataFrame({"category": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]})
    assert outlier_check(df) == []


def test_outlier_check_empty():
    assert outlier_check(pd.DataFrame({"col": []})) == []


# ── leakage_check ───────────────────────────────────────────────────────

def test_leakage_check_direct_copy():
    df = pd.DataFrame(
        {"leak": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )
    findings = leakage_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].check_name == "target_leakage"
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].affected_columns == ["leak"]
    assert "ROC AUC" in findings[0].evidence


def test_leakage_check_clean():
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "a": np.random.randint(0, 100, 100),
            "b": np.random.choice(["x", "y", "z"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )
    assert leakage_check(df, target_column="target") == []


def test_leakage_check_id_column_not_leakage():
    df = pd.DataFrame({"id": list(range(10)), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    findings = leakage_check(df, target_column="target", suspicious_columns=["id"])
    assert findings == []


def test_leakage_check_continuous_direct_copy():
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
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "salary": np.random.randn(100) * 10000 + 50000,
        }
    )
    assert leakage_check(df, target_column="salary") == []


def test_leakage_check_no_target_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    findings = leakage_check(df)
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


# ── bias_check ──────────────────────────────────────────────────────────

def test_bias_check_critical_disparate_impact():
    # 60/100 vs 10/100 -> DI 0.167, p≈0 -> CRITICAL
    df = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "target": [1] * 60 + [0] * 40 + [1] * 10 + [0] * 90,
        }
    )
    findings = bias_check(df, target_column="target", sensitive_columns=["group"])
    assert len(findings) == 1
    assert findings[0].check_name == "bias"
    assert findings[0].severity == Severity.CRITICAL


def test_bias_check_no_finding_when_rates_close():
    df = pd.DataFrame(
        {"group": ["A"] * 100 + ["B"] * 100, "target": [1] * 50 + [0] * 50 + [1] * 48 + [0] * 52}
    )
    findings = bias_check(df, target_column="target", sensitive_columns=["group"])
    assert findings == []


def test_bias_check_no_sensitive_columns_is_info_note():
    df = pd.DataFrame({"gender": ["M", "F"] * 40, "target": [0, 1] * 40})
    findings = bias_check(df, target_column="target")
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_bias_check_skips_non_binary_target():
    """Multiclass target returns an empty list (check gates on binary/boolean)."""
    df = pd.DataFrame(
        {"group": ["A"] * 60 + ["B"] * 60, "target": ["red", "blue", "green"] * 40}
    )
    findings = bias_check(df, target_column="target", sensitive_columns=["group"])
    assert findings == []


# ── odd column robustness ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "check_fn",
    [duplicates_check, schema_check, missing_check],
)
def test_checks_handle_empty_column_name(check_fn):
    df = pd.DataFrame({"": [1, 2, 3], "col2": [4, 5, 6], "target": [0, 1, 0]})
    findings = check_fn(df)
    assert isinstance(findings, list)


@pytest.mark.parametrize(
    "check_fn",
    [duplicates_check, schema_check, missing_check],
)
def test_checks_handle_duplicate_column_names(check_fn):
    df = pd.DataFrame([["a", "a", "target"], ["b", "b", "target"]])
    findings = check_fn(df)
    assert isinstance(findings, list)


@pytest.mark.parametrize(
    "check_fn",
    [duplicates_check, schema_check, missing_check],
)
def test_checks_handle_all_null_column(check_fn):
    df = pd.DataFrame({"a": [None, None, None], "target": [0, 1, 0]})
    findings = check_fn(df)
    assert isinstance(findings, list)


# ── Finding sorting ─────────────────────────────────────────────────────

def test_finding_sorting_highest_first():
    f_low = Finding(check_name="a", severity=Severity.LOW, evidence="x")
    f_crit = Finding(check_name="b", severity=Severity.CRITICAL, evidence="y")
    f_med = Finding(check_name="c", severity=Severity.MEDIUM, evidence="z")
    sorted_list = sorted([f_low, f_crit, f_med])
    assert [f.severity for f in sorted_list] == [
        Severity.CRITICAL,
        Severity.MEDIUM,
        Severity.LOW,
    ]


# ── semantic gating / skip note ──────────────────────────────────────────


def _gating_csv(df, name="gating.csv"):
    path = os.path.join(tempfile.mkdtemp(), name)
    df.to_csv(path, index=False)
    return path


def _gating_state(csv_path, profile, target_column="target"):
    return {
        "csv_path": csv_path,
        "target_column": target_column,
        "profile": profile,
        "check_params": {},
    }


def _run_skewness(csv_path, profile, target_column="target"):
    spec = get_check("skewness")
    return run_check(spec, _gating_state(csv_path, profile, target_column))


def _skip_note(findings):
    for f in findings:
        if f.evidence.startswith("Skipped") or f.evidence.startswith(
            "Semantic profile unavailable"
        ):
            return f.evidence
    return None


def test_skip_note_lists_only_excluded_columns_grouped_by_reason():
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "amount": [float(i) for i in range(100)],  # numeric, eligible
            "note": ["free text entry " + str(i) for i in range(100)],  # text
        }
    )
    csv_path = _gating_csv(df)
    profile = profile_dataset(csv_path, target_column="target")
    findings = _run_skewness(csv_path, profile)
    note = _skip_note(findings)
    assert note is not None
    assert "target" not in note
    assert "amount" not in note
    assert "note" in note
    assert "wrong type" in note


def test_skip_note_absent_when_all_eligible():
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "amount": [float(i) for i in range(100)],
            "score": [float(i * 0.5) for i in range(100)],
        }
    )
    csv_path = _gating_csv(df)
    profile = profile_dataset(csv_path, target_column="target")
    findings = _run_skewness(csv_path, profile)
    assert _skip_note(findings) is None


def test_skip_note_all_non_target_excluded():
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "note1": ["text " + str(i) for i in range(100)],
            "note2": ["more text " + str(i) for i in range(100)],
        }
    )
    csv_path = _gating_csv(df)
    profile = profile_dataset(csv_path, target_column="target")
    findings = _run_skewness(csv_path, profile)
    note = _skip_note(findings)
    assert note is not None
    assert "target" not in note
    assert "note1" in note
    assert "note2" in note


def test_skip_note_empty_lookup_runs_ungated():
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "amount": [float(i) for i in range(100)],
        }
    )
    csv_path = _gating_csv(df)
    state = _gating_state(csv_path, {}, target_column="target")
    spec = get_check("skewness")
    findings = run_check(spec, state)
    note = _skip_note(findings)
    assert note is not None
    assert "Semantic profile unavailable" in note


def test_skip_note_message_groups_by_reason():
    msg = _skip_note_message("skewness", ["date", "street"], ["view", "condition"])
    assert "wrong type: date, street" in msg
    assert "ambiguous type: view, condition" in msg
    assert "Skipped 4 column(s) not eligible for the skewness check" in msg


def test_cap_names_short_list():
    assert _cap_names(["a", "b"]) == "a, b"


def test_cap_names_long_list_capped():
    assert (
        _cap_names(["a", "b", "c", "d", "e", "f", "g", "h"], limit=6)
        == "a, b, c, d, e, f and 2 more"
    )


# ── target handling & degenerate IQR (Step 5b + Step 3) ─────────────────


def _target_frame(target_col, target_values, n=None):
    if n is None:
        n = len(target_values)
    return pd.DataFrame({
        target_col: list(target_values)[:n],
        "skewed_feat": ([1.0, 2.0, 3.0, 4.0] * ((n - 1) // 4 + 1))[:n - 1] + [100.0],
        "outlier_feat": list(range(1, n - 14)) + [1000.0] * 10 + [-1000.0] * 5,
        "normal_feat": [float(i) for i in range(n)],
    })


def _run_checks_on(df, target_column):
    path = os.path.join(tempfile.mkdtemp(), "target_frame.csv")
    df.to_csv(path, index=False)
    profile = profile_dataset(path, target_column=target_column)
    state = {
        "csv_path": path,
        "target_column": target_column,
        "profile": profile,
        "check_params": {},
    }
    out = {}
    for cn in ("skewness", "outliers"):
        spec = get_check(cn)
        findings = run_check(spec, state)
        out[cn] = [
            (f.affected_columns[0] if f.affected_columns else "(none)",
             f.severity.value,
             f.evidence)
            for f in findings
        ]
    return out


def _cols(findings_by_check, check_name):
    return {col for col, sev, _ in findings_by_check.get(check_name, []) if sev != "info"}


def _notes(findings_by_check, check_name):
    return [ev for col, sev, ev in findings_by_check.get(check_name, []) if sev == "info"]


def test_is_classification_target():
    assert is_classification_target(pd.Series([0, 1, 0, 1])) is True
    assert is_classification_target(pd.Series([True, False, True])) is True
    assert is_classification_target(pd.Series(["a", "b", "c"] * 10)) is True
    assert is_classification_target(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])) is False
    assert is_classification_target(pd.Series([1, 1, 1])) is False


def test_skewness_outliers_exclude_only_classification_target():
    """Binary/boolean/multiclass targets: only the target column is excluded;
    eligible feature columns are still analyzed."""
    n = 100
    for target_values in ([0, 1] * 50, [True, False] * 50, [0, 1, 2, 3] * 25):
        df = _target_frame("target", target_values, n)
        res = _run_checks_on(df, "target")
        assert "target" not in _cols(res, "skewness"), res
        assert "target" not in _cols(res, "outliers"), res
        assert "skewed_feat" in _cols(res, "skewness"), res
        assert "outlier_feat" in _cols(res, "outliers"), res
        assert "normal_feat" not in _cols(res, "skewness"), res
        assert "normal_feat" not in _cols(res, "outliers"), res


def test_skewness_outliers_continuous_target_analyzed():
    n = 100
    df = _target_frame("target", [float(i) for i in range(n)], n)
    res = _run_checks_on(df, "target")
    # Continuous target is analyzed (uniform here -> no finding); features still analyzed.
    assert "target" not in _cols(res, "skewness"), res
    assert "target" not in _cols(res, "outliers"), res
    assert "skewed_feat" in _cols(res, "skewness"), res
    assert "outlier_feat" in _cols(res, "outliers"), res


def test_skewness_finding_on_skewed_continuous_target():
    n = 100
    target = [1.0] * 80 + [float(i) for i in range(100, 120)]  # skewed, continuous
    df = _target_frame("target", target, n)
    res = _run_checks_on(df, "target")
    target_sev = [sev for col, sev, _ in res["skewness"] if col == "target"]
    assert target_sev, res


def test_outlier_finding_on_continuous_target_with_hint():
    n = 100
    target = [float(i) for i in range(85)] + [1000.0] * 10 + [-1000.0] * 5
    df = _target_frame("target", target, n)
    res = _run_checks_on(df, "target")
    target_findings = [(col, sev, ev) for col, sev, ev in res["outliers"] if col == "target"]
    assert len(target_findings) == 1, res
    assert target_findings[0][1] == "high", res
    hint = ("This is the target column: review these rows before removing them; "
            "consider a log transform or a robust loss.")
    assert hint in target_findings[0][2], target_findings[0][2]


def test_zero_iqr_column_skipped_with_note():
    n = 100
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "zero_heavy": [0.0] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0],
        "normal_outlier": list(range(1, 86)) + [1000.0] * 10 + [-1000.0] * 5,
    })
    res = _run_checks_on(df, "target")
    assert "zero_heavy" not in _cols(res, "outliers"), res
    assert "normal_outlier" in _cols(res, "outliers"), res
    notes = _notes(res, "outliers")
    assert any("too few distinct values for IQR outlier detection" in note for note in notes), notes
    assert any("zero_heavy" in note for note in notes), notes


def test_no_note_when_no_degenerate_column():
    n = 100
    df = _target_frame("target", [0, 1] * 50, n)
    res = _run_checks_on(df, "target")
    assert _notes(res, "outliers") == [], res
