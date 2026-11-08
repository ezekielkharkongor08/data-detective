"""Unit tests for dataset profiling (src/tools/profiling.py).

Covers shape, missing counts, numeric stats, text markers, boolean columns,
and column semantic classification.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.check_config import check_config
from src.tools.profiling import (
    SEMANTIC_TYPES,
    _profile_column,
    classify_column,
    classify_columns,
    get_semantic_lookup,
    profile_dataset,
)


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ── profile_dataset ─────────────────────────────────────────────────────

def test_profile_dataset_shape_and_columns():
    df = pd.DataFrame(
        {"age": [25, 30, None, 45], "city": ["NY", "LA", "NY", None], "flag": [0, 1, 0, 1]}
    )
    profile = profile_dataset(_write_csv(df))
    assert profile["n_rows"] == 4
    assert profile["n_cols"] == 3
    names = {c["name"] for c in profile["columns"]}
    assert names == {"age", "city", "flag"}
    age = next(c for c in profile["columns"] if c["name"] == "age")
    assert age["dtype"] in ("float64", "Float64")
    assert age["missing"] == 1
    assert age["missing_pct"] == 25.0
    assert "stats" in age
    city = next(c for c in profile["columns"] if c["name"] == "city")
    assert city["missing"] == 1
    assert "unique" in city
    assert "sample_values" in city
    assert len(profile["sample_rows"]) == 3


def test_profile_dataset_missing_percent():
    df = pd.DataFrame({"a": [1, None, None, 4]})
    profile = profile_dataset(_write_csv(df))
    col = profile["columns"][0]
    assert col["missing"] == 2
    assert col["missing_pct"] == 50.0


def test_profile_dataset_numeric_stats():
    df = pd.DataFrame({"score": [10.0, 20.0, 30.0]})
    profile = profile_dataset(_write_csv(df))
    stats = profile["columns"][0]["stats"]
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
    assert stats["mean"] == 20.0


def test_profile_counts_text_markers_as_missing():
    series = pd.Series(["?", "Private", "?", "Self-emp"], name="workclass")
    col = _profile_column(series)
    assert col["missing"] == 2
    assert col["missing_pct"] == 50.0
    assert col["text_markers"] == 2


def test_profile_counts_nan_and_markers_together():
    series = pd.Series(["?", None, "Private", "NA"], name="workclass")
    col = _profile_column(series)
    assert col["missing"] == 3  # 1 NaN + 2 markers
    assert col["missing_pct"] == 75.0
    assert col["text_markers"] == 2


def test_profile_dataset_marks_text_markers_from_csv():
    df = pd.DataFrame({"workclass": ["?", "Private", "?", "Self-emp"]})
    profile = profile_dataset(_write_csv(df))
    col = profile["columns"][0]
    assert col["missing"] == 2
    assert col["text_markers"] == 2


def test_profile_dataset_boolean_column():
    series = pd.Series([True, False, True, None], dtype="boolean", name="active")
    col = _profile_column(series)
    assert col["dtype"] == "boolean"
    assert col["type_category"] == "boolean"
    assert col["true_count"] == 2
    assert col["true_pct"] == 66.67


# ── semantic classification ─────────────────────────────────────────────

@pytest.mark.parametrize("t", SEMANTIC_TYPES)
def test_semantic_types_are_stable(t):
    assert t in SEMANTIC_TYPES


def test_classify_column_returns_known_keys():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    res = classify_column(s, config=check_config.semantic)
    assert set(res) == {"type", "confidence", "ambiguous", "reasons"}
    assert res["type"] in SEMANTIC_TYPES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["reasons"]) <= 2


def test_classify_column_never_raises():
    weird = pd.Series([np.nan, "?", np.inf, -np.inf, "x"], name="weird")
    res = classify_column(weird, config=check_config.semantic)
    assert res["type"] in SEMANTIC_TYPES
    assert 0.0 <= res["confidence"] <= 1.0


def test_senior_citizen_zero_one_is_binary():
    s = pd.Series([0, 1, 0, 1, 0, 1] * 20, name="SeniorCitizen")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "binary"
    assert not res["ambiguous"]


def test_id_columns_are_identifiers():
    s = pd.Series([f"ID-{i}" for i in range(200)], name="customerID")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "identifier"
    assert not res["ambiguous"]


def test_constant_column():
    s = pd.Series([7, 7, 7, 7, 7] * 40, name="const")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "constant"
    assert not res["ambiguous"]


def test_high_cardinality_text_is_text():
    s = pd.Series([f"long unique value number {i} here" for i in range(200)], name="comment")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "text"
    assert not res["ambiguous"]


def test_float_all_unique_is_measurement_not_identifier():
    s = pd.Series([float(i) + 0.5 for i in range(200)], name="measurement")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "numeric_measurement"
    assert res["type"] != "identifier"


def test_twenty_row_dataset_is_ambiguous():
    df = pd.DataFrame({"m": range(20), "c": ["a", "b"] * 10, "id": [f"I{i}" for i in range(20)]})
    p = profile_dataset(_write_csv(df))
    for col in p["columns"]:
        assert col["semantic"]["ambiguous"], col["name"]


def test_yes_no_maybe_not_binary():
    s = pd.Series(["Yes", "No", "Maybe"] * 40, name="answer")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "categorical"
    assert not res["ambiguous"]


def test_date_string_column_is_datetime():
    s = pd.Series(["2014-05-02", "2014-06-03", "2014-07-04"] * 40, name="date")
    res = classify_column(s, config=check_config.semantic)
    assert res["type"] == "datetime"
    assert not res["ambiguous"]


def test_currency_and_percent_strings_are_numeric():
    money = pd.Series(["$12.50", "$1,234.00", " 45.00 ", "$0.99"] * 40, name="price")
    pct = pd.Series(["45%", "100%", "7.5%", "0%"] * 40, name="rate")
    assert classify_column(money, config=check_config.semantic)["type"] == "numeric_measurement"
    assert classify_column(pct, config=check_config.semantic)["type"] == "numeric_measurement"


def test_get_semantic_lookup_tolerates_missing_keys():
    profile = {"columns": [{"name": "a"}, {"name": "b", "semantic": {}}]}
    lookup = get_semantic_lookup(profile)
    assert lookup == {"a": {}, "b": {}}


def test_classification_matrix_no_crash_and_sane():
    rng = np.random.default_rng(42)
    n_rows = 200
    df = pd.DataFrame(
        {
            "measurement": rng.normal(0, 1, n_rows),
            "discrete": rng.integers(1, 6, n_rows),
            "binary": rng.integers(0, 2, n_rows),
            "categorical": rng.choice(["a", "b", "c"], n_rows),
            "identifier": [f"ID-{i}" for i in range(n_rows)],
            "text": [f"unique long value {i} token" for i in range(n_rows)],
        }
    )
    p = profile_dataset(_write_csv(df))
    lookup = get_semantic_lookup(p)
    assert len(lookup) == df.shape[1]
    for col in p["columns"]:
        sem = col["semantic"]
        assert sem["type"] in SEMANTIC_TYPES, (col["name"], sem)
        assert 0.0 <= sem["confidence"] <= 1.0
        assert len(sem["reasons"]) <= 2