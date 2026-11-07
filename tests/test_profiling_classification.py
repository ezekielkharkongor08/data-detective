"""Unit tests for the column semantic classifier (profiling.py).

Covers the required probe columns and a synthetic matrix across dataset sizes
and missing-value ratios. The classifier is deterministic and never raises.
"""

import numpy as np
import pandas as pd
import pytest

from src.check_config import check_config
from src.tools.profiling import (
    SEMANTIC_TYPES,
    classify_column,
    classify_columns,
    get_semantic_lookup,
    profile_dataset,
)


def _write_csv(df: pd.DataFrame) -> str:
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def _one(series: pd.Series, **kw) -> dict:
    """Classify one series through the public wrapper."""
    return classify_column(series, config=check_config.semantic, **kw)


# ---------------------------------------------------------------------------
# Shape / contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t", SEMANTIC_TYPES)
def test_semantic_types_are_stable(t):
    assert t in SEMANTIC_TYPES


def test_classify_column_returns_known_keys():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    res = _one(s)
    assert set(res) == {"type", "confidence", "ambiguous", "reasons"}
    assert res["type"] in SEMANTIC_TYPES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["reasons"]) <= 2


def test_classify_column_never_raises():
    """Unclear / weird input must become unknown+ambiguous, not crash."""
    weird = pd.Series([np.nan, "?", np.inf, -np.inf, "x"], name="weird")
    res = _one(weird)
    assert res["type"] in SEMANTIC_TYPES
    assert 0.0 <= res["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Required probe columns
# ---------------------------------------------------------------------------

def test_senior_citizen_zero_one_is_binary():
    """Telco SeniorCitizen (0/1 int) must NOT be treated as a measurement."""
    s = pd.Series([0, 1, 0, 1, 0, 1] * 20, name="SeniorCitizen")
    res = _one(s)
    assert res["type"] == "binary"
    assert not res["ambiguous"]


def test_education_num_is_encoded_categorical():
    """Adult education.num is an ordinal code duplicating education -> categorical."""
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] * 40,
                  name="education.num")
    # One text label per code, aligned positionally -> a true 1:1 encoding.
    texts = pd.Series([f"edu_{int(v)}" for v in s], name="education")
    res = classify_column(
        s, column_series={"education.num": s, "education": texts},
        config=check_config.semantic,
    )
    assert res["type"] == "categorical"
    assert not res["ambiguous"]
    assert "education" in " ".join(res["reasons"])


def test_bedrooms_is_measurement_not_rejected():
    """House bedrooms (a real count, stored as float) must be numeric_measurement."""
    s = pd.Series([3.0, 5.0, 3.0, 4.0, 2.0, 6.0, 3.0, 4.0] * 100, name="bedrooms")
    res = _one(s)
    assert res["type"] == "numeric_measurement"
    assert not res["ambiguous"]


def test_id_columns_are_identifiers():
    s = pd.Series([f"ID-{i}" for i in range(200)], name="customerID")
    res = _one(s)
    assert res["type"] == "identifier"
    assert not res["ambiguous"]


def test_constant_column():
    s = pd.Series([7, 7, 7, 7, 7] * 40, name="const")
    res = _one(s)
    assert res["type"] == "constant"
    assert not res["ambiguous"]


def test_high_cardinality_text_is_text():
    s = pd.Series([f"long unique value number {i} here" for i in range(200)],
                  name="comment")
    res = _one(s)
    assert res["type"] == "text"
    assert not res["ambiguous"]


# ---------------------------------------------------------------------------
# Synthetic matrix: sizes x missing ratios
# ---------------------------------------------------------------------------

def _make_df(n_rows: int, missing_ratio: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "measurement": rng.normal(0, 1, n_rows),
        "discrete": rng.integers(1, 6, n_rows),
        "binary": rng.integers(0, 2, n_rows),
        "categorical": rng.choice(["a", "b", "c"], n_rows),
        "identifier": [f"ID-{i}" for i in range(n_rows)],
        "text": [f"unique long value {i} token" for i in range(n_rows)],
    })
    for col in df.columns:
        mask = rng.random(n_rows) < missing_ratio
        df.loc[mask, col] = np.nan
    return df


@pytest.mark.parametrize("n_rows", [20, 200, 20000])
@pytest.mark.parametrize("missing_ratio", [0.0, 0.2, 0.9])
def test_classification_matrix_no_crash_and_sane(n_rows, missing_ratio):
    """Every combination must classify without raising and return sane dicts."""
    df = _make_df(n_rows, missing_ratio)
    p = profile_dataset(_write_csv(df))
    lookup = get_semantic_lookup(p)
    assert len(lookup) == df.shape[1]
    for col in p["columns"]:
        sem = col["semantic"]
        assert sem["type"] in SEMANTIC_TYPES, (col["name"], sem)
        assert 0.0 <= sem["confidence"] <= 1.0
        assert len(sem["reasons"]) <= 2
        # n_eff < MIN_ROWS => ambiguous (rule 2)
        n_eff = int(df[col["name"]].notna().sum())
        if n_eff < check_config.semantic.MIN_ROWS:
            assert sem["ambiguous"], (col["name"], sem)


def test_large_clean_dataset_classifies_core_types():
    """20,000 clean rows: core columns reach their expected types."""
    df = _make_df(20000, 0.0)
    p = profile_dataset(_write_csv(df))
    lookup = get_semantic_lookup(p)
    assert lookup["measurement"]["type"] == "numeric_measurement"
    assert lookup["binary"]["type"] == "binary"
    assert lookup["identifier"]["type"] == "identifier"
    assert lookup["categorical"]["type"] == "categorical"
    assert lookup["text"]["type"] == "text"


# ---------------------------------------------------------------------------
# Specific ambiguous / non-binary cases
# ---------------------------------------------------------------------------

def test_twenty_row_dataset_is_ambiguous():
    df = _make_df(20, 0.0)
    p = profile_dataset(_write_csv(df))
    for col in p["columns"]:
        assert col["semantic"]["ambiguous"], col["name"]


def test_status_code_one_to_five_ambiguous():
    """Status codes 1..5 are a small contiguous range -> ambiguous (override)."""
    s = pd.Series([1, 2, 3, 4, 5] * 40, name="status_code")
    res = _one(s)
    assert res["type"] == "unknown"
    assert res["ambiguous"]
    assert "semantic-override" in " ".join(res["reasons"])


def test_zipcode_and_year_columns_ambiguous():
    zips = pd.Series([90210, 94105, 10001, 60601, 98101] * 40, name="zipcode")
    years = pd.Series([1990, 2005, 2012, 1987, 2020] * 40, name="year")
    assert _one(zips)["type"] == "unknown" and _one(zips)["ambiguous"]
    assert _one(years)["type"] == "unknown" and _one(years)["ambiguous"]


def test_yes_no_maybe_not_binary():
    """Three distinct values -> categorical, not binary."""
    s = pd.Series(["Yes", "No", "Maybe"] * 40, name="answer")
    res = _one(s)
    assert res["type"] == "categorical"
    assert not res["ambiguous"]


def test_date_string_column_is_datetime():
    s = pd.Series(["2014-05-02", "2014-06-03", "2014-07-04"] * 40, name="date")
    res = _one(s)
    assert res["type"] == "datetime"
    assert not res["ambiguous"]


def test_currency_and_percent_strings_are_numeric():
    """Text numbers with $ and % must take the numeric path."""
    money = pd.Series(["$12.50", "$1,234.00", " 45.00 ", "$0.99"] * 40, name="price")
    pct = pd.Series(["45%", "100%", "7.5%", "0%"] * 40, name="rate")
    assert _one(money)["type"] == "numeric_measurement"
    assert _one(pct)["type"] == "numeric_measurement"


def test_float_all_unique_is_measurement_not_identifier():
    """All-unique floats are numeric_measurement, never identifier."""
    s = pd.Series([float(i) + 0.5 for i in range(200)], name="measurement")
    res = _one(s)
    assert res["type"] == "numeric_measurement"
    assert res["type"] != "identifier"


# ---------------------------------------------------------------------------
# Determinism / order independence
# ---------------------------------------------------------------------------

def test_classification_is_row_order_independent():
    rng = np.random.default_rng(1)
    df = _make_df(2000, 0.1, seed=1)
    p1 = profile_dataset(_write_csv(df))
    df_shuf = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    p2 = profile_dataset(_write_csv(df_shuf))
    l1 = get_semantic_lookup(p1)
    l2 = get_semantic_lookup(p2)
    for col in df.columns:
        assert l1[col]["type"] == l2[col]["type"], col
        assert l1[col]["ambiguous"] == l2[col]["ambiguous"], col


def test_classification_is_column_order_independent():
    df = _make_df(2000, 0.0, seed=3)
    cols = list(df.columns)
    p1 = profile_dataset(_write_csv(df))
    df_rev = df[cols[::-1]]
    p2 = profile_dataset(_write_csv(df_rev))
    l1 = get_semantic_lookup(p1)
    l2 = get_semantic_lookup(p2)
    for col in cols:
        assert l1[col]["type"] == l2[col]["type"], col


# ---------------------------------------------------------------------------
# get_semantic_lookup on a profile missing semantic keys
# ---------------------------------------------------------------------------

def test_get_semantic_lookup_tolerates_missing_keys():
    profile = {"columns": [{"name": "a"}, {"name": "b", "semantic": {}}]}
    lookup = get_semantic_lookup(profile)
    assert lookup == {"a": {}, "b": {}}
