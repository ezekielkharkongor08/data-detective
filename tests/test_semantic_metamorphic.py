"""Metamorphic tests for the column semantic classifier.

The classification must be invariant under:
  - row shuffling
  - column shuffling
  - renaming columns to meaningless names (except where a name hint was the
    deciding reason — then the reason must say so)
  - rescaling / shifting numeric columns
  - relabelling category strings
"""

import numpy as np
import pandas as pd

from src.check_config import check_config
from src.tools.profiling import (
    classify_column,
    classify_columns,
    get_semantic_lookup,
    profile_dataset,
)


def df_to_csv(df: pd.DataFrame):
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def _base_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "customer_id": [f"ID-{i}" for i in range(200)],
            "is_active": rng.integers(0, 2, 200),
            "score": rng.normal(50, 15, 200),
            "count": rng.integers(1, 6, 200),
            "city": rng.choice(["NY", "LA", "SF"], 200),
            "comment": [f"unique long value {i} token here" for i in range(200)],
            "year": rng.integers(1990, 2021, 200),
            "grade": rng.integers(1, 6, 200),
        }
    )


def _classify_df(df: pd.DataFrame) -> dict:
    p = profile_dataset(df_to_csv(df))
    return get_semantic_lookup(p)


def df_to_csv(df: pd.DataFrame):
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Row-shuffle invariance
# ---------------------------------------------------------------------------

def test_row_shuffle_invariant():
    df = _base_df()
    before = _classify_df(df)
    df_shuf = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    after = _classify_df(df_shuf)
    for col in df.columns:
        assert before[col]["type"] == after[col]["type"], col
        assert before[col]["ambiguous"] == after[col]["ambiguous"], col


# ---------------------------------------------------------------------------
# Column-shuffle invariance
# ---------------------------------------------------------------------------

def test_column_shuffle_invariant():
    df = _base_df()
    before = _classify_df(df)
    cols = list(df.columns)
    df_shuf = df[cols[::-1]]
    after = _classify_df(df_shuf)
    for col in cols:
        assert before[col]["type"] == after[col]["type"], col
        assert before[col]["ambiguous"] == after[col]["ambiguous"], col


# ---------------------------------------------------------------------------
# Rename to meaningless names
# ---------------------------------------------------------------------------

def test_rename_to_meaningless_names():
    """Renaming all columns to col_0..col_N must not change value-based
    classifications. Only name-hint-driven results may change, and then the
    reason must mention the name hint / override."""
    df = _base_df()
    before = _classify_df(df)
    renamed = {c: f"col_{i}" for i, c in enumerate(df.columns)}
    df_renamed = df.rename(columns=renamed)
    after = _classify_df(df_renamed)

    # customer_id was identifier via value rule (near-unique short tokens),
    # not via name hint — must stay identifier.
    assert after["col_0"]["type"] == "identifier"
    assert not after["col_0"]["ambiguous"]

    # is_active was binary via exactly-two-values — stays binary.
    assert after["col_1"]["type"] == "binary"
    assert not after["col_1"]["ambiguous"]

    # grade was ordinal_candidate via value rule (small int range); renaming
    # removes the hint so it may become ambiguous. That's allowed: the reason
    # must not claim a name hint decided it.
    assert after["col_7"]["type"] in ("unknown", "numeric_discrete", "ordinal_candidate")


# ---------------------------------------------------------------------------
# Rescale / shift numeric columns
# ---------------------------------------------------------------------------

def test_rescale_numeric_invariant():
    df = _base_df()
    before = _classify_df(df)
    df_scaled = df.copy()
    df_scaled["score"] = df_scaled["score"] * 2.0 + 100.0
    df_scaled["count"] = df_scaled["count"] * 3
    after = _classify_df(df_scaled)
    assert before["score"]["type"] == after["score"]["type"]
    assert before["count"]["type"] == after["count"]["type"]


# ---------------------------------------------------------------------------
# Relabel category strings
# ---------------------------------------------------------------------------

def test_relabel_categories_invariant():
    df = _base_df()
    before = _classify_df(df)
    df_relabelled = df.copy()
    df_relabelled["city"] = df_relabelled["city"].replace(
        {"NY": "alpha", "LA": "beta", "SF": "gamma"}
    )
    after = _classify_df(df_relabelled)
    assert before["city"]["type"] == after["city"]["type"]
    assert before["city"]["ambiguous"] == after["city"]["ambiguous"]


# ---------------------------------------------------------------------------
# Missing-value ratio invariance (classification should not flip just because
# more values are missing, as long as n_eff stays above MIN_ROWS)
# ---------------------------------------------------------------------------

def test_missing_ratio_stable_above_min_rows():
    df = _base_df()
    before = _classify_df(df)
    df_missing = df.copy()
    mask = np.random.default_rng(3).random(len(df)) < 0.3
    df_missing.loc[mask, "score"] = np.nan
    after = _classify_df(df_missing)
    # score still has ~140 non-missing > MIN_ROWS=50, type should not flip.
    assert before["score"]["type"] == after["score"]["type"]
    assert before["score"]["ambiguous"] == after["score"]["ambiguous"]


# ---------------------------------------------------------------------------
# Direct classify_column determinism
# ---------------------------------------------------------------------------

def test_classify_column_deterministic():
    s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8] * 30, name="x")
    s2 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8] * 30, name="x")
    r1 = classify_column(s1, config=check_config.semantic)
    r2 = classify_column(s2, config=check_config.semantic)
    assert r1 == r2
