import os
import tempfile

import pandas as pd

from src.tools.profiling import _profile_column, profile_dataset


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def test_profile_dataset_shape_and_columns():
    """profile_dataset reports correct shape and per-column info."""
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 45],
            "city": ["NY", "LA", "NY", None],
            "flag": [0, 1, 0, 1],
        }
    )
    csv = _write_csv(df)

    profile = profile_dataset(csv)

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
    """missing_pct is computed correctly for an all-missing-ish column."""
    df = pd.DataFrame({"a": [1, None, None, 4]})
    csv = _write_csv(df)

    profile = profile_dataset(csv)

    col = profile["columns"][0]
    assert col["missing"] == 2
    assert col["missing_pct"] == 50.0


def test_profile_dataset_numeric_stats():
    """Numeric columns get min/max/mean/std stats."""
    df = pd.DataFrame({"score": [10.0, 20.0, 30.0]})
    csv = _write_csv(df)

    profile = profile_dataset(csv)
    col = profile["columns"][0]
    stats = col["stats"]
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
    assert stats["mean"] == 20.0


def test_profile_counts_text_markers_as_missing():
    """'?' and similar markers count as missing so the report is not misleading."""
    series = pd.Series(["?", "Private", "?", "Self-emp"], name="workclass")

    col = _profile_column(series)

    assert col["missing"] == 2
    assert col["missing_pct"] == 50.0
    assert col["text_markers"] == 2


def test_profile_counts_nan_and_markers_together():
    """True NaN and text markers are added, not treated as one or the other."""
    series = pd.Series(["?", None, "Private", "NA"], name="workclass")

    col = _profile_column(series)

    assert col["missing"] == 3  # 1 NaN + 2 markers
    assert col["missing_pct"] == 75.0
    assert col["text_markers"] == 2


def test_profile_dataset_marks_text_markers_from_csv():
    """The marker count survives a real CSV read."""
    df = pd.DataFrame({"workclass": ["?", "Private", "?", "Self-emp"]})
    profile = profile_dataset(_write_csv(df))

    col = profile["columns"][0]
    assert col["missing"] == 2
    assert col["text_markers"] == 2


def test_profile_dataset_boolean_column():
    """Boolean columns get type_category, true_count, and true_pct."""
    series = pd.Series([True, False, True, None], dtype="boolean", name="active")
    col = _profile_column(series)

    assert col["dtype"] == "boolean"
    assert col["type_category"] == "boolean"
    assert col["true_count"] == 2
    assert col["true_pct"] == 66.67  # 2 True out of 3 non-null


def test_profile_dataset_boolean_column_all_null():
    """An all-null boolean column doesn't crash."""
    series = pd.Series([None, None, None], dtype="boolean", name="active")
    col = _profile_column(series)

    assert col["type_category"] == "boolean"
    assert col["true_count"] == 0
    assert col["true_pct"] == 0.0
