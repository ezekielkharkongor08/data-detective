"""Performance test for the column semantic classifier.

Target: classifying a 100,000-row by 300-column synthetic frame in a few
seconds. Parse and pairwise tests run on a fixed deterministic sample
(SAMPLE_SIZE = 2000), so cost is linear per column.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.check_config import check_config
from src.tools.profiling import classify_columns, profile_dataset


def _make_wide_frame(n_rows: int = 100_000, n_cols: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, pd.Series] = {}
    for i in range(n_cols):
        kind = i % 8
        if kind == 0:
            data[f"measurement_{i}"] = rng.normal(0, 1, n_rows)
        elif kind == 1:
            data[f"discrete_{i}"] = rng.integers(1, 6, n_rows)
        elif kind == 2:
            data[f"binary_{i}"] = rng.integers(0, 2, n_rows)
        elif kind == 3:
            data[f"categorical_{i}"] = rng.choice(["a", "b", "c", "d"], n_rows)
        elif kind == 4:
            data[f"text_{i}"] = [f"unique token {i} row {j}" for j in range(n_rows)]
        elif kind == 5:
            data[f"identifier_{i}"] = [f"ID-{i}-{j}" for j in range(n_rows)]
        elif kind == 6:
            data[f"flag_{i}"] = rng.choice(["Yes", "No"], n_rows)
        else:
            data[f"code_{i}"] = rng.integers(1000, 9999, n_rows)
    return pd.DataFrame(data)


def test_classify_wide_frame_is_fast():
    """100k rows x 300 cols should classify in a few seconds."""
    df = _make_wide_frame()
    start = time.perf_counter()
    columns_info = [{"name": c} for c in df.columns]
    results = classify_columns(df, columns_info, config=check_config.semantic)
    elapsed = time.perf_counter() - start

    assert len(results) == 300
    for res in results:
        assert res["type"] in (
            "numeric_measurement",
            "numeric_discrete",
            "binary",
            "categorical",
            "ordinal_candidate",
            "datetime",
            "identifier",
            "text",
            "constant",
            "unknown",
        )
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["reasons"]) <= 2

    # Report timing (target: a few seconds).
    print(f"\nClassified {len(df.columns)} columns x {len(df)} rows in {elapsed:.2f}s")
    assert elapsed < 30.0, f"classification too slow: {elapsed:.2f}s"


def test_profile_dataset_on_wide_frame():
    """profile_dataset end-to-end on a wide frame includes semantic keys."""
    df = _make_wide_frame(n_rows=5000, n_cols=50)
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "wide.csv")
    df.to_csv(path, index=False)
    start = time.perf_counter()
    profile = profile_dataset(path)
    elapsed = time.perf_counter() - start

    assert len(profile["columns"]) == 50
    for col in profile["columns"]:
        assert "semantic" in col
        sem = col["semantic"]
        assert set(sem) == {"type", "confidence", "ambiguous", "reasons"}
        assert len(sem["reasons"]) <= 2
    print(f"\nProfiled 50 columns x 5000 rows in {elapsed:.2f}s")
