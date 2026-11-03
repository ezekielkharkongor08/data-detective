"""One shared definition of "missing" for profiling and checks.

Real CSVs often encode missing values as text markers (``?``, ``NA``, ``-``,
blank strings) instead of true NaN. If profiling counted only NaN while the
missing check counted markers too, the report would claim ``missing=0`` for a
column that is actually full of ``?``. That is misleading, so both places use
this module.

Design rule: pandas does the counting; the LLM only reads the result.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_string_dtype

# Common non-NaN missing-value markers found in real-world CSVs.
MISSING_MARKERS = {"?", "NA", "N/A", "null", "NULL", "", " ", "-", "--"}


def count_missing(series: pd.Series) -> tuple[int, int]:
    """Return ``(nan_count, marker_count)`` for one column.

    ``nan_count`` counts true NaN/NaT values. ``marker_count`` counts string
    cells (after stripping spaces) that match a known missing marker.
    Markers are only counted for string columns; numbers are never matched.
    """
    nan_count = int(series.isna().sum())

    marker_count = 0
    if is_string_dtype(series):
        marker_count = int(
            series.astype(str).str.strip().isin(MISSING_MARKERS).sum()
        )

    return nan_count, marker_count


__all__ = ["MISSING_MARKERS", "count_missing"]
