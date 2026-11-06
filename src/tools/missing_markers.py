"""One shared definition of "missing" for profiling and checks.

Real CSVs often encode missing values as text markers (``?``, ``NA``, ``-``,
blank strings) instead of true NaN. If profiling counted only NaN while the
missing check counted markers too, the report would claim ``missing=0`` for a
column that is actually full of ``?``. That is misleading, so both places use
this module.

Missing markers are configurable via the ``DATA_DETECTIVE_MISSING_MARKERS``
environment variable (comma-separated list) or by passing ``extra_markers``
to :func:`count_missing`.

Design rule: pandas does the counting; the LLM only reads the result.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
from pandas.api.types import is_string_dtype

# Common non-NaN missing-value markers found in real-world CSVs.
DEFAULT_MISSING_MARKERS = frozenset({"?", "NA", "N/A", "null", "NULL", "", " ", "-", "--"})


def _load_missing_markers() -> frozenset[str]:
    """Load missing markers, optionally overridden by env var.

    ``DATA_DETECTIVE_MISSING_MARKERS`` is a comma-separated list of extra
    markers. They are merged with the defaults so callers never lose the
    built-in set.
    """
    extra = os.environ.get("DATA_DETECTIVE_MISSING_MARKERS", "").strip()
    if not extra:
        return frozenset(DEFAULT_MISSING_MARKERS)
    added = {m.strip() for m in extra.split(",") if m.strip()}
    return frozenset(DEFAULT_MISSING_MARKERS | added)


# Module-level marker set (loaded once at import time).
MISSING_MARKERS: frozenset[str] = _load_missing_markers()


def count_missing(
    series: pd.Series, extra_markers: Iterable[str] | None = None
) -> tuple[int, int]:
    """Return ``(nan_count, marker_count)`` for one column.

    ``nan_count`` counts true NaN/NaT values. ``marker_count`` counts string
    cells (after stripping spaces) that match a known missing marker.
    Markers are only counted for string columns; numbers are never matched.

    Args:
        series: The column to scan.
        extra_markers: Additional markers to treat as missing for this call,
            merged with the module-level set.
    """
    nan_count = int(series.isna().sum())

    marker_count = 0
    if is_string_dtype(series):
        markers = MISSING_MARKERS
        if extra_markers:
            markers = markers | {m.strip() for m in extra_markers if m.strip()}
        marker_count = int(
            series.astype(str).str.strip().isin(markers).sum()
        )

    return nan_count, marker_count


__all__ = ["MISSING_MARKERS", "DEFAULT_MISSING_MARKERS", "count_missing"]
