"""Detect what kind of target a dataset has.

Checks declare the target types they support (see
:mod:`src.tools.checks.registry`). The registry runner uses this helper to
skip a check with one informational note instead of crashing on an
unsupported target.

Target types
------------
- ``binary``: exactly two distinct non-null values (``0/1``, yes/no, True/False)
- ``multiclass``: more than two distinct values that are not continuous
- ``continuous``: numeric with many distinct values relative to row count
- ``boolean``: a real ``bool`` dtype
- ``unknown``: empty, all-null, or a single distinct value
"""

from __future__ import annotations

import pandas as pd

from src.check_config import check_config

# A numeric target is treated as continuous when this share of its non-null
# values are distinct (for example salary or price).
# Value lives in the shared config: check_config.target_type.continuous_unique_ratio


def detect_target_type(series: pd.Series) -> str:
    """Return the target type of ``series``.

    Args:
        series: The target column.

    Returns:
        One of ``binary``, ``multiclass``, ``continuous``, ``boolean``,
        ``unknown``.
    """
    clean = series.dropna()
    if clean.empty:
        return "unknown"

    if pd.api.types.is_bool_dtype(clean):
        return "boolean"

    nunique = int(clean.nunique())
    if nunique <= 1:
        return "unknown"
    if nunique == 2:
        return "binary"

    if pd.api.types.is_numeric_dtype(clean):
        if nunique / len(clean) > check_config.target_type.continuous_unique_ratio:
            return "continuous"
    return "multiclass"


__all__ = ["detect_target_type"]
