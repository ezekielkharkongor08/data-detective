"""Detect what kind of target a dataset has.

Checks declare the target types they support; the registry runner uses this
helper to skip a check with one informational note instead of crashing.
Types: binary, multiclass, continuous, boolean, unknown.
"""

from __future__ import annotations

import pandas as pd

from src.check_config import check_config

# A numeric target is treated as continuous when this share of its non-null
# values are distinct (for example salary or price).
# Value lives in the shared config: check_config.target_type.continuous_unique_ratio


def detect_target_type(series: pd.Series) -> str:
    """Return one of ``binary``, ``multiclass``, ``continuous``,
    ``boolean``, or ``unknown`` for ``series``."""
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


def is_classification_target(series: pd.Series) -> bool:
    """Return True when ``series`` is a classification target
    (binary, boolean, or multiclass).

    Class labels have no meaningful skew or outliers, so only the target
    column itself is excluded from the skewness and outlier checks rather
    than skipping the whole check.
    """
    return detect_target_type(series) in ("binary", "boolean", "multiclass")


__all__ = ["detect_target_type", "is_classification_target"]
