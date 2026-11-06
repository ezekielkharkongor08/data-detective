"""Tests for configurable missing markers (src/tools/missing_markers.py)."""

import os

import pandas as pd
import pytest

from src.tools.missing_markers import (
    DEFAULT_MISSING_MARKERS,
    MISSING_MARKERS,
    count_missing,
)


def test_default_missing_markers_contains_common():
    """Defaults include common markers like '?', 'NA', ''."""
    assert "?" in MISSING_MARKERS
    assert "NA" in MISSING_MARKERS
    assert "" in MISSING_MARKERS
    assert "-" in MISSING_MARKERS


def test_count_missing_counts_known_markers():
    """Known text markers count as missing."""
    series = pd.Series(["?", "Private", "NA", "Self-emp"], name="workclass")
    nan_count, marker_count = count_missing(series)
    assert nan_count == 0
    assert marker_count == 2  # '?' and 'NA'


def test_count_missing_extra_markers_parameter():
    """extra_markers are merged with defaults for one call."""
    series = pd.Series(["N/A", "NA", "missing", "Private"], name="workclass")
    nan_count, marker_count = count_missing(series, extra_markers=["missing"])
    assert marker_count == 3  # 'N/A', 'NA', 'missing'


def test_count_missing_numbers_never_matched():
    """Numeric columns never match text markers even if values look like them."""
    series = pd.Series([1, 2, 3, 4], name="num")
    nan_count, marker_count = count_missing(series)
    assert nan_count == 0
    assert marker_count == 0


def test_env_var_adds_extra_markers(monkeypatch):
    """DATA_DETECTIVE_MISSING_MARKERS env var adds markers."""
    # Reload the module to pick up the env var
    monkeypatch.setenv("DATA_DETECTIVE_MISSING_MARKERS", "my_marker,NOMARK")
    import importlib
    import src.tools.missing_markers as mm
    importlib.reload(mm)

    series = pd.Series(["my_marker", "NOMARK", "Private"], name="workclass")
    nan_count, marker_count = mm.count_missing(series)
    assert marker_count == 2  # 'my_marker' and 'NOMARK'

    # Restore
    importlib.reload(mm)


def test_default_markers_is_frozenset():
    """DEFAULT_MISSING_MARKERS is a frozenset (can't be mutated accidentally)."""
    assert isinstance(DEFAULT_MISSING_MARKERS, frozenset)


__all__ = []
