"""Tests for robust CSV reading (src/tools/csv_reader.py).

Covers encoding fallback (non-UTF-8 files) and delimiter detection
(tab, semicolon, pipe) so unknown CSVs never crash the audit.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.tools.csv_reader import read_csv_robust


def _write_bytes(content: bytes, suffix: str = ".csv") -> str:
    path = os.path.join(tempfile.mkdtemp(), f"sample{suffix}")
    with open(path, "wb") as f:
        f.write(content)
    return path


def _write_text(content: str, encoding: str = "utf-8", suffix: str = ".csv") -> str:
    path = os.path.join(tempfile.mkdtemp(), f"sample{suffix}")
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(content)
    return path


# ── encoding fallback ─────────────────────────────────────────────────────

def test_read_csv_robust_utf8():
    """A plain UTF-8 CSV loads normally."""
    path = _write_text("a,b,c\n1,2,3\n4,5,6\n")
    df = read_csv_robust(path)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]


def test_read_csv_robust_latin1():
    """A Latin-1 encoded CSV (non-UTF-8 bytes) loads via fallback."""
    # 'caf\xc3\xa9' is valid UTF-8; use a byte that is NOT valid UTF-8.
    content = b"id,name\n1,Jos\xe9\n2,M\xe1ria\n"  # \xe9 = é in latin-1
    path = _write_bytes(content)
    df = read_csv_robust(path)
    assert len(df) == 2
    assert "Jos" in str(df["name"].iloc[0])


def test_read_csv_robust_utf8_sig_bom():
    """A UTF-8 file with BOM loads without the BOM polluting column names."""
    content = "\ufeffa,b\n1,2\n"
    path = _write_text(content, encoding="utf-8-sig")
    df = read_csv_robust(path)
    assert list(df.columns) == ["a", "b"]


def test_read_csv_robust_cp1252():
    """A cp1252-encoded CSV loads via fallback."""
    content = b"id,product\n1,Caf\xe9\n2,Na\xefve\n"  # \xe9 = é, \xef = ï in cp1252
    path = _write_bytes(content)
    df = read_csv_robust(path)
    assert len(df) == 2


# ── delimiter detection ───────────────────────────────────────────────────

def test_read_csv_robust_tab_delimited():
    """A tab-separated file is detected and parsed."""
    content = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
    path = _write_text(content)
    df = read_csv_robust(path)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]


def test_read_csv_robust_semicolon_delimited():
    """A semicolon-separated file (common in Europe) is detected."""
    content = "a;b;c\n1;2;3\n4;5;6\n"
    path = _write_text(content)
    df = read_csv_robust(path)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]


def test_read_csv_robust_pipe_delimited():
    """A pipe-separated file is detected."""
    content = "a|b|c\n1|2|3\n4|5|6\n"
    path = _write_text(content)
    df = read_csv_robust(path)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]


def test_read_csv_robust_comma_still_default():
    """Comma remains the default when no other delimiter fits better."""
    content = "a,b,c\n1,2,3\n4,5,6\n"
    path = _write_text(content)
    df = read_csv_robust(path)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]


# ── errors ────────────────────────────────────────────────────────────────

def test_read_csv_robust_nonexistent_file():
    """A missing file raises a clear error."""
    with pytest.raises((FileNotFoundError, OSError)):
        read_csv_robust("/tmp/does_not_exist_xyz123.csv")


__all__ = []
