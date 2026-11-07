"""Robust CSV reading with encoding fallback and delimiter detection.

``read_csv_robust`` tries several encodings and delimiters so oddly saved
CSVs never crash the audit, and caches the parsed DataFrame per path
(callers get an isolated copy so parallel nodes can't mutate the original).
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache
from typing import Optional

import pandas as pd

# Encodings tried in order. utf-8 first (most common), then BOM-aware utf-8,
# then latin-1 / cp1252 which can decode any byte sequence.
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

# Delimiters tried in order. csv.Sniffer is avoided because it is unreliable
# on short or irregular files.
_DELIMITERS = (",", "\t", ";", "|")


def _detect_delimiter(sample: str) -> str:
    """Return the delimiter that gives the most consistent column count."""
    best_delimiter = ","
    best_score = -1
    for delimiter in _DELIMITERS:
        try:
            reader = csv.reader(io.StringIO(sample), delimiter=delimiter)
            col_counts = [len(row) for row in reader]
        except csv.Error:
            continue
        if not col_counts:
            continue
        # Score: most rows agree on a column count > 1.
        mode_count = max(set(col_counts), key=col_counts.count)
        consistency = col_counts.count(mode_count) / len(col_counts)
        score = consistency * (mode_count if mode_count > 1 else 0)
        if score > best_score:
            best_score = score
            best_delimiter = delimiter
    return best_delimiter


@lru_cache(maxsize=3)
def _read_csv_robust_impl(csv_path: str) -> pd.DataFrame:
    """Read a CSV from disk with encoding fallback and delimiter detection.

    Heavy lifting runs once per unique path; later calls return the cached
    DataFrame. The public wrapper returns an isolated copy so callers cannot
    mutate the cache.
    """
    last_error: Optional[Exception] = None

    # Read raw bytes once so we can try multiple encodings on the same content.
    with open(csv_path, "rb") as f:
        raw_bytes = f.read()

    for encoding in _ENCODINGS:
        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        # Strip a BOM if present (utf-8 decodes BOM bytes but leaves them in
        # the text, which would pollute the first column name).
        if text.startswith("\ufeff"):
            text = text[1:]

        # Detect delimiter from the first 8KB (enough for header + several rows).
        sample = text[:8192]
        delimiter = _detect_delimiter(sample)

        try:
            df = pd.read_csv(io.StringIO(text), delimiter=delimiter)
        except Exception as exc:  # noqa: BLE001 - try next encoding/delimiter
            last_error = exc
            continue

        # Strip leading/trailing whitespace from column names so that CSVs
        # saved by Excel or other tools with padded headers load cleanly.
        df.columns = df.columns.str.strip()

        if df.shape[1] >= 1:
            return df

        last_error = pd.errors.ParserError(
            f"Could not parse {csv_path} with encoding {encoding!r} "
            f"and delimiter {delimiter!r}"
        )

    if last_error is not None:
        raise last_error  # type: ignore[misc]
    raise ValueError(f"Could not read CSV: {csv_path}")


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    """Read a CSV with encoding fallback and delimiter detection.

    Returns an isolated copy of the cached DataFrame for ``csv_path``;
    raises the last error if every encoding fails.
    """
    df = _read_csv_robust_impl(csv_path)
    return df.copy()


__all__ = ["read_csv_robust"]
