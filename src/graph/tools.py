"""LangChain tool wrappers around the pandas checks.

Each tool has a name, a docstring the LLM reads to decide when to call it,
and typed arguments. Tools return the checks' findings as JSON strings so
they can travel through the agent's message history.

The LLM only ever receives these results and the compact profile — never the
full CSV (design rule #1).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from src.schemas.finding import Finding
from src.tools.checks import (
    duplicates_check,
    id_like_check,
    imbalance_check,
    missing_check,
    schema_check,
)
from src.tools.checks.bias_check import bias_check
from src.tools.checks.leakage_check import leakage_check


def _serialize(findings: list[Finding]) -> str:
    """Serialize findings to a JSON string for the agent message history."""
    return json.dumps([f.model_dump(mode="json") for f in findings])


def _parse(content: Any) -> list[Finding]:
    """Parse a tool message payload back into Finding objects."""
    if isinstance(content, str):
        data = json.loads(content)
    elif isinstance(content, list):
        # Some providers return content blocks; join text parts.
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
        data = json.loads(text) if text else []
    else:
        data = content or []
    return [Finding.model_validate(item) for item in data]


@tool
def run_duplicates_check(csv_path: str) -> str:
    """Check a CSV for duplicate rows.

    Flags MEDIUM when duplicates exist and HIGH when more than 10% of rows
    are duplicates. Returns findings as JSON (empty list means clean).
    """
    return _serialize(duplicates_check(csv_path))


@tool
def run_missing_check(csv_path: str) -> str:
    """Check a CSV for missing values (NaN, blanks, and '?' markers).

    Flags LOW above 0%, HIGH above 20%, and CRITICAL above 50% missing in a
    column. Returns findings as JSON (empty list means clean).
    """
    return _serialize(missing_check(csv_path))


@tool
def run_schema_check(csv_path: str) -> str:
    """Check a CSV for schema problems such as numbers stored as text.

    Flags MEDIUM when a text column is mostly numeric. Returns findings as
    JSON (empty list means clean).
    """
    return _serialize(schema_check(csv_path))


@tool
def run_imbalance_check(csv_path: str, target_column: str) -> str:
    """Check the class balance of a classification target column.

    Flags HIGH when the minority class is under 10% and MEDIUM under 30%.
    Returns findings as JSON (empty list means balanced).
    """
    return _serialize(imbalance_check(csv_path, target_column=target_column or None))


@tool
def run_id_like_check(csv_path: str, target_column: str = "") -> str:
    """Check for ID-like columns that are unique per row.

    Flags MEDIUM for any non-target column with a unique value in every row.
    Returns findings as JSON (empty list means clean).
    """
    return _serialize(id_like_check(csv_path, target_column=target_column or None))


@tool
def run_bias_check(csv_path: str, target_column: str = "") -> str:
    """Bias / fairness check: does a sensitive attribute predict the target?

    Auto-detects candidate sensitive columns (low-cardinality string columns
    such as gender, race) and trains a baseline model using each one alone to
    predict the target. Flags CRITICAL when a sensitive column alone reaches
    >90% accuracy (classification) or R^2 > 0.70 (regression). Returns
    findings as JSON (empty list means no bias detected).
    """
    return _serialize(bias_check(csv_path, target_column=target_column or None))


@tool
def run_leakage_check(csv_path: str, target_column: str) -> str:
    """Deep leakage check: does any single column alone predict the target?

    Trains a quick baseline model per column and flags CRITICAL when a
    column alone reaches >95% accuracy (classification) or R^2 > 0.90
    (regression). Returns findings as JSON (empty list means no leakage).
    """
    return _serialize(leakage_check(csv_path, target_column=target_column or None))


# Tools the agent may choose from (order is irrelevant to the LLM).
ALL_TOOLS = [
    run_duplicates_check,
    run_missing_check,
    run_schema_check,
    run_imbalance_check,
    run_id_like_check,
    run_bias_check,
    run_leakage_check,
]


__all__ = [
    "run_duplicates_check",
    "run_missing_check",
    "run_schema_check",
    "run_imbalance_check",
    "run_id_like_check",
    "run_bias_check",
    "run_leakage_check",
    "ALL_TOOLS",
    "_parse",
    "_serialize",
]
