"""Pandas check functions and the check registry.

Importing this package registers the built-in checks. New check modules are
also discovered automatically by :func:`~src.tools.checks.registry.discover_checks`,
so adding a check never requires editing this file.
"""

from src.tools.checks.bias_check import bias_check
from src.tools.checks.duplicates_check import duplicates_check
from src.tools.checks.id_like_check import id_like_check
from src.tools.checks.imbalance_check import imbalance_check
from src.tools.checks.leakage_check import leakage_check
from src.tools.checks.missing_check import missing_check
from src.tools.checks.registry import (
    CheckSpec,
    discover_checks,
    error_finding,
    get_check,
    get_checks,
    info_note,
    register_check,
    run_check,
)
from src.tools.checks.schema_check import schema_check

__all__ = [
    # Check functions (direct imports still work).
    "bias_check",
    "duplicates_check",
    "id_like_check",
    "imbalance_check",
    "leakage_check",
    "missing_check",
    "schema_check",
    # Registry helpers.
    "CheckSpec",
    "register_check",
    "discover_checks",
    "get_checks",
    "get_check",
    "run_check",
    "info_note",
    "error_finding",
]
