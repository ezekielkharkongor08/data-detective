"""Unit tests for the idempotent findings reducer.

The reducer is the safety net that stops a repeated finding from appearing
twice when a check runs more than once (for example on a thread re-run).
"""

from src.graph.reducers import (
    finding_dedup_key,
    merge_findings,
    stable_evidence_key,
)


def _finding(
    check_name: str,
    columns: list[str],
    evidence: str,
    severity: str = "medium",
) -> dict:
    """Build a serialized Finding-like dict for tests."""
    return {
        "check_name": check_name,
        "severity": severity,
        "evidence": evidence,
        "affected_columns": columns,
        "confidence": 1.0,
    }


# ── stable_evidence_key ───────────────────────────────────────────────

def test_stable_evidence_key_ignores_surrounding_whitespace():
    assert stable_evidence_key("  same text \n") == stable_evidence_key("same text")


def test_stable_evidence_key_is_deterministic():
    assert stable_evidence_key("abc") == stable_evidence_key("abc")


def test_stable_evidence_key_differs_for_different_text():
    assert stable_evidence_key("abc") != stable_evidence_key("abd")


def test_stable_evidence_key_handles_none_and_empty():
    assert stable_evidence_key(None) == stable_evidence_key("")


# ── finding_dedup_key ─────────────────────────────────────────────────

def test_finding_dedup_key_is_column_order_insensitive():
    a = _finding("missing_values", ["c", "a"], "e")
    b = _finding("missing_values", ["a", "c"], "e")
    assert finding_dedup_key(a) == finding_dedup_key(b)


def test_finding_dedup_key_differs_by_check_name():
    a = _finding("duplicates", ["x"], "e")
    b = _finding("missing_values", ["x"], "e")
    assert finding_dedup_key(a) != finding_dedup_key(b)


def test_finding_dedup_key_differs_by_evidence():
    a = _finding("duplicates", [], "2 duplicates")
    b = _finding("duplicates", [], "3 duplicates")
    assert finding_dedup_key(a) != finding_dedup_key(b)


# ── merge_findings ────────────────────────────────────────────────────

def test_merge_findings_dedupes_exact_duplicates():
    finding = _finding("duplicates", [], "Found 1 duplicate row.")
    assert merge_findings([finding], [dict(finding)]) == [finding]


def test_merge_findings_dedupes_across_copies_with_whitespace():
    a = _finding("duplicates", [], "Found 1 duplicate row.")
    b = _finding("duplicates", [], "  Found 1 duplicate row.  ")
    assert len(merge_findings([a], [b])) == 1


def test_merge_findings_column_order_is_irrelevant():
    a = _finding("missing_values", ["a", "b"], "x")
    b = _finding("missing_values", ["b", "a"], "x")
    assert len(merge_findings([a], [b])) == 1


def test_merge_findings_keeps_distinct_columns():
    a = _finding("missing_values", ["a"], "x")
    b = _finding("missing_values", ["b"], "x")
    assert len(merge_findings([a], [b])) == 2


def test_merge_findings_keeps_distinct_evidence():
    a = _finding("duplicates", [], "2 duplicates")
    b = _finding("duplicates", [], "3 duplicates")
    assert len(merge_findings([a], [b])) == 2


def test_merge_findings_keeps_first_occurrence_order():
    first = _finding("a", ["x"], "e1")
    second = _finding("b", ["y"], "e2")
    merged = merge_findings([first], [second, dict(first)])
    assert merged == [first, second]


def test_merge_findings_is_idempotent():
    findings = [
        _finding("a", ["x"], "e1"),
        _finding("b", ["y"], "e2"),
    ]
    once = merge_findings([], findings)
    twice = merge_findings(once, findings)
    assert twice == once


def test_merge_findings_handles_none():
    finding = _finding("a", [], "e")
    assert merge_findings(None, [finding]) == [finding]
    assert merge_findings([finding], None) == [finding]
    assert merge_findings(None, None) == []


def test_merge_findings_accepts_empty_lists():
    assert merge_findings([], []) == []


__all__ = [
    "test_stable_evidence_key_ignores_surrounding_whitespace",
    "test_stable_evidence_key_is_deterministic",
    "test_stable_evidence_key_differs_for_different_text",
    "test_stable_evidence_key_handles_none_and_empty",
    "test_finding_dedup_key_is_column_order_insensitive",
    "test_finding_dedup_key_differs_by_check_name",
    "test_finding_dedup_key_differs_by_evidence",
    "test_merge_findings_dedupes_exact_duplicates",
    "test_merge_findings_dedupes_across_copies_with_whitespace",
    "test_merge_findings_column_order_is_irrelevant",
    "test_merge_findings_keeps_distinct_columns",
    "test_merge_findings_keeps_distinct_evidence",
    "test_merge_findings_keeps_first_occurrence_order",
    "test_merge_findings_is_idempotent",
    "test_merge_findings_handles_none",
    "test_merge_findings_accepts_empty_lists",
]
