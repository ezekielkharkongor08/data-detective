"""State reducers for the Data Detective graph.

LangGraph applies a reducer whenever a node returns an ``Annotated[T,
reducer]`` field. ``findings`` needs an idempotent reducer: re-invoking a
thread that already holds checkpointed findings would otherwise append a
second copy of every finding (plain concatenation is not enough).

The dedupe key is ``(check_name, sorted(affected_columns), sha256(evidence))``.
"""

from __future__ import annotations

import hashlib


def stable_evidence_key(evidence: str | None) -> str:
    """Return a deterministic SHA-256 hash of a finding's evidence text.

    Whitespace is stripped first so trivially different copies collapse to
    one key.
    """
    text = (evidence or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finding_dedup_key(finding: dict) -> tuple[str, tuple[str, ...], str]:
    """Build the identity key deciding whether two findings are equal.

    Returns ``(check_name, sorted affected columns, evidence hash)``.
    """
    return (
        str(finding.get("check_name", "")),
        tuple(sorted(finding.get("affected_columns") or [])),
        stable_evidence_key(finding.get("evidence")),
    )


def merge_findings(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list[dict]:
    """Idempotent reducer: concatenate findings and drop duplicates.

    First occurrence wins, so checkpointed findings keep their original order.
    Applying the reducer twice to the same inputs returns the same list.
    """
    merged: list[dict] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    for finding in (existing or []) + (new or []):
        key = finding_dedup_key(finding)
        if key in seen:
            continue
        seen.add(key)
        merged.append(finding)

    return merged


__all__ = ["stable_evidence_key", "finding_dedup_key", "merge_findings"]
