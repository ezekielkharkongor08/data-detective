"""State reducers for the Data Detective graph.

LangGraph applies a reducer whenever a node returns a field declared as
``Annotated[T, reducer]``. The ``findings`` list must be **idempotent**: a
check may run more than once (for example when a thread is re-invoked), and
the same finding must never appear twice in the report.

Why this exists
---------------
Before this module the field used ``operator.add``, which only concatenates.
Re-invoking a thread that already held checkpointed findings appended a
second copy of every finding. This reducer is the safety net: even if a check
runs twice, the report sees each finding once.

The dedupe key is ``(check_name, sorted(affected_columns), sha256(evidence))``.
Two findings with the same check, the same affected columns (in any order),
and the same evidence are the same finding.
"""

from __future__ import annotations

import hashlib


def stable_evidence_key(evidence: str | None) -> str:
    """Return a deterministic hash of a finding's evidence text.

    Surrounding whitespace is stripped so trivially different copies of the
    same evidence collapse to one key. The hash keeps the key short and avoids
    holding long evidence strings in a set.

    Args:
        evidence: The human-readable evidence string (may be ``None``).

    Returns:
        A lowercase SHA-256 hex digest of the stripped evidence.
    """
    text = (evidence or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finding_dedup_key(finding: dict) -> tuple[str, tuple[str, ...], str]:
    """Build the identity key used to decide whether two findings are equal.

    Args:
        finding: A serialized :class:`~src.schemas.finding.Finding` dict.

    Returns:
        ``(check_name, sorted affected columns, evidence hash)``.
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

    The first occurrence wins, so checkpointed findings keep their original
    order and severity. Applying this reducer twice to the same inputs returns
    the same list (idempotent).

    Args:
        existing: Findings already in state (may be ``None``).
        new: Findings returned by a node (may be ``None``).

    Returns:
        A deduplicated list of finding dicts.
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
