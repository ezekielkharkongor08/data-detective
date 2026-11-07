"""LLM context builder: a compact, token-budgeted digest of the profile.

The full profile stays in state, the database, and the report. The LLM only
receives this digest, which:

- groups columns by semantic type as capped name lists (no stats, no
  sample rows for the agent),
- lists the top missing columns,
- flags ambiguous columns with a reason.

When the digest exceeds ``budget_tokens`` it degrades in a fixed order so
small local models always receive something bounded:

1. drop ambiguous-column reasons
2. shrink per-type name lists (25 -> 10 -> 5 names)
3. drop text columns from the name lists
4. drop ambiguous columns entirely
5. drop per-type name lists entirely (keep type counts only)

Token estimation is deliberately conservative (1 token ~= 4 characters)
so the real prompt is almost always smaller than the budget.
"""

from __future__ import annotations

import json
from typing import Any

# Conservative estimate: 1 token ~= 4 characters.
CHARS_PER_TOKEN = 4

# Default token budget for the digest sent to the LLM.
DEFAULT_BUDGET_TOKENS = 2000


def _estimate_tokens(text: str) -> int:
    """Return a conservative token count for ``text``."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _dump(obj: Any) -> str:
    """Serialize to compact JSON for token estimation."""
    return json.dumps(obj, default=str, separators=(",", ":"))


def build_llm_digest(profile: dict[str, Any], budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> dict[str, Any]:
    """Build a token-budgeted digest of ``profile`` for the LLM.

    The returned dict is JSON-friendly and contains no per-column stats or
    sample rows. The full ``profile`` is left untouched in state, the DB,
    and the report.

    Args:
        profile: The full profiling dict (unchanged).
        budget_tokens: Max estimated tokens for the digest.

    Returns:
        A dict with keys ``n_rows``, ``n_cols``, ``target_type``,
        ``columns_by_type``, ``type_counts``, ``top_missing``,
        ``ambiguous``, and ``truncated``.
    """
    columns = profile.get("columns", []) or []
    semantic_lookup = {
        c.get("name"): (c.get("semantic") or {}) for c in columns
    }

    # Group columns by semantic type, preserving profile order.
    by_type: dict[str, list[str]] = {}
    for c in columns:
        name = c.get("name")
        if not name:
            continue
        sem = semantic_lookup.get(name, {})
        t = sem.get("type", "unknown")
        by_type.setdefault(t, []).append(name)

    # Top missing columns (by missing_pct desc, then name for determinism).
    missing_cols = sorted(
        columns,
        key=lambda c: (-float(c.get("missing_pct", 0) or 0), str(c.get("name", ""))),
    )
    top_missing = [
        {"name": c.get("name"), "missing_pct": c.get("missing_pct", 0)}
        for c in missing_cols
        if float(c.get("missing_pct", 0) or 0) > 0
    ][:10]

    # Ambiguous columns with their leading reason.
    ambiguous_all = []
    for c in columns:
        name = c.get("name")
        if not name:
            continue
        sem = semantic_lookup.get(name, {})
        if sem.get("ambiguous"):
            reasons = sem.get("reasons") or ["unclear"]
            ambiguous_all.append(
                {
                    "name": name,
                    "type": sem.get("type", "unknown"),
                    "reason": str(reasons[0]),
                }
            )

    # Degradation knobs, applied in fixed order when over budget.
    type_cap = 25        # names per semantic type
    include_text = True  # include 'text' columns in name lists
    include_ambiguous = True
    include_ambiguous_reasons = True
    include_type_lists = True

    def build() -> dict[str, Any]:
        columns_by_type: dict[str, list[str]] = {}
        if include_type_lists:
            for t, names in by_type.items():
                if t == "text" and not include_text:
                    continue
                cap = type_cap if t != "unknown" else max(1, type_cap // 3)
                columns_by_type[t] = names[:cap]

        ambiguous_out = []
        if include_ambiguous:
            for a in ambiguous_all:
                entry: dict[str, Any] = {"name": a["name"], "type": a["type"]}
                if include_ambiguous_reasons:
                    entry["reason"] = a["reason"]
                ambiguous_out.append(entry)

        return {
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "target_type": profile.get("target_type", "unknown"),
            "columns_by_type": columns_by_type,
            "type_counts": {t: len(names) for t, names in by_type.items()},
            "top_missing": top_missing,
            "ambiguous": ambiguous_out,
            "truncated": False,
        }

    digest = build()
    while _estimate_tokens(_dump(digest)) > budget_tokens:
        if include_ambiguous_reasons:
            include_ambiguous_reasons = False
            digest = build()
            continue
        if type_cap > 5:
            type_cap = max(5, type_cap // 2)
            digest = build()
            continue
        if include_text:
            include_text = False
            digest = build()
            continue
        if include_ambiguous:
            include_ambiguous = False
            digest = build()
            continue
        if include_type_lists:
            include_type_lists = False
            digest = build()
            continue
        # Nothing left to degrade.
        digest["truncated"] = True
        break

    return digest


__all__ = ["build_llm_digest", "DEFAULT_BUDGET_TOKENS"]
