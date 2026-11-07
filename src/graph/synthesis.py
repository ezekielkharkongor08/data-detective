"""LLM synthesis logic for Data Detective.

Extracts the summary-generation logic from ``nodes.py`` into a dedicated
module so that ``nodes.py`` stays focused on graph node wiring.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.llm_context import build_llm_digest, DEFAULT_BUDGET_TOKENS
from src.graph.state import GraphState
from src.schemas.finding import Finding, Severity
from src.schemas.report import AuditReport, SynthesisResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FINDINGS_IN_PROMPT = 15      # keep the prompt small for local models
MAX_EVIDENCE_CHARS = 400         # trim very long evidence strings
MIN_SUMMARY_WORDS = 40           # below this, treat the LLM output as unusable


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are Data Detective, a data-quality reviewer. You receive a JSON payload with 'profile', 'findings', and 'facts'.\n\n"
    "Write your response using EXACTLY this structure:\n\n"
    "OVERVIEW:\n"
    "<Write a detailed 3-4 sentence summary describing the ACTUAL dataset shape (rows/columns), what the data appears to be about, and its overall health. You MUST explicitly summarize any HIGH or CRITICAL severity issues (like target leakage) in this overview paragraph rather than calling the data healthy.>\n\n"
    "KEY ISSUES & NEXT STEPS:\n"
    "<Create EXACTLY ONE bullet point for EVERY distinct check_name in the findings. Merge all affected columns for that check into one bullet. DO NOT skip any findings.>\n"
    "- [SEVERITY] <Description of the problem including all affected columns and key numbers>. Next step: <Your own dynamic, specific advice based on those exact columns and numbers.>\n\n"
    "RULES:\n"
    "1. Separate the OVERVIEW and KEY ISSUES sections with a double newline (\\n\\n).\n"
    "2. Put each bullet on its own line (\\n).\n"
    "3. You MUST include every issue present in the findings. Do not drop any.\n"
    "4. Generate your own contextual 'Next step'.\n"
    "5. Do not invent numbers, columns, or sensitive/bias issues that are not in the JSON.\n"
)

_STRUCTURED_SUFFIX = (
    "\n8. Put the whole text in the `summary` field. Never return only a "
    "heading."
)
_PLAIN_SUFFIX = "\n8. Reply with plain text only. Do not use JSON."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sev_name(finding: dict) -> str:
    """Return the severity of a finding dict as an upper-case string."""
    sev = finding.get("severity")
    sev = getattr(sev, "value", sev)
    return str(sev).upper()


def _sort_findings(findings: list) -> list:
    """Sort findings by severity (highest first), then by check name.

    Uses :class:`Finding.__lt__` for severity ranking (backed by the
    :class:`Severity` enum declaration order). A stable two-pass sort
    preserves check_name ordering for ties.
    """
    # First sort by check name (secondary key).
    by_name = sorted(findings, key=lambda f: str(f.get("check_name", "")))
    # Then sort by severity using Finding.__lt__ (primary key).
    # Stable sort preserves check_name order for equal severities.
    finding_objs = [Finding.model_validate(f) for f in by_name]
    finding_objs.sort()  # Highest severity first via Finding.__lt__
    return [f.model_dump(mode="json") for f in finding_objs]


def _group_findings(findings: list[dict]) -> list[dict]:
    """Aggregate findings by check_name + severity so the LLM receives one
    bullet per check type per severity level.

    Grouping key is a tuple (check_name, severity) so that a LOW-severity
    missing_values finding never gets merged under a CRITICAL severity
    finding for the same check type.

    For findings with the same (check_name, severity):
    - Keep the severity from the group key.
    - Combine all affected_columns into a single deduplicated list.
    - Combine evidence strings so the LLM still gets all numerical facts.
    - Keep the highest confidence.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in findings:
        key = (str(f.get("check_name", "")), str(f.get("severity", "")))
        groups.setdefault(key, []).append(f)

    grouped: list[dict] = []
    for key, items in groups.items():
        if len(items) == 1:
            grouped.append(items[0])
            continue

        check_name, severity = key

        # Combine affected_columns (deduplicated, order-preserving).
        all_cols: list[str] = []
        seen_cols: set[str] = set()
        for i in items:
            for col in i.get("affected_columns") or []:
                if col not in seen_cols:
                    seen_cols.add(col)
                    all_cols.append(col)

        # Combine evidence strings so the LLM gets all numerical facts.
        evidence_parts = [
            str(i.get("evidence", "")).strip()
            for i in items
            if i.get("evidence")
        ]
        combined_evidence = " ".join(evidence_parts)

        # Highest confidence.
        best_confidence = max(float(i.get("confidence", 1.0)) for i in items)

        grouped.append(
            {
                "check_name": check_name,
                "severity": severity,
                "evidence": combined_evidence,
                "affected_columns": all_cols,
                "confidence": best_confidence,
            }
        )

    return grouped


def _compact_finding(f: dict) -> dict:
    """Shrink one finding to the fields the LLM needs."""
    evidence = str(f.get("evidence", ""))
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[:MAX_EVIDENCE_CHARS] + "..."
    return {
        "severity": _sev_name(f),
        "check": f.get("check_name"),
        "columns": f.get("affected_columns"),
        "evidence": evidence,
    }


def _build_payload(profile: dict, findings: list) -> str:
    """Build the JSON payload sent to the LLM.

    Findings are aggregated by (check_name, severity) before being sent, so
    the LLM receives one bullet per check type per severity level and can
    never emit duplicate bullets for the same check+severity combination.
    """
    from src.graph.nodes import _highest_severity

    grouped = _group_findings(findings)
    ordered = _sort_findings(grouped)
    shown = ordered[:MAX_FINDINGS_IN_PROMPT]
    counts: dict[str, int] = {}
    for f in ordered:
        name = _sev_name(f)
        counts[name] = counts.get(name, 0) + 1
    # The LLM receives a compact digest, never the full profile. The full
    # profile stays in state, the DB, and the report.
    payload = {
        "profile": build_llm_digest(profile, DEFAULT_BUDGET_TOKENS),
        "findings": [_compact_finding(f) for f in shown],
        "facts": {
            "total_findings": len(ordered),
            "findings_not_shown": max(0, len(ordered) - len(shown)),
            "severity_counts": counts,
            "highest_severity": _highest_severity(ordered).value,
        },
    }
    return json.dumps(payload, default=str)


def _fallback_summary(profile: dict, findings: list, reason: str) -> str:
    """Plain summary built without the LLM, so the user always gets something."""
    n_rows = profile.get("n_rows", "?")
    n_cols = profile.get("n_cols", "?")
    lines = [
        f"(Automatic summary - LLM output unavailable: {reason})",
        f"Overview: the dataset has {n_rows} rows and {n_cols} columns.",
    ]
    ordered = _sort_findings(findings)
    if not ordered:
        lines.append("The checks that ran found no issues.")
    else:
        lines.append("Key issues:")
        for f in ordered[:5]:
            lines.append(
                f"- [{_sev_name(f)}] {f.get('check_name')}: "
                f"{str(f.get('evidence', ''))[:200]}"
            )
        if len(ordered) > 5:
            lines.append(f"- ...and {len(ordered) - 5} more finding(s).")
    lines.append(f"Overall: highest severity is {_highest_severity(ordered).value}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_synthesize_findings_node(llm: Any) -> Callable[[GraphState], GraphState]:
    """Factory for the node that asks the LLM for a prioritized summary."""

    def synthesize_findings_node(state: GraphState) -> GraphState:
        findings = state.get("findings") or []
        profile = state.get("profile") or {}

        payload = _build_payload(profile, findings)
        human = HumanMessage(content=payload)
        structured_system = SystemMessage(content=_SYSTEM_PROMPT + _STRUCTURED_SUFFIX)
        plain_system = SystemMessage(content=_SYSTEM_PROMPT + _PLAIN_SUFFIX)

        summary = ""
        error_note = ""

        # Attempt 1: structured output (JSON mode).
        try:
            structured = llm.with_structured_output(SynthesisResult)
            result = structured.invoke([structured_system, human])
            summary = (result.summary or "").strip()
        except Exception as exc:  # noqa: BLE001 - structured output may fail
            error_note = f"{type(exc).__name__}"

        # Attempt 2: plain text, if the first reply is missing or too short.
        # Small local models sometimes return only a heading in JSON mode.
        if len(summary.split()) < MIN_SUMMARY_WORDS:
            try:
                raw = llm.invoke([plain_system, human])
                text = raw.content if isinstance(raw.content, str) else str(raw.content)
                text = text.strip()
                if len(text.split()) > len(summary.split()):
                    summary = text
            except Exception as exc:  # noqa: BLE001
                error_note = error_note or f"{type(exc).__name__}"

        # Attempt 3: build a summary from the findings, no LLM needed.
        if len(summary.split()) < MIN_SUMMARY_WORDS:
            summary = _fallback_summary(profile, findings, error_note or "reply too short")

        # Severity is computed from the findings, not trusted to the LLM.
        from src.graph.nodes import _highest_severity

        return {
            "summary": summary,
            "overall_severity": _highest_severity(findings).value,
        }

    return synthesize_findings_node


__all__ = [
    "make_synthesize_findings_node",
    "_build_payload",
    "_fallback_summary",
    "_group_findings",
    "_sort_findings",
]
