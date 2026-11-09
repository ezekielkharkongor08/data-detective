"""LLM synthesis logic for Data Detective.

Extracts the summary-generation logic from ``nodes.py`` into a dedicated
module so that ``nodes.py`` stays focused on graph node wiring.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

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
    "You are a data quality auditor. You receive JSON with dataset facts and the findings of automated checks. Your reply is the audit summary and nothing else. Never repeat or list these instructions.\n\n"
    "Rules:\n"
    "- Take every fact from the JSON: counts, the target column, column names, percentages and statistics. Copy numbers exactly. Never compute or recall statistics. Never claim that something is absent (for example, no leakage), because the JSON does not say so.\n"
    "- Write one bullet per finding, in the order given (most severe first). Never merge different severities.\n"
    "- Findings with severity INFO and no columns are notes. Put them last, one short bullet each, keeping the column names from the evidence, saying only what the finding says and ending with Next step: None.\n"
    "- A finding with check bias is a fairness finding about outcome rates between groups; never call it leakage. Use the word leakage only for a finding whose check is target_leakage.\n"
    "- Make each next step specific to that bullet's columns and numbers. Different findings get different advice. If a bullet covers several columns, give advice that fits each one. Do not add filler such as to avoid potential bias unless the check is bias.\n"
    "- Advice by check. outliers: review the flagged rows and check whether the values are legitimate, then consider a transform or robust scaling; never suggest removing the column or deleting rows. skewness: consider a log or power transform; tree-based models are less sensitive. missing_values: read the percentage in the evidence. For a small or moderate share, impute inside the modeling pipeline. For a very large share (more than half the rows), first add an indicator flag to test whether missingness carries signal, and only then consider dropping the column. Never mention these thresholds or percentages-as-rules in the advice. id_like: exclude the columns from model features; identifiers are not predictive and this is not leakage. class_imbalance: class weights, resampling, or PR-AUC and F1. target_leakage: investigate the column's relation to the target and exclude it if it would not be known at prediction time. bias: review outcome rates between the groups and whether the gap comes from the data or the target definition; do not recommend deleting the column. In the bullet, give the group rates and the disparate impact ratio from the evidence. duplicates: check they are legitimate before removing. schema: convert the columns to the numeric type they appear to hold. Other checks: practical advice based only on that finding's evidence.\n"
    "- Length: about 400-550 words in total. The overview is the longest paragraph; each bullet has two or three sentences.\n\n"
    "Reply in exactly this layout, with a blank line between the two sections and each bullet on its own line:\n\n"
    "- Describe each check in plain words in the text, for example identifier-like columns instead of id_like, and never write the raw check name.\n"
    "OVERVIEW: <One paragraph of about 8 to 10 sentences, written in this order. (1) What the dataset appears to be about, its rows and columns, and the target column if given. (2) The mix of column types, using the type counts in the JSON profile. (3) The total number of findings, how many there are at each severity, and the highest severity, from the JSON facts. (4) One sentence per severity level that has findings, most severe first, naming the affected columns and giving the key numbers (percentages, counts, ratios, rate differences). (5) One sentence on skipped checks or automatic runs, only if such findings exist. (6) A closing sentence on overall data quality. No advice in the overview.>\n\n"
    "KEY ISSUES & NEXT STEPS:\n"
    "- [SEVERITY] <what was found, naming the affected columns and keeping the key numbers>. Next step: <one or two sentences of specific advice.>\n\n"
    "Write each severity in capitals inside square brackets: [CRITICAL], [HIGH], [MEDIUM], [LOW], [INFO]. Begin with OVERVIEW: and stop after the last bullet."
)

_STRUCTURED_SUFFIX = (
    "\n\nPut the whole text in the `summary` field and write every line break "
    "as \\n. Never return only a heading."
)
_PLAIN_SUFFIX = "\n\nReply with plain text only. Do not use JSON."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sev_name(finding: dict) -> str:
    """Return the severity of a finding dict as an upper-case string."""
    sev = finding.get("severity")
    sev = getattr(sev, "value", sev)
    return str(sev).upper()


def _sort_findings(findings: list) -> list:
    """Sort findings by severity (highest first), then check name."""
    # Two-pass stable sort: name first, then severity, so ties keep name order.
    by_name = sorted(findings, key=lambda f: str(f.get("check_name", "")))
    finding_objs = [Finding.model_validate(f) for f in by_name]
    finding_objs.sort()  # highest severity first via Finding.__lt__
    return [f.model_dump(mode="json") for f in finding_objs]


def _group_findings(findings: list[dict]) -> list[dict]:
    """Aggregate findings by (check_name, severity) so the LLM gets one
    bullet per check type per severity level.

    Columns are merged into one deduplicated list, evidence strings are
    joined for all facts, and the highest confidence is kept.
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

        # Column merge: deduplicated, order-preserving.
        all_cols: list[str] = []
        seen_cols: set[str] = set()
        for i in items:
            for col in i.get("affected_columns") or []:
                if col not in seen_cols:
                    seen_cols.add(col)
                    all_cols.append(col)

        # Combine evidence strings so the LLM keeps all numerical facts.
        evidence_parts = [
            str(i.get("evidence", "")).strip()
            for i in items
            if i.get("evidence")
        ]
        combined_evidence = " ".join(evidence_parts)

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


@traceable(name="build_llm_payload", run_type="chain")
def _highest_severity(findings: list[dict]) -> Severity:
    """Return the highest severity among findings (INFO when empty)."""
    if not findings:
        return Severity.INFO
    finding_objs = [Finding.model_validate(f) for f in findings]
    return min(finding_objs).severity


def _build_payload(profile: dict, findings: list) -> str:
    """Build the JSON payload sent to the LLM.

    Findings are aggregated by (check_name, severity) before being sent, so
    the LLM receives one bullet per check type per severity level and can
    never emit duplicate bullets for the same check+severity combination.
    """
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

    @traceable(name="synthesize_findings", run_type="chain")
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
