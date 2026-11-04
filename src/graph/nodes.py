"""Graph nodes: load_csv -> profile -> [parallel checks] -> merge_findings ->
router -> [deep_checks?] -> report.

Each node takes the current GraphState and returns a dict of field updates
to merge into the state. Calculation logic is kept in pure helper functions
so it can be tested without LangGraph.
"""

import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.agent import run_checks_with_agent
from src.graph.state import GraphState
from src.schemas.finding import Finding, Severity
from src.schemas.report import AuditReport, SynthesisResult
from src.tools.checks import (
    duplicates_check,
    id_like_check,
    imbalance_check,
    missing_check,
    schema_check,
)
from src.tools.checks.leakage_check import leakage_check
from src.tools.profiling import profile_dataset


def load_csv_node(state: GraphState) -> GraphState:
    """Validate the CSV file exists and record its path + dataset id."""
    csv_path = state.get("csv_path")
    if not csv_path:
        return {"error": "No csv_path provided in state."}

    path = Path(csv_path)
    if not path.exists():
        return {"error": f"CSV file not found: {csv_path}"}

    return {
        "csv_path": str(path),
        "dataset_id": path.stem,
        "error": None,
    }


def _finding_to_dict(f: Finding) -> dict:
    """Convert a :class:`Finding` to a plain dict for checkpointer serialization."""
    return f.model_dump(mode="json")


def _wrap_check(check_fn: Callable[..., list[Finding]], state: GraphState) -> dict:
    """Run a check function safely so one failure can't crash the parallel step.

    Returns a dict with a ``findings`` key containing the check's findings as
    plain dicts (serializable by the LangGraph checkpointer) or a single error
    finding if the check raised.
    """
    csv_path = state.get("csv_path")
    target_column = state.get("target_column")
    try:
        findings = check_fn(csv_path=csv_path, target_column=target_column)
    except Exception as exc:  # noqa: BLE001 — wrapper safety
        findings = [
            Finding(
                check_name=getattr(check_fn, "__name__", "unknown"),
                severity=Severity.HIGH,
                evidence=f"Check failed with error: {exc}",
                affected_columns=[],
                suggested_fix="Investigate the check implementation.",
                confidence=0.0,
            )
        ]
    return {"findings": [_finding_to_dict(f) for f in findings]}


def make_check_node(check_fn: Callable[..., list[Finding]]) -> Callable[[GraphState], dict]:
    """Factory that wraps a check function into a LangGraph node."""
    def _node(state: GraphState) -> dict:
        return _wrap_check(check_fn, state)
    _node.__name__ = f"check_{getattr(check_fn, '__name__', 'unknown')}"
    return _node


def merge_findings_node(state: GraphState) -> GraphState:
    """Merge findings from all parallel check branches.

    Relies on the ``Annotated[list[Finding], operator.add]`` reducer in
    GraphState: LangGraph already concatenates the lists. This node is a
    no-op passthrough that ensures the merge step exists in the graph.
    """
    return {}


def router_node(state: GraphState) -> GraphState:
    """Decide whether to run deep leakage checks.

    Examines merged findings. If any finding suggests a column could be
    leaking the target (e.g., an ID-like column or a text column that is
    mostly numeric), the graph routes to deep_checks. Otherwise it skips
    straight to the report.

    Sets ``leakage_suspected`` and ``suspect_columns`` in state.
    """
    findings = state.get("findings") or []
    target_column = state.get("target_column")

    suspect_columns: list[str] = []
    for f in findings:
        # ID-like columns are the most common leakage vector.
        if f["check_name"] == "id_like":
            for col in f["affected_columns"]:
                if col != target_column and col not in suspect_columns:
                    suspect_columns.append(col)
        # A text column that is mostly numeric may be target encoded as text.
        elif f["check_name"] == "schema":
            for col in f["affected_columns"]:
                if col != target_column and col not in suspect_columns:
                    suspect_columns.append(col)

    return {
        "leakage_suspected": len(suspect_columns) > 0,
        "suspect_columns": suspect_columns,
    }


def deep_checks_node(state: GraphState) -> GraphState:
    """Run deep leakage check on all columns.

    When the router suspects leakage, runs leakage_check across all
    non-target columns to find the actual leaking column(s). Results are
    stored in ``deep_check_results`` and any leakage findings are returned
    to be merged into the findings list via the operator.add reducer.
    """
    csv_path = state.get("csv_path")
    target_column = state.get("target_column")

    if not csv_path or not target_column:
        return {"deep_check_results": []}

    # Test all columns for leakage, not just the ones the router flagged.
    # The router's job is to decide *whether* to run deep checks; the
    # deep check itself examines every column.
    findings = leakage_check(
        csv_path=csv_path,
        target_column=target_column,
        suspicious_columns=None,
    )

    results = [
        {
            "column": f.affected_columns[0] if f.affected_columns else "?",
            "severity": f.severity.value,
            "evidence": f.evidence,
        }
        for f in findings
    ]

    return {
        "deep_check_results": results,
        "findings": [_finding_to_dict(f) for f in findings],
    }


_TRUSTED_SEVERITY_ORDER = [
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
]


def _highest_severity(findings: list[dict]) -> Severity:
    """Return the highest severity among findings (INFO when empty).

    ``findings`` is a list of plain dicts (serialized Finding objects).
    """
    if not findings:
        return Severity.INFO
    best = max(
        findings,
        key=lambda f: _TRUSTED_SEVERITY_ORDER.index(Severity(f["severity"])),
    )
    return Severity(best["severity"])


def _run_all_checks_deterministically(state: GraphState) -> list[dict]:
    """Fallback: run every check without the LLM (used if the agent fails).

    Returns findings as plain dicts for checkpointer serialization.
    """
    csv_path = state.get("csv_path")
    target_column = state.get("target_column")
    if not csv_path:
        return []

    findings: list[Finding] = []
    findings += duplicates_check(csv_path)
    findings += missing_check(csv_path)
    findings += schema_check(csv_path)
    findings += imbalance_check(csv_path, target_column=target_column)
    findings += id_like_check(csv_path, target_column=target_column)
    if target_column:
        findings += leakage_check(csv_path, target_column=target_column)
    return [_finding_to_dict(f) for f in findings]


def make_agent_checks_node(llm: Any) -> Callable[[GraphState], GraphState]:
    """Factory for the M4 agent node that chooses and runs checks via the LLM."""

    def agent_checks_node(state: GraphState) -> GraphState:
        csv_path = state.get("csv_path")
        if not csv_path:
            return {"error": "No csv_path in state; cannot run checks."}

        profile = state.get("profile") or {}
        try:
            findings, _messages = run_checks_with_agent(
                csv_path=csv_path,
                target_column=state.get("target_column"),
                profile=profile,
                llm=llm,
            )
            findings = [_finding_to_dict(f) for f in findings]
        except Exception as exc:  # noqa: BLE001 — LLM may be unreachable
            # Fall back to deterministic checks so the audit still completes.
            findings = _run_all_checks_deterministically(state)
            findings.append(
                _finding_to_dict(
                    Finding(
                        check_name="agent",
                        severity=Severity.INFO,
                        evidence=(
                            "LLM agent unavailable "
                            f"({type(exc).__name__}: {exc}); ran deterministic checks."
                        ),
                        affected_columns=[],
                        suggested_fix="Check the LLM settings in .env and that Ollama is running.",
                        confidence=1.0,
                    )
                )
            )

        if not findings:
            # The model finished without a single finding. Re-run the checks
            # in Python so a real problem is never silently missed.
            findings = _run_all_checks_deterministically(state)

        return {"findings": findings, "error": None}

    return agent_checks_node


def make_synthesize_findings_node(llm: Any) -> Callable[[GraphState], GraphState]:
    """Factory for the node that asks the LLM for a structured summary."""

    def synthesize_findings_node(state: GraphState) -> GraphState:
        findings = state.get("findings") or []
        profile = state.get("profile") or {}

        system = SystemMessage(
            content=(
                "You are Data Detective. Given a dataset profile and a list "
                "of findings, write a DETAILED plain-English overview. "
                "Put the whole text in the `summary` field: at least 150 words, "
                "3-5 short paragraphs or bullet lists. Never return only a heading. "
                "Cover: (1) what the dataset appears to be and its shape (rows/columns); "
                "(2) what the column groups contain, using their real names; "
                "(3) every problem found, why it matters, and which columns it hits; "
                "(4) concrete next steps. "
                "Only describe what the profile and findings actually show."
            )
        )

        payload = json.dumps(
            {
                "profile": profile,
                "findings": findings,
            },
            default=str,
        )

        try:
            structured = llm.with_structured_output(SynthesisResult)
            result = structured.invoke([system, HumanMessage(content=payload)])
        except Exception as exc:  # noqa: BLE001 — structured output may fail
            return {
                "summary": f"(LLM summary unavailable: {type(exc).__name__}: {exc})",
                "overall_severity": _highest_severity(findings).value,
            }

        summary = result.summary

        # JSON mode on small local models sometimes returns only a heading.
        # Ask again in plain text so the user still gets a real summary.
        if len(summary.strip()) < 80:
            try:
                raw = llm.invoke([system, HumanMessage(content=payload)])
                text = raw.content if isinstance(raw.content, str) else str(raw.content)
                if len(text.strip()) > len(summary.strip()):
                    summary = text
            except Exception:  # noqa: BLE001 - keep the short summary instead
                pass

        # Severity is computed from the findings, not trusted to the LLM.
        return {
            "summary": summary,
            "overall_severity": _highest_severity(findings).value,
        }

    return synthesize_findings_node


def render_text_report(report: AuditReport) -> str:
    """Render a plain-text report from an :class:`AuditReport` object.

    The report object is the single source of truth (also serialized to JSON).
    This function is just one view of it — findings are sorted by severity
    (highest first).
    """
    lines: list[str] = []
    lines.append("Data Detective — Profile Report")
    lines.append("=" * 32)
    lines.append(f"Dataset: {report.dataset_id}")
    lines.append(f"Rows: {report.rows}, Cols: {report.cols}")
    if report.duplicate_rows is not None:
        lines.append(f"Duplicate rows: {report.duplicate_rows}")
    lines.append("")

    if report.error:
        lines.append(f"Error: {report.error}")
        lines.append("")

    # LLM plain-language summary (M4). Indent every line so multi-line
    # detailed summaries stay visually nested under the heading.
    if report.summary:
        lines.append("Summary:")
        lines.append("-" * 32)
        for line in report.summary.splitlines():
            lines.append(f"  {line}" if line.strip() else "")
        lines.append("")

    # Findings section (M2 addition)
    if report.findings:
        sorted_findings = sorted(report.findings)
        lines.append("Findings:")
        lines.append("-" * 32)
        for f in sorted_findings:
            cols = ", ".join(f.affected_columns) if f.affected_columns else "(none)"
            lines.append(
                f"  [{f.severity.value.upper()}] {f.check_name}: {f.evidence} "
                f"(columns: {cols}, confidence: {f.confidence})"
            )
            if f.suggested_fix:
                lines.append(f"    -> {f.suggested_fix}")
        lines.append("")

    # Deep check results section (M3 addition)
    if report.deep_check_results:
        lines.append("Deep Leakage Check Results:")
        lines.append("-" * 32)
        for r in report.deep_check_results:
            lines.append(
                f"  [{r['severity'].upper()}] Column '{r['column']}': {r['evidence']}"
            )
        lines.append("")

    lines.append("Columns:")
    lines.append("-" * 32)
    for col in report.columns:
        name = col.get("name", "?")
        dtype = col.get("dtype", "?")
        missing = col.get("missing", 0)
        missing_pct = col.get("missing_pct", 0)
        markers = col.get("text_markers", 0)
        missing_txt = f"missing={missing:<4d} ({missing_pct}%)"
        if markers:
            missing_txt += f" [{markers} text marker(s)]"
        detail = ""
        if "stats" in col:
            s = col["stats"]
            detail = f"stats=min={s['min']} max={s['max']} mean={s['mean']} std={s['std']}"
        elif col.get("type_category") == "boolean":
            detail = f"boolean true_count={col.get('true_count')} true_pct={col.get('true_pct')}%"
        else:
            detail = f"unique={col.get('unique')} sample={col.get('sample_values')}"
        lines.append(
            f"  {name:<20s} | {dtype:<12s} | {missing_txt} | {detail}"
        )

    lines.append("")
    lines.append("Sample rows (first 3):")
    lines.append("-" * 32)
    for row in report.sample_rows:
        lines.append(f"  {row}")

    return "\n".join(lines)


def profile_node(state: GraphState) -> GraphState:
    """Build the compact profile from the CSV on disk."""
    csv_path = state.get("csv_path")
    if not csv_path:
        return {"error": "No csv_path in state; cannot profile."}

    try:
        profile = profile_dataset(csv_path)
    except Exception as exc:  # noqa: BLE001 — surface any load error in state
        return {"error": f"Failed to profile {csv_path}: {exc}"}

    return {"profile": profile, "error": None}


def report_node(state: GraphState) -> GraphState:
    """Render the profile + merged findings + deep check results as a report.

    Builds a single :class:`AuditReport` object — the source of truth — and
    stores both its JSON serialization (``state["report"]``) and a plain-text
    rendering (``state["report_text"]``).

    If profiling failed (no profile), still renders a minimal report so
    the error is surfaced to the user instead of being dropped.
    """
    profile = state.get("profile")
    dataset_id = state.get("dataset_id", "unknown")
    error = state.get("error")
    findings_raw = state.get("findings") or []
    # Reconstitute Finding objects from dicts for the AuditReport.
    findings = [Finding.model_validate(f) for f in findings_raw]
    deep_check_results = state.get("deep_check_results") or []
    summary = state.get("summary")

    report_obj = AuditReport(
        dataset_id=dataset_id,
        rows=profile.get("n_rows", 0) if profile else 0,
        cols=profile.get("n_cols", 0) if profile else 0,
        duplicate_rows=profile.get("duplicate_rows") if profile else None,
        summary=summary,
        overall_severity=state.get("overall_severity"),
        findings=findings,
        columns=profile.get("columns", []) if profile else [],
        sample_rows=profile.get("sample_rows", []) if profile else [],
        deep_check_results=deep_check_results or None,
        error=error,
    )

    return {
        "report": report_obj.to_json(),
        "report_text": render_text_report(report_obj),
    }
