"""Graph nodes: load_csv -> profile -> [parallel checks] -> merge_findings ->
router -> [deep_checks?] -> report.

Each node takes the current GraphState and returns a dict of field updates
to merge into the state. Calculation logic is kept in pure helper functions
so it can be tested without LangGraph.

LLM synthesis logic (prompts, payload builders, fallback summary) lives in
:mod:`src.graph.synthesis`.
"""

import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.synthesis import make_synthesize_findings_node
from src.graph.state import GraphState
from src.schemas.finding import Finding, Severity
from src.schemas.report import AuditReport
from src.tools.checks import (
    CheckSpec,
    error_finding,
    get_check,
    get_checks,
    info_note,
    run_check,
)
from src.check_config import check_config
from src.tools.csv_reader import read_csv_robust
from src.tools.profiling import get_semantic_lookup, profile_dataset


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
    """Run a raw check function safely so one failure can't crash a step.

    Loads the CSV once and passes the pre-loaded DataFrame to the check,
    matching the new check signature ``(df, target_column=None, ...)``.
    Kept for direct callers and tests. Registry-driven nodes use
    :func:`make_check_node`, which also applies the registry contract.
    """
    csv_path = state.get("csv_path")
    if not csv_path or (isinstance(csv_path, str) and not csv_path.strip()):
        name = getattr(check_fn, "__name__", "unknown")
        return {
            "findings": [
                _finding_to_dict(error_finding(name, ValueError("No csv_path provided.")))
            ]
        }

    # Load the CSV once, centrally.
    try:
        df = read_csv_robust(csv_path)
    except Exception as exc:
        name = getattr(check_fn, "__name__", "unknown")
        return {"findings": [_finding_to_dict(error_finding(name, exc))]}

    target_column = state.get("target_column")
    try:
        findings = check_fn(df=df, target_column=target_column)
    except Exception as exc:  # noqa: BLE001 — wrapper safety
        name = getattr(check_fn, "__name__", "unknown")
        findings = [error_finding(name, exc)]
    return {"findings": [_finding_to_dict(f) for f in findings]}


def _run_registered_check(
    spec: CheckSpec,
    state: GraphState,
    run_fn: Callable[..., list[Finding]] | None = None,
) -> dict:
    """Run one registered check through the registry contract.

    Missing inputs and unsupported target types become one informational
    note; unexpected exceptions become one HIGH error finding. Either way the
    rest of the audit continues.
    """
    try:
        findings = run_check(spec, state, run_fn=run_fn)
    except Exception as exc:  # noqa: BLE001 — one check must not crash the run
        findings = [error_finding(spec.name, exc)]
    return {"findings": [_finding_to_dict(f) for f in findings]}


def make_check_node(
    spec: CheckSpec,
    run_fn: Callable[..., list[Finding]] | None = None,
) -> Callable[[GraphState], dict]:
    """Factory that wraps a registered check into a LangGraph node.

    ``run_fn`` lets tests patch ``builder.CHECKS`` without losing the
    registry metadata (required inputs, params, target gating).
    """
    def _node(state: GraphState) -> dict:
        return _run_registered_check(spec, state, run_fn=run_fn)
    _node.__name__ = f"check_{spec.name}"
    return _node


def merge_findings_node(state: GraphState) -> GraphState:
    """Merge findings from all parallel check branches.

    The actual merging is done by the ``merge_findings`` reducer in
    GraphState: LangGraph concatenates the lists and drops duplicates. This
    node is a no-op passthrough that gives the join an explicit step.
    """
    return {}


def _eligible_suspect_columns(
    csv_path: str,
    profile: dict[str, Any],
    target_column: str | None,
    max_count: int,
) -> tuple[list[str], int]:
    """Return feature columns eligible for deep leakage detection.

    Reads the CSV to compute nunique for high-uniqueness exclusion.
    Excludes the target column, identifier/constant/text columns, and
    high-uniqueness columns that leakage_check itself would skip.
    Returns ``(capped_columns, total_eligible)`` where ``capped_columns``
    holds at most ``max_count`` columns in profile order.
    """
    if not csv_path or not target_column or not profile:
        return [], 0
    lookup = get_semantic_lookup(profile)
    columns = profile.get("columns") or []
    n_rows = int(profile.get("n_rows", 0))
    eligible: list[str] = []
    try:
        df = read_csv_robust(csv_path)
    except Exception:
        return [], 0
    nunique_by_col = {
        str(col.get("name")): int(df[col.get("name")].nunique(dropna=True))
        for col in columns
        if col.get("name") in df.columns
    }
    for col_info in columns:
        col = str(col_info.get("name") or "")
        if not col or col == target_column:
            continue
        sem = lookup.get(col) or {}
        col_type = sem.get("type", "unknown")
        if col_type in ("identifier", "constant", "text"):
            continue
        # High-uniqueness columns are skipped by leakage_check itself.
        nunique = nunique_by_col.get(col)
        if nunique is not None and n_rows >= 50:
            if nunique / n_rows >= check_config.leakage.high_cardinality_ratio:
                continue
        eligible.append(col)
    total_eligible = len(eligible)
    capped = eligible[:max_count]
    return capped, total_eligible


def router_node(state: GraphState) -> GraphState:
    """Decide whether to run deep leakage checks.

    When a target column exists, every eligible feature column is treated
    as a leakage suspect (excluded: target, identifier, constant, text,
    high-uniqueness). ``leakage_suspected`` is True when at least one
    eligible column exists. ``id_like`` and ``schema`` findings are no
    longer triggers -- an identifier is not leakage.
    """
    csv_path = state.get("csv_path")
    target_column = state.get("target_column")
    profile = state.get("profile") or {}
    max_count = int(check_config.leakage.max_leakage_suspects)

    suspect_columns, total_eligible = _eligible_suspect_columns(
        csv_path, profile, target_column, max_count
    )
    findings: list[dict[str, Any]] = []
    if total_eligible > len(suspect_columns):
        findings.append(
            info_note(
                "target_leakage",
                (
                    f"Capped deep-leakage check at {max_count} suspect "
                    f"column(s); {total_eligible - len(suspect_columns)} "
                    f"further eligible column(s) were not checked."
                ),
            )
        )

    # State field names kept for API compatibility with the agent path and
    # the checkpointer.
    return {
        "leakage_suspected": len(suspect_columns) > 0,
        "suspect_columns": suspect_columns,
        "findings": findings,
    }


def deep_checks_node(state: GraphState) -> GraphState:
    """Run the deep leakage check on non-target columns (leakage suspected
    only); results go to ``deep_check_results`` and their findings are merged.
    """
    csv_path = state.get("csv_path")
    target_column = state.get("target_column")

    if not csv_path or not target_column:
        return {"deep_check_results": []}

    # Deterministic path: router_node already set suspect_columns. Agent path:
    # compute them here from the same rule so both paths agree.
    suspect_columns = state.get("suspect_columns")
    if suspect_columns is None:
        csv_path = state.get("csv_path")
        profile = state.get("profile") or {}
        max_count = int(check_config.leakage.max_leakage_suspects)
        suspect_columns, _ = _eligible_suspect_columns(
            csv_path, profile, target_column, max_count
        )

    spec = get_check("target_leakage")
    deep_state = dict(state)
    deep_state["check_params"] = {
        **(state.get("check_params") or {}),
        "target_leakage": {"suspicious_columns": suspect_columns},
    }
    findings = run_check(spec, deep_state)

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


def _highest_severity(findings: list[dict]) -> Severity:
    """Return the highest severity among findings (INFO when empty)."""
    if not findings:
        return Severity.INFO
    finding_objs = [Finding.model_validate(f) for f in findings]
    return min(finding_objs).severity


def _run_all_checks_deterministically(state: GraphState) -> list[dict]:
    """Fallback: run every check without the LLM (used if the agent fails).

    Checks come from the registry's ``fallback`` group, so a new check joins
    the fallback automatically unless it opts out (for example, fairness,
    which must only run with user-specified sensitive columns).

    Returns findings as plain dicts for checkpointer serialization.
    """
    if not state.get("csv_path"):
        return []

    findings: list[Finding] = []
    for spec in get_checks(fallback=True):
        try:
            findings += run_check(spec, state)
        except Exception as exc:  # noqa: BLE001 — one check must not stop the rest
            findings.append(error_finding(spec.name, exc))
    return [_finding_to_dict(f) for f in findings]


def make_agent_checks_node(llm: Any) -> Callable[[GraphState], GraphState]:
    """Factory for the M4 agent node that chooses and runs checks via the LLM."""

    def agent_checks_node(state: GraphState) -> GraphState:
        csv_path = state.get("csv_path")
        if not csv_path:
            return {"error": "No csv_path in state; cannot run checks."}

        profile = state.get("profile") or {}
        # Honor user-supplied per-check params (for example --sensitive /
        # --favorable) in both the agent's tool calls and the completeness
        # guard, so CLI values are not silently dropped in agent mode.
        check_params = state.get("check_params") or {}
        try:
            from src.graph.agent import (
                run_checks_with_agent,
                run_uncalled_agent_checks,
            )
            findings, messages = run_checks_with_agent(
                csv_path=csv_path,
                target_column=state.get("target_column"),
                profile=profile,
                llm=llm,
                check_params=check_params,
            )
            # Guard: guarantee every registered check is represented, even
            # when the model skips one (for example bias). Only checks the
            # agent never called are run; the agent still chooses the tools.
            # An agent exception propagates to the except block below, so the
            # guard never intercepts it.
            findings += run_uncalled_agent_checks(
                messages,
                csv_path=csv_path,
                target_column=state.get("target_column"),
                profile=profile,
                check_params=check_params,
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

    # LLM plain-language summary (M4); indented so multi-line summaries
    # stay visually nested under the heading.
    if report.summary:
        lines.append("Summary:")
        lines.append("-" * 32)
        for line in report.summary.splitlines():
            lines.append(f"  {line}" if line.strip() else "")
        lines.append("")

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
        lines.append("")

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

    overrides = state.get("semantic_overrides") or {}
    try:
        profile = profile_dataset(
            csv_path,
            target_column=state.get("target_column"),
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 — surface any load error in state
        return {"error": f"Failed to profile {csv_path}: {exc}"}

    return {"profile": profile, "error": None}


def report_node(state: GraphState) -> GraphState:
    """Render the profile + merged findings + deep check results as a report.

    Builds one :class:`AuditReport` — the source of truth — storing its JSON
    (``state["report"]``) and plain-text rendering (``state["report_text"]``).
    Even with no profile, renders a minimal report so the error surfaces.
    """
    profile = state.get("profile")
    dataset_id = state.get("dataset_id", "unknown")
    error = state.get("error")
    findings_raw = state.get("findings") or []
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


__all__ = [
    "load_csv_node",
    "profile_node",
    "merge_findings_node",
    "router_node",
    "deep_checks_node",
    "make_agent_checks_node",
    "make_check_node",
    "render_text_report",
    "report_node",
    "_highest_severity",
]
