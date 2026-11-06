"""Graph-level tests for M3: conditional routing and deep leakage checks."""

import os
import tempfile

import pandas as pd

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.nodes import render_text_report, router_node
from src.schemas.finding import Finding, Severity
from src.schemas.report import AuditReport


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ── router node ───────────────────────────────────────────────────────

def test_router_flags_id_like_as_leakage_suspect():
    """An id_like finding makes the router suspect leakage."""
    findings = [
        {
            "check_name": "id_like",
            "severity": "medium",
            "evidence": "Column 'customerID' is unique per row",
            "affected_columns": ["customerID"],
            "confidence": 0.85,
        }
    ]
    state = {"findings": findings, "target_column": "Churn"}
    result = router_node(state)
    assert result["leakage_suspected"] is True
    assert "customerID" in result["suspect_columns"]
    assert "Churn" not in result["suspect_columns"]  # target is skipped


def test_router_flags_schema_as_leakage_suspect():
    """A schema finding makes the router suspect leakage."""
    findings = [
        {
            "check_name": "schema",
            "severity": "medium",
            "evidence": "Column 'target_text' is stored as text but looks numeric",
            "affected_columns": ["target_text"],
            "confidence": 0.9,
        }
    ]
    state = {"findings": findings, "target_column": "Churn"}
    result = router_node(state)
    assert result["leakage_suspected"] is True
    assert "target_text" in result["suspect_columns"]


def test_router_no_suspects_when_clean():
    """Clean findings mean no leakage suspected."""
    findings = [
        {
            "check_name": "missing_values",
            "severity": "low",
            "evidence": "Column 'age' has 1 missing value",
            "affected_columns": ["age"],
            "confidence": 1.0,
        }
    ]
    state = {"findings": findings, "target_column": "Churn"}
    result = router_node(state)
    assert result["leakage_suspected"] is False
    assert result["suspect_columns"] == []


def test_router_empty_findings():
    """No findings means no leakage suspected."""
    state = {"findings": [], "target_column": "Churn"}
    result = router_node(state)
    assert result["leakage_suspected"] is False
    assert result["suspect_columns"] == []


# ── graph structure ───────────────────────────────────────────────────

def test_graph_run_clean_csv_skips_deep_checks():
    """A clean CSV with no ID-like/schema findings skips deep_checks."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 25, 30],
            "city": ["NY", "LA", "NY", "LA"],
            "target": [0, 1, 0, 1],
        }
    )
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m3-clean"}}
    final_state = graph.invoke({"csv_path": csv, "target_column": "target"}, config=config)


    assert final_state.get("error") is None, final_state.get("error")
    assert final_state.get("leakage_suspected") is False
    assert final_state.get("deep_check_results") == [] or final_state.get("deep_check_results") is None
    report = final_state.get("report")
    assert report is not None and len(report) > 0


def test_graph_run_dirty_csv_runs_deep_checks():
    """A CSV with an ID-like column runs deep_checks and finds leakage."""
    # customerID is unique per row (triggers router via id_like).
    # 'leak' column equals the target (deep_checks finds the actual leakage).
    df = pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(20)],
            "leak": [0, 1] * 10,
            "target": [0, 1] * 10,
        }
    )
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m3-dirty"}}
    final_state = graph.invoke({"csv_path": csv, "target_column": "target"}, config=config)


    assert final_state.get("error") is None, final_state.get("error")
    assert final_state.get("leakage_suspected") is True
    deep_results = final_state.get("deep_check_results") or []
    assert len(deep_results) > 0, "Expected deep check results from leakage"
    leaked_columns = {r["column"] for r in deep_results}
    assert "leak" in leaked_columns, f"Expected 'leak' in {leaked_columns}"
    assert any(r["severity"] == "critical" for r in deep_results)


def test_graph_deep_check_findings_merged():
    """Leakage findings from deep_checks are merged into the findings list."""
    df = pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(20)],
            "leak": [0, 1] * 10,
            "target": [0, 1] * 10,
        }
    )
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m3-merged"}}
    final_state = graph.invoke({"csv_path": csv, "target_column": "target"}, config=config)


    findings = final_state.get("findings", [])
    leakage_findings = [f for f in findings if f["check_name"] == "target_leakage"]
    assert len(leakage_findings) > 0, "Expected target_leakage finding in merged results"
    assert Severity(leakage_findings[0]["severity"]) == Severity.CRITICAL


# ── report includes deep check results ────────────────────────────────

def test_render_text_report_includes_deep_checks():
    """The report includes a Deep Leakage Check Results section."""
    report = render_text_report(
        AuditReport(
            dataset_id="test_ds",
            rows=4,
            cols=3,
            findings=[],
            deep_check_results=[
                {
                    "column": "leak",
                    "severity": "critical",
                    "evidence": "Column 'leak' alone predicts target with 100.0% accuracy",
                }
            ],
        )
    )

    assert "Deep Leakage Check Results:" in report
    assert "leak" in report
    assert "CRITICAL" in report


def test_report_no_deep_checks_when_clean():
    """Report has no deep check section when no leakage suspected."""
    report = render_text_report(
        AuditReport(
            dataset_id="test_ds",
            rows=4,
            cols=3,
            findings=[],
        )
    )
    assert "Deep Leakage Check Results" not in report


__all__ = [
    "test_router_flags_id_like_as_leakage_suspect",
    "test_router_flags_schema_as_leakage_suspect",
    "test_router_no_suspects_when_clean",
    "test_router_empty_findings",
    "test_graph_run_clean_csv_skips_deep_checks",
    "test_graph_run_dirty_csv_runs_deep_checks",
    "test_graph_deep_check_findings_merged",
    "test_render_text_report_includes_deep_checks",
    "test_report_no_deep_checks_when_clean",
]
