"""Graph-level tests for M2: parallel checks, fan-out/fan-in, reducers."""

import json
import os
import tempfile
import time

import pandas as pd

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph, CHECKS
from src.graph.nodes import render_text_report
from src.schemas.finding import Finding, Severity
from src.schemas.report import AuditReport


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


# ── graph structure ───────────────────────────────────────────────────

def test_checks_registry_has_six_checks():
    """The CHECKS registry registers exactly 6 checks (5 M2 + bias)."""
    assert len(CHECKS) == 6
    names = {name for name, _ in CHECKS}
    assert names == {"bias", "duplicates", "missing_values", "schema", "class_imbalance", "id_like"}


def test_graph_run_clean_csv_produces_report():
    """A clean CSV runs through all checks and produces a report."""
    df = pd.DataFrame(
        {"age": [25, 30, 35, 40], "city": ["NY", "LA", "NY", "LA"]}
    )
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m2"}}
    final_state = graph.invoke({"csv_path": csv}, config=config)


    assert final_state.get("error") is None, final_state.get("error")
    report = final_state.get("report")
    report_text = final_state.get("report_text")
    assert report is not None and len(report) > 0
    assert report_text is not None and len(report_text) > 0
    assert "Dataset: sample" in report_text
    # JSON is valid.
    json.loads(report)


def test_graph_run_dirty_csv_collects_findings():
    """A dirty CSV produces merged findings from all checks."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "target": [0, 0, 0, 1],
            "dup": ["x", "x", "y", "z"],
        }
    )
    # Add a duplicate row
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m2-dirty"}}
    final_state = graph.invoke({"csv_path": csv, "target_column": "target"}, config=config)


    assert final_state.get("error") is None, final_state.get("error")
    findings = final_state.get("findings", [])
    assert len(findings) > 0, "Expected at least one finding from dirty CSV"
    # Check that findings came from multiple check types
    check_names = {f["check_name"] for f in findings}
    assert len(check_names) >= 2, f"Expected multiple check types, got {check_names}"


def test_findings_reducer_merges_parallel_results():
    """The merge_findings reducer merges findings from parallel branches."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "target": [0, 0, 1],
        }
    )
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m2-reducer"}}
    final_state = graph.invoke({"csv_path": csv, "target_column": "target"}, config=config)


    findings = final_state.get("findings", [])
    assert isinstance(findings, list)
    # Should have at least id_like (id column unique) and possibly imbalance
    assert len(findings) >= 1


# ── conditional error edge ────────────────────────────────────────────

def test_missing_csv_skips_checks_goes_to_report():
    """If load_csv sets an error, checks are skipped and report still renders."""
    graph = build_graph()
    final_state = graph.invoke({"csv_path": "does_not_exist.csv"})

    assert final_state.get("error") is not None
    assert final_state.get("findings") == [] or final_state.get("findings") is None
    report = final_state.get("report")
    assert report is not None and len(report) > 0


# ── report includes findings sorted by severity ───────────────────────

def test_render_text_report_sorts_findings_by_severity():
    """Findings in the report are sorted highest severity first."""
    report = render_text_report(
        AuditReport(
            dataset_id="test_ds",
            rows=4,
            cols=2,
            findings=[
                Finding(check_name="low_check", severity=Severity.LOW, evidence="low"),
                Finding(check_name="crit_check", severity=Severity.CRITICAL, evidence="crit"),
                Finding(check_name="med_check", severity=Severity.MEDIUM, evidence="med"),
            ],
        )
    )

    crit_pos = report.index("crit_check")
    med_pos = report.index("med_check")
    low_pos = report.index("low_check")
    assert crit_pos < med_pos < low_pos, "Findings should be sorted highest first"


def test_report_contains_finding_fields():
    """Report shows check name, severity, evidence, columns, confidence."""
    report = render_text_report(
        AuditReport(
            dataset_id="test_ds",
            rows=4,
            cols=2,
            findings=[
                Finding(
                    check_name="duplicates",
                    severity=Severity.HIGH,
                    evidence="Found 2 duplicate rows",
                    affected_columns=["col_a"],
                    confidence=0.95,
                )
            ],
        )
    )

    assert "[HIGH]" in report
    assert "duplicates" in report
    assert "Found 2 duplicate rows" in report
    assert "col_a" in report
    assert "Remove duplicate rows" in report
    assert "0.95" in report


# ── sleep-based parallel proof ────────────────────────────────────────

def test_checks_run_in_parallel_sleep_proof():
    """
    Prove the check branches run in parallel, not serially.

    Every entry in the builder's CHECKS registry is patched to sleep 0.4s.
    If serial, total wall time would be ~2.4s (6 checks). If parallel, it
    should be ~0.4-0.5s.
    """
    import time
    from unittest import mock

    import src.graph.builder as builder_mod

    def _make_sleeping_check(orig):
        def _sleeping(*args, **kwargs):
            time.sleep(0.4)
            return orig(*args, **kwargs)
        return _sleeping

    sleeping = [
        (name, _make_sleeping_check(fn)) for name, fn in builder_mod.CHECKS
    ]

    with mock.patch.object(builder_mod, "CHECKS", sleeping):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        csv = _write_csv(df)

        graph = build_graph()
        start = time.time()
        final_state = graph.invoke({"csv_path": csv})
        elapsed = time.time() - start

    assert final_state.get("error") is None, final_state.get("error")
    # Parallel: ~0.4s. Serial: ~2.4s. Use 1.5s as the threshold.
    assert elapsed < 1.5, (
        f"Checks appear to run serially: {elapsed:.2f}s elapsed "
        f"(expected < 1.5s for parallel)"
    )


__all__ = [
    "test_checks_registry_has_six_checks",
    "test_graph_run_clean_csv_produces_report",
    "test_graph_run_dirty_csv_collects_findings",
    "test_findings_reducer_merges_parallel_results",
    "test_missing_csv_skips_checks_goes_to_report",
    "test_render_text_report_sorts_findings_by_severity",
    "test_report_contains_finding_fields",
    "test_checks_run_in_parallel_sleep_proof",
]
