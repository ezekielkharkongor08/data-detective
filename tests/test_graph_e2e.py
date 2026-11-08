"""End-to-end integration test for the full audit pipeline.

Data Loading -> Profiling -> parallel checks -> router -> (deep leakage) ->
text report.

This is the CI/CD gatekeeper: a single CSV is run through the compiled graph
and every structural invariant is asserted. Leakage is exercised separately
in ``tests/test_checks.py``; here we keep the dataset free of a perfect
leak column so the small-sample CV path stays stable.
"""

import json

import pandas as pd

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph


def _write_csv(df: pd.DataFrame) -> str:
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def _make_dirty_dataset() -> pd.DataFrame:
    """100-row CSV exercising duplicates, missing, imbalance, schema,
    skewness, outliers, and bias (info note). No leak column so the router
    does not trigger deep_checks."""
    n = 100
    return pd.DataFrame(
        {
            "target": [0] * 80 + [1] * 20,
            "dup": ["x", "x", "y", "z"] * 25,
            "missing": [1, None] * 50,
            "num": (["1", "2", "3", "4", "5", "x"] * 16) + ["1", "2", "3", "4"],
            "skewed": [1, 2, 3, 4, 1000] * 20,
            "outlier_col": list(range(1, 21)) * 5,
        }
    )


def test_e2e_pipeline_runs_checks_and_produces_report():
    """Run the full graph on a dirty CSV; multiple checks produce findings
    and a report (JSON + plain text) is produced."""
    df = _make_dirty_dataset()
    csv_path = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-e2e"}}
    final_state = graph.invoke(
        {"csv_path": csv_path, "target_column": "target"}, config=config
    )

    # 1. No crash.
    assert final_state.get("error") is None, final_state.get("error")

    # 2. Profile was built.
    profile = final_state.get("profile")
    assert profile is not None
    assert profile["n_rows"] == 100
    assert profile["n_cols"] == 6
    assert profile["target_type"] in ("binary", "boolean")

    # 3. Multiple checks produced well-formed findings.
    findings = final_state.get("findings") or []
    check_names = {f["check_name"] for f in findings}
    assert len(check_names) >= 3, f"Expected findings from multiple checks, got {check_names}"
    for f in findings:
        assert f["check_name"]
        assert f["severity"]
        assert f["evidence"]
        assert isinstance(f["affected_columns"], list)
        assert 0.0 <= f["confidence"] <= 1.0

    # 4. JSON report is valid and complete.
    report = final_state.get("report")
    assert report is not None and len(report) > 0
    data = json.loads(report)
    assert data["dataset_id"] == "sample"
    assert data["rows"] == 100
    assert data["cols"] == 6
    assert "overall_severity" in data
    assert isinstance(data["findings"], list)

    # 5. Plain-text report rendered from the same JSON object.
    report_text = final_state.get("report_text")
    assert report_text is not None and len(report_text) > 0
    assert "Dataset: sample" in report_text
    assert "Rows: 100, Cols: 6" in report_text


def test_e2e_missing_csv_still_renders_error_report():
    """A missing CSV path produces an error state and an error report
    instead of crashing."""
    graph = build_graph()
    final_state = graph.invoke({"csv_path": "does_not_exist.csv"})

    assert final_state.get("error") is not None
    report = final_state.get("report")
    assert report is not None and len(report) > 0
    report_text = final_state.get("report_text")
    assert report_text is not None and "Error:" in report_text


def test_e2e_clean_csv_skips_deep_leakage_check():
    """A clean CSV with no id_like/schema findings skips deep_checks."""
    df = pd.DataFrame(
        {"age": [25, 30, 25, 30], "city": ["NY", "LA", "NY", "LA"], "target": [0, 1, 0, 1]}
    )
    csv_path = _write_csv(df)
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-e2e-clean"}}
    final_state = graph.invoke(
        {"csv_path": csv_path, "target_column": "target"}, config=config
    )
    assert final_state.get("error") is None, final_state.get("error")
    assert final_state.get("leakage_suspected") is False
    deep = final_state.get("deep_check_results") or []
    assert deep == []
    report = final_state.get("report")
    assert report is not None
    json.loads(report)