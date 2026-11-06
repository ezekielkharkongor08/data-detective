"""Tests for M1 graph: load -> profile -> text report."""

import json
import os
import tempfile

import pandas as pd

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.nodes import render_text_report
from src.schemas.report import AuditReport


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def test_render_text_report_contains_key_fields():
    """The plain-text report includes dataset id, shape, and column info."""
    report = render_text_report(
        AuditReport(
            dataset_id="sample",
            rows=4,
            cols=3,
            columns=[
                {
                    "name": "age",
                    "dtype": "float64",
                    "missing": 1,
                    "missing_pct": 25.0,
                    "stats": {
                        "min": 25.0,
                        "max": 45.0,
                        "mean": 35.0,
                        "std": 10.0,
                    },
                },
                {
                    "name": "city",
                    "dtype": "object",
                    "missing": 1,
                    "missing_pct": 25.0,
                    "unique": 2,
                    "sample_values": ["NY", "LA"],
                },
            ],
            sample_rows=[{"age": 25.0, "city": "NY"}],
        )
    )

    assert "Dataset: sample" in report
    assert "Rows: 4, Cols: 3" in report
    assert "age" in report
    assert "city" in report
    assert "NY" in report


def test_render_text_report_shows_text_markers():
    """The column table names text markers instead of hiding them."""
    report = render_text_report(
        AuditReport(
            dataset_id="sample",
            rows=4,
            cols=1,
            columns=[
                {
                    "name": "workclass",
                    "dtype": "str",
                    "missing": 2,
                    "missing_pct": 50.0,
                    "text_markers": 2,
                    "unique": 2,
                    "sample_values": ["?", "Private"],
                }
            ],
            sample_rows=[],
        )
    )

    assert "missing=2" in report
    assert "[2 text marker(s)]" in report


def test_graph_run_produces_report():
    """Invoking the compiled graph on a CSV yields JSON + text reports."""
    df = pd.DataFrame({"age": [25, 30, None, 45], "city": ["NY", "LA", "NY", None]})
    csv = _write_csv(df)

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m1"}}
    final_state = graph.invoke({"csv_path": csv}, config=config)

    assert final_state.get("error") is None, final_state.get("error")
    assert final_state.get("dataset_id") == "sample"

    report = final_state.get("report")
    report_text = final_state.get("report_text")

    # JSON is valid and has the expected fields.
    assert report is not None and len(report) > 0
    data = json.loads(report)
    assert data["dataset_id"] == "sample"
    assert data["rows"] == 4
    assert data["cols"] == 2

    # Plain text is rendered from the same JSON object.
    assert report_text is not None and len(report_text) > 0
    assert "Dataset: sample" in report_text
    assert "Rows: 4, Cols: 2" in report_text


def test_graph_run_missing_file_sets_error():
    """A missing CSV path produces an error in state and an error report."""
    graph = build_graph()
    final_state = graph.invoke({"csv_path": "does_not_exist.csv"})

    assert final_state.get("error") is not None
    report_text = final_state.get("report_text")
    assert report_text is not None and len(report_text) > 0
    assert "Error:" in report_text
