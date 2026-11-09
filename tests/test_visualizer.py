"""Unit tests for the deterministic plot selector.

Uses synthetic DataFrames, tmp_path CSVs, the matplotlib Agg backend, and
monkeypatches ``PLOTS_DIR`` so nothing is written into the real plots/ folder.
No LLM, no database, no network.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import json
import os
import sys
import tempfile
from io import StringIO

import numpy as np
import pandas as pd
import pytest

import src.workbench.visualizer.visualizer_tools as vt
from src.tools.profiling import profile_dataset
from src.workbench.visualizer.visualizer_tools import (
    generate_plots,
    PlotReport,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _write_csv(df: pd.DataFrame, tmp_path, name: str = "data.csv") -> str:
    path = os.path.join(str(tmp_path), name)
    df.to_csv(path, index=False)
    return path


def _profile(columns: list[dict], target_type: str = "binary") -> dict:
    return {"columns": columns, "target_type": target_type}


def _col(name: str, stype: str, ambiguous: bool = False) -> dict:
    return {"name": name, "semantic": {"type": stype, "ambiguous": ambiguous}}


def _png_ok(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == PNG_MAGIC
    except OSError:
        return False


def _plots(report: PlotReport) -> list[str]:
    return list(report.plots)


def _skipped_reasons(report: PlotReport) -> list[str]:
    return [s.get("reason", "") for s in report.skipped]


def _set_plots_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(vt, "PLOTS_DIR", tmp_path)


# ── outliers / skewness per-column plots ───────────────────────────────


def test_outliers_and_skewness_produce_plots(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
            "num2": [float(i) for i in range(100)],
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [_col("num1", "numeric_measurement"), _col("num2", "numeric_measurement")],
        target_type="binary",
    )
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": ["num1"], "evidence": "e1"},
        {"check_name": "skewness", "severity": "high", "affected_columns": ["num1"], "evidence": "e2"},
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    assert report.plots, report.skipped
    assert all(_png_ok(p) for p in report.plots)
    names = [os.path.basename(p) for p in report.plots]
    assert any("outliers" in n and "num1" in n for n in names), names
    assert any("skewness" in n and "num1" in n for n in names), names
    # classification target always gets a class-balance plot.
    assert any("imbalance" in n and "target" in n for n in names), names
    # target column excluded from outliers/skewness for classification.
    assert not any("outliers" in n and "target" in n for n in names), names
    assert not any("skewness" in n and "target" in n for n in names), names


# ── classification target ──────────────────────────────────────────────


def test_classification_target_excludes_target_and_makes_imbalance(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0] * 80 + [1] * 20,
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [_col("target", "binary"), _col("num1", "numeric_measurement")],
        target_type="binary",
    )
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": ["target", "num1"], "evidence": "e1"},
        {"check_name": "class_imbalance", "severity": "medium", "affected_columns": ["target"], "evidence": "e2"},
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    names = [os.path.basename(p) for p in report.plots]
    # imbalance plot produced for classification target.
    assert any("imbalance" in n and "target" in n for n in names), names
    # target column NOT plotted as outliers/skewness.
    assert not any("outliers" in n and "target" in n for n in names), names
    assert not any("skewness" in n and "target" in n for n in names), names
    # target distribution skipped: class balance covers the target.
    assert any("class balance plot covers the target" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


# ── continuous target ──────────────────────────────────────────────────


def test_continuous_target_skips_imbalance_but_makes_other_plots(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [float(i) for i in range(100)],
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [_col("target", "numeric_measurement"), _col("num1", "numeric_measurement")],
        target_type="continuous",
    )
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": ["target", "num1"], "evidence": "e1"},
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    names = [os.path.basename(p) for p in report.plots]
    # target column included in outliers for continuous target.
    assert any("outliers" in n and "target" in n for n in names), names
    assert any("outliers" in n and "num1" in n for n in names), names
    # imbalance skipped with reason for continuous target.
    assert any("continuous" in r for r in _skipped_reasons(report)), _skipped_reasons(report)
    assert not any("imbalance" in n for n in names), names


# ── classification target: class balance always plotted ────────────────


def test_balanced_binary_target_gets_class_balance_plot(tmp_path, monkeypatch):
    """A balanced binary target with no class_imbalance finding still gets a
    class-balance plot."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,  # balanced
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [_col("target", "binary"), _col("num1", "numeric_measurement")],
        target_type="binary",
    )
    # No class_imbalance finding.
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": ["num1"], "evidence": "e1"},
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    names = [os.path.basename(p) for p in report.plots]
    assert any("imbalance" in n and "target" in n for n in names), names


def test_classification_skip_reason_class_balance_covers_target(
    tmp_path, monkeypatch
):
    """The target-distribution skip reason for classification targets."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0] * 80 + [1] * 20,
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [_col("target", "binary"), _col("num1", "numeric_measurement")],
        target_type="binary",
    )
    findings = []
    report = generate_plots(findings, profile, csv_path, "target")
    assert any(
        "class balance plot covers the target" in r for r in _skipped_reasons(report)
    ), _skipped_reasons(report)


def test_imbalance_finding_does_not_create_second_plot(tmp_path, monkeypatch):
    """A class_imbalance finding produces exactly one imbalance plot."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0] * 80 + [1] * 20,
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [_col("target", "binary"), _col("num1", "numeric_measurement")],
        target_type="binary",
    )
    findings = [
        {
            "check_name": "class_imbalance",
            "severity": "medium",
            "affected_columns": ["target"],
            "evidence": "e2",
        },
        {
            "check_name": "class_imbalance",
            "severity": "medium",
            "affected_columns": ["target"],
            "evidence": "e3",
        },
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    imbalance_plots = [p for p in report.plots if "imbalance" in os.path.basename(p)]
    assert len(imbalance_plots) == 1, imbalance_plots


# ── per-column cap ─────────────────────────────────────────────────────


def test_per_column_cap_adds_skipped_entry(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    cols = [f"c{i}" for i in range(20)]
    df = pd.DataFrame({"target": [0, 1] * 50, **{c: [float(i) for i in range(100)] for c in cols}})
    csv_path = _write_csv(df, tmp_path)
    profile = _profile([_col(c, "numeric_measurement") for c in cols] + [_col("target", "binary")], target_type="binary")
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": cols, "evidence": "e"},
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    outlier_plots = [p for p in report.plots if "outliers" in os.path.basename(p)]
    assert len(outlier_plots) == vt.MAX_PLOTS_PER_TYPE, len(outlier_plots)
    assert any("more column(s) not plotted" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


# ── correlation ────────────────────────────────────────────────────────


def test_correlation_caps_and_notes(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    cols = [f"c{i}" for i in range(35)]
    df = pd.DataFrame({c: [float(i) for i in range(100)] for c in cols})
    csv_path = _write_csv(df, tmp_path)
    profile = _profile([_col(c, "numeric_measurement") for c in cols], target_type="continuous")
    report = generate_plots([], profile, csv_path, None)
    names = [os.path.basename(p) for p in report.plots]
    assert any("correlation" in n for n in names), names
    assert any("capped at" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


def test_correlation_skipped_when_fewer_than_two_numeric(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
    csv_path = _write_csv(df, tmp_path)
    profile = _profile([_col("only", "numeric_measurement")], target_type="continuous")
    report = generate_plots([], profile, csv_path, None)
    assert report.plots == [], report.plots
    assert any("two numeric columns" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


def test_correlation_cap_passes_exactly_cap_columns(tmp_path, monkeypatch):
    """More numeric columns than the cap → heatmap uses exactly the cap and
    notes how many were left out."""
    _set_plots_dir(monkeypatch, tmp_path)
    cols = [f"c{i}" for i in range(35)]
    df = pd.DataFrame({c: [float(i) for i in range(100)] for c in cols})
    csv_path = _write_csv(df, tmp_path)
    profile = _profile([_col(c, "numeric_measurement") for c in cols], target_type="continuous")
    captured: dict = {}

    def spy(csv_path, columns=None):
        captured["columns"] = list(columns) if columns is not None else None
        return vt._result(
            "run_plot_correlation",
            "fake.png",
            "ok",
            columns or [],
            "x",
        )

    monkeypatch.setattr(vt.run_plot_correlation, "func", spy)
    report = generate_plots([], profile, csv_path, None)
    assert captured["columns"] is not None, captured
    assert len(captured["columns"]) == vt.MAX_CORRELATION_COLUMNS, captured
    assert any("capped at" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


def test_correlation_default_behavior_unchanged(tmp_path, monkeypatch):
    """Calling the correlation tool without ``columns`` still plots all
    numeric columns (signature backward compatible)."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    csv_path = _write_csv(df, tmp_path)
    result = vt.run_plot_correlation.func(csv_path=csv_path)
    payload = vt._to_dict(result)
    assert payload.get("status") == "ok", payload
    assert len(payload.get("columns_affected", [])) == 2, payload


# ── missing values and categorical ─────────────────────────────────────


def test_missing_values_and_categorical_and_target_distribution(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
            "cat1": ["a", "b", "c"] * 33 + ["a"],
            "missing": [1, None, 3, None, 5] * 20,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile(
        [
            _col("num1", "numeric_measurement"),
            _col("cat1", "categorical"),
            _col("missing", "numeric_measurement"),
            _col("target", "binary"),
        ],
        target_type="binary",
    )
    findings = [
        {"check_name": "missing_values", "severity": "low", "affected_columns": ["missing"], "evidence": "e"},
        {"check_name": "class_imbalance", "severity": "medium", "affected_columns": ["target"], "evidence": "e"},
    ]
    report = generate_plots(findings, profile, csv_path, "target")
    names = [os.path.basename(p) for p in report.plots]
    assert any("missing_values" in n for n in names), names
    assert any("categorical_counts" in n for n in names), names
    # classification target → target distribution skipped: class balance covers it.
    assert any("class balance plot covers the target" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


# ── empty findings ─────────────────────────────────────────────────────


def test_empty_findings_nothing_to_plot(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame({"target": [0, 1] * 50, "num1": [float(i) for i in range(100)]})
    csv_path = _write_csv(df, tmp_path)
    # Empty profile → no usable profile facts.
    report = generate_plots([], {}, csv_path, "target")
    assert report.plots == [], report.plots
    assert report.skipped == [{"plot": None, "reason": "nothing to plot"}], report.skipped


# ── all-text DataFrame ─────────────────────────────────────────────────


def test_all_text_dataframe_records_reasons(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame({"cat1": ["a", "b", "c"] * 34})
    csv_path = _write_csv(df, tmp_path)
    profile = _profile([_col("cat1", "text")], target_type="continuous")
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": ["cat1"], "evidence": "e"},
    ]
    report = generate_plots(findings, profile, csv_path, None)
    assert report.plots == [], report.plots
    assert any("text" in r or "only applies" in r for r in _skipped_reasons(report)), _skipped_reasons(report)


# ── nonexistent CSV ────────────────────────────────────────────────────


def test_nonexistent_csv_no_exception(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    report = generate_plots([], {}, os.path.join(str(tmp_path), "does_not_exist.csv"), None)
    assert report.plots == [], report.plots
    assert len(report.skipped) == 1
    assert "Failed to read CSV" in report.skipped[0]["reason"], report.skipped


# ── determinism ────────────────────────────────────────────────────────


def test_deterministic_plot_paths(tmp_path, monkeypatch):
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "num1": [1.0, 2.0, 3.0, 100.0] * 25,
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = _profile([_col("num1", "numeric_measurement"), _col("target", "binary")], target_type="binary")
    findings = [
        {"check_name": "outliers", "severity": "high", "affected_columns": ["num1"], "evidence": "e"},
    ]
    r1 = generate_plots(findings, profile, csv_path, "target")
    r2 = generate_plots(findings, profile, csv_path, "target")
    assert r1.plots == r2.plots
    assert r1.skipped == r2.skipped


# ── PlotReport shape ───────────────────────────────────────────────────


def test_plot_report_is_dataclass():
    report = PlotReport(plots=["a.png"], skipped=[{"plot": "b.png", "reason": "x"}])
    assert report.plots == ["a.png"]
    assert report.skipped == [{"plot": "b.png", "reason": "x"}]


# ── main.py handler (run-id path) ──────────────────────────────────────


import importlib

import main as main_module  # noqa: E402  (imported at end to reuse fixtures)


def _make_run(findings, csv_path="data/raw/sample.csv", target_column="target"):
    return {
        "run_id": "run-123",
        "csv_path": csv_path,
        "target_column": target_column,
        "report_json": {"findings": findings},
    }


def _write_real_csv(tmp_path, name="sample.csv"):
    df = pd.DataFrame({"target": [0, 1] * 50, "num1": [float(i) for i in range(100)]})
    path = os.path.join(str(tmp_path), name)
    df.to_csv(path, index=False)
    return path


def _call_handler(run, monkeypatch, tmp_path):
    """Invoke _run_visualizer_workbench with generate_plots and profile
    monkeypatched, capturing stdout. Returns (captured_output, generate_plots_call_args)."""
    _set_plots_dir(monkeypatch, tmp_path)
    call_args = {}

    def spy_generate_plots(*, findings, profile, csv_path, target_column):
        call_args["findings"] = findings
        call_args["profile"] = profile
        call_args["csv_path"] = csv_path
        call_args["target_column"] = target_column
        return PlotReport(
            plots=["/tmp/plots/sample/outliers_num.png"],
            skipped=[{"plot": None, "reason": "some reason"}],
        )

    monkeypatch.setattr(main_module, "generate_plots", spy_generate_plots)
    monkeypatch.setattr(main_module, "profile_dataset", lambda *a, **k: {"columns": []})

    out = StringIO()
    old_stdout = sys.stdout
    sys.stdout = out
    try:
        main_module._run_visualizer_workbench(run)
    finally:
        sys.stdout = old_stdout
    return out.getvalue(), call_args


def test_handler_calls_generate_plots_once(tmp_path, monkeypatch):
    import sys
    from io import StringIO

    run = _make_run(
        findings=[{"check_name": "outliers", "severity": "high", "affected_columns": ["num1"], "evidence": "e"}],
        csv_path=_write_real_csv(tmp_path),
    )
    output, args = _call_handler(run, monkeypatch, tmp_path)
    assert args["findings"] == run["report_json"]["findings"]
    assert args["csv_path"] == run["csv_path"]
    assert args["target_column"] == run["target_column"]
    assert args["profile"] == {"columns": []}
    assert "some reason" in output


def test_handler_prints_paths_and_skip_reasons(tmp_path, monkeypatch):
    import sys
    from io import StringIO

    run = _make_run(findings=[], csv_path=_write_real_csv(tmp_path))
    output, _ = _call_handler(run, monkeypatch, tmp_path)
    assert "/tmp/plots/sample/outliers_num.png" in output
    assert "some reason" in output
    assert "plot(s) generated" in output


def test_handler_missing_csv_prints_message_no_crash(tmp_path, monkeypatch):
    import sys
    from io import StringIO

    run = _make_run(findings=[], csv_path="data/raw/does_not_exist.csv")
    called = {"n": 0}

    def spy_generate_plots(**kwargs):
        called["n"] += 1
        return PlotReport(plots=[], skipped=[])

    monkeypatch.setattr(main_module, "generate_plots", spy_generate_plots)
    monkeypatch.setattr(main_module, "profile_dataset", lambda *a, **k: {"columns": []})

    out = StringIO()
    old_stdout = sys.stdout
    sys.stdout = out
    try:
        main_module._run_visualizer_workbench(run)
    finally:
        sys.stdout = old_stdout
    output = out.getvalue()
    assert called["n"] == 0
    assert "CSV file not found" in output
    assert run["csv_path"] in output


def test_handler_unknown_run_keeps_error_message(monkeypatch):
    called = {"n": 0}

    def spy_get_run(run_id):
        called["n"] += 1
        return None

    monkeypatch.setattr(main_module, "get_run", spy_get_run)
    out = StringIO()
    old_stdout = sys.stdout
    sys.stdout = out
    try:
        run = main_module.get_run("no-such-run")
        if run is None:
            print(f"Error: no audit run found for run-id no-such-run.")
    finally:
        sys.stdout = old_stdout
    output = out.getvalue()
    assert called["n"] == 1
    assert "Error: no audit run found for run-id no-such-run." in output


def test_handler_no_plots_prints_skip_reasons_not_bare_message(tmp_path, monkeypatch):
    import sys
    from io import StringIO

    run = _make_run(findings=[], csv_path=_write_real_csv(tmp_path))
    called = {"n": 0}

    def spy_generate_plots(**kwargs):
        called["n"] += 1
        return PlotReport(
            plots=[],
            skipped=[{"plot": None, "reason": "nothing to plot"}],
        )

    monkeypatch.setattr(main_module, "generate_plots", spy_generate_plots)
    monkeypatch.setattr(main_module, "profile_dataset", lambda *a, **k: {"columns": []})

    out = StringIO()
    old_stdout = sys.stdout
    sys.stdout = out
    try:
        main_module._run_visualizer_workbench(run)
    finally:
        sys.stdout = old_stdout
    output = out.getvalue()
    assert called["n"] == 1
    assert "nothing to plot" in output
    assert "No plots were generated" not in output


def test_print_plot_report_lists_paths_count_reasons(tmp_path, monkeypatch):
    import sys
    from io import StringIO

    report = PlotReport(
        plots=["/tmp/plots/sample/outliers_num.png", "/tmp/plots/sample/skewness_num.png"],
        skipped=[
            {"plot": None, "reason": "continuous target"},
            {"plot": None, "reason": "nothing to plot"},
        ],
    )
    out = StringIO()
    old_stdout = sys.stdout
    sys.stdout = out
    try:
        main_module._print_plot_report(report)
    finally:
        sys.stdout = old_stdout
    output = out.getvalue()
    assert "/tmp/plots/sample/outliers_num.png" in output
    assert "/tmp/plots/sample/skewness_num.png" in output
    assert "2 plot(s) generated" in output
    assert "continuous target" in output
    assert "nothing to plot" in output


# ── main.py audit path: --plots / --no-plots / prompt ─────────────────


def _make_argv(
    tmp_path,
    plots: bool = False,
    no_plots: bool = False,
    csv_path="data/raw/sample.csv",
    target="target",
    agent: bool = False,
    workbench=None,
    run_id=None,
    list_flag: bool = False,
) -> list[str]:
    argv = ["main.py"]
    if list_flag:
        argv.append("--list")
    elif workbench:
        argv.extend(["--workbench", workbench])
        if run_id:
            argv.extend(["--run-id", run_id])
    else:
        argv.append(csv_path)
        argv.extend(["--target", target])
        if agent:
            argv.append("--agent")
    if plots:
        argv.append("--plots")
    if no_plots:
        argv.append("--no-plots")
    return argv


def _run_audit_with_prompt(
    monkeypatch,
    tmp_path,
    argv: list[str],
    prompt_input: str | None = None,
    is_tty: bool = True,
):
    """Run main() with argv, generate_plots, and stdin monkeypatched. Returns
    (captured_stdout, generate_plots_call_args_or_None, prompted_or_None)."""
    _set_plots_dir(monkeypatch, tmp_path)
    call_args: dict | None = None
    prompted: list[str] = []

    def spy_generate_plots(*, findings, profile, csv_path, target_column):
        nonlocal call_args
        call_args = {
            "findings": findings,
            "profile": profile,
            "csv_path": csv_path,
            "target_column": target_column,
        }
        return PlotReport(
            plots=["/tmp/plots/sample/outliers_num.png"],
            skipped=[{"plot": None, "reason": "some reason"}],
        )

    def fake_input(prompt: str = "") -> str:
        prompted.append(prompt)
        if prompt_input is None:
            raise EOFError
        return prompt_input

    monkeypatch.setattr(main_module, "generate_plots", spy_generate_plots)
    monkeypatch.setattr(main_module, "profile_dataset", lambda *a, **k: {"columns": []})
    monkeypatch.setattr(main_module, "ensure_schema", lambda: None)
    monkeypatch.setattr(main_module, "save_run", lambda **kwargs: None)
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: is_tty)
    monkeypatch.setattr(sys, "argv", argv)

    out = StringIO()
    old_stdout = sys.stdout
    sys.stdout = out
    try:
        main_module.main()
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout
    return out.getvalue(), call_args, prompted


def test_plots_flag_calls_generate_plots_without_prompt(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, plots=True, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv
    )
    assert prompted == [], prompted
    assert call_args is not None, call_args
    assert call_args["csv_path"] == real_csv
    assert call_args["target_column"] == "target"
    assert "some reason" in output


def test_no_plots_flag_never_calls_or_asks(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, no_plots=True, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv
    )
    assert prompted == [], prompted
    assert call_args is None
    assert "Generate plots?" not in output


def test_not_tty_without_flags_no_question_no_plots(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, is_tty=False
    )
    assert prompted == [], prompted
    assert call_args is None
    assert "Generate plots?" not in output


def test_tty_yes_plots(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input="y"
    )
    assert call_args is not None
    assert "Generate plots? [y/N] " in prompted[0]


def test_tty_yes_word_plots(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input="yes"
    )
    assert call_args is not None


def test_tty_empty_no_no_plots(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input=""
    )
    assert call_args is None


def test_tty_n_no_plots(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input="n"
    )
    assert call_args is None


def test_tty_no_word_no_plots(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input="no"
    )
    assert call_args is None


def test_tty_eof_counts_as_no(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input=None
    )
    assert call_args is None


def test_both_flags_argparse_error(monkeypatch, tmp_path):
    import argparse as _argparse

    p = _argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plots", action="store_true")
    g.add_argument("--no-plots", action="store_true")
    with pytest.raises(SystemExit):
        p.parse_args(["--plots", "--no-plots"])


def test_list_and_workbench_never_prompt(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(
        tmp_path, workbench="visualize", run_id="run-1", csv_path=real_csv
    )
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv
    )
    assert prompted == [], prompted
    assert call_args is None
    assert "Generate plots?" not in output


def test_generate_plots_receives_in_memory_inputs(tmp_path, monkeypatch):
    real_csv = _write_real_csv(tmp_path)
    argv = _make_argv(tmp_path, plots=True, csv_path=real_csv)
    output, call_args, prompted = _run_audit_with_prompt(
        monkeypatch, tmp_path, argv, prompt_input="y"
    )
    assert call_args is not None
    assert isinstance(call_args["findings"], list)
    assert isinstance(call_args["profile"], dict)
    assert call_args["csv_path"] == real_csv
    assert call_args["target_column"] == "target"


# ── correlation heatmap column selection ────────────────────────────


def _titanic_like_frame() -> pd.DataFrame:
    """Titanic-like frame: numeric features, numeric 0/1 target, identifier,
    text and datetime columns (>= MIN_ROWS so the classifier is confident)."""
    n = 100
    base = {
        "Pclass": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3] * 10,
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, 27.0, 14.0, 4.0, 58.0, 20.0] * 10,
        "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 15.5, 5.0, 31.0, 16.7] * 10,
        "SibSp": [1, 1, 0, 0, 0, 0, 0, 1, 0, 0] * 10,
        "Parch": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1] * 10,
        "target": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 10,
        "Name": ["n%s" % i for i in range(n)],
    }
    return pd.DataFrame(
        {"PassengerId": list(range(1, n + 1)), **base, "date": pd.date_range("2020-01-01", periods=n)}
    )


def _capture_correlation_columns(monkeypatch):
    """Spy on run_plot_correlation.func and return the columns it receives."""
    captured: dict = {}

    def spy(csv_path, columns=None):
        captured["columns"] = list(columns) if columns is not None else None
        return vt._result("run_plot_correlation", "fake.png", "ok", columns or [], "x")

    monkeypatch.setattr(vt.run_plot_correlation, "func", spy)
    return captured


def test_correlation_heatmap_includes_numeric_target_first(
    tmp_path, monkeypatch
):
    """Heatmap contains Age, Fare, Pclass, SibSp, Parch and the target
    (first), but not the identifier; text/datetime columns are excluded."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = _titanic_like_frame()
    csv_path = _write_csv(df, tmp_path)
    profile = profile_dataset(csv_path, target_column="target")
    captured = _capture_correlation_columns(monkeypatch)
    report = generate_plots([], profile, csv_path, "target")
    cols = captured["columns"]
    assert cols is not None, cols
    assert cols[0] == "target", cols
    for col in ("Age", "Fare", "Pclass", "SibSp", "Parch"):
        assert col in cols, cols
    assert "PassengerId" not in cols, cols
    assert "Name" not in cols, cols
    assert "date" not in cols, cols
    assert report.plots, report.skipped


def test_correlation_heatmap_text_or_datetime_excluded(tmp_path, monkeypatch):
    """A text column and a datetime column are never in the heatmap."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "num1": [float(i) for i in range(100)],
            "text": ["t%s" % (i % 5) for i in range(100)],
            "stamp": pd.date_range("2020-01-01", periods=100),
        }
    )
    csv_path = _write_csv(df, tmp_path)
    profile = profile_dataset(csv_path, target_column="target")
    captured = _capture_correlation_columns(monkeypatch)
    generate_plots([], profile, csv_path, "target")
    cols = captured["columns"]
    assert cols is not None, cols
    assert cols[0] == "target", cols
    assert "num1" in cols, cols
    assert "text" not in cols, cols
    assert "stamp" not in cols, cols


def test_correlation_heatmap_skip_when_fewer_than_two(tmp_path, monkeypatch):
    """Fewer than two usable numeric columns gives a skip reason."""
    _set_plots_dir(monkeypatch, tmp_path)
    df = pd.DataFrame({"target": [0, 1] * 50, "text": ["x"] * 100})
    csv_path = _write_csv(df, tmp_path)
    profile = profile_dataset(csv_path, target_column="target")
    captured = _capture_correlation_columns(monkeypatch)
    report = generate_plots([], profile, csv_path, "target")
    # The correlation heatmap is not produced (tool not called due to skip);
    # only the imbalance plot may be present for a classification target.
    assert captured.get("columns") is None, captured
    assert any(
        "two numeric columns" in r for r in _skipped_reasons(report)
    ), _skipped_reasons(report)


def test_correlation_cap_ranks_by_correlation_not_variance(
    tmp_path, monkeypatch
):
    """More columns than the cap keeps the target plus exactly the cap,
    ranked by correlation with the target, not by raw variance. A
    large-scale noise column (highest variance, low correlation) is dropped."""
    _set_plots_dir(monkeypatch, tmp_path)
    n = 200
    rng = np.random.RandomState(42)
    target = rng.randint(0, 2, size=n)
    # Features positively correlated with the target.
    features = {
        "feat%s" % i: (target + 0.1 * rng.standard_normal(n)).tolist()
        for i in range(33)
    }
    # Noise column with the highest variance but essentially no correlation.
    noise = (1e6 * rng.standard_normal(n)).tolist()
    df = pd.DataFrame(
        {"target": target.tolist(), "noise": noise, **features}
    )
    csv_path = _write_csv(df, tmp_path)
    profile = profile_dataset(csv_path, target_column="target")
    captured = _capture_correlation_columns(monkeypatch)
    report = generate_plots([], profile, csv_path, "target")
    cols = captured["columns"]
    assert cols is not None, cols
    assert cols[0] == "target", cols
    # noise has the highest variance but is dropped (low correlation).
    assert "noise" not in cols, cols
    # target plus exactly MAX_CORRELATION_COLUMNS columns total.
    assert len(cols) == vt.MAX_CORRELATION_COLUMNS, cols
    assert any("capped at" in r for r in _skipped_reasons(report)), _skipped_reasons(report)
