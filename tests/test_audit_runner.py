"""Tests for the shared audit runner (M7 milestone 1).

Every test uses a ``tmp_path`` SQLite database; PostgreSQL is never involved.
No LLM and no network are required. Reuses the tmp-database pattern from
``tests/test_db_backend.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
import src.db.connection as conn_mod
import src.db.repository as repo_mod
import src.graph.audit_runner as runner_mod
from src.db.repository import ensure_schema, get_run
from src.graph.audit_runner import AuditResult, run_audit


def _make_csv(tmp_path) -> str:
    """Synthetic CSV with a binary target and a skewed sensitive column."""
    df = pd.DataFrame(
        {
            "target": [1] * 50 + [0] * 50,
            "sex": ["M"] * 50 + ["F"] * 50,
            "num": list(range(100)),
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def _use_sqlite_at(tmp_path, monkeypatch) -> None:
    """Point cached settings + the module engine at a tmp_path SQLite DB."""
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "detective.db"))
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.backend_name == "sqlite"
    engine = create_engine(
        settings.database_url, connect_args={"check_same_thread": False}
    )
    conn_mod.engine = engine
    conn_mod.SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    repo_mod.SessionLocal = conn_mod.SessionLocal
    ensure_schema()


class _FakeGraph:
    """Graph stand-in whose invoke returns a fixed state dict."""

    def __init__(self, state: dict):
        self.state = state

    def invoke(self, input_data: dict, config=None):  # noqa: ARG002
        return self.state


class _FakeGraphRaises:
    """Graph stand-in whose invoke raises."""

    def invoke(self, input_data: dict, config=None):  # noqa: ARG002
        raise RuntimeError("boom")


# ── deterministic run ──────────────────────────────────────────────────


def test_deterministic_run_persists_and_is_readable(tmp_path, monkeypatch):
    _use_sqlite_at(tmp_path, monkeypatch)

    csv_path = _make_csv(tmp_path)

    result = run_audit(
        run_id="run-det-001",
        csv_path=csv_path,
        target_column="target",
        check_params={},
        semantic_overrides={},
        mode="deterministic",
    )

    assert isinstance(result, AuditResult)
    assert result.status == "completed"
    assert result.mode == "deterministic"
    assert result.error is None
    assert isinstance(result.findings, list)
    assert result.report_json is not None
    assert result.report_text is not None
    assert result.report_json["rows"] == 100

    run = get_run("run-det-001")
    assert run is not None
    assert run["run_id"] == "run-det-001"
    assert run["status"] == "completed"
    assert run["report_json"] is not None
    assert run["report_text"] is not None


# ── nonexistent CSV ───────────────────────────────────────────────────


def test_nonexistent_csv_returns_error_without_raising(tmp_path, monkeypatch):
    _use_sqlite_at(tmp_path, monkeypatch)

    result = run_audit(
        run_id="run-missing-001",
        csv_path="does_not_exist.csv",
        target_column="target",
        check_params={},
        semantic_overrides={},
        mode="deterministic",
    )

    assert result.status == "completed"
    assert result.error is not None
    assert "does_not_exist.csv" in result.error
    assert result.findings == []
    assert result.report_json is None


# ── bias check via check_params ────────────────────────────────────────


def test_bias_check_params_produce_bias_finding(tmp_path, monkeypatch):
    _use_sqlite_at(tmp_path, monkeypatch)

    csv_path = _make_csv(tmp_path)

    result = run_audit(
        run_id="run-bias-001",
        csv_path=csv_path,
        target_column="target",
        check_params={"bias": {"sensitive_columns": ["sex"], "favorable_value": "1"}},
        semantic_overrides={},
        mode="deterministic",
    )

    assert result.error is None, result.error
    check_names = {f["check_name"] for f in result.findings}
    assert "bias" in check_names, f"Expected a bias finding, got {check_names}"


# ── mode selects the builder ───────────────────────────────────────────


def test_mode_selects_builder(monkeypatch, tmp_path):
    _use_sqlite_at(tmp_path, monkeypatch)
    csv_path = _make_csv(tmp_path)

    base_state = {
        "error": None,
        "report": '{"dataset_id":"sample","rows":100,"cols":3,"overall_severity":"low","findings":[]}',
        "report_text": "Dataset: sample\nRows: 100, Cols: 3",
        "findings": [],
        "overall_severity": "low",
        "profile": {},
    }
    calls: list[str] = []

    def fake_build_deterministic(checkpointer=None):  # noqa: ARG001
        calls.append("deterministic")
        return _FakeGraph(dict(base_state))

    def fake_build_agent(checkpointer=None):  # noqa: ARG001
        calls.append("agent")
        return _FakeGraph(dict(base_state))

    monkeypatch.setattr(runner_mod, "build_graph", fake_build_deterministic)
    monkeypatch.setattr(runner_mod, "build_agent_graph", fake_build_agent)

    run_audit(
        run_id="run-mode-det",
        csv_path=csv_path,
        target_column="target",
        check_params={},
        semantic_overrides={},
        mode="deterministic",
    )
    assert calls == ["deterministic"]

    calls.clear()
    run_audit(
        run_id="run-mode-agent",
        csv_path=csv_path,
        target_column="target",
        check_params={},
        semantic_overrides={},
        mode="agent",
    )
    assert calls == ["agent"]


# ── unknown mode ───────────────────────────────────────────────────────


def test_unknown_mode_raises_value_error(tmp_path, monkeypatch):
    _use_sqlite_at(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        run_audit(
            run_id="run-bad-mode",
            csv_path=_make_csv(tmp_path),
            target_column="target",
            check_params={},
            semantic_overrides={},
            mode="not_a_mode",
        )


# ── run_audit does not print ───────────────────────────────────────────


def test_run_audit_prints_only_backend_line(tmp_path, monkeypatch, capsys):
    _use_sqlite_at(tmp_path, monkeypatch)

    run_audit(
        run_id="run-noprint",
        csv_path=_make_csv(tmp_path),
        target_column="target",
        check_params={},
        semantic_overrides={},
        mode="deterministic",
    )

    captured = capsys.readouterr()
    # The only output is the backend startup line from the checkpointer setup.
    assert captured.out.strip() == "Using sqlite backend."
    assert captured.err == ""


# ── exception from the graph is returned as an error result ────────────


def test_graph_exception_returned_as_error_result(
    tmp_path, monkeypatch, capsys
):
    _use_sqlite_at(tmp_path, monkeypatch)
    csv_path = _make_csv(tmp_path)

    monkeypatch.setattr(runner_mod, "build_graph", lambda *a, **k: _FakeGraphRaises())
    monkeypatch.setattr(
        runner_mod, "build_agent_graph", lambda *a, **k: _FakeGraphRaises()
    )

    result = run_audit(
        run_id="run-exc-001",
        csv_path=csv_path,
        target_column="target",
        check_params={},
        semantic_overrides={},
        mode="deterministic",
    )

    assert result.status == "completed"
    assert result.mode == "deterministic"
    assert result.error == "boom"
    assert result.findings == []
    assert result.report_json is None
    assert result.report_text is None
