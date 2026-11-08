"""SQLite backend tests (Milestone 6).

Every test uses a ``tmp_path`` SQLite database; PostgreSQL is never involved.
Covers backend selection, the save/list/get round trip (including JSON report
content), and the main.py audit path end to end on SQLite.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
import src.db.connection as conn_mod
import src.db.repository as repo_mod
from src.db.repository import ensure_schema, get_run, list_runs, save_run


def _make_csv(tmp_path) -> str:
    df = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "num": list(range(100)),
            "cat": ["a", "b"] * 50,
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def _use_sqlite_at(tmp_path, monkeypatch) -> None:
    """Point cached settings + the module engine at a tmp_path SQLite DB."""
    # DATABASE_URL in .env points to postgres; override it via env var
    # (env vars take priority over env_file in pydantic-settings).
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


# ── backend selection ──────────────────────────────────────────────────


def test_backend_sqlite_when_database_url_unset(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")  # overrides the .env value
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.backend_name == "sqlite"
    assert settings.database_url.startswith("sqlite:///")
    assert settings.is_sqlite and not settings.is_postgres


def test_backend_postgres_when_database_url_set(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/data_detective"
    )
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.backend_name == "postgres"
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.is_postgres and not settings.is_sqlite


def test_database_path_overrides_default(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("DATABASE_PATH", "/tmp/custom_detective.db")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.database_url == "sqlite:////tmp/custom_detective.db"


# ── repository round trip ──────────────────────────────────────────────


def test_save_run_list_runs_get_run_round_trip(tmp_path, monkeypatch):
    _use_sqlite_at(tmp_path, monkeypatch)

    findings = [
        {
            "check_name": "missing_values",
            "severity": "high",
            "evidence": "12 missing cells",
            "affected_columns": ["age", "salary"],
            "confidence": 0.9,
        }
    ]
    report_json = {
        "dataset_id": "sample",
        "rows": 100,
        "cols": 3,
        "overall_severity": "high",
        "findings": findings,
    }

    run_id = save_run(
        run_id="run-abc-123",
        csv_path="data/sample.csv",
        target_column="target",
        status="completed",
        overall_severity="high",
        findings=findings,
        report_json=report_json,
        report_text="Audit report text.",
    )
    assert run_id

    # list_runs returns the run, newest first.
    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-abc-123"
    assert runs[0]["csv_path"] == "data/sample.csv"
    assert runs[0]["target_column"] == "target"
    assert runs[0]["status"] == "completed"
    assert runs[0]["overall_severity"] == "high"
    assert runs[0]["findings_count"] == 1

    # get_run loads the full run including the JSON report.
    run = get_run("run-abc-123")
    assert run is not None
    assert run["run_id"] == "run-abc-123"
    assert run["report_json"] == report_json
    assert run["report_json"]["findings"][0]["check_name"] == "missing_values"
    assert run["report_text"] == "Audit report text."

    # Unknown run_id returns None.
    assert get_run("does-not-exist") is None


# ── main.py audit path end to end on SQLite ────────────────────────────


def test_main_audit_path_runs_on_sqlite(tmp_path, monkeypatch, capsys):
    _use_sqlite_at(tmp_path, monkeypatch)

    from main import main as cli_main

    csv_path = _make_csv(tmp_path)

    old_argv = sys.argv
    sys.argv = ["main.py", csv_path, "--target", "target"]
    try:
        cli_main()
    finally:
        sys.argv = old_argv

    output = capsys.readouterr().out

    # One startup line names the backend.
    assert "Using sqlite backend." in output, output

    # The audit completed and produced a report (no crash).
    assert "Error:" not in output, output
    assert "Dataset: sample" in output or "(no report produced)" not in output

    # The run was persisted to the SQLite DB.
    runs = list_runs()
    assert len(runs) >= 1, runs
    run = get_run(runs[0]["run_id"])
    assert run is not None
    assert run["report_json"] is not None
    assert run["report_json"]["rows"] == 100
    assert isinstance(run["report_json"]["findings"], list)
