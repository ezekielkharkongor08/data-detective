"""FastAPI API tests (M7).

Uses FastAPI TestClient with a ``tmp_path`` SQLite database; no LLM, no network,
no real Ollama. ``run_audit`` is monkeypatched with a fake that writes a stub
saved run, and ``BackgroundTasks.add_task`` is disabled so audits are driven by
calling the background function directly (reliable, no threading).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.api.app as app_module
from src.config import get_settings
import src.db.connection as conn_mod
import src.db.repository as repo_mod
from src.db.repository import ensure_schema, get_run, save_run
from src.graph.audit_runner import AuditResult
from src.workbench.visualizer.visualizer_tools import generate_plots, profile_dataset


def _make_csv(tmp_path, name: str = "sample.csv") -> str:
    df = pd.DataFrame(
        {
            "target": [1, 0, 1, 0],
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    path = tmp_path / name
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


def _make_fake_run_audit(error: str | None = None, slow: bool = False,
                         records: list | None = None):
    def fake(*, run_id: str, csv_path: str, target_column: str | None,
             check_params: dict[str, dict], semantic_overrides: dict[str, str],
             mode: str = "deterministic") -> AuditResult:
        if slow:
            if records is not None:
                records.append(("start", run_id, time.time()))
            time.sleep(0.1)
            if records is not None:
                records.append(("end", run_id, time.time()))
        if error:
            return AuditResult(
                run_id=run_id, status="completed", mode=mode,
                overall_severity=None, findings=[], report_json=None,
                report_text=None, profile=None, error=error,
            )
        save_run(
            run_id=run_id, csv_path=csv_path, target_column=target_column,
            status="completed", overall_severity="low",
            findings=[{"check_name": "schema", "severity": "low", "evidence": "ok",
                       "affected_columns": [], "confidence": 1.0}],
            report_json={"dataset_id": "sample", "rows": 4, "cols": 3,
                         "overall_severity": "low", "findings": []},
            report_text="Dataset: sample\nRows: 4, Cols: 3",
        )
        return AuditResult(
            run_id=run_id, status="completed", mode=mode,
            overall_severity="low",
            findings=[{"check_name": "schema", "severity": "low", "evidence": "ok",
                       "affected_columns": [], "confidence": 1.0}],
            report_json={"dataset_id": "sample", "rows": 4, "cols": 3,
                         "overall_severity": "low", "findings": []},
            report_text="Dataset: sample\nRows: 4, Cols: 3",
            profile={}, error=None,
        )
    return fake


@pytest.fixture
def client(tmp_path, monkeypatch):
    _use_sqlite_at(tmp_path, monkeypatch)
    app_module._RUNS.clear()
    tmp_uploads = tmp_path / "uploads"
    tmp_plots = tmp_path / "plots"
    monkeypatch.setattr(app_module, "_uploads_dir", lambda: tmp_uploads)
    monkeypatch.setattr(app_module, "_plots_dir", lambda run_id: tmp_plots / run_id)
    monkeypatch.setattr(app_module, "run_audit", _make_fake_run_audit())
    monkeypatch.setattr(app_module, "_ollama_reachable", lambda: True)
    # Disable FastAPI background-task scheduling so tests drive the background
    # function directly and deterministically (no threading).
    monkeypatch.setattr(app_module.BackgroundTasks, "add_task", lambda *a, **k: None)
    return TestClient(app_module.app)


# ── POST /audits + status flow ──────────────────────────────────────────

def test_start_and_get_audit_returns_completed(client, tmp_path):
    csv_path = _make_csv(tmp_path)
    with open(csv_path, "rb") as fh:
        content = fh.read()
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", content, "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]
    assert run_id in app_module._RUNS
    assert app_module._RUNS[run_id]["status"] == "queued"

    app_module._run_audit_background(
        run_id=run_id, csv_path=csv_path, target_column="target",
        check_params={}, semantic_overrides={}, mode="deterministic",
    )
    r = client.get(f"/audits/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["report_json"] is not None
    assert body["report_text"] is not None
    assert body["overall_severity"] == "low"


def test_failing_run_ends_failed(client, tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setattr(app_module, "run_audit", _make_fake_run_audit(error="boom"))
    with open(csv_path, "rb") as fh:
        content = fh.read()
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", content, "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    app_module._run_audit_background(
        run_id=run_id, csv_path=csv_path, target_column="target",
        check_params={}, semantic_overrides={}, mode="deterministic",
    )
    r = client.get(f"/audits/{run_id}")
    assert r.json()["status"] == "failed"
    assert r.json()["error_message"] == "boom"


def test_unknown_run_id_returns_404(client):
    r = client.get("/audits/does-not-exist")
    assert r.status_code == 404


def test_plots_before_completion_return_409(client):
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", b"target,num\n1,1.0\n2,2.0\n", "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    r = client.post(f"/audits/{run_id}/plots")
    assert r.status_code == 409


def test_old_database_run_can_be_fetched_and_plotted(client, tmp_path):
    csv_path = _make_csv(tmp_path)
    save_run(
        run_id="old-run-1", csv_path=csv_path, target_column="target",
        status="completed", overall_severity="low",
        findings=[{"check_name": "schema", "severity": "low", "evidence": "ok",
                   "affected_columns": [], "confidence": 1.0}],
        report_json={"dataset_id": "sample", "rows": 4, "cols": 3,
                     "overall_severity": "low", "findings": []},
        report_text="Dataset: sample\nRows: 4, Cols: 3",
    )
    r = client.get("/audits/old-run-1")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["report_json"] is not None

    r = client.post("/audits/old-run-1/plots")
    assert r.status_code == 200
    body = r.json()
    assert "plots" in body and "skipped" in body


# ── upload validation ───────────────────────────────────────────────────

def test_non_csv_file_returns_400(client):
    resp = client.post(
        "/audits",
        files={"file": ("data.txt", b"target,num\n1,1.0\n", "text/plain")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 400


def test_missing_target_column_returns_400(client, tmp_path):
    csv_path = _make_csv(tmp_path)
    with open(csv_path, "rb") as fh:
        content = fh.read()
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", content, "text/csv")},
        data={"target": "nonexistent", "mode": "deterministic"},
    )
    assert resp.status_code == 400


def test_unparseable_csv_returns_400(client):
    resp = client.post(
        "/audits",
        files={"file": ("bad.csv", b"\x00\x01\x02not-a-csv", "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 400


def test_oversized_upload_returns_413(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", b"target,num\n1,1.0\n2,2.0\n", "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 413


# ── plot path traversal ─────────────────────────────────────────────────

def test_plot_path_traversal_returns_400(client):
    # Percent-encoded traversal reaches the server decoded as ".." / "\"
    # and is rejected with 400. (Raw "../x.png" is normalized away by the
    # HTTP client before it reaches the app, which is also safe.)
    for name in ["%2E%2E", "%2E%2E%5Cx.png", "foo%2E%2E"]:
        r = client.get(f"/audits/some-run/plots/{name}")
        assert r.status_code == 400, f"{name!r} -> {r.status_code}"


# ── Ollama gating ───────────────────────────────────────────────────────

def test_agent_mode_requires_ollama(client, tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setattr(app_module, "_ollama_reachable", lambda: False)
    with open(csv_path, "rb") as fh:
        content = fh.read()
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", content, "text/csv")},
        data={"target": "target", "mode": "agent"},
    )
    assert resp.status_code == 503


def test_deterministic_mode_skips_ollama(client, tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    called: list[bool] = []
    monkeypatch.setattr(
        app_module, "_ollama_reachable", lambda: (called.append(True), False)[1]
    )
    with open(csv_path, "rb") as fh:
        content = fh.read()
    resp = client.post(
        "/audits",
        files={"file": ("sample.csv", content, "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert resp.status_code == 202, resp.text
    assert called == [], "Ollama must not be consulted for deterministic mode"


# ── health ──────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"api", "database", "backend", "ollama"}
    assert body["api"] == "ok"
    assert body["database"] == "ok"
    assert body["backend"] == "sqlite"
    assert body["ollama"] in ("ok", "unreachable")


# ── concurrency (lock serialises audits) ────────────────────────────────

def test_two_audits_do_not_run_at_the_same_time(client, tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    records: list[tuple[str, str, float]] = []
    monkeypatch.setattr(
        app_module, "run_audit", _make_fake_run_audit(slow=True, records=records)
    )
    with open(csv_path, "rb") as fh:
        content = fh.read()
    r1 = client.post(
        "/audits", files={"file": ("a.csv", content, "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    r2 = client.post(
        "/audits", files={"file": ("b.csv", content, "text/csv")},
        data={"target": "target", "mode": "deterministic"},
    )
    assert r1.status_code == 202 and r2.status_code == 202
    run_id_1 = r1.json()["run_id"]
    run_id_2 = r2.json()["run_id"]

    t1 = threading.Thread(
        target=app_module._run_audit_background,
        kwargs=dict(run_id=run_id_1, csv_path=csv_path, target_column="target",
                    check_params={}, semantic_overrides={}, mode="deterministic"),
    )
    t2 = threading.Thread(
        target=app_module._run_audit_background,
        kwargs=dict(run_id=run_id_2, csv_path=csv_path, target_column="target",
                    check_params={}, semantic_overrides={}, mode="deterministic"),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    by_id: dict[str, dict[str, float]] = {}
    for event, run_id, ts in records:
        by_id.setdefault(run_id, {})[event] = ts
    assert len(by_id) == 2
    intervals = sorted((d["start"], d["end"]) for d in by_id.values())
    assert intervals[0][1] <= intervals[1][0], "audits overlapped"


# ── generate_plots output_dir ──────────────────────────────────────────

def test_generate_plots_output_dir_writes_to_given_folder(tmp_path, monkeypatch):
    import src.workbench.visualizer.visualizer_tools as vt

    csv_path = _make_csv(tmp_path)
    profile = profile_dataset(csv_path, target_column="target")
    out_dir = tmp_path / "custom_plots"
    report = generate_plots(
        findings=[{"check_name": "schema", "severity": "low", "evidence": "ok",
                   "affected_columns": [], "confidence": 1.0}],
        profile=profile, csv_path=csv_path, target_column="target",
        output_dir=str(out_dir),
    )
    assert isinstance(report.plots, list) and isinstance(report.skipped, list)

    default_plots = tmp_path / "default_plots"
    monkeypatch.setattr(vt, "PLOTS_DIR", default_plots)
    report2 = generate_plots(
        findings=[], profile=profile, csv_path=csv_path, target_column="target",
        output_dir=None,
    )
    assert isinstance(report2.plots, list) and isinstance(report2.skipped, list)
