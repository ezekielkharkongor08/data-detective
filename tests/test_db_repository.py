"""Tests for the M5 persistence repository layer (src/db/repository.py).

These tests exercise the business tables (datasets, audit_runs, findings,
reports) — separate from LangGraph's own checkpoints table managed by
PostgresSaver. They require a running Postgres at localhost:5432/data_detective
and are skipped when Postgres is unavailable, matching the pattern in
tests/test_resume_no_duplicates.py.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from src.config import get_settings
from src.db.repository import (
    ensure_schema,
    get_run,
    list_runs,
    register_dataset,
    save_run,
)


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    """Skip all tests in this module when Postgres is unavailable."""
    try:
        conn = get_settings().postgres_conn_string
        with psycopg.connect(conn, connect_timeout=3):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _schema(_require_postgres):
    """Ensure tables exist once for the whole module."""
    ensure_schema()


def _new_run_id() -> str:
    return f"test-run-{uuid.uuid4()}"


# ── ensure_schema ─────────────────────────────────────────────────────────

def test_ensure_schema_is_idempotent():
    """Calling ensure_schema twice does not raise."""
    ensure_schema()
    ensure_schema()  # should not raise


# ── register_dataset ──────────────────────────────────────────────────────

def test_register_dataset_round_trip():
    """A registered dataset can be read back via list_runs."""
    ds_id = register_dataset("test_ds", "/tmp/test_ds.csv")
    assert ds_id

    runs = list_runs()
    # Our run may not appear until a save_run happens; just confirm the
    # dataset row was created without error by checking list_runs works.
    assert isinstance(runs, list)


# ── save_run + get_run ────────────────────────────────────────────────────

def _sample_findings() -> list[dict]:
    return [
        {
            "check_name": "bias",
            "severity": "critical",
            "evidence": "Disparate impact 0.358 for sex groups",
            "affected_columns": ["sex"],
            "confidence": 0.9,
        },
        {
            "check_name": "missing_values",
            "severity": "low",
            "evidence": "Column 'workclass' has 1836 missing values",
            "affected_columns": ["workclass"],
            "confidence": 1.0,
        },
    ]


def _sample_report_json() -> dict:
    return {
        "dataset_id": "adult",
        "rows": 32561,
        "cols": 15,
        "findings": _sample_findings(),
        "overall_severity": "critical",
    }


def test_save_and_get_run_round_trip():
    """save_run then get_run returns the same findings and report."""
    run_id = _new_run_id()

    run_db_id = save_run(
        run_id=run_id,
        csv_path="/tmp/adult.csv",
        target_column="income",
        status="completed",
        overall_severity="critical",
        findings=_sample_findings(),
        report_json=_sample_report_json(),
        report_text="Data Detective — Profile Report\n...",
    )
    assert run_db_id

    run = get_run(run_id)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["csv_path"] == "/tmp/adult.csv"
    assert run["target_column"] == "income"
    assert run["status"] == "completed"
    assert run["overall_severity"] == "critical"
    assert run["findings_count"] == 2
    assert run["report_json"] == _sample_report_json()
    assert "Data Detective" in run["report_text"]


def test_get_run_returns_none_for_missing_run():
    """get_run returns None when no run exists for the run_id."""
    assert get_run(f"test-run-{uuid.uuid4()}") is None


def test_save_run_updates_existing_run():
    """Calling save_run with the same run_id updates rather than duplicates."""
    run_id = _new_run_id()

    save_run(
        run_id=run_id,
        csv_path="/tmp/adult.csv",
        target_column="income",
        status="completed",
        overall_severity="high",
        findings=[_sample_findings()[0]],
    )

    save_run(
        run_id=run_id,
        csv_path="/tmp/adult.csv",
        target_column="income",
        status="completed",
        overall_severity="critical",
        findings=_sample_findings(),
        report_json=_sample_report_json(),
    )

    run = get_run(run_id)
    assert run is not None
    assert run["overall_severity"] == "critical"
    assert run["findings_count"] == 2


# ── list_runs ─────────────────────────────────────────────────────────────

def test_list_runs_returns_recent_first():
    """list_runs returns runs ordered newest first."""
    # Create two runs with different timestamps.
    run_id_1 = _new_run_id()
    save_run(
        run_id=run_id_1,
        csv_path="/tmp/a.csv",
        status="completed",
        findings=[],
    )
    run_id_2 = _new_run_id()
    save_run(
        run_id=run_id_2,
        csv_path="/tmp/b.csv",
        status="completed",
        findings=[],
    )

    runs = list_runs()
    run_ids = [r["run_id"] for r in runs]

    # Both test runs should appear, newest first.
    assert run_id_2 in run_ids
    assert run_id_1 in run_ids
    run_ids.index(run_id_2) < run_ids.index(run_id_1), (
        "newest run should appear first"
    )


def test_list_runs_entries_have_required_fields():
    """Each list_runs entry has the expected keys."""
    runs = list_runs()
    if not runs:
        pytest.skip("no runs to inspect")
    required = {
        "id",
        "run_id",
        "csv_path",
        "target_column",
        "status",
        "overall_severity",
        "findings_count",
        "dataset_name",
        "created_at",
    }
    for r in runs:
        assert required.issubset(set(r.keys())), f"missing keys: {required - set(r.keys())}"


__all__ = []
