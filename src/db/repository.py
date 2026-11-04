"""Repository layer for M5 persistence.

Higher-level CRUD over the SQLAlchemy models: list audit runs, save a completed
run (findings + report), load a run by run_id, and register datasets.

Note: LangGraph checkpoints are managed by ``PostgresSaver`` in its own tables.
This repository only touches business tables: ``datasets``, ``audit_runs``,
``findings``, ``reports``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, joinedload

from src.db.connection import SessionLocal, init_db
from src.db.models import AuditRun, Dataset, Finding, Report


def ensure_schema() -> None:
    """Create all tables if they don't exist yet (call once at startup)."""
    init_db()


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent audit runs, newest first."""
    with SessionLocal() as session:
        rows = (
            session.query(AuditRun)
            .options(joinedload(AuditRun.dataset))
            .order_by(AuditRun.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "csv_path": r.csv_path,
                "target_column": r.target_column,
                "status": r.status,
                "overall_severity": r.overall_severity,
                "findings_count": r.findings_count,
                "dataset_name": r.dataset.name if r.dataset else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def get_run(run_id: str) -> dict[str, Any] | None:
    """Load a single audit run by run_id (== LangGraph thread_id)."""
    with SessionLocal() as session:
        run = (
            session.query(AuditRun)
            .options(joinedload(AuditRun.dataset), joinedload(AuditRun.report))
            .filter_by(run_id=run_id)
            .first()
        )
        if not run:
            return None

        return {
            "id": run.id,
            "run_id": run.run_id,
            "csv_path": run.csv_path,
            "target_column": run.target_column,
            "status": run.status,
            "overall_severity": run.overall_severity,
            "findings_count": run.findings_count,
            "dataset_name": run.dataset.name if run.dataset else None,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "report_json": run.report.report_json if run.report else None,
            "report_text": run.report.report_text if run.report else None,
        }


def save_run(
    *,
    run_id: str,
    csv_path: str,
    target_column: str | None = None,
    dataset_id: str | None = None,
    status: str = "completed",
    overall_severity: str | None = None,
    findings: list[dict[str, Any]] | None = None,
    report_json: dict[str, Any] | None = None,
    report_text: str | None = None,
) -> str:
    """Persist (or update) a completed audit run.

    Creates/updates the AuditRun row, inserts Finding rows, inserts/updates the
    Report row. Does NOT touch checkpoints — that is PostgresSaver's job.

    Returns the audit run id.
    """
    with SessionLocal() as session:  # type: Session
        run = session.query(AuditRun).filter_by(run_id=run_id).first()
        if not run:
            run = AuditRun(
                run_id=run_id,
                csv_path=csv_path,
                target_column=target_column,
                dataset_id=dataset_id,
                status=status,
                overall_severity=overall_severity,
                findings_count=len(findings or []),
            )
            session.add(run)
            session.flush()
        else:
            run.status = status
            run.overall_severity = overall_severity
            if findings is not None:
                run.findings_count = len(findings)
            session.flush()

        # Replace findings for this run
        if findings is not None:
            session.query(Finding).filter_by(audit_run_id=run.id).delete()
            for f in findings:
                session.add(
                    Finding(
                        audit_run_id=run.id,
                        check_name=f.get("check_name", ""),
                        severity=f.get("severity", "info"),
                        evidence=f.get("evidence", ""),
                        affected_columns=f.get("affected_columns", []),
                        suggested_fix=f.get("suggested_fix"),
                        confidence=int(float(f.get("confidence", 1.0)) * 100),
                    )
                )

        # Upsert report
        if report_json is not None:
            existing_report = session.query(Report).filter_by(audit_run_id=run.id).first()
            if existing_report:
                existing_report.report_json = report_json
                existing_report.report_text = report_text
            else:
                session.add(
                    Report(
                        audit_run_id=run.id,
                        report_json=report_json,
                        report_text=report_text,
                    )
                )

        session.commit()
        return run.id


def register_dataset(name: str, csv_path: str) -> str:
    """Register a dataset and return its id."""
    with SessionLocal() as session:  # type: Session
        ds = Dataset(name=name, csv_path=csv_path)
        session.add(ds)
        session.commit()
        return ds.id


__all__ = ["ensure_schema", "list_runs", "get_run", "save_run", "register_dataset"]
