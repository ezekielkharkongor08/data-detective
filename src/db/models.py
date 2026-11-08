"""SQLAlchemy ORM models for M5 persistence.

One table per concept: datasets, audit runs, findings, and reports. LangGraph
checkpoints live in separate ``PostgresSaver`` tables and must not be mixed
with these business tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.db.connection import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    csv_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    runs: Mapped[list["AuditRun"]] = relationship(back_populates="dataset")


class AuditRun(Base):
    """One audit run. ``run_id`` equals the LangGraph ``thread_id`` value."""

    __tablename__ = "audit_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    csv_path: Mapped[str] = mapped_column(String, nullable=False)
    target_column: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")  # running | completed | failed
    overall_severity: Mapped[str | None] = mapped_column(String, nullable=True)
    findings_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    dataset: Mapped["Dataset | None"] = relationship(back_populates="runs")
    findings: Mapped[list["Finding"]] = relationship(back_populates="audit_run", cascade="all, delete-orphan")
    report: Mapped["Report | None"] = relationship(back_populates="audit_run", uselist=False, cascade="all, delete-orphan")


class Finding(Base):
    """A structured Finding extracted and stored from an audit run."""

    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audit_run_id: Mapped[str] = mapped_column(ForeignKey("audit_runs.id"), nullable=False, index=True)
    check_name: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    affected_columns: Mapped[list[str] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Integer, nullable=False, default=100)  # stored as 0-100 int
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    audit_run: Mapped["AuditRun"] = relationship(back_populates="findings")


class Report(Base):
    """Final audit report (JSON source of truth + plain-text rendering)."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audit_run_id: Mapped[str] = mapped_column(ForeignKey("audit_runs.id"), nullable=False, unique=True)
    report_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    report_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    audit_run: Mapped["AuditRun"] = relationship(back_populates="report")


__all__ = ["Dataset", "AuditRun", "Finding", "Report"]
