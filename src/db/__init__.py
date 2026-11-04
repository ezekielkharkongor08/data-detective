"""Database layer (M5 persistence): PostgreSQL + SQLAlchemy + LangGraph PostgresSaver.

LangGraph checkpoints are managed by ``PostgresSaver`` (from
``langgraph_checkpoint.postgres``) in its own tables. Business data
(datasets, audit runs, findings, reports) lives in SQLAlchemy models below.
"""

from src.db.connection import Base, engine, SessionLocal, get_db, init_db
from src.db.models import AuditRun, Dataset, Finding, Report
from src.db.repository import (
    ensure_schema,
    get_run,
    list_runs,
    register_dataset,
    save_run,
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "AuditRun",
    "Dataset",
    "Finding",
    "Report",
    "ensure_schema",
    "get_run",
    "list_runs",
    "register_dataset",
    "save_run",
]
