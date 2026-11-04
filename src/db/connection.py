"""Database connection setup using SQLAlchemy + PostgreSQL.

All database access goes through SQLAlchemy sessions. The connection URL is
read from ``DATABASE_URL`` in ``.env`` (never hardcoded, per AGENTS.md rule).
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""


_settings = get_settings()
engine = create_engine(_settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and close it when done.

    Intended for use as a FastAPI dependency or context-managed helper.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined on ``Base``.

    Safe to call on startup; tables that already exist are left untouched.
    """
    Base.metadata.create_all(bind=engine)


__all__ = ["Base", "engine", "SessionLocal", "get_db", "init_db"]
