"""Database connection setup using SQLAlchemy + PostgreSQL.

All database access goes through SQLAlchemy sessions. The URL comes from
``DATABASE_URL`` in ``.env`` (never hardcoded).
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, text
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


def _admin_connection_string() -> str:
    """Build a connection string to the postgres admin database.

    CREATE DATABASE cannot run inside a transaction, so we connect to the
    ``postgres`` maintenance database and run it with AUTOCOMMIT.
    """
    return (
        f"postgresql+psycopg://{_settings.postgres_user}:{_settings.postgres_password}"
        f"@{_settings.postgres_host}:{_settings.postgres_port}/postgres"
    )


def create_database_if_not_exists() -> None:
    """Create the target PostgreSQL database if it does not exist.

    SQLAlchemy cannot create a database while connected to it, so this
    connects to the ``postgres`` admin database first.
    """
    admin_engine = create_engine(_admin_connection_string(), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": _settings.postgres_db},
            )
            if result.fetchone() is None:
                conn.execute(text(f'CREATE DATABASE "{_settings.postgres_db}"'))
    finally:
        admin_engine.dispose()


def init_db() -> None:
    """Create all tables defined on ``Base``.

    Safe to call on startup; tables that already exist are left untouched.
    Automatically creates the target database if it does not exist.
    """
    create_database_if_not_exists()
    Base.metadata.create_all(bind=engine)


__all__ = ["Base", "engine", "SessionLocal", "get_db", "init_db", "create_database_if_not_exists"]
