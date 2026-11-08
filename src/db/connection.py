"""Database connection setup using SQLAlchemy.

Supports PostgreSQL (when ``DATABASE_URL`` is set) and SQLite (default, at
``./data_detective.db`` or ``DATABASE_PATH``). All database access goes
through SQLAlchemy sessions. The URL comes from ``DATABASE_URL`` in ``.env``
(never hardcoded).
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""


_settings = get_settings()
_is_sqlite = _settings.database_url.startswith("sqlite")

_engine_kwargs: dict = {"pool_pre_ping": True}
if _is_sqlite:
    # SQLite needs check_same_thread=False so the checkpointer and the app can
    # share the connection across threads.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}


def _make_engine() -> Engine:
    engine = create_engine(_settings.database_url, **_engine_kwargs)

    if _is_sqlite:
        # Enforce foreign keys on every SQLite connection.
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, connection_record):  # noqa: ARG001
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


engine = _make_engine()
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


def create_database_if_not_exists() -> None:
    """Create the target database if it does not exist (Postgres only).

    For SQLite the database file is created automatically on first connect, so
    this is a no-op. SQLAlchemy cannot create a database while connected to it,
    so Postgres connects to the ``postgres`` admin database first.
    """
    if _is_sqlite:
        return
    admin_connection_string = (
        f"postgresql+psycopg://{_settings.postgres_user}:{_settings.postgres_password}"
        f"@{_settings.postgres_host}:{_settings.postgres_port}/postgres"
    )
    admin_engine = create_engine(admin_connection_string, isolation_level="AUTOCOMMIT")
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
    Automatically creates the target database if it does not exist (Postgres).
    """
    create_database_if_not_exists()
    Base.metadata.create_all(bind=engine)


__all__ = ["Base", "engine", "SessionLocal", "get_db", "init_db", "create_database_if_not_exists"]
