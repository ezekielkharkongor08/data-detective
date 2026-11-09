"""Shared audit runner used by the CLI and (later) the FastAPI layer.

Runs one audit end to end: builds the checkpointer-backed graph, invokes it,
persists the result with save_run, and returns a plain result object. It never
prints or prompts, so it can run in a CLI, a background task, or a worker
thread without change.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - only when the sqlite dep is missing
    SqliteSaver = None  # type: ignore[assignment]

from src.config import get_settings
from src.db.repository import save_run
from src.graph.builder import build_agent_graph, build_graph


def _parse_columns(values: list[str]) -> list[str]:
    """Flatten repeatable/comma-separated values into unique column names."""
    columns: list[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if name and name not in columns:
                columns.append(name)
    return columns


def build_check_params(sensitive: list[str], favorable: str | None) -> dict[str, dict]:
    """Build check_params from sensitive columns and a favorable value.

    Mirrors the CLI's ``--sensitive``/``--favorable`` translation so the API
    and the CLI construct identical ``check_params``.
    """
    check_params: dict[str, dict] = {}
    bias_params: dict[str, object] = {}
    if sensitive:
        bias_params["sensitive_columns"] = sensitive
    if favorable:
        bias_params["favorable_value"] = favorable
    if bias_params:
        check_params["bias"] = bias_params
    return check_params


@dataclass(frozen=True)
class AuditResult:
    """Outcome of a single audit run.

    All fields are plain data so callers (CLI, API, worker) can serialize or
    inspect them without importing LangGraph state.
    """

    run_id: str
    status: str
    mode: str
    overall_severity: str | None
    findings: list[dict[str, Any]]
    report_json: dict[str, Any] | None
    report_text: str | None
    profile: dict[str, Any] | None
    error: str | None


def _build_graph(use_agent: bool, checkpointer: Any) -> Any:
    """Return the deterministic or agent graph compiled with a checkpointer."""
    return (
        build_agent_graph(checkpointer=checkpointer)
        if use_agent
        else build_graph(checkpointer=checkpointer)
    )


@contextmanager
def _persisted_graph(use_agent: bool):
    """Create the backend checkpointer, set up its tables, and build a graph.

    Postgres -> PostgresSaver (existing behavior). SQLite -> SqliteSaver.
    Prints one line at startup naming the backend. A Postgres connection failure
    is surfaced as a clear error asking to unset DATABASE_URL; it never silently
    falls back to SQLite.
    """
    settings = get_settings()
    print(f"Using {settings.backend_name} backend.")

    if settings.is_postgres:
        try:
            with PostgresSaver.from_conn_string(settings.postgres_conn_string) as saver:
                saver.setup()  # safe to call repeatedly: only creates missing tables
                yield _build_graph(use_agent, saver)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
            print(
                "Error: could not connect to PostgreSQL using DATABASE_URL.\n"
                "Unset DATABASE_URL (or set DATABASE_PATH) to use the local "
                "SQLite backend instead."
            )
            raise
    else:
        if SqliteSaver is None:
            raise RuntimeError(
                "SQLite backend selected but langgraph-checkpoint-sqlite is not "
                "installed. Install it or set DATABASE_URL to use PostgreSQL."
            )
        with SqliteSaver.from_conn_string(settings.database_path) as saver:
            saver.setup()
            yield _build_graph(use_agent, saver)


def run_audit(
    *,
    run_id: str,
    csv_path: str,
    target_column: str | None,
    check_params: dict[str, dict],
    semantic_overrides: dict[str, str],
    mode: str = "deterministic",
) -> AuditResult:
    """Run one audit end to end and persist it exactly as the CLI does.

    Builds the checkpointer-backed graph for the requested mode, invokes it with
    ``thread_id == run_id``, then saves the run with ``save_run``. Never prints,
    prompts, or calls sys.exit. Graph exceptions are caught and returned as an
    error result instead of being raised, so callers decide how to surface them.

    Raises ValueError for an unknown mode.
    """
    if mode not in ("deterministic", "agent"):
        raise ValueError(f"unknown mode: {mode!r}")

    use_agent = mode == "agent"

    try:
        with _persisted_graph(use_agent) as graph:
            config = {"configurable": {"thread_id": run_id}}

            state = graph.invoke(
                {
                    "csv_path": csv_path,
                    "target_column": target_column,
                    "check_params": check_params,
                    "semantic_overrides": semantic_overrides,
                },
                config=config,
            )
    except Exception as exc:  # noqa: BLE001 - return as error result, do not crash caller
        return AuditResult(
            run_id=run_id,
            status="completed",
            mode=mode,
            overall_severity=None,
            findings=[],
            report_json=None,
            report_text=None,
            profile=None,
            error=str(exc),
        )

    if state.get("error"):
        # Mirrors main.py: a state error stops the run before persisting.
        return AuditResult(
            run_id=run_id,
            status="completed",
            mode=mode,
            overall_severity=state.get("overall_severity"),
            findings=state.get("findings") or [],
            report_json=None,
            report_text=state.get("report_text"),
            profile=state.get("profile"),
            error=state["error"],
        )

    report = state.get("report")
    report_text = state.get("report_text")
    findings_raw = state.get("findings") or []

    findings_data = [
        {
            "check_name": f["check_name"],
            "severity": f["severity"],
            "evidence": f["evidence"],
            "affected_columns": f["affected_columns"],
            "confidence": f["confidence"],
        }
        for f in findings_raw
    ]

    save_run(
        run_id=run_id,
        csv_path=csv_path,
        target_column=target_column,
        status="completed",
        overall_severity=state.get("overall_severity"),
        findings=findings_data,
        report_json=json.loads(report) if report else None,
        report_text=report_text,
    )

    return AuditResult(
        run_id=run_id,
        status="completed",
        mode=mode,
        overall_severity=state.get("overall_severity"),
        findings=findings_data,
        report_json=json.loads(report) if report else None,
        report_text=report_text,
        profile=state.get("profile"),
        error=None,
    )


__all__ = ["run_audit", "AuditResult", "_parse_columns", "build_check_params"]
