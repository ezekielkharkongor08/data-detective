"""Data Detective CLI.

Run a pipeline audit or the LLM agent audit.

Examples:
    python main.py data/raw/adult.csv --target income
    python main.py data/raw/adult.csv --target income --agent
    python main.py --list
    python main.py data/raw/adult.csv --target income --thread-id run-001
    python main.py data/raw/adult.csv --target income --sensitive <column_name> --favorable <val>  # Runs fairness and bias checks for specific groups

The agent mode reads LLM settings from `.env`.
Plain-text output is printed to stdout; the full JSON report is persisted to
PostgreSQL.

Persistence:
    --list            List past audit runs from PostgreSQL and exit.
    --thread-id ID    Use this thread_id for the run (auto-generated if omitted).

The graph runs to completion in a single invoke; there is no human-in-the-loop
pause. PostgresSaver silently saves run history to the database.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from src.config import get_settings
from src.db.repository import ensure_schema, list_runs, save_run
from src.graph.builder import build_agent_graph, build_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a CSV dataset.")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to the CSV file to audit.")
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column (needed for imbalance/leakage checks).",
    )
    parser.add_argument(
        "--sensitive",
        action="append",
        default=[],
        metavar="COLUMN",
        help=(
            "Sensitive column for the fairness check. Repeat the flag or use "
            "a comma-separated list."
        ),
    )
    parser.add_argument(
        "--favorable",
        default=None,
        metavar="VALUE",
        help="Favorable outcome value for the fairness check (for example '>50K').",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the M4 LLM agent graph (requires the model in .env).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List past audit runs from PostgreSQL and exit.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Thread ID for this audit run (auto-generated if omitted).",
    )
    return parser.parse_args()


def _parse_columns(values: list[str]) -> list[str]:
    """Flatten repeatable/comma-separated CLI values into unique names."""
    columns: list[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if name and name not in columns:
                columns.append(name)
    return columns


def _build_check_params(args: argparse.Namespace) -> dict[str, dict]:
    """Translate friendly CLI flags into the generic ``check_params`` dict."""
    check_params: dict[str, dict] = {}

    sensitive = _parse_columns(args.sensitive)
    bias_params: dict[str, object] = {}
    if sensitive:
        bias_params["sensitive_columns"] = sensitive
    if args.favorable:
        bias_params["favorable_value"] = args.favorable
    if bias_params:
        check_params["bias"] = bias_params

    return check_params


@contextmanager
def _persisted_graph(use_agent: bool):
    """Create a PostgresSaver, set up its tables, and build a graph with it.

    Yields the compiled graph. The saver is closed when the context exits.
    """
    settings = get_settings()
    with PostgresSaver.from_conn_string(settings.postgres_conn_string) as saver:
        saver.setup()  # idempotent: creates checkpoints/checkpoint_writes tables
        graph = (
            build_agent_graph(checkpointer=saver)
            if use_agent
            else build_graph(checkpointer=saver)
        )
        yield graph


def main() -> None:
    # Print the report's unicode characters correctly on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    # Ensure the database schema exists (M5 business tables).
    ensure_schema()

    # --list: show past audit runs and exit.
    if args.list:
        runs = list_runs()
        if not runs:
            print("(no audit runs found)")
            return
        print(
            f"{'Run ID':<38s} | {'Status':<12s} | {'Approval':<11s} | {'Severity':<10s} | {'Findings':<9s} | CSV"
        )
        print("-" * 120)
        for r in runs:
            print(
                f"{r['run_id']:<38s} | {r['status']:<12s} | {r['approval_status']:<11s} | {r['overall_severity'] or '-':<10s} | {r['findings_count']:<9d} | {r['csv_path']}"
            )
        return

    if not args.csv_path:
        print("Error: csv_path is required (use --list to see past runs).")
        return

    # run_id is the LangGraph thread_id for this audit run.
    run_id = args.thread_id or str(uuid.uuid4())

    # Build the graph with the PostgresSaver checkpointer. The graph runs to
    # completion in a single invoke() — there is no human-in-the-loop pause.
    with _persisted_graph(args.agent) as graph:
        config = {"configurable": {"thread_id": run_id}}

        state = graph.invoke(
            {
                "csv_path": args.csv_path,
                "target_column": args.target,
                "check_params": _build_check_params(args),
            },
            config=config,
        )

        if state.get("error"):
            print(f"Error: {state['error']}")
            return

        report = state.get("report")
        report_text = state.get("report_text")

        # Persist the completed run to business tables (PostgresSaver already
        # saved the graph checkpoints to its own tables).
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
            csv_path=args.csv_path,
            target_column=args.target,
            status="completed",
            overall_severity=state.get("overall_severity"),
            findings=findings_data,
            report_json=json.loads(report) if report else None,
            report_text=report_text,
        )

        if report_text:
            print(report_text)
        else:
            print("(no report produced)")


if __name__ == "__main__":
    main()
