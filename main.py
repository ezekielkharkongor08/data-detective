"""Data Detective CLI.

Run a pipeline audit or the LLM agent audit.

Examples:
    python main.py data/raw/adult.csv --target income
    python main.py data/raw/adult.csv --target income --agent
    python main.py data/raw/adult.csv --target income --format json
    python main.py data/raw/adult.csv --target income --format both
    python main.py --list
    python main.py data/raw/adult.csv --target income --resume <run_id>

The agent mode reads LLM settings from `.env`.
The canonical output is JSON (state["report"]); plain text is rendered from
the same object (state["report_text"]).

M5 persistence:
    --list          List past audit runs from PostgreSQL and exit.
    --resume ID     Resume a saved audit run by run_id (== LangGraph thread_id).
    --thread-id ID  Use this thread_id for the run (auto-generated if omitted).
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
        "--agent",
        action="store_true",
        help="Use the M4 LLM agent graph (requires the model in .env).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "both"],
        default="text",
        help="Output format: text (default), json, or both.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List past audit runs from PostgreSQL and exit.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume a saved audit run by run_id (== LangGraph thread_id).",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Thread ID for this audit run (auto-generated if omitted).",
    )
    return parser.parse_args()


def _print_report(report: str | None, report_text: str | None, fmt: str) -> None:
    if fmt == "json":
        print(report if report is not None else "")
    elif fmt == "both":
        if report_text:
            print(report_text)
            print()
        print(report if report is not None else "")
    else:
        print(report_text if report_text is not None else report or "(no report produced)")


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
        print(f"{'Run ID':<38s} | {'Status':<12s} | {'Severity':<10s} | {'Findings':<9s} | CSV")
        print("-" * 100)
        for r in runs:
            print(
                f"{r['run_id']:<38s} | {r['status']:<12s} | {r['overall_severity'] or '-':<10s} | {r['findings_count']:<9d} | {r['csv_path']}"
            )
        return

    if not args.csv_path:
        print("Error: csv_path is required (use --list to see past runs).")
        return

    # run_id is the LangGraph thread_id for this audit run.
    run_id = args.thread_id or str(uuid.uuid4())

    # Build the graph with the PostgresSaver checkpointer.
    with _persisted_graph(args.agent) as graph:
        config = {"configurable": {"thread_id": run_id}}

        # Resume: LangGraph loads the last checkpoint for this thread_id
        # automatically when the same thread_id is reused. Pass minimal input.
        initial_state: dict = {}
        if not args.resume:
            # Fresh run: provide csv_path and target_column as initial input.
            initial_state = {"csv_path": args.csv_path, "target_column": args.target}

        state = graph.invoke(initial_state, config=config)

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
                "suggested_fix": f.get("suggested_fix"),
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

        _print_report(report, report_text, args.format)


if __name__ == "__main__":
    main()
