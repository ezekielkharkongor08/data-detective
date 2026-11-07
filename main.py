"""Data Detective CLI.

Run a pipeline audit or the LLM agent audit; every run is saved to PostgreSQL.

Examples:
    python main.py data/raw/adult.csv --target income
    python main.py data/raw/adult.csv --target income --agent
    python main.py --list
    python main.py data/raw/adult.csv --target income --thread-id run-001
    python main.py data/raw/adult.csv --target income --sensitive <column_name> --favorable <val>

The graph runs to completion in a single invoke; there is no human-in-the-loop
pause. LLM settings come from `.env`.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from src.config import get_settings
from src.db.repository import ensure_schema, get_run, list_runs, save_run
from src.graph.builder import build_agent_graph, build_graph
from src.workbench.visualizer.visualizer_graph import build_visualizer_graph


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
        "--semantic-override",
        action="append",
        default=[],
        metavar="COLUMN=TYPE",
        help=(
            "Override the semantic type of a column (for example "
            "'SeniorCitizen=binary'). Repeat the flag or use a comma-separated "
            "list. Overrides win over every classifier rule."
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations from a past audit run (requires --run-id).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID of a past audit to visualize (use with --visualize).",
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


def _build_semantic_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Translate ``--semantic-override COLUMN=TYPE`` flags into a dict.

    Values are validated against ``SEMANTIC_TYPES``; invalid types produce a
    warning and are skipped.
    """
    from src.tools.profiling import SEMANTIC_TYPES

    overrides: dict[str, str] = {}
    for value in args.semantic_override:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                print(
                    f"Warning: ignoring malformed --semantic-override "
                    f"{part!r} (expected COLUMN=TYPE)."
                )
                continue
            name, new_type = part.split("=", 1)
            name = name.strip()
            new_type = new_type.strip()
            if new_type not in SEMANTIC_TYPES:
                print(
                    f"Warning: ignoring --semantic-override {part!r}: "
                    f"unknown type {new_type!r}."
                )
                continue
            overrides[name] = new_type
    return overrides


@contextmanager
def _persisted_graph(use_agent: bool):
    """Create a PostgresSaver, set up its tables, and build a graph with it.

    Yields the compiled graph. The saver is closed when the context exits.
    """
    settings = get_settings()
    with PostgresSaver.from_conn_string(settings.postgres_conn_string) as saver:
        saver.setup()  # safe to call repeatedly: only creates missing tables
        graph = (
            build_agent_graph(checkpointer=saver)
            if use_agent
            else build_graph(checkpointer=saver)
        )
        yield graph


def main() -> None:
    # Windows consoles default to a non-UTF8 codepage, which garbles report symbols.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    ensure_schema()

    if args.list:
        runs = list_runs()
        if not runs:
            print("(no audit runs found)")
            return
        print(
            f"{'Run ID':<38s} | {'Status':<12s} | {'Severity':<10s} | {'Findings':<9s} | CSV"
        )
        print("-" * 120)
        for r in runs:
            print(
                f"{r['run_id']:<38s} | {r['status']:<12s} | {r['overall_severity'] or '-':<10s} | {r['findings_count']:<9d} | {r['csv_path']}"
            )
        return

    if args.visualize:
        if not args.run_id:
            print("Error: --run-id is required with --visualize.")
            return
        run = get_run(args.run_id)
        if not run:
            print(f"Error: no audit run found for run-id {args.run_id}.")
            return
        report_json = run.get("report_json") or {}
        findings_in = report_json.get("findings") or []
        csv_path = run.get("csv_path")
        target_column = run.get("target_column")

        graph = build_visualizer_graph()
        state = graph.invoke(
            {
                "findings_in": findings_in,
                "csv_path": csv_path,
                "target_column": target_column,
                "want_plots": True,
            }
        )

        plot_paths = state.get("plot_paths") or []
        if plot_paths:
            print("Visualizer Workbench complete. Plots saved:")
            for p in plot_paths:
                print(f"  {p}")
        else:
            print("Visualizer Workbench complete. No plots were generated.")
        return

    if not args.csv_path:
        print("Error: csv_path is required (use --list to see past runs).")
        return

    # run_id doubles as the LangGraph thread_id, linking the two histories.
    run_id = args.thread_id or str(uuid.uuid4())

    with _persisted_graph(args.agent) as graph:
        config = {"configurable": {"thread_id": run_id}}

        state = graph.invoke(
            {
                "csv_path": args.csv_path,
                "target_column": args.target,
                "check_params": _build_check_params(args),
                "semantic_overrides": _build_semantic_overrides(args),
            },
            config=config,
        )

        if state.get("error"):
            print(f"Error: {state['error']}")
            return

        report = state.get("report")
        report_text = state.get("report_text")

        # PostgresSaver only stores graph state; findings go to the business tables too.
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
