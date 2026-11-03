"""Data Detective CLI.

Run a pipeline audit or the LLM agent audit.

Examples:
    python main.py data/raw/adult.csv --target income
    python main.py data/raw/adult.csv --target income --agent
    python main.py data/raw/adult.csv --target income --format json
    python main.py data/raw/adult.csv --target income --format both

The agent mode reads LLM settings from `.env` (see `.env.example`).
The canonical output is JSON (state["report"]); plain text is rendered from
the same object (state["report_text"]).
"""

import argparse
import json
import sys

from src.graph.builder import build_agent_graph, build_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a CSV dataset.")
    parser.add_argument("csv_path", help="Path to the CSV file to audit.")
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
    return parser.parse_args()


def main() -> None:
    # Print the report's unicode characters correctly on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    graph = build_agent_graph() if args.agent else build_graph()
    assert args.csv_path, "csv_path is required"
    state = graph.invoke({"csv_path": args.csv_path, "target_column": args.target})

    if state.get("error"):
        print(f"Error: {state['error']}")
        return

    report = state.get("report")
    report_text = state.get("report_text")

    if args.format == "json":
        print(report if report is not None else "")
    elif args.format == "both":
        if report_text:
            print(report_text)
            print()
        print(report if report is not None else "")
    else:
        # Default: human-friendly plain text, fall back to JSON.
        print(report_text if report_text is not None else report or "(no report produced)")


if __name__ == "__main__":
    main()
