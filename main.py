"""Data Detective CLI.

Run a pipeline audit or the LLM agent audit; every run is saved to PostgreSQL.

Examples:
    python main.py data/raw/adult.csv --target income
    python main.py data/raw/adult.csv --target income --agent
    python main.py --list
    python main.py data/raw/adult.csv --target income --thread-id run-001
    python main.py data/raw/adult.csv --target income --sensitive <column_name> --favorable <val>
    python main.py --workbench visualize --run-id <uuid>

The graph runs to completion in a single invoke; there is no human-in-the-loop
pause. LLM settings come from `.env`.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from src.db.repository import ensure_schema, get_run, list_runs, save_run
from src.graph.audit_runner import build_check_params, run_audit, _parse_columns
from src.tools.profiling import profile_dataset
from src.workbench.visualizer.visualizer_tools import generate_plots


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
        "--workbench",
        choices=["visualize"],
        default=None,
        help="Run a Phase 2 workbench tool against a past audit run.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Deprecated alias for --workbench visualize (requires --run-id).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID of a past audit run (required with --workbench).",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots after the audit without prompting.",
    )
    plot_group.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation after the audit without prompting.",
    )
    return parser.parse_args()


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


def _print_plot_report(report: object) -> None:
    """Print the plot report: each saved path, a count, and every skip reason.

    Never prints only "no plots were generated" without the reasons.
    """
    plot_paths = getattr(report, "plots", []) or []
    skipped = getattr(report, "skipped", []) or []
    if plot_paths:
        print("Visualizer Workbench complete. Plots saved:")
        for p in plot_paths:
            print(f"  {p}")
    print(f"{len(plot_paths)} plot(s) generated.")
    if skipped:
        print("Skipped plots:")
        for item in skipped:
            print(f"  - {item.get('reason')}")


def _should_generate_plots(args: argparse.Namespace) -> bool:
    """Decide whether to generate plots after an audit.

    ``--plots`` forces yes, ``--no-plots`` forces no. Otherwise the user is
    prompted on a TTY; non-interactive input defaults to no.
    """
    if args.plots:
        return True
    if args.no_plots:
        return False
    if not sys.stdin.isatty():
        return False
    try:
        answer = input("Generate plots? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ("y", "yes")


def _run_visualizer_workbench(run: dict) -> None:
    """Generate charts for an existing audit run using deterministic plot selection.

    The profile is not stored in the database, so it is recomputed from the CSV
    (read-only V1 profiling). If the CSV no longer exists, prints a clear
    message naming the path and returns without crashing.
    """
    report_json = run.get("report_json") or {}
    findings = report_json.get("findings") or []
    csv_path = run.get("csv_path")
    target_column = run.get("target_column")

    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Recompute the profile from the CSV (read-only V1 profiling).
    profile = profile_dataset(csv_path, target_column=target_column)

    report = generate_plots(
        findings=findings,
        profile=profile,
        csv_path=csv_path,
        target_column=target_column,
    )

    _print_plot_report(report)


def main() -> None:
    # Windows consoles default to a non-UTF8 codepage, which garbles report symbols.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    # Keep the old `--visualize` flag working while standardizing on --workbench.
    if args.visualize and not args.workbench:
        args.workbench = "visualize"

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

    if args.workbench:
        if not args.run_id:
            print(f"Error: --run-id is required with --workbench {args.workbench}.")
            return

        run = get_run(args.run_id)
        if not run:
            print(f"Error: no audit run found for run-id {args.run_id}.")
            return

        if args.workbench == "visualize":
            _run_visualizer_workbench(run)
        return

    if not args.csv_path:
        print("Error: csv_path is required (use --list to see past runs).")
        return

    # run_id doubles as the LangGraph thread_id, linking the two histories.
    run_id = args.thread_id or str(uuid.uuid4())

    result = run_audit(
        run_id=run_id,
        csv_path=args.csv_path,
        target_column=args.target,
        check_params=build_check_params(_parse_columns(args.sensitive), args.favorable),
        semantic_overrides=_build_semantic_overrides(args),
        mode="agent" if args.agent else "deterministic",
    )

    if result.error:
        print(f"Error: {result.error}")
        return

    report_text = result.report_text
    if report_text:
        print(report_text)
    else:
        print("(no report produced)")

    if _should_generate_plots(args):
        # Generate plots from the returned audit results (no database read).
        profile = result.profile or {}
        report = generate_plots(
            findings=result.findings,
            profile=profile,
            csv_path=args.csv_path,
            target_column=args.target,
        )
        _print_plot_report(report)


if __name__ == "__main__":
    main()
