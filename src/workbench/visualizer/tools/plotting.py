"""Matplotlib plotting tools for the Visualizer Workbench (V2).

Each tool is built with LangChain's ``StructuredTool.from_function``,
mirroring V1's ``src/graph/tools.py`` pattern exactly.

V1 reuse (read-only):
- ``read_csv_robust`` is imported from ``src.tools.csv_reader`` and called
  directly. It is NOT duplicated, rewritten, or wrapped inside V2.

Charts are saved strictly as .png files to the root-level ``plots/``
directory using default matplotlib colors. No database image storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend for saving PNGs without a display
import matplotlib.pyplot as plt

from langchain_core.tools import StructuredTool

from src.tools.csv_reader import read_csv_robust


# ---------------------------------------------------------------------------
# Directory setup (root-level plots/)
# ---------------------------------------------------------------------------

# This file lives at src/workbench/visualizer/tools/plotting.py.
# Walk up from __file__ until we find the project root (marker: pyproject.toml).
_PROJECT_ROOT: Path | None = None
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "pyproject.toml").exists():
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is None:
    # Fallback: four levels up from this file.
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]

PLOTS_DIR = _PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: matplotlib.figure.Figure, filename: str) -> str:
    """Save a figure to the root plots/ directory and return its path."""
    plot_path = PLOTS_DIR / filename
    fig.savefig(plot_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return str(plot_path)


def _result(
    tool_name: str,
    plot_file: str | None,
    status: str,
    columns: list[str],
    evidence: str,
) -> str:
    """Serialize a tool result to a JSON string."""
    return json.dumps(
        {
            "tool_name": tool_name,
            "plot_file": plot_file,
            "status": status,
            "columns_affected": columns,
            "evidence": evidence,
        }
    )


# ---------------------------------------------------------------------------
# Plotting functions (strictly matplotlib, default colors)
# ---------------------------------------------------------------------------


def _plot_outliers(csv_path: str, column: str) -> str:
    """Draw a boxplot for ``column`` and save it as a PNG."""
    try:
        df = read_csv_robust(csv_path)
    except Exception as exc:  # noqa: BLE001 — tool must not crash the agent
        return _result(
            "run_plot_outliers",
            None,
            "error",
            [column],
            f"Failed to read CSV: {type(exc).__name__}: {exc}",
        )

    if column not in df.columns:
        return _result(
            "run_plot_outliers",
            None,
            "skipped",
            [column],
            f"Column '{column}' not found in dataset.",
        )

    fig, ax = plt.subplots()
    ax.boxplot(df[column].dropna(), vert=True)
    ax.set_title(f"Boxplot of {column}")
    ax.set_ylabel(column)
    plot_file = _save_fig(fig, f"outliers_{column}.png")
    return _result(
        "run_plot_outliers",
        plot_file,
        "ok",
        [column],
        f"Boxplot saved for column '{column}'.",
    )


def _plot_skewness(csv_path: str, column: str) -> str:
    """Draw a histogram for ``column`` and save it as a PNG."""
    try:
        df = read_csv_robust(csv_path)
    except Exception as exc:  # noqa: BLE001 — tool must not crash the agent
        return _result(
            "run_plot_skewness",
            None,
            "error",
            [column],
            f"Failed to read CSV: {type(exc).__name__}: {exc}",
        )

    if column not in df.columns:
        return _result(
            "run_plot_skewness",
            None,
            "skipped",
            [column],
            f"Column '{column}' not found in dataset.",
        )

    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins="auto")
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plot_file = _save_fig(fig, f"skewness_{column}.png")
    return _result(
        "run_plot_skewness",
        plot_file,
        "ok",
        [column],
        f"Histogram saved for column '{column}'.",
    )


def _plot_imbalance(csv_path: str, target_column: str) -> str:
    """Draw a bar chart of class distribution for ``target_column``."""
    try:
        df = read_csv_robust(csv_path)
    except Exception as exc:  # noqa: BLE001 — tool must not crash the agent
        return _result(
            "run_plot_imbalance",
            None,
            "error",
            [target_column],
            f"Failed to read CSV: {type(exc).__name__}: {exc}",
        )

    if target_column not in df.columns:
        return _result(
            "run_plot_imbalance",
            None,
            "skipped",
            [target_column],
            f"Column '{target_column}' not found in dataset.",
        )

    counts = df[target_column].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Class Distribution of {target_column}")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    plot_file = _save_fig(fig, f"imbalance_{target_column}.png")
    return _result(
        "run_plot_imbalance",
        plot_file,
        "ok",
        [target_column],
        f"Class distribution saved for target '{target_column}'.",
    )


# ---------------------------------------------------------------------------
# LangChain StructuredTools (mirrors V1's src/graph/tools.py)
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str, func: Any) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
    )


run_plot_outliers = _make_tool(
    name="run_plot_outliers",
    description=(
        "Draw a boxplot for a numeric column to visualize outliers. "
        "Takes csv_path and column. Returns the saved PNG path as JSON."
    ),
    func=_plot_outliers,
)

run_plot_skewness = _make_tool(
    name="run_plot_skewness",
    description=(
        "Draw a histogram for a numeric column to visualize skewness / "
        "distribution. Takes csv_path and column. Returns the saved PNG "
        "path as JSON."
    ),
    func=_plot_skewness,
)

run_plot_imbalance = _make_tool(
    name="run_plot_imbalance",
    description=(
        "Draw a bar chart of class distribution for a target column to "
        "visualize imbalance. Takes csv_path and target_column. Returns "
        "the saved PNG path as JSON."
    ),
    func=_plot_imbalance,
)

ALL_TOOLS = [run_plot_outliers, run_plot_skewness, run_plot_imbalance]


__all__ = [
    "run_plot_outliers",
    "run_plot_skewness",
    "run_plot_imbalance",
    "ALL_TOOLS",
    "PLOTS_DIR",
]
