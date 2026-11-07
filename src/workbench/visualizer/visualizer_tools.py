"""Matplotlib plotting tools for the Visualizer Workbench (V2).

Built with LangChain ``StructuredTool.from_function``, mirroring V1's
``src/graph/tools.py``; ``read_csv_robust`` is reused read-only. Charts go
to ``plots/<dataset>/*.png`` with default matplotlib colors. Figures use the
``Figure`` API instead of ``pyplot`` because ``ToolNode`` can run tools in
parallel and ``pyplot`` is not thread-safe.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from matplotlib.figure import Figure

from src.tools.csv_reader import read_csv_robust
from src.tools.profiling import get_semantic_lookup, profile_dataset
from src.tools.target_type import detect_target_type


# --------------------------------------------------------------------- config

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # src/workbench/visualizer/visualizer_tools.py -> project root is parents[3]
    return here.parents[min(3, len(here.parents) - 1)]


_PROJECT_ROOT = _find_project_root()
PLOTS_DIR = _PROJECT_ROOT / "plots"

_MIN_NUMERIC_SHARE = 0.9   # text columns count as numeric if >= 90% parse
_MAX_HIST_BINS = 200       # guards against huge bin counts on skewed data
_MAX_CLASSES = 50          # guards against unreadable bar charts
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


# -------------------------------------------------------------------- helpers

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


def _safe_filename(name: str) -> str:
    """Make a column name safe to use inside a file name."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", str(name)).strip(" .")
    return (cleaned or "column")[:100]


def _dataset_subfolder(csv_path: str) -> str:
    """Return a dataset-name subfolder derived from the CSV file name.

    ``House.csv`` -> ``House``. This keeps plots for each dataset in their
    own folder under ``plots/`` so re-runs and different datasets don't
    collide.
    """
    return Path(csv_path).stem


def _save_fig(fig: Figure, filename: str, subfolder: str | None = None) -> str:
    """Save a figure under ``plots/`` (optionally inside a dataset subfolder).

    Returns the saved path.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = PLOTS_DIR / subfolder if subfolder else PLOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    plot_path = target_dir / filename
    fig.savefig(plot_path, bbox_inches="tight", dpi=100)
    return str(plot_path)


@lru_cache(maxsize=2)
def _read_cached(csv_path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    # mtime_ns and size are cache keys, so an edited file is re-read.
    return read_csv_robust(csv_path)


def _load_csv(csv_path: str) -> pd.DataFrame:
    """Read the CSV once per file version. The returned frame is read-only."""
    try:
        stat = Path(csv_path).stat()
    except OSError:
        return read_csv_robust(csv_path)  # let V1 raise its usual error
    return _read_cached(str(csv_path), stat.st_mtime_ns, stat.st_size)


def _get_column(
    tool_name: str, csv_path: str, column: str
) -> tuple[pd.Series | None, str | None]:
    """Return ``(series, None)`` or ``(None, result_json)`` when the tool must stop."""
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        return None, _result(
            tool_name,
            None,
            "error",
            [column],
            f"Failed to read CSV: {type(exc).__name__}: {exc}",
        )

    if column not in df.columns:
        return None, _result(
            tool_name,
            None,
            "skipped",
            [column],
            f"Column '{column}' not found in dataset.",
        )

    series = df[column]
    if isinstance(series, pd.DataFrame):  # duplicate column names
        series = series.iloc[:, 0]
    return series, None


def _numeric_values(series: pd.Series) -> np.ndarray | None:
    """Return finite numeric values, or None if the column is not numeric."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        numbers = pd.to_numeric(series, errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(series):
        return None
    else:
        non_null = series.dropna()
        if non_null.empty:
            return None
        numbers = pd.to_numeric(series, errors="coerce")
        if numbers.notna().sum() < _MIN_NUMERIC_SHARE * len(non_null):
            return None
    values = numbers.astype("float64").to_numpy()
    values = values[np.isfinite(values)]
    return values if values.size else None


def _hist_bins(values: np.ndarray) -> int:
    """Bin count that matches numpy's ``bins="auto"`` rule, capped at ``_MAX_HIST_BINS``.

    The rule is written out here so it behaves the same on every numpy
    version. Older numpy versions could ask for millions of bins when a column
    has a tiny IQR and a huge range (for example money amounts that are mostly
    zero).
    """
    n = values.size
    low, high = float(values.min()), float(values.max())
    span = high - low
    if span == 0:
        return 1
    q75, q25 = np.percentile(values, [75, 25])
    fd_width = 2.0 * (q75 - q25) * n ** (-1.0 / 3.0)
    sqrt_width = span / np.sqrt(n)
    sturges_width = span / (np.log2(n) + 1.0)
    width = min(max(fd_width, sqrt_width / 2.0), sturges_width)
    return int(min(np.ceil(span / width), _MAX_HIST_BINS))


def _render(
    tool_name: str,
    column: str,
    prefix: str,
    evidence: str,
    draw: Callable[[Any], None],
    subfolder: str | None = None,
) -> str:
    """Create a figure, draw on it, save it, and return the JSON result."""
    try:
        fig = Figure()
        draw(fig.subplots())
        plot_file = _save_fig(
            fig, f"{prefix}_{_safe_filename(column)}.png", subfolder=subfolder
        )
    except Exception as exc:
        return _result(
            tool_name,
            None,
            "error",
            [column],
            f"Failed to plot: {type(exc).__name__}: {exc}",
        )
    return _result(tool_name, plot_file, "ok", [column], evidence)


def _numeric_or_stop(
    tool_name: str, csv_path: str, column: str
) -> tuple[np.ndarray | None, str | None]:
    """Load a column as numbers, or return the JSON result that ends the tool."""
    series, stop = _get_column(tool_name, csv_path, column)
    if stop is not None:
        return None, stop
    values = _numeric_values(series)
    if values is None:
        return None, _result(
            tool_name,
            None,
            "skipped",
            [column],
            f"Column '{column}' has no numeric values to plot.",
        )
    return values, None


def _semantic_or_stop(
    tool_name: str, csv_path: str, column: str, accepted: frozenset[str]
) -> tuple[bool, str | None]:
    """Return ``(True, None)`` if the column is semantically eligible for
    the tool, or ``(False, result_json)`` if it should be skipped.

    Uses the profile's semantic classification so that skewness/outlier
    plots are only drawn on ``numeric_measurement`` / ``numeric_discrete``
    columns — never on binary flags, integer-coded categoricals, IDs, or
    constants.
    """
    try:
        profile = profile_dataset(csv_path)
    except Exception as exc:  # noqa: BLE001 — surface profile errors cleanly
        return False, _result(
            tool_name,
            None,
            "error",
            [column],
            f"Failed to profile CSV: {type(exc).__name__}: {exc}",
        )
    lookup = get_semantic_lookup(profile)
    sem = lookup.get(column, {}) or {}
    col_type = sem.get("type", "unknown")
    ambiguous = sem.get("ambiguous", True)
    if col_type not in accepted or ambiguous:
        return False, _result(
            tool_name,
            None,
            "skipped",
            [column],
            f"Column '{column}' is classified as '{col_type}' "
            f"({'ambiguous' if ambiguous else 'unambiguous'}); "
            f"{tool_name} only applies to "
            f"{', '.join(sorted(accepted))} columns.",
        )
    return True, None


# ---------------------------------------------------------------------- tools

def _plot_outliers(csv_path: str, column: str) -> str:
    """Draw a boxplot for ``column`` and save it as a PNG.

    Only runs on ``numeric_measurement`` / ``numeric_discrete`` columns;
    binary flags, integer-coded categoricals, IDs, and constants are
    skipped because outliers are meaningless on them.
    """
    ok, stop = _semantic_or_stop(
        "run_plot_outliers",
        csv_path,
        column,
        frozenset({"numeric_measurement", "numeric_discrete"}),
    )
    if not ok:
        return stop
    values, stop = _numeric_or_stop("run_plot_outliers", csv_path, column)
    if stop is not None:
        return stop

    subfolder = _dataset_subfolder(csv_path)

    def draw(ax: Any) -> None:
        ax.boxplot(values)
        ax.set_title(f"Boxplot of {column}")
        ax.set_ylabel(column)

    return _render(
        "run_plot_outliers",
        column,
        "outliers",
        f"Boxplot saved for column '{column}'.",
        draw,
        subfolder=subfolder,
    )


def _plot_skewness(csv_path: str, column: str) -> str:
    """Draw a histogram for ``column`` and save it as a PNG.

    Only runs on ``numeric_measurement`` / ``numeric_discrete`` columns;
    binary flags, integer-coded categoricals, IDs, and constants are
    skipped because skewness is meaningless on them.
    """
    ok, stop = _semantic_or_stop(
        "run_plot_skewness",
        csv_path,
        column,
        frozenset({"numeric_measurement", "numeric_discrete"}),
    )
    if not ok:
        return stop
    values, stop = _numeric_or_stop("run_plot_skewness", csv_path, column)
    if stop is not None:
        return stop

    subfolder = _dataset_subfolder(csv_path)

    def draw(ax: Any) -> None:
        ax.hist(values, bins=_hist_bins(values))
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    return _render(
        "run_plot_skewness",
        column,
        "skewness",
        f"Histogram saved for column '{column}'.",
        draw,
        subfolder=subfolder,
    )


def _plot_imbalance(csv_path: str, target_column: str) -> str:
    """Draw a bar chart of class distribution for ``target_column``.

    Only runs for binary/multiclass targets. Continuous targets are skipped
    because class imbalance is meaningless for regression.
    """
    tool_name = "run_plot_imbalance"
    series, stop = _get_column(tool_name, csv_path, target_column)
    if stop is not None:
        return stop

    # Skip continuous targets — imbalance check is meaningless for regression.
    target_type = detect_target_type(series)
    if target_type == "continuous":
        return _result(
            tool_name,
            None,
            "skipped",
            [target_column],
            f"Column '{target_column}' is continuous; imbalance check is not "
            f"applicable to regression targets.",
        )

    counts = series.value_counts()
    if counts.empty:
        return _result(
            tool_name,
            None,
            "skipped",
            [target_column],
            f"Column '{target_column}' has no values to plot.",
        )

    truncated = len(counts) > _MAX_CLASSES
    if truncated:
        counts = counts.nlargest(_MAX_CLASSES)
    try:
        counts = counts.sort_index()
    except TypeError:  # mixed types cannot be ordered
        counts = counts.sort_values(ascending=False)

    labels = [str(label) for label in counts.index]
    title = f"Class Distribution of {target_column}"
    if truncated:
        title += f" (top {_MAX_CLASSES})"

    subfolder = _dataset_subfolder(csv_path)

    def draw(ax: Any) -> None:
        ax.bar(range(len(labels)), counts.to_numpy(), width=0.5)
        ax.set_xticks(range(len(labels)))
        if len(labels) > 6 or max(len(s) for s in labels) > 8:
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel(target_column)
        ax.set_ylabel("Count")

    return _render(
        tool_name,
        target_column,
        "imbalance",
        f"Class distribution saved for target '{target_column}'.",
        draw,
        subfolder=subfolder,
    )


def _plot_correlation(csv_path: str) -> str:
    """Draw a heatmap of the correlation matrix for all numeric columns."""
    tool_name = "run_plot_correlation"
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        return _result(
            tool_name,
            None,
            "error",
            [],
            f"Failed to read CSV: {type(exc).__name__}: {exc}",
        )

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return _result(
            tool_name,
            None,
            "skipped",
            [],
            "Need at least two numeric columns for a correlation matrix.",
        )

    corr = numeric_df.corr()
    columns = list(corr.columns)
    values = corr.to_numpy()

    n = len(columns)
    fig = Figure(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    ax = fig.subplots()
    im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()

    subfolder = _dataset_subfolder(csv_path)
    plot_file = _save_fig(fig, "correlation_matrix.png", subfolder=subfolder)
    return _result(
        tool_name,
        plot_file,
        "ok",
        columns,
        f"Correlation matrix saved for {n} numeric column(s).",
    )


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

run_plot_correlation = _make_tool(
    name="run_plot_correlation",
    description=(
        "Draw a heatmap of the full correlation matrix for all numeric "
        "columns (each column's correlation with every other column). "
        "Takes csv_path only. Returns the saved PNG path as JSON."
    ),
    func=_plot_correlation,
)

ALL_TOOLS = [run_plot_outliers, run_plot_skewness, run_plot_imbalance, run_plot_correlation]


__all__ = [
    "run_plot_outliers",
    "run_plot_skewness",
    "run_plot_imbalance",
    "run_plot_correlation",
    "ALL_TOOLS",
    "PLOTS_DIR",
]