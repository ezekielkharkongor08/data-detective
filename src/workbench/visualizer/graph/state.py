"""LangGraph state for the Visualizer Workbench (V2).

State holds IDs, file paths, and findings — never a DataFrame and never
image bytes. Tools load the CSV from disk when needed via V1's
``read_csv_robust`` (imported read-only).
"""

from typing import Optional, TypedDict


class GraphState(TypedDict, total=False):
    """State carried through the visualizer graph.

    All fields are optional so a node can return only the fields it updates.

    ``findings_in`` receives V1 findings (from the DB or CLI) so the LLM can
    decide which charts to draw. ``plot_paths`` accumulates the .png file
    paths produced by the plotting tools.
    """

    findings_in: list[dict]
    csv_path: Optional[str]
    target_column: Optional[str]
    plot_paths: list[str]
    error: Optional[str]
