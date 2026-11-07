"""LangGraph state for the Visualizer Workbench (V2).
 
State holds IDs, file paths, and findings — never a DataFrame and never
image bytes. Tools load the CSV from disk when needed via V1's
``read_csv_robust`` (imported read-only).
"""
 
from typing import Annotated, Optional, TypedDict
 
from langgraph.graph.message import add_messages
 
 
class GraphState(TypedDict, total=False):
 
    messages: Annotated[list, add_messages]
    findings_in: list[dict]
    csv_path: Optional[str]
    target_column: Optional[str]
    plot_paths: list[str]
    want_plots: Optional[bool]
    error: Optional[str]
 