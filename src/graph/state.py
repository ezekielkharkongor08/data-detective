"""LangGraph state for the Data Detective audit graph.

State holds IDs, file paths, and findings — never a DataFrame.
Tools load the CSV from disk when needed.
"""

from operator import add
from typing import Annotated, Optional, TypedDict

from src.schemas.finding import Finding


class GraphState(TypedDict, total=False):
    """State carried through the audit graph.

    All fields are optional so a node can return only the fields it updates.
    """

    dataset_id: str
    csv_path: str
    target_column: Optional[str]
    profile: Optional[dict]

    # Merged findings from all parallel checks. ``operator.add`` is the
    # reducer: LangGraph concatenates lists returned by parallel branches.
    findings: Annotated[list[Finding], add]

    # Set by the router when a finding suggests potential leakage.
    leakage_suspected: Optional[bool]
    # Suspect column names passed from router to deep_checks.
    suspect_columns: Optional[list[str]]
    # Results from deep leakage check(s).
    deep_check_results: Optional[list[dict]]

    #LLM-produced plain-language summary and overall severity.
    summary: Optional[str]
    overall_severity: Optional[str]

    # JSON report (single source of truth, serialized AuditReport).
    report: Optional[str]
    # Plain-text report rendered from the JSON report object.
    report_text: Optional[str]
    # Error message if a node failed.
    error: Optional[str]


__all__ = ["GraphState"]
