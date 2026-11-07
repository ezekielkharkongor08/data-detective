"""LangGraph state for the Data Detective audit graph.

State holds IDs, file paths, and findings — never a DataFrame.
Tools load the CSV from disk when needed.
"""

from typing import Annotated, Optional, TypedDict

from src.graph.reducers import merge_findings


class GraphState(TypedDict, total=False):
    """State carried through the audit graph.

    All fields are optional so a node can return only the fields it updates.

    ``findings`` is stored as a list of plain dicts (via ``Finding.model_dump``)
    so the LangGraph checkpointer can serialize them to JSON without needing a
    custom serializer for Pydantic models.
    """

    dataset_id: str
    csv_path: str
    target_column: Optional[str]
    profile: Optional[dict]

    # Per-check parameters, keyed by check name. Example:
    #   check_params["bias"] = {"sensitive_columns": ["sex"], "favorable_value": ">50K"}
    # Checks declare which keys they read; the registry only passes the ones
    # declared, so adding a check needs no state-field changes.
    check_params: dict[str, dict]

    # Semantic type overrides from --semantic-override CLI flag:
    #   {column_name: semantic_type}. Applied by classify_columns in profile_node.
    #   Rule 0: overrides win over every classifier rule.
    semantic_overrides: dict[str, str]

    # Merged findings from all parallel checks. ``merge_findings`` is the
    # reducer: it concatenates lists returned by parallel branches and drops
    # duplicates, so a repeated finding can never appear twice in the report.
    findings: Annotated[list[dict], merge_findings]

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
