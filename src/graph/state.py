"""LangGraph state for the Data Detective audit graph.

State holds IDs, file paths, and findings — never a DataFrame.
Tools load the CSV from disk when needed.
"""

from typing import Annotated, Optional, TypedDict

from src.graph.reducers import merge_findings


class GraphState(TypedDict, total=False):
    """State carried through the audit graph.

    All fields are optional so a node returns only what it updates.
    ``findings`` is a list of plain dicts (from ``Finding.model_dump``) so the
    checkpointer can serialize them without a custom Pydantic serializer.
    """

    dataset_id: str
    csv_path: str
    target_column: Optional[str]
    profile: Optional[dict]

    # Per-check parameters keyed by check name, for example
    # check_params["bias"] = {"sensitive_columns": ["sex"]}.
    # Checks declare the keys they read, so adding a check needs no state changes.
    check_params: dict[str, dict]

    # {column_name: semantic_type} from --semantic-override; applied in
    # profile_node. Rule 0: overrides win over every classifier rule.
    semantic_overrides: dict[str, str]

    # Reducer concatenates findings from parallel branches and drops dupes.
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
    # Plain-text rendering of the JSON report.
    report_text: Optional[str]
    # Set by a node when it fails.
    error: Optional[str]


__all__ = ["GraphState"]
