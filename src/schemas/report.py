"""Structured output schemas for the agent (LLM structured output) and the audit report.

The LLM is forced to return ``SynthesisResult`` instead of free text.
``AuditReport`` is the single source of truth for the final report: it is
serialized to JSON (``state["report"]``) and rendered to plain text
(:func:`src.graph.nodes.render_text_report`, stored in ``state["report_text"]``).
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.finding import Finding


class SynthesisResult(BaseModel):
    """Detailed plain-language overview produced by the LLM from the findings.

    Only the narrative lives here. The overall severity is computed in Python
    from the findings.
    """

    summary: str = Field(
        ...,
        description=(
            "The complete summary text using EXACTLY this structure with all-caps "
            "section headers:\n\n"
            "OVERVIEW:\n"
            "<3-4 sentences describing the dataset shape (rows/columns), what the "
            "data appears to be about using real column names, and its overall health.\n\n"
            "KEY ISSUES & NEXT STEPS:\n"
            "- [SEVERITY] <Merged problem description with key numbers and column names>. "
            "Next step: <Your own dynamic, specific advice based on those exact columns and numbers.>\n"
            "<Additional bullets as needed, most severe first. Strictly merge ALL findings "
            "of the same check type into ONE bullet.>\n\n"
            "Cover: (1) what the dataset is and its shape; (2) every problem found, why it "
            "matters, and which columns it hits; (3) concrete, dynamic next steps. Only "
            "describe what the profile and findings actually show. Never return only a "
            "heading or a title."
        ),
    )


class AuditReport(BaseModel):
    """Complete audit report as structured data.

    JSON (via :meth:`to_json`) is the canonical output. Plain text is derived
    from the same object, so both views always stay in sync.
    """

    dataset_id: str
    rows: int
    cols: int
    duplicate_rows: int | None = None
    summary: Optional[str] = None
    overall_severity: Optional[str] = None
    findings: list[Finding] = Field(default_factory=list)
    columns: list[dict] = Field(default_factory=list)
    sample_rows: list[dict] = Field(default_factory=list)
    deep_check_results: Optional[list[dict]] = None
    error: Optional[str] = None
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON — the canonical report output."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-compatible)."""
        return self.model_dump(mode="json")


__all__ = ["SynthesisResult", "AuditReport"]
