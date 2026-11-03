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
            "The FULL detailed overview, at least 150 words, as 3-5 short "
            "paragraphs or bullet lists. Never return only a heading or a title. "
            "Cover: (1) what the dataset appears to be and its shape "
            "(rows/columns); (2) what the column groups contain, using their "
            "real names; (3) every problem found, why it matters, and which "
            "columns it hits; (4) concrete next steps. Only describe what the "
            "profile and findings actually show."
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
