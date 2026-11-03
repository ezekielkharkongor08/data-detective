"""Pydantic models for audit findings.

Every check returns a list of ``Finding`` objects. An empty list means
no problem was found.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity of a finding, from least to most concerning."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Finding(BaseModel):
    """A structured, evidence-backed data-quality finding.

    Attributes:
        check_name: Name of the check that produced this finding.
        severity: Severity level (see :class:`Severity`).
        evidence: Human-readable description of the evidence.
        affected_columns: Columns involved in the finding.
        suggested_fix: Recommended action to remediate.
        confidence: Confidence score from 0.0 to 1.0.
    """

    check_name: str = Field(..., description="Name of the check that produced this finding.")
    severity: Severity = Field(..., description="Severity level.")
    evidence: str = Field(..., description="Human-readable description of the evidence.")
    affected_columns: list[str] = Field(
        default_factory=list, description="Columns involved in the finding."
    )
    suggested_fix: Optional[str] = Field(
        default=None, description="Recommended action to remediate."
    )
    confidence: float = Field(
        default=1.0, description="Confidence score from 0.0 to 1.0.",
    )

    def __lt__(self, other: "Finding") -> bool:
        """Sort by severity (highest first)."""
        order = {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFO: 1,
        }
        return order[self.severity] > order[other.severity]


__all__ = ["Finding", "Severity"]
