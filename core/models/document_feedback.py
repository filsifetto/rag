"""
Pydantic models for document feedback and reference suggestions.

Used by the document feedback service and by the CLI when displaying
or exporting feedback.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SuggestedReference(BaseModel):
    """A single suggestion for where to add a citation in the user's document."""

    place_in_document: str = Field(
        ...,
        description="Quote or sentence where the citation should be added.",
    )
    citation_apa: str = Field(
        ...,
        description="APA-style citation (inline or full) to insert.",
    )
    source_snippet: str = Field(
        ...,
        description="Excerpt from the subject material that supports this citation.",
    )
    reason: str = Field(
        ...,
        description="Why this citation fits at this place.",
    )
    full_reference_with_page: str = Field(
        ...,
        description="Full APA 7 reference for the references section at the bottom, including page number, so the user can look up and double-check the source.",
    )

    @field_validator("place_in_document")
    @classmethod
    def place_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("place_in_document must be non-empty")
        return v


class SuggestedEdit(BaseModel):
    """An optional suggested text change (e.g. rephrase, add clause)."""

    location_or_quote: str = Field(
        ...,
        description="Where in the document (quote or description).",
    )
    suggestion: str = Field(
        ...,
        description="Suggested replacement or addition.",
    )
    reason: Optional[str] = Field(None, description="Why this change helps.")


class DocumentFeedback(BaseModel):
    """Structured feedback on a user document with reference suggestions."""

    feedback_summary: str = Field(
        ...,
        description="Overall feedback on the document (quality, clarity, gaps).",
    )
    suggested_references: List[SuggestedReference] = Field(
        default_factory=list,
        description="Specific places to add citations with APA reference and snippet.",
    )
    suggested_edits: List[SuggestedEdit] = Field(
        default_factory=list,
        description="Optional suggested text changes.",
    )
