"""
Tests for document feedback Pydantic models.
"""

import pytest
from pydantic import ValidationError

from core.models.document_feedback import (
    SuggestedReference,
    SuggestedEdit,
    DocumentFeedback,
)


class TestSuggestedReference:
    """Tests for SuggestedReference."""

    def test_valid_instance(self):
        ref = SuggestedReference(
            place_in_document="In the introduction paragraph.",
            citation_apa="(Smith, 2020, p. 12)",
            source_snippet="Customer value is defined as...",
            reason="Supports the claim about value.",
            full_reference_with_page="Smith, J. (2020). Title. Publisher, p. 12.",
        )
        assert ref.place_in_document == "In the introduction paragraph."
        assert ref.citation_apa == "(Smith, 2020, p. 12)"
        assert ref.reason == "Supports the claim about value."
        assert "p. 12" in ref.full_reference_with_page

    def test_empty_place_rejected(self):
        with pytest.raises(ValidationError):
            SuggestedReference(
                place_in_document="",
                citation_apa="(Smith, 2020)",
                source_snippet="Snippet.",
                reason="Reason.",
                full_reference_with_page="Smith (2020). Ref.",
            )

    def test_whitespace_only_place_rejected(self):
        with pytest.raises(ValidationError):
            SuggestedReference(
                place_in_document="   ",
                citation_apa="(Smith, 2020)",
                source_snippet="Snippet.",
                reason="Reason.",
                full_reference_with_page="Smith (2020). Ref.",
            )


class TestSuggestedEdit:
    """Tests for SuggestedEdit."""

    def test_valid_instance(self):
        edit = SuggestedEdit(
            location_or_quote="First paragraph.",
            suggestion="Add a transition sentence.",
            reason="Improves flow.",
        )
        assert edit.reason == "Improves flow."

    def test_reason_optional(self):
        edit = SuggestedEdit(
            location_or_quote="Here",
            suggestion="Change to X",
        )
        assert edit.reason is None


class TestDocumentFeedback:
    """Tests for DocumentFeedback."""

    def test_valid_minimal(self):
        fb = DocumentFeedback(feedback_summary="Overall the document is clear.")
        assert fb.feedback_summary == "Overall the document is clear."
        assert fb.suggested_references == []
        assert fb.suggested_edits == []

    def test_valid_with_references(self):
        ref = SuggestedReference(
            place_in_document="Para 1",
            citation_apa="(Author, 2020)",
            source_snippet="Snippet",
            reason="Fits here.",
            full_reference_with_page="Author, A. (2020). Work. Pub, p. 1.",
        )
        fb = DocumentFeedback(
            feedback_summary="Good draft.",
            suggested_references=[ref],
        )
        assert len(fb.suggested_references) == 1
        assert fb.suggested_references[0].citation_apa == "(Author, 2020)"

    def test_round_trip(self):
        fb = DocumentFeedback(
            feedback_summary="Summary.",
            suggested_references=[
                SuggestedReference(
                    place_in_document="P1",
                    citation_apa="(A, 2020)",
                    source_snippet="Snip",
                    reason="R",
                    full_reference_with_page="A. (2020). T. P., p. 1.",
                )
            ],
            suggested_edits=[
                SuggestedEdit(location_or_quote="Q", suggestion="S", reason="R")
            ],
        )
        d = fb.model_dump()
        fb2 = DocumentFeedback.model_validate(d)
        assert fb2.feedback_summary == fb.feedback_summary
        assert len(fb2.suggested_references) == 1
        assert len(fb2.suggested_edits) == 1
