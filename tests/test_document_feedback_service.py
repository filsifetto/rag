"""Tests for the document feedback service (mocked LLM)."""

import pytest
from unittest.mock import Mock

from core.models.search_result import SearchResult, SearchResultType
from core.models.document_feedback import DocumentFeedback
from core.services.document_feedback_service import DocumentFeedbackService


def _make_mock_search_result(
    content: str = "Some source content.",
    title: str = "Test Source",
    author: str = "Author",
    year: int = 2020,
    page_number: int = 5,
) -> SearchResult:
    return SearchResult(
        id="test-id-1",
        content=content,
        metadata={
            "title": title,
            "author": author,
            "year": year,
            "page_number": page_number,
        },
        vector_score=0.9,
        keyword_score=0.1,
        combined_score=0.85,
        explanation="test",
        result_type=SearchResultType.CHUNK,
    )


@pytest.fixture
def mock_settings():
    s = Mock()
    s.openai_api_key = "test-key"
    s.openai_model = "gpt-4"
    s.max_response_tokens = 1000
    s.max_sources_per_response = 5
    return s


class TestDocumentFeedbackServiceEmptyInputs:
    """Behaviour when document or results are empty."""

    @pytest.mark.asyncio
    async def test_empty_document_returns_minimal_feedback(self, mock_settings):
        svc = DocumentFeedbackService(mock_settings)
        result = await svc.generate_feedback("", [_make_mock_search_result()])
        assert isinstance(result, DocumentFeedback)
        assert "No document text" in result.feedback_summary or "no document" in result.feedback_summary.lower()
        assert result.suggested_references == []

    @pytest.mark.asyncio
    async def test_whitespace_document_returns_minimal_feedback(self, mock_settings):
        svc = DocumentFeedbackService(mock_settings)
        result = await svc.generate_feedback("   \n\n  ", [])
        assert isinstance(result, DocumentFeedback)
        assert result.suggested_references == []

    @pytest.mark.asyncio
    async def test_no_search_results_returns_message(self, mock_settings):
        svc = DocumentFeedbackService(mock_settings)
        result = await svc.generate_feedback("Real document text here.", [])
        assert isinstance(result, DocumentFeedback)
        assert "no reference" in result.feedback_summary.lower() or "No reference" in result.feedback_summary


class TestDocumentFeedbackServiceContextBuilding:
    """Context built from search results includes source content and citation info."""

    def test_build_sources_context_includes_metadata_and_content(self, mock_settings):
        svc = DocumentFeedbackService(mock_settings)
        ctx = svc._build_sources_context([_make_mock_search_result()])
        assert "Test Source" in ctx
        assert "Author" in ctx
        assert "Some source content" in ctx
        assert "2020" in ctx
        assert "page_number" in ctx or "5" in ctx
