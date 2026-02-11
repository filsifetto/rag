"""
Document feedback service: LLM-produced feedback and reference suggestions.

Takes the user's document text and retrieved subject-matter excerpts (search results),
and returns structured feedback with suggested places to add citations.
Does not perform search or file I/O; callers provide document text and results.
"""

from typing import List, Optional
import logging
import time
import instructor
from openai import OpenAI

from ..models.search_result import SearchResult
from ..models.document_feedback import DocumentFeedback
from ..citation import format_apa_inline, format_apa_reference, build_citation_key
from ..config import Settings

# Max characters of source content per result (to fit in prompt).
MAX_CONTENT_CHARS_PER_SOURCE = 1500


class DocumentFeedbackService:
    """Generate feedback and reference suggestions from document text and search results."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = instructor.patch(OpenAI(api_key=settings.openai_api_key))
        self.model = settings.openai_model
        self.max_tokens = settings.max_response_tokens
        self.temperature = 0.2
        self.logger = logging.getLogger(__name__)

    def _build_sources_context(self, search_results: List[SearchResult]) -> str:
        """Build a string of source excerpts with citation info for the LLM."""
        parts = []
        for i, result in enumerate(search_results[: self.settings.max_sources_per_response], 1):
            meta = result.metadata
            key = build_citation_key(meta)
            inline = format_apa_inline(meta)
            content = result.content
            if len(content) > MAX_CONTENT_CHARS_PER_SOURCE:
                content = content[:MAX_CONTENT_CHARS_PER_SOURCE] + "... [truncated]"
            meta_str = ", ".join(
                f"{k}: {v}" for k, v in meta.items()
                if k in ("title", "author", "year", "page_number", "chapter") and v is not None
            )
            parts.append(
                f"Source [{key}] {inline}\n"
                f"Metadata: {meta_str or 'None'}\n"
                f"Content: {content}\n---"
            )
        return "\n\n".join(parts)

    def _system_prompt(self, custom_instructions: Optional[str] = None) -> str:
        base = """You are an academic writing advisor. You have been given:
1. A document (e.g. essay or report) written by a student or author.
2. Excerpts from reference material (course readings, papers, books) that are relevant to the document.

Your task:
- Provide concise, constructive feedback on the document (clarity, argument strength, gaps).
- Suggest specific places in the document where the author should add citations from the reference material. For each suggestion provide:
  - place_in_document: the exact sentence or short quote where the citation should be added (this will be shown with a yellow marker so the user sees where to add it).
  - citation_apa: the APA 7 in-text citation to insert there (e.g. (Author, Year, p. X)).
  - source_snippet: the relevant excerpt from the reference that supports this citation.
  - reason: why this citation fits at this place.
  - full_reference_with_page: the full APA 7 reference for the references section at the bottom of the output, including page number (e.g. "Author, A. A. (Year). Title. Publisher, p. X." or journal format with pages). The user will use this to look up and double-check the source.
- Optionally suggest_edits: small improvements (rephrasing, adding a transition) with location_or_quote (the text to change, shown with a marker), suggestion, and optional reason.

Use only the provided reference excerpts; do not invent sources. If the document does not need more citations, return an empty suggested_references list. Always include full_reference_with_page for every suggested reference so the user can verify the source."""
        if custom_instructions:
            base += f"\n\nAdditional instructions:\n{custom_instructions}"
        return base

    def _user_prompt(self, document_text: str, sources_context: str) -> str:
        return f"""AUTHOR'S DOCUMENT:
{document_text}

---
REFERENCE MATERIAL (excerpts from subject/course sources):
{sources_context}

---
Provide your feedback_summary, suggested_references (each with place_in_document, citation_apa, source_snippet, reason, and full_reference_with_page for the bottom references list), and optionally suggested_edits."""

    async def generate_feedback(
        self,
        document_text: str,
        search_results: List[SearchResult],
        custom_instructions: Optional[str] = None,
    ) -> DocumentFeedback:
        """Generate structured feedback and reference suggestions.

        Args:
            document_text: The full text of the user's document.
            search_results: Retrieved subject-matter chunks (e.g. from hybrid search).
            custom_instructions: Optional extra instructions for the LLM.

        Returns:
            DocumentFeedback with feedback_summary, suggested_references, suggested_edits.
        """
        start = time.time()
        if not document_text or not document_text.strip():
            self.logger.warning("Empty document text; returning minimal feedback.")
            return DocumentFeedback(
                feedback_summary="No document text was provided.",
                suggested_references=[],
                suggested_edits=[],
            )
        if not search_results:
            self.logger.warning("No search results; feedback will not cite sources.")
            return DocumentFeedback(
                feedback_summary="No reference material was retrieved for this document. You may still add general feedback after re-running with a subject that has ingested material.",
                suggested_references=[],
                suggested_edits=[],
            )

        sources_context = self._build_sources_context(search_results)
        system = self._system_prompt(custom_instructions)
        user = self._user_prompt(document_text, sources_context)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_model=DocumentFeedback,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self.logger.info(
                f"Document feedback generated in {time.time() - start:.3f}s, "
                f"suggested_references={len(response.suggested_references)}"
            )
            return response
        except Exception as e:
            self.logger.error(f"Document feedback generation failed: {e}")
            return DocumentFeedback(
                feedback_summary=f"Feedback generation failed: {e}",
                suggested_references=[],
                suggested_edits=[],
            )
