"""
Intelligent response generation with multi-provider support and quality analysis.

This module provides advanced response generation capabilities with structured output,
quality assessment, and support for multiple LLM providers.
"""

from typing import List, Dict, Any, Optional
import instructor
from openai import OpenAI
import logging
import time
from datetime import datetime

from ..models.search_result import SearchResult, ResponseAnalysis
from ..citation import format_apa_inline, format_apa_reference, build_citation_key
from ..config import Settings


class ResponseGenerator:
    """Advanced response generation with multi-provider support and quality analysis."""
    
    def __init__(self, settings: Settings):
        """Initialize response generator."""
        self.settings = settings
        self.openai_client = instructor.patch(OpenAI(api_key=settings.openai_api_key))
        self.logger = logging.getLogger(__name__)
        
        # Response configuration
        self.max_sources = settings.max_sources_per_response
        self.temperature = settings.response_temperature
        self.max_tokens = settings.max_response_tokens
        self.model = settings.openai_model
    
    async def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        max_sources: Optional[int] = None,
        temperature: Optional[float] = None,
        custom_instructions: Optional[str] = None
    ) -> ResponseAnalysis:
        """Generate intelligent response with quality analysis."""
        start_time = time.time()
        
        max_sources = max_sources or self.max_sources
        temperature = temperature or self.temperature
        
        if not search_results:
            return self._create_no_results_response(query, start_time)
        
        # Select top sources based on combined score
        top_sources = sorted(search_results, key=lambda x: x.combined_score, reverse=True)[:max_sources]
        
        self.logger.info(f"Generating response using {len(top_sources)} sources")
        
        # Build context from sources
        context = self._build_context(top_sources)
        
        # Create system prompt with response guidelines
        system_prompt = self._create_system_prompt(custom_instructions)
        
        # Generate structured response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._create_user_prompt(query, context)}
        ]
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=ResponseAnalysis,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            # Enhance response with additional metadata
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            response.query = query
            response.search_results_count = len(search_results)
            response.model_used = self.model
            response.token_count = len(response.answer.split())  # Rough estimate
            
            # Add detailed source information
            response.source_details = [result.get_source_info() for result in top_sources]
            
            self.logger.info(
                f"Response generated in {processing_time:.3f}s, "
                f"confidence: {response.confidence_score:.3f}, "
                f"sources used: {len(response.sources_used)}"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._create_error_response(query, str(e), start_time)
    
    # Maximum characters of content to include per source (~375 tokens)
    MAX_CONTENT_CHARS_PER_SOURCE = 1500

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Build structured context from search results, truncating to fit token budget.

        Each source is labelled with an APA-style citation key so the LLM can
        produce proper in-text citations.
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            meta = result.metadata
            citation_key = build_citation_key(meta)
            inline_cite = format_apa_inline(meta)

            # Build a compact metadata line (include page_number for precise citations)
            metadata_items = []
            for key in ("title", "author", "year", "journal", "publisher", "chapter", "page_number"):
                value = meta.get(key)
                if value is not None:
                    metadata_items.append(f"{key}: {value}")
            metadata_str = ", ".join(metadata_items) if metadata_items else "No metadata"
            
            # Truncate content to stay within token budget
            content = result.content
            if len(content) > self.MAX_CONTENT_CHARS_PER_SOURCE:
                content = content[:self.MAX_CONTENT_CHARS_PER_SOURCE] + "... [truncated]"
            
            context_parts.append(f"""
Source [{citation_key}] {inline_cite}:
Content: {content}
Metadata: {metadata_str}
Relevance Score: {result.combined_score:.3f}
---""")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, custom_instructions: Optional[str] = None) -> str:
        """Create comprehensive system prompt for response generation."""
        base_prompt = """You are an academic research assistant that synthesizes information from provided sources using APA 7th edition referencing.

CITATION RULES:
1. Base your response ONLY on the provided sources.
2. Use APA 7 in-text citations. Each source is labelled with its citation key in square brackets (e.g. [Sommerville, 2015]). Cite like: (Sommerville, 2015) or (Crispin & Gregory, 2008, Ch. 6) or (Author, Year, p. 12).
3. When a source has a chapter, include it: (Author, Year, Ch. X).
4. When a source has a page_number, include it: (Author, Year, p. X) or (Author, Year, Ch. X, p. X).
5. When paraphrasing, place the citation at the end of the relevant sentence or clause.
6. When multiple sources support the same point, combine them: (Author1, Year; Author2, Year).
7. After your answer, include a "References" section with full APA 7 reference entries for every source you cited. Use ONLY the references provided in the source metadata — do not invent references.
8. Provide step-by-step reasoning.
9. Calculate confidence (0-1) based on source quality and consensus.
10. Note any gaps or limitations.

CONFIDENCE LEVELS: Very High (0.9-1.0), High (0.75-0.89), Medium (0.5-0.74), Low (0.25-0.49), Very Low (0.0-0.24).

REFERENCE FORMAT EXAMPLES:
- Book: Author, A. A. (Year). *Title of work* (Xth ed.). Publisher.
- Journal: Author, A. A. (Year). Title of article. *Journal Name*, *Volume*(Issue), Pages."""
        
        if custom_instructions:
            base_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}"
        
        return base_prompt
    
    def _build_reference_list(self, search_results: List[SearchResult]) -> str:
        """Build an APA 7 reference list from the search results metadata."""
        seen: set = set()
        refs: list = []
        for result in search_results:
            ref_entry = format_apa_reference(result.metadata)
            if ref_entry not in seen:
                seen.add(ref_entry)
                refs.append(ref_entry)
        return "\n".join(refs)

    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create structured user prompt with query and context."""
        return f"""QUESTION: {query}

SOURCES:
{context}

Provide a comprehensive answer based on the sources above. Use APA 7 in-text citations and end with a References section."""
    
    def _create_no_results_response(self, query: str, start_time: float) -> ResponseAnalysis:
        """Create response when no search results are available."""
        processing_time = time.time() - start_time
        
        return ResponseAnalysis(
            answer="I apologize, but I couldn't find any relevant information to answer your question. This could be because the query didn't match any documents in the knowledge base, or the search criteria were too restrictive.",
            confidence_score=0.0,
            source_coverage=0.0,
            reasoning_steps=[
                "No search results were returned for the query",
                "Cannot provide an answer without source material",
                "Suggesting query refinement or broader search terms"
            ],
            sources_used=[],
            source_details=[],
            limitations="No relevant sources found in the knowledge base for this query.",
            processing_time=processing_time,
            query=query,
            search_results_count=0,
            model_used=self.model
        )
    
    def _create_error_response(self, query: str, error_message: str, start_time: float) -> ResponseAnalysis:
        """Create response when an error occurs during generation."""
        processing_time = time.time() - start_time
        
        return ResponseAnalysis(
            answer="I encountered an error while generating a response to your question. Please try again or rephrase your query.",
            confidence_score=0.0,
            source_coverage=0.0,
            reasoning_steps=[
                "Error occurred during response generation",
                f"Error details: {error_message}",
                "Unable to complete analysis"
            ],
            sources_used=[],
            source_details=[],
            limitations=f"Response generation failed due to technical error: {error_message}",
            processing_time=processing_time,
            query=query,
            search_results_count=0,
            model_used=self.model
        )
    
    def generate_summary(self, search_results: List[SearchResult], max_length: int = 200) -> str:
        """Generate a brief summary of search results."""
        if not search_results:
            return "No results found."
        
        # Extract key information
        total_results = len(search_results)
        avg_score = sum(r.combined_score for r in search_results) / total_results
        top_score = max(r.combined_score for r in search_results)
        
        # Get source types
        source_types = set(r.result_type for r in search_results)
        
        # Create summary
        summary = f"Found {total_results} results with average relevance of {avg_score:.2f}. "
        summary += f"Top result scored {top_score:.2f}. "
        summary += f"Results include: {', '.join(source_types)}."
        
        return summary[:max_length]
    
    def validate_response_quality(self, response: ResponseAnalysis) -> Dict[str, Any]:
        """Validate and assess response quality."""
        quality_metrics = {
            "overall_quality": "good",
            "issues": [],
            "recommendations": []
        }
        
        # Check confidence level
        if response.confidence_score < 0.3:
            quality_metrics["issues"].append("Low confidence score")
            quality_metrics["recommendations"].append("Consider gathering more sources")
        
        # Check source usage
        if response.source_coverage < 0.5:
            quality_metrics["issues"].append("Low source coverage")
            quality_metrics["recommendations"].append("Utilize more available sources")
        
        # Check answer length
        if len(response.answer.split()) < 20:
            quality_metrics["issues"].append("Answer may be too brief")
            quality_metrics["recommendations"].append("Provide more detailed explanation")
        
        # Check reasoning steps
        if len(response.reasoning_steps) < 2:
            quality_metrics["issues"].append("Insufficient reasoning detail")
            quality_metrics["recommendations"].append("Include more reasoning steps")
        
        # Determine overall quality
        if len(quality_metrics["issues"]) == 0:
            quality_metrics["overall_quality"] = "excellent"
        elif len(quality_metrics["issues"]) <= 2:
            quality_metrics["overall_quality"] = "good"
        else:
            quality_metrics["overall_quality"] = "needs_improvement"
        
        return quality_metrics
