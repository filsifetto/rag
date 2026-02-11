#!/usr/bin/env python3
"""
Document feedback CLI: upload a file, get LLM feedback and reference suggestions.

Reads the file (no ingestion), queries the subject's RAG collection for
relevant material, and returns structured feedback with suggested places
to add citations from the subject material.
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import Settings, apply_subject
from core.database.qdrant_client import QdrantManager
from core.loaders import load_user_document
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine
from core.services.document_query_strategy import document_to_search_queries
from core.services.document_feedback_service import DocumentFeedbackService
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text


def _dedupe_results(results: list) -> list:
    """Deduplicate by result id, preserving order."""
    seen = set()
    out = []
    for r in results:
        if r.id not in seen:
            seen.add(r.id)
            out.append(r)
    return out


async def run_feedback(
    file_path: Path,
    subject: str,
    max_sources: int = 10,
    search_limit: int = 15,
) -> None:
    """Load document, search subject collection, generate and print feedback."""
    console = Console()
    settings = apply_subject(Settings(), subject)

    console.print(f"[bold]Loading document:[/bold] {file_path}")
    try:
        doc = load_user_document(file_path)
    except FileNotFoundError:
        console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(f"[dim]Subject collection: {settings.qdrant_collection_name}[/dim]\n")

    qdrant = QdrantManager(settings)
    embedder = EmbeddingService(settings)
    search_engine = HybridSearchEngine(qdrant, embedder, settings)
    feedback_service = DocumentFeedbackService(settings)

    queries = document_to_search_queries(doc.content)
    if not queries:
        console.print("[yellow]Document has no text to search.[/yellow]")
        feedback = await feedback_service.generate_feedback(doc.content, [])
    else:
        console.print(f"[dim]Document split into {len(queries)} search query window(s) (sliding window, full coverage).[/dim]")
        all_results = []
        for q in queries:
            results = await search_engine.search(q, limit=search_limit)
            all_results.extend(results)
        all_results = _dedupe_results(all_results)
        all_results = sorted(all_results, key=lambda x: x.combined_score, reverse=True)[:max_sources]
        console.print(f"[dim]Retrieved {len(all_results)} source chunks.[/dim]\n")
        feedback = await feedback_service.generate_feedback(doc.content, all_results)

    console.print(Panel(Markdown(feedback.feedback_summary), title="Feedback", border_style="blue"))

    # --- Proposed changes with yellow-highlight markers (for easy double-check) ---
    if feedback.suggested_references:
        console.print()
        console.print("[bold]Suggested reference insertions[/bold] (marked text = where to add the citation):")
        for i, ref in enumerate(feedback.suggested_references, 1):
            # Show the excerpt from the document with yellow background so the user sees exactly where to add the citation
            console.print(f"  [bold]{i}.[/bold] In your text, add the citation after this sentence/excerpt:")
            console.print(Text(ref.place_in_document, style="on yellow black"))
            console.print(f"  [green]→ Insert in-text citation:[/green] {ref.citation_apa}")
            console.print(f"  [dim]Reason:[/dim] {ref.reason}")
            console.print(Panel(
                ref.source_snippet[:500] + ("..." if len(ref.source_snippet) > 500 else ""),
                title="Supporting excerpt from source",
                border_style="dim",
            ))
            console.print()

    if feedback.suggested_edits:
        console.print("[bold]Suggested edits[/bold] (marked text = location of the change):")
        for i, edit in enumerate(feedback.suggested_edits, 1):
            console.print(f"  [bold]{i}.[/bold] Location in your text:")
            console.print(Text(edit.location_or_quote[:300] + ("..." if len(edit.location_or_quote) > 300 else ""), style="on yellow black"))
            console.print(f"  [green]→ Suggested change:[/green] {edit.suggestion[:300]}{'...' if len(edit.suggestion) > 300 else ''}")
            if edit.reason:
                console.print(f"  [dim]Reason:[/dim] {edit.reason}")
            console.print()

    # --- References at the bottom (with page numbers) so the user can look up and double-check ---
    if feedback.suggested_references:
        console.print(Panel(
            "\n".join(ref.full_reference_with_page for ref in feedback.suggested_references),
            title="References (with page numbers) — use these to look up and double-check the sources",
            border_style="green",
        ))

    if not feedback.suggested_references and not feedback.suggested_edits and "failed" not in feedback.feedback_summary.lower():
        console.print("[dim]No specific reference or edit suggestions.[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get feedback and reference suggestions for a document against a subject's RAG collection.",
    )
    parser.add_argument("--file", "-f", type=Path, required=True, help="Path to the document (.txt, .md, .pdf)")
    parser.add_argument("--subject", "-s", type=str, required=True, help="Subject id (e.g. TIØ4165) — uses that collection")
    parser.add_argument("--max-sources", type=int, default=10, help="Max source chunks to pass to the LLM (default 10)")
    parser.add_argument("--search-limit", type=int, default=15, help="Search results per query (default 15)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    asyncio.run(run_feedback(args.file, args.subject, args.max_sources, args.search_limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
