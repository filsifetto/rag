#!/usr/bin/env python3
"""
Full document ingestion pipeline for QdrantRAG-Pro.

Handles:
  - Unzipping archives (.zip)
  - Extracting text from PDFs (.pdf) via PyMuPDF
  - Loading JSON document files (.json)
  - Loading plain text/markdown (.txt, .md)
  - Chunking, embedding, and upserting into Qdrant
"""

import sys
import os
import json
import asyncio
import logging
import zipfile
import tempfile
import shutil
import re
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fitz  # PyMuPDF

from core.config import Settings, apply_subject, subject_documents_dir
from core.database.qdrant_client import QdrantManager
from core.database.document_store import DocumentStore
from core.services.embedding_service import EmbeddingService
from core.models.document import Document, DocumentMetadata, DocumentType
from core.parsers.pdf import PDFMetadataExtractor
from core.citation import enrich_metadata
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel


console = Console()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

# Shared extractor instance
_pdf_extractor = PDFMetadataExtractor()


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, int]:
    """Extract text from a PDF file using PyMuPDF. Returns (text, page_count)."""
    text, meta = _pdf_extractor.extract_text_and_metadata(pdf_path)
    return text, meta.page_count


def extract_text_and_metadata_from_pdf(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Extract text and full metadata from a PDF file.

    Returns (text, metadata_dict) where metadata_dict is compatible with
    DocumentMetadata fields.
    """
    text, meta = _pdf_extractor.extract_text_and_metadata(pdf_path)
    return text, meta.to_document_metadata_dict(source=pdf_path.name)


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------

def unzip_archives(documents_dir: Path, extract_dir: Path) -> List[Path]:
    """Unzip all .zip files into extract_dir, return list of extracted files."""
    extracted_files = []
    zip_files = list(documents_dir.glob("*.zip"))

    for zf in zip_files:
        console.print(f"  📦 Unzipping: [cyan]{zf.name}[/cyan]")
        with zipfile.ZipFile(zf, 'r') as z:
            for member in z.namelist():
                # Skip macOS resource forks and directories
                if member.startswith("__MACOSX") or member.endswith("/"):
                    continue
                # Extract to flat directory to avoid nested paths
                filename = Path(member).name
                if not filename:
                    continue
                target = extract_dir / filename
                with z.open(member) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                extracted_files.append(target)
                console.print(f"    └─ {filename}")

    return extracted_files


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------

def discover_documents(documents_dir: Path, extra_files: List[Path] = None) -> List[Dict[str, Any]]:
    """
    Discover all ingestible documents:
      - PDFs (top-level + extracted from zips)
      - JSON document files
      - Plain text / markdown
    Returns a list of raw document dicts ready for ingestion.
    """
    raw_docs: List[Dict[str, Any]] = []
    all_files: List[Path] = []

    # Collect top-level files (non-zip) and recurse into subdirectories
    for f in documents_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() != ".zip":
            # Skip files inside the _extracted directory (handled separately)
            try:
                f.relative_to(documents_dir / "_extracted")
                continue
            except ValueError:
                pass
            all_files.append(f)

    # Add extracted files
    if extra_files:
        all_files.extend(extra_files)

    # Build set of text file stems so we can skip duplicate PDFs
    txt_stems = {f.stem for f in all_files if f.suffix.lower() in (".txt", ".md")}

    for file_path in sorted(all_files):
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            # Skip PDFs that have a matching .txt file (text is cleaner for embeddings)
            if file_path.stem in txt_stems:
                console.print(f"  ⏭️  Skipping PDF (text version exists): {file_path.name}")
                continue
            try:
                text, pdf_meta = extract_text_and_metadata_from_pdf(file_path)
                if not text or len(text) < 50:
                    console.print(f"  ⚠️  Skipping (too little text): {file_path.name}")
                    continue

                # Start from the extracted PDF metadata, then layer on
                # heuristic values for category / tags / language.
                meta = dict(pdf_meta)
                meta.setdefault("title", file_path.stem)
                meta["category"] = _guess_category(file_path.name)
                meta["language"] = "en"

                # Enrich with citation registry data (APA metadata)
                meta = enrich_metadata(meta, file_path.name)

                # Merge heuristic tags with any keywords from PDF metadata
                existing_tags = meta.get("tags", [])
                guessed_tags = _guess_tags(file_path.name)
                meta["tags"] = list(dict.fromkeys(existing_tags + guessed_tags))

                page_count = meta.get("page_count", 0)
                word_count = meta.get("word_count", 0)

                author_info = f" by {meta['author']}" if meta.get("author") else ""
                raw_docs.append({
                    "content": text,
                    "metadata": meta,
                })
                console.print(
                    f"  📄 PDF: [green]{file_path.name}[/green]"
                    f" ({page_count} pages, {word_count:,} words{author_info})"
                )
            except Exception as e:
                console.print(f"  ❌ PDF failed: {file_path.name}: {e}")

        elif ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if "content" in item:
                        raw_docs.append(item)
                console.print(f"  📋 JSON: [green]{file_path.name}[/green] ({len(items)} documents)")
            except Exception as e:
                console.print(f"  ❌ JSON failed: {file_path.name}: {e}")

        elif ext in (".txt", ".md"):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    doc_type = "markdown" if ext == ".md" else "text"
                    word_count = len(text.split())
                    meta = {
                        "title": file_path.stem,
                        "source": file_path.name,
                        "document_type": doc_type,
                        "category": _guess_category(file_path.name),
                        "word_count": word_count,
                        "language": "en",
                        "tags": _guess_tags(file_path.name),
                    }
                    # Enrich with citation registry data (APA metadata)
                    meta = enrich_metadata(meta, file_path.name)
                    raw_docs.append({
                        "content": text,
                        "metadata": meta,
                    })
                    console.print(f"  📝 {ext.upper()}: [green]{file_path.name}[/green] ({word_count:,} words)")
            except Exception as e:
                console.print(f"  ❌ Text failed: {file_path.name}: {e}")

    return raw_docs


def _guess_category(filename: str) -> str:
    """Guess a category from the filename."""
    name = filename.lower()
    if any(w in name for w in ["scrum", "kanban", "agile", "xp"]):
        return "agile-methodologies"
    if any(w in name for w in ["testing", "crispin", "gregory"]):
        return "software-testing"
    if any(w in name for w in ["sommerville", "software engineering"]):
        return "software-engineering"
    if any(w in name for w in ["cohn", "user stor"]):
        return "requirements-engineering"
    if any(w in name for w in ["meyer", "becker", "babb", "stray", "runeson", "waterman", "dingsøyr", "dingsoyr"]):
        return "research-articles"
    return "general"


def _guess_tags(filename: str) -> List[str]:
    """Guess tags from the filename."""
    tags = []
    name = filename.lower()
    keyword_map = {
        "agile": "agile", "scrum": "scrum", "kanban": "kanban", "xp": "xp",
        "testing": "testing", "test": "testing",
        "software engineering": "software-engineering",
        "user stor": "user-stories", "requirements": "requirements",
        "chapter": "textbook-chapter",
    }
    for keyword, tag in keyword_map.items():
        if keyword in name:
            tags.append(tag)
    return tags if tags else ["document"]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

async def ingest_all(raw_docs: List[Dict[str, Any]], settings: Settings) -> Dict[str, Any]:
    """Ingest all documents into Qdrant with progress display."""
    qdrant = QdrantManager(settings)
    embedding_service = EmbeddingService(settings)
    document_store = DocumentStore(qdrant, settings)

    # Health check
    if not qdrant.health_check():
        console.print("[red]❌ Cannot connect to Qdrant! Is it running?[/red]")
        return {"successful": 0, "failed": len(raw_docs)}

    # Ensure collection exists
    qdrant.initialize_collection()
    console.print("✅ Qdrant collection ready\n")

    # Convert to Document objects
    documents: List[Document] = []
    for doc_data in raw_docs:
        try:
            meta = doc_data.get("metadata", {})
            metadata = DocumentMetadata(**meta)
            doc = Document(content=doc_data["content"], metadata=metadata)
            documents.append(doc)
        except Exception as e:
            console.print(f"  ⚠️  Skipping invalid doc: {e}")

    # Cost estimate
    contents = [d.content for d in documents]
    cost = embedding_service.estimate_cost(contents)
    console.print(f"📊 Total tokens: [cyan]{cost['total_tokens']:,}[/cyan]  |  "
                  f"Estimated cost: [cyan]${cost['estimated_cost_usd']:.4f}[/cyan]  |  "
                  f"Batches: [cyan]{cost['batch_count']}[/cyan]\n")

    successful = 0
    failed = 0
    total_chunks = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=len(documents))

        # Process one-by-one to give clear feedback
        for i, doc in enumerate(documents):
            title = doc.metadata.title or f"Document {i+1}"
            try:
                # Embed document (main + chunks)
                main_emb, chunk_embs = await embedding_service.embed_document_with_chunks(doc.content)

                # If chunks were created, store the chunk texts on the document
                chunk_embeddings = None
                if chunk_embs:
                    doc.chunks = embedding_service.chunk_text(doc.content)
                    chunk_embeddings = [ce.embedding for ce in chunk_embs]
                    total_chunks += len(chunk_embs)

                # Ingest
                result = document_store.ingest_document(
                    document=doc,
                    embedding=main_emb.embedding,
                    chunk_embeddings=chunk_embeddings,
                )

                if result.success:
                    chunks_info = f" (+{result.chunk_count} chunks)" if result.chunk_count else ""
                    console.print(f"  ✅ {title}{chunks_info}")
                    successful += 1
                else:
                    console.print(f"  ❌ {title}: {result.message}")
                    failed += 1
            except Exception as e:
                console.print(f"  ❌ {title}: {e}")
                failed += 1

            progress.update(task, advance=1)

    elapsed = time.time() - start_time

    return {
        "successful": successful,
        "failed": failed,
        "total_chunks": total_chunks,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="QdrantRAG-Pro — Full Document Ingestion")
    parser.add_argument(
        "--subject", "-s",
        type=str,
        default=None,
        help="Subject name (e.g. 'software-engineering'). "
             "Uses data/subjects/<subject>/documents/ and a dedicated Qdrant collection."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    # Determine documents directory
    if args.subject:
        documents_dir = subject_documents_dir(args.subject)
    else:
        documents_dir = project_root / "data" / "documents"

    if not documents_dir.exists():
        console.print(f"[red]❌ Documents directory not found: {documents_dir}[/red]")
        if args.subject:
            console.print(f"[yellow]Create it with: mkdir -p {documents_dir}[/yellow]")
        return 1

    subject_label = f"  |  Subject: [bold]{args.subject}[/bold]" if args.subject else ""
    console.print(Panel.fit(
        "[bold blue]QdrantRAG-Pro — Full Document Ingestion[/bold blue]\n"
        f"Source: {documents_dir}{subject_label}",
        border_style="blue"
    ))

    # Step 1: Unzip archives into a temp directory
    console.print("\n[bold]Step 1: Extracting archives[/bold]")
    extract_dir = documents_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)

    try:
        extracted_files = unzip_archives(documents_dir, extract_dir)
        if not extracted_files:
            console.print("  (no zip files found)")

        # Step 2: Discover all documents
        console.print("\n[bold]Step 2: Discovering & reading documents[/bold]")
        raw_docs = discover_documents(documents_dir, extracted_files)

        if not raw_docs:
            console.print("[yellow]⚠️  No documents found to ingest![/yellow]")
            return 0

        console.print(f"\n📚 Total documents discovered: [bold]{len(raw_docs)}[/bold]\n")

        # Step 3: Ingest into Qdrant
        console.print("[bold]Step 3: Embedding & ingesting into Qdrant[/bold]")
        settings = Settings()
        if args.subject:
            settings = apply_subject(settings, args.subject)
            console.print(f"📂 Collection: [cyan]{settings.qdrant_collection_name}[/cyan]")
        results = await ingest_all(raw_docs, settings)

        # Summary
        summary = Table(title="Ingestion Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Documents ingested", str(results["successful"]))
        summary.add_row("Chunks created", str(results["total_chunks"]))
        summary.add_row("Failed", str(results["failed"]))
        summary.add_row("Total time", f"{results['elapsed']:.1f}s")
        console.print(summary)

        if results["successful"] > 0:
            console.print(Panel.fit(
                "[bold green]🎉 Ingestion complete![/bold green]\n"
                "Search your documents with:\n"
                "  [cyan]python scripts/interactive_search.py[/cyan]",
                border_style="green"
            ))

    finally:
        # Clean up extracted files
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
