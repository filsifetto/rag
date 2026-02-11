"""
Load a single file into text and metadata (no ingestion).

Provides load_user_document() for use by document feedback and by the
ingestion script so "file → content" logic lives in one place.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ..citation import enrich_metadata
from ..parsers.pdf import PDFMetadataExtractor


@dataclass
class UserDocument:
    """Content and metadata from a single file (not yet ingested)."""

    content: str
    metadata: dict
    page_list: Optional[List[Tuple[int, str]]] = None

    def to_ingest_dict(self) -> dict:
        """Dict suitable for ingestion: content, metadata, optional page_list."""
        out = {"content": self.content, "metadata": self.metadata}
        if self.page_list is not None:
            out["page_list"] = self.page_list
        return out


def parse_page_markers(content: str) -> Optional[List[Tuple[int, str]]]:
    """Parse '--- Page N ---' or '--- Slide N ---' markers in text.

    Returns a list of (page_number, page_text) for page-aware chunking,
    or None if no markers are found.
    """
    pattern = re.compile(
        r"^--- (?:Page|Slide) (\d+) ---\s*",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(content))
    if not matches:
        return None
    page_list = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        page_text = content[start:end].strip()
        page_list.append((page_num, page_text))
    return page_list if page_list else None


def load_user_document(
    file_path: Path,
    *,
    title: Optional[str] = None,
    category: Optional[str] = None,
) -> UserDocument:
    """Load a single file (.txt, .md, or .pdf) into content and metadata.

    The file is not ingested; this only reads it. For document feedback
    you get the text to send to the LLM and to derive search queries.
    For ingestion, call to_ingest_dict() and pass the result to the
    ingestion pipeline.

    Args:
        file_path: Path to the file.
        title: Optional override for document title.
        category: Optional category (e.g. for ingestion).

    Returns:
        UserDocument with content, metadata, and optional page_list
        (for PDFs and TXT with --- Page N --- markers).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file or format is unsupported.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path, title=title, category=category)
    if suffix in (".txt", ".md"):
        return _load_text_or_markdown(file_path, suffix, title=title, category=category)

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: .txt, .md, .pdf"
    )


def _load_pdf(
    file_path: Path,
    *,
    title: Optional[str] = None,
    category: Optional[str] = None,
) -> UserDocument:
    extractor = PDFMetadataExtractor()
    content, page_list, pdf_meta = extractor.extract_text_and_metadata_by_page(
        file_path
    )
    if not content.strip():
        raise ValueError(f"No extractable text found in PDF: {file_path}")

    meta = pdf_meta.to_document_metadata_dict(source=str(file_path.resolve()))
    if title is not None:
        meta["title"] = title
    elif not meta.get("title"):
        meta["title"] = file_path.stem
    if category is not None:
        meta["category"] = category
    meta = enrich_metadata(meta, file_path.name)

    return UserDocument(content=content, metadata=meta, page_list=page_list)


def _load_text_or_markdown(
    file_path: Path,
    suffix: str,
    *,
    title: Optional[str] = None,
    category: Optional[str] = None,
) -> UserDocument:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc_type = "markdown" if suffix == ".md" else "text"
    meta = {
        "title": title or file_path.stem,
        "source": str(file_path.resolve()),
        "document_type": doc_type,
        "file_size": file_path.stat().st_size,
    }
    if category is not None:
        meta["category"] = category
    meta = enrich_metadata(meta, file_path.name)

    page_list = parse_page_markers(content)
    return UserDocument(content=content, metadata=meta, page_list=page_list)
