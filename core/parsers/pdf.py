"""
PDF metadata extraction service for QdrantRAG-Pro.

Extracts embedded metadata from PDF files using PyMuPDF (fitz),
including title, author, subject, keywords, creation/modification dates,
producer application, PDF version, and page-level information.
"""

import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFMetadata:
    """Structured container for PDF metadata."""

    def __init__(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        subject: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        creator: Optional[str] = None,
        producer: Optional[str] = None,
        creation_date: Optional[datetime] = None,
        modification_date: Optional[datetime] = None,
        pdf_version: Optional[str] = None,
        page_count: int = 0,
        word_count: int = 0,
        file_size: int = 0,
        is_encrypted: bool = False,
        has_toc: bool = False,
        toc_entries: Optional[List[str]] = None,
        page_labels: Optional[List[str]] = None,
    ):
        self.title = title
        self.author = author
        self.subject = subject
        self.keywords = keywords or []
        self.creator = creator
        self.producer = producer
        self.creation_date = creation_date
        self.modification_date = modification_date
        self.pdf_version = pdf_version
        self.page_count = page_count
        self.word_count = word_count
        self.file_size = file_size
        self.is_encrypted = is_encrypted
        self.has_toc = has_toc
        self.toc_entries = toc_entries or []
        self.page_labels = page_labels or []

    def to_dict(self) -> Dict[str, Any]:
        """Serialise metadata to a plain dictionary suitable for storage."""
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "modification_date": self.modification_date.isoformat() if self.modification_date else None,
            "pdf_version": self.pdf_version,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "file_size": self.file_size,
            "is_encrypted": self.is_encrypted,
            "has_toc": self.has_toc,
            "toc_entries": self.toc_entries,
            "page_labels": self.page_labels,
        }

    def to_document_metadata_dict(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Return a dict compatible with ``DocumentMetadata`` fields.

        Extra PDF-specific fields are stored inside ``custom_fields``.
        """
        meta: Dict[str, Any] = {
            "document_type": "pdf",
            "page_count": self.page_count,
            "word_count": self.word_count,
            "file_size": self.file_size,
        }

        if self.title:
            meta["title"] = self.title
        if self.author:
            meta["author"] = self.author
        if source:
            meta["source"] = source
        if self.creation_date:
            meta["created_at"] = self.creation_date
        if self.modification_date:
            meta["modified_at"] = self.modification_date
        if self.keywords:
            meta["tags"] = self.keywords

        # PDF-specific fields go into custom_fields
        custom: Dict[str, Any] = {}
        if self.subject:
            custom["subject"] = self.subject
        if self.creator:
            custom["creator_tool"] = self.creator
        if self.producer:
            custom["producer"] = self.producer
        if self.pdf_version:
            custom["pdf_version"] = self.pdf_version
        if self.is_encrypted:
            custom["is_encrypted"] = self.is_encrypted
        if self.has_toc:
            custom["has_toc"] = True
            custom["toc_entries"] = self.toc_entries
        if self.page_labels:
            custom["page_labels"] = self.page_labels

        if custom:
            meta["custom_fields"] = custom

        return meta

    def __repr__(self) -> str:
        parts = [f"PDFMetadata(title={self.title!r}"]
        if self.author:
            parts.append(f"author={self.author!r}")
        parts.append(f"pages={self.page_count}")
        parts.append(f"words={self.word_count}")
        return ", ".join(parts) + ")"


class PDFMetadataExtractor:
    """Extract metadata and text from PDF files using PyMuPDF."""

    # PDF date format: D:YYYYMMDDHHmmSS+HH'mm' (with many optional parts)
    _PDF_DATE_RE = re.compile(
        r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
    )

    # ---- public API --------------------------------------------------------

    def extract_metadata(self, pdf_path: Path) -> PDFMetadata:
        """Extract metadata from a PDF file without reading the full text.

        This is a lightweight operation that only reads the document info
        dictionary and structural information (TOC, page count, etc.).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        try:
            return self._build_metadata(doc, pdf_path)
        finally:
            doc.close()

    def extract_text_and_metadata(
        self,
        pdf_path: Path,
    ) -> Tuple[str, PDFMetadata]:
        """Extract both full text and metadata from a PDF.

        Returns ``(full_text, PDFMetadata)``.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        try:
            text = self._extract_text(doc)
            metadata = self._build_metadata(doc, pdf_path, text=text)
            return text, metadata
        finally:
            doc.close()

    def extract_text_and_metadata_by_page(
        self,
        pdf_path: Path,
    ) -> Tuple[str, List[Tuple[int, str]], PDFMetadata]:
        """Extract full text, per-page text (for page-aware chunking), and metadata.

        Returns ``(full_text, [(page_number, page_text), ...], PDFMetadata)``.
        Page numbers are 1-based.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        try:
            page_list: List[Tuple[int, str]] = []
            pages: List[str] = []
            for i, page in enumerate(doc):
                page_num = i + 1
                text = page.get_text("text")
                if text.strip():
                    page_list.append((page_num, text))
                    pages.append(text)

            full_text = "\n\n".join(pages)
            full_text = re.sub(r"\f", "\n", full_text)
            full_text = re.sub(r"\n{3,}", "\n\n", full_text)
            full_text = full_text.strip()

            metadata = self._build_metadata(doc, pdf_path, text=full_text)
            return full_text, page_list, metadata
        finally:
            doc.close()

    # ---- internal helpers --------------------------------------------------

    def _extract_text(self, doc: fitz.Document) -> str:
        """Extract and clean text from all pages."""
        pages: List[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

        full_text = "\n\n".join(pages)
        full_text = re.sub(r"\f", "\n", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        return full_text.strip()

    def _build_metadata(
        self,
        doc: fitz.Document,
        pdf_path: Path,
        text: Optional[str] = None,
    ) -> PDFMetadata:
        """Build a ``PDFMetadata`` instance from a fitz Document."""
        info: Dict[str, Any] = doc.metadata or {}

        title = self._clean_str(info.get("title"))
        author = self._clean_str(info.get("author"))
        subject = self._clean_str(info.get("subject"))
        creator = self._clean_str(info.get("creator"))
        producer = self._clean_str(info.get("producer"))

        keywords = self._parse_keywords(info.get("keywords"))

        creation_date = self._parse_pdf_date(info.get("creationDate"))
        modification_date = self._parse_pdf_date(info.get("modDate"))

        # PDF version – available through the catalog in some builds
        pdf_version: Optional[str] = None
        try:
            # fitz exposes version info via metadata or xref data
            if hasattr(doc, "pdf_catalog"):
                pdf_version = f"{doc.metadata.get('format', '')}"
            if not pdf_version:
                pdf_version = info.get("format")
        except Exception:
            pass

        # Table of contents
        toc = doc.get_toc(simple=True)
        toc_entries = [entry[1] for entry in toc] if toc else []

        # Page count (pages with extractable text)
        page_count = doc.page_count

        # Page labels (if present)
        page_labels: List[str] = []
        try:
            for i in range(min(page_count, 50)):  # cap to avoid huge lists
                label = doc[i].get_label()
                if label:
                    page_labels.append(label)
        except Exception:
            pass

        word_count = len(text.split()) if text else 0
        file_size = pdf_path.stat().st_size if pdf_path.exists() else 0

        return PDFMetadata(
            title=title,
            author=author,
            subject=subject,
            keywords=keywords,
            creator=creator,
            producer=producer,
            creation_date=creation_date,
            modification_date=modification_date,
            pdf_version=pdf_version,
            page_count=page_count,
            word_count=word_count,
            file_size=file_size,
            is_encrypted=doc.is_encrypted,
            has_toc=bool(toc_entries),
            toc_entries=toc_entries,
            page_labels=page_labels,
        )

    # ---- parsing utilities -------------------------------------------------

    @classmethod
    def _clean_str(cls, value: Any) -> Optional[str]:
        """Return a cleaned non-empty string, or ``None``."""
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    @classmethod
    def _parse_keywords(cls, raw: Any) -> List[str]:
        """Parse the PDF keywords field into a list of tags."""
        if not raw:
            return []
        text = str(raw)
        # Keywords may be separated by commas, semicolons, or spaces
        tokens = re.split(r"[;,]+", text)
        return [t.strip() for t in tokens if t.strip()]

    @classmethod
    def _parse_pdf_date(cls, raw: Any) -> Optional[datetime]:
        """Parse a PDF date string (``D:YYYYMMDDHHmmSS…``) into a datetime."""
        if not raw:
            return None
        text = str(raw).strip()
        m = cls._PDF_DATE_RE.match(text)
        if not m:
            # Try a plain ISO-style fallback
            try:
                return datetime.fromisoformat(text)
            except (ValueError, TypeError):
                logger.debug("Unable to parse PDF date: %s", text)
                return None

        parts = list(m.groups())
        year = int(parts[0])
        month = int(parts[1]) if parts[1] else 1
        day = int(parts[2]) if parts[2] else 1
        hour = int(parts[3]) if parts[3] else 0
        minute = int(parts[4]) if parts[4] else 0
        second = int(parts[5]) if parts[5] else 0

        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            logger.debug("Invalid PDF date components: %s", text)
            return None
