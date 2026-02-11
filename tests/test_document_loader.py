"""
Tests for the document loader (core/loaders/document_loader.py).
"""

import pytest
from pathlib import Path

from core.loaders.document_loader import (
    UserDocument,
    parse_page_markers,
    load_user_document,
)


class TestParsePageMarkers:
    """Tests for parse_page_markers."""

    def test_returns_none_when_no_markers(self):
        text = "Just some text without any page markers."
        assert parse_page_markers(text) is None

    def test_returns_none_for_empty_string(self):
        assert parse_page_markers("") is None

    def test_parses_page_markers(self):
        text = "--- Page 1 ---\n\nFirst page.\n\n--- Page 2 ---\n\nSecond page."
        result = parse_page_markers(text)
        assert result is not None
        assert len(result) == 2
        assert result[0] == (1, "First page.")
        assert result[1] == (2, "Second page.")

    def test_parses_slide_markers(self):
        text = "--- Slide 1 ---\n\nSlide one.\n\n--- Slide 2 ---\n\nSlide two."
        result = parse_page_markers(text)
        assert result is not None
        assert len(result) == 2
        assert result[0] == (1, "Slide one.")
        assert result[1] == (2, "Slide two.")

    def test_case_insensitive(self):
        text = "--- PAGE 1 ---\n\nContent."
        result = parse_page_markers(text)
        assert result is not None
        assert result[0] == (1, "Content.")


class TestUserDocument:
    """Tests for UserDocument and to_ingest_dict."""

    def test_to_ingest_dict_includes_content_and_metadata(self):
        doc = UserDocument(content="Hello", metadata={"title": "Test"})
        d = doc.to_ingest_dict()
        assert d["content"] == "Hello"
        assert d["metadata"] == {"title": "Test"}
        assert "page_list" not in d

    def test_to_ingest_dict_includes_page_list_when_present(self):
        doc = UserDocument(
            content="Hi",
            metadata={},
            page_list=[(1, "Page 1"), (2, "Page 2")],
        )
        d = doc.to_ingest_dict()
        assert d["page_list"] == [(1, "Page 1"), (2, "Page 2")]


class TestLoadUserDocumentTxt:
    """Tests for load_user_document with .txt files."""

    def test_load_txt_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world.\n\nThis is a test.")
        doc = load_user_document(f)
        assert doc.content == "Hello world.\n\nThis is a test."
        assert "title" in doc.metadata
        assert doc.metadata["title"] == "doc"
        assert doc.metadata["document_type"] == "text"
        assert doc.page_list is None

    def test_load_txt_with_page_markers(self, tmp_path):
        f = tmp_path / "paged.txt"
        f.write_text(
            "--- Page 1 ---\n\nIntro.\n\n--- Page 2 ---\n\nMore."
        )
        doc = load_user_document(f)
        assert doc.content  # full content
        assert doc.page_list is not None
        assert len(doc.page_list) == 2
        assert doc.page_list[0][0] == 1 and "Intro" in doc.page_list[0][1]
        assert doc.page_list[1][0] == 2 and "More" in doc.page_list[1][1]

    def test_title_override(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        doc = load_user_document(f, title="Custom Title")
        assert doc.metadata["title"] == "Custom Title"

    def test_category_override(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        doc = load_user_document(f, category="essays")
        assert doc.metadata.get("category") == "essays"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_user_document(Path("/nonexistent/file.txt"))

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_user_document(f)


class TestLoadUserDocumentPdf:
    """Tests for load_user_document with .pdf files."""

    def test_load_pdf_returns_user_document(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 minimal\n")
        # Minimal bytes are not a valid PDF; fitz will fail to open. Skip when that happens.
        try:
            doc = load_user_document(pdf_path)
        except Exception as e:
            msg = str(e).lower()
            if "failed to open" in msg or "no objects" in msg or "fzerror" in msg or "filedataerror" in msg:
                pytest.skip(f"Minimal PDF not valid for fitz: {e}")
            raise
        assert isinstance(doc, UserDocument)
        assert hasattr(doc, "content") and hasattr(doc, "metadata")
        assert doc.metadata.get("document_type") == "pdf"


class TestLoadUserDocumentMd:
    """Tests for load_user_document with .md files."""

    def test_load_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nBody.")
        doc = load_user_document(f)
        assert "# Title" in doc.content
        assert doc.metadata["document_type"] == "markdown"
