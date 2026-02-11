"""
Document loaders for reading files into content and metadata.

Used by ingestion and by document feedback; provides a single place for
"file path → content + metadata" so both flows stay consistent.
"""

from .document_loader import (
    UserDocument,
    parse_page_markers,
    load_user_document,
)

__all__ = [
    "UserDocument",
    "parse_page_markers",
    "load_user_document",
]
