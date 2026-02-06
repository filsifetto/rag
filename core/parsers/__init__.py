"""Document parsers for extracting text and metadata from various file formats."""

from .pdf import PDFMetadataExtractor, PDFMetadata

__all__ = ["PDFMetadataExtractor", "PDFMetadata"]
