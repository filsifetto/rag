"""
RAG system core library.

Provides configuration, database access, document models,
embedding / search / response-generation services, and document parsers.
"""

__version__ = "1.0.0"
__author__ = "QdrantRAG-Pro Team"
__email__ = "contact@qdrantrag-pro.com"

from .config import Settings
from .database.qdrant_client import QdrantManager
from .database.document_store import DocumentStore
from .services.embedding_service import EmbeddingService
from .services.search_engine import HybridSearchEngine
from .services.response_generator import ResponseGenerator
from .parsers.pdf import PDFMetadataExtractor

__all__ = [
    "Settings",
    "QdrantManager",
    "DocumentStore",
    "EmbeddingService",
    "HybridSearchEngine",
    "ResponseGenerator",
    "PDFMetadataExtractor",
]
