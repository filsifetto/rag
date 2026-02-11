"""
Turn a user document into one or more search queries for the subject RAG.

Used by the document feedback flow to retrieve relevant subject material
based on the uploaded document.
"""

from typing import List

# Default max characters to send as a single query (avoid token overflow).
DEFAULT_MAX_QUERY_CHARS = 2000


def document_to_search_queries(
    document_text: str,
    max_chars: int = DEFAULT_MAX_QUERY_CHARS,
) -> List[str]:
    """Produce one or more search queries from the document text.

    Simple strategy: one query built from the start of the document,
    truncated to max_chars. Later we can add per-section or chunk-based
    strategies.

    Args:
        document_text: Full text of the user's document.
        max_chars: Maximum characters to use for the query (default 2000).

    Returns:
        A list of query strings (currently always length 1).
    """
    if not document_text or not document_text.strip():
        return []
    text = document_text.strip()
    if len(text) <= max_chars:
        return [text]
    query = text[:max_chars].rstrip()
    # Avoid cutting in the middle of a word if possible
    if len(text) > max_chars and text[max_chars].isalnum() and query:
        last_space = query.rfind(" ")
        if last_space > max_chars // 2:
            query = query[: last_space + 1].rstrip()
    return [query] if query else []
