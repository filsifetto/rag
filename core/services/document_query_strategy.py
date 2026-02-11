"""
Turn a user document into one or more search queries for the subject RAG.

Used by the document feedback flow to retrieve relevant subject material
based on the uploaded document. Uses a sliding-window strategy so that
(1) queries are collectively exhaustive (every part of the document is
covered) and (2) consecutive windows overlap to improve recall at boundaries.
"""

from typing import List

# Default window size (chars per query) and step (chars to advance between windows).
# step < window → overlapping queries. step=1000, window=2000 → 50% overlap.
DEFAULT_WINDOW_CHARS = 2000
DEFAULT_STEP_CHARS = 1000


def document_to_search_queries(
    document_text: str,
    window_size: int = DEFAULT_WINDOW_CHARS,
    step_size: int = DEFAULT_STEP_CHARS,
) -> List[str]:
    """Produce search queries that cover the whole document with overlap.

    Uses a sliding window: each query is a contiguous slice of the document.
    Windows overlap (step_size < window_size) so content at boundaries is
    still retrieved. Together the windows are collectively exhaustive.

    Args:
        document_text: Full text of the user's document.
        window_size: Maximum characters per query (keeps embeddings within limits).
        step_size: Characters to advance between windows. Use step_size < window_size
            for overlap (e.g. step 1000, window 2000 → 50% overlap).

    Returns:
        A list of query strings, one per window, in document order.
    """
    if not document_text or not document_text.strip():
        return []
    text = document_text.strip()

    # Single window suffices
    if len(text) <= window_size:
        return [text]

    # Ensure step is positive and at most window (otherwise we'd have gaps)
    step_size = max(1, min(step_size, window_size))

    queries: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        chunk = text[start:end]

        # Trim end to word boundary when we're not at end of document and cut mid-word
        if end < len(text) and chunk and chunk[-1].isalnum():
            last_space = chunk.rfind(" ")
            if last_space > len(chunk) // 2:
                chunk = chunk[: last_space + 1].rstrip()

        if chunk.strip():
            queries.append(chunk.strip())

        start += step_size

    return queries
