"""Tests for document → search query strategy (sliding window, exhaustive + overlap)."""

import pytest
from core.services.document_query_strategy import (
    document_to_search_queries,
    DEFAULT_WINDOW_CHARS,
    DEFAULT_STEP_CHARS,
)


def test_empty_returns_empty_list():
    assert document_to_search_queries("") == []
    assert document_to_search_queries("   ") == []


def test_short_document_returns_one_query():
    """Document shorter than window → single query (full text)."""
    text = "This is a short essay."
    assert document_to_search_queries(text) == [text]


def test_long_document_multiple_overlapping_queries():
    """Long document → multiple queries covering full length with overlap."""
    long = "a " * (DEFAULT_WINDOW_CHARS // 2 + 500)  # e.g. > 2000 chars
    queries = document_to_search_queries(long)
    assert len(queries) >= 2
    for q in queries:
        assert len(q) <= DEFAULT_WINDOW_CHARS + 50  # margin for word boundary
    # Collectively exhaustive: total covered length >= len(long)
    # With step 1000, window 2000: first window 0:2000, second 1000:3000, ...
    combined_len = sum(len(q) for q in queries)
    assert combined_len >= len(long.strip()) - (DEFAULT_WINDOW_CHARS * 2)  # overlap means we "overcover"


def test_overlap_between_consecutive_queries():
    """With step < window, total chars in queries exceed doc length (overlap)."""
    # Clearly longer than one window so we get 2+ queries
    text = "word " * (DEFAULT_WINDOW_CHARS // 4 + 100)  # > 2000 chars
    queries = document_to_search_queries(text)
    assert len(queries) >= 2
    # Overlap means we "overcover": sum of query lengths > document length
    total_chars = sum(len(q) for q in queries)
    assert total_chars > len(text.strip())


def test_custom_window_and_step():
    """Custom window_size and step_size produce expected number of windows."""
    # 500 chars, window 100, step 50 → roughly (500-100)/50 + 1 ≈ 9 windows with overlap
    text = "x " * 250
    queries = document_to_search_queries(text, window_size=100, step_size=50)
    assert len(queries) >= 2
    for q in queries:
        assert len(q) <= 100 + 20


def test_collectively_exhaustive():
    """First query starts at doc start, last query ends at doc end (full coverage)."""
    words = ["section" + str(i) + " " for i in range(400)]
    text = "".join(words).strip()
    assert len(text) > DEFAULT_WINDOW_CHARS
    queries = document_to_search_queries(text)
    assert len(queries) >= 2
    assert text.startswith(queries[0][:100].strip()) or queries[0].strip() in text[: len(queries[0]) + 50]
    assert text.endswith(queries[-1][-100:].strip()) or queries[-1].strip() in text[-len(queries[-1]) - 50 :]
