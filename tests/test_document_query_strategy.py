"""Tests for document → search query strategy."""

import pytest
from core.services.document_query_strategy import (
    document_to_search_queries,
    DEFAULT_MAX_QUERY_CHARS,
)


def test_empty_returns_empty_list():
    assert document_to_search_queries("") == []
    assert document_to_search_queries("   ") == []


def test_short_document_returns_one_query():
    text = "This is a short essay."
    assert document_to_search_queries(text) == [text]


def test_long_document_truncated():
    long = "a" * (DEFAULT_MAX_QUERY_CHARS + 500)
    queries = document_to_search_queries(long)
    assert len(queries) == 1
    assert len(queries[0]) <= DEFAULT_MAX_QUERY_CHARS + 50  # allow some margin for word boundary


def test_custom_max_chars():
    text = "Hello world. " * 200
    queries = document_to_search_queries(text, max_chars=100)
    assert len(queries) == 1
    assert len(queries[0]) <= 100 + 20
