"""
APA 7 citation registry and formatting utilities for QdrantRAG-Pro.

This module provides:
  - A registry mapping source filenames to full bibliographic metadata.
  - Functions to look up citation data during ingestion.
  - APA 7 formatting helpers for in-text citations and reference lists.

The registry is loaded from a JSON file (default: data/citation_registry.json).
Override with the CITATION_REGISTRY_PATH environment variable.

Registry file format: JSON array of objects. Each object must have:
  - "pattern": regex string matched against the document filename (first match wins).
  - APA fields: author, year, title, publication_type ("book" | "journal" | "other"),
    and optionally chapter, edition, publisher, isbn, journal, volume, issue, pages, doi.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry path and cache
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = _PROJECT_ROOT / "data" / "citation_registry.json"

_registry_cache: Optional[List[Dict[str, Any]]] = None


def clear_registry_cache() -> None:
    """Clear the in-memory registry cache. Used when switching registry path or in tests."""
    global _registry_cache
    _registry_cache = None


def _get_registry_path() -> Path:
    """Resolve path to the citation registry file (env override supported)."""
    path = os.environ.get("CITATION_REGISTRY_PATH")
    return Path(path) if path else DEFAULT_REGISTRY_PATH


def load_registry(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Load the citation registry from the configured JSON file.

    Entries are cached. Use force_reload=True to re-read from disk.
    Each entry must have a "pattern" (regex string) and APA metadata fields.
    Invalid patterns are skipped and logged.
    """
    global _registry_cache
    if _registry_cache is not None and not force_reload:
        return _registry_cache

    path = _get_registry_path()
    if not path.exists():
        logger.debug("Citation registry not found at %s; using empty registry.", path)
        _registry_cache = []
        return _registry_cache

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load citation registry from %s: %s", path, e)
        _registry_cache = []
        return _registry_cache

    if not isinstance(raw, list):
        logger.warning("Citation registry must be a JSON array; got %s", type(raw).__name__)
        _registry_cache = []
        return _registry_cache

    compiled: List[Dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or "pattern" not in entry:
            logger.debug("Skipping registry entry %s: missing 'pattern' or not an object.", i)
            continue
        pattern = entry.get("pattern")
        if not isinstance(pattern, str):
            continue
        try:
            re_obj = re.compile(pattern)
        except re.error as e:
            logger.warning("Invalid regex in citation registry entry %s: %s", i, e)
            continue
        compiled.append({**entry, "_re": re_obj})

    _registry_cache = compiled
    return _registry_cache


def lookup_citation(filename: str) -> Optional[Dict[str, Any]]:
    """Look up citation metadata for a given filename.

    Returns the first matching registry entry (without internal keys), or ``None`` if no match.
    """
    registry = load_registry()
    for entry in registry:
        if entry["_re"].search(filename):
            return {k: v for k, v in entry.items() if k != "pattern" and k != "_re"}
    return None


def enrich_metadata(metadata: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Enrich a metadata dict with citation information from the registry.

    Registry values are added *only* for keys that are not already set in
    ``metadata``, so explicit values always take precedence.
    """
    citation = lookup_citation(filename)
    if citation is None:
        return metadata

    for key, value in citation.items():
        if key not in metadata or metadata[key] is None:
            metadata[key] = value

    return metadata


# ---------------------------------------------------------------------------
# APA 7 formatting helpers
# ---------------------------------------------------------------------------

def format_apa_inline(metadata: Dict[str, Any]) -> str:
    """Produce an APA 7 in-text citation key, e.g. ``(Sommerville, 2015, Ch. 1, p. 12)``.

    Falls back to the title or source if author/year are unavailable.
    """
    author_raw = metadata.get("author", "")
    year = metadata.get("year")
    chapter = metadata.get("chapter")
    page_number = metadata.get("page_number")

    if author_raw:
        # Extract the first surname for in-text use
        # Handles "Surname, I." and "Surname, I., & Surname2, J."
        surnames = _extract_surnames(author_raw)
        if len(surnames) == 1:
            author_part = surnames[0]
        elif len(surnames) == 2:
            author_part = f"{surnames[0]} & {surnames[1]}"
        else:
            author_part = f"{surnames[0]} et al."
    else:
        author_part = metadata.get("title") or metadata.get("source") or "Unknown"

    parts = [author_part]
    if year:
        parts.append(str(year))
    if chapter:
        parts.append(f"Ch. {chapter}")
    if page_number is not None:
        parts.append(f"p. {page_number}")

    return f"({', '.join(parts)})"


def format_apa_reference(metadata: Dict[str, Any]) -> str:
    """Produce a full APA 7 reference-list entry.

    Supports books (with optional edition/chapter) and journal articles.
    """
    pub_type = metadata.get("publication_type", "other")
    author = metadata.get("author", "Unknown")
    year = metadata.get("year", "n.d.")
    title = metadata.get("title", "Untitled")

    if pub_type == "journal":
        journal = metadata.get("journal", "")
        volume = metadata.get("volume", "")
        issue = metadata.get("issue", "")
        pages = metadata.get("pages", "")
        doi = metadata.get("doi", "")

        ref = f"{author} ({year}). {title}. "
        if journal:
            ref += f"*{journal}*"
            if volume:
                ref += f", *{volume}*"
            if issue:
                ref += f"({issue})"
            if pages:
                ref += f", {pages}"
            ref += "."
        if doi:
            ref += f" https://doi.org/{doi}"
        return ref

    # Default: book
    publisher = metadata.get("publisher", "")
    edition = metadata.get("edition", "")
    isbn = metadata.get("isbn", "")

    title_part = f"*{title}*"
    if edition:
        title_part += f" ({edition} ed.)"

    ref = f"{author} ({year}). {title_part}."
    if publisher:
        ref += f" {publisher}."
    if isbn:
        ref += f" ISBN: {isbn}."
    return ref


def build_citation_key(metadata: Dict[str, Any]) -> str:
    """Build a short citation key for context labelling, e.g. ``Sommerville, 2015``.

    Used as a source identifier in the context passed to the LLM.
    """
    author_raw = metadata.get("author", "")
    year = metadata.get("year")

    if author_raw:
        surnames = _extract_surnames(author_raw)
        if len(surnames) == 1:
            key = surnames[0]
        elif len(surnames) == 2:
            key = f"{surnames[0]} & {surnames[1]}"
        else:
            key = f"{surnames[0]} et al."
    else:
        key = metadata.get("title") or metadata.get("source") or "Unknown"

    if year:
        key += f", {year}"
    return key


def _extract_surnames(author_string: str) -> List[str]:
    """Extract surname(s) from an APA author string.

    Handles formats like:
      - ``Sommerville, I.``
      - ``Crispin, L., & Gregory, J.``
      - ``Becker, C., Betz, S., Chitchyan, R., ...``
    """
    # Split on " & " or ", & " first, then by comma pairs
    # APA format: "Last, F. M., Last2, F., & Last3, G."
    surnames: List[str] = []
    # Normalise ampersand forms
    cleaned = author_string.replace(", &", " &").replace("&", ",")
    # Split by comma and iterate in pairs (surname, initials)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    for part in parts:
        # A surname is a part that doesn't look like bare initials (e.g. "I." or "D. I. K.")
        if not re.match(r'^[A-Z]\.(\s*[A-Z]\.)*$', part):
            surnames.append(part)
    return surnames if surnames else [author_string.strip()]
