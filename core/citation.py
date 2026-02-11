"""
APA 7 citation registry and formatting utilities for QdrantRAG-Pro.

This module provides:
  - A registry mapping source filenames to full bibliographic metadata.
  - Functions to look up citation data during ingestion.
  - APA 7 formatting helpers for in-text citations and reference lists.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Citation registry
# ---------------------------------------------------------------------------
# Each entry maps a filename pattern (regex) to citation metadata.
# Fields follow APA 7 conventions.
# ---------------------------------------------------------------------------

CITATION_REGISTRY: List[Dict[str, Any]] = [
    # --- Books / book chapters ---
    {
        "pattern": r"(?i)sommerville.*chapter\s*1\b",
        "author": "Sommerville, I.",
        "year": 2015,
        "title": "Software Engineering",
        "chapter": "1",
        "edition": "10th",
        "publisher": "Pearson Education",
        "isbn": "9781292096131",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)sommerville.*chapter\s*2\b",
        "author": "Sommerville, I.",
        "year": 2015,
        "title": "Software Engineering",
        "chapter": "2",
        "edition": "10th",
        "publisher": "Pearson Education",
        "isbn": "9781292096131",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)sommerville.*chapter\s*3\b",
        "author": "Sommerville, I.",
        "year": 2015,
        "title": "Software Engineering",
        "chapter": "3",
        "edition": "10th",
        "publisher": "Pearson Education",
        "isbn": "9781292096131",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)sommerville.*chapter\s*6",
        "author": "Sommerville, I.",
        "year": 2015,
        "title": "Software Engineering",
        "chapter": "6",
        "edition": "10th",
        "publisher": "Pearson Education",
        "isbn": "9781292096131",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)cohn.*chapter\s*1\b",
        "author": "Cohn, M.",
        "year": 2004,
        "title": "User Stories Applied: For Agile Software Development",
        "chapter": "1",
        "publisher": "Addison-Wesley Professional",
        "isbn": "9780321205681",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)cohn.*chapter\s*2\b",
        "author": "Cohn, M.",
        "year": 2004,
        "title": "User Stories Applied: For Agile Software Development",
        "chapter": "2",
        "publisher": "Addison-Wesley Professional",
        "isbn": "9780321205681",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)crispin.*gregory.*chapter\s*6\b",
        "author": "Crispin, L., & Gregory, J.",
        "year": 2008,
        "title": "Agile Testing: A Practical Guide for Testers and Agile Teams",
        "chapter": "6",
        "publisher": "Addison-Wesley Professional",
        "isbn": "9780321534460",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)crispin.*gregory.*chapter\s*10\b",
        "author": "Crispin, L., & Gregory, J.",
        "year": 2008,
        "title": "Agile Testing: A Practical Guide for Testers and Agile Teams",
        "chapter": "10",
        "publisher": "Addison-Wesley Professional",
        "isbn": "9780321534460",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)kanban\s+and\s+scrum",
        "author": "Kniberg, H., & Skarin, M.",
        "year": 2010,
        "title": "Kanban and Scrum: Making the Most of Both",
        "publisher": "C4Media",
        "publication_type": "book",
    },
    {
        "pattern": r"(?i)scrum\s+and\s+xp\s+from\s+the\s+trenches",
        "author": "Kniberg, H.",
        "year": 2015,
        "title": "Scrum and XP from the Trenches",
        "edition": "2nd",
        "publisher": "C4Media",
        "publication_type": "book",
    },

    # --- Journal articles ---
    {
        "pattern": r"(?i)\bmeyer\b",
        "author": "Meyer, B.",
        "year": 2018,
        "title": "Making sense of agile methods",
        "journal": "IEEE Software",
        "volume": "35",
        "issue": "2",
        "pages": "91-94",
        "publication_type": "journal",
    },
    {
        "pattern": r"(?i)\bbecker\b",
        "author": "Becker, C., Betz, S., Chitchyan, R., Duboc, L., Easterbrook, S. M., Penzenstadler, B., Seyff, N., & Venters, C. C.",
        "year": 2016,
        "title": "Requirements: The key to sustainability",
        "journal": "IEEE Software",
        "volume": "33",
        "issue": "1",
        "pages": "56-65",
        "publication_type": "journal",
    },
    {
        "pattern": r"(?i)\bbabb\b",
        "author": "Babb, J., Hoda, R., & Nørbjerg, J.",
        "year": 2014,
        "title": "Embedding reflection and learning into agile software development",
        "journal": "IEEE Software",
        "volume": "31",
        "issue": "4",
        "pages": "51-57",
        "publication_type": "journal",
    },
    {
        "pattern": r"(?i)\bstray\b",
        "author": "Stray, V., Moe, N. B., & Sjøberg, D. I. K.",
        "year": 2020,
        "title": "Daily stand-up meetings: Start breaking the rules",
        "journal": "IEEE Software",
        "volume": "37",
        "issue": "3",
        "pages": "70-77",
        "publication_type": "journal",
    },
    {
        "pattern": r"(?i)\bruneson\b",
        "author": "Runeson, P., Andersson, C., Thelin, T., Andrews, A., & Berber, T.",
        "year": 2006,
        "title": "What do we know about defect detection methods?",
        "journal": "IEEE Software",
        "volume": "23",
        "issue": "3",
        "pages": "82-90",
        "publication_type": "journal",
    },
    {
        "pattern": r"(?i)\bwaterman\b",
        "author": "Waterman, M.",
        "year": 2018,
        "title": "Agility, risk, and uncertainty, part 1: Designing an agile architecture",
        "journal": "IEEE Software",
        "volume": "35",
        "issue": "2",
        "pages": "99-101",
        "publication_type": "journal",
    },
    {
        "pattern": r"(?i)dings.yr",
        "author": "Dingsøyr, T., Strode, D., & Lindsjørn, Y.",
        "year": 2022,
        "title": "Right thoughts & right action: How to make agile teamwork effective",
        "journal": "Amplify",
        "volume": "35",
        "issue": "2",
        "pages": "12-17",
        "publication_type": "journal",
    },
]


def lookup_citation(filename: str) -> Optional[Dict[str, Any]]:
    """Look up citation metadata for a given filename.

    Returns the first matching registry entry, or ``None`` if no match.
    """
    for entry in CITATION_REGISTRY:
        if re.search(entry["pattern"], filename):
            # Return a copy without the regex pattern
            return {k: v for k, v in entry.items() if k != "pattern"}
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
