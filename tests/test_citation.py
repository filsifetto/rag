"""
Tests for the citation registry and APA formatting (core/citation.py).
"""

import os
import pytest

from core.citation import (
    lookup_citation,
    enrich_metadata,
    load_registry,
    clear_registry_cache,
    format_apa_inline,
    format_apa_reference,
    build_citation_key,
)


def _restore_registry_env_and_cache(env_prev):
    """Restore CITATION_REGISTRY_PATH and clear cache so default registry is used again."""
    if env_prev is None:
        os.environ.pop("CITATION_REGISTRY_PATH", None)
    else:
        os.environ["CITATION_REGISTRY_PATH"] = env_prev
    clear_registry_cache()


class TestLookupCitation:
    """Tests for lookup_citation using the default registry file."""

    def test_match_returns_metadata(self):
        # Default registry has Sommerville chapter 1 pattern
        result = lookup_citation("Sommerville - Chapter 1.pdf")
        assert result is not None
        assert result.get("author") == "Sommerville, I."
        assert result.get("year") == 2015
        assert result.get("title") == "Software Engineering"
        assert "pattern" not in result

    def test_no_match_returns_none(self):
        assert lookup_citation("UnknownFile123.pdf") is None


class TestEnrichMetadata:
    """Tests for enrich_metadata."""

    def test_fills_missing_keys(self):
        meta = {}
        out = enrich_metadata(meta, "Sommerville - Chapter 1.pdf")
        assert out.get("author") == "Sommerville, I."
        assert out.get("year") == 2015

    def test_preserves_existing_values(self):
        meta = {"author": "Custom Author"}
        out = enrich_metadata(meta, "Sommerville - Chapter 1.pdf")
        assert out["author"] == "Custom Author"
        assert out.get("year") == 2015

    def test_fills_none_values(self):
        """Registry fills keys that are present but None (treated as missing)."""
        meta = {"author": None, "year": 2020}
        out = enrich_metadata(meta, "Sommerville - Chapter 1.pdf")
        assert out["author"] == "Sommerville, I."
        assert out["year"] == 2020


class TestLoadRegistry:
    """Tests for load_registry."""

    def test_returns_list(self):
        reg = load_registry()
        assert isinstance(reg, list)

    def test_entries_have_compiled_pattern(self):
        reg = load_registry()
        if reg:
            entry = reg[0]
            assert "_re" in entry
            assert hasattr(entry["_re"], "search")

    def test_force_reload_rereads_from_disk(self):
        """force_reload=True bypasses cache and re-reads the file (new list, same content)."""
        reg1 = load_registry()
        reg2 = load_registry(force_reload=True)
        assert len(reg2) == len(reg1)
        # Content equivalent (e.g. first entry has same author)
        if reg1 and reg2:
            assert reg1[0].get("author") == reg2[0].get("author")


class TestRegistryFromCustomPath:
    """Tests that use a temporary registry file via CITATION_REGISTRY_PATH."""

    def test_loads_from_temp_file_and_lookup_works(self, tmp_path):
        """When CITATION_REGISTRY_PATH points to a valid JSON file, lookup uses it."""
        registry_json = tmp_path / "citations.json"
        registry_json.write_text(
            '[{"pattern": "(?i)mybook", "author": "Test, A.", "year": 2000, '
            '"title": "My Book", "publication_type": "book"}]',
            encoding="utf-8",
        )
        prev = os.environ.get("CITATION_REGISTRY_PATH")
        try:
            os.environ["CITATION_REGISTRY_PATH"] = str(registry_json)
            clear_registry_cache()
            result = lookup_citation("mybook.pdf")
            assert result is not None
            assert result["author"] == "Test, A."
            assert result["year"] == 2000
            assert result["title"] == "My Book"
        finally:
            _restore_registry_env_and_cache(prev)

    def test_missing_file_returns_empty_registry(self, tmp_path):
        """When the registry file does not exist, load_registry returns [] and lookup returns None."""
        missing = tmp_path / "nonexistent.json"
        assert not missing.exists()
        prev = os.environ.get("CITATION_REGISTRY_PATH")
        try:
            os.environ["CITATION_REGISTRY_PATH"] = str(missing)
            clear_registry_cache()
            reg = load_registry(force_reload=True)
            assert reg == []
            assert lookup_citation("anything.pdf") is None
        finally:
            _restore_registry_env_and_cache(prev)

    def test_invalid_json_returns_empty_registry(self, tmp_path):
        """Malformed JSON results in empty registry and no exception."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json {", encoding="utf-8")
        prev = os.environ.get("CITATION_REGISTRY_PATH")
        try:
            os.environ["CITATION_REGISTRY_PATH"] = str(bad_json)
            clear_registry_cache()
            reg = load_registry(force_reload=True)
            assert reg == []
            assert lookup_citation("anything.pdf") is None
        finally:
            _restore_registry_env_and_cache(prev)

    def test_non_array_json_returns_empty_registry(self, tmp_path):
        """Registry must be a JSON array; object or other types yield empty list."""
        obj_json = tmp_path / "obj.json"
        obj_json.write_text('{"pattern": "x"}', encoding="utf-8")
        prev = os.environ.get("CITATION_REGISTRY_PATH")
        try:
            os.environ["CITATION_REGISTRY_PATH"] = str(obj_json)
            clear_registry_cache()
            reg = load_registry(force_reload=True)
            assert reg == []
        finally:
            _restore_registry_env_and_cache(prev)

    def test_invalid_regex_entry_skipped_others_loaded(self, tmp_path):
        """Entries with invalid regex are skipped; valid entries still work."""
        registry_json = tmp_path / "mixed.json"
        registry_json.write_text(
            '['
            '{"pattern": "[invalid", "author": "Bad"},'
            '{"pattern": "(?i)good", "author": "Good, A.", "year": 1999, "title": "Good", "publication_type": "book"}'
            ']',
            encoding="utf-8",
        )
        prev = os.environ.get("CITATION_REGISTRY_PATH")
        try:
            os.environ["CITATION_REGISTRY_PATH"] = str(registry_json)
            clear_registry_cache()
            reg = load_registry(force_reload=True)
            assert len(reg) == 1
            assert reg[0].get("author") == "Good, A."
            result = lookup_citation("good.pdf")
            assert result is not None
            assert result["author"] == "Good, A."
            assert lookup_citation("bad.pdf") is None
        finally:
            _restore_registry_env_and_cache(prev)

    def test_entry_without_pattern_skipped(self, tmp_path):
        """Entries that are not objects or lack 'pattern' are skipped."""
        registry_json = tmp_path / "nopattern.json"
        registry_json.write_text(
            '[{"author": "NoPattern", "year": 2000}, {"pattern": "(?i)yes", "author": "Yes", "year": 2001, "title": "Y", "publication_type": "book"}]',
            encoding="utf-8",
        )
        prev = os.environ.get("CITATION_REGISTRY_PATH")
        try:
            os.environ["CITATION_REGISTRY_PATH"] = str(registry_json)
            clear_registry_cache()
            reg = load_registry(force_reload=True)
            assert len(reg) == 1
            assert lookup_citation("yes.pdf") is not None
            assert lookup_citation("nopattern.pdf") is None
        finally:
            _restore_registry_env_and_cache(prev)


class TestRegistryCache:
    """Tests for cache behavior."""

    def test_second_load_returns_same_list_without_force_reload(self):
        """Without force_reload, load_registry returns the same cached list."""
        clear_registry_cache()
        reg1 = load_registry()
        reg2 = load_registry()
        assert reg1 is reg2

    def test_clear_registry_cache_forces_reload_on_next_load(self):
        """After clear_registry_cache(), next load_registry() reads from disk again."""
        load_registry()
        clear_registry_cache()
        reg = load_registry()
        assert reg is not None
        assert isinstance(reg, list)


class TestApaFormatting:
    """Tests for APA formatting helpers (use metadata dicts, no registry)."""

    def test_format_apa_inline(self):
        meta = {"author": "Smith, J.", "year": 2020, "page_number": 5}
        assert "Smith" in format_apa_inline(meta)
        assert "2020" in format_apa_inline(meta)
        assert "p. 5" in format_apa_inline(meta)

    def test_format_apa_reference_journal(self):
        meta = {
            "author": "Doe, J.",
            "year": 2019,
            "title": "A Study",
            "journal": "Journal of X",
            "volume": "10",
            "publication_type": "journal",
        }
        ref = format_apa_reference(meta)
        assert "Doe" in ref and "2019" in ref and "Journal of X" in ref

    def test_build_citation_key(self):
        meta = {"author": "Smith, J.", "year": 2020}
        assert build_citation_key(meta) == "Smith, 2020"
