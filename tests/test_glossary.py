"""Tests for kb glossary commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kb.db import Database


@pytest.fixture
def glossary_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "memory").mkdir()
        # Create a minimal glossary.md
        (root / "memory" / "glossary.md").write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n|------|----------|\n| AC | Lattice Co |\n"
        )
        db = Database(root / "data")
        yield db, root
        db.close()


class TestGlossaryAdd:
    def test_add_term(self, glossary_env):
        from kb.glossary import add_term

        _db, root = glossary_env
        add_term(root, "SCRT", "Secrets Detection", section="Acronyms")
        content = (root / "memory" / "glossary.md").read_text()
        assert "| SCRT | Secrets Detection |" in content

    def test_add_term_creates_section(self, glossary_env):
        from kb.glossary import add_term

        _db, root = glossary_env
        add_term(root, "Haiku", "Fast small model", section="Internal Jargon")
        content = (root / "memory" / "glossary.md").read_text()
        assert "## Internal Jargon" in content
        assert "| Haiku | Fast small model |" in content


class TestGlossaryDelete:
    def test_delete_term(self, glossary_env):
        from kb.glossary import delete_term

        _db, root = glossary_env
        delete_term(root, "AC")
        content = (root / "memory" / "glossary.md").read_text()
        assert "| AC |" not in content


class TestGlossaryEdit:
    def test_edit_term_updates_expansion(self, glossary_env):
        from kb.glossary import edit_term

        _db, root = glossary_env
        result = edit_term(root, "AC", "Lattice Cooration")
        assert result["term"] == "AC"
        assert result["expansion"] == "Lattice Cooration"
        content = (root / "memory" / "glossary.md").read_text()
        assert "| AC | Lattice Cooration |" in content
        assert "| AC | Lattice Co |" not in content

    def test_edit_term_not_found_raises(self, glossary_env):
        from kb.glossary import edit_term

        _db, root = glossary_env
        with pytest.raises(ValueError, match="Term not found"):
            edit_term(root, "NONEXISTENT", "whatever")

    def test_edit_term_preserves_other_rows(self, glossary_env):
        from kb.glossary import add_term, edit_term

        _db, root = glossary_env
        add_term(root, "SCRT", "Secrets Detection", section="Acronyms")
        edit_term(root, "AC", "Lattice Cooration")
        content = (root / "memory" / "glossary.md").read_text()
        assert "| SCRT | Secrets Detection |" in content
        assert "| AC | Lattice Cooration |" in content


class TestGlossaryList:
    def test_list_terms(self, glossary_env):
        from kb.glossary import list_terms

        _db, root = glossary_env
        terms = list_terms(root)
        assert any(t.term == "AC" for t in terms)

    def test_list_terms_returns_glossary_entries(self, glossary_env):
        from kb.glossary import list_terms
        from kb.types import GlossaryEntry

        _db, root = glossary_env
        terms = list_terms(root)
        assert len(terms) > 0
        assert isinstance(terms[0], GlossaryEntry)
        # Verify fields are accessible as attributes
        entry = terms[0]
        assert entry.term == "AC"
        assert entry.expansion == "Lattice Co"
        assert entry.section == "Acronyms"
