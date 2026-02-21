"""Tests for Pydantic strict models."""

from __future__ import annotations

import pytest


class TestStrictBase:
    def test_strict_rejects_wrong_type(self):
        """Pydantic strict mode rejects type coercion (e.g., int where str expected)."""
        from pydantic import ValidationError

        from kb.types import EntityData

        with pytest.raises(ValidationError):
            EntityData(name=123, entity_type="person", aliases=[], metadata={})

    def test_frozen_prevents_mutation(self):
        """Frozen models reject attribute assignment."""
        from kb.types import GlossaryEntry

        entry = GlossaryEntry(term="AC", expansion="Lattice Co", section="Acronyms")
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            entry.term = "XX"


class TestEntityModels:
    def test_entity_data_construction(self):
        """EntityData (parser output) constructs with all required fields."""
        from kb.types import EntityData

        e = EntityData(
            name="Jane Doe",
            entity_type="person",
            aliases=["JD"],
            metadata={"role": "CTO"},
            source_path="memory/people/jane-doe.md",
        )
        assert e.name == "Jane Doe"
        assert e.entity_type == "person"
        assert e.aliases == ["JD"]
        assert e.metadata == {"role": "CTO"}
        assert e.source_path == "memory/people/jane-doe.md"
        assert e.pinned is False

    def test_entity_data_model_dump(self):
        """EntityData serializes to dict matching the old dict format."""
        from kb.types import EntityData

        e = EntityData(
            name="Test",
            entity_type="person",
            aliases=[],
            metadata={},
            source_path=None,
        )
        d = e.model_dump()
        assert d["name"] == "Test"
        assert d["pinned"] is False

    def test_entity_construction(self):
        """Entity model (from DB) constructs with id."""
        from kb.types import Entity

        e = Entity(
            id=1,
            name="Jane",
            entity_type="person",
            aliases=["JD"],
            metadata={"role": "CTO"},
            source_path="memory/people/jane.md",
            pinned=False,
        )
        assert e.id == 1
        assert e.name == "Jane"

    def test_entity_not_frozen(self):
        """Entity is mutable (needed for crud.py edit operations)."""
        from kb.types import Entity

        e = Entity(
            id=1,
            name="Jane",
            entity_type="person",
            aliases=[],
            metadata={},
            source_path=None,
            pinned=False,
        )
        e.metadata["role"] = "CTO"
        assert e.metadata["role"] == "CTO"

    def test_entity_mention(self):
        """EntityMention constructs correctly."""
        from kb.types import EntityMention

        m = EntityMention(entity_id=1, mention_type="tagged")
        assert m.entity_id == 1
        assert m.mention_type == "tagged"


class TestDocumentModels:
    def test_chunk_construction(self):
        from kb.types import Chunk

        c = Chunk(index=0, heading="Intro", content="Hello world")
        assert c.index == 0
        assert c.heading == "Intro"

    def test_parsed_document_construction(self):
        from kb.types import ParsedDocument

        doc = ParsedDocument(
            path="meetings/2026/01/01/test.md",
            title="Test Meeting",
            date="2026-01-01",
            doc_type="notes",
            source_system="granola",
            source_id="abc12345",
        )
        assert doc.path == "meetings/2026/01/01/test.md"
        assert doc.tags == []
        assert doc.chunks == []

    def test_parsed_document_mutable(self):
        """ParsedDocument is mutable (indexer clears chunks/raw_body after processing)."""
        from kb.types import ParsedDocument

        doc = ParsedDocument(
            path="test.md",
            title="T",
            date=None,
            doc_type="notes",
            source_system="memory",
            source_id="test",
        )
        doc.raw_body = None
        doc.chunks = []

    def test_index_result_construction(self):
        from kb.types import IndexResult

        r = IndexResult()
        assert r.documents_indexed == 0
        assert r.errors == []


class TestSearchModels:
    def test_search_result_construction(self):
        from kb.types import SearchResult

        r = SearchResult(
            chunk_id=1,
            document_id=10,
            title="Test Meeting",
            path="meetings/2026/01/01/test.md",
            date="2026-01-01",
            doc_type="notes",
            score=0.85,
            section="Introduction",
            snippet="This is a test...",
            entities=["Jane Doe"],
        )
        assert r.score == 0.85
        assert r.entities == ["Jane Doe"]

    def test_search_response(self):
        from kb.types import SearchMeta, SearchResponse, SearchResult

        meta = SearchMeta(query="test", total=1, limit=10, sort_by="score", execution_ms=5.0)
        r = SearchResult(
            chunk_id=1,
            document_id=10,
            title="T",
            path="p",
            date=None,
            doc_type="notes",
            score=0.5,
            section=None,
            snippet="s",
            entities=[],
        )
        resp = SearchResponse(results=[r], meta=meta)
        assert resp.meta.total == 1
        assert len(resp.results) == 1


class TestContextModels:
    def test_context_stats(self):
        from kb.types import ContextStats, DateRange

        dr = DateRange(earliest="2025-01-01", latest="2026-01-01")
        s = ContextStats(documents=100, entities=50, date_range=dr)
        assert s.documents == 100

    def test_context_entity_summary(self):
        from kb.types import ContextEntitySummary

        s = ContextEntitySummary(name="Jane", entity_type="person", mention_count=10)
        assert s.mention_count == 10

    def test_context_output(self):
        from kb.types import ContextOutput, ContextStats, DateRange

        co = ContextOutput(
            text="[KB:50 entities]",
            stats=ContextStats(
                documents=100,
                entities=50,
                date_range=DateRange(earliest=None, latest=None),
            ),
            entities=[],
        )
        assert co.text.startswith("[KB:")


class TestGlossaryModel:
    def test_glossary_entry(self):
        from kb.types import GlossaryEntry

        g = GlossaryEntry(term="AC", expansion="Lattice Co", section="Acronyms")
        assert g.term == "AC"

    def test_glossary_entry_frozen(self):
        from kb.types import GlossaryEntry

        g = GlossaryEntry(term="AC", expansion="Lattice Co", section="Acronyms")
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            g.term = "XX"


class TestCrudModels:
    def test_entity_type_config(self):
        from kb.types import EntityTypeConfig

        c = EntityTypeConfig(
            directory="memory/people",
            alias_label="Also known as",
            fields=[("Email", "email"), ("Role", "role")],
        )
        assert c.directory == "memory/people"
        assert len(c.fields) == 2


class TestChunkerPydantic:
    """Verify that chunker.py re-exports Pydantic models (not dataclasses)."""

    def test_chunk_is_pydantic(self):
        from pydantic import BaseModel

        from kb.chunker import Chunk

        assert issubclass(Chunk, BaseModel)

    def test_parsed_document_is_pydantic(self):
        from pydantic import BaseModel

        from kb.chunker import ParsedDocument

        assert issubclass(ParsedDocument, BaseModel)


class TestIndexerPydantic:
    """Verify that indexer.py re-exports Pydantic IndexResult (not dataclass)."""

    def test_index_result_is_pydantic(self):
        from pydantic import BaseModel

        from kb.indexer import IndexResult

        assert issubclass(IndexResult, BaseModel)
