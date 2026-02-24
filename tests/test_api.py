"""Tests for the KnowledgeBase API."""

from __future__ import annotations

import pytest


@pytest.fixture
def project_root(tmp_path):
    """Create a minimal project root with memory/ directory."""
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "glossary.md").write_text("# Glossary\n")
    return tmp_path


@pytest.fixture
def kb_instance(tmp_path, project_root):
    """Create a KnowledgeBase instance with isolated data dir."""
    from kb.api import KnowledgeBase

    data_dir = tmp_path / "data"
    kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
    yield kb
    kb.close()


class TestLifecycle:
    """Constructor, close, context manager."""

    def test_constructor_creates_db(self, tmp_path, project_root):
        from kb.api import KnowledgeBase

        data_dir = tmp_path / "data"
        kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
        assert (data_dir / "metadata.db").exists()
        kb.close()

    def test_context_manager(self, tmp_path, project_root):
        from kb.api import KnowledgeBase

        data_dir = tmp_path / "data"
        with KnowledgeBase(project_root=project_root, data_dir=data_dir) as kb:
            assert (data_dir / "metadata.db").exists()
        # After exit, db should be closed (no error on double-close)
        kb.close()

    def test_thread_safe_mode(self, tmp_path, project_root):
        from kb.api import KnowledgeBase

        data_dir = tmp_path / "data"
        kb = KnowledgeBase(project_root=project_root, data_dir=data_dir, thread_safe=True)
        conn = kb._db.get_sqlite_conn()
        # Verify WAL mode is enabled
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        kb.close()

    def test_import_from_package(self):
        from kb import KnowledgeBase

        assert KnowledgeBase is not None

    def test_count_documents_empty(self, kb_instance):
        assert kb_instance.count_documents() == 0


@pytest.fixture
def kb_with_entities(kb_instance):
    """Seed some test entities into the database."""
    conn = kb_instance._db.get_sqlite_conn()
    conn.execute(
        "INSERT INTO entities (name, entity_type, aliases, metadata, source_path)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            "Talia Ström",
            "person",
            '["Talia"]',
            '{"role": "Engineering Leader", "team": "Platform"}',
            "memory/people/eve.md",
        ),
    )
    conn.execute(
        "INSERT INTO entities (name, entity_type, aliases, metadata, source_path)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            "Helix Refactor",
            "project",
            '["helix-refactor"]',
            '{"status": "In Progress"}',
            None,
        ),
    )
    conn.execute(
        "INSERT INTO documents (path, title, doc_date, doc_type, source_system,"
        " source_id, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "meetings/standup.md",
            "Standup",
            "2026-02-20",
            "meeting",
            "granola",
            "abc",
            "hash1",
        ),
    )
    conn.execute(
        "INSERT INTO entity_mentions (entity_id, document_id, mention_type)"
        " VALUES (1, 1, 'discussed')"
    )
    conn.execute(
        "INSERT INTO entity_mentions (entity_id, document_id, mention_type)"
        " VALUES (1, 1, 'participant')"
    )
    conn.commit()
    return kb_instance


class TestEntityOperations:
    """list_entities, get_entity, find_entities."""

    def test_list_entities_all(self, kb_with_entities):
        result = kb_with_entities.list_entities()
        assert len(result) == 2
        names = {e.name for e in result}
        assert "Talia Ström" in names
        assert "Helix Refactor" in names

    def test_list_entities_by_type(self, kb_with_entities):
        result = kb_with_entities.list_entities(entity_type="person")
        assert len(result) == 1
        assert result[0].name == "Talia Ström"
        assert result[0].mention_count == 2

    def test_list_entities_pinned_first(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Helix Refactor'")
        conn.commit()
        result = kb_with_entities.list_entities()
        assert result[0].name == "Helix Refactor"
        assert result[0].pinned is True

    def test_list_entities_returns_entity_summary(self, kb_with_entities):
        from kb.types import EntitySummary

        result = kb_with_entities.list_entities()
        assert isinstance(result[0], EntitySummary)

    def test_get_entity_found(self, kb_with_entities):
        result = kb_with_entities.get_entity("Talia Ström")
        assert result is not None
        assert result.name == "Talia Ström"
        assert result.metadata["role"] == "Engineering Leader"
        assert result.aliases == ["Talia"]
        assert result.mention_count == 2

    def test_get_entity_not_found(self, kb_with_entities):
        result = kb_with_entities.get_entity("Nobody")
        assert result is None

    def test_get_entity_case_insensitive(self, kb_with_entities):
        result = kb_with_entities.get_entity("eve perrin")
        assert result is not None
        assert result.name == "Talia Ström"

    def test_get_entity_returns_entity_detail(self, kb_with_entities):
        from kb.types import EntityDetail

        result = kb_with_entities.get_entity("Talia Ström")
        assert isinstance(result, EntityDetail)

    def test_get_entity_with_facts(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()
        result = kb_with_entities.get_entity("Talia Ström")
        assert result is not None
        assert len(result.facts) == 1
        assert result.facts[0].text == "Promoted to Lead"

    def test_find_entities_exact(self, kb_with_entities):
        result = kb_with_entities.find_entities("Talia Ström")
        assert len(result) == 1
        assert result[0].name == "Talia Ström"

    def test_find_entities_alias(self, kb_with_entities):
        result = kb_with_entities.find_entities("Talia")
        assert len(result) == 1
        assert result[0].name == "Talia Ström"

    def test_find_entities_partial(self, kb_with_entities):
        result = kb_with_entities.find_entities("Perrin")
        assert len(result) == 1

    def test_find_entities_no_match(self, kb_with_entities):
        result = kb_with_entities.find_entities("zzz_nonexistent")
        assert result == []


class TestEntityTimeline:
    def test_timeline_returns_entries(self, kb_with_entities):
        result = kb_with_entities.get_entity_timeline("Talia Ström")
        assert len(result) == 1
        assert result[0].title == "Standup"
        assert result[0].date == "2026-02-20"

    def test_timeline_by_alias(self, kb_with_entities):
        result = kb_with_entities.get_entity_timeline("Talia")
        assert len(result) == 1

    def test_timeline_not_found(self, kb_with_entities):
        result = kb_with_entities.get_entity_timeline("Nobody")
        assert result == []

    def test_timeline_respects_limit(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        for i in range(5):
            conn.execute(
                "INSERT INTO documents (path, title, doc_date, doc_type, source_system,"
                " source_id, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"meetings/m{i}.md",
                    f"Meeting {i}",
                    f"2026-02-{10 + i:02d}",
                    "meeting",
                    "granola",
                    f"id{i}",
                    f"hash{i}",
                ),
            )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type)"
                f" VALUES (1, {i + 2}, 'discussed')"
            )
        conn.commit()
        result = kb_with_entities.get_entity_timeline("Talia", limit=3)
        assert len(result) == 3

    def test_timeline_returns_timeline_entry(self, kb_with_entities):
        from kb.types import TimelineEntry

        result = kb_with_entities.get_entity_timeline("Talia")
        assert isinstance(result[0], TimelineEntry)


class TestEntityPin:
    def test_toggle_pin_on(self, kb_with_entities):
        result = kb_with_entities.toggle_entity_pin("Talia Ström")
        assert result.pinned is True
        assert result.name == "Talia Ström"

    def test_toggle_pin_off(self, kb_with_entities):
        kb_with_entities.toggle_entity_pin("Talia Ström")  # on
        result = kb_with_entities.toggle_entity_pin("Talia Ström")  # off
        assert result.pinned is False

    def test_toggle_pin_by_alias(self, kb_with_entities):
        result = kb_with_entities.toggle_entity_pin("Talia")
        assert result.pinned is True
        assert result.name == "Talia Ström"

    def test_toggle_pin_not_found(self, kb_with_entities):
        with pytest.raises(ValueError, match="Entity not found"):
            kb_with_entities.toggle_entity_pin("Nobody")


class TestDocumentOperations:
    def test_get_document_pin_false(self, kb_with_entities):
        assert kb_with_entities.get_document_pin("meetings/standup.md") is False

    def test_toggle_document_pin(self, kb_with_entities):
        result = kb_with_entities.toggle_document_pin("meetings/standup.md")
        assert result.pinned is True
        assert result.path == "meetings/standup.md"

    def test_toggle_document_pin_off(self, kb_with_entities):
        kb_with_entities.toggle_document_pin("meetings/standup.md")  # on
        result = kb_with_entities.toggle_document_pin("meetings/standup.md")  # off
        assert result.pinned is False

    def test_toggle_document_pin_not_found(self, kb_with_entities):
        with pytest.raises(ValueError, match="Document not found"):
            kb_with_entities.toggle_document_pin("nonexistent.md")

    def test_list_pinned_documents_empty(self, kb_instance):
        result = kb_instance.list_pinned_documents()
        assert result == []

    def test_list_pinned_documents(self, kb_with_entities):
        kb_with_entities.toggle_document_pin("meetings/standup.md")
        result = kb_with_entities.list_pinned_documents()
        assert len(result) == 1
        assert result[0].path == "meetings/standup.md"
        assert result[0].title == "Standup"

    def test_count_documents(self, kb_with_entities):
        assert kb_with_entities.count_documents() == 1


class TestMemoryOperations:
    def test_read_memory_file(self, kb_instance, project_root):
        (project_root / "memory" / "notes").mkdir(parents=True, exist_ok=True)
        (project_root / "memory" / "notes" / "test.md").write_text("# Test\nHello")
        result = kb_instance.read_memory_file("notes/test.md")
        assert result == "# Test\nHello"

    def test_read_memory_file_not_found(self, kb_instance):
        result = kb_instance.read_memory_file("notes/nonexistent.md")
        assert result is None

    def test_read_memory_file_traversal_blocked(self, kb_instance):
        result = kb_instance.read_memory_file("../../../etc/passwd")
        assert result is None

    def test_read_memory_file_non_md_blocked(self, kb_instance):
        result = kb_instance.read_memory_file("notes/test.txt")
        assert result is None

    def test_write_memory_file(self, kb_instance, project_root):
        (project_root / "memory" / "notes").mkdir(parents=True, exist_ok=True)
        (project_root / "memory" / "notes" / "test.md").write_text("old")
        ok = kb_instance.write_memory_file("notes/test.md", "new content")
        assert ok is True
        assert (project_root / "memory" / "notes" / "test.md").read_text() == "new content"

    def test_write_memory_file_no_create(self, kb_instance):
        ok = kb_instance.write_memory_file("notes/new.md", "content")
        assert ok is False

    def test_list_memory_tree(self, kb_instance, project_root):
        from kb.types import MemoryTreeNode

        (project_root / "memory" / "notes").mkdir(parents=True, exist_ok=True)
        (project_root / "memory" / "notes" / "a.md").write_text("# A")
        (project_root / "memory" / "notes" / "b.md").write_text("# B")
        result = kb_instance.list_memory_tree()
        assert len(result) >= 1
        notes = [n for n in result if n.name == "notes"]
        assert len(notes) == 1
        assert notes[0].node_type == "dir"
        assert notes[0].count == 2
        assert isinstance(notes[0], MemoryTreeNode)

    def test_list_memory_tree_pinned_state(self, kb_instance, project_root):
        (project_root / "memory" / "notes").mkdir(parents=True, exist_ok=True)
        (project_root / "memory" / "notes" / "pinned.md").write_text("# Pinned")
        conn = kb_instance._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO documents (path, title, doc_date, doc_type, source_system,"
            " source_id, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "memory/notes/pinned.md",
                "Pinned",
                None,
                "memory",
                "memory",
                "memory/notes/pinned.md",
                "hash",
            ),
        )
        conn.execute("UPDATE documents SET pinned = 1 WHERE path = 'memory/notes/pinned.md'")
        conn.commit()
        result = kb_instance.list_memory_tree()
        notes = next(n for n in result if n.name == "notes")
        pinned_file = next(c for c in notes.children if c.name == "pinned.md")
        assert pinned_file.pinned is True


class TestGlossary:
    def test_list_glossary_terms_empty(self, kb_instance):
        result = kb_instance.list_glossary_terms()
        assert result == []

    def test_list_glossary_terms(self, kb_instance, project_root):
        (project_root / "memory" / "glossary.md").write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n"
            "|------|----------|\n| API | Application Programming Interface |\n"
        )
        result = kb_instance.list_glossary_terms()
        assert len(result) == 1
        assert result[0].term == "API"

    def test_list_glossary_returns_glossary_entry(self, kb_instance, project_root):
        from kb.types import GlossaryEntry

        (project_root / "memory" / "glossary.md").write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n"
            "|------|----------|\n| API | Application Programming Interface |\n"
        )
        result = kb_instance.list_glossary_terms()
        assert isinstance(result[0], GlossaryEntry)


class TestSearch:
    def test_search_empty_db(self, kb_instance):
        result = kb_instance.search("anything", fast=True)
        assert result.results == []
        assert result.meta.query == "anything"

    def test_search_returns_search_response(self, kb_instance):
        from kb.types import SearchResponse

        result = kb_instance.search("test", fast=True)
        assert isinstance(result, SearchResponse)


class TestContext:
    def test_context_empty_db(self, kb_instance):
        result = kb_instance.context()
        assert result.text is not None
        assert result.stats.documents == 0

    def test_context_human_format(self, kb_instance):
        result = kb_instance.context(fmt="human")
        assert isinstance(result.text, str)

    def test_context_returns_context_output(self, kb_instance):
        from kb.types import ContextOutput

        result = kb_instance.context()
        assert isinstance(result, ContextOutput)


class TestFactCounts:
    """get_fact_counts() batch method."""

    def test_empty_db_returns_empty_dict(self, kb_instance):
        result = kb_instance.get_fact_counts()
        assert result == {}

    def test_counts_multiple_entities(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        # Add 2 facts for entity 1 and 1 fact for entity 2
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Joined Platform team', '2025-06-01')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (2, 'Migration kicked off', '2026-02-01')"
        )
        conn.commit()
        result = kb_with_entities.get_fact_counts()
        assert result == {1: 2, 2: 1}

    def test_filtered_by_entity_ids(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (2, 'Migration kicked off', '2026-02-01')"
        )
        conn.commit()
        # Only ask for entity 1
        result = kb_with_entities.get_fact_counts(entity_ids=[1])
        assert result == {1: 1}
        assert 2 not in result

    def test_entities_with_no_facts_absent(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        # Only add facts for entity 1, not entity 2
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()
        result = kb_with_entities.get_fact_counts()
        assert 1 in result
        assert 2 not in result  # absent, not zero

    def test_filtered_empty_list_returns_empty(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()
        result = kb_with_entities.get_fact_counts(entity_ids=[])
        assert result == {}


class TestIndex:
    def test_index_with_glossary(self, kb_instance):
        # project_root fixture creates memory/glossary.md, so indexer finds it
        result = kb_instance.index()
        assert result.documents_indexed >= 0

    def test_index_returns_index_result(self, kb_instance):
        from kb.types import IndexResult

        result = kb_instance.index()
        assert isinstance(result, IndexResult)
