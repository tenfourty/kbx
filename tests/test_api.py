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
