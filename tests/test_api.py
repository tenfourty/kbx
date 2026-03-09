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
    """Create a KnowledgeBase instance with isolated data dir.

    Embedder is disabled (text-only indexing) so tests never hit the network.
    """
    from kb.api import KnowledgeBase

    data_dir = tmp_path / "data"
    kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
    kb._embedder_failed = True  # force text-only indexing
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

    def test_get_entity_facts_include_id(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()
        result = kb_with_entities.get_entity("Talia Ström")
        assert result is not None
        fact = result.facts[0]
        assert isinstance(fact.id, int)
        assert fact.id > 0

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

    def test_edit_glossary_term(self, kb_instance, project_root):
        (project_root / "memory" / "glossary.md").write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n"
            "|------|----------|\n| API | Application Programming Interface |\n"
        )
        result = kb_instance.edit_glossary_term("API", "App Programming Interface")
        assert result["term"] == "API"
        assert result["expansion"] == "App Programming Interface"
        # Verify it persisted
        terms = kb_instance.list_glossary_terms()
        assert terms[0].expansion == "App Programming Interface"

    def test_edit_glossary_term_not_found(self, kb_instance, project_root):
        with pytest.raises(ValueError, match="Term not found"):
            kb_instance.edit_glossary_term("NOPE", "whatever")


class TestNoteDelete:
    def test_delete_note(self, kb_instance, project_root):
        """delete_note removes file + DB records."""
        notes_dir = project_root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "2026-02-28-test.md").write_text(
            "---\ntitle: Test Note\ndate: 2026-02-28\n---\nBody here\n"
        )
        kb_instance.index()
        assert kb_instance.count_documents() >= 1

        result = kb_instance.delete_note("memory/notes/2026-02-28-test.md")
        assert result["deleted"] is True
        assert result["title"] == "Test Note"
        assert not (notes_dir / "2026-02-28-test.md").exists()

    def test_delete_note_not_found(self, kb_instance):
        with pytest.raises(ValueError, match="not found"):
            kb_instance.delete_note("memory/notes/nonexistent.md")

    def test_delete_note_refuses_non_note(self, kb_with_entities):
        """Cannot delete meeting docs via delete_note."""
        with pytest.raises(ValueError, match="Not a memory note"):
            kb_with_entities.delete_note("meetings/standup.md")

    def test_delete_note_cleans_db(self, kb_instance, project_root):
        """All DB records (documents, chunks, entity_mentions) are cleaned."""
        notes_dir = project_root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "2026-02-28-cleanup.md").write_text(
            "---\ntitle: Cleanup Test\ndate: 2026-02-28\n---\nSome content\n"
        )
        kb_instance.index()

        conn = kb_instance._get_conn()
        doc = conn.execute(
            "SELECT id FROM documents WHERE path = 'memory/notes/2026-02-28-cleanup.md'"
        ).fetchone()
        assert doc is not None
        doc_id = doc["id"]

        # Verify chunks exist
        chunks = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchone()
        assert chunks["cnt"] > 0

        kb_instance.delete_note("memory/notes/2026-02-28-cleanup.md")

        # Verify everything is gone
        assert (
            conn.execute(
                "SELECT COUNT(*) AS cnt FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()["cnt"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) AS cnt FROM chunks WHERE document_id = ?", (doc_id,)
            ).fetchone()["cnt"]
            == 0
        )


class TestFactDelete:
    def test_delete_fact_by_id(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Joined Platform team', '2025-06-01')"
        )
        conn.commit()

        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Promoted to Lead'"
        ).fetchone()
        fact_id = fact_row["id"]

        result = kb_with_entities.delete_fact(fact_id)
        assert result["deleted"] is True
        assert result["fact_text"] == "Promoted to Lead"

        # Verify fact is gone from DB
        remaining = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        assert remaining is None

        # Other fact still exists
        other = conn.execute(
            "SELECT * FROM facts WHERE fact_text = 'Joined Platform team'"
        ).fetchone()
        assert other is not None

    def test_delete_fact_not_found(self, kb_instance):
        with pytest.raises(ValueError, match="Fact not found"):
            kb_instance.delete_fact(99999)

    def test_delete_fact_removes_from_entity_file(self, kb_with_entities, project_root):
        """If entity has a source file with ## Recent Facts, the line is removed."""
        people_dir = project_root / "memory" / "people"
        people_dir.mkdir(parents=True, exist_ok=True)
        (people_dir / "eve.md").write_text(
            "# Talia Ström\n\n**Role:** Engineering Leader\n\n"
            "## Recent Facts\n"
            "- [2026-01-15] Promoted to Lead\n"
            "- [2025-06-01] Joined Platform team\n"
        )

        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute("UPDATE entities SET source_path = 'memory/people/eve.md' WHERE id = 1")
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()

        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Promoted to Lead'"
        ).fetchone()

        kb_with_entities.delete_fact(fact_row["id"])

        content = (people_dir / "eve.md").read_text()
        assert "Promoted to Lead" not in content
        assert "Joined Platform team" in content


class TestFactEdit:
    def test_edit_fact_text(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()

        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Promoted to Lead'"
        ).fetchone()

        result = kb_with_entities.edit_fact(fact_row["id"], text="Promoted to Staff")
        assert result["fact_text"] == "Promoted to Staff"

        updated = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_row["id"],)).fetchone()
        assert updated["fact_text"] == "Promoted to Staff"

    def test_edit_fact_date(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()
        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Promoted to Lead'"
        ).fetchone()

        result = kb_with_entities.edit_fact(fact_row["id"], date="2026-02-01")
        assert result["fact_date"] == "2026-02-01"

    def test_edit_fact_not_found(self, kb_instance):
        with pytest.raises(ValueError, match="Fact not found"):
            kb_instance.edit_fact(99999, text="new text")

    def test_edit_fact_no_changes(self, kb_with_entities):
        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Some fact', '2026-01-15')"
        )
        conn.commit()
        fact_row = conn.execute("SELECT id FROM facts WHERE fact_text = 'Some fact'").fetchone()
        with pytest.raises(ValueError, match="No changes"):
            kb_with_entities.edit_fact(fact_row["id"])

    def test_edit_fact_updates_entity_file(self, kb_with_entities, project_root):
        """If entity has a source file, the fact line is updated."""
        people_dir = project_root / "memory" / "people"
        people_dir.mkdir(parents=True, exist_ok=True)
        (people_dir / "eve.md").write_text(
            "# Talia Ström\n\n**Role:** Engineering Leader\n\n"
            "## Recent Facts\n"
            "- [2026-01-15] Promoted to Lead\n"
        )

        conn = kb_with_entities._db.get_sqlite_conn()
        conn.execute("UPDATE entities SET source_path = 'memory/people/eve.md' WHERE id = 1")
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()

        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Promoted to Lead'"
        ).fetchone()

        kb_with_entities.edit_fact(fact_row["id"], text="Promoted to Staff")

        content = (people_dir / "eve.md").read_text()
        assert "Promoted to Staff" in content
        assert "Promoted to Lead" not in content


class TestSearch:
    def test_search_empty_db(self, kb_instance):
        result = kb_instance.search("anything", fast=True)
        assert result.results == []
        assert result.meta.query == "anything"

    def test_search_returns_search_response(self, kb_instance):
        from kb.types import SearchResponse

        result = kb_instance.search("test", fast=True)
        assert isinstance(result, SearchResponse)

    def test_search_accepts_fts_weight(self, kb_instance):
        result = kb_instance.search("test", fast=True, fts_weight=2.0)
        assert result.meta.query == "test"

    def test_search_accepts_vector_weight(self, kb_instance):
        result = kb_instance.search("test", fast=True, vector_weight=0.5)
        assert result.meta.query == "test"

    def test_search_accepts_dedupe(self, kb_instance):
        result = kb_instance.search("test", fast=True, dedupe=True)
        assert result.meta.query == "test"

    def test_search_passes_params_through(self, kb_instance, monkeypatch):
        """Verify fts_weight, vector_weight, dedupe are forwarded to internal search."""
        captured: dict = {}

        def fake_search(db, embedder, query, **kwargs):
            captured.update(kwargs)
            from kb.types import SearchMeta, SearchResponse

            return SearchResponse(
                results=[],
                meta=SearchMeta(query=query, total=0, limit=10, sort_by="score", execution_ms=0),
            )

        import kb.search

        monkeypatch.setattr(kb.search, "search", fake_search)

        kb_instance.search("test", fast=True, fts_weight=2.5, vector_weight=0.3, dedupe=True)
        assert captured["fts_weight"] == 2.5
        assert captured["vector_weight"] == 0.3
        assert captured["dedupe"] is True


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


class TestWarmEmbedder:
    def test_warm_embedder_loads_model(self, kb_instance, monkeypatch):
        """warm_embedder() returns True and calls embed_query on the embedder."""
        calls: list[str] = []

        class FakeEmbedder:
            def embed_query(self, text: str):
                calls.append(text)
                import numpy as np

                return np.zeros((1, 1024), dtype=np.float32)

            def release_gpu_memory(self) -> None:
                pass

        monkeypatch.setattr(kb_instance, "_get_embedder", lambda: FakeEmbedder())
        result = kb_instance.warm_embedder()
        assert result is True
        assert calls == ["warmup"]

    def test_warm_embedder_returns_false_when_no_embed(self, kb_instance, monkeypatch):
        """warm_embedder() returns False when embedder is unavailable."""
        monkeypatch.setattr(kb_instance, "_get_embedder", lambda: None)
        result = kb_instance.warm_embedder()
        assert result is False

    def test_warm_embedder_does_not_trigger_reindex(self, kb_instance, monkeypatch):
        """warm_embedder() must NOT call auto_reindex_if_stale."""
        reindex_called = False

        def fake_reindex(*args, **kwargs):
            nonlocal reindex_called
            reindex_called = True

        monkeypatch.setattr("kb.staleness.auto_reindex_if_stale", fake_reindex)
        # Ensure embedder is unavailable so we don't need real model
        monkeypatch.setattr(kb_instance, "_get_embedder", lambda: None)
        kb_instance.warm_embedder()
        assert reindex_called is False


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


class TestPostMutationReindex:
    """Mutations that change memory files should update the index."""

    def test_delete_note_cleans_document_attendees(self, kb_instance, project_root):
        """delete_note removes document_attendees rows too."""
        notes_dir = project_root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "2026-02-28-att.md").write_text(
            "---\ntitle: Attendee Test\ndate: 2026-02-28\n---\nBody\n"
        )
        kb_instance.index()

        conn = kb_instance._get_conn()
        doc = conn.execute(
            "SELECT id FROM documents WHERE path = 'memory/notes/2026-02-28-att.md'"
        ).fetchone()
        assert doc is not None
        doc_id = doc["id"]

        # Manually insert a document_attendees row
        conn.execute(
            "INSERT INTO document_attendees (document_id, entity_id, name, email, matched_by)"
            " VALUES (?, 1, 'Test', 'test@example.com', 'email')",
            (doc_id,),
        )
        conn.commit()

        kb_instance.delete_note("memory/notes/2026-02-28-att.md")

        # document_attendees should be cleaned
        cnt = conn.execute(
            "SELECT COUNT(*) AS cnt FROM document_attendees WHERE document_id = ?",
            (doc_id,),
        ).fetchone()["cnt"]
        assert cnt == 0

    def test_delete_note_cleans_fts(self, kb_instance, project_root):
        """delete_note removes content from FTS index."""
        notes_dir = project_root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "2026-02-28-fts.md").write_text(
            "---\ntitle: FTS Cleanup\ndate: 2026-02-28\n---\nXylophone unique word\n"
        )
        kb_instance.index()

        conn = kb_instance._get_conn()
        # Verify FTS has the content
        fts_before = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'Xylophone'"
        ).fetchone()["cnt"]
        assert fts_before > 0

        kb_instance.delete_note("memory/notes/2026-02-28-fts.md")

        # FTS should be clean (triggers fire on chunk DELETE)
        fts_after = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'Xylophone'"
        ).fetchone()["cnt"]
        assert fts_after == 0

    def test_edit_glossary_reindexes(self, kb_instance, project_root):
        """edit_glossary_term updates the FTS index with new expansion."""
        (project_root / "memory" / "glossary.md").write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n"
            "|------|----------|\n| API | Application Programming Interface |\n"
        )
        kb_instance.index()

        conn = kb_instance._get_conn()
        # Verify old expansion is searchable
        fts_old = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'Application'"
        ).fetchone()["cnt"]
        assert fts_old > 0

        # Edit the term
        kb_instance.edit_glossary_term("API", "App Prog Iface")

        # New expansion should be searchable
        fts_new = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'Iface'"
        ).fetchone()["cnt"]
        assert fts_new > 0

        # Old expansion should be gone
        fts_old_after = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'Application'"
        ).fetchone()["cnt"]
        assert fts_old_after == 0

    def test_edit_fact_reindexes_source_file(self, kb_instance, project_root):
        """edit_fact re-indexes the entity's source file."""
        people_dir = project_root / "memory" / "people"
        people_dir.mkdir(parents=True, exist_ok=True)
        (people_dir / "jane.md").write_text(
            "# Jane\n\n**Role:** Engineer\n\n## Recent Facts\n- [2026-01-15] Promoted to Lead\n"
        )
        kb_instance.index()

        conn = kb_instance._get_conn()
        # Seed entity with source_path
        conn.execute(
            "INSERT OR IGNORE INTO entities (name, entity_type, aliases, metadata, source_path)"
            " VALUES ('Jane', 'person', '[]', '{}', 'memory/people/jane.md')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES ((SELECT id FROM entities WHERE name = 'Jane'), 'Promoted to Lead', '2026-01-15')"
        )
        conn.commit()

        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Promoted to Lead'"
        ).fetchone()

        # Verify old text in FTS
        fts_before = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'Promoted'"
        ).fetchone()["cnt"]
        assert fts_before > 0

        kb_instance.edit_fact(fact_row["id"], text="Became Staff Engineer")

        # New text should be in FTS
        fts_after = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH '\"Staff Engineer\"'"
        ).fetchone()["cnt"]
        assert fts_after > 0

    def test_delete_fact_reindexes_source_file(self, kb_instance, project_root):
        """delete_fact re-indexes the entity's source file."""
        people_dir = project_root / "memory" / "people"
        people_dir.mkdir(parents=True, exist_ok=True)
        (people_dir / "bob.md").write_text(
            "# Soren\n\n**Role:** SRE\n\n## Recent Facts\n- [2026-01-15] Deployed zephyr service\n"
        )
        kb_instance.index()

        conn = kb_instance._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO entities (name, entity_type, aliases, metadata, source_path)"
            " VALUES ('Soren', 'person', '[]', '{}', 'memory/people/bob.md')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES ((SELECT id FROM entities WHERE name = 'Soren'), 'Deployed zephyr service', '2026-01-15')"
        )
        conn.commit()

        fact_row = conn.execute(
            "SELECT id FROM facts WHERE fact_text = 'Deployed zephyr service'"
        ).fetchone()

        kb_instance.delete_fact(fact_row["id"])

        # "zephyr" should be gone from FTS (the fact line was removed from file, file re-indexed)
        fts_after = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'zephyr'"
        ).fetchone()["cnt"]
        assert fts_after == 0


class TestIndex:
    def test_index_with_glossary(self, kb_instance):
        # project_root fixture creates memory/glossary.md, so indexer finds it
        result = kb_instance.index()
        assert result.documents_indexed >= 0

    def test_index_returns_index_result(self, kb_instance):
        from kb.types import IndexResult

        result = kb_instance.index()
        assert isinstance(result, IndexResult)
