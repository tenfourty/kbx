"""Extra tests targeting specific uncovered lines across modules to push coverage toward 90%."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import invoke_cli

from kb.db import Database

# ---------------------------------------------------------------------------
# search.py gaps: _is_strong_signal, _build_fts_query, _sanitize_fts_input
# ---------------------------------------------------------------------------


class TestIsStrongSignal:
    def test_empty_results(self):
        from kb.search import _is_strong_signal

        assert _is_strong_signal([]) is False

    def test_single_result(self):
        from kb.search import _is_strong_signal

        results = [{"bm25_score": 1.0}]
        assert _is_strong_signal(results) is True

    def test_two_results(self):
        from kb.search import _is_strong_signal

        results = [{"bm25_score": 1.0}, {"bm25_score": 0.5}]
        assert _is_strong_signal(results) is True

    def test_many_results_with_large_gap(self):
        from kb.search import _is_strong_signal

        results = [
            {"bm25_score": 1.0},
            {"bm25_score": 0.8},
            {"bm25_score": 0.5},
            {"bm25_score": 0.3},
        ]
        assert _is_strong_signal(results) is True

    def test_many_results_no_gap(self):
        from kb.search import _is_strong_signal

        results = [
            {"bm25_score": 1.0},
            {"bm25_score": 0.95},
            {"bm25_score": 0.9},
            {"bm25_score": 0.85},
        ]
        assert _is_strong_signal(results) is False

    def test_low_top_score(self):
        from kb.search import _is_strong_signal

        results = [
            {"bm25_score": 0.8},
            {"bm25_score": 0.5},
            {"bm25_score": 0.3},
        ]
        assert _is_strong_signal(results) is False


class TestBuildFtsQuery:
    def test_single_word(self):
        from kb.search import _build_fts_query

        variants = _build_fts_query("MFA")
        assert variants == ["MFA"]

    def test_multi_word(self):
        from kb.search import _build_fts_query

        variants = _build_fts_query("Helix refactor")
        assert len(variants) == 4
        assert variants[0] == '"Helix refactor"'
        assert "NEAR" in variants[1]
        assert variants[2] == "Helix refactor"  # implicit AND
        assert "OR" in variants[3]

    def test_empty_after_sanitize(self):
        from kb.search import _build_fts_query

        variants = _build_fts_query("AND OR NOT")
        # After sanitizing FTS operators, the query is empty
        assert len(variants) >= 1


class TestSanitizeFtsInput:
    def test_strips_operators(self):
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input("hello AND world")
        assert "AND" not in result
        assert "hello" in result
        assert "world" in result

    def test_strips_special_chars(self):
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input('test*"query^')
        assert "*" not in result
        assert '"' not in result
        assert "^" not in result

    def test_collapses_whitespace(self):
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input("  hello   world  ")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# search.py: _enrich_results with empty input
# ---------------------------------------------------------------------------


class TestEnrichResults:
    def test_empty_chunk_ids(self):
        from kb.search import _enrich_results

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            result = _enrich_results(db, [])
            assert result == {}
            db.close()


# ---------------------------------------------------------------------------
# CLI: glossary commands
# ---------------------------------------------------------------------------


@pytest.fixture
def glossary_env(tmp_path):
    """Create project root with glossary file and a DB."""
    glossary = tmp_path / "memory" / "glossary.md"
    glossary.parent.mkdir(parents=True)
    glossary.write_text(
        "# Glossary\n\n## Acronyms\n\n"
        "| Term | Expansion | Notes |\n"
        "|------|-----------|-------|\n"
        "| AC | Lattice Co | Company |\n"
    )
    # Create memory dirs for entity create
    (tmp_path / "memory" / "people").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "projects").mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    db = Database(data_dir)
    yield db, tmp_path, data_dir
    db.close()


class TestGlossaryCommands:
    def test_glossary_list_json(self, runner, glossary_env):
        """kb glossary list --json returns terms."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            result = invoke_cli(runner, ["glossary", "list", "--json"], str(data_dir))
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert any(t["term"] == "AC" for t in data["results"])

    def test_glossary_add_json(self, runner, glossary_env):
        """kb glossary add creates a term."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            result = invoke_cli(
                runner,
                ["glossary", "add", "MR", "Merge Request", "--json"],
                str(data_dir),
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["term"] == "MR"

    def test_glossary_delete(self, runner, glossary_env):
        """kb glossary delete removes a term."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            result = invoke_cli(runner, ["glossary", "delete", "AC"], str(data_dir))
        assert result.exit_code == 0
        assert "Deleted" in result.output


# ---------------------------------------------------------------------------
# CLI: person/project edit and delete
# ---------------------------------------------------------------------------


class TestEntityEditDelete:
    def test_person_edit_role(self, runner, glossary_env):
        """kb person edit --set role='New Role' updates entity."""
        _, root, data_dir = glossary_env
        # First create a person
        with patch("kb.cli._find_project_root", return_value=root):
            invoke_cli(
                runner,
                ["person", "create", "Test Person", "--role", "Developer", "--json"],
                str(data_dir),
            )
            result = invoke_cli(
                runner,
                ["person", "edit", "Test Person", "--role", "Senior Developer", "--json"],
                str(data_dir),
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Test Person"

    def test_project_edit_status(self, runner, glossary_env):
        """kb project edit --status 'Done' updates project."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            invoke_cli(
                runner,
                ["project", "create", "My Project", "--status", "Active", "--json"],
                str(data_dir),
            )
            result = invoke_cli(
                runner,
                ["project", "edit", "My Project", "--status", "Completed", "--json"],
                str(data_dir),
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "My Project"

    def test_person_delete(self, runner, glossary_env):
        """kb person delete removes entity."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            invoke_cli(
                runner,
                ["person", "create", "Deletable Person", "--json"],
                str(data_dir),
            )
            result = invoke_cli(
                runner,
                ["person", "delete", "Deletable Person", "--yes"],
                str(data_dir),
            )
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_project_delete(self, runner, glossary_env):
        """kb project delete removes entity."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            invoke_cli(
                runner,
                ["project", "create", "Deletable Project", "--json"],
                str(data_dir),
            )
            result = invoke_cli(
                runner,
                ["project", "delete", "Deletable Project", "--yes"],
                str(data_dir),
            )
        assert result.exit_code == 0
        assert "Deleted" in result.output


# ---------------------------------------------------------------------------
# CLI: _fuzzy_suggest, _fuzzy_suggest_entity
# ---------------------------------------------------------------------------


class TestFuzzySuggest:
    def test_substring_match(self):
        from kb.cli import _fuzzy_suggest

        paths = ["a/b/meeting.md", "a/b/notes.md", "a/b/review.md"]
        suggestions = _fuzzy_suggest("meet", paths)
        assert "a/b/meeting.md" in suggestions

    def test_no_match(self):
        from kb.cli import _fuzzy_suggest

        paths = ["a/b/meeting.md"]
        suggestions = _fuzzy_suggest("zzz", paths)
        assert len(suggestions) <= 3


class TestFuzzySuggestEntity:
    def test_exact_substring(self):
        from kb.cli import _fuzzy_suggest_entity

        names = ["Talia Ström", "Soren Vance", "Soren Vance"]
        suggestions = _fuzzy_suggest_entity("Talia", names)
        assert "Talia Ström" in suggestions

    def test_close_typo(self):
        from kb.cli import _fuzzy_suggest_entity

        names = ["Talia Ström", "Soren Vance"]
        suggestions = _fuzzy_suggest_entity("Taliz", names)
        assert "Talia Ström" in suggestions

    def test_no_close_match(self):
        from kb.cli import _fuzzy_suggest_entity

        names = ["Talia Ström"]
        suggestions = _fuzzy_suggest_entity("Zzzzzzzzz", names)
        assert len(suggestions) == 0


# ---------------------------------------------------------------------------
# CLI: _entity_not_found (JSON vs table mode)
# ---------------------------------------------------------------------------


class TestEntityNotFound:
    def test_entity_not_found_table_mode(self, runner, glossary_env):
        """Entity not found in table mode shows text error."""
        _, _root, data_dir = glossary_env
        db = Database(data_dir)
        conn = db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
            ("Talia Ström", "person", '["Talia"]', "{}"),
        )
        conn.commit()
        db.close()

        result = invoke_cli(runner, ["person", "find", "Nonexistent"], str(data_dir))
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: list with --from / --to
# ---------------------------------------------------------------------------


class TestListDateFilters:
    def test_list_with_from_only(self, runner, glossary_env):
        """kb list --from filters by start date."""
        db, _root, data_dir = glossary_env
        conn = db.get_sqlite_conn()
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("a.md", "Doc A", "2026-01-15", "notes", "granola", "[]", "h1", 1),
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("b.md", "Doc B", "2026-02-15", "notes", "granola", "[]", "h2", 1),
        )
        conn.commit()

        result = invoke_cli(runner, ["list", "--from", "2026-02-01", "--json"], str(data_dir))
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            if r.get("date"):
                assert r["date"] >= "2026-02-01"


# ---------------------------------------------------------------------------
# CLI: view with --format jsonl
# ---------------------------------------------------------------------------


class TestSearchFormatJsonl:
    def test_search_format_jsonl(self, runner, glossary_env):
        """kb search --format jsonl returns valid JSONL."""
        db, _, data_dir = glossary_env
        conn = db.get_sqlite_conn()
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("test.md", "Test Doc", "2026-01-15", "notes", "granola", "[]", "testhash", 1),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (1, 0, None, "Some test content about testing."),
        )
        conn.commit()

        result = invoke_cli(
            runner, ["search", "test", "--fast", "--format", "jsonl"], str(data_dir)
        )
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        for line in lines:
            if line.strip():
                json.loads(line)  # should not raise


# ---------------------------------------------------------------------------
# CLI: context --json output format
# ---------------------------------------------------------------------------


class TestContextJsonOutput:
    def test_context_json_has_expected_keys(self, runner, glossary_env):
        """kb context --json returns {text, stats, entities}."""
        _, root, data_dir = glossary_env
        with patch("kb.cli._find_project_root", return_value=root):
            result = invoke_cli(runner, ["context", "--json"], str(data_dir))
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "text" in data
        assert "stats" in data
        assert "entities" in data


# ---------------------------------------------------------------------------
# CLI: search --sort date
# ---------------------------------------------------------------------------


class TestSearchSortDate:
    def test_search_sort_date_json(self, runner, glossary_env):
        """kb search --sort date returns results sorted by date."""
        db, _, data_dir = glossary_env
        conn = db.get_sqlite_conn()
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old.md", "Old Meeting", "2025-01-15", "notes", "granola", "[]", "old1", 1),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (1, 0, None, "Old meeting about refactor."),
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("new.md", "New Meeting", "2026-02-15", "notes", "granola", "[]", "new1", 1),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (2, 0, None, "New meeting about refactor."),
        )
        conn.commit()

        result = invoke_cli(
            runner, ["search", "refactor", "--fast", "--sort", "date", "--json"], str(data_dir)
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        dates = [r["date"] for r in data["results"] if r.get("date")]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# entities.py: _parse_field, _parse_title, _stem_to_first_name
# ---------------------------------------------------------------------------


class TestEntityHelpers:
    def test_parse_field_found(self):
        from kb.entities import _parse_field

        text = "**Role:** Engineering Lead\n**Team:** Platform"
        assert _parse_field(text, "Role") == "Engineering Lead"
        assert _parse_field(text, "Team") == "Platform"

    def test_parse_field_not_found(self):
        from kb.entities import _parse_field

        text = "Some plain text"
        assert _parse_field(text, "Role") is None

    def test_parse_title(self):
        from kb.entities import _parse_title

        text = "# Talia Ström\n\n**Role:** Lead"
        assert _parse_title(text) == "Talia Ström"

    def test_parse_title_missing(self):
        from kb.entities import _parse_title

        assert _parse_title("No heading here") is None

    def test_stem_to_first_name(self):
        from kb.entities import _stem_to_first_name

        assert _stem_to_first_name("Talia Ström") == "Talia"
        assert _stem_to_first_name("OnlyOneName") is None


# ---------------------------------------------------------------------------
# entities.py: build_entity_patterns, _build_name_patterns
# ---------------------------------------------------------------------------


class TestEntityPatterns:
    def test_build_entity_patterns_basic(self):
        from kb.entities import build_entity_patterns
        from kb.types import Entity

        entities = [
            Entity(id=1, name="Talia Ström", entity_type="person", aliases=["Talia"], metadata={}),
            Entity(id=2, name="AC", entity_type="org", aliases=[], metadata={}),
        ]
        patterns = build_entity_patterns(entities)
        assert len(patterns) > 0

    def test_short_uppercase_alias(self):
        from kb.entities import _build_name_patterns
        from kb.types import Entity

        entity = Entity(id=1, name="Lattice Co", entity_type="org", aliases=["AC"], metadata={})
        patterns = _build_name_patterns(entity)
        # "AC" should get a case-sensitive pattern
        assert any(p[1] == 2 for p in patterns)  # len("AC") = 2

    def test_file_stem_alias_skipped(self):
        from kb.entities import _build_name_patterns
        from kb.types import Entity

        entity = Entity(
            id=1,
            name="Talia Ström",
            entity_type="person",
            aliases=["eve-perrin"],
            metadata={},
        )
        patterns = _build_name_patterns(entity)
        # "eve-perrin" is a file stem — should be skipped
        pattern_strs = [p[0].pattern for p in patterns]
        assert not any("eve-perrin" in ps for ps in pattern_strs)


# ---------------------------------------------------------------------------
# db.py: _apply_migrations edge cases
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_all_migrations_applied(self):
        """Fresh DB should have all migrations applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            rows = conn.execute("SELECT name FROM migrations ORDER BY name").fetchall()
            names = [r["name"] for r in rows]
            assert "001_add_file_mtime" in names
            assert "002_create_facts_table" in names
            assert "003_add_entity_pinned" in names
            assert "004_create_document_attendees" in names
            assert "005_normalize_paths_nfc" in names
            assert "006_unique_entity_name_type" in names
            assert "007_add_document_pinned" in names
            db.close()

    def test_document_attendees_table_exists(self):
        """Migration 004 should create document_attendees table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "document_attendees" in tables
            db.close()

    def test_entities_pinned_column_exists(self):
        """Migration 003 should add pinned column to entities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            # Insert entity with pinned
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, pinned) VALUES (?, ?, ?, ?, ?)",
                ("Test", "person", "[]", "{}", 1),
            )
            row = conn.execute("SELECT pinned FROM entities WHERE name = 'Test'").fetchone()
            assert row["pinned"] == 1
            db.close()

    def test_documents_pinned_column_exists(self):
        """Migration 007 should add pinned column to documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test.md", "Test", None, "notes", "test", "[]", "h1", 0, 1),
            )
            row = conn.execute("SELECT pinned FROM documents WHERE path = 'test.md'").fetchone()
            assert row["pinned"] == 1
            db.close()


# ---------------------------------------------------------------------------
# db.py: get_lance_schema
# ---------------------------------------------------------------------------


class TestLanceSchema:
    def test_get_lance_schema(self):
        from kb.db import get_lance_schema

        schema = get_lance_schema()
        field_names = [f.name for f in schema]
        assert "chunk_id" in field_names
        assert "embedding" in field_names
        assert "document_id" in field_names


# ---------------------------------------------------------------------------
# CLI: person timeline with --from and --to
# ---------------------------------------------------------------------------


class TestPersonTimelineFilters:
    def test_timeline_with_from_date(self, runner, glossary_env):
        """kb person timeline --from filters docs."""
        db, _, data_dir = glossary_env
        conn = db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
            ("Test Person", "person", "[]", "{}"),
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("jan.md", "Jan Meet", "2026-01-15", "notes", "granola", "[]", "jh", 1),
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("feb.md", "Feb Meet", "2026-02-15", "notes", "granola", "[]", "fh", 1),
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 1, 'discussed')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 2, 'discussed')"
        )
        conn.commit()

        result = invoke_cli(
            runner,
            ["person", "timeline", "Test Person", "--from", "2026-02-01", "--json"],
            str(data_dir),
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for doc in data["documents"]:
            if doc.get("date"):
                assert doc["date"] >= "2026-02-01"

    def test_timeline_with_limit(self, runner, glossary_env):
        """kb person timeline --limit 1 caps results."""
        db, _, data_dir = glossary_env
        conn = db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
            ("Lim Person", "person", "[]", "{}"),
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("x.md", "X", "2026-01-01", "notes", "test", "[]", "xh", 1),
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("y.md", "Y", "2026-02-01", "notes", "test", "[]", "yh", 1),
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 1, 'discussed')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 2, 'discussed')"
        )
        conn.commit()

        result = invoke_cli(
            runner,
            ["person", "timeline", "Lim Person", "--limit", "1", "--json"],
            str(data_dir),
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["documents"]) <= 1


# ---------------------------------------------------------------------------
# CLI: _append_fact_to_file edge cases
# ---------------------------------------------------------------------------


class TestAppendFactEdgeCases:
    def test_fact_to_file_without_existing_section(self, tmp_path):
        """Appending to a file without Recent Facts section creates it."""
        md = tmp_path / "person.md"
        md.write_text("# Person\n\nJust some text.\n")
        from unittest.mock import MagicMock

        from kb.api import KnowledgeBase

        # _append_fact_to_file is an instance method that calls self._atomic_write
        mock_kb = MagicMock(spec=KnowledgeBase)
        mock_kb._atomic_write = KnowledgeBase._atomic_write
        KnowledgeBase._append_fact_to_file(mock_kb, md, "New fact", "2026-02-20")
        content = md.read_text()
        assert "## Recent Facts" in content
        assert "[2026-02-20] New fact" in content


# ---------------------------------------------------------------------------
# context.py: _format_person with long roles, missing roles
# ---------------------------------------------------------------------------


class TestContextFormatHelpers:
    def test_format_person_with_long_role(self):
        from kb.context import _format_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=1,
            name="Talia Ström",
            entity_type="person",
            aliases=["Talia"],
            metadata={"role": "A" * 30},  # > 25 chars
            mention_count=3,
            pinned=False,
        )
        result = _format_person(entity)
        assert "..." in result

    def test_format_person_team_only(self):
        from kb.context import _format_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=2,
            name="Someone",
            entity_type="person",
            aliases=["Someone"],
            metadata={"team": "Platform"},
            mention_count=1,
            pinned=False,
        )
        result = _format_person(entity)
        assert "Platform" in result

    def test_format_person_no_meta(self):
        from kb.context import _format_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=3,
            name="Nobody",
            entity_type="person",
            aliases=[],
            metadata={},
            mention_count=1,
            pinned=False,
        )
        result = _format_person(entity)
        assert result == "Nobody"

    def test_format_project_with_lead_and_status(self):
        from kb.context import _format_project
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=4,
            name="Helix Refactor",
            entity_type="project",
            aliases=[],
            metadata={"lead": "Linnea", "status": "Active"},
            mention_count=1,
            pinned=False,
        )
        result = _format_project(entity)
        assert "Linnea" in result
        assert "Active" in result

    def test_format_team_with_abbrev(self):
        from kb.context import _format_team
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=5,
            name="Platform",
            entity_type="team",
            aliases=["PLAT"],
            metadata={},
            mention_count=1,
            pinned=False,
        )
        result = _format_team(entity)
        assert "PLAT" in result

    def test_format_team_no_abbrev(self):
        from kb.context import _format_team
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=6,
            name="Some Team",
            entity_type="team",
            aliases=["some-team"],
            metadata={},
            mention_count=1,
            pinned=False,
        )
        result = _format_team(entity)
        assert result == "Some Team"


# ---------------------------------------------------------------------------
# config.py: find_entities (multi-match)
# ---------------------------------------------------------------------------


class TestFindEntities:
    def test_find_entities_multiple(self):
        from kb.config import find_entities

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Anders Holt", "person", '["Anders M.", "Anders"]', '{"role": "CEO"}'),
            )
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Kit Larsen", "person", '["Kit K.", "Anders"]', '{"role": "VP"}'),
            )
            conn.commit()

            results = find_entities(conn, "Anders")
            assert len(results) >= 2
            db.close()

    def test_find_entities_no_match(self):
        from kb.config import find_entities

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            results = find_entities(conn, "Nonexistent")
            assert len(results) == 0
            db.close()

    def test_find_entities_partial_name_match(self):
        from kb.config import find_entities

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Linnea Holm", "person", '["Linnea"]', "{}"),
            )
            conn.commit()
            results = find_entities(conn, "Linnea")
            assert len(results) >= 1
            db.close()

    def test_find_entities_alias_partial_match(self):
        from kb.config import find_entities

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Helix Refactor", "project", '["helix-refactor", "RM"]', "{}"),
            )
            conn.commit()
            results = find_entities(conn, "RM")
            assert len(results) >= 1
            db.close()


# ---------------------------------------------------------------------------
# _update_frontmatter_pinned (cli.py)
# ---------------------------------------------------------------------------


class TestUpdateFrontmatterPinned:
    def test_inject_pinned_into_existing_frontmatter(self):
        from kb.cli import _update_frontmatter_pinned

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f = root / "test.md"
            f.write_text("---\ntitle: Test\ndate: 2026-01-01\n---\nSome content.\n")
            _update_frontmatter_pinned(root, "test.md", pinned=True)
            content = f.read_text()
            assert "pinned: true" in content
            assert "title: Test" in content

    def test_update_existing_pinned_false(self):
        from kb.cli import _update_frontmatter_pinned

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f = root / "test.md"
            f.write_text("---\ntitle: Test\npinned: true\n---\nContent.\n")
            _update_frontmatter_pinned(root, "test.md", pinned=False)
            content = f.read_text()
            assert "pinned: false" in content

    def test_no_frontmatter_pin_injects_block(self):
        from kb.cli import _update_frontmatter_pinned

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f = root / "test.md"
            f.write_text("Just plain markdown content.\n")
            _update_frontmatter_pinned(root, "test.md", pinned=True)
            content = f.read_text()
            assert content.startswith("---\npinned: true\n---\n")

    def test_no_frontmatter_unpin_noop(self):
        from kb.cli import _update_frontmatter_pinned

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f = root / "test.md"
            original = "Just plain markdown content.\n"
            f.write_text(original)
            _update_frontmatter_pinned(root, "test.md", pinned=False)
            assert f.read_text() == original

    def test_missing_file_noop(self):
        from kb.cli import _update_frontmatter_pinned

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Should not raise
            _update_frontmatter_pinned(root, "nonexistent.md", pinned=True)


# ---------------------------------------------------------------------------
# CLI view with line range (cli.py lines 255-258)
# ---------------------------------------------------------------------------


class TestViewLineRange:
    def test_view_with_line_number(self, runner):

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test/doc.md", "Test Doc", "2026-01-01", "notes", "test", "[]", "hash123", 1),
            )
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (1, 0, None, "Line content here."),
            )
            conn.commit()
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["view", "test/doc.md:10", "--json"],
                env={"KB_DATA_DIR": tmpdir},
                catch_exceptions=False,
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI index status (cli.py lines 1127-1182)
# ---------------------------------------------------------------------------


class TestIndexStatusCommand:
    def test_index_status_json(self):
        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test.md", "Test", "2026-01-01", "notes", "test", "[]", "hash1", 1),
            )
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (1, 0, None, "Content"),
            )
            conn.commit()
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["index", "status", "--json"],
                env={"KB_DATA_DIR": tmpdir},
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "documents" in data
            assert "chunks" in data
            assert "entities" in data
            assert data["documents"] == 1
            assert data["chunks"] == 1


# ---------------------------------------------------------------------------
# CLI entity create duplicate error (cli.py lines 746-748)
# ---------------------------------------------------------------------------


class TestEntityCreateDuplicateError:
    def test_person_create_duplicate(self):
        from unittest.mock import patch as _patch

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            people_dir = project_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            # Write a person file
            (people_dir / "wren-kasper.md").write_text(
                "---\ntitle: Wren Kasper\n---\n# Wren Kasper\n"
            )

            data_dir = project_root / "kbx" / "data"
            db = Database(data_dir)
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                ("Wren Kasper", "person", "[]", "{}", "memory/people/wren-kasper.md"),
            )
            conn.commit()
            db.close()

            runner = CliRunner()
            with _patch("kb.cli._find_project_root", return_value=project_root):
                result = runner.invoke(
                    cli,
                    ["person", "create", "Wren Kasper", "--json"],
                    env={"KB_DATA_DIR": str(data_dir)},
                )
            # Should fail with EntityExistsError
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI entity edit not found (cli.py lines 771-773)
# ---------------------------------------------------------------------------


class TestEntityEditNotFound:
    def test_person_edit_not_found(self):
        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["person", "edit", "NonexistentPerson", "--set", "role=Manager"],
                env={"KB_DATA_DIR": tmpdir},
            )
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI entity delete not found (cli.py lines 783-785)
# ---------------------------------------------------------------------------


class TestEntityDeleteNotFound:
    def test_person_delete_not_found(self):
        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["person", "delete", "NonexistentPerson"],
                env={"KB_DATA_DIR": tmpdir},
            )
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI glossary delete not found (cli.py lines 1055-1057)
# ---------------------------------------------------------------------------


class TestGlossaryDeleteNotFound:
    def test_glossary_delete_nonexistent(self):
        from unittest.mock import patch as _patch

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create glossary file without the term
            project_root = Path(tmpdir)
            glossary = project_root / "memory" / "glossary.md"
            glossary.parent.mkdir(parents=True)
            glossary.write_text(
                "# Glossary\n\n## Acronyms\n\n"
                "| Term | Expansion | Notes |\n|------|-----------|-------|\n"
                "| AC | Lattice Co | |\n"
            )

            db = Database(project_root / "kbx" / "data")
            db.close()

            runner = CliRunner()
            with _patch("kb.cli._find_project_root", return_value=project_root):
                result = runner.invoke(
                    cli,
                    ["glossary", "delete", "NONEXISTENT"],
                    env={"KB_DATA_DIR": str(project_root / "kbx" / "data")},
                )
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# DB: _migrate_005_normalize_paths (db.py lines 130-150)
# ---------------------------------------------------------------------------


class TestMigrate005:
    def test_normalize_paths_migration_normalizes_nfd(self):
        import unicodedata

        from kb.db import _migrate_005_normalize_paths, normalize_path

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()

            # Insert a document with NFD path
            nfd_path = unicodedata.normalize("NFD", "meetings/Linnea.md")
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (nfd_path, "Test", "2026-01-01", "notes", "test", "[]", "hash1", 1),
            )
            # Insert an entity with NFD source path
            nfd_src = unicodedata.normalize("NFD", "memory/Linnea.md")
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                ("Linnea", "person", "[]", "{}", nfd_src),
            )
            conn.commit()

            _migrate_005_normalize_paths(conn)
            conn.commit()

            # Check document path is NFC
            row = conn.execute("SELECT path FROM documents WHERE id = 1").fetchone()
            assert row["path"] == normalize_path(nfd_path)
            assert row["path"] == unicodedata.normalize("NFC", nfd_path)

            # Check entity source_path is NFC
            row = conn.execute("SELECT source_path FROM entities WHERE id = 1").fetchone()
            assert row["source_path"] == normalize_path(nfd_src)
            db.close()


# ---------------------------------------------------------------------------
# context.py: _render_human with full data (covers lines 485-558)
# ---------------------------------------------------------------------------


class TestRenderHumanFull:
    def test_render_human_with_glossary_and_pinned(self):
        """Generate context in human format with all sections to cover _render_human."""
        from kb.context import generate_context

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create glossary with both sections
            glossary = root / "memory" / "glossary.md"
            glossary.parent.mkdir(parents=True)
            glossary.write_text(
                "# Glossary\n\n## Acronyms\n\n"
                "| Term | Expansion | Notes |\n|------|-----------|-------|\n"
                "| AC | Lattice Co | Company name |\n"
                "| MR | Merge Request | GitLab |\n"
                "\n## Internal Language\n\n"
                "| Term | Meaning |\n|------|--------|\n"
                "| Wards | Main GIM repo |\n"
            )

            db = Database(root / "kbx" / "data")
            conn = db.get_sqlite_conn()

            # Insert docs, with one pinned
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "meetings/test.md",
                    "Test Meeting",
                    "2026-01-01",
                    "notes",
                    "test",
                    "[]",
                    "h1",
                    1,
                    1,
                ),
            )
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (1, 0, "Intro", "Meeting content here"),
            )

            # Insert entities of all types
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                (
                    "Talia Ström",
                    "person",
                    '["Talia"]',
                    '{"role": "Engineering Leader", "team": "Platform"}',
                    None,
                ),
            )
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                (
                    "Helix Refactor",
                    "project",
                    '["helix-refactor"]',
                    '{"status": "Active", "lead": "Linnea"}',
                ),
            )
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Platform", "team", '["PLAT"]', "{}"),
            )
            # Add entity mentions for all entity types
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                (1, 1, "discussed"),
            )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                (2, 1, "discussed"),
            )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                (3, 1, "discussed"),
            )
            conn.commit()

            result = generate_context(db, root, fmt="human")
            text = result.text

            # Human format should have markdown headings
            assert "## Key People" in text
            assert "## Projects" in text
            assert "## Teams" in text
            assert "## GG Acronyms" in text
            assert "## GG Jargon" in text
            # Pinned docs section
            assert "## Pinned Documents" in text
            assert "Test Meeting" in text
            db.close()


# ---------------------------------------------------------------------------
# entities.py: _add_claude_md_extra_people (lines 198-224)
# ---------------------------------------------------------------------------


class TestAddClaudeMdExtraPeople:
    def test_adds_extra_people_from_claude_md(self):
        from kb.entities import _add_claude_md_extra_people

        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text(
                "# People\n\n"
                "| Who | Role |\n"
                "|------|-------|\n"
                "| **Anders** | Soren Vance (CEO) |\n"
                "| **Soren** | Linnea Holm (CoS) |\n"
            )
            existing = set()
            result = _add_claude_md_extra_people(claude_md, existing)
            names = [e.name for e in result]
            assert "Soren Vance" in names
            assert "Linnea Holm" in names


# ---------------------------------------------------------------------------
# entities.py: _merge_claude_md_aliases (lines 227-258)
# ---------------------------------------------------------------------------


class TestMergeClaudeMdAliases:
    def test_merges_short_name_into_existing_entity(self):
        from kb.entities import EntityData, _merge_claude_md_aliases

        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text(
                "# People\n\n| Who | Role |\n|------|-------|\n| **Anders** | Anders Holt (CEO) |\n"
            )
            entities = [
                EntityData(
                    name="Anders Holt",
                    entity_type="person",
                    aliases=["Anders M."],
                    metadata={"role": "CEO"},
                    source_path="memory/people/david-marchand.md",
                )
            ]
            result = _merge_claude_md_aliases(entities, claude_md)
            assert "Anders" in result[0].aliases
            assert "Anders M." in result[0].aliases

    def test_non_person_entities_untouched(self):
        from kb.entities import EntityData, _merge_claude_md_aliases

        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text("# People\n\n| Who | Role |\n|------|-------|\n")
            entities = [
                EntityData(
                    name="Helix Refactor",
                    entity_type="project",
                    aliases=["RM"],
                    metadata={},
                    source_path=None,
                )
            ]
            result = _merge_claude_md_aliases(entities, claude_md)
            assert result[0].aliases == ["RM"]


# ---------------------------------------------------------------------------
# entities.py: _build_name_patterns edge cases (lines 479-491)
# ---------------------------------------------------------------------------


class TestBuildNamePatternsEdgeCases:
    def test_single_char_name_skipped(self):
        from kb.entities import Entity, _build_name_patterns

        entity = Entity(id=1, name="X", aliases=[], entity_type="person")
        patterns = _build_name_patterns(entity)
        assert len(patterns) == 0

    def test_common_word_case_sensitive(self):
        from kb.entities import Entity, _build_name_patterns

        entity = Entity(id=1, name="Product", aliases=[], entity_type="team")
        patterns = _build_name_patterns(entity)
        # Should have at least one pattern
        assert len(patterns) >= 1
        # The pattern should be case-sensitive for common words
        pat = patterns[0][0]
        assert pat.search("Product") is not None
        assert pat.search("product") is None

    def test_short_first_name_skipped_in_content(self):
        from kb.entities import Entity, _build_name_patterns

        entity = Entity(id=1, name="Anders Holt", aliases=["Anders"], entity_type="person")
        patterns = _build_name_patterns(entity)
        # "Anders" (6 chars, single word, Title case) should be skipped
        alias_names = [p[0].pattern for p in patterns]
        # Only the full name pattern should be present
        assert any("Anders\\ Holt" in p for p in alias_names)


# ---------------------------------------------------------------------------
# MCP: handle_kb_memory_add note with entity link (mcp_server.py lines 329-333)
# ---------------------------------------------------------------------------


class TestMcpMemoryAddNoteWithEntity:
    def test_note_with_entity_links_entity(self):
        from kb.mcp_server import handle_kb_memory_add

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            notes_dir = root / "memory" / "notes"
            notes_dir.mkdir(parents=True)

            db = Database(root / "kbx" / "data")
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Talia Ström", "person", '["Talia"]', "{}"),
            )
            conn.commit()

            result_str = handle_kb_memory_add(
                db, root, "Test note title", body="Some body content", entity="Talia"
            )
            result = json.loads(result_str)
            assert result["type"] == "note"
            assert result["status"] == "ok"
            db.close()


# ---------------------------------------------------------------------------
# CLI: memory add note collision (cli.py lines 1487-1489)
# ---------------------------------------------------------------------------


class TestNoteCollision:
    def test_note_filename_collision_increments(self):
        from unittest.mock import patch as _patch

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            notes_dir = root / "memory" / "notes"
            notes_dir.mkdir(parents=True)

            db = Database(root / "kbx" / "data")
            db.close()

            runner = CliRunner()
            with _patch("kb.cli._find_project_root", return_value=root):
                # Create first note
                result1 = runner.invoke(
                    cli,
                    ["memory", "add", "Test note", "--date", "2026-01-01"],
                    env={"KB_DATA_DIR": str(root / "kbx" / "data")},
                    catch_exceptions=False,
                )
                assert result1.exit_code == 0

                # Create second note with same title+date (collision)
                result2 = runner.invoke(
                    cli,
                    ["memory", "add", "Test note", "--date", "2026-01-01"],
                    env={"KB_DATA_DIR": str(root / "kbx" / "data")},
                    catch_exceptions=False,
                )
                assert result2.exit_code == 0

            # Check that two different files exist
            files = list(notes_dir.glob("2026-01-01-test-note*.md"))
            assert len(files) == 2


# ---------------------------------------------------------------------------
# db.py: ensure_lance_table (lines 293-306)
# ---------------------------------------------------------------------------


class TestEnsureLanceTable:
    def test_ensure_lance_table_creates_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            table = db.ensure_lance_table()
            assert table is not None
            assert table.count_rows() == 0
            db.close()

    def test_ensure_lance_table_with_data(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            data = [
                {
                    "chunk_id": 1,
                    "document_id": 1,
                    "embedding": np.zeros(1024, dtype=np.float32),
                    "doc_type": "notes",
                    "doc_date": "2026-01-01",
                    "tags": "",
                    "entity_ids": "",
                }
            ]
            table = db.ensure_lance_table(data=data)
            assert table is not None
            assert table.count_rows() == 1
            db.close()

    def test_ensure_lance_table_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            t1 = db.ensure_lance_table()
            t2 = db.ensure_lance_table()
            assert t1 is t2
            db.close()


# ---------------------------------------------------------------------------
# CLI: _find_document_by_target edge cases (cli.py lines 383-464)
# ---------------------------------------------------------------------------


class TestFindDocumentByTargetCli:
    def test_hash_lookup_not_found(self):
        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            result = _find_document_by_target(conn, "#nonexistent")
            assert result is None
            db.close()

    def test_hash_lookup_found(self):
        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test.md", "Test", "2026-01-01", "notes", "test", "[]", "abc123def", 1),
            )
            conn.commit()
            result = _find_document_by_target(conn, "#abc123")
            assert result is not None
            assert result["title"] == "Test"
            db.close()

    def test_title_match(self):
        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "deep/nested/path.md",
                    "My Unique Title",
                    "2026-01-01",
                    "notes",
                    "test",
                    "[]",
                    "hash1",
                    1,
                ),
            )
            conn.commit()
            result = _find_document_by_target(conn, "My Unique Title")
            assert result is not None
            assert result["path"] == "deep/nested/path.md"
            db.close()

    def test_glob_match(self):
        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "meetings/2026/01/unique_meeting.md",
                    "Unique",
                    "2026-01-01",
                    "notes",
                    "test",
                    "[]",
                    "h1",
                    1,
                ),
            )
            conn.commit()
            result = _find_document_by_target(conn, "*unique_meeting*")
            assert result is not None
            db.close()

    def test_ambiguous_title(self):
        import pytest

        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("path1.md", "Same Title", "2026-01-01", "notes", "test", "[]", "h1", 1),
            )
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("path2.md", "Same Title", "2026-01-02", "notes", "test", "[]", "h2", 1),
            )
            conn.commit()
            with pytest.raises(SystemExit):
                _find_document_by_target(conn, "Same Title")
            db.close()


# ---------------------------------------------------------------------------
# context.py: _render_compact line wrapping (covers lines 403-406, 425-427)
# ---------------------------------------------------------------------------


class TestRenderCompactLinePacking:
    def test_compact_with_many_entities_wraps_lines(self):
        """Compact format with many entities should pack into 80-char lines."""
        from kb.context import generate_context

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            glossary = root / "memory" / "glossary.md"
            glossary.parent.mkdir(parents=True)
            glossary.write_text("# Glossary\n")

            db = Database(root / "kbx" / "data")
            conn = db.get_sqlite_conn()

            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test.md", "Test", "2026-01-01", "notes", "test", "[]", "h1", 1),
            )
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (1, 0, None, "Content"),
            )

            # Insert many people to force line wrapping (2+ fields to qualify as key)
            for i in range(15):
                conn.execute(
                    "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                    (
                        f"Person{i} Longname{i}",
                        "person",
                        f'["P{i}"]',
                        f'{{"role": "Role {i}", "team": "Team{i}"}}',
                    ),
                )
            # Insert a team
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Infrastructure", "team", '["INFRA"]', "{}"),
            )
            # Add entity mentions so entities appear in context
            for entity_id in range(1, 17):
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                    (entity_id, 1, "discussed"),
                )
            conn.commit()

            result = generate_context(db, root, fmt="compact")
            assert "[People:key]" in result.text
            assert "[Teams:" in result.text
            db.close()


# ---------------------------------------------------------------------------
# output.py: format_table truncation edge case (line 97)
# ---------------------------------------------------------------------------


class TestOutputEdgeCases:
    def test_render_table_no_results_key(self):
        """render with table fmt should handle dict without results key."""
        from kb.output import render

        data = {"name": "Wren"}
        result = render(data, fmt="table")
        assert "Wren" in result

    def test_format_csv_none_values(self):
        from kb.output import format_csv_str

        rows = [{"name": "Wren", "age": None}]
        result = format_csv_str(rows)
        assert "Wren" in result


# ---------------------------------------------------------------------------
# config.py: find_entities partial alias match (lines 102-104)
# ---------------------------------------------------------------------------


class TestFindEntitiesPartialAlias:
    def test_partial_alias_match(self):
        from kb.config import find_entities

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Helix Refactor", "project", '["helix-refactor", "RM"]', "{}"),
            )
            conn.commit()
            # "helix" matches as partial in alias "helix-refactor"
            results = find_entities(conn, "helix")
            assert len(results) >= 1
            assert results[0]["name"] == "Helix Refactor"
            db.close()

    def test_partial_alias_no_name_match(self):
        """Partial alias match when name doesn't match."""
        from kb.config import find_entities

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Atlas Pipeline", "project", '["atlas-pipeline", "CLM"]', "{}"),
            )
            conn.commit()
            # "CLM" matches as exact alias, won't reach partial
            # "atlas" matches partial in name
            results = find_entities(conn, "atlas")
            assert len(results) >= 1
            db.close()


# ---------------------------------------------------------------------------
# cli.py: _find_document_by_target suffix dedup (lines 423-431)
# ---------------------------------------------------------------------------


class TestFindDocumentByTargetSuffix:
    def test_suffix_match_single(self):
        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "deep/nested/unique_doc.md",
                    "Deep Doc",
                    "2026-01-01",
                    "notes",
                    "test",
                    "[]",
                    "h1",
                    1,
                ),
            )
            conn.commit()
            result = _find_document_by_target(conn, "unique_doc.md")
            assert result is not None
            assert result["title"] == "Deep Doc"
            db.close()

    def test_suffix_ambiguous_raises(self):
        import pytest

        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("path1/doc.md", "Doc 1", "2026-01-01", "notes", "test", "[]", "h1", 1),
            )
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("path2/doc.md", "Doc 2", "2026-01-02", "notes", "test", "[]", "h2", 1),
            )
            conn.commit()
            with pytest.raises(SystemExit):
                _find_document_by_target(conn, "doc.md")
            db.close()

    def test_ambiguous_glob_multiple(self):
        """Multiple glob matches should raise SystemExit(2)."""
        import pytest

        from kb.cli import _find_document_by_target

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("meetings/a_meeting.md", "A", "2026-01-01", "notes", "test", "[]", "h1", 1),
            )
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("meetings/b_meeting.md", "B", "2026-01-02", "notes", "test", "[]", "h2", 1),
            )
            conn.commit()
            with pytest.raises(SystemExit):
                _find_document_by_target(conn, "*_meeting.md")
            db.close()


# ---------------------------------------------------------------------------
# cli.py: timeline to_date and notes dedup (lines 692-693, 715)
# ---------------------------------------------------------------------------


class TestTimelineToDateAndDedup:
    def test_timeline_with_to_date(self):

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            # Person entity
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Talia Ström", "person", '["Talia"]', "{}"),
            )
            # Two docs mentioning Talia
            for i, (date, dtype) in enumerate(
                [("2026-01-10", "notes"), ("2026-02-15", "notes")], 1
            ):
                conn.execute(
                    """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (f"doc{i}.md", f"Doc {i}", date, dtype, "test", "[]", f"h{i}", 1),
                )
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                    (1, i, "discussed"),
                )
            conn.commit()
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["person", "timeline", "Talia", "--to", "2026-01-31", "--json"],
                env={"KB_DATA_DIR": tmpdir},
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data["documents"]) == 1  # Only Jan doc

    def test_timeline_dedup_prefers_notes(self):

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Talia Ström", "person", '["Talia"]', "{}"),
            )
            # Two docs with same date and title: one transcript, one notes
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("transcript.md", "Meeting X", "2026-01-10", "transcript", "test", "[]", "h1", 1),
            )
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("notes.md", "Meeting X", "2026-01-10", "notes", "test", "[]", "h2", 1),
            )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                (1, 1, "discussed"),
            )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                (1, 2, "discussed"),
            )
            conn.commit()
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["person", "timeline", "Talia", "--json"],
                env={"KB_DATA_DIR": tmpdir},
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            # Should dedup to 1 and prefer notes
            assert len(data["documents"]) == 1
            assert data["documents"][0]["doc_type"] == "notes"


# ---------------------------------------------------------------------------
# cli.py: entity unpin not found (lines 653-654)
# ---------------------------------------------------------------------------


class TestEntityUnpinNotFound:
    def test_person_unpin_not_found(self):
        from unittest.mock import patch as _patch

        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db = Database(root / "kbx" / "data")
            db.close()

            runner = CliRunner()
            with _patch("kb.cli._find_project_root", return_value=root):
                result = runner.invoke(
                    cli,
                    ["person", "unpin", "NonexistentPerson"],
                    env={"KB_DATA_DIR": str(root / "kbx" / "data")},
                )
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# context.py: _render_human line wrapping for many people/projects (lines 500-501, 515-516, 554-555)
# ---------------------------------------------------------------------------


class TestRenderHumanLineWrapping:
    def test_human_format_wraps_long_people_lines(self):
        """Human format with many people should wrap at 80 chars."""
        from kb.context import generate_context

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            glossary = root / "memory" / "glossary.md"
            glossary.parent.mkdir(parents=True)
            # Create enough jargon terms to force line wrapping
            jargon_lines = "| Term | Meaning |\n|------|--------|\n"
            for i in range(20):
                jargon_lines += f"| JargonTerm{i}LongName | Meaning {i} |\n"
            glossary.write_text("# Glossary\n\n## Internal Language\n\n" + jargon_lines)

            db = Database(root / "kbx" / "data")
            conn = db.get_sqlite_conn()

            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test.md", "Test", "2026-01-01", "notes", "test", "[]", "h1", 1),
            )
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (1, 0, None, "Content"),
            )

            # Insert many people to force wrapping
            for i in range(10):
                conn.execute(
                    "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                    (
                        f"Person{i} VeryLongLastName{i}",
                        "person",
                        f'["P{i}"]',
                        f'{{"role": "Very Long Role Name Here {i}", "team": "Team{i}"}}',
                    ),
                )
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                    (i + 1, 1, "discussed"),
                )
            # Insert many projects to force wrapping
            for i in range(5):
                eid = 11 + i
                conn.execute(
                    "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                    (
                        f"Project{i} With Long Name",
                        "project",
                        "[]",
                        f'{{"status": "In Progress", "lead": "PersonLead{i}"}}',
                    ),
                )
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                    (eid, 1, "discussed"),
                )
            conn.commit()

            result = generate_context(db, root, fmt="human")
            text = result.text

            assert "## Key People" in text
            assert "## Projects" in text
            assert "## GG Jargon" in text
            # Check that output has multiple lines (wrapping happened)
            lines = text.split("\n")
            # There should be many lines (wrapping produces more lines)
            assert len(lines) > 10
            db.close()


# ---------------------------------------------------------------------------
# mcp_server.py: MCP handler error paths (exception handlers)
# ---------------------------------------------------------------------------


class TestMcpHandlerErrors:
    def test_search_handles_exception(self):
        """Search handler should return error JSON on exception."""
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_search

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB connection failed")

        result = json.loads(handle_kb_search(mock_db, "test query", fast=True, limit=5))
        assert "error" in result

    def test_person_find_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_person_find

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_person_find(mock_db, "test"))
        assert "error" in result

    def test_person_timeline_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_person_timeline

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_person_timeline(mock_db, "test"))
        assert "error" in result

    def test_view_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_view

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_view(mock_db, "test"))
        assert "error" in result

    def test_context_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_context

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_context(mock_db, Path("/tmp")))
        assert "error" in result

    def test_pin_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_pin

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_pin(mock_db, "test"))
        assert "error" in result

    def test_unpin_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_unpin

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_unpin(mock_db, "test"))
        assert "error" in result

    def test_usage_handles_exception(self):
        import json as _json
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_usage

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = handle_kb_usage(mock_db)
        data = _json.loads(result)
        assert "error" in data

    def test_memory_add_handles_exception(self):
        from unittest.mock import MagicMock

        from kb.mcp_server import handle_kb_memory_add

        mock_db = MagicMock()
        mock_db.get_sqlite_conn.side_effect = RuntimeError("DB error")
        result = json.loads(handle_kb_memory_add(mock_db, Path("/tmp"), "test"))
        assert "error" in result


# ---------------------------------------------------------------------------
# cli.py: list with limit overflow (line 1704)
# ---------------------------------------------------------------------------


class TestListLimitOverflow:
    def test_list_more_docs_than_limit(self):
        from click.testing import CliRunner

        from kb.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            for i in range(5):
                conn.execute(
                    """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"doc{i}.md",
                        f"Doc {i}",
                        f"2026-01-0{i + 1}",
                        "notes",
                        "test",
                        "[]",
                        f"hash{i}",
                        1,
                    ),
                )
            conn.commit()
            db.close()

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["list", "--limit", "2", "--json"],
                env={"KB_DATA_DIR": tmpdir},
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data["results"]) == 2
