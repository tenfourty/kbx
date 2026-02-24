"""Tests for kb.cli — CLI commands using Click CliRunner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import invoke_cli

from kb.db import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_db():
    """Create a small test database with known data for CLI tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()

        # Insert test documents
        docs = [
            (
                "meetings/2026/01/27/jeremy_charles.notes.md",
                "Wren / Soren",
                "2026-01-27",
                "notes",
                "granola",
                "id1",
                '["Soren"]',
                "abcdef123456",
                2,
            ),
            (
                "meetings/2026/01/20/rust_migration.notes.md",
                "Helix Refactor Status",
                "2026-01-20",
                "notes",
                "granola",
                "id2",
                "[]",
                "fedcba654321",
                2,
            ),
            (
                "memory/people/eve.md",
                "Talia Ström",
                None,
                "memory_person",
                "memory",
                None,
                "[]",
                "111222333444",
                1,
            ),
        ]
        for path, title, date, dtype, src, sid, tags, chash, cc in docs:
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (path, title, date, dtype, src, sid, tags, chash, cc),
            )

        # Insert test chunks
        chunks = [
            (
                1,
                0,
                "Overview",
                "[Meeting: Wren / Soren | Date: 2026-01-27]\nPerformance review discussion with Soren.",
            ),
            (
                1,
                1,
                "Next Steps",
                "[Meeting: Wren / Soren | Date: 2026-01-27]\nSchedule follow-up meeting in February.",
            ),
            (
                2,
                0,
                "Status",
                "[Meeting: Helix Refactor Status | Date: 2026-01-20]\nRust migration at 30% completion.",
            ),
            (
                2,
                1,
                "Performance",
                "[Meeting: Helix Refactor Status | Date: 2026-01-20]\nRust engine is 5x faster than Python.",
            ),
            (3, 0, None, "Talia Ström is the Infrastructure/Platform lead."),
        ]
        for doc_id, idx, heading, content in chunks:
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (doc_id, idx, heading, content),
            )

        # Insert entities
        entities = [
            (
                "Talia Ström",
                "person",
                '["Talia"]',
                '{"role": "Engineering Leader", "team": "Platform"}',
                "memory/people/eve.md",
            ),
            (
                "Soren Vance",
                "person",
                '["Soren"]',
                '{"team": "DevEfficiency"}',
                "memory/people/thomas.md",
            ),
            (
                "Helix Refactor",
                "project",
                '["helix-refactor"]',
                '{"status": "In Progress"}',
                "memory/projects/helix-refactor.md",
            ),
            (
                "Soren Vance",
                "person",
                '["Anders M.", "Anders"]',
                '{"role": "CEO"}',
                "memory/people/david-marchand.md",
            ),
            (
                "Kit Larsen",
                "person",
                '["Kit K.", "Anders"]',
                '{"role": "VP Eng"}',
                "memory/people/dave-kowalski.md",
            ),
        ]
        for name, etype, aliases, meta, src in entities:
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                (name, etype, aliases, meta, src),
            )

        # Add a __document__ summary chunk (tests issue #1: section display)
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (
                1,
                2,
                "__document__",
                "[Summary: Wren / Soren | Date: 2026-01-27]\nPerformance review with Soren discussing improvements.",
            ),
        )

        # Add a document with a different date for date-range search tests
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "meetings/2026/02/05/weekly_sync.notes.md",
                "Weekly Sync",
                "2026-02-05",
                "notes",
                "granola",
                "id4",
                "[]",
                "444555666777",
                1,
            ),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (4, 0, "Updates", "Weekly sync meeting with team updates and recruitment discussion."),
        )

        # Entity mentions
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (2, 1, 'participant')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (3, 2, 'discussed')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 2, 'discussed')"
        )

        conn.commit()
        yield db, Path(tmpdir)
        db.close()


# ---------------------------------------------------------------------------
# Step 1: search command
# ---------------------------------------------------------------------------


class TestSearchCommand:
    def test_search_fast_json(self, runner, cli_db):
        """kb search 'test' --fast --json returns valid JSON."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["search", "Soren", "--fast", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "meta" in data

    def test_search_fast_table(self, runner, cli_db):
        """kb search 'Rust' --fast returns table output."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["search", "Rust", "--fast"], db_path)
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_search_with_limit(self, runner, cli_db):
        """kb search with --limit returns at most N results."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["search", "meeting", "--fast", "--json", "--limit", "1"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["results"]) <= 1

    def test_search_with_format_csv(self, runner, cli_db):
        """kb search --format csv returns CSV output."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["search", "Rust", "--fast", "--format", "csv"], db_path)
        assert result.exit_code == 0
        assert "title" in result.output or "path" in result.output

    def test_search_with_fields(self, runner, cli_db):
        """kb search --fields title,score returns only those fields."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["search", "Soren", "--fast", "--json", "--fields", "title,score"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        if data["results"]:
            r = data["results"][0]
            assert "title" in r
            assert "score" in r
            assert "path" not in r

    def test_search_deep_placeholder(self, runner, cli_db):
        """kb search --deep prints not yet implemented."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["search", "test", "--deep"], db_path)
        # Should mention not yet implemented somewhere in output
        assert "not yet implemented" in result.output.lower()

    def test_search_with_type_filter(self, runner, cli_db):
        """kb search --type notes filters results."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["search", "Rust", "--fast", "--json", "--type", "notes"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            assert r["doc_type"] == "notes"

    def test_search_no_query(self, runner, cli_db):
        """kb search without query shows usage error."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["search"], db_path)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Step 2: view command
# ---------------------------------------------------------------------------


class TestViewCommand:
    def test_view_by_path(self, runner, cli_db):
        """kb view with a known path returns document metadata."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--json"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "Wren / Soren"

    def test_view_by_hash(self, runner, cli_db):
        """kb view #abcdef resolves by content hash prefix."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["view", "#abcdef", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "Wren / Soren"

    def test_view_by_suffix(self, runner, cli_db):
        """kb view with partial path (suffix match)."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["view", "jeremy_charles.notes.md", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Wren" in data["title"]

    def test_view_not_found(self, runner, cli_db):
        """kb view with unknown path returns non-zero exit code."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["view", "nonexistent.md"], db_path)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Issue #11: kb view --plain for raw content output
# ---------------------------------------------------------------------------


class TestViewPlainFlag:
    def test_view_plain_outputs_raw_content(self, runner, cli_db):
        """kb view --plain should output just the content, no JSON wrapper or table."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner,
            ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--plain"],
            db_path,
        )
        assert result.exit_code == 0
        output = result.output
        # Should contain the actual content from chunks
        assert "Performance review discussion with Soren." in output
        assert "Schedule follow-up meeting in February." in output

    def test_view_plain_no_json_wrapper(self, runner, cli_db):
        """kb view --plain should not wrap output in JSON."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner,
            ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--plain"],
            db_path,
        )
        assert result.exit_code == 0
        output = result.output.strip()
        # Should NOT be valid JSON (not a dict/list wrapper)
        assert not output.startswith("{")
        assert not output.startswith("[")
        # Should NOT contain JSON keys like "title", "path", "chunks"
        assert '"title"' not in output
        assert '"chunks"' not in output

    def test_view_plain_strips_metadata_prefix(self, runner, cli_db):
        """kb view --plain should strip [Meeting: ...] metadata prefixes."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner,
            ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--plain"],
            db_path,
        )
        assert result.exit_code == 0
        assert "[Meeting:" not in result.output

    def test_view_plain_includes_section_headings(self, runner, cli_db):
        """kb view --plain should include section headings as ## markers."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner,
            ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--plain"],
            db_path,
        )
        assert result.exit_code == 0
        assert "## Overview" in result.output
        assert "## Next Steps" in result.output

    def test_view_plain_excludes_document_summary_chunks(self, runner, cli_db):
        """kb view --plain should skip __document__ summary chunks."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner,
            ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--plain"],
            db_path,
        )
        assert result.exit_code == 0
        assert "__document__" not in result.output

    def test_view_plain_with_json_flag_plain_wins(self, runner, cli_db):
        """When both --plain and --json are given, --plain takes precedence."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner,
            ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--plain", "--json"],
            db_path,
        )
        assert result.exit_code == 0
        output = result.output.strip()
        # --plain should win, so no JSON wrapper
        assert not output.startswith("{")


# ---------------------------------------------------------------------------
# Step 3: list command
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_list_json(self, runner, cli_db):
        """kb list --json returns {results, meta}."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "results" in data
        assert "meta" in data
        assert len(data["results"]) == 4

    def test_list_type_filter(self, runner, cli_db):
        """kb list --type notes filters by doc_type."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list", "--type", "notes", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        for doc in data["results"]:
            assert doc["doc_type"] == "notes"

    def test_list_limit(self, runner, cli_db):
        """kb list --limit 1 returns at most 1 document."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list", "--limit", "1", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["results"]) <= 1

    def test_list_date_range(self, runner, cli_db):
        """kb list --from --to filters by date range."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["list", "--from", "2026-01-25", "--to", "2026-01-31", "--json"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for doc in data["results"]:
            assert doc["date"] >= "2026-01-25"
            assert doc["date"] <= "2026-01-31"

    def test_list_default_table(self, runner, cli_db):
        """kb list returns table output by default."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list"], db_path)
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0


# ---------------------------------------------------------------------------
# Step 4: entity commands
# ---------------------------------------------------------------------------


class TestPersonCommands:
    def test_person_list_json(self, runner, cli_db):
        """kb person list --json returns {results, meta} with only people."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "results" in data
        for e in data["results"]:
            assert e["entity_type"] == "person"

    def test_person_find_json(self, runner, cli_db):
        """kb person find 'Talia' --json returns compact profile."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "Talia", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Talia Ström"
        # New compact shape
        assert "facts" in data
        assert isinstance(data["facts"], list)
        assert "document_count" in data
        assert isinstance(data["document_count"], int)
        assert "source_path" in data
        assert "breadcrumbs" in data
        # Old shape is gone
        assert "documents" not in data

    def test_person_find_includes_facts(self, runner, cli_db):
        """kb person find includes facts from the facts table."""
        db, db_path = cli_db
        conn = db.get_sqlite_conn()
        # Insert a fact for Talia (entity_id=1)
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (1, 'Leads platform team', '2026-02-23')"
        )
        conn.commit()
        result = invoke_cli(runner, ["person", "find", "Talia", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["facts"]) == 1
        assert data["facts"][0]["text"] == "Leads platform team"
        assert data["facts"][0]["date"] == "2026-02-23"

    def test_person_find_breadcrumbs(self, runner, cli_db):
        """kb person find includes breadcrumbs for deeper exploration."""
        import datetime

        _db, db_path = cli_db

        # Mock date.today() to make breadcrumbs deterministic
        fake_today = datetime.date(2026, 2, 23)
        with patch("kb.cli.date") as mock_date:
            mock_date.today.return_value = fake_today
            mock_date.side_effect = lambda *a, **kw: datetime.date(*a, **kw)
            result = invoke_cli(runner, ["person", "find", "Talia", "--json"], db_path)

        assert result.exit_code == 0
        data = json.loads(result.output)
        bc = data["breadcrumbs"]
        assert "timeline" in bc
        assert "Talia Ström" in bc["timeline"]
        assert "recent" in bc
        assert "--from 2026-01-24" in bc["recent"]
        assert "profile" in bc
        assert bc["profile"] == "kbx view memory/people/eve.md"

    def test_person_find_document_count(self, runner, cli_db):
        """kb person find shows document_count instead of full doc list."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "Talia", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Talia has 1 mention in cli_db fixture (entity_id=1 -> doc_id=2)
        assert data["document_count"] == 1
        assert "documents" not in data

    def test_person_find_me(self, runner, cli_db, tmp_path):
        """kb person find me resolves to [user] name from config."""
        _db, db_path = cli_db
        # Write a config with [user] name
        config = tmp_path / "kbx.toml"
        config.write_text('[user]\nname = "Talia Ström"\n')
        env = {"KB_DATA_DIR": str(db_path), "KBX_CONFIG": str(config)}
        from kb.cli import cli

        result = runner.invoke(
            cli, ["person", "find", "me", "--json"], env=env, catch_exceptions=False
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Talia Ström"

    def test_me_shortcut(self, runner, cli_db, tmp_path):
        """kb me is a shortcut for person find me."""
        _db, db_path = cli_db
        config = tmp_path / "kbx.toml"
        config.write_text('[user]\nname = "Talia Ström"\n')
        env = {"KB_DATA_DIR": str(db_path), "KBX_CONFIG": str(config)}
        from kb.cli import cli

        result = runner.invoke(cli, ["me", "--json"], env=env, catch_exceptions=False)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Talia Ström"

    def test_person_find_partial_match(self, runner, cli_db):
        """kb person find with partial name (case-insensitive)."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "thomas", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Soren" in data["name"]

    def test_person_find_not_found(self, runner, cli_db):
        """kb person find with unknown name returns non-zero."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "zzzznotexist"], db_path)
        assert result.exit_code != 0

    def test_person_timeline_json(self, runner, cli_db):
        """kb person timeline 'Talia' returns chronological docs."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "timeline", "Talia", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "name" in data
        assert "documents" in data

    def test_person_create_json(self, runner):
        """kb person create 'Test Person' --json creates entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "memory" / "people").mkdir(parents=True)
            (root / "memory" / "projects").mkdir(parents=True)
            db_path = root / "data"
            Database(db_path)  # initialize DB schema

            with patch("kb.cli._find_project_root", return_value=root):
                result = invoke_cli(
                    runner,
                    ["person", "create", "Test Person", "--role", "Engineer", "--json"],
                    str(db_path),
                )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["name"] == "Test Person"


class TestProjectCommands:
    def test_project_list_json(self, runner, cli_db):
        """kb project list --json returns only projects."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["project", "list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        for e in data["results"]:
            assert e["entity_type"] == "project"

    def test_project_find_json(self, runner, cli_db):
        """kb project find 'Helix Refactor' --json returns profile."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["project", "find", "Helix Refactor", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Helix Refactor" in data["name"]

    def test_project_find_compact(self, runner, cli_db):
        """kb project find also uses compact output."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["project", "find", "Helix Refactor", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "facts" in data
        assert "document_count" in data
        assert "breadcrumbs" in data
        assert "documents" not in data
        assert "timeline" in data["breadcrumbs"]

    def test_project_create_json(self, runner):
        """kb project create 'New Project' --json creates entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "memory" / "people").mkdir(parents=True)
            (root / "memory" / "projects").mkdir(parents=True)
            db_path = root / "data"
            Database(db_path)

            with patch("kb.cli._find_project_root", return_value=root):
                result = invoke_cli(
                    runner,
                    ["project", "create", "New Project", "--status", "Active", "--json"],
                    str(db_path),
                )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["name"] == "New Project"


# ---------------------------------------------------------------------------
# Step 5: index commands
# ---------------------------------------------------------------------------


class TestIndexCommands:
    def test_index_status_json(self, runner, cli_db):
        """kb index status --json returns health info."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["index", "status", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "documents" in data
        assert "chunks" in data
        assert "entities" in data

    def test_index_run_no_embed_flag(self, runner, cli_db):
        """--no-embed flag runs text-only indexing without requiring sentence_transformers."""
        from kb.indexer import IndexResult

        _db, db_path = cli_db
        mock_result = IndexResult(
            documents_indexed=1,
            documents_skipped=0,
            chunks_created=3,
            entities_linked=1,
            embeddings_skipped=True,
        )
        with patch("kb.indexer.index_all", return_value=mock_result):
            result = invoke_cli(runner, ["index", "run", "--no-embed"], db_path)
        assert result.exit_code == 0
        output = result.output.lower()
        assert "text-only" in output
        assert "embeddings: skipped" in output

    def test_usage_contains_no_embed(self, runner, cli_db):
        """kb usage output documents --no-embed flag."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["usage"], db_path)
        assert result.exit_code == 0
        assert "--no-embed" in result.output


# ---------------------------------------------------------------------------
# Step 6: usage command
# ---------------------------------------------------------------------------


class TestUsageCommand:
    def test_usage_contains_search(self, runner, cli_db):
        """kb usage output contains 'kb search'."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["usage"], db_path)
        assert result.exit_code == 0
        assert "kb search" in result.output

    def test_usage_contains_score_interpretation(self, runner, cli_db):
        """kb usage output contains 'Score Interpretation'."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["usage"], db_path)
        assert result.exit_code == 0
        assert "Score Interpretation" in result.output


# ---------------------------------------------------------------------------
# Step 7: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_bad_search_no_query(self, runner, cli_db):
        """Missing query returns non-zero exit code."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["search"], db_path)
        assert result.exit_code != 0

    def test_view_bad_path(self, runner, cli_db):
        """Bad path in view returns non-zero exit code."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["view", "totally_nonexistent_file.md"], db_path)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Issue #1: entity find error should be JSON when --json is passed
# ---------------------------------------------------------------------------


class TestPersonFindErrorJSON:
    def test_person_not_found_json_is_valid_json(self, runner, cli_db):
        """person find with --json should return valid JSON even on not-found."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "zzzznotexist", "--json"], db_path)
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "error" in data

    def test_person_not_found_json_has_suggestion(self, runner, cli_db):
        """Not-found JSON error should include suggestion or available_actions."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "Evee", "--json"], db_path)
        # "Evee" is close to "Talia" — should suggest it
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "error" in data
        # Should have a suggestion or available_actions
        assert "suggestion" in data or "available_actions" in data

    def test_person_timeline_not_found_json(self, runner, cli_db):
        """person timeline with --json should return valid JSON on not-found."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "timeline", "zzzznotexist", "--json"], db_path)
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "error" in data


# ---------------------------------------------------------------------------
# Issue #2: kb view --json should have a content field
# ---------------------------------------------------------------------------


class TestViewContentField:
    def test_view_json_has_content(self, runner, cli_db):
        """kb view --json should have a top-level 'content' field."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--json"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "content" in data
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0

    def test_view_json_content_strips_metadata_prefix(self, runner, cli_db):
        """View content should strip [Meeting: ...] prefixes."""
        _db, db_path = cli_db
        result = invoke_cli(
            runner, ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--json"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert not data["content"].startswith("[Meeting:")


# ---------------------------------------------------------------------------
# Issue #3: kb list --json should return {results, meta} not bare list
# ---------------------------------------------------------------------------


class TestListConsistentSchema:
    def test_list_json_has_results_and_meta(self, runner, cli_db):
        """kb list --json should return {results: [...], meta: {...}}."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "results" in data
        assert "meta" in data
        assert isinstance(data["results"], list)

    def test_list_json_meta_has_total(self, runner, cli_db):
        """kb list meta should include total count."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["meta"]["total"] == len(data["results"])

    def test_list_json_results_same_fields(self, runner, cli_db):
        """kb list results should have same fields as before (id, path, title, date, doc_type)."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        for doc in data["results"]:
            assert "title" in doc
            assert "path" in doc
            assert "doc_type" in doc

    def test_person_list_json_has_results_and_meta(self, runner, cli_db):
        """kb person list --json should also return {results, meta}."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "results" in data
        assert "meta" in data


# ---------------------------------------------------------------------------
# Issue #6: entity find "Anders" (ambiguous) should return all matches
# ---------------------------------------------------------------------------


class TestPersonAmbiguous:
    def test_person_find_ambiguous_returns_multiple(self, runner, cli_db):
        """person find 'Anders' should return multiple matches when ambiguous."""
        _db, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "Anders", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should indicate ambiguity — either multiple results or an error with candidates
        if "results" in data:
            assert len(data["results"]) >= 2
        elif "candidates" in data:
            assert len(data["candidates"]) >= 2
        else:
            # Single result is wrong — both Davids should appear
            raise AssertionError("Ambiguous 'Anders' should return multiple matches")


# ---------------------------------------------------------------------------
# QA Issue #1: __document__ section should be null in search results
# ---------------------------------------------------------------------------


class TestDocumentSectionDisplay:
    def test_search_document_section_is_null(self, runner, cli_db):
        """Search results from __document__ summary chunks should have section: null."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["search", "performance review", "--fast", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            assert r["section"] != "__document__", (
                f"Section should be null, not '__document__' (title: {r['title']})"
            )

    def test_view_excludes_document_chunks_from_content(self, runner, cli_db):
        """kb view --json content field should not include __document__ summary chunks."""
        _, db_path = cli_db
        result = invoke_cli(
            runner, ["view", "meetings/2026/01/27/jeremy_charles.notes.md", "--json"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "__document__" not in data["content"]


# ---------------------------------------------------------------------------
# QA Issue #2: kb search --from / --to date filters
# ---------------------------------------------------------------------------


class TestSearchDateFilters:
    def test_search_with_from_date(self, runner, cli_db):
        """kb search --from should filter results to that date onward."""
        _, db_path = cli_db
        result = invoke_cli(
            runner, ["search", "meeting", "--fast", "--json", "--from", "2026-02-01"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            assert r["date"] is not None and r["date"] >= "2026-02-01", (
                f"Result date {r['date']} should be >= 2026-02-01"
            )

    def test_search_with_to_date(self, runner, cli_db):
        """kb search --to should filter results up to that date."""
        _, db_path = cli_db
        result = invoke_cli(
            runner, ["search", "meeting", "--fast", "--json", "--to", "2026-01-25"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            assert r["date"] is not None and r["date"] <= "2026-01-25", (
                f"Result date {r['date']} should be <= 2026-01-25"
            )

    def test_search_from_to_range(self, runner, cli_db):
        """kb search --from --to combined narrows to a range."""
        _, db_path = cli_db
        result = invoke_cli(
            runner,
            [
                "search",
                "Rust OR sync",
                "--fast",
                "--json",
                "--from",
                "2026-01-15",
                "--to",
                "2026-01-31",
            ],
            db_path,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            assert "2026-01-15" <= r["date"] <= "2026-01-31"

    def test_search_from_excludes_earlier(self, runner, cli_db):
        """--from 2026-02-01 should exclude Jan 27 and Jan 20 documents."""
        _, db_path = cli_db
        result = invoke_cli(
            runner,
            ["search", "meeting OR sync", "--fast", "--json", "--from", "2026-02-01"],
            db_path,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        jan_dates = [r for r in data["results"] if r["date"] and r["date"] < "2026-02-01"]
        assert len(jan_dates) == 0, f"Found results before 2026-02-01: {jan_dates}"


# ---------------------------------------------------------------------------
# QA Issue #4: Entity timeline deduplication (notes + transcript same meeting)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QA Issue #5: Fuzzy entity suggestions should be relevant
# ---------------------------------------------------------------------------


class TestSearchSortOption:
    def test_search_sort_date_json(self, runner, cli_db):
        """kb search --sort date --json should sort by date."""
        _, db_path = cli_db
        result = invoke_cli(
            runner, ["search", "meeting", "--fast", "--sort", "date", "--json"], db_path
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        dates = [r["date"] for r in data["results"] if r["date"]]
        assert dates == sorted(dates, reverse=True)


class TestIngestCommand:
    def test_ingest_uses_sys_executable(self, runner, cli_db, tmp_path):
        """ingest should invoke organise_granola.py with sys.executable."""
        import sys
        from unittest.mock import patch as mock_patch

        _, db_path = cli_db
        # Create a fake organise script
        script = tmp_path / "meetings" / "organise_granola.py"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env python3\nprint('ok')\n")
        with (
            mock_patch("kb.cli._find_project_root", return_value=tmp_path),
            mock_patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = type("R", (), {"returncode": 0})()
            invoke_cli(runner, ["ingest"], db_path)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == sys.executable


class TestMemoryListWrapper:
    """kb memory list should return {results, meta} wrapper like other list commands."""

    def test_memory_list_json_empty_has_wrapper(self, runner, cli_db):
        """Empty memory list should return {results: [], meta: {...}}."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["memory", "list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "meta" in data
        assert data["results"] == []

    def test_memory_list_json_with_facts_has_wrapper(self, runner, cli_db):
        """Memory list with facts should return {results: [...], meta: {total: N}}."""
        _, db_path = cli_db
        invoke_cli(
            runner,
            ["memory", "add", "Some fact", "--entity", "Talia", "--date", "2026-02-01"],
            db_path,
        )
        result = invoke_cli(runner, ["memory", "list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "meta" in data
        assert len(data["results"]) == 1
        assert data["meta"]["total"] == 1

    def test_memory_list_table_empty_shows_hint(self, runner, cli_db):
        """Empty memory list table output should show a helpful message."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["memory", "list"], db_path)
        assert result.exit_code == 0
        assert "No facts" in result.output or "no facts" in result.output.lower()


class TestFuzzyPersonSuggestions:
    def test_unknown_person_no_wild_suggestions(self, runner, cli_db):
        """Completely unrelated names should not be suggested."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "Elon Musk", "--json"], db_path)
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "error" in data
        # Should NOT suggest unrelated names like "Helix Refactor"
        if "available_actions" in data:
            for action in data["available_actions"]:
                assert "Helix Refactor" not in action, "Should not suggest unrelated entities"

    def test_typo_suggestion_is_relevant(self, runner, cli_db):
        """Close typo like 'Evee' should suggest Talia Ström."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["person", "find", "Evee", "--json"], db_path)
        assert result.exit_code != 0
        data = json.loads(result.output)
        if "suggestion" in data:
            assert "Talia" in data["suggestion"]


class TestPersonTimelineDedup:
    def test_person_timeline_no_duplicate_meetings(self, runner, cli_db):
        """Person timeline should not show both notes and transcript for same meeting date+title."""
        db, db_path = cli_db
        conn = db.get_sqlite_conn()
        # Add a transcript version of the same Jan 27 meeting
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "meetings/2026/01/27/jeremy_charles.transcript.md",
                "Wren / Soren",
                "2026-01-27",
                "transcript",
                "granola",
                "id1t",
                "[]",
                "aaabbb111222",
                1,
            ),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (5, 0, None, "Soren: I think the performance review was fair."),
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (2, 5, 'participant')"
        )
        conn.commit()

        result = invoke_cli(runner, ["person", "timeline", "Soren", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Count Jan 27 entries — should be 1 (deduplicated), not 2
        jan27 = [d for d in data["documents"] if d.get("date") == "2026-01-27"]
        assert len(jan27) == 1, (
            f"Expected 1 entry for Jan 27 (deduplicated), got {len(jan27)}: {jan27}"
        )


# ---------------------------------------------------------------------------
# Unicode NFC/NFD normalization
# ---------------------------------------------------------------------------


class TestUnicodeNormalization:
    def test_view_nfc_finds_nfd_path(self, runner, cli_db):
        """kb view with NFC input should find a document stored with NFD path."""
        import unicodedata

        db, db_path = cli_db
        conn = db.get_sqlite_conn()
        # Insert a document with NFD path (decomposed: e + combining accent)
        nfd_path = unicodedata.normalize("NFD", "meetings/2026/01/23/Camille_d7ae74bb.notes.md")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                nfd_path,
                "Linnea Meeting",
                "2026-01-23",
                "notes",
                "granola",
                "[]",
                "uuu111222333",
                1,
            ),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (5, 0, None, "Meeting with Linnea about Rust."),
        )
        conn.commit()

        # Query with NFC input (precomposed: é as single codepoint)
        nfc_path = unicodedata.normalize("NFC", "meetings/2026/01/23/Camille_d7ae74bb.notes.md")
        result = invoke_cli(runner, ["view", nfc_path, "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "Linnea Meeting"

    def test_view_nfd_finds_nfc_path(self, runner, cli_db):
        """kb view with NFD input should find a document stored with NFC path."""
        import unicodedata

        db, db_path = cli_db
        conn = db.get_sqlite_conn()
        # Insert with NFC path
        nfc_path = unicodedata.normalize("NFC", "meetings/2026/02/10/Hugo_abc12345.notes.md")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                nfc_path,
                "Mei Meeting",
                "2026-02-10",
                "notes",
                "granola",
                "[]",
                "vvv444555666",
                1,
            ),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (5, 0, None, "Meeting with Mei."),
        )
        conn.commit()

        # Query with NFD input
        nfd_input = unicodedata.normalize("NFD", "meetings/2026/02/10/Hugo_abc12345.notes.md")
        result = invoke_cli(runner, ["view", nfd_input, "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "Mei Meeting"


# ---------------------------------------------------------------------------
# Glob / fuzzy path matching in kb view
# ---------------------------------------------------------------------------


class TestViewGlobMatching:
    def test_view_glob_star(self, runner, cli_db):
        """kb view '*jeremy_charles*' should find a matching document via glob."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["view", "*jeremy_charles*", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Wren" in data["title"]

    def test_view_filename_substring(self, runner, cli_db):
        """kb view 'jeremy_charles' (no glob) should find via filename substring."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["view", "jeremy_charles", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Wren" in data["title"]

    def test_view_glob_ambiguous_returns_exit_2(self, runner, cli_db):
        """kb view with glob matching multiple docs should return exit code 2."""
        _, db_path = cli_db
        # Glob "*notes*" should match multiple documents
        result = invoke_cli(runner, ["view", "*notes*"], db_path)
        assert result.exit_code == 2

    def test_view_glob_no_match_still_suggests(self, runner, cli_db):
        """kb view with non-matching glob should still show suggestions."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["view", "*zzzznotexist*"], db_path)
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "did you mean" in result.output.lower()

    def test_view_question_mark_glob(self, runner, cli_db):
        """kb view with ? glob should match single characters."""
        _, db_path = cli_db
        # "jeremy_charle?.notes.md" should match "jeremy_charles.notes.md"
        result = invoke_cli(runner, ["view", "*jeremy_charle?.notes*", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Wren" in data["title"]


class TestNormalizePath:
    def test_nfd_to_nfc(self):
        """normalize_path should convert NFD to NFC."""
        import unicodedata

        from kb.db import normalize_path

        nfd = unicodedata.normalize("NFD", "Linnea")
        result = normalize_path(nfd)
        assert result == unicodedata.normalize("NFC", "Linnea")

    def test_nfc_unchanged(self):
        """normalize_path should leave NFC unchanged."""
        import unicodedata

        from kb.db import normalize_path

        nfc = unicodedata.normalize("NFC", "Linnea")
        assert normalize_path(nfc) == nfc

    def test_ascii_unchanged(self):
        """normalize_path should leave ASCII paths unchanged."""
        from kb.db import normalize_path

        assert normalize_path("meetings/2026/01/test.md") == "meetings/2026/01/test.md"


class TestContextHumanFlag:
    def test_context_default_is_compact(self, runner, cli_db):
        """kb context should output compact format by default."""
        _, db_path = cli_db
        with patch("kb.cli._find_project_root", return_value=Path("/nonexistent")):
            result = invoke_cli(runner, ["context"], db_path)
        assert result.exit_code == 0
        assert "[People:key]" in result.output

    def test_context_human_flag(self, runner, cli_db):
        """kb context --human should output markdown format."""
        _, db_path = cli_db
        with patch("kb.cli._find_project_root", return_value=Path("/nonexistent")):
            result = invoke_cli(runner, ["context", "--human"], db_path)
        assert result.exit_code == 0
        assert "## Key People" in result.output


class TestPersonPinUnpin:
    def test_person_pin_sets_flag(self, runner, cli_db):
        """kb person pin 'Talia' should set pinned=1."""
        db, db_path = cli_db
        result = invoke_cli(runner, ["person", "pin", "Talia"], db_path)
        assert result.exit_code == 0
        row = (
            db.get_sqlite_conn()
            .execute("SELECT pinned FROM entities WHERE name = 'Talia Ström'")
            .fetchone()
        )
        assert row["pinned"] == 1

    def test_person_unpin_clears_flag(self, runner, cli_db):
        """kb person unpin 'Talia' should set pinned=0."""
        db, db_path = cli_db
        # Pin first, then unpin
        invoke_cli(runner, ["person", "pin", "Talia"], db_path)
        result = invoke_cli(runner, ["person", "unpin", "Talia"], db_path)
        assert result.exit_code == 0
        row = (
            db.get_sqlite_conn()
            .execute("SELECT pinned FROM entities WHERE name = 'Talia Ström'")
            .fetchone()
        )
        assert row["pinned"] == 0

    def test_person_pin_not_found(self, runner, cli_db):
        """kb person pin unknown name returns error."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["person", "pin", "NonexistentPerson"], db_path)
        assert result.exit_code != 0

    def test_person_pin_json_output(self, runner, cli_db):
        """Pin command confirms with entity name."""
        _, db_path = cli_db
        result = invoke_cli(runner, ["person", "pin", "Talia"], db_path)
        assert result.exit_code == 0
        assert "Talia Ström" in result.output


# ---------------------------------------------------------------------------
# Notes & Pinning
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_env():
    """Create a project root with memory dirs and a small indexed DB for note tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "memory" / "people").mkdir(parents=True)
        (root / "memory" / "projects").mkdir(parents=True)
        (root / "memory" / "notes").mkdir(parents=True)
        data_dir = root / "data"
        db = Database(data_dir)
        conn = db.get_sqlite_conn()

        # Insert a test document for pin testing
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "memory/priorities/cirs.md",
                "CIRs",
                "2026-02-01",
                "memory",
                "memory",
                None,
                "[]",
                "aaa111bbb222",
                3,
            ),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (1, 0, "Immediate", "Production incidents, key person departures"),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (1, 1, "Daily", "Calendar conflicts, overdue items"),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (1, 2, "Weekly", "Initiative progress, team health"),
        )

        # Insert an entity for entity-linked notes
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
            (
                "Soren Vance",
                "person",
                '["Anders"]',
                '{"role": "CEO"}',
                "memory/people/david-marchand.md",
            ),
        )

        conn.commit()
        yield db, root, data_dir
        db.close()


def invoke_cli_with_root(runner, args, data_dir, project_root):
    """Invoke CLI with KB_DATA_DIR and patched project root."""
    from kb.cli import cli

    env = {"KB_DATA_DIR": str(data_dir)}
    with patch("kb.cli._find_project_root", return_value=project_root):
        return runner.invoke(cli, args, env=env, catch_exceptions=False)


class TestDocumentPin:
    def test_pin_by_path(self, runner, notes_env):
        """kb pin <path> sets the pinned flag."""
        db, root, data_dir = notes_env
        result = invoke_cli_with_root(runner, ["pin", "memory/priorities/cirs.md"], data_dir, root)
        assert result.exit_code == 0
        assert "Pinned" in result.output

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT pinned FROM documents WHERE path = 'memory/priorities/cirs.md'"
        ).fetchone()
        assert row["pinned"] == 1

    def test_pin_by_title(self, runner, notes_env):
        """kb pin <title> resolves by title."""
        db, root, data_dir = notes_env
        result = invoke_cli_with_root(runner, ["pin", "CIRs"], data_dir, root)
        assert result.exit_code == 0

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT pinned FROM documents WHERE title = 'CIRs'").fetchone()
        assert row["pinned"] == 1

    def test_unpin(self, runner, notes_env):
        """kb unpin removes the pinned flag."""
        db, root, data_dir = notes_env
        conn = db.get_sqlite_conn()

        # Pin first
        conn.execute("UPDATE documents SET pinned = 1 WHERE path = 'memory/priorities/cirs.md'")
        conn.commit()

        result = invoke_cli_with_root(runner, ["unpin", "CIRs"], data_dir, root)
        assert result.exit_code == 0
        assert "Unpinned" in result.output

        row = conn.execute("SELECT pinned FROM documents WHERE title = 'CIRs'").fetchone()
        assert row["pinned"] == 0

    def test_pin_not_found(self, runner, notes_env):
        """kb pin on unknown document exits with error."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(runner, ["pin", "nonexistent_doc"], data_dir, root)
        assert result.exit_code == 1


class TestMemoryAddNote:
    def test_quick_note_no_entity(self, runner, notes_env):
        """kb memory add 'text' (no entity) creates a note file."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner, ["memory", "add", "Budget approved for Q2"], data_dir, root
        )
        assert result.exit_code == 0
        assert "Note created" in result.output

        # Check file was written
        note_files = list((root / "memory" / "notes").glob("*budget-approved*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "title: Budget approved for Q2" in content
        assert "Budget approved for Q2" in content

    def test_note_with_body(self, runner, notes_env):
        """kb memory add with --body creates structured note."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner,
            ["memory", "add", "ExCom Takeaways", "--body", "## Decisions\n- Approved roadmap"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        note_files = list((root / "memory" / "notes").glob("*excom-takeaways*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "## Decisions" in content
        assert "Approved roadmap" in content

    def test_note_with_tags(self, runner, notes_env):
        """kb memory add with --tags writes tags to frontmatter."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner,
            ["memory", "add", "Hiring update", "--tags", "hiring,budget"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        note_files = list((root / "memory" / "notes").glob("*hiring-update*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "tags: [hiring, budget]" in content

    def test_note_with_pin(self, runner, notes_env):
        """kb memory add --pin creates and pins the note."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner,
            ["memory", "add", "Important note", "--pin"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        assert "Pinned to context" in result.output

        # Re-open DB to see committed writes from the CLI
        fresh_db = Database(data_dir)
        conn = fresh_db.get_sqlite_conn()
        row = conn.execute("SELECT pinned FROM documents WHERE title = 'Important note'").fetchone()
        assert row is not None
        assert row["pinned"] == 1
        fresh_db.close()

    def test_note_with_entity_link(self, runner, notes_env):
        """kb memory add with --entity and --body creates note + entity mention."""
        db, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner,
            [
                "memory",
                "add",
                "CEO feedback",
                "--body",
                "Anders shared Q3 plans",
                "--entity",
                "Soren Vance",
            ],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        assert "Note created" in result.output

        conn = db.get_sqlite_conn()
        doc_row = conn.execute("SELECT id FROM documents WHERE title = 'CEO feedback'").fetchone()
        assert doc_row is not None
        mention = conn.execute(
            "SELECT * FROM entity_mentions WHERE document_id = ? AND mention_type = 'tagged'",
            (doc_row["id"],),
        ).fetchone()
        assert mention is not None

    def test_fact_path_unchanged(self, runner, notes_env):
        """kb memory add with --entity only (no body/tags/pin) uses fact path."""
        _, root, data_dir = notes_env
        # Create entity source file for fact append
        entity_file = root / "memory" / "people" / "david-marchand.md"
        entity_file.write_text("# Soren Vance\n\n**Role:** CEO\n", encoding="utf-8")

        result = invoke_cli_with_root(
            runner,
            ["memory", "add", "Promoted to chairman", "--entity", "Soren Vance"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        assert "Fact recorded" in result.output

        # Verify no note file was created
        note_files = list((root / "memory" / "notes").glob("*promoted*"))
        assert len(note_files) == 0


class TestFrontmatterPinning:
    def test_note_with_pin_writes_frontmatter(self, runner, notes_env):
        """Note created with --pin should have pinned: true in frontmatter."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner, ["memory", "add", "Pinned note test", "--pin"], data_dir, root
        )
        assert result.exit_code == 0

        note_files = list((root / "memory" / "notes").glob("*pinned-note-test*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "pinned: true" in content

    def test_reindex_preserves_pinned_from_frontmatter(self, runner, notes_env):
        """Full re-index should set pinned=1 for notes with pinned: true in frontmatter."""
        _, root, data_dir = notes_env

        # Write a note file with pinned: true in frontmatter directly
        note_path = root / "memory" / "notes" / "2026-02-20-test-reindex.md"
        note_path.write_text(
            "---\ntitle: Test Reindex\ndate: 2026-02-20\npinned: true\n---\nSome content.\n",
            encoding="utf-8",
        )

        # Run a full index (text-only)
        from kb.db import Database
        from kb.indexer import index_all

        db = Database(data_dir)
        index_all(db, None, root, full=True)

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT pinned FROM documents WHERE path = 'memory/notes/2026-02-20-test-reindex.md'"
        ).fetchone()
        assert row is not None, "Document should be indexed"
        assert row["pinned"] == 1, "pinned should be 1 from frontmatter"
        db.close()

    def test_pin_updates_frontmatter(self, runner, notes_env):
        """kb pin should add pinned: true to the note's frontmatter."""
        _, root, data_dir = notes_env

        # Create a note file without pinned in frontmatter
        note_path = root / "memory" / "notes" / "2026-02-20-test-pin-fm.md"
        note_path.write_text(
            "---\ntitle: Test Pin FM\ndate: 2026-02-20\n---\nSome content.\n",
            encoding="utf-8",
        )

        # Index it so DB knows about it
        from kb.db import Database
        from kb.indexer import index_all

        db = Database(data_dir)
        index_all(db, None, root, full=True)

        # Pin via CLI
        result = invoke_cli_with_root(
            runner,
            ["pin", "memory/notes/2026-02-20-test-pin-fm.md"],
            data_dir,
            root,
        )
        assert result.exit_code == 0

        # Check frontmatter on disk
        content = note_path.read_text()
        assert "pinned: true" in content

    def test_unpin_removes_frontmatter_pinned(self, runner, notes_env):
        """kb unpin should set pinned: false in the note's frontmatter."""
        _, root, data_dir = notes_env

        # Create a note file with pinned: true
        note_path = root / "memory" / "notes" / "2026-02-20-test-unpin-fm.md"
        note_path.write_text(
            "---\ntitle: Test Unpin FM\ndate: 2026-02-20\npinned: true\n---\nSome content.\n",
            encoding="utf-8",
        )

        # Index it so DB knows about it
        from kb.db import Database
        from kb.indexer import index_all

        db = Database(data_dir)
        index_all(db, None, root, full=True)

        # Unpin via CLI
        result = invoke_cli_with_root(
            runner,
            ["unpin", "memory/notes/2026-02-20-test-unpin-fm.md"],
            data_dir,
            root,
        )
        assert result.exit_code == 0

        # Check frontmatter on disk — should NOT have pinned: true
        content = note_path.read_text()
        assert "pinned: true" not in content
        assert "pinned: false" in content


class TestPinnedInContext:
    def test_pinned_docs_in_compact_context(self, notes_env):
        """Pinned documents appear in compact context output."""
        db, root, _ = notes_env
        conn = db.get_sqlite_conn()

        # Pin the CIRs doc
        conn.execute("UPDATE documents SET pinned = 1 WHERE path = 'memory/priorities/cirs.md'")
        conn.commit()

        from kb.context import generate_context

        result = generate_context(db, root)
        assert "[Pinned:1]" in result.text
        assert "CIRs" in result.text
        assert "Immediate" in result.text  # heading digest

    def test_pinned_docs_in_human_context(self, notes_env):
        """Pinned documents appear in human-readable context."""
        db, root, _ = notes_env
        conn = db.get_sqlite_conn()

        conn.execute("UPDATE documents SET pinned = 1 WHERE path = 'memory/priorities/cirs.md'")
        conn.commit()

        from kb.context import generate_context

        result = generate_context(db, root, fmt="human")
        assert "## Pinned Documents" in result.text
        assert "CIRs" in result.text
        assert "Immediate" in result.text

    def test_no_pinned_section_when_empty(self, notes_env):
        """No [Pinned] section when nothing is pinned."""
        db, root, _ = notes_env
        from kb.context import generate_context

        result = generate_context(db, root)
        assert "[Pinned" not in result.text


# ---------------------------------------------------------------------------
# Tag filter on search
# ---------------------------------------------------------------------------


class TestSearchTagFilter:
    def test_search_with_tag_filter(self, runner, notes_env):
        """kb search --tag should filter results to matching tags."""
        _, root, data_dir = notes_env
        # Create two notes with different tags
        invoke_cli_with_root(
            runner,
            [
                "memory",
                "add",
                "Infrastructure decision",
                "--tags",
                "decision,infra",
                "--body",
                "Use Postgres",
            ],
            data_dir,
            root,
        )
        invoke_cli_with_root(
            runner,
            [
                "memory",
                "add",
                "Finance decision",
                "--tags",
                "decision,finance",
                "--body",
                "Budget approved",
            ],
            data_dir,
            root,
        )
        # Search with tag filter
        result = invoke_cli_with_root(
            runner,
            ["search", "decision", "--fast", "--json", "--tag", "infra"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should find results, all from the infra-tagged doc
        assert len(data["results"]) > 0
        # The finance decision should NOT appear
        for r in data["results"]:
            assert "Finance" not in r["title"]

    def test_search_without_tag_returns_both(self, runner, notes_env):
        """kb search without --tag should return all matching results."""
        _, root, data_dir = notes_env
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Alpha note", "--tags", "alpha", "--body", "Content A"],
            data_dir,
            root,
        )
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Beta note", "--tags", "beta", "--body", "Content B"],
            data_dir,
            root,
        )
        result = invoke_cli_with_root(
            runner,
            ["search", "note", "--fast", "--json"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        titles = [r["title"] for r in data["results"]]
        assert any("Alpha" in t for t in titles)
        assert any("Beta" in t for t in titles)


# ---------------------------------------------------------------------------
# Note list command
# ---------------------------------------------------------------------------


class TestNoteList:
    def test_note_list_returns_notes(self, runner, notes_env):
        """Create 2 notes, verify kb note list --json returns both."""
        _, root, data_dir = notes_env
        invoke_cli_with_root(
            runner,
            ["memory", "add", "First note", "--body", "Content A"],
            data_dir,
            root,
        )
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Second note", "--body", "Content B"],
            data_dir,
            root,
        )
        result = invoke_cli_with_root(
            runner,
            ["note", "list", "--json"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "meta" in data
        titles = [r["title"] for r in data["results"]]
        assert any("First" in t for t in titles)
        assert any("Second" in t for t in titles)

    def test_note_list_filter_by_tag(self, runner, notes_env):
        """Create notes with different tags, verify --tag filters correctly."""
        _, root, data_dir = notes_env
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Special note", "--tags", "special,work", "--body", "X"],
            data_dir,
            root,
        )
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Normal note", "--tags", "general", "--body", "Y"],
            data_dir,
            root,
        )
        result = invoke_cli_with_root(
            runner,
            ["note", "list", "--tag", "special", "--json"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["results"]) >= 1
        titles = [r["title"] for r in data["results"]]
        assert any("Special" in t for t in titles)
        assert not any("Normal" in t for t in titles)

    def test_note_list_pinned_only(self, runner, notes_env):
        """Create pinned and unpinned notes, verify --pinned shows only pinned."""
        _, root, data_dir = notes_env
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Pinned note", "--pin", "--body", "Important"],
            data_dir,
            root,
        )
        invoke_cli_with_root(
            runner,
            ["memory", "add", "Unpinned note", "--body", "Trivial"],
            data_dir,
            root,
        )
        result = invoke_cli_with_root(
            runner,
            ["note", "list", "--pinned", "--json"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["results"]) >= 1
        for r in data["results"]:
            assert r["pinned"] is True
        titles = [r["title"] for r in data["results"]]
        assert any("Pinned" in t for t in titles)
        assert not any("Unpinned" in t for t in titles)


# ---------------------------------------------------------------------------
# Note edit command
# ---------------------------------------------------------------------------


class TestNoteEdit:
    def _create_note(
        self, runner, root, data_dir, title, body="Some content", tags=None, pin=False
    ):
        """Helper: create a note and return the note file path."""
        args = ["memory", "add", title, "--body", body]
        if tags:
            args.extend(["--tags", tags])
        if pin:
            args.append("--pin")
        result = invoke_cli_with_root(runner, args, data_dir, root)
        assert result.exit_code == 0
        return result

    def test_note_edit_replace_body(self, runner, notes_env):
        """kb note edit <target> --body 'new' replaces body content."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Editable note", body="Original body")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Editable note", "--body", "Replaced body"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

        # Verify file on disk has new body
        note_files = list((root / "memory" / "notes").glob("*editable-note*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "Replaced body" in content
        assert "Original body" not in content

    def test_note_edit_append(self, runner, notes_env):
        """kb note edit <target> --append 'extra' appends to body."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Appendable note", body="First line")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Appendable note", "--append", "\nSecond line"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

        note_files = list((root / "memory" / "notes").glob("*appendable-note*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "First line" in content
        assert "Second line" in content

    def test_note_edit_update_tags(self, runner, notes_env):
        """kb note edit <target> --tags 'new,tags' replaces tags."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Tagged note", tags="old,outdated")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Tagged note", "--tags", "fresh,updated"],
            data_dir,
            root,
        )
        assert result.exit_code == 0

        note_files = list((root / "memory" / "notes").glob("*tagged-note*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "fresh" in content
        assert "updated" in content
        assert "old" not in content
        assert "outdated" not in content

    def test_note_edit_pin(self, runner, notes_env):
        """kb note edit <target> --pin pins the note."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Pinnable note")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Pinnable note", "--pin"],
            data_dir,
            root,
        )
        assert result.exit_code == 0

        note_files = list((root / "memory" / "notes").glob("*pinnable-note*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "pinned: true" in content

    def test_note_edit_unpin(self, runner, notes_env):
        """kb note edit <target> --unpin unpins the note."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Unpinnable note", pin=True)

        # Verify it's pinned first
        note_files = list((root / "memory" / "notes").glob("*unpinnable-note*.md"))
        assert len(note_files) == 1
        assert "pinned: true" in note_files[0].read_text()

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Unpinnable note", "--unpin"],
            data_dir,
            root,
        )
        assert result.exit_code == 0

        content = note_files[0].read_text()
        assert "pinned: true" not in content

    def test_note_edit_not_found(self, runner, notes_env):
        """kb note edit on unknown target exits with error."""
        _, root, data_dir = notes_env
        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "nonexistent note", "--body", "hello"],
            data_dir,
            root,
        )
        assert result.exit_code == 1

    def test_note_edit_not_a_note(self, runner, notes_env):
        """kb note edit on a meeting doc (not a note) exits with error."""
        db, root, data_dir = notes_env
        conn = db.get_sqlite_conn()
        # Insert a meeting doc
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "meetings/2026/02/10/standup.notes.md",
                "Standup",
                "2026-02-10",
                "notes",
                "granola",
                "[]",
                "mtg111222333",
                1,
            ),
        )
        conn.commit()

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Standup", "--body", "hacked"],
            data_dir,
            root,
        )
        assert result.exit_code == 1
        assert "not a note" in result.output.lower() or "not a memory note" in result.output.lower()

    def test_note_edit_body_and_append_conflict(self, runner, notes_env):
        """kb note edit with both --body and --append exits with error."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Conflict note")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Conflict note", "--body", "new", "--append", "extra"],
            data_dir,
            root,
        )
        assert result.exit_code == 1
        assert "cannot" in result.output.lower() or "conflict" in result.output.lower()

    def test_note_edit_no_options(self, runner, notes_env):
        """kb note edit with no edit options shows current info or error."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "No edit note")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "No edit note"],
            data_dir,
            root,
        )
        assert result.exit_code == 1
        assert "no edit" in result.output.lower() or "specify" in result.output.lower()

    def test_note_edit_json_output(self, runner, notes_env):
        """kb note edit --json returns structured output."""
        _, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "JSON note", body="Content", tags="test")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "JSON note", "--body", "New content", "--json"],
            data_dir,
            root,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data
        assert "title" in data

    def test_note_edit_reindexes(self, runner, notes_env):
        """After note edit, the document is re-indexed with updated content."""
        db, root, data_dir = notes_env
        self._create_note(runner, root, data_dir, "Reindex note", body="Before edit")

        result = invoke_cli_with_root(
            runner,
            ["note", "edit", "Reindex note", "--body", "After edit"],
            data_dir,
            root,
        )
        assert result.exit_code == 0

        # Verify the chunk content was updated by re-indexing
        conn = db.get_sqlite_conn()
        doc = conn.execute("SELECT id FROM documents WHERE title = 'Reindex note'").fetchone()
        assert doc is not None
        chunks = conn.execute(
            "SELECT content FROM chunks WHERE document_id = ?", (doc["id"],)
        ).fetchall()
        chunk_text = " ".join(c["content"] for c in chunks)
        assert "After edit" in chunk_text
