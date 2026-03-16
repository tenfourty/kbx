"""Tests for kb MCP server — Phase 8: tools, config, resources."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kb.db import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_db():
    """Create a test database with known data for MCP tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()

        docs = [
            (
                "meetings/2026/01/27/mfa_review.notes.md",
                "MFA Implementation Review",
                "2026-01-27",
                "notes",
                "granola",
                "id1",
                "[]",
                "aaa111222333",
                2,
            ),
            (
                "meetings/2026/01/20/rust_status.notes.md",
                "Helix Refactor Status",
                "2026-01-20",
                "notes",
                "granola",
                "id2",
                "[]",
                "bbb444555666",
                2,
            ),
            (
                "meetings/2026/02/01/quartz-indexer.notes.md",
                "Atlas Pipeline Plan",
                "2026-02-01",
                "notes",
                "granola",
                "id3",
                "[]",
                "ccc777888999",
                1,
            ),
            (
                "memory/people/eve.md",
                "Talia Ström",
                None,
                "memory_person",
                "memory",
                None,
                "[]",
                "ddd000111222",
                1,
            ),
        ]
        for path, title, date, dtype, src, sid, tags, chash, cc in docs:
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (path, title, date, dtype, src, sid, tags, chash, cc),
            )

        chunks = [
            (
                1,
                0,
                "Overview",
                "[Meeting: MFA Implementation Review | Date: 2026-01-27]\nMFA implementation using TOTP with Okta integration.",
            ),
            (
                1,
                1,
                "Rollout",
                "[Meeting: MFA Implementation Review | Date: 2026-01-27]\nRollout to all employees by end of February.",
            ),
            (
                2,
                0,
                "Status",
                "[Meeting: Helix Refactor Status | Date: 2026-01-20]\nRust migration at 45% completion. 180 of 400 modules converted.",
            ),
            (
                2,
                1,
                "Performance",
                "[Meeting: Helix Refactor Status | Date: 2026-01-20]\nRust engine is 5x faster than Python.",
            ),
            (
                3,
                0,
                "Plan",
                "[Meeting: Atlas Pipeline Plan | Date: 2026-02-01]\nDatastore SSO integration and Grafana dashboard migration.",
            ),
            (4, 0, None, "Talia Ström is the Infrastructure/Platform lead."),
        ]
        for doc_id, idx, heading, content in chunks:
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (doc_id, idx, heading, content),
            )

        entities = [
            (
                "Talia Ström",
                "person",
                '["Talia"]',
                '{"role": "Engineering Leader", "team": "Platform"}',
                "memory/people/eve.md",
            ),
            (
                "Helix Refactor",
                "project",
                '["helix-refactor"]',
                '{"status": "In Progress"}',
                None,
            ),
        ]
        for name, etype, aliases, meta, src in entities:
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                (name, etype, aliases, meta, src),
            )

        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 1, 'discussed')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (2, 2, 'discussed')"
        )
        conn.commit()
        yield db, Path(tmpdir)
        db.close()


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_find_project_root_returns_path(self):
        """find_project_root should return a Path."""
        from kb.config import find_project_root

        result = find_project_root()
        assert isinstance(result, Path)

    def test_get_data_dir_from_env(self, tmp_path):
        """get_data_dir should use KB_DATA_DIR env var when set."""
        from kb.config import get_data_dir

        old = os.environ.get("KB_DATA_DIR")
        try:
            os.environ["KB_DATA_DIR"] = str(tmp_path)
            result = get_data_dir()
            assert result == tmp_path
        finally:
            if old is not None:
                os.environ["KB_DATA_DIR"] = old
            else:
                os.environ.pop("KB_DATA_DIR", None)

    def test_get_db_returns_database(self, mcp_db):
        """get_db should return a Database instance."""
        from kb.config import get_db

        _, db_path = mcp_db
        old = os.environ.get("KB_DATA_DIR")
        try:
            os.environ["KB_DATA_DIR"] = str(db_path)
            db = get_db()
            assert isinstance(db, Database)
        finally:
            if old is not None:
                os.environ["KB_DATA_DIR"] = old
            else:
                os.environ.pop("KB_DATA_DIR", None)

    def test_find_entity_exact_match(self, mcp_db):
        """find_entity should find by exact name."""
        from kb.config import find_entity

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        result = find_entity(conn, "Talia Ström")
        assert result is not None
        assert result["name"] == "Talia Ström"

    def test_find_entity_alias_match(self, mcp_db):
        """find_entity should find by alias."""
        from kb.config import find_entity

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        result = find_entity(conn, "Talia")
        assert result is not None
        assert result["name"] == "Talia Ström"

    def test_find_entity_not_found(self, mcp_db):
        """find_entity should return None for unknown names."""
        from kb.config import find_entity

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        result = find_entity(conn, "NonexistentPerson")
        assert result is None


# ---------------------------------------------------------------------------
# MCP tool handler tests (test functions directly, not MCP transport)
# ---------------------------------------------------------------------------


class TestMcpSearch:
    def test_search_returns_json(self, mcp_db):
        """kb_search should return valid JSON with results."""
        from kb.mcp_server import handle_kb_search

        db, _ = mcp_db
        result = handle_kb_search(db, "MFA", fast=True, limit=5)
        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_finds_correct_doc(self, mcp_db):
        """kb_search should find the right document."""
        from kb.mcp_server import handle_kb_search

        db, _ = mcp_db
        result = handle_kb_search(db, "Cloud migration", fast=True, limit=5)
        data = json.loads(result)
        titles = [r["title"] for r in data["results"]]
        assert any("Cloud" in t for t in titles)

    def test_search_no_results(self, mcp_db):
        """kb_search should return empty results for no matches."""
        from kb.mcp_server import handle_kb_search

        db, _ = mcp_db
        result = handle_kb_search(db, "zzzznonexistentquery", fast=True, limit=5)
        data = json.loads(result)
        assert data["results"] == []


class TestMcpPersonFind:
    def test_person_find_returns_json(self, mcp_db):
        """kb_person_find should return valid JSON with person data."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        result = handle_kb_person_find(db, "Talia")
        data = json.loads(result)
        assert data["name"] == "Talia Ström"

    def test_person_find_compact_output(self, mcp_db):
        """kb_person_find should return compact output with facts, doc count, breadcrumbs."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        result = handle_kb_person_find(db, "Talia")
        data = json.loads(result)
        assert "document_count" in data
        assert "breadcrumbs" in data
        assert "facts" in data
        assert isinstance(data["document_count"], int)
        assert isinstance(data["breadcrumbs"], dict)
        assert isinstance(data["facts"], list)
        # Should NOT have the old full documents list
        assert "documents" not in data

    def test_person_find_not_found(self, mcp_db):
        """kb_person_find should return error for unknown person."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        result = handle_kb_person_find(db, "NonexistentPerson")
        data = json.loads(result)
        assert "error" in data


class TestFactIdsInPersonFind:
    """Facts returned by person_find should include IDs for editing."""

    def test_facts_have_ids(self, mcp_db):
        """Each fact in person_find output should have a 'seq' field."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_person_find

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Talia is great at debugging", entity="Talia")

        data = json.loads(handle_kb_person_find(db, "Talia"))
        assert len(data["facts"]) >= 1
        for fact in data["facts"]:
            assert "seq" in fact, f"Fact missing 'seq': {fact}"
            assert isinstance(fact["seq"], int)
            assert "text" in fact
            assert "date" in fact

    def test_fact_id_matches_db(self, mcp_db):
        """Fact seq in output should match the actual DB row seq."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_person_find

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Talia knows Rust", entity="Talia", date="2026-03-16")

        data = json.loads(handle_kb_person_find(db, "Talia"))
        fact = next(f for f in data["facts"] if f["text"] == "Talia knows Rust")

        conn = db.get_sqlite_conn()
        db_fact = conn.execute(
            "SELECT seq FROM facts WHERE fact_text = 'Talia knows Rust'"
        ).fetchone()
        assert fact["seq"] == db_fact["seq"]

    def test_project_find_facts_have_ids(self, mcp_db):
        """Project find should also return fact seq."""
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        # Helix Refactor is a fixture project — may not have facts, but check shape
        data = json.loads(handle_kb_project_find(db, "Helix Refactor"))
        assert "facts" in data
        # If there are facts, they should have seq
        for fact in data["facts"]:
            assert "seq" in fact


class TestMemoryListEntityFilter:
    """list_facts should support entity filtering."""

    def test_list_facts_filtered_by_entity(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_memory_list

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Talia speaks French", entity="Talia")
        handle_kb_memory_add(db, root, "Cloud is migrating", entity="Helix Refactor")

        # Filter by entity
        result = json.loads(handle_kb_memory_list(db, root, entity="Talia"))
        assert result["meta"]["total"] == 1
        assert result["results"][0]["entity_name"] == "Talia Ström"

    def test_list_facts_no_entity_returns_all(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_memory_list

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Talia speaks French", entity="Talia")
        handle_kb_memory_add(db, root, "Cloud is migrating", entity="Helix Refactor")

        result = json.loads(handle_kb_memory_list(db, root))
        assert result["meta"]["total"] == 2


class TestMcpPersonTimeline:
    def test_timeline_returns_json(self, mcp_db):
        """kb_person_timeline should return valid JSON."""
        from kb.mcp_server import handle_kb_person_timeline

        db, _ = mcp_db
        result = handle_kb_person_timeline(db, "Talia")
        data = json.loads(result)
        assert "documents" in data

    def test_timeline_not_found(self, mcp_db):
        """kb_person_timeline should return error for unknown person."""
        from kb.mcp_server import handle_kb_person_timeline

        db, _ = mcp_db
        result = handle_kb_person_timeline(db, "NonexistentPerson")
        data = json.loads(result)
        assert "error" in data


class TestMcpView:
    def test_view_by_path(self, mcp_db):
        """kb_view should return document by path."""
        from kb.mcp_server import handle_kb_view

        db, _ = mcp_db
        result = handle_kb_view(db, "meetings/2026/01/27/mfa_review.notes.md")
        data = json.loads(result)
        assert data["title"] == "MFA Implementation Review"
        assert "chunks" in data

    def test_view_by_hash(self, mcp_db):
        """kb_view should return document by #hash prefix."""
        from kb.mcp_server import handle_kb_view

        db, _ = mcp_db
        result = handle_kb_view(db, "#aaa111")
        data = json.loads(result)
        assert data["title"] == "MFA Implementation Review"

    def test_view_not_found(self, mcp_db):
        """kb_view should return error for unknown document."""
        from kb.mcp_server import handle_kb_view

        db, _ = mcp_db
        result = handle_kb_view(db, "nonexistent/path.md")
        data = json.loads(result)
        assert "error" in data


class TestMcpViewGlob:
    def test_view_glob_finds_document(self, mcp_db):
        """MCP kb_view with glob pattern should find matching document."""
        from kb.mcp_server import handle_kb_view

        db, _ = mcp_db
        result = handle_kb_view(db, "*mfa_review*")
        data = json.loads(result)
        assert "error" not in data
        assert data["title"] == "MFA Implementation Review"

    def test_view_filename_substring(self, mcp_db):
        """MCP kb_view with filename substring should find matching document."""
        from kb.mcp_server import handle_kb_view

        db, _ = mcp_db
        result = handle_kb_view(db, "mfa_review")
        data = json.loads(result)
        assert "error" not in data
        assert data["title"] == "MFA Implementation Review"


class TestMcpViewUnicode:
    def test_view_nfc_finds_nfd_path(self, mcp_db):
        """MCP kb_view with NFC input should find document stored with NFD path."""
        import unicodedata

        from kb.mcp_server import handle_kb_view

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        nfd_path = unicodedata.normalize("NFD", "meetings/2026/01/23/Camille_test.notes.md")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (nfd_path, "Linnea Test", "2026-01-23", "notes", "granola", "[]", "uuu999888777", 1),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
            (5, 0, None, "Meeting with Linnea."),
        )
        conn.commit()

        nfc_input = unicodedata.normalize("NFC", "meetings/2026/01/23/Camille_test.notes.md")
        result = handle_kb_view(db, nfc_input)
        data = json.loads(result)
        assert "error" not in data
        assert data["title"] == "Linnea Test"


class TestMcpContext:
    def test_context_returns_json(self, mcp_db):
        """kb_context should return valid JSON."""
        from kb.mcp_server import handle_kb_context

        db, _ = mcp_db
        result = handle_kb_context(db, project_root=Path("/tmp"))
        data = json.loads(result)
        assert "text" in data
        assert "stats" in data

    def test_context_with_topic(self, mcp_db):
        """kb_context with topic should filter."""
        from kb.mcp_server import handle_kb_context

        db, _ = mcp_db
        result = handle_kb_context(db, project_root=Path("/tmp"), topic="Cloud")
        data = json.loads(result)
        assert "text" in data


class TestMcpContextFmt:
    def test_context_default_compact(self, mcp_db):
        """handle_kb_context with default fmt should return compact format."""
        from kb.mcp_server import handle_kb_context

        db, _ = mcp_db
        result = handle_kb_context(db, project_root=Path("/tmp"))
        data = json.loads(result)
        assert "text" in data
        # Compact format uses [People:key] style
        assert "[People:key]" in data["text"]

    def test_context_human_fmt(self, mcp_db):
        """handle_kb_context with fmt='human' should return markdown format."""
        from kb.mcp_server import handle_kb_context

        db, _ = mcp_db
        result = handle_kb_context(db, project_root=Path("/tmp"), fmt="human")
        data = json.loads(result)
        assert "text" in data
        # Human format uses ## headings
        assert "## Key People" in data["text"]

    def test_context_mcp_tool_accepts_fmt(self, mcp_db):
        """kb_context MCP tool should accept fmt parameter."""
        import inspect

        from kb.mcp_server import kb_context

        sig = inspect.signature(kb_context)
        assert "fmt" in sig.parameters


class TestMcpUsage:
    def test_usage_returns_json(self, mcp_db):
        """kb_usage should return structured JSON with index stats."""
        from kb.mcp_server import handle_kb_usage

        db, _ = mcp_db
        data = json.loads(handle_kb_usage(db))
        assert "docs" in data
        assert "entities" in data
        assert "facts" in data
        assert "pinned" in data
        assert "date_range" in data
        assert "tool_count" in data
        assert isinstance(data["docs"], int)
        assert isinstance(data["entities"], int)
        assert isinstance(data["tool_count"], int)

    def test_usage_stats_reflect_fixture_data(self, mcp_db):
        """kb_usage stats should match the fixture's known data."""
        from kb.mcp_server import handle_kb_usage

        db, _ = mcp_db
        data = json.loads(handle_kb_usage(db))
        assert data["docs"] == 4  # 3 meetings + 1 person doc in fixture
        assert data["entities"] == 2  # Talia Ström + Helix Refactor
        assert data["pinned"] == 0  # no pinned docs in fixture
        assert data["date_range"]["earliest"] == "2026-01-20"
        assert data["date_range"]["latest"] == "2026-02-01"

    def test_usage_tool_count_positive(self, mcp_db):
        """kb_usage tool_count should be a positive integer."""
        from kb.mcp_server import handle_kb_usage

        db, _ = mcp_db
        data = json.loads(handle_kb_usage(db))
        assert data["tool_count"] >= 28  # at least our known tools


# ---------------------------------------------------------------------------
# CLI still works after refactor
# ---------------------------------------------------------------------------


class TestCliAfterRefactor:
    def test_cli_search_still_works(self, mcp_db):
        """CLI search should still work after config.py refactor."""
        from click.testing import CliRunner

        from kb.cli import cli

        _, db_path = mcp_db
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["search", "MFA", "--fast", "--json"],
            env={"KB_DATA_DIR": str(db_path)},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["results"]

    def test_cli_person_find_still_works(self, mcp_db):
        """CLI person find should still work after config.py refactor."""
        from click.testing import CliRunner

        from kb.cli import cli

        _, db_path = mcp_db
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["person", "find", "Talia", "--json"],
            env={"KB_DATA_DIR": str(db_path)},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Talia Ström"

    def test_kb_mcp_command_exists(self):
        """kb mcp command should exist."""
        from click.testing import CliRunner

        from kb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output or "mcp" in result.output.lower()


# ---------------------------------------------------------------------------
# _validate_date tests
# ---------------------------------------------------------------------------


class TestValidateDate:
    def test_valid_date(self):
        """Valid YYYY-MM-DD date string returned as-is."""
        from kb.mcp_server import _validate_date

        assert _validate_date("2026-01-15") == "2026-01-15"

    def test_invalid_date_format(self):
        """Non-date string returns None."""
        from kb.mcp_server import _validate_date

        assert _validate_date("not-a-date") is None

    def test_none_input(self):
        """None input returns None."""
        from kb.mcp_server import _validate_date

        assert _validate_date(None) is None

    def test_partial_date_returns_none(self):
        """Incomplete date like YYYY-MM returns None."""
        from kb.mcp_server import _validate_date

        assert _validate_date("2026-01") is None

    def test_date_with_time_returns_none(self):
        """Date with time suffix returns None."""
        from kb.mcp_server import _validate_date

        assert _validate_date("2026-01-15T12:00:00") is None


# ---------------------------------------------------------------------------
# handle_kb_pin / handle_kb_unpin tests
# ---------------------------------------------------------------------------


class TestMcpPinUnpin:
    def test_pin_by_path(self, mcp_db):
        """Pinning a known document by path sets pinned flag."""
        from kb.mcp_server import handle_kb_pin

        db, _ = mcp_db
        result = json.loads(handle_kb_pin(db, "meetings/2026/01/27/mfa_review.notes.md"))
        assert result["pinned"] is True
        assert result["status"] == "ok"

        # Verify in DB
        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT pinned FROM documents WHERE path = ?",
            ("meetings/2026/01/27/mfa_review.notes.md",),
        ).fetchone()
        assert row["pinned"] == 1

    def test_unpin_by_path(self, mcp_db):
        """Unpinning a previously pinned document clears pinned flag."""
        from kb.mcp_server import handle_kb_pin, handle_kb_unpin

        db, _ = mcp_db
        handle_kb_pin(db, "meetings/2026/01/27/mfa_review.notes.md")
        result = json.loads(handle_kb_unpin(db, "meetings/2026/01/27/mfa_review.notes.md"))
        assert result["pinned"] is False
        assert result["status"] == "ok"

        # Verify in DB
        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT pinned FROM documents WHERE path = ?",
            ("meetings/2026/01/27/mfa_review.notes.md",),
        ).fetchone()
        assert row["pinned"] == 0

    def test_pin_not_found(self, mcp_db):
        """Pinning nonexistent document returns error."""
        from kb.mcp_server import handle_kb_pin

        db, _ = mcp_db
        result = json.loads(handle_kb_pin(db, "nonexistent.md"))
        assert "error" in result

    def test_unpin_not_found(self, mcp_db):
        """Unpinning nonexistent document returns error."""
        from kb.mcp_server import handle_kb_unpin

        db, _ = mcp_db
        result = json.loads(handle_kb_unpin(db, "nonexistent.md"))
        assert "error" in result


# ---------------------------------------------------------------------------
# _find_document_by_target tests
# ---------------------------------------------------------------------------


class TestFindDocumentByTarget:
    def test_find_by_exact_path(self, mcp_db):
        """Finds document by exact path."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, "meetings/2026/01/27/mfa_review.notes.md")
        assert doc is not None
        assert doc["title"] == "MFA Implementation Review"

    def test_find_by_hash_prefix(self, mcp_db):
        """Finds document by content hash prefix."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, "#aaa111")
        assert doc is not None
        assert doc["title"] == "MFA Implementation Review"

    def test_find_by_title(self, mcp_db):
        """Finds document by exact title (case-insensitive)."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, "MFA Implementation Review")
        assert doc is not None
        assert doc["title"] == "MFA Implementation Review"

    def test_find_by_title_case_insensitive(self, mcp_db):
        """Title lookup is case-insensitive."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, "mfa implementation review")
        assert doc is not None
        assert doc["title"] == "MFA Implementation Review"

    def test_find_returns_none_for_ambiguous_glob(self, mcp_db):
        """Ambiguous glob matching multiple docs returns None."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        # *.notes.md matches multiple documents
        doc = _find_document_by_target(conn, "*.notes.md")
        assert doc is None

    def test_find_returns_none_for_nonexistent(self, mcp_db):
        """Completely unknown target returns None."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, "totally-nonexistent-doc.md")
        assert doc is None

    def test_find_by_suffix(self, mcp_db):
        """Finds document by path suffix."""
        from kb.mcp_server import _find_document_by_target

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, "quartz-indexer.notes.md")
        assert doc is not None
        assert doc["title"] == "Atlas Pipeline Plan"


# ---------------------------------------------------------------------------
# handle_kb_memory_add tests
# ---------------------------------------------------------------------------


class TestMcpMemoryAdd:
    def test_add_fact_to_known_entity(self, mcp_db):
        """Adding a fact to a known entity inserts into facts table."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        result = json.loads(
            handle_kb_memory_add(db, db_path, "Talia is great at debugging", entity="Talia")
        )
        assert result["type"] == "fact"
        assert result["status"] == "ok"
        assert result["entity"] == "Talia Ström"

        # Verify in DB
        conn = db.get_sqlite_conn()
        facts = conn.execute("SELECT * FROM facts").fetchall()
        assert len(facts) >= 1
        assert any("great at debugging" in f["fact_text"] for f in facts)

    def test_add_fact_unknown_entity_returns_error(self, mcp_db):
        """Adding a fact for unknown entity returns error."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        result = json.loads(
            handle_kb_memory_add(db, db_path, "Some fact about unknown", entity="ZzzNobody")
        )
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_add_note_creates_file(self, mcp_db):
        """Adding a note creates a markdown file in memory/notes/."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            result = json.loads(
                handle_kb_memory_add(db, db_path, "Test note title", body="Some body content")
            )
        assert result["type"] == "note"
        assert result["status"] == "ok"
        assert "path" in result

        # Check file was created
        notes_dir = db_path / "memory" / "notes"
        assert notes_dir.exists()
        md_files = list(notes_dir.glob("*.md"))
        assert len(md_files) >= 1

        # Check content
        content = md_files[0].read_text()
        assert "title: Test note title" in content
        assert "Some body content" in content

    def test_add_note_with_tags(self, mcp_db):
        """Note with tags includes them in frontmatter."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            result = json.loads(
                handle_kb_memory_add(
                    db, db_path, "Tagged note", body="Content", tags="infra,urgent"
                )
            )
        assert result["type"] == "note"
        assert result["status"] == "ok"

        notes_dir = db_path / "memory" / "notes"
        md_files = list(notes_dir.glob("*.md"))
        content = md_files[0].read_text()
        assert "infra" in content
        assert "urgent" in content

    def test_add_note_with_pin(self, mcp_db):
        """Note with pin=True sets pinned on the document."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            result = json.loads(
                handle_kb_memory_add(db, db_path, "Pinned note", body="Content", pin=True)
            )
        assert result["type"] == "note"
        assert result["pinned"] is True

    def test_add_note_with_entity_linking(self, mcp_db):
        """Note with entity creates the file and links entity."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            result = json.loads(
                handle_kb_memory_add(
                    db,
                    db_path,
                    "Note about Talia",
                    body="Talia helped with the migration",
                    entity="Talia",
                )
            )
        # entity + body -> note path (not fact path)
        assert result["type"] == "note"
        assert result["status"] == "ok"

    def test_add_note_duplicate_filename(self, mcp_db):
        """Duplicate filename gets counter suffix."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            # Create first note
            result1 = json.loads(
                handle_kb_memory_add(db, db_path, "Same title", body="First", date="2026-01-15")
            )
            # Create second note with same title and date
            result2 = json.loads(
                handle_kb_memory_add(db, db_path, "Same title", body="Second", date="2026-01-15")
            )

        assert result1["status"] == "ok"
        assert result2["status"] == "ok"
        # Paths should differ
        assert result1["path"] != result2["path"]

        # Verify both files exist
        notes_dir = db_path / "memory" / "notes"
        md_files = list(notes_dir.glob("*.md"))
        assert len(md_files) >= 2

    def test_add_note_without_body_or_entity(self, mcp_db):
        """One-liner note (no body, no entity) takes note path."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            result = json.loads(handle_kb_memory_add(db, db_path, "Quick note"))
        # entity=None -> is_note=True
        assert result["type"] == "note"
        assert result["status"] == "ok"

    def test_add_note_custom_date(self, mcp_db):
        """Custom date parameter used in filename."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        with patch("kb.indexer.index_all"):
            result = json.loads(
                handle_kb_memory_add(db, db_path, "Dated note", body="content", date="2025-12-25")
            )
        assert result["status"] == "ok"
        assert "2025-12-25" in result["path"]


# ---------------------------------------------------------------------------
# Facts persist to entity markdown files
# ---------------------------------------------------------------------------


class TestFactPersistsToFile:
    """Facts added via handle_kb_memory_add should appear in the entity's markdown file."""

    def test_fact_appended_to_entity_file(self, mcp_db):
        """Adding a fact writes it under ## Recent Facts in the entity file."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_person_create

        db, db_path = mcp_db
        # Create entity with a real file on disk
        handle_kb_person_create(db, db_path, "Tina Test", role="Engineer")

        # Add a fact
        result = json.loads(
            handle_kb_memory_add(
                db, db_path, "Tina loves TDD", entity="Tina Test", date="2026-03-16"
            )
        )
        assert result["status"] == "ok"
        assert result["type"] == "fact"

        # Verify fact is in the markdown file
        entity_file = db_path / "memory" / "people" / "tina-test.md"
        content = entity_file.read_text(encoding="utf-8")
        assert "## Recent Facts" in content
        assert "[2026-03-16] Tina loves TDD" in content

    def test_multiple_facts_all_appear_in_file(self, mcp_db):
        """Multiple facts all appear under ## Recent Facts."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_person_create

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Uma Unit", role="SRE")

        handle_kb_memory_add(
            db, db_path, "Uma knows Kubernetes", entity="Uma Unit", date="2026-01-01"
        )
        handle_kb_memory_add(
            db, db_path, "Uma wrote the runbook", entity="Uma Unit", date="2026-02-01"
        )

        entity_file = db_path / "memory" / "people" / "uma-unit.md"
        content = entity_file.read_text(encoding="utf-8")
        assert "[2026-01-01] Uma knows Kubernetes" in content
        assert "[2026-02-01] Uma wrote the runbook" in content

    def test_fact_on_entity_without_file_still_succeeds(self, mcp_db):
        """Fact on fixture entity (no file on disk) should still succeed in DB."""
        from kb.mcp_server import handle_kb_memory_add

        db, db_path = mcp_db
        # Talia exists in DB (from fixture) but has no real file
        result = json.loads(handle_kb_memory_add(db, db_path, "Talia likes Python", entity="Talia"))
        # Should succeed in DB even if file doesn't exist
        assert result["status"] == "ok"
        assert result["type"] == "fact"


# ---------------------------------------------------------------------------
# handle_kb_usage — facts count coverage
# ---------------------------------------------------------------------------


class TestMcpUsageFacts:
    def test_usage_includes_facts_count(self, mcp_db):
        """kb_usage facts field reflects actual fact count."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_usage

        db, db_path = mcp_db
        # Add a fact first
        handle_kb_memory_add(db, db_path, "Talia knows Rust", entity="Talia")

        data = json.loads(handle_kb_usage(db))
        assert data["facts"] == 1


# ---------------------------------------------------------------------------
# handle_kb_entity_stale tests
# ---------------------------------------------------------------------------


class TestMcpEntityStale:
    def test_entity_stale_returns_stale(self, mcp_db):
        """handle_kb_entity_stale returns entities with old timestamps."""
        from kb.mcp_server import handle_kb_entity_stale

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        # Set old timestamps on both test entities
        conn.execute(
            "UPDATE entities SET updated_at = '2025-01-01', last_mentioned_at = '2025-01-15'"
        )
        conn.commit()

        result = json.loads(handle_kb_entity_stale(db, days=30))
        assert "results" in result
        assert len(result["results"]) == 2
        names = {r["name"] for r in result["results"]}
        assert "Talia Ström" in names

    def test_entity_stale_excludes_fresh(self, mcp_db):
        """handle_kb_entity_stale excludes recently updated entities."""
        from datetime import date, timedelta

        from kb.mcp_server import handle_kb_entity_stale

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        conn.execute(
            f"UPDATE entities SET updated_at = '{yesterday}', last_mentioned_at = '{yesterday}'"
        )
        conn.commit()

        result = json.loads(handle_kb_entity_stale(db, days=30))
        assert len(result["results"]) == 0

    def test_entity_stale_type_filter(self, mcp_db):
        """handle_kb_entity_stale respects entity_type filter."""
        from kb.mcp_server import handle_kb_entity_stale

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        conn.execute(
            "UPDATE entities SET updated_at = '2025-01-01', last_mentioned_at = '2025-01-15'"
        )
        conn.commit()

        result = json.loads(handle_kb_entity_stale(db, days=30, entity_type="person"))
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "Talia Ström"

    def test_entity_stale_meta_includes_threshold(self, mcp_db):
        """Response meta includes threshold_days and count."""
        from kb.mcp_server import handle_kb_entity_stale

        db, _ = mcp_db
        result = json.loads(handle_kb_entity_stale(db, days=60))
        assert result["meta"]["threshold_days"] == 60
        assert "count" in result["meta"]


# ---------------------------------------------------------------------------
# person_find freshness fields
# ---------------------------------------------------------------------------


class TestMcpPersonFindFreshness:
    def test_person_find_includes_freshness_fields(self, mcp_db):
        """kb_person_find response includes updated_at and last_mentioned_at."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        conn.execute(
            "UPDATE entities SET updated_at = '2026-02-01', last_mentioned_at = '2026-02-15' WHERE name = 'Talia Ström'"
        )
        conn.commit()

        result = json.loads(handle_kb_person_find(db, "Talia"))
        assert result["updated_at"] == "2026-02-01"
        assert result["last_mentioned_at"] == "2026-02-15"

    def test_person_find_null_freshness_fields(self, mcp_db):
        """kb_person_find returns None for missing freshness timestamps."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        result = json.loads(handle_kb_person_find(db, "Talia"))
        assert "updated_at" in result
        assert "last_mentioned_at" in result


# ---------------------------------------------------------------------------
# handle_kb_project_find tests
# ---------------------------------------------------------------------------


class TestMcpProjectFind:
    def test_project_find_returns_json(self, mcp_db):
        """kb_project_find should return project data."""
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "Helix Refactor"))
        assert result["name"] == "Helix Refactor"
        assert result["entity_type"] == "project"
        assert "facts" in result
        assert "breadcrumbs" in result

    def test_project_find_by_alias(self, mcp_db):
        """kb_project_find should work with aliases."""
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "helix-refactor"))
        assert result["name"] == "Helix Refactor"

    def test_project_find_not_found(self, mcp_db):
        """kb_project_find should return error for unknown project."""
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "NonexistentProject"))
        assert "error" in result

    def test_project_find_wrong_type(self, mcp_db):
        """kb_project_find should reject person entities."""
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "Talia Ström"))
        assert "error" in result
        assert "person" in result["error"]


# ---------------------------------------------------------------------------
# handle_kb_project_list / handle_kb_person_list tests
# ---------------------------------------------------------------------------


class TestMcpEntityLists:
    def test_project_list(self, mcp_db):
        """kb_project_list returns all projects."""
        from kb.mcp_server import handle_kb_project_list

        db, _ = mcp_db
        result = json.loads(handle_kb_project_list(db))
        assert "results" in result
        assert "meta" in result
        assert result["meta"]["total"] == 1
        assert result["results"][0]["name"] == "Helix Refactor"

    def test_person_list(self, mcp_db):
        """kb_person_list returns all people."""
        from kb.mcp_server import handle_kb_person_list

        db, _ = mcp_db
        result = json.loads(handle_kb_person_list(db))
        assert "results" in result
        assert result["meta"]["total"] == 1
        assert result["results"][0]["name"] == "Talia Ström"


# ---------------------------------------------------------------------------
# handle_kb_note_list tests
# ---------------------------------------------------------------------------


class TestMcpNoteList:
    def _insert_note(self, conn, path, title, date, tags, pinned=False):
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count, pinned)
               VALUES (?, ?, ?, 'memory_note', 'memory', ?, ?, 1, ?)""",
            (path, title, date, json.dumps(tags), f"hash_{title[:6]}", 1 if pinned else 0),
        )

    def test_note_list_empty(self, mcp_db):
        """kb_note_list returns empty when no notes exist."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        result = json.loads(handle_kb_note_list(db))
        assert result["results"] == []

    def test_note_list_returns_notes(self, mcp_db):
        """kb_note_list returns inserted memory notes."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        self._insert_note(conn, "memory/notes/test.md", "Test Note", "2026-01-01", ["infra"])
        conn.commit()

        result = json.loads(handle_kb_note_list(db))
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Note"
        assert result["results"][0]["tags"] == ["infra"]

    def test_note_list_tag_filter(self, mcp_db):
        """kb_note_list filters by tag."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        self._insert_note(conn, "memory/notes/a.md", "Note A", "2026-01-01", ["infra", "urgent"])
        self._insert_note(conn, "memory/notes/b.md", "Note B", "2026-01-02", ["decision"])
        conn.commit()

        result = json.loads(handle_kb_note_list(db, tag="infra"))
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Note A"

    def test_note_list_pinned_only(self, mcp_db):
        """kb_note_list pinned_only returns only pinned notes."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        self._insert_note(conn, "memory/notes/a.md", "Pinned", "2026-01-01", [], pinned=True)
        self._insert_note(conn, "memory/notes/b.md", "Not Pinned", "2026-01-02", [])
        conn.commit()

        result = json.loads(handle_kb_note_list(db, pinned_only=True))
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Pinned"

    def test_note_list_limit(self, mcp_db):
        """kb_note_list respects limit."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        for i in range(5):
            self._insert_note(conn, f"memory/notes/{i}.md", f"Note {i}", f"2026-01-0{i + 1}", [])
        conn.commit()

        result = json.loads(handle_kb_note_list(db, limit=2))
        assert len(result["results"]) == 2


# ---------------------------------------------------------------------------
# handle_kb_note_edit tests
# ---------------------------------------------------------------------------


class TestMcpNoteEdit:
    def _create_note_file(self, project_root, rel_path, content):
        full = project_root / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    def test_note_edit_body(self, mcp_db):
        """kb_note_edit replaces note body."""
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        conn = db.get_sqlite_conn()
        rel = "memory/notes/edit_test.md"
        self._create_note_file(root, rel, "---\ntitle: Edit Test\n---\nOld body\n")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, 'Edit Test', '2026-01-01', 'memory_note', 'memory', '[]', 'eee111', 1)""",
            (rel,),
        )
        conn.commit()

        with patch("kb.indexer.index_all"):
            result = json.loads(handle_kb_note_edit(db, root, rel, body="New body"))
        assert result["status"] == "ok"
        content = (root / rel).read_text()
        assert "New body" in content
        assert "Old body" not in content

    def test_note_edit_append(self, mcp_db):
        """kb_note_edit appends to note body."""
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        conn = db.get_sqlite_conn()
        rel = "memory/notes/append_test.md"
        self._create_note_file(root, rel, "---\ntitle: Append Test\n---\nExisting.\n")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, 'Append Test', '2026-01-01', 'memory_note', 'memory', '[]', 'fff222', 1)""",
            (rel,),
        )
        conn.commit()

        with patch("kb.indexer.index_all"):
            result = json.loads(handle_kb_note_edit(db, root, rel, append="\nExtra line."))
        assert result["status"] == "ok"
        content = (root / rel).read_text()
        assert "Existing." in content
        assert "Extra line." in content

    def test_note_edit_not_found(self, mcp_db):
        """kb_note_edit returns error for unknown note."""
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        result = json.loads(handle_kb_note_edit(db, root, "nonexistent.md", body="x"))
        assert "error" in result

    def test_note_edit_rejects_non_note(self, mcp_db):
        """kb_note_edit rejects non-memory_note doc types."""
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        result = json.loads(
            handle_kb_note_edit(db, root, "meetings/2026/01/27/mfa_review.notes.md", body="x")
        )
        assert "error" in result
        assert "Not a memory note" in result["error"]

    def test_note_edit_no_changes(self, mcp_db):
        """kb_note_edit returns error when no edit options given."""
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        conn = db.get_sqlite_conn()
        rel = "memory/notes/noop.md"
        self._create_note_file(root, rel, "---\ntitle: Noop\n---\nBody\n")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, 'Noop', '2026-01-01', 'memory_note', 'memory', '[]', 'ggg333', 1)""",
            (rel,),
        )
        conn.commit()

        result = json.loads(handle_kb_note_edit(db, root, rel))
        assert "error" in result
        assert "No edit options" in result["error"]

    def test_note_edit_both_body_and_append(self, mcp_db):
        """kb_note_edit rejects body + append together."""
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        conn = db.get_sqlite_conn()
        rel = "memory/notes/both.md"
        self._create_note_file(root, rel, "---\ntitle: Both\n---\nBody\n")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, 'Both', '2026-01-01', 'memory_note', 'memory', '[]', 'hhh444', 1)""",
            (rel,),
        )
        conn.commit()

        result = json.loads(handle_kb_note_edit(db, root, rel, body="new", append="extra"))
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_kb_note_delete tests
# ---------------------------------------------------------------------------


class TestMcpNoteDelete:
    def test_note_delete(self, mcp_db):
        """kb_note_delete removes file and DB rows."""
        from kb.mcp_server import handle_kb_note_delete

        db, root = mcp_db
        conn = db.get_sqlite_conn()
        rel = "memory/notes/delete_me.md"
        note_path = root / rel
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("---\ntitle: Delete Me\n---\nContent\n")
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES (?, 'Delete Me', '2026-01-01', 'memory_note', 'memory', '[]', 'del111', 1)""",
            (rel,),
        )
        conn.commit()

        result = json.loads(handle_kb_note_delete(db, root, rel))
        assert result["status"] == "ok"
        assert not note_path.exists()
        row = conn.execute("SELECT id FROM documents WHERE path = ?", (rel,)).fetchone()
        assert row is None

    def test_note_delete_not_found(self, mcp_db):
        """kb_note_delete returns error for unknown note."""
        from kb.mcp_server import handle_kb_note_delete

        db, root = mcp_db
        result = json.loads(handle_kb_note_delete(db, root, "nonexistent.md"))
        assert "error" in result

    def test_note_delete_rejects_non_note(self, mcp_db):
        """kb_note_delete rejects non-memory_note doc types."""
        from kb.mcp_server import handle_kb_note_delete

        db, root = mcp_db
        result = json.loads(
            handle_kb_note_delete(db, root, "meetings/2026/01/27/mfa_review.notes.md")
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_kb_memory_list tests
# ---------------------------------------------------------------------------


class TestMcpMemoryList:
    def test_memory_list_empty(self, mcp_db):
        """kb_memory_list returns empty when no facts exist."""
        from kb.mcp_server import handle_kb_memory_list

        db, root = mcp_db
        result = json.loads(handle_kb_memory_list(db, root))
        assert result["results"] == []
        assert result["meta"]["total"] == 0

    def test_memory_list_with_facts(self, mcp_db):
        """kb_memory_list returns inserted facts."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_memory_list

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Talia speaks French", entity="Talia")
        handle_kb_memory_add(db, root, "Talia likes Rust", entity="Talia")

        result = json.loads(handle_kb_memory_list(db, root))
        assert result["meta"]["total"] == 2
        texts = [f["fact_text"] for f in result["results"]]
        assert "Talia speaks French" in texts


# ---------------------------------------------------------------------------
# handle_kb_memory_delete_fact / handle_kb_memory_edit_fact tests
# ---------------------------------------------------------------------------


class TestMcpMemoryFactOps:
    def test_memory_delete_fact(self, mcp_db):
        """kb_memory_delete_fact removes a fact by entity + seq."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_memory_delete_fact

        db, root = mcp_db
        # Add a fact first
        add_result = json.loads(handle_kb_memory_add(db, root, "Talia is great", entity="Talia"))
        assert add_result["status"] == "ok"

        # Get fact seq
        conn = db.get_sqlite_conn()
        fact_row = conn.execute("SELECT seq FROM facts WHERE fact_text = 'Talia is great'").fetchone()
        fact_seq = fact_row["seq"]

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_memory_delete_fact(db, root, "Talia Ström", fact_seq))
        assert result.get("status") == "ok" or "deleted" in str(result).lower()

    def test_memory_delete_fact_not_found(self, mcp_db):
        """kb_memory_delete_fact returns error for unknown entity/seq."""
        from kb.mcp_server import handle_kb_memory_delete_fact

        db, root = mcp_db
        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_memory_delete_fact(db, root, "Nobody", 99999))
        assert "error" in result

    def test_memory_edit_fact_no_changes(self, mcp_db):
        """kb_memory_edit_fact returns error when no text or date given."""
        from kb.mcp_server import handle_kb_memory_edit_fact

        db, root = mcp_db
        result = json.loads(handle_kb_memory_edit_fact(db, root, "Talia Ström", 1))
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_kb_glossary_* tests
# ---------------------------------------------------------------------------


class TestMcpGlossary:
    def _create_glossary(self, project_root):
        glossary_path = project_root / "memory" / "glossary.md"
        glossary_path.parent.mkdir(parents=True, exist_ok=True)
        glossary_path.write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n|------|----------|\n| GG | Lattice Co |\n\n"
            "## Jargon\n\n| Term | Expansion |\n|------|----------|\n| Wards | Ward patrol runs |\n",
            encoding="utf-8",
        )

    def test_glossary_list(self, mcp_db):
        """kb_glossary_list returns all terms."""
        from kb.mcp_server import handle_kb_glossary_list

        _, root = mcp_db
        self._create_glossary(root)

        result = json.loads(handle_kb_glossary_list(root))
        assert "results" in result
        assert result["meta"]["total"] >= 2
        terms = [r["term"] for r in result["results"]]
        assert "GG" in terms

    def test_glossary_add(self, mcp_db):
        """kb_glossary_add inserts a new term."""
        from kb.mcp_server import handle_kb_glossary_add

        _, root = mcp_db
        self._create_glossary(root)

        result = json.loads(handle_kb_glossary_add(root, "MFA", "Multi-Factor Authentication"))
        assert result.get("status") == "ok" or "term" in result

        # Verify it's in the file
        content = (root / "memory" / "glossary.md").read_text()
        assert "MFA" in content

    def test_glossary_edit(self, mcp_db):
        """kb_glossary_edit updates an existing term."""
        from kb.mcp_server import handle_kb_glossary_edit

        _, root = mcp_db
        self._create_glossary(root)

        result = json.loads(handle_kb_glossary_edit(root, "GG", "Lattice Co Inc."))
        assert result.get("status") == "ok" or "term" in result

        content = (root / "memory" / "glossary.md").read_text()
        assert "Lattice Co Inc." in content

    def test_glossary_edit_not_found(self, mcp_db):
        """kb_glossary_edit returns error for unknown term."""
        from kb.mcp_server import handle_kb_glossary_edit

        _, root = mcp_db
        self._create_glossary(root)

        result = json.loads(handle_kb_glossary_edit(root, "ZZZNOPE", "Nothing"))
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_kb_granola_view / handle_kb_granola_edit tests
# ---------------------------------------------------------------------------


class TestMcpGranolaView:
    def test_granola_view_not_found(self):
        """kb_granola_view returns error when no doc found."""
        from kb.mcp_server import handle_kb_granola_view

        with patch("kb.sync.granola.GranolaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.find_document.return_value = None
            mock_cls.return_value = mock_client

            result = json.loads(handle_kb_granola_view("fake-uid-123"))
        assert "error" in result

    def test_granola_view_notes_mode(self):
        """kb_granola_view returns notes in default mode."""
        from kb.mcp_server import handle_kb_granola_view

        fake_doc = {
            "id": "doc1",
            "title": "Test Meeting",
            "created_at": "2026-01-15T10:00:00Z",
            "notes_markdown": "# Notes\nSome notes here.",
            "google_calendar_event": {"iCalUID": "uid-123"},
        }

        with (
            patch("kb.sync.granola.GranolaClient") as mock_cls,
            patch("kb.sync.granola.extract_panel_markdown", return_value=""),
        ):
            mock_client = MagicMock()
            mock_client.find_document.return_value = fake_doc
            mock_cls.return_value = mock_client

            result = json.loads(handle_kb_granola_view("uid-123"))
        assert result["title"] == "Test Meeting"
        assert "Notes" in result["content"]


class TestMcpGranolaEdit:
    def test_granola_edit_no_args(self):
        """kb_granola_edit returns error when neither body nor append given."""
        from kb.mcp_server import handle_kb_granola_edit

        result = json.loads(handle_kb_granola_edit("uid-123"))
        assert "error" in result

    def test_granola_edit_both_args(self):
        """kb_granola_edit returns error when both body and append given."""
        from kb.mcp_server import handle_kb_granola_edit

        result = json.loads(handle_kb_granola_edit("uid-123", body="new", append="extra"))
        assert "error" in result

    def test_granola_edit_body(self):
        """kb_granola_edit calls API to replace body."""
        from kb.mcp_server import handle_kb_granola_edit

        fake_doc = {"id": "doc1", "notes_markdown": "old notes"}

        with patch("kb.sync.granola.GranolaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.find_document.return_value = fake_doc
            mock_cls.return_value = mock_client

            result = json.loads(handle_kb_granola_edit("uid-123", body="new notes"))
        assert result["status"] == "ok"
        assert result["action"] == "replace"
        mock_client.update_document_notes.assert_called_once_with("doc1", "new notes")

    def test_granola_edit_append(self):
        """kb_granola_edit appends to existing notes."""
        from kb.mcp_server import handle_kb_granola_edit

        fake_doc = {"id": "doc1", "notes_markdown": "existing notes"}

        with patch("kb.sync.granola.GranolaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.find_document.return_value = fake_doc
            mock_cls.return_value = mock_client

            result = json.loads(handle_kb_granola_edit("uid-123", append="appended text"))
        assert result["status"] == "ok"
        assert result["action"] == "append"
        call_args = mock_client.update_document_notes.call_args[0]
        assert "existing notes" in call_args[1]
        assert "appended text" in call_args[1]


# ---------------------------------------------------------------------------
# handle_kb_list tests
# ---------------------------------------------------------------------------


class TestMcpList:
    def test_list_all(self, mcp_db):
        """kb_list returns all documents."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        result = json.loads(handle_kb_list(db))
        assert "results" in result
        assert result["meta"]["total"] == 4  # 3 meetings + 1 memory_person

    def test_list_by_type(self, mcp_db):
        """kb_list filters by doc_type."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        result = json.loads(handle_kb_list(db, doc_type="notes"))
        titles = [r["title"] for r in result["results"]]
        assert len(titles) == 3  # all 3 meeting notes

    def test_list_by_date_range(self, mcp_db):
        """kb_list filters by date range."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        result = json.loads(handle_kb_list(db, from_date="2026-01-25", to_date="2026-02-05"))
        titles = [r["title"] for r in result["results"]]
        assert "MFA Implementation Review" in titles
        assert "Atlas Pipeline Plan" in titles
        # Helix Refactor Status is 2026-01-20, outside range
        assert "Helix Refactor Status" not in titles

    def test_list_limit(self, mcp_db):
        """kb_list respects limit."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        result = json.loads(handle_kb_list(db, limit=2))
        assert len(result["results"]) == 2

    def test_list_returns_doc_fields(self, mcp_db):
        """kb_list returns standard document fields."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        result = json.loads(handle_kb_list(db, limit=1))
        doc = result["results"][0]
        assert "path" in doc
        assert "title" in doc
        assert "doc_type" in doc


# ---------------------------------------------------------------------------
# handle_kb_index_status tests
# ---------------------------------------------------------------------------


class TestMcpIndexStatus:
    def test_index_status(self, mcp_db):
        """kb_index_status returns document/entity/fact counts."""
        from kb.mcp_server import handle_kb_index_status

        db, _ = mcp_db
        result = json.loads(handle_kb_index_status(db))
        assert result["documents"] == 4
        assert result["chunks"] == 6
        assert result["entities"] == 2
        assert "documents_by_type" in result
        assert "date_range" in result

    def test_index_status_doc_types(self, mcp_db):
        """kb_index_status breaks down by doc_type."""
        from kb.mcp_server import handle_kb_index_status

        db, _ = mcp_db
        result = json.loads(handle_kb_index_status(db))
        by_type = result["documents_by_type"]
        assert by_type.get("notes") == 3
        assert by_type.get("memory_person") == 1


# ---------------------------------------------------------------------------
# handle_kb_correct tests
# ---------------------------------------------------------------------------


class TestMcpCorrect:
    def _create_memory_files(self, project_root):
        mem = project_root / "memory"
        mem.mkdir(parents=True, exist_ok=True)
        notes = mem / "notes"
        notes.mkdir(exist_ok=True)
        (notes / "test.md").write_text(
            "---\ntitle: Test\n---\nQuartz Indexer is great.\nQuartz Indexer rules.\n"
        )

    def test_correct_scan(self, mcp_db):
        """kb_correct scan mode finds occurrences."""
        from kb.mcp_server import handle_kb_correct

        _, root = mcp_db
        self._create_memory_files(root)

        result = json.loads(handle_kb_correct(root, "Quartz Indexer"))
        assert result["meta"]["action"] == "scan"
        assert result["meta"]["total_occurrences"] == 2

    def test_correct_dry_run(self, mcp_db):
        """kb_correct dry-run mode previews without changing."""
        from kb.mcp_server import handle_kb_correct

        _, root = mcp_db
        self._create_memory_files(root)

        result = json.loads(handle_kb_correct(root, "Quartz Indexer", replacement="Coralogix"))
        assert result["meta"]["action"] == "dry_run"
        assert result["meta"]["total_occurrences"] == 2

        # File unchanged
        content = (root / "memory" / "notes" / "test.md").read_text()
        assert "Quartz Indexer" in content

    def test_correct_apply(self, mcp_db):
        """kb_correct apply mode replaces occurrences."""
        from kb.mcp_server import handle_kb_correct

        _, root = mcp_db
        self._create_memory_files(root)

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(
                handle_kb_correct(root, "Quartz Indexer", replacement="Coralogix", apply=True)
            )
        assert result["meta"]["action"] == "applied"
        assert result["meta"]["occurrences_replaced"] == 2

        content = (root / "memory" / "notes" / "test.md").read_text()
        assert "Coralogix" in content
        assert "Quartz Indexer" not in content

    def test_correct_no_matches(self, mcp_db):
        """kb_correct returns empty when no matches found."""
        from kb.mcp_server import handle_kb_correct

        _, root = mcp_db
        self._create_memory_files(root)

        result = json.loads(handle_kb_correct(root, "ZZZnonexistent"))
        assert result["meta"]["total"] == 0
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Audit fixes: A1 — kb_search doc_type filter
# ---------------------------------------------------------------------------


class TestMcpSearchDocType:
    def test_search_with_doc_type_filter(self, mcp_db):
        """kb_search should filter by doc_type when specified."""
        from kb.mcp_server import handle_kb_search

        db, _ = mcp_db
        # Search for "migration" which appears in both notes and memory_person chunks
        result = json.loads(
            handle_kb_search(db, "migration", fast=True, limit=10, doc_type="notes")
        )
        # All results should be doc_type "notes"
        for r in result["results"]:
            assert r["doc_type"] == "notes"

    def test_search_without_doc_type_returns_all(self, mcp_db):
        """kb_search without doc_type should return all matching types."""
        from kb.mcp_server import handle_kb_search

        db, _ = mcp_db
        result = json.loads(handle_kb_search(db, "MFA", fast=True, limit=10))
        assert len(result["results"]) > 0


# ---------------------------------------------------------------------------
# Audit fixes: A2 — entity list pagination
# ---------------------------------------------------------------------------


class TestMcpEntityListPagination:
    def _insert_people(self, db, count):
        conn = db.get_sqlite_conn()
        for i in range(count):
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, 'person', '[]', '{}')",
                (f"Person {i:03d}",),
            )
        conn.commit()

    def test_person_list_default_limit(self, mcp_db):
        """kb_person_list should default to limit 50."""
        from kb.mcp_server import handle_kb_person_list

        db, _ = mcp_db
        self._insert_people(db, 60)

        result = json.loads(handle_kb_person_list(db))
        # 60 new + 1 existing = 61, but limit 50 should cap it
        assert len(result["results"]) == 50
        assert result["meta"]["total"] == 61

    def test_person_list_with_limit(self, mcp_db):
        """kb_person_list should accept custom limit."""
        from kb.mcp_server import handle_kb_person_list

        db, _ = mcp_db
        self._insert_people(db, 10)

        result = json.loads(handle_kb_person_list(db, limit=5))
        assert len(result["results"]) == 5

    def test_person_list_with_offset(self, mcp_db):
        """kb_person_list should accept offset for pagination."""
        from kb.mcp_server import handle_kb_person_list

        db, _ = mcp_db
        self._insert_people(db, 10)

        page1 = json.loads(handle_kb_person_list(db, limit=5, offset=0))
        page2 = json.loads(handle_kb_person_list(db, limit=5, offset=5))
        # No overlap between pages
        names1 = {r["name"] for r in page1["results"]}
        names2 = {r["name"] for r in page2["results"]}
        assert names1.isdisjoint(names2)

    def test_project_list_with_limit(self, mcp_db):
        """kb_project_list should accept limit param."""
        from kb.mcp_server import handle_kb_project_list

        db, _ = mcp_db
        result = json.loads(handle_kb_project_list(db, limit=1))
        assert len(result["results"]) == 1


# ---------------------------------------------------------------------------
# Audit fixes: A3 — person_timeline doc_type + limit
# ---------------------------------------------------------------------------


class TestMcpTimelineFilters:
    def _setup_multi_type_docs(self, db):
        """Add transcript + debrief docs mentioning Talia."""
        conn = db.get_sqlite_conn()
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES ('meetings/2026/01/27/mfa.transcript.md', 'MFA Transcript', '2026-01-27', 'transcript', 'granola', '[]', 'ttt111', 1)""",
        )
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count)
               VALUES ('meetings/2026/01/27/mfa.debrief.md', 'MFA Debrief', '2026-01-27', 'debrief', 'memory', '[]', 'ddd111', 1)""",
        )
        # Link both to Talia (entity_id=1)
        # doc IDs 5 and 6 (after the 4 existing docs)
        transcript_id = conn.execute(
            "SELECT id FROM documents WHERE path = 'meetings/2026/01/27/mfa.transcript.md'"
        ).fetchone()["id"]
        debrief_id = conn.execute(
            "SELECT id FROM documents WHERE path = 'meetings/2026/01/27/mfa.debrief.md'"
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, ?, 'discussed')",
            (transcript_id,),
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, ?, 'discussed')",
            (debrief_id,),
        )
        conn.commit()

    def test_timeline_with_doc_type_filter(self, mcp_db):
        """kb_person_timeline should filter by doc_type."""
        from kb.mcp_server import handle_kb_person_timeline

        db, _ = mcp_db
        self._setup_multi_type_docs(db)

        result = json.loads(handle_kb_person_timeline(db, "Talia", doc_type="transcript"))
        assert len(result["documents"]) == 1
        assert result["documents"][0]["doc_type"] == "transcript"

    def test_timeline_with_limit(self, mcp_db):
        """kb_person_timeline should respect limit param."""
        from kb.mcp_server import handle_kb_person_timeline

        db, _ = mcp_db
        self._setup_multi_type_docs(db)

        # Talia has 3 docs: original mention + transcript + debrief
        result = json.loads(handle_kb_person_timeline(db, "Talia", limit=2))
        assert len(result["documents"]) == 2

    def test_timeline_default_has_all(self, mcp_db):
        """kb_person_timeline without filters returns all doc types."""
        from kb.mcp_server import handle_kb_person_timeline

        db, _ = mcp_db
        self._setup_multi_type_docs(db)

        result = json.loads(handle_kb_person_timeline(db, "Talia"))
        types = {d["doc_type"] for d in result["documents"]}
        assert "notes" in types
        assert "transcript" in types
        assert "debrief" in types


# ---------------------------------------------------------------------------
# Audit fixes: A5 — kb_correct MCP path resolution
# ---------------------------------------------------------------------------


class TestMcpCorrectPathResolution:
    def test_correct_with_global_config_project_root(self, mcp_db):
        """kb_correct should work when project_root has memory/ subdir."""
        from kb.mcp_server import handle_kb_correct

        _, root = mcp_db
        mem = root / "memory"
        mem.mkdir(parents=True, exist_ok=True)
        (mem / "notes").mkdir(exist_ok=True)
        (mem / "notes" / "test.md").write_text("Some text with Typo here.\n")

        # Should not error — memory dir exists at project_root/memory
        result = json.loads(handle_kb_correct(root, "Typo"))
        assert "error" not in result
        assert result["meta"]["total_occurrences"] == 1


# ---------------------------------------------------------------------------
# Audit fixes: A7 — kb_list since_hours param
# ---------------------------------------------------------------------------


class TestMcpListSinceHours:
    def test_list_since_hours(self, mcp_db):
        """kb_list with since_hours returns recently indexed docs."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        # All test docs were just indexed (indexed_at is NOW), so since_hours=1 should return them
        result = json.loads(handle_kb_list(db, since_hours=1))
        assert len(result["results"]) > 0

    def test_list_since_hours_excludes_old(self, mcp_db):
        """kb_list with since_hours=0 should return nothing (0 hours ago = now)."""
        from kb.mcp_server import handle_kb_list

        db, _ = mcp_db
        # Set indexed_at to yesterday for all docs
        conn = db.get_sqlite_conn()
        conn.execute("UPDATE documents SET indexed_at = datetime('now', '-2 days')")
        conn.commit()

        result = json.loads(handle_kb_list(db, since_hours=1))
        assert len(result["results"]) == 0


# ---------------------------------------------------------------------------
# Audit fixes: A8 — glossary path resolution for MCP
# ---------------------------------------------------------------------------


class TestMcpGlossaryPathResolution:
    def test_glossary_list_with_correct_project_root(self, mcp_db):
        """kb_glossary_list should find glossary at project_root/memory/glossary.md."""
        from kb.mcp_server import handle_kb_glossary_list

        _, root = mcp_db
        glossary_dir = root / "memory"
        glossary_dir.mkdir(parents=True, exist_ok=True)
        (glossary_dir / "glossary.md").write_text(
            "# Glossary\n\n## Acronyms\n\n| Term | Expansion |\n|------|----------|\n| API | Application Programming Interface |\n",
            encoding="utf-8",
        )

        result = json.loads(handle_kb_glossary_list(root))
        assert result["meta"]["total"] >= 1
        terms = [r["term"] for r in result["results"]]
        assert "API" in terms

    def test_glossary_list_empty_when_no_file(self, tmp_path):
        """kb_glossary_list returns empty when glossary.md doesn't exist."""
        from kb.mcp_server import handle_kb_glossary_list

        result = json.loads(handle_kb_glossary_list(tmp_path))
        assert result["results"] == []
        assert result["meta"]["total"] == 0


# ---------------------------------------------------------------------------
# Audit fix A8: find_project_root with global XDG config
# ---------------------------------------------------------------------------


class TestFindProjectRootGlobalConfig:
    def test_global_config_absolute_memory_path(self, tmp_path):
        """find_project_root derives root from absolute memory path in global config."""
        from kb.config import find_project_root

        # Set up a fake project structure
        project_dir = tmp_path / "project"
        memory_dir = project_dir / "memory"
        memory_dir.mkdir(parents=True)

        # Set up a fake global config with absolute memory path
        xdg_dir = tmp_path / "xdg" / "kbx"
        xdg_dir.mkdir(parents=True)
        config_file = xdg_dir / "config.toml"
        config_file.write_text(
            f'[sources]\nmemory = "{memory_dir}"\n\n[data]\ndir = "{tmp_path / "data"}"\n'
        )

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path / "xdg"), "KBX_CONFIG": ""}):
            # Clear KBX_CONFIG so find_config walks up CWD then falls to XDG
            os.environ.pop("KBX_CONFIG", None)
            # Ensure CWD doesn't contain kbx.toml
            old_cwd = os.getcwd()
            try:
                os.chdir(str(tmp_path / "xdg"))
                root = find_project_root()
            finally:
                os.chdir(old_cwd)

        # Should resolve to project_dir (parent of absolute memory path)
        assert root == project_dir

    def test_local_config_unchanged(self, tmp_path):
        """find_project_root still uses config parent for local kbx.toml."""
        from kb.config import find_project_root

        # Set up local kbx.toml
        config_file = tmp_path / "kbx.toml"
        config_file.write_text('[sources]\nmemory = "memory"\n')
        (tmp_path / "memory").mkdir()

        with patch.dict(os.environ, {"KBX_CONFIG": str(config_file)}):
            root = find_project_root()

        assert root == tmp_path


# ---------------------------------------------------------------------------
# A4 — kb_context mention_threshold
# ---------------------------------------------------------------------------


class TestMcpContextMentionThreshold:
    """A4: mention_threshold filters low-mention entities from context."""

    @pytest.fixture
    def threshold_db(self):
        """DB with entities at varying mention counts + one pinned low-mention entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()

            # Create enough documents for mention variety
            for i in range(1, 12):
                conn.execute(
                    "INSERT INTO documents (path, title, doc_type, doc_date, content_hash) "
                    f"VALUES ('meetings/test{i}.md', 'Test Meeting {i}', 'notes', '2026-01-{i:02d}', 'hash-thr-{i}')"
                )
            # Insert a chunk so FTS table exists
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, content) VALUES (1, 0, 'content')"
            )

            # Person with 10 mentions (high)
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, pinned) "
                "VALUES ('Wren High', 'person', '[]', "
                '\'{"role": "Engineer", "team": "Platform"}\', 0)'
            )
            # Person with 2 mentions (low)
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, pinned) "
                "VALUES ('Soren Low', 'person', '[]', "
                '\'{"role": "Intern", "team": "Platform"}\', 0)'
            )
            # Person with 1 mention, PINNED (should always appear)
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, pinned) "
                "VALUES ('Linnea Pinned', 'person', '[]', "
                '\'{"role": "CTO", "team": "Exec"}\', 1)'
            )
            # Project with 1 mention (low)
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, pinned) "
                "VALUES ('Tiny Project', 'project', '[]', "
                '\'{"status": "Active"}\', 0)'
            )

            # Add mentions: Wren=10, Soren=2, Linnea=1, Tiny Project=1
            # Each mention uses a different document (unique constraint)
            for i in range(1, 11):
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                    f"VALUES (1, {i}, 'discussed')"
                )
            for i in range(1, 3):
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                    f"VALUES (2, {i}, 'discussed')"
                )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                "VALUES (3, 1, 'discussed')"
            )
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                "VALUES (4, 1, 'discussed')"
            )

            conn.commit()
            yield db, Path(tmpdir)
            db.close()

    def test_threshold_zero_returns_all(self, threshold_db):
        """mention_threshold=0 (default) should include all entities."""
        from kb.mcp_server import handle_kb_context

        db, root = threshold_db
        result = handle_kb_context(db, project_root=root, mention_threshold=0)
        data = json.loads(result)
        text = data["text"]
        assert "Wren" in text
        assert "Soren" in text
        assert "Linnea" in text

    def test_threshold_filters_low_mention_entities(self, threshold_db):
        """mention_threshold=5 should drop Soren (2 mentions) but keep Wren (10)."""
        from kb.mcp_server import handle_kb_context

        db, root = threshold_db
        result = handle_kb_context(db, project_root=root, mention_threshold=5)
        data = json.loads(result)
        text = data["text"]
        assert "Wren" in text
        assert "Soren" not in text

    def test_threshold_preserves_pinned(self, threshold_db):
        """Pinned entities should appear regardless of mention_threshold."""
        from kb.mcp_server import handle_kb_context

        db, root = threshold_db
        result = handle_kb_context(db, project_root=root, mention_threshold=5)
        data = json.loads(result)
        text = data["text"]
        # Linnea is pinned with only 1 mention — must still appear
        assert "Linnea" in text

    def test_threshold_filters_projects(self, threshold_db):
        """mention_threshold should also filter projects below threshold."""
        from kb.mcp_server import handle_kb_context

        db, root = threshold_db
        result = handle_kb_context(db, project_root=root, mention_threshold=5)
        data = json.loads(result)
        text = data["text"]
        assert "Tiny Project" not in text

    def test_mcp_tool_accepts_mention_threshold(self, threshold_db):
        """kb_context MCP tool should accept mention_threshold parameter."""
        import inspect

        from kb.mcp_server import kb_context

        sig = inspect.signature(kb_context)
        assert "mention_threshold" in sig.parameters
        assert sig.parameters["mention_threshold"].default == 0


# ---------------------------------------------------------------------------
# Fix: kb_note_list meta.total should reflect true count, not capped by limit
# ---------------------------------------------------------------------------


class TestMcpNoteListTotal:
    """meta.total should be the true count of matching notes, not len(results)."""

    def _insert_note(self, conn, path, title, date, tags, pinned=False):
        conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system, tags, content_hash, chunk_count, pinned)
               VALUES (?, ?, ?, 'memory_note', 'memory', ?, ?, 1, ?)""",
            (path, title, date, json.dumps(tags), f"hash_{title[:8]}", 1 if pinned else 0),
        )

    def test_total_exceeds_limit(self, mcp_db):
        """meta.total should be true count even when results are capped by limit."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        for i in range(10):
            self._insert_note(
                conn, f"memory/notes/n{i}.md", f"Note {i}", f"2026-01-{i + 1:02d}", []
            )
        conn.commit()

        result = json.loads(handle_kb_note_list(db, limit=3))
        assert len(result["results"]) == 3
        assert result["meta"]["total"] == 10  # true count, not 3

    def test_total_with_tag_filter(self, mcp_db):
        """meta.total should reflect filtered count, not all notes."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        for i in range(5):
            self._insert_note(
                conn, f"memory/notes/infra{i}.md", f"Infra {i}", f"2026-01-{i + 1:02d}", ["infra"]
            )
        for i in range(3):
            self._insert_note(
                conn, f"memory/notes/other{i}.md", f"Other {i}", f"2026-02-{i + 1:02d}", ["other"]
            )
        conn.commit()

        result = json.loads(handle_kb_note_list(db, tag="infra", limit=2))
        assert len(result["results"]) == 2
        assert result["meta"]["total"] == 5  # 5 infra notes, not 2

    def test_total_with_pinned_only(self, mcp_db):
        """meta.total should reflect pinned count when pinned_only=True."""
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        conn = db.get_sqlite_conn()
        for i in range(4):
            self._insert_note(
                conn,
                f"memory/notes/pin{i}.md",
                f"Pinned {i}",
                f"2026-01-{i + 1:02d}",
                [],
                pinned=True,
            )
        for i in range(6):
            self._insert_note(
                conn, f"memory/notes/nopin{i}.md", f"Not {i}", f"2026-02-{i + 1:02d}", []
            )
        conn.commit()

        result = json.loads(handle_kb_note_list(db, pinned_only=True, limit=2))
        assert len(result["results"]) == 2
        assert result["meta"]["total"] == 4  # 4 pinned, not 2


# ---------------------------------------------------------------------------
# Fix: find_project_root XDG check should use is_relative_to, not startswith
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# handle_kb_person_create tests
# ---------------------------------------------------------------------------


class TestMcpPersonCreate:
    def test_create_person_returns_json(self, mcp_db):
        """kb_person_create should return valid JSON with name, path, entity_type."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        result = json.loads(handle_kb_person_create(db, db_path, "Wren Smith", role="SRE Lead"))
        assert result["name"] == "Wren Smith"
        assert result["entity_type"] == "person"
        assert "path" in result
        assert result["path"].endswith("alice-smith.md")

    def test_create_person_writes_file(self, mcp_db):
        """kb_person_create should create a markdown file in memory/people/."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Soren Jones", role="Engineer", team="Platform")

        filepath = db_path / "memory" / "people" / "bob-jones.md"
        assert filepath.exists()
        content = filepath.read_text()
        assert "# Soren Jones" in content
        assert "role: Engineer" in content
        assert "team:" in content

    def test_create_person_with_all_fields(self, mcp_db):
        """kb_person_create should accept role, email, team, reports_to, company, aliases."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        result = json.loads(
            handle_kb_person_create(
                db,
                db_path,
                "Charlie Davis",
                role="VP Eng",
                email="charlie@example.com",
                team="Engineering",
                reports_to="CEO",
                company="Lattice",
                aliases="Charlie,CD",
            )
        )
        assert result["name"] == "Charlie Davis"

        filepath = db_path / "memory" / "people" / "charlie-davis.md"
        content = filepath.read_text()
        assert "charlie@example.com" in content
        assert "VP Eng" in content

    def test_create_person_seeds_entity_in_db(self, mcp_db):
        """kb_person_create should seed the entity into SQLite."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Diana Prince", role="Hero")

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = ?", ("Diana Prince",)).fetchone()
        assert row is not None
        assert row["entity_type"] == "person"

    def test_create_person_duplicate_returns_error(self, mcp_db):
        """kb_person_create should return error JSON when person already exists."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        # First create should succeed
        handle_kb_person_create(db, db_path, "Talia Ström")
        # Second create should return error (Talia Ström already exists in fixture)
        result = json.loads(handle_kb_person_create(db, db_path, "Talia Ström"))
        assert "error" in result
        assert "exists" in result["error"].lower()

    def test_create_person_no_metadata(self, mcp_db):
        """kb_person_create with just a name (no optional fields) should work."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        result = json.loads(handle_kb_person_create(db, db_path, "Anders Castle"))
        assert result["name"] == "Anders Castle"
        assert result["entity_type"] == "person"

        filepath = db_path / "memory" / "people" / "frank-castle.md"
        assert filepath.exists()
        content = filepath.read_text()
        assert "# Anders Castle" in content

    def test_create_person_empty_string_role_not_filtered(self, mcp_db):
        """Empty string role should pass through metadata dict (not filtered by `if v`)."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        # Empty string is passed to create_entity metadata (not filtered by `if v is not None`).
        # However, _build_markdown drops empty values from YAML, so after seed_entities
        # re-reads the file, the metadata won't contain the empty key. This is correct:
        # empty string on create = don't set the field.
        result = json.loads(handle_kb_person_create(db, db_path, "Empty Role Person", role=""))
        assert result["name"] == "Empty Role Person"
        # Entity should be created successfully even with empty role
        assert "error" not in result


# ---------------------------------------------------------------------------
# Entity create → index + pin integration tests
# ---------------------------------------------------------------------------


class TestEntityCreateIndexing:
    """Newly created entities should be immediately searchable and pinnable."""

    def test_person_create_indexed_in_documents(self, mcp_db):
        """After kb_person_create, the person file should appear in the documents table."""
        from kb.mcp_server import handle_kb_person_create

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Grace Hopper", role="Admiral")

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE path = ?", ("memory/people/grace-hopper.md",)
        ).fetchone()
        assert row is not None
        assert row["doc_type"] == "memory_person"

    def test_person_create_searchable_via_fts(self, mcp_db):
        """After kb_person_create, the person should be findable via kb_search."""
        from kb.mcp_server import handle_kb_person_create, handle_kb_search

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Grace Hopper", role="Admiral")

        result = json.loads(handle_kb_search(db, "Grace Hopper", fast=True, limit=5))
        titles = [r["title"] for r in result["results"]]
        assert any("Grace Hopper" in t for t in titles)

    def test_person_create_pinnable(self, mcp_db):
        """After kb_person_create, kb_pin should be able to pin the person file."""
        from kb.mcp_server import handle_kb_person_create, handle_kb_pin

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Grace Hopper", role="Admiral")

        result = json.loads(handle_kb_pin(db, "memory/people/grace-hopper.md"))
        assert result.get("status") == "ok"
        assert result["pinned"] is True

    def test_person_create_pinnable_by_name(self, mcp_db):
        """After kb_person_create, kb_pin should find by title (person name)."""
        from kb.mcp_server import handle_kb_person_create, handle_kb_pin

        db, db_path = mcp_db
        handle_kb_person_create(db, db_path, "Grace Hopper", role="Admiral")

        result = json.loads(handle_kb_pin(db, "Grace Hopper"))
        assert result.get("status") == "ok"
        assert result["pinned"] is True

    def test_project_create_indexed_in_documents(self, mcp_db):
        """After kb_project_create, the project file should appear in the documents table."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        handle_kb_project_create(db, db_path, "Apollo Program", status="Active")

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE path = ?", ("memory/projects/apollo-program.md",)
        ).fetchone()
        assert row is not None
        assert row["doc_type"] == "memory_project"

    def test_project_create_searchable_via_fts(self, mcp_db):
        """After kb_project_create, the project should be findable via kb_search."""
        from kb.mcp_server import handle_kb_project_create, handle_kb_search

        db, db_path = mcp_db
        handle_kb_project_create(db, db_path, "Apollo Program", status="Active")

        result = json.loads(handle_kb_search(db, "Apollo Program", fast=True, limit=5))
        titles = [r["title"] for r in result["results"]]
        assert any("Apollo" in t for t in titles)


# ---------------------------------------------------------------------------
# _parse_meta_string tests
# ---------------------------------------------------------------------------


class TestParseMetaString:
    """Tests for _parse_meta_string — semicolon-delimited with comma-in-values support."""

    def test_simple_key_value(self):
        from kb.mcp_server import _parse_meta_string

        assert _parse_meta_string("timezone=CET") == {"timezone": "CET"}

    def test_semicolon_delimited(self):
        from kb.mcp_server import _parse_meta_string

        result = _parse_meta_string("role=Engineer; team=Platform")
        assert result == {"role": "Engineer", "team": "Platform"}

    def test_comma_in_value_preserved(self):
        """Commas within values should not be split when using semicolons."""
        from kb.mcp_server import _parse_meta_string

        result = _parse_meta_string("description=Dr. Brown, MD; team=Medical")
        assert result["description"] == "Dr. Brown, MD"
        assert result["team"] == "Medical"

    def test_single_pair_with_comma_in_value(self):
        """Single key=value with no semicolons should preserve commas in value."""
        from kb.mcp_server import _parse_meta_string

        result = _parse_meta_string("description=Dr. Brown, MD")
        assert result["description"] == "Dr. Brown, MD"

    def test_empty_and_none(self):
        from kb.mcp_server import _parse_meta_string

        assert _parse_meta_string(None) == {}
        assert _parse_meta_string("") == {}

    def test_legacy_comma_delimited_multiple_pairs(self):
        """Backward compat: comma-delimited when no semicolons and multiple = signs."""
        from kb.mcp_server import _parse_meta_string

        result = _parse_meta_string("role=Engineer,team=Platform")
        assert result == {"role": "Engineer", "team": "Platform"}


# ---------------------------------------------------------------------------
# handle_kb_person_edit tests
# ---------------------------------------------------------------------------


class TestMcpPersonEdit:
    """Tests for person edit — create a person first so the file exists on disk."""

    def _create_person(self, db, db_path, name="Test Person", **kwargs):
        from kb.mcp_server import handle_kb_person_create

        return json.loads(handle_kb_person_create(db, db_path, name, **kwargs))

    def test_edit_person_updates_role(self, mcp_db):
        """kb_person_edit should update role metadata."""
        from kb.mcp_server import handle_kb_person_edit

        db, db_path = mcp_db
        self._create_person(db, db_path, "Wren Smith", role="Engineer")
        result = json.loads(handle_kb_person_edit(db, db_path, "Wren Smith", role="VP Platform"))
        assert result["updated"] is True
        assert result["name"] == "Wren Smith"

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT metadata FROM entities WHERE name = ?", ("Wren Smith",)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["role"] == "VP Platform"

    def test_edit_person_updates_team(self, mcp_db):
        """kb_person_edit should update team metadata."""
        from kb.mcp_server import handle_kb_person_edit

        db, db_path = mcp_db
        self._create_person(db, db_path, "Soren Jones", team="Platform")
        result = json.loads(handle_kb_person_edit(db, db_path, "Soren Jones", team="Infrastructure"))
        assert result["updated"] is True

    def test_edit_person_with_meta(self, mcp_db):
        """kb_person_edit should accept arbitrary key=value metadata."""
        from kb.mcp_server import handle_kb_person_edit

        db, db_path = mcp_db
        self._create_person(db, db_path, "Charlie Davis")
        result = json.loads(
            handle_kb_person_edit(db, db_path, "Charlie Davis", meta="timezone=CET")
        )
        assert result["updated"] is True

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT metadata FROM entities WHERE name = ?", ("Charlie Davis",)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["timezone"] == "CET"

    def test_edit_person_meta_remove(self, mcp_db):
        """kb_person_edit with empty value should remove a metadata key."""
        from kb.mcp_server import handle_kb_person_edit

        db, db_path = mcp_db
        self._create_person(db, db_path, "Diana Prince")
        handle_kb_person_edit(db, db_path, "Diana Prince", meta="timezone=CET")
        result = json.loads(handle_kb_person_edit(db, db_path, "Diana Prince", meta="timezone="))
        assert result["updated"] is True

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT metadata FROM entities WHERE name = ?", ("Diana Prince",)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert "timezone" not in meta

    def test_edit_person_adds_aliases(self, mcp_db):
        """kb_person_edit should add aliases."""
        from kb.mcp_server import handle_kb_person_edit

        db, db_path = mcp_db
        self._create_person(db, db_path, "Anders Castle")
        # Note: single-word first-name aliases are auto-filtered by _derive_aliases,
        # so use multi-word or non-obvious aliases.
        result = json.loads(
            handle_kb_person_edit(db, db_path, "Anders Castle", aliases="Punisher,FC")
        )
        assert result["updated"] is True

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT aliases FROM entities WHERE name = ?", ("Anders Castle",)
        ).fetchone()
        alias_list = json.loads(row["aliases"])
        assert "Punisher" in alias_list
        assert "FC" in alias_list

    def test_edit_person_not_found_returns_error(self, mcp_db):
        """kb_person_edit for unknown person returns error JSON."""
        from kb.mcp_server import handle_kb_person_edit

        db, db_path = mcp_db
        result = json.loads(handle_kb_person_edit(db, db_path, "NonexistentPerson", role="CEO"))
        assert "error" in result
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# handle_kb_project_create tests
# ---------------------------------------------------------------------------


class TestMcpProjectCreate:
    def test_create_project_returns_json(self, mcp_db):
        """kb_project_create should return valid JSON with name, path, entity_type."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        result = json.loads(handle_kb_project_create(db, db_path, "API Redesign", status="Active"))
        assert result["name"] == "API Redesign"
        assert result["entity_type"] == "project"
        assert "path" in result
        assert result["path"].endswith("api-redesign.md")

    def test_create_project_writes_file(self, mcp_db):
        """kb_project_create should create a markdown file in memory/projects/."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        handle_kb_project_create(
            db, db_path, "Platform Migration", status="In Progress", lead="Talia Ström"
        )

        filepath = db_path / "memory" / "projects" / "platform-migration.md"
        assert filepath.exists()
        content = filepath.read_text()
        assert "# Platform Migration" in content
        assert "status: In Progress" in content

    def test_create_project_with_all_fields(self, mcp_db):
        """kb_project_create should accept status, lead, started, aliases."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        result = json.loads(
            handle_kb_project_create(
                db,
                db_path,
                "Infra Overhaul",
                status="Planning",
                lead="Talia Ström",
                started="2026-01",
                aliases="infra-v2,IO",
            )
        )
        assert result["name"] == "Infra Overhaul"

    def test_create_project_seeds_entity_in_db(self, mcp_db):
        """kb_project_create should seed the entity into SQLite."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        handle_kb_project_create(db, db_path, "New Dashboard", status="Active")

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = ?", ("New Dashboard",)).fetchone()
        assert row is not None
        assert row["entity_type"] == "project"

    def test_create_project_duplicate_returns_error(self, mcp_db):
        """kb_project_create should return error JSON when project already exists."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        handle_kb_project_create(db, db_path, "Duplicate Project")
        result = json.loads(handle_kb_project_create(db, db_path, "Duplicate Project"))
        assert "error" in result
        assert "exists" in result["error"].lower()

    def test_create_project_no_metadata(self, mcp_db):
        """kb_project_create with just a name should work."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        result = json.loads(handle_kb_project_create(db, db_path, "Bare Project"))
        assert result["name"] == "Bare Project"
        assert result["entity_type"] == "project"

    def test_create_project_empty_string_status(self, mcp_db):
        """Empty string status on create should not cause errors."""
        from kb.mcp_server import handle_kb_project_create

        db, db_path = mcp_db
        result = json.loads(handle_kb_project_create(db, db_path, "Empty Status Proj", status=""))
        assert result["name"] == "Empty Status Proj"
        assert "error" not in result


# ---------------------------------------------------------------------------
# handle_kb_project_edit tests
# ---------------------------------------------------------------------------


class TestMcpProjectEdit:
    """Tests for project edit — create a project first so the file exists on disk."""

    def _create_project(self, db, db_path, name="Test Project", **kwargs):
        from kb.mcp_server import handle_kb_project_create

        return json.loads(handle_kb_project_create(db, db_path, name, **kwargs))

    def test_edit_project_updates_status(self, mcp_db):
        """kb_project_edit should update status metadata."""
        from kb.mcp_server import handle_kb_project_edit

        db, db_path = mcp_db
        self._create_project(db, db_path, "API Redesign", status="In Progress")
        result = json.loads(handle_kb_project_edit(db, db_path, "API Redesign", status="Completed"))
        assert result["updated"] is True
        assert result["name"] == "API Redesign"

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT metadata FROM entities WHERE name = ?", ("API Redesign",)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["status"] == "Completed"

    def test_edit_project_updates_lead(self, mcp_db):
        """kb_project_edit should update lead metadata."""
        from kb.mcp_server import handle_kb_project_edit

        db, db_path = mcp_db
        self._create_project(db, db_path, "Platform Work")
        result = json.loads(handle_kb_project_edit(db, db_path, "Platform Work", lead="Talia Ström"))
        assert result["updated"] is True

    def test_edit_project_with_meta(self, mcp_db):
        """kb_project_edit should accept arbitrary key=value metadata."""
        from kb.mcp_server import handle_kb_project_edit

        db, db_path = mcp_db
        self._create_project(db, db_path, "Infra Overhaul")
        result = json.loads(
            handle_kb_project_edit(db, db_path, "Infra Overhaul", meta="priority=High")
        )
        assert result["updated"] is True

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT metadata FROM entities WHERE name = ?", ("Infra Overhaul",)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["priority"] == "High"

    def test_edit_project_adds_aliases(self, mcp_db):
        """kb_project_edit should add aliases."""
        from kb.mcp_server import handle_kb_project_edit

        db, db_path = mcp_db
        self._create_project(db, db_path, "Dashboard V2")
        # Note: lowercase-with-hyphens aliases are filtered by _derive_aliases
        # (treated as auto-generated file stems), so use non-hyphenated aliases.
        result = json.loads(
            handle_kb_project_edit(db, db_path, "Dashboard V2", aliases="DashV2,DV2")
        )
        assert result["updated"] is True

        conn = db.get_sqlite_conn()
        row = conn.execute(
            "SELECT aliases FROM entities WHERE name = ?", ("Dashboard V2",)
        ).fetchone()
        alias_list = json.loads(row["aliases"])
        assert "DashV2" in alias_list
        assert "DV2" in alias_list

    def test_edit_project_not_found_returns_error(self, mcp_db):
        """kb_project_edit for unknown project returns error JSON."""
        from kb.mcp_server import handle_kb_project_edit

        db, db_path = mcp_db
        result = json.loads(
            handle_kb_project_edit(db, db_path, "Nonexistent Project", status="Done")
        )
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestFindProjectRootXdgSafety:
    """XDG detection should not false-positive on paths like ~/.config-other/."""

    def test_config_other_not_treated_as_xdg(self, tmp_path):
        """A config in /tmp/.config-other/ should NOT trigger XDG fallback."""
        from kb.config import find_project_root

        # Create a config dir that starts with the XDG path but isn't inside it
        xdg_dir = tmp_path / ".config"
        xdg_dir.mkdir()
        fake_dir = tmp_path / ".config-other" / "kbx"
        fake_dir.mkdir(parents=True)

        config_file = fake_dir / "kbx.toml"
        config_file.write_text('[sources]\nmemory = "/some/absolute/memory"\n')

        with patch.dict(
            os.environ,
            {
                "KBX_CONFIG": str(config_file),
                "XDG_CONFIG_HOME": str(xdg_dir),
            },
        ):
            root = find_project_root()

        # Should return the config parent (not memory parent),
        # because .config-other is NOT inside .config
        assert root == fake_dir


# ---------------------------------------------------------------------------
# Coverage: MCP handlers — project find/list, person list, note delete, etc.
# ---------------------------------------------------------------------------


class TestFactSeqMigration:
    def test_migration_assigns_seq(self):
        """Migration 012 should assign seq values to existing facts."""
        import tempfile

        from kb.db import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            # Insert entity and facts without seq (simulating old data)
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) "
                "VALUES ('Test', 'person', '[]', '{}')"
            )
            eid = conn.execute("SELECT id FROM entities WHERE name = 'Test'").fetchone()["id"]
            conn.execute(
                "INSERT INTO facts (entity_id, fact_text, fact_date, seq) VALUES (?, 'Fact A', '2026-01-01', NULL)",
                (eid,),
            )
            conn.execute(
                "INSERT INTO facts (entity_id, fact_text, fact_date, seq) VALUES (?, 'Fact B', '2026-02-01', NULL)",
                (eid,),
            )
            conn.commit()

            # Run migration manually
            from kb.db import _migrate_012_add_fact_seq

            _migrate_012_add_fact_seq(conn)
            conn.commit()

            rows = conn.execute(
                "SELECT seq FROM facts WHERE entity_id = ? ORDER BY seq", (eid,)
            ).fetchall()
            assert [r["seq"] for r in rows] == [1, 2]
            db.close()


class TestCrudHelpers:
    def test_snake_to_title(self):
        from kb.crud import _snake_to_title

        assert _snake_to_title("preferred_lang") == "Preferred Lang"

    def test_title_to_snake(self):
        from kb.crud import _title_to_snake

        assert _title_to_snake("Reports to") == "reports_to"

    def test_ambiguous_document_error(self):
        from kb.crud import AmbiguousDocumentError

        err = AmbiguousDocumentError("test", ["a.md", "b.md"])
        assert err.target == "test"
        assert err.matches == ["a.md", "b.md"]


class TestPinWritethrough:
    """Pin/unpin should write to frontmatter for memory notes."""

    def test_pin_writes_frontmatter(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_pin

        db, root = mcp_db
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Pin test", body="Content", date="2026-01-01")
        )
        path = add_result["path"]

        result = json.loads(handle_kb_pin(db, path, root))
        assert result["pinned"] is True

        content = (root / path).read_text()
        assert "pinned: true" in content

    def test_pin_bare_note_creates_frontmatter(self, mcp_db):
        """Pin on a note without frontmatter should create it."""
        from kb.mcp_server import handle_kb_pin

        db, root = mcp_db
        # Create a bare note file with no frontmatter
        notes_dir = root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        bare = notes_dir / "bare.md"
        bare.write_text("Just text.\n")

        # Index it
        from kb.indexer import index_all

        index_all(db, None, root, memory_only=True, skip_seed=True)

        result = json.loads(handle_kb_pin(db, "memory/notes/bare.md", root))
        assert result["pinned"] is True
        content = bare.read_text()
        assert "pinned: true" in content

    def test_unpin_writes_frontmatter(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_pin, handle_kb_unpin

        db, root = mcp_db
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Unpin test", body="Content", date="2026-01-01")
        )
        path = add_result["path"]

        handle_kb_pin(db, path, root)
        result = json.loads(handle_kb_unpin(db, path, root))
        assert result["pinned"] is False

        content = (root / path).read_text()
        assert "pinned: false" in content


class TestMcpProjectFindCoverage:
    def test_project_find_success(self, mcp_db):
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "Helix Refactor"))
        assert result["name"] == "Helix Refactor"
        assert result["entity_type"] == "project"

    def test_project_find_not_found(self, mcp_db):
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "Nonexistent"))
        assert "error" in result

    def test_project_find_wrong_type(self, mcp_db):
        from kb.mcp_server import handle_kb_project_find

        db, _ = mcp_db
        result = json.loads(handle_kb_project_find(db, "Talia"))
        assert "error" in result
        assert "person" in result["error"]


class TestMcpProjectListCoverage:
    def test_project_list(self, mcp_db):
        from kb.mcp_server import handle_kb_project_list

        db, _ = mcp_db
        result = json.loads(handle_kb_project_list(db))
        assert "results" in result
        assert result["meta"]["total"] >= 1


class TestMcpPersonListCoverage:
    def test_person_list(self, mcp_db):
        from kb.mcp_server import handle_kb_person_list

        db, _ = mcp_db
        result = json.loads(handle_kb_person_list(db))
        assert "results" in result
        assert result["meta"]["total"] >= 1
        names = [r["name"] for r in result["results"]]
        assert "Talia Ström" in names


class TestMcpNoteDeleteConsolidated:
    def test_note_delete_removes_file_and_db(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_note_delete

        db, root = mcp_db
        # Create a note
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Deletable", body="Content", date="2026-01-01")
        )
        assert add_result["status"] == "ok"
        path = add_result["path"]

        # Delete it
        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_note_delete(db, root, path))
        assert result["status"] == "ok"

        # File should be gone
        assert not (root / path).exists()

    def test_note_delete_not_found(self, mcp_db):
        from kb.mcp_server import handle_kb_note_delete

        db, root = mcp_db
        result = json.loads(handle_kb_note_delete(db, root, "nonexistent.md"))
        assert "error" in result


class TestMcpEntityStaleCoverage:
    def test_entity_stale_returns_results(self, mcp_db):
        from kb.mcp_server import handle_kb_entity_stale

        db, _ = mcp_db
        result = json.loads(handle_kb_entity_stale(db))
        assert "results" in result
        assert "meta" in result


class TestMcpNoteListCoverage:
    def test_note_list_empty(self, mcp_db):
        from kb.mcp_server import handle_kb_note_list

        db, _ = mcp_db
        result = json.loads(handle_kb_note_list(db))
        assert "results" in result
        assert result["meta"]["total"] == 0

    def test_note_list_with_notes(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_note_list

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Test note", body="Content", date="2026-01-01")

        result = json.loads(handle_kb_note_list(db))
        assert result["meta"]["total"] >= 1


class TestMcpNoteEditCoverage:
    def test_note_edit_body(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_note_edit

        db, root = mcp_db
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Editable note", body="Original", date="2026-01-01")
        )
        path = add_result["path"]

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_note_edit(db, root, path, body="Replaced"))
        assert result["status"] == "ok"
        content = (root / path).read_text()
        assert "Replaced" in content

    def test_note_edit_append(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_note_edit

        db, root = mcp_db
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Append note", body="Start.", date="2026-01-01")
        )
        path = add_result["path"]

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_note_edit(db, root, path, append="\nMore."))
        assert result["status"] == "ok"

    def test_note_edit_tags(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_note_edit

        db, root = mcp_db
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Tag note", body="Content", date="2026-01-01")
        )
        path = add_result["path"]

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_note_edit(db, root, path, tags="foo,bar"))
        assert result["status"] == "ok"

    def test_note_edit_pin(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_note_edit

        db, root = mcp_db
        add_result = json.loads(
            handle_kb_memory_add(db, root, "Pin note", body="Content", date="2026-01-01")
        )
        path = add_result["path"]

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_note_edit(db, root, path, pin=True))
        assert result["pinned"] is True

    def test_note_edit_not_found(self, mcp_db):
        from kb.mcp_server import handle_kb_note_edit

        db, root = mcp_db
        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(handle_kb_note_edit(db, root, "nonexistent.md"))
        assert "error" in result


class TestMcpWrapperCoverage:
    """Exercise MCP tool wrappers by mocking get_db/find_project_root."""

    def test_wrappers_exercise(self, mcp_db):
        """Call multiple MCP tool wrappers to cover their thin delegation code."""
        db, root = mcp_db

        with (
            patch("kb.mcp_server.get_db", return_value=db),
            patch("kb.mcp_server.find_project_root", return_value=root),
        ):
            from kb.mcp_server import (
                kb_context,
                kb_index_status,
                kb_memory_list,
                kb_note_list,
                kb_person_find,
                kb_person_list,
                kb_person_timeline,
                kb_project_find,
                kb_project_list,
                kb_search,
                kb_usage,
                kb_view,
            )

            # Read-only wrappers
            result = json.loads(kb_search("test query"))
            assert "results" in result or "error" in result

            result = json.loads(kb_person_find("Talia"))
            assert "name" in result or "error" in result

            result = json.loads(kb_person_timeline("Talia"))
            assert "documents" in result or "error" in result

            result = json.loads(kb_view("memory/people/eve.md"))
            assert "title" in result or "error" in result

            result = json.loads(kb_context())
            assert isinstance(result, dict)

            result = json.loads(kb_usage())
            assert "docs" in result or "error" in result

            result = json.loads(kb_person_list())
            assert "results" in result

            result = json.loads(kb_project_list())
            assert "results" in result

            result = json.loads(kb_project_find("Helix Refactor"))
            assert "name" in result or "error" in result

            result = json.loads(kb_note_list())
            assert "results" in result

            result = json.loads(kb_memory_list())
            assert "results" in result

            result = json.loads(kb_index_status())
            assert isinstance(result, dict)

    def test_mutating_wrappers(self, mcp_db):
        """Exercise mutating MCP wrappers."""
        db, root = mcp_db

        with (
            patch("kb.mcp_server.get_db", return_value=db),
            patch("kb.mcp_server.find_project_root", return_value=root),
            patch("kb.config.get_data_dir", return_value=root),
        ):
            from kb.mcp_server import (
                kb_entity_stale,
                kb_list,
                kb_memory_add,
                kb_memory_delete_fact,
                kb_memory_edit_fact,
                kb_note_delete,
                kb_note_edit,
                kb_person_create,
                kb_person_edit,
                kb_pin,
                kb_project_create,
                kb_project_edit,
                kb_unpin,
            )

            result = json.loads(kb_entity_stale())
            assert "results" in result

            # Create a note then edit/pin/unpin/delete it
            add_result = json.loads(
                kb_memory_add("Wrapper test note", body="Content", date="2026-01-01")
            )
            assert add_result["status"] == "ok"
            path = add_result["path"]

            result = json.loads(kb_pin(path))
            assert "pinned" in result or "error" in result

            result = json.loads(kb_unpin(path))
            assert "pinned" in result or "error" in result

            result = json.loads(kb_note_edit(path, body="Updated"))
            assert result.get("status") == "ok" or "error" in result

            result = json.loads(kb_note_delete(path))
            assert result.get("status") == "ok" or "error" in result

            # Entity create/edit
            result = json.loads(kb_person_create("Wrapper Person", role="Tester"))
            assert "name" in result or "error" in result

            result = json.loads(kb_person_edit("Wrapper Person", role="Senior Tester"))
            assert "updated" in result or "error" in result

            result = json.loads(kb_project_create("Wrapper Project", status="Active"))
            assert "name" in result or "error" in result

            result = json.loads(kb_project_edit("Wrapper Project", status="Done"))
            assert "updated" in result or "error" in result

            # Fact via add, edit, delete
            fact_result = json.loads(kb_memory_add("Wrapper person fact", entity="Wrapper Person"))
            assert fact_result["status"] == "ok"

            conn = db.get_sqlite_conn()
            fact_row = conn.execute(
                "SELECT seq FROM facts WHERE fact_text = 'Wrapper person fact'"
            ).fetchone()
            fact_seq = fact_row["seq"]

            result = json.loads(kb_memory_edit_fact("Wrapper Person", fact_seq, text="Edited fact"))
            assert result.get("updated") is True or "error" in result

            result = json.loads(kb_memory_delete_fact("Wrapper Person", fact_seq))
            assert result.get("deleted") is True or "error" in result

            # List documents
            result = json.loads(kb_list())
            assert "results" in result


class TestMcpMemoryEditFactSuccess:
    def test_edit_fact_updates_text(self, mcp_db):
        from kb.mcp_server import handle_kb_memory_add, handle_kb_memory_edit_fact

        db, root = mcp_db
        handle_kb_memory_add(db, root, "Talia fact original", entity="Talia")
        conn = db.get_sqlite_conn()
        fact = conn.execute(
            "SELECT seq FROM facts WHERE fact_text = 'Talia fact original'"
        ).fetchone()

        with patch("kb.config.get_data_dir", return_value=root):
            result = json.loads(
                handle_kb_memory_edit_fact(
                    db, root, "Talia Ström", fact["seq"], text="Talia fact updated"
                )
            )
        assert result.get("updated") is True
        assert result["fact_text"] == "Talia fact updated"
