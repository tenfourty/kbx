"""Tests for kb MCP server — Phase 8: tools, config, resources."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

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
                '{"team": "Platform"}',
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

    def test_person_find_includes_documents(self, mcp_db):
        """kb_person_find should include linked documents."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        result = handle_kb_person_find(db, "Talia")
        data = json.loads(result)
        assert "documents" in data

    def test_person_find_not_found(self, mcp_db):
        """kb_person_find should return error for unknown person."""
        from kb.mcp_server import handle_kb_person_find

        db, _ = mcp_db
        result = handle_kb_person_find(db, "NonexistentPerson")
        data = json.loads(result)
        assert "error" in data


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
        # Compact format uses [People: ...] style
        assert "[People:" in data["text"]

    def test_context_human_fmt(self, mcp_db):
        """handle_kb_context with fmt='human' should return markdown format."""
        from kb.mcp_server import handle_kb_context

        db, _ = mcp_db
        result = handle_kb_context(db, project_root=Path("/tmp"), fmt="human")
        data = json.loads(result)
        assert "text" in data
        # Human format uses ## headings
        assert "## People" in data["text"]

    def test_context_mcp_tool_accepts_fmt(self, mcp_db):
        """kb_context MCP tool should accept fmt parameter."""
        import inspect

        from kb.mcp_server import kb_context

        sig = inspect.signature(kb_context)
        assert "fmt" in sig.parameters


class TestMcpUsage:
    def test_usage_returns_string(self, mcp_db):
        """kb_usage should return usage text with stats."""
        from kb.mcp_server import handle_kb_usage

        db, _ = mcp_db
        result = handle_kb_usage(db)
        assert "kb_search" in result
        assert "documents" in result.lower() or "entities" in result.lower()


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
# handle_kb_usage — facts count coverage
# ---------------------------------------------------------------------------


class TestMcpUsageFacts:
    def test_usage_includes_facts_count(self, mcp_db):
        """kb_usage output includes facts count."""
        from kb.mcp_server import handle_kb_memory_add, handle_kb_usage

        db, db_path = mcp_db
        # Add a fact first
        handle_kb_memory_add(db, db_path, "Talia knows Rust", entity="Talia")

        result = handle_kb_usage(db)
        # Should include "1 facts" in the output
        assert "1 facts" in result
