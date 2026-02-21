"""Integration tests for kb — Phase 6: ingest, migrations, eval, end-to-end."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from conftest import invoke_cli

from kb.db import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_db():
    """Create a test database with known data for integration tests."""
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
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (2, 2, 'discussed')"
        )
        conn.commit()
        yield db, Path(tmpdir)
        db.close()


# ---------------------------------------------------------------------------
# Step 1: ingest command
# ---------------------------------------------------------------------------


def _mock_embedder_module():
    """Create a mock kb.embeddings module where Embedder() raises."""
    mock_mod = MagicMock()
    mock_mod.Embedder.side_effect = RuntimeError("No model in test")
    return mock_mod


class TestIngestCommand:
    def test_ingest_skip_organise(self, runner, integration_db):
        """kb ingest --skip-organise should run without error (index step only)."""
        _, db_path = integration_db
        # Temporarily replace kb.embeddings so Embedder() fails fast
        orig = sys.modules.get("kb.embeddings")
        sys.modules["kb.embeddings"] = _mock_embedder_module()
        try:
            result = invoke_cli(runner, ["ingest", "--skip-organise"], db_path)
        finally:
            if orig is not None:
                sys.modules["kb.embeddings"] = orig
            else:
                sys.modules.pop("kb.embeddings", None)
        assert result.exit_code == 0

    def test_ingest_dry_run(self, runner, integration_db):
        """kb ingest --dry-run should not modify data."""
        _, db_path = integration_db
        result = invoke_cli(runner, ["ingest", "--dry-run", "--skip-organise"], db_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Step 2: .gitignore
# ---------------------------------------------------------------------------


class TestGitignore:
    def test_gitignore_exists(self):
        """Project root .gitignore should exist and contain data/."""
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        assert gitignore.exists(), ".gitignore should exist at project root"
        content = gitignore.read_text()
        assert "data/" in content


# ---------------------------------------------------------------------------
# Step 3: Schema migrations
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_migrations_table_exists(self):
        """Fresh DB should have a migrations table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert "migrations" in tables
            db.close()

    def test_migrations_applied(self):
        """Fresh DB should have migrations recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            rows = conn.execute("SELECT name FROM migrations").fetchall()
            names = [r["name"] for r in rows]
            assert len(names) >= 1
            assert "001_add_file_mtime" in names
            db.close()

    def test_migrations_idempotent(self):
        """Creating DB twice should not fail or duplicate migrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = Database(Path(tmpdir))
            count1 = db1.get_sqlite_conn().execute("SELECT COUNT(*) FROM migrations").fetchone()[0]
            db1.close()

            db2 = Database(Path(tmpdir))
            count2 = db2.get_sqlite_conn().execute("SELECT COUNT(*) FROM migrations").fetchone()[0]
            db2.close()

            assert count1 == count2


# ---------------------------------------------------------------------------
# Step 4: Search quality eval (pytest-compatible)
# ---------------------------------------------------------------------------


class TestSearchQualityEval:
    """Run a few search quality queries against the fixture DB."""

    EVAL_QUERIES: ClassVar[list[dict[str, str]]] = [
        {"query": "MFA implementation", "expect_title": "MFA"},
        {"query": "Cloud migration", "expect_title": "Cloud"},
        {"query": "Datastore", "expect_title": "Datastore"},
    ]

    def test_eval_queries_hit_at_5(self, integration_db):
        """All eval queries should have their expected result in top 5."""
        from kb.search import search

        db, _ = integration_db
        for q in self.EVAL_QUERIES:
            results = search(db, None, q["query"], fast=True, limit=5)
            titles = [r.title for r in results.results]
            assert any(q["expect_title"] in t for t in titles), (
                f"Query '{q['query']}' expected title containing '{q['expect_title']}' "
                f"in top 5, got: {titles}"
            )

    def test_eval_queries_return_results(self, integration_db):
        """All eval queries should return at least 1 result."""
        from kb.search import search

        db, _ = integration_db
        for q in self.EVAL_QUERIES:
            results = search(db, None, q["query"], fast=True)
            assert results.meta.total > 0, f"Query '{q['query']}' returned no results"


# ---------------------------------------------------------------------------
# Step 5: End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_search_then_view(self, runner, integration_db):
        """Search for a document, then view it by path."""
        _, db_path = integration_db
        # Search
        search_result = invoke_cli(runner, ["search", "MFA", "--fast", "--json"], db_path)
        assert search_result.exit_code == 0
        data = json.loads(search_result.output)
        assert data["results"]

        # View the first result
        path = data["results"][0]["path"]
        view_result = invoke_cli(runner, ["view", path, "--json"], db_path)
        assert view_result.exit_code == 0
        view_data = json.loads(view_result.output)
        assert view_data["title"] == "MFA Implementation Review"

    def test_index_status_structure(self, runner, integration_db):
        """kb index status --json returns valid structure."""
        _, db_path = integration_db
        result = invoke_cli(runner, ["index", "status", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "documents" in data
        assert "chunks" in data
        assert "entities" in data
        assert isinstance(data["documents"], int)
        assert isinstance(data["chunks"], int)

    def test_list_and_entity_list_json(self, runner, integration_db):
        """Both kb list --json and kb entity list --json return valid JSON."""
        _, db_path = integration_db

        list_result = invoke_cli(runner, ["list", "--json"], db_path)
        assert list_result.exit_code == 0
        list_data = json.loads(list_result.output)
        assert isinstance(list_data, dict)
        assert "results" in list_data
        assert len(list_data["results"]) > 0

        person_result = invoke_cli(runner, ["person", "list", "--json"], db_path)
        assert person_result.exit_code == 0
        person_data = json.loads(person_result.output)
        assert isinstance(person_data, dict)
        assert "results" in person_data
        assert len(person_data["results"]) > 0

    def test_search_returns_correct_fields(self, runner, integration_db):
        """Search results have all required fields."""
        _, db_path = integration_db
        result = invoke_cli(runner, ["search", "Rust", "--fast", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        required = {
            "chunk_id",
            "document_id",
            "title",
            "path",
            "date",
            "doc_type",
            "score",
            "section",
            "snippet",
            "entities",
        }
        for r in data["results"]:
            assert required.issubset(r.keys())
