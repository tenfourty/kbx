"""Tests for Database context manager and edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kb.db import Database


class TestDatabaseContextManager:
    def test_context_manager_opens_and_closes(self):
        with tempfile.TemporaryDirectory() as tmpdir, Database(Path(tmpdir)) as db:
            conn = db.get_sqlite_conn()
            conn.execute("SELECT 1")
            # After exiting, connection should be closed

    def test_context_manager_on_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with Database(Path(tmpdir)) as db:
                    _conn = db.get_sqlite_conn()
                    raise ValueError("test error")
            except ValueError:
                pass
            # DB should be cleaned up despite exception


class TestSchemaIntegrity:
    def test_schema_has_expected_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "documents" in tables
            assert "chunks" in tables
            assert "entities" in tables
            assert "entity_mentions" in tables
            assert "migrations" in tables
            assert "facts" in tables
            db.close()

    def test_fts_table_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "chunks_fts" in tables
            db.close()


class TestMigration008EntityFreshness:
    def test_migration_008_entity_freshness_columns(self):
        """Migration 008 adds updated_at and last_mentioned_at to entities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            cols = {row[1] for row in conn.execute("PRAGMA table_info(entities)").fetchall()}
            assert "updated_at" in cols, "updated_at column missing"
            assert "last_mentioned_at" in cols, "last_mentioned_at column missing"
            db.close()


class TestMigration009BackfillFreshness:
    def test_migration_009_backfills_last_mentioned_at(self):
        """Migration 009 backfills last_mentioned_at from entity_mentions + documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()

            # Remove 009 migration record so we can re-trigger it after inserting data
            conn.execute("DELETE FROM migrations WHERE name = '009_backfill_last_mentioned_at'")

            # Insert a person entity (freshness columns exist but are NULL)
            conn.execute(
                "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
                ("Wren", "person"),
            )
            entity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Insert a document with a known date
            conn.execute(
                "INSERT INTO documents (path, content_hash, doc_date) VALUES (?, ?, ?)",
                ("test.md", "abc123", "2026-02-15"),
            )
            doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Link entity to document
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                "VALUES (?, ?, ?)",
                (entity_id, doc_id, "discussed"),
            )
            conn.commit()

            # Re-run migrations to trigger backfill
            from kb.db import _apply_migrations

            _apply_migrations(conn)

            row = conn.execute(
                "SELECT last_mentioned_at FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            assert row["last_mentioned_at"] == "2026-02-15"
            db.close()

    def test_migration_009_picks_max_date(self):
        """Backfill picks the most recent doc_date across all mentions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()

            # Remove 009 migration record so we can re-trigger it after inserting data
            conn.execute("DELETE FROM migrations WHERE name = '009_backfill_last_mentioned_at'")

            conn.execute(
                "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
                ("Soren", "person"),
            )
            entity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Two documents with different dates
            for path, hash_, date in [
                ("old.md", "h1", "2025-12-01"),
                ("new.md", "h2", "2026-02-20"),
            ]:
                conn.execute(
                    "INSERT INTO documents (path, content_hash, doc_date) VALUES (?, ?, ?)",
                    (path, hash_, date),
                )
                did = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                    "VALUES (?, ?, ?)",
                    (entity_id, did, "discussed"),
                )
            conn.commit()

            from kb.db import _apply_migrations

            _apply_migrations(conn)

            row = conn.execute(
                "SELECT last_mentioned_at FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            assert row["last_mentioned_at"] == "2026-02-20"
            db.close()


class TestMigration009DDLCommitBeforeBackfill:
    def test_backfill_works_when_008_and_009_both_pending(self):
        """When 008 (ALTER TABLE) and 009 (backfill) are both pending,
        the backfill must see the new columns — requires DDL commit first.

        Creates a raw DB without the freshness columns, inserts entity data
        and mentions, then runs _apply_migrations to apply 008 + 009 together.
        """
        import sqlite3 as _sqlite3

        from kb.db import SCHEMA_SQL, _apply_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "metadata.db"
            conn = _sqlite3.connect(str(db_path))
            conn.row_factory = _sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")

            # Create base schema (entities table has NO updated_at/last_mentioned_at)
            conn.executescript(SCHEMA_SQL)

            # Create facts table (migration 002 would have created it, but SCHEMA_SQL
            # doesn't include it — simulate a DB that had 002 applied)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id INTEGER REFERENCES entities(id),
                    fact_text TEXT NOT NULL,
                    fact_date TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )
            conn.commit()

            # Register only migrations 001-007 as applied (008/009 are pending)
            pre_008 = [
                "001_add_file_mtime",
                "002_create_facts_table",
                "003_add_entity_pinned",
                "004_create_document_attendees",
                "005_normalize_paths_nfc",
                "006_unique_entity_name_type",
                "007_add_document_pinned",
            ]
            for name in pre_008:
                conn.execute("INSERT INTO migrations (name) VALUES (?)", (name,))
            conn.commit()

            # Verify freshness columns do NOT exist yet
            cols = {row[1] for row in conn.execute("PRAGMA table_info(entities)").fetchall()}
            assert "updated_at" not in cols
            assert "last_mentioned_at" not in cols

            # Insert entity + document + mention
            conn.execute(
                "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
                ("TestPerson", "person"),
            )
            entity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO documents (path, content_hash, doc_date) VALUES (?, ?, ?)",
                ("meeting.md", "hash1", "2026-02-10"),
            )
            doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) "
                "VALUES (?, ?, ?)",
                (entity_id, doc_id, "discussed"),
            )
            conn.commit()

            # Run _apply_migrations — should apply 008 DDL + 009 backfill
            _apply_migrations(conn)

            # last_mentioned_at should be populated by the backfill
            row = conn.execute(
                "SELECT last_mentioned_at FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            assert row["last_mentioned_at"] == "2026-02-10", (
                f"Backfill should populate last_mentioned_at but got {row['last_mentioned_at']}"
            )
            conn.close()


class TestNormalizePath:
    def test_nfc_normalization(self):
        import unicodedata

        from kb.db import normalize_path

        # NFD string (decomposed)
        nfd = unicodedata.normalize("NFD", "Linnea")
        result = normalize_path(nfd)
        assert result == unicodedata.normalize("NFC", "Linnea")

    def test_ascii_unchanged(self):
        from kb.db import normalize_path

        assert normalize_path("simple/path.md") == "simple/path.md"

    def test_nfc_with_accented_characters(self):
        import unicodedata

        from kb.db import normalize_path

        # NFD decomposed accented character
        nfd = unicodedata.normalize("NFD", "cafe\u0301")  # cafe + combining accent
        result = normalize_path(nfd)
        assert result == unicodedata.normalize("NFC", "cafe\u0301")
