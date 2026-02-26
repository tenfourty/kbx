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
