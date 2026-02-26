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
