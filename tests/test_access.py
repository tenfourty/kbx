"""Tests for kb.access — touch + reset helpers (issue #67)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kb.access import (
    reset_document_access,
    reset_entity_access,
    touch_document,
    touch_entity,
)
from kb.db import Database


@pytest.fixture
def db():
    """Fresh in-memory-ish DB with a couple of docs and entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_obj = Database(Path(tmpdir))
        conn = db_obj.get_sqlite_conn()

        conn.execute(
            """INSERT INTO documents (path, title, content_hash) VALUES (?, ?, ?)""",
            ("doc1.md", "Doc One", "h1"),
        )
        conn.execute(
            """INSERT INTO documents (path, title, content_hash) VALUES (?, ?, ?)""",
            ("doc2.md", "Doc Two", "h2"),
        )
        conn.execute(
            """INSERT INTO entities (name, entity_type) VALUES (?, ?)""",
            ("Wren", "person"),
        )
        conn.execute(
            """INSERT INTO entities (name, entity_type) VALUES (?, ?)""",
            ("Soren", "person"),
        )
        conn.commit()
        yield db_obj


class TestTouchDocument:
    def test_first_touch_sets_count_to_one(self, db):
        conn = db.get_sqlite_conn()
        touch_document(conn, 1)
        row = conn.execute(
            "SELECT access_count, last_accessed_at FROM documents WHERE id = 1"
        ).fetchone()
        assert row["access_count"] == 1
        assert row["last_accessed_at"] is not None

    def test_multiple_touches_increment(self, db):
        conn = db.get_sqlite_conn()
        for _ in range(5):
            touch_document(conn, 1)
        row = conn.execute("SELECT access_count FROM documents WHERE id = 1").fetchone()
        assert row["access_count"] == 5

    def test_touches_are_independent(self, db):
        """Touching doc1 does not affect doc2."""
        conn = db.get_sqlite_conn()
        touch_document(conn, 1)
        touch_document(conn, 1)
        row = conn.execute(
            "SELECT access_count FROM documents WHERE id = 2"
        ).fetchone()
        assert row["access_count"] == 0

    def test_missing_id_is_silent(self, db):
        """Touching a non-existent doc is a no-op (no exception)."""
        conn = db.get_sqlite_conn()
        touch_document(conn, 9999)  # ID doesn't exist


class TestTouchEntity:
    def test_first_touch_sets_count_to_one(self, db):
        conn = db.get_sqlite_conn()
        touch_entity(conn, 1)
        row = conn.execute(
            "SELECT access_count, last_accessed_at FROM entities WHERE id = 1"
        ).fetchone()
        assert row["access_count"] == 1
        assert row["last_accessed_at"] is not None

    def test_increments_per_touch(self, db):
        conn = db.get_sqlite_conn()
        for _ in range(3):
            touch_entity(conn, 2)
        row = conn.execute("SELECT access_count FROM entities WHERE id = 2").fetchone()
        assert row["access_count"] == 3


class TestResetDocumentAccess:
    def test_reset_specific_doc(self, db):
        conn = db.get_sqlite_conn()
        touch_document(conn, 1)
        touch_document(conn, 1)
        touch_document(conn, 2)
        n = reset_document_access(conn, document_id=1)
        assert n == 1
        # doc1 cleared
        row = conn.execute(
            "SELECT access_count, last_accessed_at FROM documents WHERE id = 1"
        ).fetchone()
        assert row["access_count"] == 0
        assert row["last_accessed_at"] is None
        # doc2 untouched
        row2 = conn.execute("SELECT access_count FROM documents WHERE id = 2").fetchone()
        assert row2["access_count"] == 1

    def test_reset_all_docs(self, db):
        conn = db.get_sqlite_conn()
        touch_document(conn, 1)
        touch_document(conn, 2)
        n = reset_document_access(conn)
        assert n == 2
        rows = conn.execute("SELECT access_count FROM documents").fetchall()
        assert all(r["access_count"] == 0 for r in rows)


class TestResetEntityAccess:
    def test_reset_specific_entity(self, db):
        conn = db.get_sqlite_conn()
        touch_entity(conn, 1)
        touch_entity(conn, 2)
        n = reset_entity_access(conn, entity_id=1)
        assert n == 1
        rows = conn.execute(
            "SELECT id, access_count FROM entities ORDER BY id"
        ).fetchall()
        assert rows[0]["access_count"] == 0
        assert rows[1]["access_count"] == 1

    def test_reset_all_entities(self, db):
        conn = db.get_sqlite_conn()
        touch_entity(conn, 1)
        touch_entity(conn, 2)
        n = reset_entity_access(conn)
        assert n == 2
        rows = conn.execute("SELECT access_count FROM entities").fetchall()
        assert all(r["access_count"] == 0 for r in rows)
