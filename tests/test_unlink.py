"""Integration tests for entity↔document unlink/relink suppression (#35).

The suppression must survive a full reindex (and, by extension, Granola sync's
file regeneration) — that persistence is the whole point of the sidecar store.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from kb.cli import cli


def _count_mentions(conn: sqlite3.Connection, entity_name: str, doc_like: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM entity_mentions em
        JOIN entities e ON e.id = em.entity_id
        JOIN documents d ON d.id = em.document_id
        WHERE e.name = ? AND d.path LIKE ?
        """,
        (entity_name, doc_like),
    ).fetchone()
    return int(row["n"])


def _seed_project(root: Path) -> str:
    """Create a person entity + a meeting note that mentions them. Returns note rel-path."""
    people = root / "memory" / "people"
    people.mkdir(parents=True)
    (people / "alex-tanner.md").write_text("# Alex Tanner\n\n**Role:** Engineer\n")
    meetings = root / "memory" / "meetings" / "2026" / "05" / "24"
    meetings.mkdir(parents=True)
    (meetings / "ab12cd34_Sync.granola.notes.md").write_text(
        "---\ntitle: Sync\ndate: 2026-05-24\ntype: notes\ngranola_id: ab12cd34\n---\n\n"
        "## Notes\n\nAlex Tanner presented the roadmap.\n"
    )
    return "ab12cd34_Sync.granola.notes.md"


class TestUnlinkRelink:
    def test_unlink_drops_mention_and_persists_through_reindex(self, tmp_db):
        from kb.api import KnowledgeBase
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            note = _seed_project(root)

            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alex Tanner", "%Sync%") >= 1, "expected a baseline link"

            kb = KnowledgeBase._from_existing(db=db, project_root=root)
            result = kb.unlink_entity("Alex Tanner", note)
            assert result["unlinked"] is True
            assert result["entity"] == "Alex Tanner"
            assert _count_mentions(conn, "Alex Tanner", "%Sync%") == 0, "live mention not dropped"

            # The whole point of #35: the suppression survives a full reindex.
            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alex Tanner", "%Sync%") == 0, (
                "suppression lost on reindex"
            )

    def test_relink_restores_mention_on_reindex(self, tmp_db):
        from kb.api import KnowledgeBase
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            note = _seed_project(root)

            index_all(db, None, root, full=True)
            kb = KnowledgeBase._from_existing(db=db, project_root=root)
            kb.unlink_entity("Alex Tanner", note)
            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alex Tanner", "%Sync%") == 0

            result = kb.relink_entity("Alex Tanner", note)
            assert result["relinked"] is True

            # Re-derivation happens on the next index of the document.
            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alex Tanner", "%Sync%") >= 1, "link not restored"

    def test_unlink_unknown_entity_raises(self, tmp_db):
        from kb.api import KnowledgeBase
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            note = _seed_project(root)
            index_all(db, None, root, full=True)
            kb = KnowledgeBase._from_existing(db=db, project_root=root)
            try:
                kb.unlink_entity("Nonexistent Person", note)
            except ValueError as exc:
                assert "not found" in str(exc).lower()
            else:  # pragma: no cover - guard
                raise AssertionError("expected ValueError for unknown entity")

    def test_unlink_unknown_document_raises(self, tmp_db):
        from kb.api import KnowledgeBase
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _seed_project(root)
            index_all(db, None, root, full=True)
            kb = KnowledgeBase._from_existing(db=db, project_root=root)
            try:
                kb.unlink_entity("Alex Tanner", "no-such-document.md")
            except ValueError as exc:
                assert "not found" in str(exc).lower()
            else:  # pragma: no cover - guard
                raise AssertionError("expected ValueError for unknown document")


class TestUnlinkRelinkCLI:
    def _index(self, root: Path, data_dir: Path) -> None:
        from kb.db import Database
        from kb.indexer import index_all

        db = Database(data_dir)
        index_all(db, None, root, full=True)
        db.close()

    def test_entity_unlink_then_relink_cli(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        note = _seed_project(root)
        data_dir = tmp_path / "data"
        self._index(root, data_dir)

        with (
            patch("kb.cli._find_project_root", return_value=root),
            patch("kb.cli._get_data_dir", return_value=data_dir),
        ):
            runner = CliRunner()
            unlinked = runner.invoke(
                cli,
                ["entity", "unlink", "Alex Tanner", note, "--no-commit", "--json"],
                catch_exceptions=False,
            )
            assert unlinked.exit_code == 0, unlinked.output
            assert json.loads(unlinked.output)["unlinked"] is True

            relinked = runner.invoke(
                cli,
                ["entity", "relink", "Alex Tanner", note, "--no-commit", "--json"],
                catch_exceptions=False,
            )
            assert relinked.exit_code == 0, relinked.output
            assert json.loads(relinked.output)["relinked"] is True

    def test_entity_unlink_unknown_entity_exits_nonzero(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        note = _seed_project(root)
        data_dir = tmp_path / "data"
        self._index(root, data_dir)

        with (
            patch("kb.cli._find_project_root", return_value=root),
            patch("kb.cli._get_data_dir", return_value=data_dir),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["entity", "unlink", "Ghost Person", note, "--no-commit"],
                catch_exceptions=False,
            )
            assert result.exit_code == 1
            assert "not found" in result.output.lower()
