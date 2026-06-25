"""Integration tests for bare first-name disambiguation through index_all (#36)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path


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


def _seed_two_alexandres(root: Path) -> None:
    people = root / "memory" / "people"
    people.mkdir(parents=True)
    (people / "alexandre-dupont.md").write_text(
        "# Alexandre Dupont\n\n**Also known as:** Alexandre\n\n**Role:** Engineer\n"
    )
    (people / "alexandre-martin.md").write_text(
        "# Alexandre Martin\n\n**Also known as:** Alexandre\n\n**Role:** Designer\n"
    )


def _write_meeting(root: Path, name: str, body: str, attendees_yaml: str = "") -> None:
    d = root / "memory" / "meetings" / "2026" / "05" / "24"
    d.mkdir(parents=True, exist_ok=True)
    front = "---\ntitle: Sync\ndate: 2026-05-24\ntype: notes\ngranola_id: " + name[:8] + "\n"
    front += attendees_yaml
    front += "---\n\n## Notes\n\n" + body + "\n"
    (d / f"{name}.granola.notes.md").write_text(front)


class TestFirstNameDisambiguationIntegration:
    def test_ambiguous_bare_first_name_not_linked(self, tmp_db):
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _seed_two_alexandres(root)
            _write_meeting(root, "aaaa1111_bare", "Alexandre walked us through the roadmap.")

            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alexandre Dupont", "%bare%") == 0
            assert _count_mentions(conn, "Alexandre Martin", "%bare%") == 0

    def test_full_name_corroborates_one_alexandre(self, tmp_db):
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _seed_two_alexandres(root)
            _write_meeting(
                root,
                "bbbb2222_full",
                "Alexandre Dupont opened. Later Alexandre summarised the actions.",
            )

            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alexandre Dupont", "%full%") >= 1
            assert _count_mentions(conn, "Alexandre Martin", "%full%") == 0

    def test_attendee_corroborates_one_alexandre(self, tmp_db):
        from kb.indexer import index_all

        db, _ = tmp_db
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _seed_two_alexandres(root)
            _write_meeting(
                root,
                "cccc3333_att",
                "Alexandre walked us through the roadmap.",
                attendees_yaml="attendees:\n  - name: Alexandre Dupont\n    email: ad@example.com\n",
            )

            index_all(db, None, root, full=True)
            conn = db.get_sqlite_conn()
            assert _count_mentions(conn, "Alexandre Dupont", "%att%") >= 1
            assert _count_mentions(conn, "Alexandre Martin", "%att%") == 0
