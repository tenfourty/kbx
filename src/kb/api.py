"""KnowledgeBase -- public Python API for kbx.

Single entry point for all knowledge base operations. Owns DB + config +
embedder lifecycle. Consumers create one instance and call methods.

    from kb import KnowledgeBase

    with KnowledgeBase() as kb:
        results = kb.search("cloud migration")
        people = kb.list_entities(entity_type="person")
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from kb.db import Database
from kb.types import EntityDetail, EntityFact, EntitySummary

if TYPE_CHECKING:
    from pathlib import Path

    from kb.embeddings import Embedder


class KnowledgeBase:
    """Public API for the kbx knowledge base.

    Parameters
    ----------
    project_root:
        Path to the project root (contains ``memory/``, ``meetings/``).
        Auto-discovered from ``kbx.toml`` or CWD walk-up if not provided.
    data_dir:
        Path to the database directory (contains ``metadata.db``, ``vectors/``).
        Auto-discovered from config / ``KB_DATA_DIR`` / ``~/.config/kbx/`` if not provided.
    thread_safe:
        If True, opens the SQLite connection with ``check_same_thread=False``
        and enables WAL mode. Use this when sharing the instance across threads
        (e.g. FastAPI route handlers).
    """

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        data_dir: Path | None = None,
        thread_safe: bool = False,
    ) -> None:
        if project_root is None:
            from kb.config import find_project_root

            project_root = find_project_root()
        if data_dir is None:
            from kb.config import get_data_dir

            data_dir = get_data_dir()

        self._project_root = project_root
        self._data_dir = data_dir
        self._thread_safe = thread_safe
        self._embedder: Embedder | None = None
        self._embedder_failed = False

        self._db = Database(data_dir)

        if thread_safe:
            self._replace_conn_thread_safe()

    def _replace_conn_thread_safe(self) -> None:
        """Replace the DB connection with a thread-safe one."""
        old_conn = self._db.get_sqlite_conn()
        db_path = str(self._data_dir / "metadata.db")
        new_conn = sqlite3.connect(db_path, check_same_thread=False)
        new_conn.row_factory = sqlite3.Row
        new_conn.execute("PRAGMA foreign_keys=ON")
        new_conn.execute("PRAGMA synchronous=NORMAL")
        new_conn.execute("PRAGMA cache_size=-64000")
        new_conn.execute("PRAGMA journal_mode=WAL")
        new_conn.execute("PRAGMA busy_timeout=5000")
        old_conn.close()
        self._db._sqlite_conn = new_conn

    def _get_conn(self) -> sqlite3.Connection:
        """Get the SQLite connection."""
        return self._db.get_sqlite_conn()

    def _get_embedder(self) -> Embedder | None:
        """Lazy-load the embedder, returning None if unavailable."""
        if self._embedder is not None:
            return self._embedder
        if self._embedder_failed:
            return None
        try:
            from kb.embeddings import Embedder as _Embedder

            self._embedder = _Embedder()
            return self._embedder
        except Exception:
            self._embedder_failed = True
            return None

    def close(self) -> None:
        """Release all resources (DB connection, embedder GPU memory)."""
        self._db.close()
        self._embedder = None
        self._embedder_failed = False

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """Return the total number of indexed documents."""
        row = self._get_conn().execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()
        return int(row["cnt"])

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def list_entities(self, entity_type: str | None = None) -> list[EntitySummary]:
        """List entities, optionally filtered by type. Pinned first, then by name."""
        conn = self._get_conn()
        if entity_type:
            rows = conn.execute(
                "SELECT id, name, entity_type, metadata, pinned"
                " FROM entities WHERE entity_type = ? ORDER BY name",
                (entity_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, entity_type, metadata, pinned FROM entities ORDER BY name"
            ).fetchall()

        mention_rows = conn.execute(
            "SELECT entity_id, COUNT(*) AS cnt FROM entity_mentions GROUP BY entity_id"
        ).fetchall()
        mention_map: dict[int, int] = {r["entity_id"]: r["cnt"] for r in mention_rows}

        results = [
            EntitySummary(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                mention_count=mention_map.get(r["id"], 0),
                pinned=bool(r["pinned"]),
            )
            for r in rows
        ]
        results.sort(key=lambda e: (not e.pinned, e.name.lower()))
        return results

    def get_entity(self, name: str) -> EntityDetail | None:
        """Get full entity detail by name (case-insensitive, supports aliases).

        Returns None if not found.
        """
        from kb.config import find_entity

        conn = self._get_conn()
        row = find_entity(conn, name)
        if row is None:
            return None

        entity_id: int = row["id"]
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        aliases = json.loads(row["aliases"]) if row["aliases"] else []

        mention_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM entity_mentions WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()

        fact_rows = conn.execute(
            "SELECT fact_text, fact_date FROM facts WHERE entity_id = ?"
            " ORDER BY fact_date DESC, id DESC",
            (entity_id,),
        ).fetchall()

        return EntityDetail(
            id=entity_id,
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=aliases,
            metadata=meta,
            mention_count=mention_row["cnt"] if mention_row else 0,
            pinned=bool(row["pinned"]),
            source_path=row["source_path"],
            facts=[EntityFact(text=f["fact_text"], date=f["fact_date"]) for f in fact_rows],
        )

    def find_entities(self, name: str) -> list[EntitySummary]:
        """Find entities by name/alias (case-insensitive, partial match).

        Returns matches in priority order: exact name > exact alias > partial.
        """
        from kb.config import find_entities as _find_entities

        conn = self._get_conn()
        rows = _find_entities(conn, name)
        if not rows:
            return []

        mention_rows = conn.execute(
            "SELECT entity_id, COUNT(*) AS cnt FROM entity_mentions GROUP BY entity_id"
        ).fetchall()
        mention_map: dict[int, int] = {r["entity_id"]: r["cnt"] for r in mention_rows}

        return [
            EntitySummary(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                mention_count=mention_map.get(r["id"], 0),
                pinned=bool(r["pinned"]),
            )
            for r in rows
        ]
