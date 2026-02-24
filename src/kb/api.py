"""KnowledgeBase -- public Python API for kbx.

Single entry point for all knowledge base operations. Owns DB + config +
embedder lifecycle. Consumers create one instance and call methods.

    from kb import KnowledgeBase

    with KnowledgeBase() as kb:
        results = kb.search("cloud migration")
        people = kb.list_entities(entity_type="person")
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from kb.db import Database

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
