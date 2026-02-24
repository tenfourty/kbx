"""KnowledgeBase -- public Python API for kbx.

Single entry point for all knowledge base operations. Owns DB + config +
embedder lifecycle. Consumers create one instance and call methods.

    from kb import KnowledgeBase

    with KnowledgeBase() as kb:
        results = kb.search("cloud migration")
        people = kb.list_entities(entity_type="person")
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from typing import TYPE_CHECKING

from kb.db import Database
from kb.types import (
    DocumentPinResult,
    EntityDetail,
    EntityFact,
    EntityPinResult,
    EntitySummary,
    GlossaryEntry,
    MemoryTreeNode,
    PinnedDocument,
    TimelineEntry,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kb.embeddings import Embedder
    from kb.types import ContextOutput, IndexResult, SearchResponse


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

    @staticmethod
    def _mention_counts(conn: sqlite3.Connection, entity_ids: list[int]) -> dict[int, int]:
        """Return {entity_id: mention_count} for the given IDs only."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" * len(entity_ids))
        rows = conn.execute(
            f"SELECT entity_id, COUNT(*) AS cnt FROM entity_mentions"
            f" WHERE entity_id IN ({placeholders}) GROUP BY entity_id",
            entity_ids,
        ).fetchall()
        return {r["entity_id"]: r["cnt"] for r in rows}

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
        if self._embedder is not None:
            self._embedder.release_gpu_memory()
        self._db.close()
        self._embedder = None
        self._embedder_failed = False

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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

        mention_map = self._mention_counts(conn, [r["id"] for r in rows])

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

        mention_map = self._mention_counts(conn, [r["id"] for r in rows])

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

    def get_entity_timeline(self, name: str, limit: int = 10) -> list[TimelineEntry]:
        """Return recent documents mentioning an entity.

        Resolves *name* via alias/partial matching. Returns empty list if not found.
        """
        from kb.config import find_entity

        conn = self._get_conn()
        entity = find_entity(conn, name)
        if entity is None:
            return []

        rows = conn.execute(
            "SELECT DISTINCT d.title, d.doc_date AS date, d.path"
            " FROM entity_mentions em"
            " JOIN documents d ON d.id = em.document_id"
            " WHERE em.entity_id = ?"
            " ORDER BY d.doc_date DESC"
            " LIMIT ?",
            (entity["id"], limit),
        ).fetchall()

        return [TimelineEntry(title=r["title"], date=r["date"], path=r["path"]) for r in rows]

    def toggle_entity_pin(self, name: str) -> EntityPinResult:
        """Toggle an entity's pinned state. Raises ValueError if not found.

        Resolves *name* via alias/partial matching (same as get_entity).
        """
        from kb.config import find_entity

        conn = self._get_conn()
        row = find_entity(conn, name)
        if row is None:
            raise ValueError(f"Entity not found: {name}")

        new_pinned = not bool(row["pinned"])
        conn.execute("UPDATE entities SET pinned = ? WHERE id = ?", (int(new_pinned), row["id"]))
        conn.commit()
        return EntityPinResult(name=row["name"], pinned=new_pinned)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """Return the total number of indexed documents."""
        row = self._get_conn().execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()
        return int(row["cnt"])

    def get_document_pin(self, path: str) -> bool:
        """Return whether a document is pinned."""
        row = (
            self._get_conn()
            .execute("SELECT pinned FROM documents WHERE path = ?", (path,))
            .fetchone()
        )
        if row is None:
            return False
        return bool(row["pinned"])

    def toggle_document_pin(self, path: str) -> DocumentPinResult:
        """Toggle a document's pinned state. Raises ValueError if not found."""
        conn = self._get_conn()
        row = conn.execute("SELECT id, pinned FROM documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise ValueError(f"Document not found: {path}")

        new_pinned = not bool(row["pinned"])
        conn.execute("UPDATE documents SET pinned = ? WHERE id = ?", (int(new_pinned), row["id"]))
        conn.commit()
        return DocumentPinResult(path=path, pinned=new_pinned)

    def list_pinned_documents(self) -> list[PinnedDocument]:
        """Return all pinned documents with their section headings."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, path, title FROM documents WHERE pinned = 1 ORDER BY title"
        ).fetchall()
        pinned: list[PinnedDocument] = []
        for r in rows:
            headings_rows = conn.execute(
                "SELECT DISTINCT heading FROM chunks"
                " WHERE document_id = ? AND heading IS NOT NULL ORDER BY chunk_index",
                (r["id"],),
            ).fetchall()
            headings = [h["heading"] for h in headings_rows]
            pinned.append(PinnedDocument(path=r["path"], title=r["title"], headings=headings))
        return pinned

    # ------------------------------------------------------------------
    # Memory file operations
    # ------------------------------------------------------------------

    def _memory_dir(self) -> Path:
        return self._project_root / "memory"

    def _validate_memory_path(self, relative_path: str) -> Path | None:
        """Validate a relative path within memory/. Returns None if invalid."""
        if ".." in relative_path:
            return None
        if not relative_path.endswith(".md"):
            return None
        mem_dir = self._memory_dir()
        resolved = (mem_dir / relative_path).resolve()
        if not resolved.is_relative_to(mem_dir.resolve()):
            return None
        return resolved

    def read_memory_file(self, relative_path: str) -> str | None:
        """Read a markdown file from memory/. Returns None if invalid or not found."""
        resolved = self._validate_memory_path(relative_path)
        if resolved is None or not resolved.is_file():
            return None
        return resolved.read_text(encoding="utf-8")

    def write_memory_file(self, relative_path: str, content: str) -> bool:
        """Write content to an existing markdown file in memory/.

        Returns False if the path is invalid or the file doesn't exist.
        Does not create new files. Uses atomic temp-file-then-rename.
        """
        resolved = self._validate_memory_path(relative_path)
        if resolved is None or not resolved.is_file():
            return False
        fd, tmp_path = tempfile.mkstemp(dir=str(resolved.parent), suffix=".md.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, str(resolved))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        return True

    def list_memory_tree(self) -> list[MemoryTreeNode]:
        """Return the memory/ directory as a nested tree with pinned state."""
        mem_dir = self._memory_dir()
        if not mem_dir.is_dir():
            return []

        pinned_paths: set[str] = set()
        try:
            rows = (
                self._get_conn().execute("SELECT path FROM documents WHERE pinned = 1").fetchall()
            )
            pinned_paths = {r["path"] for r in rows}
        except Exception:
            pass

        def _scan(directory: Path, prefix: str = "") -> list[MemoryTreeNode]:
            items: list[MemoryTreeNode] = []
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                return items

            for entry in entries:
                rel = f"{prefix}/{entry.name}".lstrip("/") if prefix else entry.name
                full_rel = f"memory/{rel}"

                if entry.is_dir():
                    children = _scan(entry, rel)
                    file_count = sum(1 for c in children if c.node_type == "file") + sum(
                        c.count for c in children if c.node_type == "dir"
                    )
                    items.append(
                        MemoryTreeNode(
                            name=entry.name,
                            node_type="dir",
                            path=rel,
                            children=children,
                            count=file_count,
                        )
                    )
                elif entry.suffix == ".md":
                    items.append(
                        MemoryTreeNode(
                            name=entry.name,
                            node_type="file",
                            path=rel,
                            pinned=full_rel in pinned_paths,
                        )
                    )
            return items

        return _scan(mem_dir)

    # ------------------------------------------------------------------
    # Glossary
    # ------------------------------------------------------------------

    def list_glossary_terms(self) -> list[GlossaryEntry]:
        """Return all glossary terms from memory/glossary.md."""
        from kb.glossary import list_terms

        return list_terms(self._project_root)

    # ------------------------------------------------------------------
    # Embedder
    # ------------------------------------------------------------------

    def warm_embedder(self) -> bool:
        """Pre-load the embedding model without triggering search or reindex.

        Returns True if the embedder was loaded, False if embeddings are disabled.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return False
        embedder.embed_query("warmup")
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        fast: bool = False,
        recency: float = 0.15,
        doc_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        tag: str | None = None,
        sort_by: str = "score",
    ) -> SearchResponse:
        """Run a hybrid search (FTS5 + vector + RRF fusion).

        Set ``fast=True`` for FTS-only (no embedder loaded).
        The embedder is lazy-loaded on first non-fast search and reused.
        """
        from kb.search import search as do_search
        from kb.staleness import auto_reindex_if_stale

        auto_reindex_if_stale(self._db, self._project_root)

        embedder = None if fast else self._get_embedder()
        return do_search(
            self._db,
            embedder,
            query,
            limit=limit,
            fast=fast or embedder is None,
            recency=recency,
            doc_type=doc_type,
            from_date=from_date,
            to_date=to_date,
            tag=tag,
            sort_by=sort_by,
        )

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def context(
        self,
        topic: str | None = None,
        fmt: str = "compact",
    ) -> ContextOutput:
        """Generate compressed entity context for AI agents."""
        from kb.context import generate_context
        from kb.staleness import auto_reindex_if_stale

        auto_reindex_if_stale(self._db, self._project_root)
        return generate_context(self._db, self._project_root, topic=topic, fmt=fmt)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, *, full: bool = False) -> IndexResult:
        """Run the indexing pipeline. Reuses the shared embedder instance."""
        from kb.indexer import index_all

        embedder = self._get_embedder()
        return index_all(self._db, embedder, self._project_root, full=full)
