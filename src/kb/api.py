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
from typing import TYPE_CHECKING, Any

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
        self._owns_db = True

        self._db = Database(data_dir)

        if thread_safe:
            self._replace_conn_thread_safe()

    @classmethod
    def _from_existing(cls, *, db: Database, project_root: Path) -> KnowledgeBase:
        """Create a KnowledgeBase that reuses an existing Database instance.

        The caller retains ownership of the db — close() will NOT close it.
        Embedder is disabled (text-only indexing).
        """
        inst = object.__new__(cls)
        inst._project_root = project_root
        inst._data_dir = None  # type: ignore[assignment]
        inst._thread_safe = False
        inst._embedder = None
        inst._embedder_failed = True  # skip embedder in shared-db context
        inst._owns_db = False
        inst._db = db
        return inst

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

    def get_fact_counts(self, entity_ids: list[int] | None = None) -> dict[int, int]:
        """Return {entity_id: fact_count} in a single query.

        If *entity_ids* is provided, only count facts for those entities.
        An empty list returns an empty dict immediately.
        Entities with zero facts are absent from the result.
        """
        conn = self._get_conn()
        if entity_ids is not None:
            if not entity_ids:
                return {}
            placeholders = ",".join("?" * len(entity_ids))
            rows = conn.execute(
                f"SELECT entity_id, COUNT(*) AS cnt FROM facts"
                f" WHERE entity_id IN ({placeholders}) GROUP BY entity_id",
                entity_ids,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT entity_id, COUNT(*) AS cnt FROM facts GROUP BY entity_id"
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
        """Release all resources (DB connection, embedder GPU memory).

        When created via _from_existing(), the DB is not owned and won't be closed.
        """
        if self._embedder is not None:
            self._embedder.release_gpu_memory()
        if self._owns_db:
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
                "SELECT id, name, entity_type, aliases, metadata, pinned"
                " FROM entities WHERE entity_type = ? ORDER BY name",
                (entity_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, entity_type, aliases, metadata, pinned"
                " FROM entities ORDER BY name"
            ).fetchall()

        mention_map = self._mention_counts(conn, [r["id"] for r in rows])

        results = [
            EntitySummary(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                aliases=json.loads(r["aliases"]) if r["aliases"] else [],
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                mention_count=mention_map.get(r["id"], 0),
                pinned=bool(r["pinned"]),
            )
            for r in rows
        ]
        results.sort(key=lambda e: (not e.pinned, e.name.lower()))
        return results

    def match_tasks_to_projects(
        self,
        tasks: list[dict[str, Any]],
        *,
        min_keyword_len: int = 4,
    ) -> dict[str, list[dict[str, Any]]]:
        """Match tasks to projects using the two-tier algorithm.

        Loads project entities from the DB, converts to ``ProjectInput`` dicts,
        delegates to ``matching.match_tasks_to_projects()``.

        Parameters
        ----------
        tasks:
            ``TaskInput``-shaped dicts with ``title`` and optional ``description``.
        min_keyword_len:
            Word-boundary threshold (see ``matching`` module docs).

        Returns
        -------
        ``{project_name: [matched_tasks]}``
        """
        from typing import cast

        from kb.matching import ProjectInput, TaskInput
        from kb.matching import match_tasks_to_projects as _match

        projects = self.list_entities(entity_type="project")
        project_dicts = [
            ProjectInput(
                name=p.name,
                aliases=p.aliases,
                metadata=p.metadata,
            )
            for p in projects
        ]
        task_inputs = cast("list[TaskInput]", tasks)
        result = _match(project_dicts, task_inputs, min_keyword_len=min_keyword_len)
        return cast("dict[str, list[dict[str, Any]]]", result)

    def get_entity(self, name: str, *, _track_access: bool = True) -> EntityDetail | None:
        """Get full entity detail by name (case-insensitive, supports aliases).

        Returns None if not found. When ``_track_access=True`` (the default for
        external callers via API/CLI/MCP), the entity's ``access_count`` is
        incremented for hotness scoring (#67). Internal lookups that should
        not pollute the hotness signal pass ``_track_access=False``.
        """
        from kb.access import touch_entity
        from kb.config import find_entity

        conn = self._get_conn()
        row = find_entity(conn, name)
        if row is None:
            return None

        entity_id: int = row["id"]
        if _track_access:
            touch_entity(conn, entity_id)
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        aliases = json.loads(row["aliases"]) if row["aliases"] else []

        mention_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM entity_mentions WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()

        fact_rows = conn.execute(
            "SELECT seq, fact_text, fact_date FROM facts WHERE entity_id = ? ORDER BY seq ASC",
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
            facts=[
                EntityFact(seq=f["seq"], text=f["fact_text"], date=f["fact_date"])
                for f in fact_rows
            ],
        )

    def get_entity_profile(self, name: str) -> dict[str, Any] | None:
        """Get a compact entity profile with facts (incl. IDs), timestamps, and breadcrumbs.

        Returns None if the entity is not found. This is the shared method
        backing both CLI and MCP entity find commands.
        """
        import shlex
        from datetime import date, timedelta

        entity = self.get_entity(name)
        if entity is None:
            return None

        conn = self._get_conn()
        ts_row = conn.execute(
            "SELECT updated_at, last_mentioned_at FROM entities WHERE id = ?",
            (entity.id,),
        ).fetchone()

        entity_type = entity.entity_type
        source_path = entity.source_path

        quoted_name = shlex.quote(entity.name)
        thirty_ago = (date.today() - timedelta(days=30)).isoformat()

        # CLI breadcrumbs (command-line hints)
        cli_breadcrumbs: dict[str, str] = {}
        if entity_type in ("person", "project"):
            cli_breadcrumbs["timeline"] = f"kbx {entity_type} timeline {quoted_name} --limit 20"
            cli_breadcrumbs["recent"] = (
                f"kbx {entity_type} timeline {quoted_name} --from {thirty_ago}"
            )
        else:
            cli_breadcrumbs["search"] = f"kbx search {quoted_name} --limit 20"
        if source_path:
            cli_breadcrumbs["profile"] = f"kbx view {source_path}"

        # MCP breadcrumbs (tool references for agents)
        mcp_breadcrumbs: list[dict[str, Any]] = []
        if entity_type in ("person", "project"):
            mcp_breadcrumbs.append(
                {
                    "tool": "kb_person_timeline",
                    "params": {"name": entity.name, "limit": 20},
                }
            )
            mcp_breadcrumbs.append(
                {
                    "tool": "kb_person_timeline",
                    "params": {"name": entity.name, "from_date": thirty_ago},
                }
            )
        else:
            mcp_breadcrumbs.append(
                {
                    "tool": "kb_search",
                    "params": {"query": entity.name, "limit": 20},
                }
            )
        if source_path:
            mcp_breadcrumbs.append(
                {
                    "tool": "kb_view",
                    "params": {"target": source_path},
                }
            )

        return {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity_type,
            "aliases": entity.aliases,
            "metadata": entity.metadata,
            "updated_at": ts_row["updated_at"] if ts_row else None,
            "last_mentioned_at": ts_row["last_mentioned_at"] if ts_row else None,
            "facts": [{"seq": f.seq, "text": f.text, "date": f.date} for f in entity.facts],
            "source_path": source_path,
            "document_count": entity.mention_count,
            "breadcrumbs": cli_breadcrumbs,
            "mcp_breadcrumbs": mcp_breadcrumbs,
        }

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
                aliases=json.loads(r["aliases"]) if r["aliases"] else [],
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                mention_count=mention_map.get(r["id"], 0),
                pinned=bool(r["pinned"]),
            )
            for r in rows
        ]

    def get_entity_timeline(
        self,
        name: str,
        limit: int | None = 10,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        doc_type: str | None = None,
    ) -> list[TimelineEntry]:
        """Return documents mentioning an entity, newest first.

        Resolves *name* via alias/partial matching. Returns empty list if not found.
        Increments the entity's ``access_count`` for hotness scoring (#67).
        """
        from kb.access import touch_entity
        from kb.config import find_entity

        conn = self._get_conn()
        entity = find_entity(conn, name)
        if entity is None:
            return []

        touch_entity(conn, entity["id"])
        sql = (
            "SELECT d.title, d.doc_date AS date, d.path, d.doc_type,"
            " GROUP_CONCAT(DISTINCT em.mention_type) AS mention_type"
            " FROM entity_mentions em"
            " JOIN documents d ON d.id = em.document_id"
            " WHERE em.entity_id = ?"
        )
        params: list[Any] = [entity["id"]]

        if from_date:
            sql += " AND d.doc_date >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND d.doc_date <= ?"
            params.append(to_date)
        if doc_type:
            sql += " AND d.doc_type = ?"
            params.append(doc_type)

        sql += " GROUP BY d.id ORDER BY d.doc_date DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()

        return [
            TimelineEntry(
                title=r["title"],
                date=r["date"],
                path=r["path"],
                doc_type=r["doc_type"],
                mention_type=r["mention_type"],
            )
            for r in rows
        ]

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

    def get_stale_entities(
        self,
        days: int = 30,
        entity_type: str | None = None,
    ) -> list[dict[str, object]]:
        """Return entities where both updated_at and last_mentioned_at are older than *days*."""
        from datetime import date, timedelta

        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = self._get_conn()

        query = """
            SELECT id, name, entity_type, metadata, updated_at, last_mentioned_at, pinned
            FROM entities
            WHERE (updated_at IS NULL OR updated_at < ?)
              AND (last_mentioned_at IS NULL OR last_mentioned_at < ?)
        """
        params: list[object] = [cutoff, cutoff]
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)
        query += " ORDER BY COALESCE(last_mentioned_at, updated_at, '') ASC"

        rows = conn.execute(query, params).fetchall()
        results: list[dict[str, object]] = []
        for r in rows:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            most_recent = max(r["updated_at"] or "", r["last_mentioned_at"] or "")
            age_days = (
                (date.today() - date.fromisoformat(most_recent)).days if most_recent else None
            )
            results.append(
                {
                    "name": r["name"],
                    "entity_type": r["entity_type"],
                    "role": meta.get("role"),
                    "team": meta.get("team"),
                    "updated_at": r["updated_at"],
                    "last_mentioned_at": r["last_mentioned_at"],
                    "age_days": age_days,
                    "pinned": bool(r["pinned"]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Fact operations
    # ------------------------------------------------------------------

    def add_fact(
        self,
        entity_name: str,
        text: str,
        date: str | None = None,
    ) -> dict[str, object]:
        """Add a fact: DB insert + append to entity markdown file.

        Raises ValueError if the entity is not found.
        """
        from datetime import date as date_cls

        from kb.config import find_entity

        conn = self._get_conn()
        if date is None:
            date = date_cls.today().isoformat()

        entity_row = find_entity(conn, entity_name)
        if entity_row is None:
            raise ValueError(f"Entity not found: {entity_name}")

        # Compute next seq for this entity
        max_seq_row = conn.execute(
            "SELECT MAX(seq) as m FROM facts WHERE entity_id = ?",
            (entity_row["id"],),
        ).fetchone()
        next_seq = (max_seq_row["m"] or 0) + 1

        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date, seq) VALUES (?, ?, ?, ?)",
            (entity_row["id"], text, date, next_seq),
        )
        conn.commit()

        # Write-through: append fact to entity markdown file
        source_path = entity_row["source_path"]
        if source_path:
            source_file = self._project_root / source_path
            if source_file.exists():
                self._append_fact_to_file(source_file, text, date)

        return {"status": "ok", "type": "fact", "entity": entity_row["name"]}

    def _append_fact_to_file(self, path: Path, fact_text: str, fact_date: str) -> None:
        """Append a fact to a markdown file under a ## Recent Facts section."""
        import re as _re

        content = path.read_text(encoding="utf-8")
        fact_line = f"- [{fact_date}] {fact_text}\n"

        if "## Recent Facts" in content:
            header_match = _re.search(r"^## Recent Facts\s*$", content, _re.MULTILINE)
            if header_match is None:
                return  # pragma: no cover — defensive guard
            next_section = _re.search(r"^## ", content[header_match.end() :], _re.MULTILINE)
            if next_section:
                insert_pos = header_match.end() + next_section.start()
                content = content[:insert_pos] + fact_line + "\n" + content[insert_pos:]
            else:
                content = content.rstrip("\n") + "\n" + fact_line
        else:
            content = content.rstrip("\n") + "\n\n## Recent Facts\n" + fact_line

        self._atomic_write(path, content)

    def _resolve_fact(self, entity: str, seq: int) -> dict[str, Any]:
        """Resolve (entity_name, seq) to a fact row. Raises ValueError if not found."""
        from kb.config import find_entity

        conn = self._get_conn()
        entity_row = find_entity(conn, entity)
        if entity_row is None:
            raise ValueError(f"Entity not found: {entity}")

        row = conn.execute(
            "SELECT f.id, f.entity_id, f.fact_text, f.fact_date, f.seq, e.source_path "
            "FROM facts f JOIN entities e ON f.entity_id = e.id "
            "WHERE f.entity_id = ? AND f.seq = ?",
            (entity_row["id"], seq),
        ).fetchone()
        if row is None:
            raise ValueError(f"Fact #{seq} not found on entity '{entity_row['name']}'")
        return dict(row)

    def delete_fact(self, entity: str, seq: int) -> dict[str, object]:
        """Delete a fact by entity name and sequence number.

        Raises ValueError if the entity or fact is not found.
        """
        row = self._resolve_fact(entity, seq)
        conn = self._get_conn()

        fact_text = row["fact_text"]
        fact_date = row["fact_date"]
        source_path = row["source_path"]

        conn.execute("DELETE FROM facts WHERE id = ?", (row["id"],))
        conn.commit()

        if source_path:
            self._remove_fact_from_file(source_path, fact_text, fact_date or "")
            self._reindex_memory_file(source_path)

        return {"entity": entity, "seq": seq, "fact_text": fact_text, "deleted": True}

    def edit_fact(
        self,
        entity: str,
        seq: int,
        *,
        text: str | None = None,
        date: str | None = None,
    ) -> dict[str, object]:
        """Edit a fact's text and/or date by entity name and sequence number.

        Raises ValueError if entity/fact not found or no changes specified.
        """
        import re as _re

        if text is None and date is None:
            raise ValueError("No changes specified. Provide text and/or date.")

        row = self._resolve_fact(entity, seq)
        conn = self._get_conn()

        old_text = row["fact_text"]
        old_date = row["fact_date"]
        new_text = text if text is not None else old_text
        new_date = date if date is not None else old_date
        source_path = row["source_path"]

        conn.execute(
            "UPDATE facts SET fact_text = ?, fact_date = ? WHERE id = ?",
            (new_text, new_date, row["id"]),
        )
        conn.commit()

        if source_path:
            file_path = self._project_root / source_path
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                old_line_pattern = (
                    rf"^- \[{_re.escape(old_date or '')}\] {_re.escape(old_text)}\s*$"
                )
                new_line = f"- [{new_date or ''}] {new_text}"
                new_content = _re.sub(
                    old_line_pattern, new_line, content, count=1, flags=_re.MULTILINE
                )
                if new_content != content:
                    self._atomic_write(file_path, new_content)
            self._reindex_memory_file(source_path)

        return {
            "entity": entity,
            "seq": seq,
            "fact_text": new_text,
            "fact_date": new_date,
            "updated": True,
        }

    def _remove_fact_from_file(self, source_path: str, fact_text: str, fact_date: str) -> None:
        """Remove a fact line from an entity's markdown file."""
        import re as _re

        file_path = self._project_root / source_path
        if not file_path.exists():
            return
        content = file_path.read_text(encoding="utf-8")
        pattern = rf"^- \[{_re.escape(fact_date)}\] {_re.escape(fact_text)}\s*$\n?"
        new_content = _re.sub(pattern, "", content, count=1, flags=_re.MULTILINE)
        if new_content != content:
            self._atomic_write(file_path, new_content)

    @staticmethod
    def _atomic_write(file_path: Path, content: str) -> None:
        """Atomic write: temp file then rename."""
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=str(file_path.parent), suffix=".md.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, str(file_path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _reindex_memory_file(self, rel_path: str) -> None:
        """Re-index a single memory file: delete old data, re-parse, re-insert.

        Updates SQLite + FTS5 (via triggers). Also cleans up old LanceDB
        vectors and re-embeds if an embedder is already loaded.
        """
        from kb.db import normalize_path
        from kb.entities import build_entity_patterns, find_entity_mentions, load_entities
        from kb.indexer import _delete_document
        from kb.sources.memory import _SUBDIR_DOC_TYPES, _parse_glossary, _parse_memory_file

        conn = self._get_conn()
        norm_path = normalize_path(rel_path)

        # Get old document_id for LanceDB cleanup before deleting from SQLite
        old_row = conn.execute("SELECT id FROM documents WHERE path = ?", (norm_path,)).fetchone()
        old_doc_id = old_row["id"] if old_row else None

        # Delete old document + chunks + mentions (cascade handles attendees)
        _delete_document(self._db, norm_path)
        conn.commit()

        # Clean up old vectors from LanceDB (best-effort)
        if old_doc_id is not None:
            lance_table = self._db.get_lance_table()
            if lance_table is not None:
                with contextlib.suppress(Exception):
                    lance_table.delete(f"document_id = {old_doc_id}")

        # Re-parse the file
        abs_path = self._project_root / rel_path
        if not abs_path.exists():
            return

        if norm_path == "memory/glossary.md":
            text = abs_path.read_text(encoding="utf-8")
            doc = _parse_glossary(text, norm_path)
        else:
            # Determine doc_type from subdirectory
            parts = norm_path.split("/")
            subdir = parts[1] if len(parts) > 2 else ""
            doc_type = _SUBDIR_DOC_TYPES.get(subdir, "memory_doc")
            doc = _parse_memory_file(abs_path, norm_path, doc_type)

        # Insert document row
        cursor = conn.execute(
            """INSERT INTO documents (path, title, doc_date, doc_type, source_system,
               source_id, tags, content_hash, chunk_count, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.path,
                doc.title,
                doc.date,
                doc.doc_type,
                doc.source_system,
                doc.source_id,
                json.dumps(doc.tags),
                doc.content_hash,
                len(doc.chunks),
                1 if doc.pinned else 0,
            ),
        )
        doc_id = cursor.lastrowid
        assert doc_id is not None

        # Insert chunks (FTS5 triggers handle the virtual table)
        chunk_ids: list[int] = []
        for chunk in doc.chunks:
            c = conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content, metadata_prefix)"
                " VALUES (?, ?, ?, ?, ?)",
                (doc_id, chunk.index, chunk.heading, chunk.content, chunk.metadata_prefix),
            )
            assert c.lastrowid is not None
            chunk_ids.append(c.lastrowid)

        # Entity linking
        entities = load_entities(self._db)
        patterns = build_entity_patterns(entities)
        section_content = " ".join(c.content for c in doc.chunks)
        mentions = find_entity_mentions(
            doc.title, doc.tags, section_content, entities, cached_patterns=patterns
        )
        entity_id_set = {m.entity_id for m in mentions}
        if mentions:
            conn.executemany(
                "INSERT OR IGNORE INTO entity_mentions (entity_id, document_id, mention_type)"
                " VALUES (?, ?, ?)",
                [(m.entity_id, doc_id, m.mention_type) for m in mentions],
            )

        conn.commit()

        # Re-embed if embedder is already loaded (don't trigger lazy load)
        if self._embedder is not None and doc.chunks and chunk_ids:
            try:
                texts = [chunk.content for chunk in doc.chunks]
                embeddings = self._embedder.embed(texts)
                entity_ids_json = json.dumps(sorted(entity_id_set))
                lance_batch = [
                    {
                        "chunk_id": int(chunk_ids[i]),
                        "embedding": embeddings[i].tolist(),
                        "doc_type": doc.doc_type or "",
                        "doc_date": doc.date or "",
                        "tags": json.dumps(doc.tags),
                        "document_id": int(doc_id),
                        "entity_ids": entity_ids_json,
                    }
                    for i in range(len(chunk_ids))
                ]
                from kb.indexer import _flush_lance_batch

                _flush_lance_batch(self._db, lance_batch)
            except Exception:
                pass  # Embedding is best-effort; FTS is always correct

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def view_document(self, target: str) -> dict[str, object]:
        """View a document by path, hash, title, glob, or substring.

        Returns dict with title, path, date, doc_type, content_hash, chunks.
        Raises ValueError if not found. Returns dict with 'ambiguous' key if ambiguous.
        Increments the document's ``access_count`` for hotness scoring (#67).
        """
        from kb.access import touch_document
        from kb.crud import AmbiguousDocumentError, find_document_by_target

        conn = self._get_conn()
        try:
            doc = find_document_by_target(conn, target, strict=True)
        except AmbiguousDocumentError as e:
            return {"error": f"Ambiguous target: {len(e.matches)} matches", "matches": e.matches}
        if doc is None:
            raise ValueError(f"Document not found: {target}")

        touch_document(conn, doc["id"])
        chunks = conn.execute(
            "SELECT heading, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc["id"],),
        ).fetchall()

        return {
            "title": doc["title"],
            "path": doc["path"],
            "date": doc.get("doc_date"),
            "doc_type": doc.get("doc_type"),
            "content_hash": doc.get("content_hash"),
            "chunks": [{"heading": c["heading"], "content": c["content"]} for c in chunks],
        }

    def list_documents(
        self,
        doc_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 25,
        since_hours: int | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        """List indexed documents with optional filters.

        Returns (results, total).
        """
        conn = self._get_conn()
        where_parts: list[str] = []
        params: list[object] = []

        if doc_type:
            where_parts.append("doc_type = ?")
            params.append(doc_type)
        if from_date:
            where_parts.append("doc_date >= ?")
            params.append(from_date)
        if to_date:
            where_parts.append("doc_date <= ?")
            params.append(to_date)
        if since_hours is not None:
            where_parts.append("indexed_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")

        where = " AND ".join(where_parts) if where_parts else "1=1"

        total_row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM documents WHERE {where}", params
        ).fetchone()
        total = int(total_row["cnt"])

        rows = conn.execute(
            f"SELECT id, path, title, doc_date, doc_type, source_system, pinned "
            f"FROM documents WHERE {where} ORDER BY doc_date DESC LIMIT ?",
            [*params, limit],
        ).fetchall()

        results = [
            {
                "id": r["id"],
                "path": r["path"],
                "title": r["title"],
                "date": r["doc_date"],
                "doc_type": r["doc_type"],
                "source_system": r["source_system"],
                "pinned": bool(r["pinned"]),
            }
            for r in rows
        ]
        return results, total

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

    def list_notes(
        self,
        tag: str | None = None,
        pinned_only: bool = False,
        limit: int = 25,
    ) -> tuple[list[dict[str, object]], int]:
        """List memory notes with optional tag/pin filters.

        Returns (results, total) where total is the full count before limiting.
        """
        conn = self._get_conn()

        where = "doc_type IN ('memory_note', 'memory_doc')"
        params: list[object] = []
        if pinned_only:
            where += " AND pinned = 1"

        required_tags: list[str] = []
        if tag:
            required_tags = [t.strip().lower() for t in tag.split(",") if t.strip()]

        if required_tags:
            all_rows = conn.execute(
                f"SELECT id, path, title, doc_date, tags, pinned FROM documents "
                f"WHERE {where} ORDER BY doc_date DESC, id DESC",
                params,
            ).fetchall()

            matching: list[dict[str, object]] = []
            for r in all_rows:
                doc_tags: list[str] = json.loads(r["tags"]) if r["tags"] else []
                lower_tags = [t.lower() for t in doc_tags]
                if all(rt in lower_tags for rt in required_tags):
                    matching.append(
                        {
                            "path": r["path"],
                            "title": r["title"],
                            "date": r["doc_date"],
                            "tags": doc_tags,
                            "pinned": bool(r["pinned"]),
                        }
                    )
            return matching[:limit], len(matching)

        total_row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM documents WHERE {where}", params
        ).fetchone()
        total = int(total_row["cnt"])

        rows = conn.execute(
            f"SELECT id, path, title, doc_date, tags, pinned FROM documents "
            f"WHERE {where} ORDER BY doc_date DESC, id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()

        results = [
            {
                "path": r["path"],
                "title": r["title"],
                "date": r["doc_date"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "pinned": bool(r["pinned"]),
            }
            for r in rows
        ]
        return results, total

    def delete_note(self, path: str) -> dict[str, object]:
        """Delete a memory note by path. Removes file + all DB records.

        Raises ValueError if the document is not found or is not a memory note.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise ValueError(f"Note not found: {path}")

        if row["doc_type"] not in ("memory_note", "memory_doc"):
            raise ValueError(
                f"Not a memory note (doc_type={row['doc_type']}). Only memory notes can be deleted."
            )

        doc_id = row["id"]

        # Delete file from disk
        file_path = self._project_root / path
        if file_path.exists():
            file_path.unlink()

        # Clean DB: chunks, entity_mentions, documents
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM entity_mentions WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

        # Clean up LanceDB vectors
        # Clean up LanceDB vectors (best-effort)
        lance_table = self._db.get_lance_table()
        if lance_table is not None:
            with contextlib.suppress(Exception):
                lance_table.delete(f"document_id = {doc_id}")

        return {"path": path, "title": row["title"], "deleted": True}

    def add_note(
        self,
        title: str,
        *,
        body: str | None = None,
        tags: list[str] | None = None,
        pin: bool = False,
        entity: str | None = None,
        date: str | None = None,
    ) -> dict[str, object]:
        """Create a memory note: build frontmatter, atomic write, index, pin, entity-link.

        Returns dict with status, type, path, pinned.
        """
        import re as _re
        from datetime import date as date_cls

        from kb.config import find_entity
        from kb.db import normalize_path

        conn = self._get_conn()
        if date is None:
            date = date_cls.today().isoformat()

        # Slugify title
        slug = _re.sub(r"[^\w\s-]", "", title.lower().strip())
        slug = _re.sub(r"[\s_]+", "-", slug)
        slug = _re.sub(r"-+", "-", slug)[:60].strip("-")

        notes_dir = self._project_root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        filepath = notes_dir / f"{date}-{slug}.md"
        counter = 2
        while filepath.exists():
            filepath = notes_dir / f"{date}-{slug}-{counter}.md"
            counter += 1

        # Build frontmatter
        frontmatter_lines = ["---", f"title: {title}", f"date: {date}"]
        if tags:
            frontmatter_lines.append(f"tags: [{', '.join(tags)}]")
        if pin:
            frontmatter_lines.append("pinned: true")
        frontmatter_lines.extend(["---", ""])
        note_body = body if body else title
        note_content = "\n".join(frontmatter_lines) + note_body + "\n"

        # Atomic write
        self._atomic_write(filepath, note_content)

        rel_path = normalize_path(str(filepath.relative_to(self._project_root)))

        # Index so the note is immediately searchable
        from kb.indexer import index_all

        index_all(self._db, None, self._project_root, memory_only=True, skip_seed=True)

        # Pin in DB
        if pin:
            conn.execute("UPDATE documents SET pinned = 1 WHERE path = ?", (rel_path,))
            conn.commit()

        # Entity linking
        if entity:
            entity_row = find_entity(conn, entity)
            if entity_row:
                doc_row = conn.execute(
                    "SELECT id FROM documents WHERE path = ?", (rel_path,)
                ).fetchone()
                if doc_row:
                    conn.execute(
                        "INSERT OR IGNORE INTO entity_mentions"
                        " (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                        (entity_row["id"], doc_row["id"], "tagged"),
                    )
                    conn.commit()

        return {"status": "ok", "type": "note", "path": rel_path, "pinned": pin}

    def edit_note(
        self,
        target: str,
        *,
        body: str | None = None,
        append: str | None = None,
        tags: list[str] | None = None,
        pin: bool | None = None,
    ) -> dict[str, object]:
        """Edit a note: modify frontmatter/body, atomic write, reindex.

        Raises ValueError if note not found, not a memory note, or no changes specified.
        """
        import re as _re

        from kb.crud import find_document_by_target
        from kb.db import normalize_path

        conn = self._get_conn()
        doc = find_document_by_target(conn, target)
        if doc is None:
            raise ValueError(f"Note not found: {target}")
        if doc["doc_type"] not in ("memory_note", "memory_doc"):
            raise ValueError(f"Not a memory note (doc_type={doc['doc_type']})")
        if body is not None and append is not None:
            raise ValueError("Cannot specify both body and append")
        if body is None and append is None and tags is None and pin is None:
            raise ValueError("No edit options. Provide body, append, tags, or pin.")

        file_path = self._project_root / doc["path"]
        if not file_path.exists():
            raise ValueError(f"Note file not found on disk: {doc['path']}")

        content = file_path.read_text(encoding="utf-8")
        fm_match = _re.match(r"^---\s*\n(.*?)\n---\s*\n", content, _re.DOTALL)
        if fm_match:
            fm_block = fm_match.group(1)
            note_body = content[fm_match.end() :]
        else:
            fm_block = ""
            note_body = content

        # Apply body/append
        if body is not None:
            note_body = body + "\n"
        elif append is not None:
            note_body = note_body.rstrip("\n") + append + "\n"

        # Apply tags
        if tags is not None:
            tags_line = f"tags: [{', '.join(tags)}]"
            if _re.search(r"^tags:\s", fm_block, _re.MULTILINE):
                fm_block = _re.sub(
                    r"^tags:\s.*$", tags_line, fm_block, count=1, flags=_re.MULTILINE
                )
            elif fm_block:
                fm_block = fm_block.rstrip("\n") + f"\n{tags_line}"
            else:
                fm_block = tags_line

        # Apply pin
        if pin is not None:
            pinned_line = f"pinned: {'true' if pin else 'false'}"
            if _re.search(r"^pinned:\s", fm_block, _re.MULTILINE):
                fm_block = _re.sub(
                    r"^pinned:\s.*$", pinned_line, fm_block, count=1, flags=_re.MULTILINE
                )
            elif fm_block:
                fm_block = fm_block.rstrip("\n") + f"\n{pinned_line}"
            else:
                fm_block = pinned_line

        new_content = f"---\n{fm_block}\n---\n{note_body}" if fm_block else note_body
        self._atomic_write(file_path, new_content)

        rel_path = normalize_path(str(file_path.relative_to(self._project_root)))

        # Re-index
        from kb.indexer import index_all

        index_all(self._db, None, self._project_root, memory_only=True, skip_seed=True)

        if pin is not None:
            conn.execute(
                "UPDATE documents SET pinned = ? WHERE path = ?",
                (1 if pin else 0, rel_path),
            )
            conn.commit()

        return {
            "status": "ok",
            "path": rel_path,
            "title": doc["title"],
            "pinned": bool(pin) if pin is not None else bool(doc["pinned"]),
        }

    # ------------------------------------------------------------------
    # Fact listing
    # ------------------------------------------------------------------

    def list_facts(
        self, since_days: int | None = None, entity: str | None = None
    ) -> list[dict[str, object]]:
        """List recorded facts, optionally filtered by recency or entity.

        Returns a list of dicts with id, entity_name, fact_text, fact_date, created_at.
        Raises ValueError if entity is specified but not found.
        """
        from kb.config import find_entity

        conn = self._get_conn()
        sql = """
            SELECT f.seq, f.fact_text, f.fact_date, f.created_at, e.name as entity_name
            FROM facts f
            LEFT JOIN entities e ON f.entity_id = e.id
        """
        wheres: list[str] = []
        params: list[object] = []
        if since_days is not None:
            wheres.append("f.created_at >= datetime('now', ?)")
            params.append(f"-{since_days} days")
        if entity is not None:
            entity_row = find_entity(conn, entity)
            if entity_row is None:
                raise ValueError(f"Entity not found: {entity}")
            wheres.append("f.entity_id = ?")
            params.append(entity_row["id"])
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY f.created_at DESC"
        rows = conn.execute(sql, params).fetchall()

        return [
            {
                "seq": r["seq"],
                "entity_name": r["entity_name"],
                "fact_text": r["fact_text"],
                "fact_date": r["fact_date"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Index stats
    # ------------------------------------------------------------------

    def get_index_stats(self) -> dict[str, object]:
        """Return database health stats: counts, date range, doc types."""
        conn = self._get_conn()
        total_docs = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()["c"]
        total_entities = conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"]
        total_facts = conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()["c"]
        pinned_docs = conn.execute(
            "SELECT COUNT(*) as c FROM documents WHERE pinned = 1"
        ).fetchone()["c"]
        chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        last_indexed = conn.execute("SELECT MAX(indexed_at) as ts FROM documents").fetchone()["ts"]

        date_range = conn.execute(
            "SELECT MIN(doc_date) as earliest, MAX(doc_date) as latest "
            "FROM documents WHERE doc_date IS NOT NULL"
        ).fetchone()

        type_rows = conn.execute(
            "SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type ORDER BY cnt DESC"
        ).fetchall()
        doc_counts = {r["doc_type"]: r["cnt"] for r in type_rows}

        return {
            "documents": total_docs,
            "documents_by_type": doc_counts,
            "entities": total_entities,
            "facts": total_facts,
            "pinned": pinned_docs,
            "chunks": chunk_count,
            "last_indexed": last_indexed,
            "date_range": {
                "earliest": date_range["earliest"] if date_range else None,
                "latest": date_range["latest"] if date_range else None,
            },
        }

    # ------------------------------------------------------------------
    # Glossary
    # ------------------------------------------------------------------

    def list_glossary_terms(self) -> list[GlossaryEntry]:
        """Return all glossary terms from memory/glossary.md."""
        from kb.glossary import list_terms

        return list_terms(self._project_root)

    def edit_glossary_term(self, term: str, expansion: str) -> dict[str, object]:
        """Update an existing glossary term's expansion. Raises ValueError if not found."""
        from kb.glossary import edit_term

        result = edit_term(self._project_root, term, expansion)
        self._reindex_memory_file("memory/glossary.md")
        return result

    # ------------------------------------------------------------------
    # Corrections
    # ------------------------------------------------------------------

    def correct_term(
        self,
        term: str,
        replacement: str | None = None,
        *,
        apply: bool = False,
        scope: str | None = None,
        file_type: str | None = None,
        word_boundary: bool = False,
        ignore_case: bool = False,
    ) -> dict[str, object]:
        """Find (and optionally replace) a term across memory files.

        Three modes:
        - scan (replacement=None): find all occurrences
        - dry_run (replacement set, apply=False): preview what would change
        - apply (replacement set, apply=True): execute replacements

        Returns dict with 'results' and 'meta' keys.
        """
        from kb.correct import apply_corrections, enrich_matches, scan

        memory_root = self._project_root / "memory"
        if not memory_root.is_dir():
            raise ValueError(f"Memory directory not found at {memory_root}")

        matches = scan(
            memory_root,
            term,
            ignore_case=ignore_case,
            word_boundary=word_boundary,
            scope=scope,
            file_type=file_type,
        )

        if not matches:
            return {"results": [], "meta": {"term": term, "total": 0, "action": "scan"}}

        if replacement is None:
            enriched = enrich_matches(memory_root, matches)
            total = sum(m.count for m in matches)
            return {
                "results": enriched,
                "meta": {
                    "term": term,
                    "total_occurrences": total,
                    "files": len(matches),
                    "action": "scan",
                },
            }

        if not apply:
            enriched = enrich_matches(memory_root, matches)
            total = sum(m.count for m in matches)
            return {
                "results": enriched,
                "meta": {
                    "term": term,
                    "replacement": replacement,
                    "total_occurrences": total,
                    "files": len(matches),
                    "action": "dry_run",
                },
            }

        # Apply corrections
        from kb.config import get_data_dir

        log_path = get_data_dir() / "corrections.log"
        result = apply_corrections(
            memory_root,
            matches,
            replacement,
            ignore_case=ignore_case,
            word_boundary=word_boundary,
            log_path=log_path,
        )
        return {
            "results": [{"path": p} for p in result.changed_paths],
            "meta": {
                "term": term,
                "replacement": replacement,
                "action": "applied",
                "files_changed": result.files_changed,
                "occurrences_replaced": result.occurrences_replaced,
            },
        }

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
        fts_weight: float = 1.0,
        vector_weight: float = 1.0,
        dedupe: bool = False,
        snippet_chars: int = 200,
        full_chunks: bool = False,
        merge_chunks: bool = False,
        highlight: bool = False,
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
            fts_weight=fts_weight,
            vector_weight=vector_weight,
            dedupe=dedupe,
            snippet_chars=snippet_chars,
            full_chunks=full_chunks,
            merge_chunks=merge_chunks,
            highlight=highlight,
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
