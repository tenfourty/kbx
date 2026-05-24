"""Access-tracking helpers (#67) — increment counts on user-initiated lookups.

These helpers are called from the API layer when a user (or agent) deliberately
asks for a document or entity (view, person find, project find, timeline).
Speculative lookups (search) deliberately do NOT call these — search hits are
candidates, not actual accesses.

The increment is best-effort: failures are swallowed so transient DB locks
never break a read-path call. Hotness is a ranking signal, not a correctness
invariant.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now_iso() -> str:
    """Current UTC time as ISO-8601 with seconds precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def touch_document(conn: sqlite3.Connection, document_id: int) -> None:
    """Increment ``documents.access_count`` and bump ``last_accessed_at``."""
    try:
        conn.execute(
            "UPDATE documents SET access_count = access_count + 1, "
            "last_accessed_at = ? WHERE id = ?",
            (_now_iso(), document_id),
        )
        conn.commit()
    except sqlite3.Error:
        # Best-effort — hotness is a ranking signal, not correctness.
        pass


def touch_entity(conn: sqlite3.Connection, entity_id: int) -> None:
    """Increment ``entities.access_count`` and bump ``last_accessed_at``."""
    try:
        conn.execute(
            "UPDATE entities SET access_count = access_count + 1, "
            "last_accessed_at = ? WHERE id = ?",
            (_now_iso(), entity_id),
        )
        conn.commit()
    except sqlite3.Error:
        pass


def reset_document_access(conn: sqlite3.Connection, document_id: int | None = None) -> int:
    """Clear access tracking for a document, or all documents when ``None``.

    Returns the number of rows affected.
    """
    if document_id is None:
        cursor = conn.execute(
            "UPDATE documents SET access_count = 0, last_accessed_at = NULL"
        )
    else:
        cursor = conn.execute(
            "UPDATE documents SET access_count = 0, last_accessed_at = NULL WHERE id = ?",
            (document_id,),
        )
    conn.commit()
    return cursor.rowcount


def reset_entity_access(conn: sqlite3.Connection, entity_id: int | None = None) -> int:
    """Clear access tracking for an entity, or all entities when ``None``.

    Returns the number of rows affected.
    """
    if entity_id is None:
        cursor = conn.execute(
            "UPDATE entities SET access_count = 0, last_accessed_at = NULL"
        )
    else:
        cursor = conn.execute(
            "UPDATE entities SET access_count = 0, last_accessed_at = NULL WHERE id = ?",
            (entity_id,),
        )
    conn.commit()
    return cursor.rowcount
