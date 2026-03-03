"""Staleness detection and lightweight auto-reindex for memory sources."""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

from kb.chunker import content_hash

if TYPE_CHECKING:
    from pathlib import Path

    from kb.db import Database


def find_stale_sources(db: Database, project_root: Path) -> list[str]:
    """Find memory source files that are newer than the index.

    Compares file content hashes against indexed hashes.
    Returns list of relative paths that need reindexing.
    """
    conn = db.get_sqlite_conn()

    # Get indexed docs with their content hashes
    rows = conn.execute("SELECT path, content_hash FROM documents").fetchall()
    indexed = {r["path"]: r["content_hash"] for r in rows}

    stale: list[str] = []

    # Check memory source files — only files already in the index
    # (new files need full `kb index run` for proper entity/mention setup).
    # NOTE: meeting files (memory/meetings/) are NOT checked here — they are
    # managed by the sync pipeline (kbx sync + kbx index run), not by ad-hoc
    # edits. Prep/debrief files (.prep.md, .debrief.md) also require explicit
    # `kbx index run` after creation. If auto-reindex for meetings is needed
    # later, add a walk of memory/meetings/ here with rglob.
    for subdir in ["memory/people", "memory/projects", "memory/context", "memory/notes"]:
        dir_path = project_root / subdir
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.md"):
            rel = str(f.relative_to(project_root))
            if rel in indexed and content_hash(f.read_text(encoding="utf-8")) != indexed[rel]:
                stale.append(rel)

    # Check glossary
    glossary = project_root / "memory" / "glossary.md"
    if glossary.exists():
        rel = "memory/glossary.md"
        if rel in indexed and content_hash(glossary.read_text(encoding="utf-8")) != indexed[rel]:
            stale.append(rel)

    return stale


def auto_reindex_if_stale(db: Database, project_root: Path) -> int:
    """Check for stale sources and reindex them silently.

    Returns number of stale files found.
    Safe to call from any context — never raises, never produces output.
    """
    try:
        # Skip if DB has no indexed documents (empty/new DB, not a staleness issue)
        conn = db.get_sqlite_conn()
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count == 0:
            return 0

        stale = find_stale_sources(db, project_root)
        if not stale:
            return 0

        # Re-index memory sources only, skip entity seeding (destructive upsert)
        # and suppress stderr (progress output corrupts CLI JSON output)
        from kb.indexer import index_all

        with contextlib.redirect_stderr(io.StringIO()):
            index_all(db, None, project_root, memory_only=True, skip_seed=True)

        return len(stale)
    except Exception:
        return 0
