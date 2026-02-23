"""MCP server for kbx — exposes knowledge base tools and resources via FastMCP."""

from __future__ import annotations

import json
import re
import sys
import traceback
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from kb.config import find_entity, find_project_root, get_db

if TYPE_CHECKING:
    from pathlib import Path

    from kb.db import Database

# ---------------------------------------------------------------------------
# Handler functions (testable without MCP transport)
# ---------------------------------------------------------------------------


def handle_kb_search(
    db: Database,
    query: str,
    fast: bool = True,
    limit: int = 5,
    from_date: str | None = None,
    to_date: str | None = None,
    tag: str | None = None,
    sort_by: str = "score",
) -> str:
    """Search the knowledge base. Returns JSON string."""
    try:
        from kb.search import search as do_search

        results = do_search(
            db,
            None,
            query,
            limit=limit,
            fast=fast,
            from_date=from_date,
            to_date=to_date,
            tag=tag,
            sort_by=sort_by,
        )
        return json.dumps(results.model_dump(), default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_search error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_person_find(db: Database, name: str) -> str:
    """Look up a person profile. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        entity_row = find_entity(conn, name)
        if entity_row is None:
            return json.dumps({"error": f"Entity not found: {name}"})

        entity_id = entity_row["id"]
        entity_name = str(entity_row["name"])
        entity_type = str(entity_row["entity_type"])
        source_path = str(entity_row["source_path"]) if entity_row["source_path"] else None

        # Facts
        fact_rows = conn.execute(
            "SELECT fact_text, fact_date FROM facts WHERE entity_id = ? "
            "ORDER BY fact_date DESC, id DESC",
            (entity_id,),
        ).fetchall()
        facts = [{"text": str(r["fact_text"]), "date": r["fact_date"]} for r in fact_rows]

        # Document count
        doc_count_row = conn.execute(
            "SELECT COUNT(DISTINCT document_id) as cnt FROM entity_mentions WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        document_count = int(doc_count_row["cnt"])

        # Breadcrumbs
        import shlex
        from datetime import date, timedelta

        quoted_name = shlex.quote(entity_name)
        thirty_ago = (date.today() - timedelta(days=30)).isoformat()
        breadcrumbs: dict[str, str] = {}
        if entity_type in ("person", "project"):
            breadcrumbs["timeline"] = f"kbx {entity_type} timeline {quoted_name} --limit 20"
            breadcrumbs["recent"] = f"kbx {entity_type} timeline {quoted_name} --from {thirty_ago}"
        else:
            breadcrumbs["search"] = f"kbx search {quoted_name} --limit 20"
        if source_path:
            breadcrumbs["profile"] = f"kbx view {source_path}"

        result = {
            "id": entity_id,
            "name": entity_name,
            "entity_type": entity_type,
            "aliases": json.loads(entity_row["aliases"]) if entity_row["aliases"] else [],
            "metadata": json.loads(entity_row["metadata"]) if entity_row["metadata"] else {},
            "facts": facts,
            "source_path": source_path,
            "document_count": document_count,
            "breadcrumbs": breadcrumbs,
        }
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_person_find error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_person_timeline(
    db: Database, name: str, from_date: str | None = None, to_date: str | None = None
) -> str:
    """Chronological docs mentioning a person. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        entity_row = find_entity(conn, name)
        if entity_row is None:
            return json.dumps({"error": f"Entity not found: {name}"})

        sql = """SELECT d.id, d.path, d.title, d.doc_date, d.doc_type, em.mention_type
                 FROM documents d
                 JOIN entity_mentions em ON d.id = em.document_id
                 WHERE em.entity_id = ?"""
        params: list[Any] = [entity_row["id"]]

        if from_date:
            sql += " AND d.doc_date >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND d.doc_date <= ?"
            params.append(to_date)

        sql += " ORDER BY d.doc_date ASC"
        docs = conn.execute(sql, params).fetchall()

        result = {
            "name": entity_row["name"],
            "entity_type": entity_row["entity_type"],
            "documents": [
                {
                    "path": d["path"],
                    "title": d["title"],
                    "date": d["doc_date"],
                    "doc_type": d["doc_type"],
                    "mention_type": d["mention_type"],
                }
                for d in docs
            ],
        }
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_person_timeline error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_view(db: Database, target: str) -> str:
    """View a specific document by path or #hash. Returns JSON string."""
    try:
        from kb.db import normalize_path

        conn = db.get_sqlite_conn()
        doc = None

        # Content-hash lookup: #abc123
        if target.startswith("#"):
            hash_prefix = target[1:]
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash LIKE ?",
                (hash_prefix + "%",),
            ).fetchone()
            if row:
                doc = dict(row)

        if doc is None and not target.startswith("#"):
            import unicodedata

            target_nfc = normalize_path(target)
            target_nfd = unicodedata.normalize("NFD", target)
            # Exact path match (try both NFC and NFD for compat with old data)
            row = conn.execute(
                "SELECT * FROM documents WHERE path = ? OR path = ?",
                (target_nfc, target_nfd),
            ).fetchone()
            if row:
                doc = dict(row)

            # Suffix match
            if doc is None:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE path LIKE ? OR path LIKE ?",
                    ("%" + target_nfc, "%" + target_nfd),
                ).fetchall()
                if len(rows) == 1:
                    doc = dict(rows[0])

        if doc is None and not target.startswith("#"):
            # Glob / substring matching
            import fnmatch

            all_paths = [r["path"] for r in conn.execute("SELECT path FROM documents").fetchall()]

            if "*" in target or "?" in target:
                matches = [p for p in all_paths if fnmatch.fnmatch(p, target)]
            else:
                matches = [p for p in all_paths if target in p.rsplit("/", 1)[-1]]

            if len(matches) == 1:
                row = conn.execute(
                    "SELECT * FROM documents WHERE path = ?", (matches[0],)
                ).fetchone()
                if row:
                    doc = dict(row)
            elif len(matches) > 1:
                return json.dumps(
                    {"error": f"Ambiguous path: {len(matches)} matches", "matches": matches}
                )

        if doc is None:
            return json.dumps({"error": f"Document not found: {target}"})

        chunks = conn.execute(
            "SELECT heading, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc["id"],),
        ).fetchall()

        result = {
            "title": doc["title"],
            "path": doc["path"],
            "date": doc["doc_date"],
            "doc_type": doc["doc_type"],
            "content_hash": doc["content_hash"],
            "chunks": [{"heading": c["heading"], "content": c["content"]} for c in chunks],
        }
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_view error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_context(
    db: Database, project_root: Path, topic: str | None = None, fmt: str = "compact"
) -> str:
    """Compressed entity index. Returns JSON string."""
    try:
        from kb.context import generate_context

        result = generate_context(db, project_root, topic=topic, fmt=fmt)
        return json.dumps(result.model_dump(), default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_context error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_memory_add(
    db: Database,
    project_root: Path,
    text: str,
    body: str | None = None,
    tags: str | None = None,
    entity: str | None = None,
    pin: bool = False,
    date: str | None = None,
) -> str:
    """Create a note or record a fact. Returns JSON string."""
    try:
        import os
        from datetime import date as date_cls

        from kb.db import normalize_path

        conn = db.get_sqlite_conn()

        if date is None:
            date = date_cls.today().isoformat()

        is_note = body is not None or tags is not None or pin or entity is None

        if not is_note:
            # FACT PATH
            assert entity is not None
            entity_row = find_entity(conn, entity)
            if entity_row is None:
                return json.dumps({"error": f"Entity not found: {entity}"})

            conn.execute(
                "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (?, ?, ?)",
                (entity_row["id"], text, date),
            )
            conn.commit()
            return json.dumps({"status": "ok", "type": "fact", "entity": entity_row["name"]})

        # NOTE PATH
        tag_list: list[str] = []
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Slugify
        slug = re.sub(r"[^\w\s-]", "", text.lower().strip())
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)[:60].strip("-")

        notes_dir = project_root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        filepath = notes_dir / f"{date}-{slug}.md"
        counter = 2
        while filepath.exists():
            filepath = notes_dir / f"{date}-{slug}-{counter}.md"
            counter += 1

        frontmatter_lines = ["---", f"title: {text}", f"date: {date}"]
        if tag_list:
            frontmatter_lines.append(f"tags: [{', '.join(tag_list)}]")
        frontmatter_lines.extend(["---", ""])
        note_body = body if body else text
        note_content = "\n".join(frontmatter_lines) + note_body + "\n"

        tmp_path = filepath.with_suffix(".md.tmp")
        tmp_path.write_text(note_content, encoding="utf-8")
        os.replace(str(tmp_path), str(filepath))

        rel_path = normalize_path(str(filepath.relative_to(project_root)))

        from kb.indexer import index_all

        index_all(db, None, project_root, memory_only=True, skip_seed=True)

        if pin:
            conn.execute("UPDATE documents SET pinned = 1 WHERE path = ?", (rel_path,))
            conn.commit()

        if entity:
            entity_row = find_entity(conn, entity)
            if entity_row:
                doc_row = conn.execute(
                    "SELECT id FROM documents WHERE path = ?", (rel_path,)
                ).fetchone()
                if doc_row:
                    conn.execute(
                        "INSERT OR IGNORE INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                        (entity_row["id"], doc_row["id"], "tagged"),
                    )
                    conn.commit()

        return json.dumps({"status": "ok", "type": "note", "path": rel_path, "pinned": pin})
    except Exception as e:
        print(f"kb_memory_add error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_pin(db: Database, target: str) -> str:
    """Pin a document to context. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Document not found: {target}"})
        conn.execute("UPDATE documents SET pinned = 1 WHERE id = ?", (doc["id"],))
        conn.commit()
        return json.dumps({"status": "ok", "path": doc["path"], "pinned": True})
    except Exception as e:
        print(f"kb_pin error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_unpin(db: Database, target: str) -> str:
    """Unpin a document from context. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Document not found: {target}"})
        conn.execute("UPDATE documents SET pinned = 0 WHERE id = ?", (doc["id"],))
        conn.commit()
        return json.dumps({"status": "ok", "path": doc["path"], "pinned": False})
    except Exception as e:
        print(f"kb_unpin error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def _find_document_by_target(conn: Any, target: str) -> dict[str, Any] | None:
    """Resolve a document by path, title, glob, or content hash. For MCP handlers."""
    import fnmatch
    import unicodedata

    from kb.db import normalize_path

    # Content hash
    if target.startswith("#"):
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash LIKE ?", (target[1:] + "%",)
        ).fetchone()
        return dict(row) if row else None

    path_nfc = normalize_path(target)
    path_nfd = unicodedata.normalize("NFD", target)

    # Exact path
    row = conn.execute(
        "SELECT * FROM documents WHERE path = ? OR path = ?", (path_nfc, path_nfd)
    ).fetchone()
    if row:
        return dict(row)

    # Suffix
    rows = conn.execute(
        "SELECT * FROM documents WHERE path LIKE ? OR path LIKE ?",
        ("%" + path_nfc, "%" + path_nfd),
    ).fetchall()
    if len(rows) == 1:
        return dict(rows[0])

    # Title
    rows = conn.execute(
        "SELECT * FROM documents WHERE title = ? COLLATE NOCASE", (target,)
    ).fetchall()
    if len(rows) == 1:
        return dict(rows[0])

    # Glob / substring
    all_paths = [r["path"] for r in conn.execute("SELECT path FROM documents").fetchall()]
    if "*" in target or "?" in target:
        matches = [p for p in all_paths if fnmatch.fnmatch(p, target)]
    else:
        matches = [p for p in all_paths if target in p.rsplit("/", 1)[-1]]
    if len(matches) == 1:
        row = conn.execute("SELECT * FROM documents WHERE path = ?", (matches[0],)).fetchone()
        return dict(row) if row else None

    return None


def handle_kb_usage(db: Database) -> str:
    """Get kb usage instructions and index status. Returns plain text."""
    try:
        conn = db.get_sqlite_conn()

        total_docs = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()["c"]
        total_entities = conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"]
        total_facts = conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()["c"]
        pinned_docs = conn.execute(
            "SELECT COUNT(*) as c FROM documents WHERE pinned = 1"
        ).fetchone()["c"]

        date_range = conn.execute(
            "SELECT MIN(doc_date) as earliest, MAX(doc_date) as latest "
            "FROM documents WHERE doc_date IS NOT NULL"
        ).fetchone()
        earliest = date_range["earliest"] or "N/A"
        latest = date_range["latest"] or "N/A"

        return f"""# kb — Agent Playbook

## 1. Quick Start
  kb_context()           # orient: pinned docs + entity index
  kb_search("topic")     # keyword search (~instant)
  kb_view("path")        # read a full document

## 2. Index Status
  {total_docs} docs | {total_entities} entities | {total_facts} facts | {pinned_docs} pinned | dates {earliest} to {latest}

## 3. Taking Notes
  kb_memory_add("title", body="markdown content", tags="t1,t2", pin=True)
  kb_memory_add("Quick note")                  # one-liner, no body
  kb_memory_add("fact", entity="Name")         # fact appended to entity file
  kb_memory_add("title", body="...", entity="Name")  # note linked to entity

## 4. When to Pin
  kb_pin("path or title or glob")     # pin any doc to context
  kb_unpin("path or title or glob")   # remove from context
  kb_memory_add("title", pin=True)    # create + pin in one step

## 5. Browsing Notes
  Use kb_search with tag filter, or CLI: kb note list --json

## 6. Finding Things
  kb_search("query")                           # keyword (FTS, instant)
  kb_search("query", fast=False, limit=10)     # hybrid (semantic + FTS, ~2s)
  kb_search("query", tag="infra")              # filter by tag
  kb_search("query", from_date="2026-01-01", to_date="2026-01-31")  # date range
  kb_search("query", sort_by="date")           # newest first
  kb_view("path or #hash")                     # full document

  Score Interpretation: 0.8+ strong | 0.5-0.8 worth reading | <0.5 noise

## 7. People & Projects
  kb_person_find("Name")              # compact profile (facts, metadata, breadcrumbs)
  kb_person_timeline("Name")          # chronological doc list
  kb_context()                        # full entity overview
  CLI: kb person/project create/edit/delete, kb glossary add/list/delete

## 8. Context & Indexing
  kb_context()                        # compact entity index (for agents)
  kb_context(fmt="human")             # markdown format (for humans)
  kb_context(topic="topic")           # filtered to a topic
  CLI: kb index run --no-embed        # text-only index (fast)
  CLI: kb index run --cpu             # full index with embeddings
"""
    except Exception as e:
        print(f"kb_usage error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return f"Error getting usage: {e}"


# ---------------------------------------------------------------------------
# FastMCP server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("kbx")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str | None) -> str | None:
    """Validate YYYY-MM-DD format. Returns the value or None if invalid."""
    if value is None:
        return None
    if not _DATE_RE.match(value):
        return None
    return value


@mcp.tool()
def kb_search(
    query: str,
    fast: bool = True,
    limit: int = 5,
    from_date: str | None = None,
    to_date: str | None = None,
    tag: str | None = None,
    sort_by: str = "score",
) -> str:
    """Search the knowledge base. Returns JSON with matching documents, ranked by relevance.
    Use fast=True (default) for instant FTS-only search.
    Optionally filter by date range (YYYY-MM-DD).
    Optionally filter by tag (comma-separated for AND, e.g. 'decision,infra').
    sort_by: 'score' (default) or 'date' (newest first)."""
    limit = max(1, min(limit, 100))
    from_date = _validate_date(from_date)
    to_date = _validate_date(to_date)
    if sort_by not in ("score", "date"):
        sort_by = "score"
    db = get_db()
    return handle_kb_search(
        db,
        query,
        fast=fast,
        limit=limit,
        from_date=from_date,
        to_date=to_date,
        tag=tag,
        sort_by=sort_by,
    )


@mcp.tool()
def kb_person_find(name: str) -> str:
    """Look up a person profile — compact output with facts, metadata, and breadcrumbs.
    Supports exact name, alias, or partial match."""
    db = get_db()
    return handle_kb_person_find(db, name)


@mcp.tool()
def kb_person_timeline(name: str, from_date: str | None = None, to_date: str | None = None) -> str:
    """Get chronological list of documents mentioning a person.
    Optionally filter by date range (YYYY-MM-DD)."""
    from_date = _validate_date(from_date)
    to_date = _validate_date(to_date)
    db = get_db()
    return handle_kb_person_timeline(db, name, from_date=from_date, to_date=to_date)


@mcp.tool()
def kb_view(target: str) -> str:
    """View a specific document by path or content-hash prefix (#abc123).
    Returns document metadata and all chunks."""
    db = get_db()
    return handle_kb_view(db, target)


@mcp.tool()
def kb_context(topic: str | None = None, fmt: str = "compact") -> str:
    """Get compressed entity index — overview of all people, projects, teams, and terms.
    Optionally filter to entities relevant to a specific topic.
    fmt: 'compact' (default, pipe-delimited) or 'human' (markdown with headings)."""
    if fmt not in ("compact", "human"):
        fmt = "compact"
    db = get_db()
    project_root = find_project_root()
    return handle_kb_context(db, project_root, topic=topic, fmt=fmt)


@mcp.tool()
def kb_usage() -> str:
    """Get kb usage instructions and current index status (document/entity counts, date range)."""
    db = get_db()
    return handle_kb_usage(db)


@mcp.tool()
def kb_memory_add(
    text: str,
    body: str | None = None,
    tags: str | None = None,
    entity: str | None = None,
    pin: bool = False,
    date: str | None = None,
) -> str:
    """Create a searchable note or record a fact about an entity.
    If entity is given without body/tags/pin, records a fact.
    Otherwise creates a note file in memory/notes/ and indexes it immediately.
    Use pin=True to make the note appear in kb_context output."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_memory_add(
        db, project_root, text, body=body, tags=tags, entity=entity, pin=pin, date=date
    )


@mcp.tool()
def kb_pin(target: str) -> str:
    """Pin a document to context so it appears in kb_context output.
    Accepts path, title, or glob pattern."""
    db = get_db()
    return handle_kb_pin(db, target)


@mcp.tool()
def kb_unpin(target: str) -> str:
    """Unpin a document from context.
    Accepts path, title, or glob pattern."""
    db = get_db()
    return handle_kb_unpin(db, target)


# ---------------------------------------------------------------------------
# MCP resources
# ---------------------------------------------------------------------------


@mcp.resource("kb://context")
def get_context() -> str:
    """Compressed entity index — use at session start for overview of all entities."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_context(db, project_root)


@mcp.resource("kb://person/{name}")
def get_person(name: str) -> str:
    """Person profile — compact output with facts, metadata, and breadcrumbs."""
    db = get_db()
    return handle_kb_person_find(db, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio transport."""
    # Redirect any accidental stdout prints to stderr
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
