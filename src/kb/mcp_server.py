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
    doc_type: str | None = None,
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
            doc_type=doc_type,
        )
        return json.dumps(results.model_dump(), default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_search error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def _resolve_me(name: str) -> str:
    """Resolve 'me' to the configured user name."""
    if name.lower() != "me":
        return name
    from kb.user_config import find_config, load_config

    config_path = find_config()
    if config_path:
        cfg = load_config(config_path)
        if cfg.user.name:
            return cfg.user.name
    return name  # fall through — let find_entity handle "me" as a literal name


def handle_kb_person_find(db: Database, name: str) -> str:
    """Look up a person profile. Returns JSON string."""
    try:
        name = _resolve_me(name)
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
            "updated_at": entity_row["updated_at"],
            "last_mentioned_at": entity_row["last_mentioned_at"],
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
    db: Database,
    name: str,
    from_date: str | None = None,
    to_date: str | None = None,
    doc_type: str | None = None,
    limit: int | None = None,
) -> str:
    """Chronological docs mentioning a person. Returns JSON string."""
    try:
        name = _resolve_me(name)
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
        if doc_type:
            sql += " AND d.doc_type = ?"
            params.append(doc_type)

        sql += " ORDER BY d.doc_date ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
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
    db: Database,
    project_root: Path,
    topic: str | None = None,
    fmt: str = "compact",
    mention_threshold: int = 0,
) -> str:
    """Compressed entity index. Returns JSON string."""
    try:
        from kb.context import generate_context

        result = generate_context(
            db, project_root, topic=topic, fmt=fmt, mention_threshold=mention_threshold
        )
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
  kb_entity_stale()                   # entities not updated/mentioned in 30+ days
  kb_entity_stale(days=60, entity_type="person")  # custom threshold + filter
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


def handle_kb_entity_stale(
    db: Database,
    days: int = 30,
    entity_type: str | None = None,
) -> str:
    """Return entities not updated or mentioned within *days*. Returns JSON string."""
    try:
        from datetime import date, timedelta

        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = db.get_sqlite_conn()

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
        return json.dumps(
            {"results": results, "meta": {"count": len(results), "threshold_days": days}},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_entity_stale error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_project_find(db: Database, name: str) -> str:
    """Look up a project profile. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        entity_row = find_entity(conn, name)
        if entity_row is None:
            return json.dumps({"error": f"Project not found: {name}"})
        if entity_row["entity_type"] != "project":
            return json.dumps(
                {"error": f"'{entity_row['name']}' is a {entity_row['entity_type']}, not a project"}
            )
        return _build_entity_result(conn, entity_row)
    except Exception as e:
        print(f"kb_project_find error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_project_list(db: Database, limit: int = 50, offset: int = 0) -> str:
    """List all projects. Returns JSON string."""
    try:
        return _entity_list(db, "project", limit=limit, offset=offset)
    except Exception as e:
        print(f"kb_project_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_person_list(db: Database, limit: int = 50, offset: int = 0) -> str:
    """List all people. Returns JSON string."""
    try:
        return _entity_list(db, "person", limit=limit, offset=offset)
    except Exception as e:
        print(f"kb_person_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def _entity_list(db: Database, entity_type: str, limit: int = 50, offset: int = 0) -> str:
    """Shared: list entities of a given type with pagination."""
    conn = db.get_sqlite_conn()

    # Get total count first
    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM entities WHERE entity_type = ?",
        (entity_type,),
    ).fetchone()["cnt"]

    rows = conn.execute(
        "SELECT id, name, entity_type, aliases, metadata, source_path FROM entities "
        "WHERE entity_type = ? ORDER BY name LIMIT ? OFFSET ?",
        (entity_type, limit, offset),
    ).fetchall()
    entities = [
        {
            "id": r["id"],
            "name": r["name"],
            "entity_type": r["entity_type"],
            "aliases": json.loads(r["aliases"]) if r["aliases"] else [],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
        }
        for r in rows
    ]
    return json.dumps(
        {"results": entities, "meta": {"total": total, "limit": limit, "offset": offset}},
        default=str,
        ensure_ascii=False,
    )


def _build_entity_result(conn: Any, entity_row: Any) -> str:
    """Build a compact entity result with facts, doc count, and breadcrumbs. Returns JSON string."""
    import shlex
    from datetime import date, timedelta

    entity_id = entity_row["id"]
    entity_name = str(entity_row["name"])
    entity_type = str(entity_row["entity_type"])
    source_path = str(entity_row["source_path"]) if entity_row["source_path"] else None

    fact_rows = conn.execute(
        "SELECT fact_text, fact_date FROM facts WHERE entity_id = ? "
        "ORDER BY fact_date DESC, id DESC",
        (entity_id,),
    ).fetchall()
    facts = [{"text": str(r["fact_text"]), "date": r["fact_date"]} for r in fact_rows]

    doc_count_row = conn.execute(
        "SELECT COUNT(DISTINCT document_id) as cnt FROM entity_mentions WHERE entity_id = ?",
        (entity_id,),
    ).fetchone()
    document_count = int(doc_count_row["cnt"])

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


def handle_kb_note_list(
    db: Database,
    tag: str | None = None,
    pinned_only: bool = False,
    limit: int = 25,
) -> str:
    """List notes with optional tag/pin filters. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        rows = conn.execute(
            "SELECT id, path, title, doc_date, tags, pinned "
            "FROM documents "
            "WHERE doc_type IN ('memory_note', 'memory_doc') "
            "ORDER BY doc_date DESC, id DESC",
        ).fetchall()

        required_tags: list[str] = []
        if tag:
            required_tags = [t.strip().lower() for t in tag.split(",") if t.strip()]

        results: list[dict[str, Any]] = []
        for r in rows:
            doc_tags: list[str] = json.loads(r["tags"]) if r["tags"] else []
            is_pinned = bool(r["pinned"])

            if required_tags:
                lower_tags = [t.lower() for t in doc_tags]
                if not all(rt in lower_tags for rt in required_tags):
                    continue
            if pinned_only and not is_pinned:
                continue

            results.append(
                {
                    "path": r["path"],
                    "title": r["title"],
                    "date": r["doc_date"],
                    "tags": doc_tags,
                    "pinned": is_pinned,
                }
            )
            if len(results) >= limit:
                break

        return json.dumps(
            {"results": results, "meta": {"total": len(results), "limit": limit}},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_note_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_note_edit(
    db: Database,
    project_root: Path,
    target: str,
    body: str | None = None,
    append: str | None = None,
    tags: str | None = None,
    pin: bool | None = None,
) -> str:
    """Edit a note's body, tags, or pin status. Returns JSON string."""
    try:
        import os
        import re as _re

        from kb.db import normalize_path

        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Note not found: {target}"})
        if doc["doc_type"] not in ("memory_note", "memory_doc"):
            return json.dumps(
                {"error": f"Not a memory note (doc_type={doc['doc_type']})"}
            )
        if body is not None and append is not None:
            return json.dumps({"error": "Cannot specify both body and append"})
        if body is None and append is None and tags is None and pin is None:
            return json.dumps(
                {"error": "No edit options. Provide body, append, tags, or pin."}
            )

        from pathlib import Path as P

        file_path = P(doc["path"])
        if not file_path.is_absolute():
            file_path = project_root / doc["path"]
        if not file_path.exists():
            return json.dumps({"error": f"Note file not found on disk: {doc['path']}"})

        content = file_path.read_text(encoding="utf-8")
        fm_match = _re.match(r"^---\s*\n(.*?)\n---\s*\n", content, _re.DOTALL)
        if fm_match:
            fm_block = fm_match.group(1)
            note_body = content[fm_match.end():]
        else:
            fm_block = ""
            note_body = content

        if body is not None:
            note_body = body + "\n"
        elif append is not None:
            note_body = note_body.rstrip("\n") + append + "\n"

        if tags is not None:
            new_tags = [t.strip() for t in tags.split(",") if t.strip()]
            tags_line = f"tags: [{', '.join(new_tags)}]"
            if _re.search(r"^tags:\s", fm_block, _re.MULTILINE):
                fm_block = _re.sub(r"^tags:\s.*$", tags_line, fm_block, count=1, flags=_re.MULTILINE)
            elif fm_block:
                fm_block = fm_block.rstrip("\n") + f"\n{tags_line}"
            else:
                fm_block = tags_line

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
        tmp_path = file_path.with_suffix(".md.tmp")
        tmp_path.write_text(new_content, encoding="utf-8")
        os.replace(str(tmp_path), str(file_path))

        rel_path = normalize_path(str(file_path.relative_to(project_root)))

        from kb.indexer import index_all

        index_all(db, None, project_root, memory_only=True, skip_seed=True)

        if pin is not None:
            conn.execute(
                "UPDATE documents SET pinned = ? WHERE path = ?",
                (1 if pin else 0, rel_path),
            )
            conn.commit()

        return json.dumps(
            {
                "status": "ok",
                "path": rel_path,
                "title": doc["title"],
                "pinned": bool(pin) if pin is not None else bool(doc.get("pinned")),
            },
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_note_edit error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_note_delete(db: Database, project_root: Path, target: str) -> str:
    """Delete a memory note (file + index). Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Note not found: {target}"})
        if doc["doc_type"] not in ("memory_note", "memory_doc"):
            return json.dumps(
                {"error": f"Not a memory note (doc_type={doc['doc_type']})"}
            )

        doc_id = doc["id"]
        rel_path = doc["path"]

        file_path = project_root / rel_path
        if file_path.exists():
            file_path.unlink()

        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM entity_mentions WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

        return json.dumps(
            {"status": "ok", "path": rel_path, "title": doc["title"]},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_note_delete error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_memory_list(db: Database, since_days: int | None = None) -> str:
    """List recorded facts. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        sql = """
            SELECT f.id, f.fact_text, f.fact_date, f.created_at, e.name as entity_name
            FROM facts f
            LEFT JOIN entities e ON f.entity_id = e.id
        """
        params: list[Any] = []
        if since_days is not None:
            sql += " WHERE f.created_at >= datetime('now', ?)"
            params.append(f"-{since_days} days")
        sql += " ORDER BY f.created_at DESC"
        rows = conn.execute(sql, params).fetchall()

        facts = [
            {
                "id": r["id"],
                "entity_name": r["entity_name"],
                "fact_text": r["fact_text"],
                "fact_date": r["fact_date"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return json.dumps(
            {"results": facts, "meta": {"total": len(facts)}},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_memory_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_memory_delete_fact(project_root: Path, fact_id: int) -> str:
    """Delete a fact by ID. Returns JSON string."""
    try:
        from kb.api import KnowledgeBase
        from kb.config import get_data_dir

        kb = KnowledgeBase(project_root=project_root, data_dir=get_data_dir())
        try:
            result = kb.delete_fact(fact_id)
            return json.dumps(result, default=str, ensure_ascii=False)
        finally:
            kb.close()
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_memory_delete_fact error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_memory_edit_fact(
    project_root: Path,
    fact_id: int,
    text: str | None = None,
    date: str | None = None,
) -> str:
    """Edit a fact's text or date. Returns JSON string."""
    try:
        if text is None and date is None:
            return json.dumps({"error": "Specify text and/or date to edit."})

        from kb.api import KnowledgeBase
        from kb.config import get_data_dir

        kb = KnowledgeBase(project_root=project_root, data_dir=get_data_dir())
        try:
            result = kb.edit_fact(fact_id, text=text, date=date)
            return json.dumps(result, default=str, ensure_ascii=False)
        finally:
            kb.close()
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_memory_edit_fact error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_glossary_list(project_root: Path) -> str:
    """List all glossary terms. Returns JSON string."""
    try:
        from kb.glossary import list_terms

        terms = list_terms(project_root)
        return json.dumps(
            {
                "results": [t.model_dump() for t in terms],
                "meta": {"total": len(terms)},
            },
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_glossary_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_glossary_add(
    project_root: Path, term: str, expansion: str, section: str = "Acronyms"
) -> str:
    """Add a glossary term. Returns JSON string."""
    try:
        from kb.glossary import add_term

        result = add_term(project_root, term, expansion, section=section)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_glossary_add error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_glossary_edit(project_root: Path, term: str, expansion: str) -> str:
    """Edit a glossary term. Returns JSON string."""
    try:
        from kb.glossary import edit_term

        result = edit_term(project_root, term, expansion)
        return json.dumps(result, default=str, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_glossary_edit error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_granola_view(
    calendar_uid: str, mode: str | None = None
) -> str:
    """View meeting notes/transcript/summary by calendar UID. Returns JSON string."""
    try:
        from kb.sync.granola import (
            GranolaClient,
            extract_panel_markdown,
            transcript_to_markdown,
        )

        client = GranolaClient()
        doc = client.find_document(calendar_uid=calendar_uid)
        if not doc:
            return json.dumps({"error": f"No Granola document found for UID '{calendar_uid}'"})

        notes_md = doc.get("notes_markdown") or ""
        summary_md = extract_panel_markdown(doc)

        transcript_md = ""
        if mode in ("transcript", "all"):
            segments = client.get_transcript(doc["id"])
            transcript_md = transcript_to_markdown(segments)

        gcal = doc.get("google_calendar_event") or {}
        result: dict[str, Any] = {
            "title": doc.get("title") or "Untitled",
            "date": (doc.get("created_at") or "")[:10],
            "granola_id": doc.get("id", ""),
            "calendar_uid": gcal.get("iCalUID") or gcal.get("id") or calendar_uid,
            "has_notes": bool(notes_md.strip()),
            "has_summary": bool(summary_md.strip()),
            "has_transcript": bool(transcript_md.strip()),
        }

        if mode == "summary":
            result["content"] = summary_md or "(no AI summary)"
        elif mode == "transcript":
            result["content"] = transcript_md or "(no transcript)"
        elif mode == "all":
            result["notes"] = notes_md or "(no notes)"
            result["summary"] = summary_md or "(no AI summary)"
            result["transcript"] = transcript_md or "(no transcript)"
        else:
            result["content"] = notes_md or "(no notes)"

        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_granola_view error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_granola_edit(
    calendar_uid: str,
    body: str | None = None,
    append: str | None = None,
) -> str:
    """Edit meeting notes in Granola (API only). Returns JSON string."""
    try:
        from kb.sync.granola import GranolaClient

        if body is None and append is None:
            return json.dumps({"error": "Provide body or append."})
        if body is not None and append is not None:
            return json.dumps({"error": "Provide only one of body or append."})

        client = GranolaClient()
        doc = client.find_document(calendar_uid=calendar_uid)
        if not doc:
            return json.dumps({"error": f"No Granola document found for UID '{calendar_uid}'"})

        if append is not None:
            existing_md = doc.get("notes_markdown") or ""
            if existing_md.strip():
                markdown = existing_md.rstrip() + "\n\n" + append
            else:
                markdown = append
        else:
            assert body is not None
            markdown = body

        client.update_document_notes(doc["id"], markdown)

        return json.dumps(
            {"status": "ok", "calendar_uid": calendar_uid, "action": "append" if append else "replace"},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_granola_edit error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_list(
    db: Database,
    doc_type: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 25,
    since_hours: int | None = None,
) -> str:
    """Browse documents by date/type. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        sql = "SELECT id, path, title, doc_date, doc_type, content_hash, chunk_count FROM documents WHERE 1=1"
        params: list[Any] = []

        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)
        if from_date:
            sql += " AND doc_date >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND doc_date <= ?"
            params.append(to_date)
        if since_hours is not None:
            sql += " AND indexed_at >= datetime('now', ?)"
            params.append(f"-{since_hours} hours")

        sql += " ORDER BY doc_date DESC NULLS LAST LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        docs = [
            {
                "id": r["id"],
                "path": r["path"],
                "title": r["title"],
                "date": r["doc_date"],
                "doc_type": r["doc_type"],
                "content_hash": r["content_hash"][:6] if r["content_hash"] else None,
                "chunks": r["chunk_count"],
            }
            for r in rows
        ]
        return json.dumps(
            {"results": docs, "meta": {"total": len(docs), "limit": limit}},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_index_status(db: Database) -> str:
    """Database health: doc counts, entity counts, freshness. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()

        type_rows = conn.execute(
            "SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type"
        ).fetchall()
        doc_counts = {r["doc_type"]: r["count"] for r in type_rows}
        total_docs = sum(doc_counts.values())

        chunk_count = conn.execute("SELECT COUNT(*) as count FROM chunks").fetchone()["count"]
        entity_count = conn.execute("SELECT COUNT(*) as count FROM entities").fetchone()["count"]
        fact_count = conn.execute("SELECT COUNT(*) as count FROM facts").fetchone()["count"]
        last_indexed = conn.execute("SELECT MAX(indexed_at) as ts FROM documents").fetchone()["ts"]

        date_range = conn.execute(
            "SELECT MIN(doc_date) as earliest, MAX(doc_date) as latest "
            "FROM documents WHERE doc_date IS NOT NULL"
        ).fetchone()

        status = {
            "documents": total_docs,
            "documents_by_type": doc_counts,
            "chunks": chunk_count,
            "entities": entity_count,
            "facts": fact_count,
            "last_indexed": last_indexed,
            "date_range": {
                "earliest": date_range["earliest"],
                "latest": date_range["latest"],
            },
        }
        return json.dumps(status, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_index_status error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_correct(
    project_root: Path,
    term: str,
    replacement: str | None = None,
    apply: bool = False,
    scope: str | None = None,
    file_type: str | None = None,
    word_boundary: bool = False,
    ignore_case: bool = False,
) -> str:
    """Find (and optionally replace) a term across memory files. Returns JSON string."""
    try:
        from kb.correct import apply_corrections, enrich_matches, scan

        memory_root = project_root / "memory"
        if not memory_root.is_dir():
            return json.dumps({"error": f"Memory directory not found at {memory_root}"})

        matches = scan(
            memory_root,
            term,
            ignore_case=ignore_case,
            word_boundary=word_boundary,
            scope=scope,
            file_type=file_type,
        )

        if not matches:
            return json.dumps(
                {"results": [], "meta": {"term": term, "total": 0, "action": "scan"}}
            )

        if replacement is None:
            enriched = enrich_matches(memory_root, matches)
            total = sum(m.count for m in matches)
            return json.dumps(
                {
                    "results": enriched,
                    "meta": {
                        "term": term,
                        "total_occurrences": total,
                        "files": len(matches),
                        "action": "scan",
                    },
                },
                default=str,
                ensure_ascii=False,
            )

        if not apply:
            enriched = enrich_matches(memory_root, matches)
            total = sum(m.count for m in matches)
            return json.dumps(
                {
                    "results": enriched,
                    "meta": {
                        "term": term,
                        "replacement": replacement,
                        "total_occurrences": total,
                        "files": len(matches),
                        "action": "dry_run",
                    },
                },
                default=str,
                ensure_ascii=False,
            )

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
        return json.dumps(
            {
                "results": [{"path": p} for p in result.changed_paths],
                "meta": {
                    "term": term,
                    "replacement": replacement,
                    "action": "applied",
                    "files_changed": result.files_changed,
                    "occurrences_replaced": result.occurrences_replaced,
                },
            },
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_correct error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


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
    doc_type: str | None = None,
) -> str:
    """Search the knowledge base. Returns JSON with matching documents, ranked by relevance.
    Use fast=True (default) for instant FTS-only search.
    Optionally filter by date range (YYYY-MM-DD).
    Optionally filter by tag (comma-separated for AND, e.g. 'decision,infra').
    Note: tags only work on memory notes, not meeting docs.
    doc_type: filter by document type (e.g. 'notes', 'transcript', 'memory_note').
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
        doc_type=doc_type,
    )


@mcp.tool()
def kb_person_find(name: str) -> str:
    """Look up a person profile — compact output with facts, metadata, and breadcrumbs.
    Supports exact name, alias, or partial match."""
    db = get_db()
    return handle_kb_person_find(db, name)


@mcp.tool()
def kb_person_timeline(
    name: str,
    from_date: str | None = None,
    to_date: str | None = None,
    doc_type: str | None = None,
    limit: int | None = None,
) -> str:
    """Get chronological list of documents mentioning a person.
    Optionally filter by date range (YYYY-MM-DD).
    doc_type: filter by document type (e.g. 'notes', 'transcript', 'debrief').
    limit: max results (default: no limit)."""
    from_date = _validate_date(from_date)
    to_date = _validate_date(to_date)
    db = get_db()
    return handle_kb_person_timeline(
        db, name, from_date=from_date, to_date=to_date, doc_type=doc_type, limit=limit
    )


@mcp.tool()
def kb_view(target: str) -> str:
    """View a specific document by path or content-hash prefix (#abc123).
    Returns document metadata and all chunks."""
    db = get_db()
    return handle_kb_view(db, target)


@mcp.tool()
def kb_context(
    topic: str | None = None, fmt: str = "compact", mention_threshold: int = 0
) -> str:
    """Get compressed entity index — overview of all people, projects, teams, and terms.
    Optionally filter to entities relevant to a specific topic.
    fmt: 'compact' (default, pipe-delimited) or 'human' (markdown with headings).
    mention_threshold: minimum mention count for non-pinned entities (0 = no filter)."""
    if fmt not in ("compact", "human"):
        fmt = "compact"
    db = get_db()
    project_root = find_project_root()
    return handle_kb_context(
        db, project_root, topic=topic, fmt=fmt, mention_threshold=mention_threshold
    )


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


@mcp.tool()
def kb_entity_stale(days: int = 30, entity_type: str | None = None) -> str:
    """List entities not updated or mentioned within a threshold.
    Returns stale entities sorted by stalest first.
    days: threshold in days (default 30).
    entity_type: optional filter ('person', 'project')."""
    days = max(1, days)
    db = get_db()
    return handle_kb_entity_stale(db, days=days, entity_type=entity_type)


@mcp.tool()
def kb_project_find(name: str) -> str:
    """Look up a project profile — compact output with facts, metadata, and breadcrumbs.
    Supports exact name, alias, or partial match."""
    db = get_db()
    return handle_kb_project_find(db, name)


@mcp.tool()
def kb_project_list(limit: int = 50, offset: int = 0) -> str:
    """List all known projects with their metadata.
    limit: max results (default 50).
    offset: skip first N results for pagination."""
    limit = max(1, min(limit, 500))
    db = get_db()
    return handle_kb_project_list(db, limit=limit, offset=offset)


@mcp.tool()
def kb_person_list(limit: int = 50, offset: int = 0) -> str:
    """List all known people with their metadata.
    limit: max results (default 50).
    offset: skip first N results for pagination."""
    limit = max(1, min(limit, 500))
    db = get_db()
    return handle_kb_person_list(db, limit=limit, offset=offset)


@mcp.tool()
def kb_note_list(
    tag: str | None = None, pinned_only: bool = False, limit: int = 25
) -> str:
    """Browse memory notes with optional filtering.
    tag: comma-separated tags (AND filter).
    pinned_only: only return pinned notes.
    limit: max results (default 25)."""
    limit = max(1, min(limit, 100))
    db = get_db()
    return handle_kb_note_list(db, tag=tag, pinned_only=pinned_only, limit=limit)


@mcp.tool()
def kb_note_edit(
    target: str,
    body: str | None = None,
    append: str | None = None,
    tags: str | None = None,
    pin: bool | None = None,
) -> str:
    """Edit a memory note's body, tags, or pin status.
    target: note path, title, or glob.
    body: replace note body entirely.
    append: append to existing body.
    tags: comma-separated tags (replaces existing).
    pin: True to pin, False to unpin."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_note_edit(
        db, project_root, target, body=body, append=append, tags=tags, pin=pin
    )


@mcp.tool()
def kb_note_delete(target: str) -> str:
    """Delete a memory note (file + index entry).
    target: note path, title, or glob."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_note_delete(db, project_root, target)


@mcp.tool()
def kb_memory_list(since_days: int | None = None) -> str:
    """List recorded facts, newest first.
    since_days: optional filter to only show facts from last N days."""
    db = get_db()
    return handle_kb_memory_list(db, since_days=since_days)


@mcp.tool()
def kb_memory_delete_fact(fact_id: int) -> str:
    """Delete a fact by its ID."""
    project_root = find_project_root()
    return handle_kb_memory_delete_fact(project_root, fact_id)


@mcp.tool()
def kb_memory_edit_fact(
    fact_id: int, text: str | None = None, date: str | None = None
) -> str:
    """Edit a fact's text or date.
    fact_id: the fact ID to edit.
    text: new fact text (optional).
    date: new date in YYYY-MM-DD (optional)."""
    project_root = find_project_root()
    return handle_kb_memory_edit_fact(project_root, fact_id, text=text, date=date)


@mcp.tool()
def kb_glossary_list() -> str:
    """List all glossary terms (acronyms and jargon)."""
    project_root = find_project_root()
    return handle_kb_glossary_list(project_root)


@mcp.tool()
def kb_glossary_add(term: str, expansion: str, section: str = "Acronyms") -> str:
    """Add a term to the glossary.
    term: the abbreviation or term.
    expansion: what it stands for.
    section: glossary section heading (default 'Acronyms')."""
    project_root = find_project_root()
    return handle_kb_glossary_add(project_root, term, expansion, section=section)


@mcp.tool()
def kb_glossary_edit(term: str, expansion: str) -> str:
    """Update an existing glossary term's expansion.
    term: the term to edit.
    expansion: the new expansion text."""
    project_root = find_project_root()
    return handle_kb_glossary_edit(project_root, term, expansion)


@mcp.tool()
def kb_granola_view(
    calendar_uid: str, mode: str | None = None
) -> str:
    """View meeting notes, AI summary, or transcript from Granola.
    calendar_uid: Google Calendar event ID or iCalUID.
    mode: 'summary', 'transcript', 'all', or None (notes only, default)."""
    if mode and mode not in ("summary", "transcript", "all"):
        mode = None
    return handle_kb_granola_view(calendar_uid, mode=mode)


@mcp.tool()
def kb_granola_edit(
    calendar_uid: str,
    body: str | None = None,
    append: str | None = None,
) -> str:
    """Edit meeting notes in Granola (writes to Granola API).
    calendar_uid: Google Calendar event ID or iCalUID.
    body: replace notes entirely.
    append: append to existing notes."""
    return handle_kb_granola_edit(calendar_uid, body=body, append=append)


@mcp.tool()
def kb_list(
    doc_type: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 25,
    since_hours: int | None = None,
) -> str:
    """Browse documents by date and type.
    doc_type: filter (e.g. 'notes', 'transcript', 'memory_person').
    from_date/to_date: YYYY-MM-DD date range.
    since_hours: only return documents indexed within the last N hours.
    limit: max results (default 25)."""
    limit = max(1, min(limit, 100))
    from_date = _validate_date(from_date)
    to_date = _validate_date(to_date)
    db = get_db()
    return handle_kb_list(
        db, doc_type=doc_type, from_date=from_date, to_date=to_date, limit=limit, since_hours=since_hours
    )


@mcp.tool()
def kb_index_status() -> str:
    """Get database health: document counts by type, entity/fact counts, date range, last indexed timestamp."""
    db = get_db()
    return handle_kb_index_status(db)


@mcp.tool()
def kb_correct(
    term: str,
    replacement: str | None = None,
    apply: bool = False,
    scope: str | None = None,
    file_type: str | None = None,
    word_boundary: bool = False,
    ignore_case: bool = False,
) -> str:
    """Find (and optionally replace) a term across all memory files.
    Scan mode (no replacement): lists all occurrences with context.
    Dry-run mode (replacement, apply=False): previews changes.
    Apply mode (replacement, apply=True): executes replacements with audit log.
    scope: glob pattern to limit search (e.g. '**/meetings/*').
    word_boundary: only match whole words.
    ignore_case: case-insensitive matching."""
    project_root = find_project_root()
    return handle_kb_correct(
        project_root,
        term,
        replacement=replacement,
        apply=apply,
        scope=scope,
        file_type=file_type,
        word_boundary=word_boundary,
        ignore_case=ignore_case,
    )


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
