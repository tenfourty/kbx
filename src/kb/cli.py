"""CLI entry point for kbx — local knowledge base with hybrid search."""

from __future__ import annotations

import contextlib
import functools
import json
import shlex
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from kb.config import find_entities as _find_entities
from kb.config import find_entity as _find_entity
from kb.config import find_project_root as _find_project_root
from kb.config import get_data_dir as _get_data_dir
from kb.config import get_db as _get_db
from kb.output import render

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Shared output options decorator
# ---------------------------------------------------------------------------


def output_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Shared decorator that adds --format, --json, --fields, --jq to a command."""

    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["table", "json", "jsonl", "csv"]),
        default="table",
        help="Output format.",
    )
    @click.option("--json", "json_flag", is_flag=True, help="Shortcut for --format json.")
    @click.option("--fields", "fields_str", default=None, help="Comma-separated field list.")
    @click.option("--jq", "jq_expr", default=None, help="jq expression for filtering.")
    @functools.wraps(f)
    def wrapper(
        *args: Any,
        fmt: str,
        json_flag: bool,
        fields_str: str | None,
        jq_expr: str | None,
        **kwargs: Any,
    ) -> Any:
        if json_flag:
            fmt = "json"
        fields = [s.strip() for s in fields_str.split(",")] if fields_str else None
        kwargs["fmt"] = fmt
        kwargs["fields"] = fields
        kwargs["jq_expr"] = jq_expr
        return f(*args, **kwargs)

    return wrapper


def kb_output(
    data: Any,
    fmt: str = "table",
    fields: list[str] | None = None,
    jq_expr: str | None = None,
    columns: list[str] | None = None,
) -> None:
    """Render data to stdout."""
    text = render(data, fmt=fmt, fields=fields, jq_expr=jq_expr, columns=columns)
    click.echo(text)


def _entity_not_found(
    conn: sqlite3.Connection, name: str, fmt: str, entity_type: str | None = None
) -> None:
    """Output entity-not-found error. Structured JSON when fmt is json/jsonl."""
    # Find close matches for suggestions
    if entity_type:
        all_entities = conn.execute(
            "SELECT name FROM entities WHERE entity_type = ?", (entity_type,)
        ).fetchall()
    else:
        all_entities = conn.execute("SELECT name FROM entities").fetchall()
    all_names = [e["name"] for e in all_entities]
    suggestions = _fuzzy_suggest_entity(name, all_names)

    cmd = f"kb {entity_type} find" if entity_type else "kb person find"
    if fmt in ("json", "jsonl"):
        error: dict[str, Any] = {"error": f"Entity not found: {name}"}
        if suggestions:
            error["suggestion"] = f"Did you mean '{suggestions[0]}'?"
            error["available_actions"] = [f'{cmd} "{s}"' for s in suggestions]
        click.echo(json.dumps(error, ensure_ascii=False))
    else:
        msg = f"Entity not found: {name}"
        if suggestions:
            msg += f"\nDid you mean: {suggestions[0]}?"
        click.echo(msg, err=True)


def _fuzzy_suggest_entity(query: str, names: list[str], max_suggestions: int = 3) -> list[str]:
    """Find close entity name matches using substring and edit-distance similarity."""
    from difflib import SequenceMatcher

    query_lower = query.lower()
    scored: list[tuple[float, str]] = []
    for name in names:
        name_lower = name.lower()
        # Substring match is strongest signal
        if query_lower in name_lower or name_lower in query_lower:
            scored.append((1.0, name))
            continue
        # Also check against first name / last name parts
        parts = name_lower.split()
        part_ratios = [SequenceMatcher(None, query_lower, p).ratio() for p in parts]
        best_part = max(part_ratios) if part_ratios else 0
        # Overall similarity
        full_ratio = SequenceMatcher(None, query_lower, name_lower).ratio()
        # Take whichever is higher
        score = max(best_part, full_ratio)
        if score >= 0.45:
            scored.append((score, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:max_suggestions]]


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """kbx — local knowledge base with hybrid search over markdown files."""
    pass


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("--fast", is_flag=True, help="FTS-only search (~instant).")
@click.option("--deep", is_flag=True, help="Hybrid + reranking (future).")
@click.option("--limit", default=10, type=int, help="Max results.")
@click.option("--recency", default=0.15, type=float, help="Recency weight (0-1).")
@click.option("--type", "doc_type", default=None, help="Filter by doc_type.")
@click.option("--from", "from_date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD).")
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["score", "date"]),
    default="score",
    help="Sort results by score or date.",
)
@click.option("--tag", "tag_filter", default=None, help="Filter by tag (comma-separated for AND).")
@output_options
def search(
    query: str,
    fast: bool,
    deep: bool,
    limit: int,
    recency: float,
    doc_type: str | None,
    from_date: str | None,
    to_date: str | None,
    sort_by: str,
    tag_filter: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Hybrid semantic + keyword search."""
    if deep:
        click.echo(
            "Deep search (reranking) is not yet implemented. Using default hybrid search.", err=True
        )

    from kb.search import search as do_search
    from kb.staleness import auto_reindex_if_stale

    auto_reindex_if_stale(_get_db(), _find_project_root())

    embedder = None
    if not fast:
        try:
            from kb.embeddings import Embedder

            embedder = Embedder()
            print(f"Embedding backend: {embedder.backend_name}", file=sys.stderr)
        except Exception:
            click.echo("Warning: Could not load embedding model. Using FTS-only search.", err=True)
            fast = True

    db = _get_db()
    results = do_search(
        db,
        embedder,
        query,
        limit=limit,
        fast=fast,
        recency=recency,
        doc_type=doc_type,
        from_date=from_date,
        to_date=to_date,
        tag=tag_filter,
        sort_by=sort_by,
    )
    kb_output(
        results.model_dump(),
        fmt=fmt,
        fields=fields,
        jq_expr=jq_expr,
        columns=["title", "date", "score", "section", "snippet"],
    )


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("target")
@output_options
def view(target: str, fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """View a specific document by path or #docid."""
    db = _get_db()
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

        from kb.db import normalize_path

        # Parse optional line range: path:LINE or path:START-END
        path_part = target
        if ":" in target:
            parts = target.rsplit(":", 1)
            if parts[1] and (parts[1].isdigit() or "-" in parts[1]):
                path_part = parts[0]

        path_nfc = normalize_path(path_part)
        path_nfd = unicodedata.normalize("NFD", path_part)

        # Exact path match (try both NFC and NFD for compat with old data)
        row = conn.execute(
            "SELECT * FROM documents WHERE path = ? OR path = ?",
            (path_nfc, path_nfd),
        ).fetchone()
        if row:
            doc = dict(row)

        # Suffix match
        if doc is None:
            rows = conn.execute(
                "SELECT * FROM documents WHERE path LIKE ? OR path LIKE ?",
                ("%" + path_nfc, "%" + path_nfd),
            ).fetchall()
            # Deduplicate by document id
            seen_ids = set()
            unique_rows = []
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_rows.append(r)
            rows = unique_rows
            if len(rows) == 1:
                doc = dict(rows[0])
            elif len(rows) > 1:
                paths = [r["path"] for r in rows]
                click.echo(f"Ambiguous path. Matches: {paths}", err=True)
                raise SystemExit(2)

    if doc is None and not target.startswith("#"):
        # Glob / substring matching
        import fnmatch

        all_rows = conn.execute("SELECT path FROM documents").fetchall()
        all_paths = [r["path"] for r in all_rows]

        if "*" in target or "?" in target:
            # Glob matching against all paths
            matches = [p for p in all_paths if fnmatch.fnmatch(p, target)]
        else:
            # Substring match on filename only
            matches = [p for p in all_paths if target in p.rsplit("/", 1)[-1]]

        if len(matches) == 1:
            row = conn.execute("SELECT * FROM documents WHERE path = ?", (matches[0],)).fetchone()
            if row:
                doc = dict(row)
        elif len(matches) > 1:
            click.echo("Ambiguous path. Matches:", err=True)
            for m in matches:
                click.echo(f"  {m}", err=True)
            raise SystemExit(2)

    if doc is None:
        # Suggest close matches
        all_paths = [r["path"] for r in conn.execute("SELECT path FROM documents").fetchall()]
        suggestions = _fuzzy_suggest(target, all_paths)
        msg = f"Document not found: {target}"
        if suggestions:
            msg += f"\nDid you mean: {suggestions[0]}?"
        click.echo(msg, err=True)
        raise SystemExit(1)

    # Build output: document metadata + chunks
    chunks = conn.execute(
        "SELECT heading, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc["id"],),
    ).fetchall()

    # Build content: concatenate chunks, stripping metadata prefixes
    from kb.search import _strip_metadata_prefix

    content_parts = []
    for c in chunks:
        if c["heading"] == "__document__":
            continue  # skip summary chunks
        clean = _strip_metadata_prefix(c["content"])
        if c["heading"]:
            content_parts.append(f"## {c['heading']}\n{clean}")
        else:
            content_parts.append(clean)

    output = {
        "title": doc["title"],
        "path": doc["path"],
        "date": doc["doc_date"],
        "doc_type": doc["doc_type"],
        "content_hash": doc["content_hash"],
        "content": "\n\n".join(content_parts),
        "chunks": [{"heading": c["heading"], "content": c["content"]} for c in chunks],
    }
    kb_output(output, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _fuzzy_suggest(target: str, paths: list[str], max_suggestions: int = 3) -> list[str]:
    """Find close matches using simple substring/Levenshtein."""
    target_lower = target.lower().lstrip("#")
    scored = []
    for p in paths:
        p_lower = p.lower()
        if target_lower in p_lower:
            scored.append((0, p))
        else:
            common = sum(1 for c in target_lower if c in p_lower)
            scored.append((1000 - common, p))
    scored.sort()
    return [p for _, p in scored[:max_suggestions]]


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    import re as _re

    slug = text.lower().strip()
    slug = _re.sub(r"[^\w\s-]", "", slug)
    slug = _re.sub(r"[\s_]+", "-", slug)
    slug = _re.sub(r"-+", "-", slug)
    return slug[:60].strip("-")


def _find_document_by_target(conn: sqlite3.Connection, target: str) -> dict[str, Any] | None:
    """Resolve a document by path, content hash, title, glob, or substring.

    Returns document dict or None. Raises SystemExit(2) for ambiguous matches.
    """
    # Content-hash lookup: #abc123
    if target.startswith("#"):
        hash_prefix = target[1:]
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash LIKE ?",
            (hash_prefix + "%",),
        ).fetchone()
        if row:
            return dict(row)
        return None

    import fnmatch
    import unicodedata

    from kb.db import normalize_path

    path_nfc = normalize_path(target)
    path_nfd = unicodedata.normalize("NFD", target)

    # Exact path match (try both NFC and NFD)
    row = conn.execute(
        "SELECT * FROM documents WHERE path = ? OR path = ?",
        (path_nfc, path_nfd),
    ).fetchone()
    if row:
        return dict(row)

    # Suffix match
    rows = conn.execute(
        "SELECT * FROM documents WHERE path LIKE ? OR path LIKE ?",
        ("%" + path_nfc, "%" + path_nfd),
    ).fetchall()
    seen_ids: set[int] = set()
    unique_rows: list[sqlite3.Row] = []
    for r in rows:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            unique_rows.append(r)
    if len(unique_rows) == 1:
        return dict(unique_rows[0])
    if len(unique_rows) > 1:
        paths = [r["path"] for r in unique_rows]
        click.echo(f"Ambiguous path. Matches: {paths}", err=True)
        raise SystemExit(2)

    # Title match
    title_rows = conn.execute(
        "SELECT * FROM documents WHERE title = ? COLLATE NOCASE",
        (target,),
    ).fetchall()
    if len(title_rows) == 1:
        return dict(title_rows[0])
    if len(title_rows) > 1:
        paths = [r["path"] for r in title_rows]
        click.echo(f"Ambiguous title. Matches: {paths}", err=True)
        raise SystemExit(2)

    # Glob / substring matching
    all_rows = conn.execute("SELECT path FROM documents").fetchall()
    all_paths = [r["path"] for r in all_rows]

    if "*" in target or "?" in target:
        matches = [p for p in all_paths if fnmatch.fnmatch(p, target)]
    else:
        matches = [p for p in all_paths if target in p.rsplit("/", 1)[-1]]

    if len(matches) == 1:
        row = conn.execute("SELECT * FROM documents WHERE path = ?", (matches[0],)).fetchone()
        if row:
            return dict(row)
    elif len(matches) > 1:
        click.echo("Ambiguous path. Matches:", err=True)
        for m in matches:
            click.echo(f"  {m}", err=True)
        raise SystemExit(2)

    return None


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option("--type", "doc_type", default=None, help="Filter by doc_type.")
@click.option("--limit", default=50, type=int, help="Max documents.")
@click.option("--from", "from_date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD).")
@output_options
def list_docs(
    doc_type: str | None,
    limit: int,
    from_date: str | None,
    to_date: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Browse documents by date/type."""
    db = _get_db()
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

    output = {
        "results": docs,
        "meta": {
            "total": len(docs),
            "limit": limit,
        },
    }
    kb_output(
        output,
        fmt=fmt,
        fields=fields,
        jq_expr=jq_expr,
        columns=["title", "date", "doc_type", "path"],
    )


# ---------------------------------------------------------------------------
# Shared entity helpers (used by person + project groups)
# ---------------------------------------------------------------------------


def _entity_list_impl(
    entity_type: str, fmt: str, fields: list[str] | None, jq_expr: str | None
) -> None:
    """List entities of a given type."""
    db = _get_db()
    conn = db.get_sqlite_conn()

    rows = conn.execute(
        "SELECT id, name, entity_type, aliases, metadata, source_path FROM entities "
        "WHERE entity_type = ? ORDER BY name",
        (entity_type,),
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

    output = {
        "results": entities,
        "meta": {"total": len(entities)},
    }
    kb_output(
        output, fmt=fmt, fields=fields, jq_expr=jq_expr, columns=["name", "entity_type", "aliases"]
    )


def _build_entity_result(conn: sqlite3.Connection, entity_row: sqlite3.Row) -> dict[str, Any]:
    """Build a compact entity result with facts, doc count, and breadcrumbs."""
    entity_id = entity_row["id"]
    entity_name = str(entity_row["name"])
    entity_type = str(entity_row["entity_type"])
    source_path = str(entity_row["source_path"]) if entity_row["source_path"] else None

    # Facts from the facts table
    fact_rows = conn.execute(
        "SELECT fact_text, fact_date FROM facts WHERE entity_id = ? ORDER BY fact_date DESC, id DESC",
        (entity_id,),
    ).fetchall()
    facts = [{"text": str(r["fact_text"]), "date": r["fact_date"]} for r in fact_rows]

    # Document count (not the full list)
    doc_count_row = conn.execute(
        "SELECT COUNT(DISTINCT document_id) as cnt FROM entity_mentions WHERE entity_id = ?",
        (entity_id,),
    ).fetchone()
    document_count = int(doc_count_row["cnt"])

    # Breadcrumbs — commands for deeper exploration
    quoted_name = shlex.quote(entity_name)
    breadcrumbs: dict[str, str] = {}
    if entity_type in ("person", "project"):
        breadcrumbs["timeline"] = f"kbx {entity_type} timeline {quoted_name} --limit 20"
        thirty_ago = (date.today() - timedelta(days=30)).isoformat()
        breadcrumbs["recent"] = f"kbx {entity_type} timeline {quoted_name} --from {thirty_ago}"
    else:
        breadcrumbs["search"] = f"kbx search {quoted_name} --limit 20"
    if source_path:
        breadcrumbs["profile"] = f"kbx view {source_path}"

    return {
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
    click.echo("'me' requires [user] name in kbx.toml or ~/.config/kbx/config.toml", err=True)
    raise SystemExit(1)


def _entity_find_impl(
    name: str, entity_type: str | None, fmt: str, fields: list[str] | None, jq_expr: str | None
) -> None:
    """Find entity by name, optionally filtering by type."""
    name = _resolve_me(name)
    db = _get_db()
    conn = db.get_sqlite_conn()

    matches = _find_entities(conn, name)
    if entity_type:
        matches = [m for m in matches if m["entity_type"] == entity_type]
    if not matches:
        _entity_not_found(conn, name, fmt, entity_type=entity_type)
        raise SystemExit(1)

    if len(matches) == 1:
        result: Any = _build_entity_result(conn, matches[0])
    else:
        result = {
            "results": [_build_entity_result(conn, m) for m in matches],
            "meta": {
                "total": len(matches),
                "note": f"Multiple entities match '{name}'. Use a more specific name.",
            },
        }
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _entity_pin_impl(name: str, entity_type: str | None = None) -> None:
    """Pin an entity so it always appears in context output."""
    db = _get_db()
    conn = db.get_sqlite_conn()
    entity_row = _find_entity(conn, name)
    if entity_row is None or (entity_type and entity_row["entity_type"] != entity_type):
        _entity_not_found(conn, name, "table", entity_type=entity_type)
        raise SystemExit(1)
    conn.execute("UPDATE entities SET pinned = 1 WHERE id = ?", (entity_row["id"],))
    conn.commit()
    from kb.writeback import write_entity_to_file

    write_entity_to_file(db, entity_row["id"], _find_project_root())
    click.echo(f"Pinned: {entity_row['name']}")


def _entity_unpin_impl(name: str, entity_type: str | None = None) -> None:
    """Unpin an entity from context output."""
    db = _get_db()
    conn = db.get_sqlite_conn()
    entity_row = _find_entity(conn, name)
    if entity_row is None or (entity_type and entity_row["entity_type"] != entity_type):
        _entity_not_found(conn, name, "table", entity_type=entity_type)
        raise SystemExit(1)
    conn.execute("UPDATE entities SET pinned = 0 WHERE id = ?", (entity_row["id"],))
    conn.commit()
    from kb.writeback import write_entity_to_file

    write_entity_to_file(db, entity_row["id"], _find_project_root())
    click.echo(f"Unpinned: {entity_row['name']}")


def _entity_timeline_impl(
    name: str,
    entity_type: str | None,
    from_date: str | None,
    to_date: str | None,
    limit: int | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Chronological docs mentioning an entity."""
    name = _resolve_me(name)
    db = _get_db()
    conn = db.get_sqlite_conn()

    entity_row = _find_entity(conn, name)
    if entity_row is None or (entity_type and entity_row["entity_type"] != entity_type):
        _entity_not_found(conn, name, fmt, entity_type=entity_type)
        raise SystemExit(1)

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
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    docs = conn.execute(sql, params).fetchall()

    # Deduplicate: same date+title → keep notes over transcript
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for d in docs:
        key = (d["doc_date"], d["title"])
        entry = {
            "path": d["path"],
            "title": d["title"],
            "date": d["doc_date"],
            "doc_type": d["doc_type"],
            "mention_type": d["mention_type"],
        }
        if key not in seen:
            seen[key] = entry
        elif d["doc_type"] == "notes":
            seen[key] = entry  # prefer notes over transcript

    result = {
        "name": entity_row["name"],
        "entity_type": entity_row["entity_type"],
        "documents": list(seen.values()),
    }
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _entity_create_impl(
    entity_type: str,
    name: str,
    metadata: dict[str, str],
    aliases: list[str],
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Create a new entity via CRUD engine."""
    from kb.crud import EntityExistsError, create_entity

    try:
        result = create_entity(
            _get_db(),
            _find_project_root(),
            entity_type,
            name,
            metadata=metadata or None,
            aliases=aliases or None,
        )
    except EntityExistsError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _entity_edit_impl(
    name: str,
    metadata: dict[str, str],
    aliases: list[str],
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Edit an entity's metadata via CRUD engine."""
    from kb.crud import EntityNotFoundError, edit_entity

    try:
        result = edit_entity(
            _get_db(),
            _find_project_root(),
            name,
            metadata=metadata or None,
            aliases=aliases or None,
        )
    except EntityNotFoundError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _entity_delete_impl(name: str) -> None:
    """Delete an entity via CRUD engine."""
    from kb.crud import EntityNotFoundError, delete_entity

    try:
        result = delete_entity(_get_db(), _find_project_root(), name)
    except EntityNotFoundError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    click.echo(f"Deleted: {result['name']}")


# ---------------------------------------------------------------------------
# person group
# ---------------------------------------------------------------------------


@cli.command()
@output_options
def me(fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """Shortcut for 'person find me' — show your own profile."""
    _entity_find_impl("me", "person", fmt, fields, jq_expr)


# ---------------------------------------------------------------------------


@cli.group()
def person() -> None:
    """Person commands — create, edit, delete, find, list, timeline."""
    pass


@person.command("create")
@click.argument("name")
@click.option("--role", default=None)
@click.option("--email", default=None)
@click.option("--team", default=None)
@click.option("--alias", "aliases", multiple=True)
@click.option("--reports-to", default=None)
@click.option("--company", default=None)
@output_options
def person_create(
    name: str,
    role: str | None,
    email: str | None,
    team: str | None,
    aliases: tuple[str, ...],
    reports_to: str | None,
    company: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Create a new person."""
    metadata = {
        k: v
        for k, v in {
            "role": role,
            "email": email,
            "team": team,
            "reports_to": reports_to,
            "company": company,
        }.items()
        if v
    }
    _entity_create_impl("person", name, metadata, list(aliases), fmt, fields, jq_expr)


@person.command("edit")
@click.argument("name")
@click.option("--role", default=None)
@click.option("--email", default=None)
@click.option("--team", default=None)
@click.option("--alias", "aliases", multiple=True)
@click.option("--reports-to", default=None)
@click.option("--company", default=None)
@output_options
def person_edit(
    name: str,
    role: str | None,
    email: str | None,
    team: str | None,
    aliases: tuple[str, ...],
    reports_to: str | None,
    company: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Edit a person's metadata."""
    metadata = {
        k: v
        for k, v in {
            "role": role,
            "email": email,
            "team": team,
            "reports_to": reports_to,
            "company": company,
        }.items()
        if v
    }
    _entity_edit_impl(name, metadata, list(aliases), fmt, fields, jq_expr)


@person.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this person?")
def person_delete(name: str) -> None:
    """Delete a person (file + database)."""
    _entity_delete_impl(name)


@person.command("find")
@click.argument("name")
@output_options
def person_find(name: str, fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """Look up person profile + linked documents."""
    _entity_find_impl(name, "person", fmt, fields, jq_expr)


@person.command("list")
@output_options
def person_list(fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """List all known people."""
    _entity_list_impl("person", fmt, fields, jq_expr)


@person.command("pin")
@click.argument("name")
def person_pin(name: str) -> None:
    """Pin a person so they always appear in context output."""
    _entity_pin_impl(name, "person")


@person.command("unpin")
@click.argument("name")
def person_unpin(name: str) -> None:
    """Unpin a person from context output."""
    _entity_unpin_impl(name, "person")


@person.command("timeline")
@click.argument("name")
@click.option("--from", "from_date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD).")
@click.option("--limit", "limit", default=None, type=int, help="Max results.")
@output_options
def person_timeline(
    name: str,
    from_date: str | None,
    to_date: str | None,
    limit: int | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Chronological docs mentioning a person."""
    _entity_timeline_impl(name, "person", from_date, to_date, limit, fmt, fields, jq_expr)


# ---------------------------------------------------------------------------
# project group
# ---------------------------------------------------------------------------


@cli.group()
def project() -> None:
    """Project commands — create, edit, delete, find, list."""
    pass


@project.command("create")
@click.argument("name")
@click.option("--status", default=None)
@click.option("--lead", default=None)
@click.option("--codename", default=None)
@click.option("--started", default=None, help="Start date (YYYY-MM).")
@output_options
def project_create(
    name: str,
    status: str | None,
    lead: str | None,
    codename: str | None,
    started: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Create a new project."""
    metadata = {k: v for k, v in {"status": status, "lead": lead, "started": started}.items() if v}
    aliases = [codename] if codename else []
    _entity_create_impl("project", name, metadata, aliases, fmt, fields, jq_expr)


@project.command("edit")
@click.argument("name")
@click.option("--status", default=None)
@click.option("--lead", default=None)
@click.option("--codename", default=None)
@click.option("--started", default=None, help="Start date (YYYY-MM).")
@output_options
def project_edit(
    name: str,
    status: str | None,
    lead: str | None,
    codename: str | None,
    started: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Edit a project's metadata."""
    metadata = {k: v for k, v in {"status": status, "lead": lead, "started": started}.items() if v}
    aliases = [codename] if codename else []
    _entity_edit_impl(name, metadata, aliases, fmt, fields, jq_expr)


@project.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this project?")
def project_delete(name: str) -> None:
    """Delete a project (file + database)."""
    _entity_delete_impl(name)


@project.command("find")
@click.argument("name")
@output_options
def project_find(name: str, fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """Look up project profile + linked documents."""
    _entity_find_impl(name, "project", fmt, fields, jq_expr)


@project.command("list")
@output_options
def project_list(fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """List all known projects."""
    _entity_list_impl("project", fmt, fields, jq_expr)


# ---------------------------------------------------------------------------
# glossary group
# ---------------------------------------------------------------------------


@cli.group()
def glossary() -> None:
    """Glossary commands — add, list, delete."""
    pass


@glossary.command("add")
@click.argument("term")
@click.argument("expansion")
@click.option("--section", default="Acronyms", help="Glossary section heading.")
@output_options
def glossary_add(
    term: str, expansion: str, section: str, fmt: str, fields: list[str] | None, jq_expr: str | None
) -> None:
    """Add a term to the glossary."""
    from kb.glossary import add_term

    result = add_term(_find_project_root(), term, expansion, section=section)
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


@glossary.command("list")
@output_options
def glossary_list(fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """List all glossary terms."""
    from kb.glossary import list_terms

    terms = list_terms(_find_project_root())
    output = {"results": [t.model_dump() for t in terms], "meta": {"total": len(terms)}}
    kb_output(
        output, fmt=fmt, fields=fields, jq_expr=jq_expr, columns=["term", "expansion", "section"]
    )


@glossary.command("delete")
@click.argument("term")
def glossary_delete(term: str) -> None:
    """Delete a term from the glossary."""
    from kb.glossary import delete_term

    try:
        delete_term(_find_project_root(), term)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    click.echo(f"Deleted: {term}")


# ---------------------------------------------------------------------------
# index group
# ---------------------------------------------------------------------------


@cli.group()
def index() -> None:
    """Index commands — run, status."""
    pass


@index.command("run")
@click.argument("paths", nargs=-1)
@click.option("--full", is_flag=True, help="Full re-index (clears existing data).")
@click.option("--cpu", is_flag=True, help="Force CPU (avoids MPS GPU memory pressure).")
@click.option(
    "--batch-size", type=int, default=None, help="Embedding batch size (default: 32 MPS, 16 CPU)."
)
@click.option(
    "--no-embed",
    is_flag=True,
    help="Text-only indexing (SQLite + FTS + entities, no vector embeddings).",
)
def index_run(
    paths: tuple[str, ...], full: bool, cpu: bool, batch_size: int | None, no_embed: bool
) -> None:
    """Run incremental or full index."""
    from kb.indexer import index_all

    project_root = _find_project_root()

    if no_embed:
        embedder = None
        click.echo("Text-only mode: SQLite, FTS, and entity linking (no embeddings).", err=True)
    else:
        try:
            from kb.embeddings import Embedder

            device = "cpu" if cpu else None
            click.echo(f"Loading embedding model (device={'cpu' if cpu else 'auto'})...", err=True)
            embedder = Embedder(device=device)
            print(f"Embedding backend: {embedder.backend_name}", file=sys.stderr)
        except ImportError:
            click.echo(
                "Warning: sentence_transformers not available. Falling back to text-only mode.",
                err=True,
            )
            embedder = None

    db = _get_db()
    click.echo(f"Indexing from {project_root} (full={full})...", err=True)
    result = index_all(db, embedder, project_root, full=full, batch_size=batch_size)

    embed_status = "skipped" if result.embeddings_skipped else "done"
    click.echo(
        f"Done: {result.documents_indexed} indexed, {result.documents_skipped} skipped, "
        f"{result.chunks_created} chunks, {result.entities_linked} entity links, "
        f"embeddings: {embed_status}, {len(result.errors)} errors.",
        err=True,
    )

    if result.errors:
        for err in result.errors[:10]:
            click.echo(f"  Error: {err}", err=True)


@index.command("status")
@output_options
def index_status(fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
    """Database health: counts, size, freshness."""
    db = _get_db()
    conn = db.get_sqlite_conn()

    # Document counts by type
    type_rows = conn.execute(
        "SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type"
    ).fetchall()
    doc_counts = {r["doc_type"]: r["count"] for r in type_rows}
    total_docs = sum(doc_counts.values())

    # Chunk count
    chunk_count = conn.execute("SELECT COUNT(*) as count FROM chunks").fetchone()["count"]

    # Entity count
    entity_count = conn.execute("SELECT COUNT(*) as count FROM entities").fetchone()["count"]

    # Last indexed timestamp
    last_indexed = conn.execute("SELECT MAX(indexed_at) as ts FROM documents").fetchone()["ts"]

    # LanceDB vector count
    lance_table = db.get_lance_table()
    vector_count = 0
    if lance_table is not None:
        with contextlib.suppress(Exception):
            vector_count = lance_table.count_rows()

    # Database file sizes
    data_dir = _get_data_dir()
    sqlite_size = 0
    sqlite_path = data_dir / "metadata.db"
    if sqlite_path.exists():
        sqlite_size = sqlite_path.stat().st_size

    lance_size = 0
    lance_dir = data_dir / "vectors"
    if lance_dir.exists():
        for f in lance_dir.rglob("*"):
            if f.is_file():
                lance_size += f.stat().st_size

    status = {
        "documents": total_docs,
        "documents_by_type": doc_counts,
        "chunks": chunk_count,
        "entities": entity_count,
        "vectors": vector_count,
        "last_indexed": last_indexed,
        "sqlite_size_bytes": sqlite_size,
        "lance_size_bytes": lance_size,
    }
    kb_output(status, fmt=fmt, fields=fields, jq_expr=jq_expr)


@index.command("compact")
def index_compact() -> None:
    """Compact LanceDB vectors: merge fragments and prune old versions."""
    db = _get_db()
    lance_table = db.get_lance_table()
    if lance_table is None:
        click.echo("No LanceDB table found.", err=True)
        return

    before = lance_table.count_rows()
    lance_dir = _get_data_dir() / "vectors"
    size_before = sum(f.stat().st_size for f in lance_dir.rglob("*") if f.is_file())

    click.echo(f"Before: {before} vectors, {size_before / 1024 / 1024:.1f} MB", err=True)
    lance_table.optimize(cleanup_older_than=timedelta(0))

    after = lance_table.count_rows()
    size_after = sum(f.stat().st_size for f in lance_dir.rglob("*") if f.is_file())
    click.echo(f"After:  {after} vectors, {size_after / 1024 / 1024:.1f} MB", err=True)
    click.echo("Compaction complete.", err=True)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("paths", nargs=-1)
@click.option("--dry-run", is_flag=True, help="Preview only, no changes.")
@click.option("--skip-organise", is_flag=True, help="Skip organise step, only index.")
def ingest(paths: tuple[str, ...], dry_run: bool, skip_organise: bool) -> None:
    """Organise Granola exports and index into kb."""
    import subprocess
    import sys

    project_root = _find_project_root()

    # Step 1: Run organise_granola.py (unless skipped)
    if not skip_organise:
        organise_script = project_root / "meetings" / "organise_granola.py"
        if not organise_script.exists():
            click.echo(f"Error: organise_granola.py not found at {organise_script}", err=True)
            raise SystemExit(1)

        cmd = [sys.executable, str(organise_script)]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(str(p) for p in paths)

        click.echo("Organising Granola exports...", err=True)
        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            click.echo("Error: organise_granola.py failed.", err=True)
            raise SystemExit(1)
        click.echo("Organise complete.", err=True)

    if dry_run:
        click.echo("Dry run — skipping index step.", err=True)
        return

    # Step 2: Index (load embedder only now)
    click.echo("Loading embedding model...", err=True)
    try:
        from kb.embeddings import Embedder

        embedder = Embedder()
        print(f"Embedding backend: {embedder.backend_name}", file=sys.stderr)
    except Exception as e:
        click.echo(f"Warning: Could not load embedding model ({e}). Skipping index.", err=True)
        return

    from kb.indexer import index_all

    db = _get_db()
    click.echo("Indexing...", err=True)
    idx_result = index_all(db, embedder, project_root)

    click.echo(
        f"Done: {idx_result.documents_indexed} indexed, {idx_result.documents_skipped} skipped, "
        f"{idx_result.chunks_created} chunks, {idx_result.entities_linked} entity links, "
        f"{len(idx_result.errors)} errors.",
        err=True,
    )


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


@cli.command()
def usage() -> None:
    """Self-documentation for AI agents."""
    import contextlib

    # Dynamic index stats
    stats_line = ""
    with contextlib.suppress(Exception):
        db = _get_db()
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
        stats_line = (
            f"  {total_docs} docs | {total_entities} entities | {total_facts} facts | "
            f"{pinned_docs} pinned | dates {earliest} to {latest}"
        )

    text = f"""# kb — Agent Playbook

## 1. Quick Start
  kb context              # orient: pinned docs + entity index
  kb search "topic" --fast --json --limit 5   # keyword search (~instant)
  kb view <path>          # read a full document

## 2. Index Status
{stats_line}

## 3. Taking Notes
  kb memory add "title" --body "markdown content" --tags t1,t2 --pin
  kb memory add "Quick note"                     # one-liner, no body
  kb memory add "fact" --entity "Name"           # fact appended to entity file
  kb memory add "title" --body "..." --entity "Name"  # note linked to entity

## 4. When to Pin
  kb pin <path|title|#hash|glob>     # pin any doc to context
  kb unpin <path|title|glob>         # remove from context
  kb memory add "title" --pin        # create + pin in one step
  kb person pin "Name"               # pin a person to context

## 5. Browsing Notes
  kb note list --json                   # all notes
  kb note list --tag decision --json    # filter by tag (AND: --tag a,b)
  kb note list --pinned --json          # pinned notes only
  kb note list --tag ops --pinned       # combine filters

## 6. Finding Things
  kb search "query" --fast --json             # keyword (FTS, instant)
  kb search "query" --json --limit 10         # hybrid (semantic + FTS, ~2s)
  kb search "query" --tag infra --fast --json # filter by tag
  kb search "query" --from 2026-01-01 --to 2026-01-31  # date range
  kb search "query" --sort date --json        # newest first
  kb view <path|#hash|glob> --json            # full document
  kb list --type notes --from 2026-02-01      # browse by type/date

  Score Interpretation: 0.8+ strong | 0.5-0.8 worth reading | <0.5 noise

## 7. People & Projects
  kb me --json                       # your own profile (shortcut for person find me)
  kb person find "Name" --json       # compact profile (facts, metadata, breadcrumbs)
  kb person timeline "Name" --json   # chronological doc list
  kb person create "Name" --role "Role" --team "Team"
  kb person edit "Name" --role "New Role"
  kb person list --json              # all people
  kb project find "Name" --json      # compact profile (facts, metadata, breadcrumbs)
  kb project create "Name" --status "Active" --lead "Name"
  kb project list --json             # all projects
  kb glossary add "TERM" "expansion"
  kb glossary list --json

## 8. Context & Indexing
  kb context                         # compact entity index (for agents)
  kb context --human                 # markdown format (for humans)
  kb context --for "topic"           # filtered to a topic
  kb index status --json             # database health
  kb index run --no-embed            # text-only index (fast, no GPU)
  kb index run --cpu                 # full index with embeddings on CPU

## Global Options
  --json | --format table|json|jsonl|csv | --fields f1,f2 | --jq expr
"""
    click.echo(text.strip())


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--global", "global_", is_flag=True, help="Write to ~/.config/kbx/config.toml.")
def init(global_: bool) -> None:
    """Initialize kbx configuration."""
    import os as _os

    if global_:
        xdg = _os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        config_dir = Path(xdg) / "kbx"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.toml"
    else:
        config_path = Path.cwd() / "kbx.toml"

    if config_path.exists():
        click.echo(f"Config already exists: {config_path}")
        return

    meetings = click.prompt("Where are your meeting transcripts?", default="meetings/organised")
    memory = click.prompt("Where are your memory files?", default="memory")

    default_data = str(Path.home() / ".local" / "share" / "kbx")
    data_dir = click.prompt("Where should kbx store its database?", default=default_data)

    config_content = (
        f'[sources]\nmeetings = "{meetings}"\nmemory = "{memory}"\n\n[data]\ndir = "{data_dir}"\n'
    )
    # Atomic write (temp file → rename) per project convention
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".toml")
    try:
        with _os.fdopen(tmp_fd, "w") as f:
            f.write(config_content)
        Path(tmp_path).replace(config_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    click.echo(f"Config written to {config_path}")

    # Resolve relative data dir against config location before mkdir
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Created database directory: {data_path}")
    click.echo("Run `kb index run` to index your files.")


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--for", "topic", default=None, help="Filter to entities relevant to a topic.")
@click.option("--json", "json_flag", is_flag=True, help="Output structured JSON.")
@click.option("--human", "human_flag", is_flag=True, help="Human-readable markdown format.")
def context(topic: str | None, json_flag: bool, human_flag: bool) -> None:
    """Compressed entity index for AI agents."""
    from kb.context import generate_context
    from kb.staleness import auto_reindex_if_stale

    db = _get_db()
    project_root = _find_project_root()
    auto_reindex_if_stale(db, project_root)
    fmt = "human" if human_flag else "compact"
    result = generate_context(db, project_root, topic=topic, fmt=fmt)

    if json_flag:
        click.echo(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    else:
        click.echo(result.text)


# ---------------------------------------------------------------------------
# memory group
# ---------------------------------------------------------------------------


@cli.group()
def memory() -> None:
    """Memory commands — add facts, list facts."""
    pass


@memory.command("add")
@click.argument("text")
@click.option("--entity", "entity_name", default=None, help="Entity name to associate with.")
@click.option("--body", "body", default=None, help="Multi-line body content (use - for stdin).")
@click.option("--tags", "tags_str", default=None, help="Comma-separated tags.")
@click.option("--pin", "pin_flag", is_flag=True, help="Pin this note to context.")
@click.option("--date", "fact_date", default=None, help="Date (YYYY-MM-DD, default: today).")
def memory_add(
    text: str,
    entity_name: str | None,
    body: str | None,
    tags_str: str | None,
    pin_flag: bool,
    fact_date: str | None,
) -> None:
    """Record a fact about an entity, or create a searchable note.

    \b
    Decision tree:
      --entity only (no --body/--tags/--pin) → fact (appended to entity file)
      everything else                        → note file in memory/notes/
    """
    import os

    from kb.db import normalize_path

    db = _get_db()
    conn = db.get_sqlite_conn()
    project_root = _find_project_root()

    if fact_date is None:
        fact_date = date.today().isoformat()

    # Decision tree: fact path vs note path
    is_note = body is not None or tags_str is not None or pin_flag or entity_name is None

    if not is_note:
        # --- FACT PATH (existing behavior, unchanged) ---
        assert entity_name is not None  # guaranteed by is_note logic
        entity_row = _find_entity(conn, entity_name)
        if entity_row is None:
            click.echo(f"Entity not found: {entity_name}", err=True)
            raise SystemExit(1)

        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (?, ?, ?)",
            (entity_row["id"], text, fact_date),
        )
        conn.commit()

        source_path = entity_row["source_path"]
        if source_path:
            source_file = Path(source_path)
            if not source_file.is_absolute():
                source_file = project_root / source_path
            if source_file.exists():
                _append_fact_to_file(source_file, text, fact_date)

        click.echo(f"Fact recorded for {entity_row['name']}.", err=True)
        return

    # --- NOTE PATH ---
    # Read body from stdin if -
    if body == "-":
        body = sys.stdin.read()

    # Parse tags
    tags: list[str] = []
    if tags_str:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    # Generate slug and filename
    slug = _slugify(text)
    notes_dir = project_root / "memory" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{fact_date}-{slug}.md"
    filepath = notes_dir / filename

    # Handle collision
    counter = 2
    while filepath.exists():
        filepath = notes_dir / f"{fact_date}-{slug}-{counter}.md"
        counter += 1

    # Build note content
    frontmatter_lines = [
        "---",
        f"title: {text}",
        f"date: {fact_date}",
    ]
    if tags:
        frontmatter_lines.append(f"tags: [{', '.join(tags)}]")
    if pin_flag:
        frontmatter_lines.append("pinned: true")
    frontmatter_lines.append("---")
    frontmatter_lines.append("")

    note_body = body if body else text
    note_content = "\n".join(frontmatter_lines) + note_body + "\n"

    # Atomic write
    tmp_path = filepath.with_suffix(".md.tmp")
    tmp_path.write_text(note_content, encoding="utf-8")
    os.replace(str(tmp_path), str(filepath))

    rel_path = normalize_path(str(filepath.relative_to(project_root)))

    # Immediate index (text-only, fast — embeddings on next kb index run)
    from kb.indexer import index_all

    index_all(db, None, project_root, memory_only=True, skip_seed=True)

    # Pin if requested
    if pin_flag:
        conn.execute("UPDATE documents SET pinned = 1 WHERE path = ?", (rel_path,))
        conn.commit()

    # Entity linking
    if entity_name:
        entity_row = _find_entity(conn, entity_name)
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
        else:
            click.echo(
                f"Warning: entity '{entity_name}' not found, note created without entity link.",
                err=True,
            )

    click.echo(f"Note created: {rel_path}", err=True)
    if pin_flag:
        click.echo("Pinned to context.", err=True)


def _append_fact_to_file(path: Path, fact_text: str, fact_date: str) -> None:
    """Append a fact to a markdown file under a ## Recent Facts section."""
    import re as _re

    content = path.read_text(encoding="utf-8")
    fact_line = f"- [{fact_date}] {fact_text}\n"

    if "## Recent Facts" in content:
        # Find the position of ## Recent Facts
        header_match = _re.search(r"^## Recent Facts\s*$", content, _re.MULTILINE)
        assert header_match is not None
        # Find the next ## heading after it
        next_section = _re.search(r"^## ", content[header_match.end() :], _re.MULTILINE)
        if next_section:
            # Insert fact before the next section
            insert_pos = header_match.end() + next_section.start()
            content = content[:insert_pos] + fact_line + "\n" + content[insert_pos:]
        else:
            # No following section — append at end
            content = content.rstrip("\n") + "\n" + fact_line
    else:
        # Add the section at the end
        content = content.rstrip("\n") + "\n\n## Recent Facts\n" + fact_line

    path.write_text(content, encoding="utf-8")


@memory.command("list")
@click.option("--since", "since_days", default=None, type=int, help="Show facts from last N days.")
@output_options
def memory_list(
    since_days: int | None, fmt: str, fields: list[str] | None, jq_expr: str | None
) -> None:
    """List recorded facts."""
    db = _get_db()
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

    if not facts and fmt == "table":
        click.echo(
            'No facts recorded. Use \'kb memory add "fact" --entity "name"\' to record facts.'
        )
        return

    meta: dict[str, Any] = {
        "total": len(facts),
    }
    if not facts:
        meta["message"] = (
            'No facts recorded. Use \'kb memory add "fact" --entity "name"\' to record facts.'
        )
    output: dict[str, Any] = {
        "results": facts,
        "meta": meta,
    }

    kb_output(
        output,
        fmt=fmt,
        fields=fields,
        jq_expr=jq_expr,
        columns=["fact_date", "entity_name", "fact_text"],
    )


# ---------------------------------------------------------------------------
# note group
# ---------------------------------------------------------------------------


@cli.group()
def note() -> None:
    """Note commands — browse and filter notes."""
    pass


@note.command("list")
@click.option("--tag", "tag_filter", default=None, help="Filter by tag (comma-separated for AND).")
@click.option("--pinned", "pinned_only", is_flag=True, help="Show only pinned notes.")
@click.option("--limit", default=50, type=int, help="Max results.")
@output_options
def note_list(
    tag_filter: str | None,
    pinned_only: bool,
    limit: int,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Browse notes with optional tag/pin filtering."""
    db = _get_db()
    conn = db.get_sqlite_conn()

    rows = conn.execute(
        "SELECT id, path, title, doc_date, tags, pinned "
        "FROM documents "
        "WHERE doc_type IN ('memory_note', 'memory_doc') "
        "ORDER BY doc_date DESC, id DESC",
    ).fetchall()

    # Parse required tags for AND filter
    required_tags: list[str] = []
    if tag_filter:
        required_tags = [t.strip().lower() for t in tag_filter.split(",") if t.strip()]

    results: list[dict[str, Any]] = []
    for r in rows:
        raw_tags = r["tags"]
        doc_tags: list[str] = json.loads(raw_tags) if raw_tags else []
        is_pinned = bool(r["pinned"])

        # Tag filter: every required tag must be present (case-insensitive)
        if required_tags:
            lower_tags = [t.lower() for t in doc_tags]
            if not all(rt in lower_tags for rt in required_tags):
                continue

        # Pinned filter
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

    output: dict[str, Any] = {
        "results": results,
        "meta": {"total": len(results), "limit": limit},
    }
    kb_output(
        output,
        fmt=fmt,
        fields=fields,
        jq_expr=jq_expr,
        columns=["title", "date", "tags", "pinned"],
    )


# ---------------------------------------------------------------------------
# pin / unpin
# ---------------------------------------------------------------------------


def _update_frontmatter_pinned(project_root: Path, doc_path: str, pinned: bool) -> None:
    """Update or inject the ``pinned:`` field in a file's YAML frontmatter.

    - If the file has frontmatter, update or inject the ``pinned:`` line.
    - If the file has no frontmatter and *pinned* is True, prepend a minimal
      frontmatter block with ``pinned: true``.
    - If the file has no frontmatter and *pinned* is False, do nothing.
    - Silently returns if the file does not exist on disk (e.g. DB-only docs).
    """
    import re as _re

    filepath = project_root / doc_path
    if not filepath.exists():
        return

    text = filepath.read_text(encoding="utf-8")
    pinned_str = "true" if pinned else "false"

    fm_match = _re.match(r"^---\s*\n(.*?)\n---\s*\n", text, _re.DOTALL)
    if fm_match:
        fm_block = fm_match.group(1)
        # Replace existing pinned: line or inject a new one
        if _re.search(r"^pinned:\s", fm_block, _re.MULTILINE):
            new_fm = _re.sub(
                r"^pinned:\s.*$",
                f"pinned: {pinned_str}",
                fm_block,
                count=1,
                flags=_re.MULTILINE,
            )
        else:
            new_fm = fm_block.rstrip("\n") + f"\npinned: {pinned_str}"
        text = f"---\n{new_fm}\n---\n" + text[fm_match.end() :]
    elif pinned:
        # No frontmatter — inject minimal block
        text = f"---\npinned: true\n---\n{text}"
    else:
        # No frontmatter and unpinning — nothing to do
        return

    filepath.write_text(text, encoding="utf-8")


@cli.command()
@click.argument("target")
def pin(target: str) -> None:
    """Pin a document to context. Accepts path, title, or glob."""
    db = _get_db()
    conn = db.get_sqlite_conn()

    doc = _find_document_by_target(conn, target)
    if doc is None:
        all_paths = [r["path"] for r in conn.execute("SELECT path FROM documents").fetchall()]
        suggestions = _fuzzy_suggest(target, all_paths)
        msg = f"Document not found: {target}"
        if suggestions:
            msg += f"\nDid you mean: {suggestions[0]}?"
        click.echo(msg, err=True)
        raise SystemExit(1)

    conn.execute("UPDATE documents SET pinned = 1 WHERE id = ?", (doc["id"],))
    conn.commit()
    _update_frontmatter_pinned(_find_project_root(), doc["path"], pinned=True)
    click.echo(f"Pinned: {doc['path']}", err=True)


@cli.command()
@click.argument("target")
def unpin(target: str) -> None:
    """Unpin a document from context."""
    db = _get_db()
    conn = db.get_sqlite_conn()

    doc = _find_document_by_target(conn, target)
    if doc is None:
        click.echo(f"Document not found: {target}", err=True)
        raise SystemExit(1)

    conn.execute("UPDATE documents SET pinned = 0 WHERE id = ?", (doc["id"],))
    conn.commit()
    _update_frontmatter_pinned(_find_project_root(), doc["path"], pinned=False)
    click.echo(f"Unpinned: {doc['path']}", err=True)


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


@cli.command()
def mcp() -> None:
    """Start MCP server on stdio transport."""
    from kb.mcp_server import main as mcp_main

    mcp_main()


# ---------------------------------------------------------------------------
# sync group
# ---------------------------------------------------------------------------


@cli.group()
def sync() -> None:
    """Sync commands — pull data from external sources."""
    pass


@sync.command("granola")
@click.option("--since", default=None, help="Sync documents since date (YYYY-MM-DD).")
@click.option("--dry-run", is_flag=True, help="Preview only, no file writes.")
@click.option("--force", is_flag=True, help="Overwrite files even if timestamps match.")
@click.option("--no-index", is_flag=True, help="Skip indexing after sync.")
@click.option(
    "--token-path", default=None, type=click.Path(exists=False), help="Path to supabase.json."
)
def sync_granola(
    since: str | None, dry_run: bool, force: bool, no_index: bool, token_path: str | None
) -> None:
    """Sync meetings from Granola API."""
    from pathlib import Path as P

    from kb.sync.granola import sync_granola as do_sync

    project_root = _find_project_root()
    data_dir = _get_data_dir()

    tp = P(token_path) if token_path else None

    result = do_sync(
        project_root=project_root,
        data_dir=data_dir,
        since=since,
        dry_run=dry_run,
        force=force,
        token_path=tp,
        on_progress=lambda msg: click.echo(msg, err=True),
    )

    click.echo(
        f"Sync complete: {result['created']} created, {result['updated']} updated, "
        f"{result['skipped']} skipped (total: {result['total']}).",
        err=True,
    )

    if not dry_run and not no_index:
        click.echo("Tip: Run `kb index run` to index new files.", err=True)
