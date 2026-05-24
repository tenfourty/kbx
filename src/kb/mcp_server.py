"""MCP server for kbx — exposes knowledge base tools and resources via FastMCP."""

from __future__ import annotations

import json
import re
import sys
import traceback
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from kb.config import find_entity, find_project_root, get_db
from kb.crud import find_document_by_target as _find_document_by_target

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
    path_filter: str | None = None,
    explain: bool = False,
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
            path_filter=path_filter,
            sort_by=sort_by,
            doc_type=doc_type,
            explain=explain,
        )
        return json.dumps(results.model_dump(), default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_search error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def _resolve_me(name: str) -> str:
    """Resolve 'me' to the configured user name."""
    from kb.user_config import resolve_me

    result = resolve_me(name)
    return result if result is not None else name


def handle_kb_person_find(db: Database, name: str) -> str:
    """Look up a person profile. Returns JSON string."""
    try:
        name = _resolve_me(name)
        return _build_entity_result(db, name)
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
    """Chronological docs mentioning a person. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        name = _resolve_me(name)
        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        entity = kb.get_entity(name)
        if entity is None:
            kb.close()
            return json.dumps({"error": f"Entity not found: {name}"})

        entries = kb.get_entity_timeline(
            name, limit=limit, from_date=from_date, to_date=to_date, doc_type=doc_type
        )
        kb.close()

        result = {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "documents": [
                {
                    "path": e.path,
                    "title": e.title,
                    "date": e.date,
                    "doc_type": e.doc_type,
                    "mention_type": e.mention_type,
                }
                for e in entries
            ],
        }
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"kb_person_timeline error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_view(db: Database, target: str) -> str:
    """View a specific document by path or #hash. Delegates to KnowledgeBase."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        try:
            result = kb.view_document(target)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        finally:
            kb.close()
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
    """Create a note or record a fact. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
        try:
            is_note = body is not None or tags is not None or pin or entity is None

            if not is_note:
                assert entity is not None
                try:
                    result = kb.add_fact(entity, text, date=date)
                except ValueError as e:
                    return json.dumps({"error": str(e)})
                return json.dumps(result)

            # NOTE PATH
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            result = kb.add_note(text, body=body, tags=tag_list, pin=pin, entity=entity, date=date)
            return json.dumps(result)
        finally:
            kb.close()
    except Exception as e:
        print(f"kb_memory_add error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def _set_pin_frontmatter(project_root: Path, doc_path: str, pinned: bool) -> None:
    """Write pinned: true/false to a memory note's YAML frontmatter."""
    file_path = project_root / doc_path
    if not file_path.exists():
        return
    content = file_path.read_text(encoding="utf-8")
    pinned_val = "true" if pinned else "false"
    pinned_line = f"pinned: {pinned_val}"

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if fm_match:
        fm_block = fm_match.group(1)
        if re.search(r"^pinned:\s", fm_block, re.MULTILINE):
            fm_block = re.sub(r"^pinned:\s.*$", pinned_line, fm_block, count=1, flags=re.MULTILINE)
        else:
            fm_block = fm_block.rstrip("\n") + f"\n{pinned_line}"
        new_content = f"---\n{fm_block}\n---\n{content[fm_match.end() :]}"
    else:
        new_content = f"---\n{pinned_line}\n---\n{content}"

    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=str(file_path.parent), suffix=".md.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(tmp, str(file_path))
    except BaseException:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def handle_kb_pin(db: Database, target: str, project_root: Path | None = None) -> str:
    """Pin a document to context. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Document not found: {target}"})
        conn.execute("UPDATE documents SET pinned = 1 WHERE id = ?", (doc["id"],))
        conn.commit()

        # Write-through: update frontmatter for memory notes
        if project_root and doc.get("doc_type", "").startswith("memory"):
            _set_pin_frontmatter(project_root, doc["path"], True)

        return json.dumps({"status": "ok", "path": doc["path"], "pinned": True})
    except Exception as e:
        print(f"kb_pin error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_unpin(db: Database, target: str, project_root: Path | None = None) -> str:
    """Unpin a document from context. Returns JSON string."""
    try:
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Document not found: {target}"})
        conn.execute("UPDATE documents SET pinned = 0 WHERE id = ?", (doc["id"],))
        conn.commit()

        # Write-through: update frontmatter for memory notes
        if project_root and doc.get("doc_type", "").startswith("memory"):
            _set_pin_frontmatter(project_root, doc["path"], False)

        return json.dumps({"status": "ok", "path": doc["path"], "pinned": False})
    except Exception as e:
        print(f"kb_unpin error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_usage(db: Database) -> str:
    """Get index health stats as structured JSON. Delegates to KnowledgeBase."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        stats = kb.get_index_stats()
        kb.close()

        # Add MCP-specific tool_count
        tool_count = len(mcp._tool_manager._tools)

        return json.dumps(
            {
                "docs": stats["documents"],
                "entities": stats["entities"],
                "facts": stats["facts"],
                "pinned": stats["pinned"],
                "date_range": stats["date_range"],
                "tool_count": tool_count,
            }
        )
    except Exception as e:
        print(f"kb_usage error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_entity_stale(
    db: Database,
    days: int = 30,
    entity_type: str | None = None,
) -> str:
    """Return entities not updated or mentioned within *days*. Delegates to KnowledgeBase."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        results = kb.get_stale_entities(days=days, entity_type=entity_type)
        kb.close()
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
        return _build_entity_result(db, entity_row["name"])
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


def handle_kb_person_create(
    db: Database,
    project_root: Path,
    name: str,
    *,
    role: str | None = None,
    email: str | None = None,
    team: str | None = None,
    reports_to: str | None = None,
    company: str | None = None,
    aliases: str | None = None,
) -> str:
    """Create a new person entity. Returns JSON string."""
    from kb.crud import EntityExistsError, create_entity

    try:
        metadata = {
            k: v
            for k, v in {
                "role": role,
                "email": email,
                "team": team,
                "reports_to": reports_to,
                "company": company,
            }.items()
            if v is not None
        }
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []

        result = create_entity(
            db, project_root, "person", name, metadata=metadata, aliases=alias_list
        )

        # Index the new file so it's immediately searchable and pinnable
        from kb.indexer import index_all

        index_all(db, None, project_root, memory_only=True, skip_seed=True)

        return json.dumps(result)
    except EntityExistsError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_person_create error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def _parse_meta_string(meta: str | None) -> dict[str, str]:
    """Parse a semicolon-separated 'key=value; key2=value2' metadata string.

    Semicolons delimit pairs; commas are allowed in values.
    Falls back to comma-delimited parsing when no semicolons are present
    and only one key=value pair exists (backward compat for simple cases).
    """
    if not meta:
        return {}
    result: dict[str, str] = {}
    # Use semicolons as the primary delimiter (supports commas in values)
    if ";" in meta:
        pairs = meta.split(";")
    elif meta.count("=") == 1:
        # Single key=value with no semicolons — treat the whole string as one pair
        pairs = [meta]
    else:
        # Multiple key=value pairs with no semicolons — fall back to comma split
        pairs = meta.split(",")
    for pair in pairs:
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        if key:
            result[key] = value.strip()
    return result


def handle_kb_person_edit(
    db: Database,
    project_root: Path,
    name: str,
    *,
    role: str | None = None,
    email: str | None = None,
    team: str | None = None,
    reports_to: str | None = None,
    company: str | None = None,
    aliases: str | None = None,
    meta: str | None = None,
) -> str:
    """Edit an existing person's metadata. Returns JSON string."""
    from kb.crud import EntityNotFoundError, edit_entity

    try:
        metadata: dict[str, Any] = {
            k: v
            for k, v in {
                "role": role,
                "email": email,
                "team": team,
                "reports_to": reports_to,
                "company": company,
            }.items()
            if v is not None
        }
        metadata.update(_parse_meta_string(meta))
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else None

        result = edit_entity(db, project_root, name, metadata=metadata or None, aliases=alias_list)
        return json.dumps(result)
    except EntityNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_person_edit error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_project_create(
    db: Database,
    project_root: Path,
    name: str,
    *,
    status: str | None = None,
    lead: str | None = None,
    started: str | None = None,
    aliases: str | None = None,
) -> str:
    """Create a new project entity. Returns JSON string."""
    from kb.crud import EntityExistsError, create_entity

    try:
        metadata = {
            k: v
            for k, v in {
                "status": status,
                "lead": lead,
                "started": started,
            }.items()
            if v is not None
        }
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []

        result = create_entity(
            db, project_root, "project", name, metadata=metadata, aliases=alias_list
        )

        # Index the new file so it's immediately searchable and pinnable
        from kb.indexer import index_all

        index_all(db, None, project_root, memory_only=True, skip_seed=True)

        return json.dumps(result)
    except EntityExistsError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_project_create error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_project_edit(
    db: Database,
    project_root: Path,
    name: str,
    *,
    status: str | None = None,
    lead: str | None = None,
    started: str | None = None,
    aliases: str | None = None,
    meta: str | None = None,
) -> str:
    """Edit an existing project's metadata. Returns JSON string."""
    from kb.crud import EntityNotFoundError, edit_entity

    try:
        metadata: dict[str, Any] = {
            k: v
            for k, v in {
                "status": status,
                "lead": lead,
                "started": started,
            }.items()
            if v is not None
        }
        metadata.update(_parse_meta_string(meta))
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else None

        result = edit_entity(db, project_root, name, metadata=metadata or None, aliases=alias_list)
        return json.dumps(result)
    except EntityNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        print(f"kb_project_edit error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def _entity_list(db: Database, entity_type: str, limit: int = 50, offset: int = 0) -> str:
    """List entities via KnowledgeBase.list_entities(). Returns JSON string."""
    from pathlib import Path as _Path

    from kb.api import KnowledgeBase

    kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
    all_entities = kb.list_entities(entity_type=entity_type)
    kb.close()
    total = len(all_entities)
    page = all_entities[offset : offset + limit]
    entities = [
        {
            "id": e.id,
            "name": e.name,
            "entity_type": e.entity_type,
            "aliases": e.aliases,
            "metadata": e.metadata,
        }
        for e in page
    ]
    return json.dumps(
        {"results": entities, "meta": {"total": total, "limit": limit, "offset": offset}},
        default=str,
        ensure_ascii=False,
    )


def _build_entity_result(db: Database, name: str) -> str:
    """Build a compact entity result via KnowledgeBase.get_entity_profile(). Returns JSON string."""
    from pathlib import Path as _Path

    from kb.api import KnowledgeBase

    kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
    profile = kb.get_entity_profile(name)
    kb.close()
    if profile is None:
        return json.dumps({"error": f"Entity not found: {name}"})
    return json.dumps(profile, default=str, ensure_ascii=False)


def handle_kb_note_list(
    db: Database,
    tag: str | None = None,
    pinned_only: bool = False,
    limit: int = 25,
) -> str:
    """List notes with optional tag/pin filters. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        results, total = kb.list_notes(tag=tag, pinned_only=pinned_only, limit=limit)
        kb.close()
        return json.dumps(
            {"results": results, "meta": {"total": total, "limit": limit}},
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
    """Edit a note's body, tags, or pin status. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            result = kb.edit_note(target, body=body, append=append, tags=tag_list, pin=pin)
            return json.dumps(result, default=str, ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        finally:
            kb.close()
    except Exception as e:
        print(f"kb_note_edit error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_note_delete(db: Database, project_root: Path, target: str) -> str:
    """Delete a memory note. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from kb.api import KnowledgeBase

        # Resolve target to path first (KnowledgeBase.delete_note takes a path)
        conn = db.get_sqlite_conn()
        doc = _find_document_by_target(conn, target)
        if doc is None:
            return json.dumps({"error": f"Note not found: {target}"})

        kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
        try:
            result = kb.delete_note(doc["path"])
            return json.dumps(
                {"status": "ok", "path": result["path"], "title": result["title"]},
                default=str,
                ensure_ascii=False,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        finally:
            kb.close()
    except Exception as e:
        print(f"kb_note_delete error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


def handle_kb_memory_list(
    db: Database, project_root: Path, since_days: int | None = None, entity: str | None = None
) -> str:
    """List recorded facts. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
        try:
            facts = kb.list_facts(since_days=since_days, entity=entity)
            return json.dumps(
                {"results": facts, "meta": {"total": len(facts)}},
                default=str,
                ensure_ascii=False,
            )
        finally:
            kb.close()
    except Exception as e:
        print(f"kb_memory_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_memory_delete_fact(
    db: Database, project_root: Path, entity: str, fact_seq: int
) -> str:
    """Delete a fact by entity name + seq. Delegates to KnowledgeBase. Returns JSON string."""
    try:
        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
        try:
            result = kb.delete_fact(entity, fact_seq)
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
    db: Database,
    project_root: Path,
    entity: str,
    fact_seq: int,
    text: str | None = None,
    date: str | None = None,
) -> str:
    """Edit a fact's text or date by entity + seq. Delegates to KnowledgeBase."""
    try:
        if text is None and date is None:
            return json.dumps({"error": "Specify text and/or date to edit."})

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
        try:
            result = kb.edit_fact(entity, fact_seq, text=text, date=date)
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


def handle_kb_granola_view(calendar_uid: str, mode: str | None = None) -> str:
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


def handle_kb_list(
    db: Database,
    doc_type: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 25,
    since_hours: int | None = None,
) -> str:
    """Browse documents by date/type. Delegates to KnowledgeBase."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        results, total = kb.list_documents(
            doc_type=doc_type,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            since_hours=since_hours,
        )
        kb.close()
        return json.dumps(
            {"results": results, "meta": {"total": total, "limit": limit}},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"kb_list error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e), "results": []})


def handle_kb_index_status(db: Database) -> str:
    """Database health: doc counts, entity counts, freshness. Delegates to KnowledgeBase."""
    try:
        from pathlib import Path as _Path

        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=db, project_root=_Path("."))
        stats = kb.get_index_stats()
        kb.close()
        return json.dumps(stats, default=str, ensure_ascii=False)
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
    """Find (and optionally replace) a term across memory files. Delegates to KnowledgeBase."""
    try:
        from kb.api import KnowledgeBase

        kb = KnowledgeBase._from_existing(db=get_db(), project_root=project_root)
        try:
            result = kb.correct_term(
                term,
                replacement,
                apply=apply,
                scope=scope,
                file_type=file_type,
                word_boundary=word_boundary,
                ignore_case=ignore_case,
            )
            return json.dumps(result, default=str, ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        finally:
            kb.close()
    except Exception as e:
        print(f"kb_correct error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# FastMCP server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("kbx")

# Tool annotation presets
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
_MUTATING_IDEMPOTENT = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True
)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str | None) -> str | None:
    """Validate YYYY-MM-DD format. Returns the value or None if invalid."""
    if value is None:
        return None
    if not _DATE_RE.match(value):
        return None
    return value


@mcp.tool(annotations=_READ_ONLY)
def kb_search(
    query: str,
    fast: bool = True,
    limit: int = 5,
    from_date: str | None = None,
    to_date: str | None = None,
    tag: str | None = None,
    sort_by: str = "score",
    doc_type: str | None = None,
    path: str | None = None,
    explain: bool = False,
) -> str:
    """Search the knowledge base. Returns JSON with matching documents, ranked by relevance.
    Use fast=True (default) for instant FTS-only search.
    Optionally filter by date range (YYYY-MM-DD).
    Optionally filter by tag (comma-separated for AND, e.g. 'decision,infra').
    Note: tags only work on memory notes, not meeting docs.
    doc_type: filter by document type (e.g. 'notes', 'transcript', 'memory_note').
    path: scope results to a path prefix or glob (e.g. 'memory/meetings/', 'memory/*/2026/*').
    Applies to both FTS and vector search consistently.
    explain: attach scoring breakdown (FTS/vector/fused/recency component scores) to each
    result and diagnostic meta. Issue #68 Phase 1 — raw data only; human-readable summary later.
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
        path_filter=path,
        explain=explain,
    )


@mcp.tool(annotations=_READ_ONLY)
def kb_person_find(name: str) -> str:
    """Look up a person profile — compact output with facts, metadata, and breadcrumbs.
    Supports exact name, alias, or partial match."""
    db = get_db()
    return handle_kb_person_find(db, name)


@mcp.tool(annotations=_READ_ONLY)
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


@mcp.tool(annotations=_READ_ONLY)
def kb_view(target: str) -> str:
    """View a specific document by path or content-hash prefix (#abc123).
    Returns document metadata and all chunks."""
    db = get_db()
    return handle_kb_view(db, target)


@mcp.tool(annotations=_READ_ONLY)
def kb_context(topic: str | None = None, fmt: str = "compact", mention_threshold: int = 0) -> str:
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


@mcp.tool(annotations=_READ_ONLY)
def kb_usage() -> str:
    """Get index health stats: document/entity/fact/pinned counts, date range, and tool count.
    Returns structured JSON — use individual tool docstrings for usage reference."""
    db = get_db()
    return handle_kb_usage(db)


@mcp.tool(annotations=_MUTATING)
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


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
def kb_pin(target: str) -> str:
    """Pin a document to context so it appears in kb_context output.
    Accepts path, title, or glob pattern."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_pin(db, target, project_root)


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
def kb_unpin(target: str) -> str:
    """Unpin a document from context.
    Accepts path, title, or glob pattern."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_unpin(db, target, project_root)


@mcp.tool(annotations=_READ_ONLY)
def kb_entity_stale(days: int = 30, entity_type: str | None = None) -> str:
    """List entities not updated or mentioned within a threshold.
    Returns stale entities sorted by stalest first.
    days: threshold in days (default 30).
    entity_type: optional filter ('person', 'project')."""
    days = max(1, days)
    db = get_db()
    return handle_kb_entity_stale(db, days=days, entity_type=entity_type)


@mcp.tool(annotations=_READ_ONLY)
def kb_project_find(name: str) -> str:
    """Look up a project profile — compact output with facts, metadata, and breadcrumbs.
    Supports exact name, alias, or partial match."""
    db = get_db()
    return handle_kb_project_find(db, name)


@mcp.tool(annotations=_READ_ONLY)
def kb_project_list(limit: int = 50, offset: int = 0) -> str:
    """List all known projects with their metadata.
    limit: max results (default 50).
    offset: skip first N results for pagination."""
    limit = max(1, min(limit, 500))
    db = get_db()
    return handle_kb_project_list(db, limit=limit, offset=offset)


@mcp.tool(annotations=_READ_ONLY)
def kb_person_list(limit: int = 50, offset: int = 0) -> str:
    """List all known people with their metadata.
    limit: max results (default 50).
    offset: skip first N results for pagination."""
    limit = max(1, min(limit, 500))
    db = get_db()
    return handle_kb_person_list(db, limit=limit, offset=offset)


@mcp.tool(annotations=_MUTATING)
def kb_person_create(
    name: str,
    role: str | None = None,
    email: str | None = None,
    team: str | None = None,
    reports_to: str | None = None,
    company: str | None = None,
    aliases: str | None = None,
) -> str:
    """Create a new person entity with a markdown file in memory/people/.
    name: full name (e.g. 'Jane Doe').
    role, email, team, reports_to, company: optional metadata fields.
    aliases: comma-separated alternative names (e.g. 'Jane,JD')."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_person_create(
        db,
        project_root,
        name,
        role=role,
        email=email,
        team=team,
        reports_to=reports_to,
        company=company,
        aliases=aliases,
    )


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
def kb_person_edit(
    name: str,
    role: str | None = None,
    email: str | None = None,
    team: str | None = None,
    reports_to: str | None = None,
    company: str | None = None,
    aliases: str | None = None,
    meta: str | None = None,
) -> str:
    """Edit an existing person's metadata. Preserves freeform content.
    name: person to edit (exact name, alias, or partial match).
    role, email, team, reports_to, company: standard fields (set to update).
    aliases: comma-separated names to add (e.g. 'Jane,JD').
    meta: comma-separated key=value pairs for arbitrary metadata (e.g. 'timezone=CET,lang=FR').
    Set a value to empty string to remove a key (e.g. 'timezone=')."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_person_edit(
        db,
        project_root,
        name,
        role=role,
        email=email,
        team=team,
        reports_to=reports_to,
        company=company,
        aliases=aliases,
        meta=meta,
    )


@mcp.tool(annotations=_MUTATING)
def kb_project_create(
    name: str,
    status: str | None = None,
    lead: str | None = None,
    started: str | None = None,
    aliases: str | None = None,
) -> str:
    """Create a new project entity with a markdown file in memory/projects/.
    name: project name (e.g. 'API Redesign').
    status, lead, started: optional metadata fields.
    aliases: comma-separated alternative names (e.g. 'api-v2,redesign')."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_project_create(
        db,
        project_root,
        name,
        status=status,
        lead=lead,
        started=started,
        aliases=aliases,
    )


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
def kb_project_edit(
    name: str,
    status: str | None = None,
    lead: str | None = None,
    started: str | None = None,
    aliases: str | None = None,
    meta: str | None = None,
) -> str:
    """Edit an existing project's metadata. Preserves freeform content.
    name: project to edit (exact name, alias, or partial match).
    status, lead, started: standard fields (set to update).
    aliases: comma-separated names to add (e.g. 'api-v2,redesign').
    meta: comma-separated key=value pairs for arbitrary metadata (e.g. 'priority=High').
    Set a value to empty string to remove a key (e.g. 'priority=')."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_project_edit(
        db,
        project_root,
        name,
        status=status,
        lead=lead,
        started=started,
        aliases=aliases,
        meta=meta,
    )


@mcp.tool(annotations=_READ_ONLY)
def kb_note_list(tag: str | None = None, pinned_only: bool = False, limit: int = 25) -> str:
    """Browse memory notes with optional filtering.
    tag: comma-separated tags (AND filter).
    pinned_only: only return pinned notes.
    limit: max results (default 25)."""
    limit = max(1, min(limit, 100))
    db = get_db()
    return handle_kb_note_list(db, tag=tag, pinned_only=pinned_only, limit=limit)


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
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


@mcp.tool(annotations=_DESTRUCTIVE)
def kb_note_delete(target: str) -> str:
    """Delete a memory note (file + index entry).
    target: note path, title, or glob."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_note_delete(db, project_root, target)


@mcp.tool(annotations=_READ_ONLY)
def kb_memory_list(since_days: int | None = None, entity: str | None = None) -> str:
    """List recorded facts, newest first.
    since_days: optional filter to only show facts from last N days.
    entity: optional entity name to filter facts for a specific person/project."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_memory_list(db, project_root, since_days=since_days, entity=entity)


@mcp.tool(annotations=_DESTRUCTIVE)
def kb_memory_delete_fact(entity: str, fact_seq: int) -> str:
    """Delete a fact by entity name and sequence number.
    entity: the entity name (e.g. 'Idris Kalmar').
    fact_seq: the fact sequence number (from kb_person_find or kb_memory_list)."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_memory_delete_fact(db, project_root, entity, fact_seq)


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
def kb_memory_edit_fact(
    entity: str, fact_seq: int, text: str | None = None, date: str | None = None
) -> str:
    """Edit a fact's text or date by entity name and sequence number.
    entity: the entity name.
    fact_seq: the fact sequence number.
    text: new fact text (optional).
    date: new date in YYYY-MM-DD (optional)."""
    db = get_db()
    project_root = find_project_root()
    return handle_kb_memory_edit_fact(db, project_root, entity, fact_seq, text=text, date=date)


@mcp.tool(annotations=_READ_ONLY)
def kb_glossary_list() -> str:
    """List all glossary terms (acronyms and jargon)."""
    project_root = find_project_root()
    return handle_kb_glossary_list(project_root)


@mcp.tool(annotations=_MUTATING)
def kb_glossary_add(term: str, expansion: str, section: str = "Acronyms") -> str:
    """Add a term to the glossary.
    term: the abbreviation or term.
    expansion: what it stands for.
    section: glossary section heading (default 'Acronyms')."""
    project_root = find_project_root()
    return handle_kb_glossary_add(project_root, term, expansion, section=section)


@mcp.tool(annotations=_MUTATING_IDEMPOTENT)
def kb_glossary_edit(term: str, expansion: str) -> str:
    """Update an existing glossary term's expansion.
    term: the term to edit.
    expansion: the new expansion text."""
    project_root = find_project_root()
    return handle_kb_glossary_edit(project_root, term, expansion)


@mcp.tool(annotations=_READ_ONLY)
def kb_granola_view(calendar_uid: str, mode: str | None = None) -> str:
    """View meeting notes, AI summary, or transcript from Granola.
    calendar_uid: Google Calendar event ID or iCalUID.
    mode: 'summary', 'transcript', 'all', or None (notes only, default)."""
    if mode and mode not in ("summary", "transcript", "all"):
        mode = None
    return handle_kb_granola_view(calendar_uid, mode=mode)


@mcp.tool(annotations=_READ_ONLY)
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
        db,
        doc_type=doc_type,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        since_hours=since_hours,
    )


@mcp.tool(annotations=_READ_ONLY)
def kb_index_status() -> str:
    """Get database health: document counts by type, entity/fact counts, date range, last indexed timestamp."""
    db = get_db()
    return handle_kb_index_status(db)


@mcp.tool(annotations=_MUTATING)
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
