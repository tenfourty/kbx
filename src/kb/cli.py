"""CLI entry point for kbx — local knowledge base with hybrid search."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Strip SOCKS proxy env vars before any httpx imports.
#
# Some environments (e.g. Claude Code sandbox) set ALL_PROXY=socks5h://...
# which causes httpx to require the 'socksio' package.  kbx only needs
# plain HTTPS to reach Granola/Notion/HuggingFace APIs, so we strip the
# SOCKS-specific vars while leaving HTTP_PROXY/HTTPS_PROXY intact.
# ---------------------------------------------------------------------------
for _proxy_var in (
    "ALL_PROXY",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
    "GRPC_PROXY",
    "grpc_proxy",
    "RSYNC_PROXY",
):
    os.environ.pop(_proxy_var, None)

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
# Custom Group with enhanced --help
# ---------------------------------------------------------------------------


def _get_init_status() -> str:
    """Build a compact initialization status block for --help output."""
    import contextlib

    lines: list[str] = []
    with contextlib.suppress(Exception):
        from kb.user_config import find_config

        config_path = find_config()
        data_dir = _get_data_dir()
        lines.append(f"  Config:   {config_path or 'not found'}")
        lines.append(f"  Data dir: {data_dir}")

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
        lines.append(
            f"  Index:    {total_docs} docs | {total_entities} entities | "
            f"{total_facts} facts | {pinned_docs} pinned | "
            f"dates {earliest} to {latest}"
        )

    if not lines:
        lines.append("  Status:   not initialized (run 'kbx init')")

    return "\n".join(lines)


# _AGENT_PLAYBOOK anchor convention (issue #5):
# Sections use stable, conceptual '## <name> contract' headings (NOT numbered).
# These headings are the external-consumer contract surface — cross-repo hooks and
# skill prompts anchor to them BY NAME rather than inlining flag lists (mirrors
# guten-morgen's '## Scoping axes' anchor). Renaming a heading is a breaking change
# for those consumers, so treat the set as stable: add new sections freely, but do
# not casually rename existing ones. tests/test_cli.py::TestHelpOutput pins the set.
_AGENT_PLAYBOOK = """\

# kb — Agent Playbook

## Quick start contract
  kb context              # orient: pinned docs + entity index
  kb search "topic" --fast --json --limit 5   # keyword search (~instant)
  kb view <path>          # read a full document

## Memory contract
  kb memory add "title" --body "markdown content" --tags t1,t2 --pin
  kb memory add "Quick note"                     # one-liner, no body
  kb memory add "fact" --entity "Name"           # fact appended to entity file
  kb memory add "title" --body "..." --entity "Name"  # note linked to entity
  kb memory delete-fact <id>                         # delete a fact by ID
  kb memory edit-fact <id> --text "new text"         # update fact text
  kb memory edit-fact <id> --date 2026-03-01         # update fact date
  kb memory similar "text" --entity "Name" --json    # dedup check: cosine similar facts (#71)
  kb memory similar "text" --path memory/foo.md      # scope to one document's chunks
  kb memory similar "text" --threshold 0.80 --limit 5  # tune cutoff + cap

## Pinning contract
  kb pin <path|title|#hash|glob>     # pin any doc to context
  kb unpin <path|title|glob>         # remove from context
  kb memory add "title" --pin        # create + pin in one step
  kb person pin "Name"               # pin a person to context

## Notes contract
  kb note list --json                   # all notes
  kb note list --tag decision --json    # filter by tag (AND: --tag a,b)
  kb note list --pinned --json          # pinned notes only
  kb note list --tag ops --pinned       # combine filters
  kb note edit "title" --body "new content"  # replace body
  kb note edit "title" --append "extra"      # append to body
kb note edit "person.md" --insert-under "## Open Items" --body "- [2026-05-25] item (from: Meeting)"
                                            # section-anchored insertion, most-recent-first,
                                            # creates section if absent, idempotent on duplicate
  kb note edit "title" --tags "a,b,c"        # replace tags
  kb note edit "title" --pin                 # pin note
  kb note edit "title" --unpin               # unpin note
  kb note edit "#hash" --tags "x" --pin      # combine edits
  kb note delete "title"                     # delete note (file + index)
  kb note delete "#hash" --force             # skip confirmation

## Search contract
  kb search "query" --fast --json             # keyword (FTS, instant)
  kb search "query" --json --limit 10         # hybrid (semantic + FTS, ~2s)
  kb search "query" --tag infra --fast --json # filter by tag
  kb search "query" --path memory/meetings/    # scope to path prefix (FTS + vector)
  kb search "query" --path "memory/*/2026/*"   # scope with glob (* and ? supported)
  kb search "query" --from 2026-01-01 --to 2026-01-31  # date range
  kb search "query" --sort date --json        # newest first
  kb search "query" --dedupe --json           # one result per document
  kb search "query" --full-chunks --json      # include full chunk text in results
  kb search "query" --dedupe --full-chunks --merge-chunks --json  # merge all chunks per doc
  kb search "query" --snippet-chars 500 --json  # longer snippets (default 200)
  kb search "query" --fts-weight 2.0 --json   # boost FTS (keywords)
  kb search "query" --vector-weight 2.0 --json  # boost vector (semantic)
  kb search "query" --explain                  # readable per-result scoring breakdown (#68 P2)
  kb search "query" --explain --json           # raw structured explain fields (#68 P1)
  kb search "query" --explain --verbose        # + per-phase timing breakdown (#3)
  kb search "query" --why-not PATH             # why a specific doc didn't surface (#3)
  kb search "query" --no-hotness --json        # disable hotness blend (#67)
  kb search "query" --no-hierarchy --json      # force flat (skip Pass 1 entity match, #69)
  kb search "query" --hierarchy-alpha 0.7 --json  # tune doc-vs-entity blend (#69)
  kb access reset                              # clear all hotness access counts
  # JSON results include `abstract`: extractive one-sentence summary per doc (#66 P1)
  kb search "query" --include-overview --json  # also include L1 paragraph overview (#66 P4, ~500 chars)
  kb view <path|#hash|glob> --json            # full document
  kb view <path|#hash|glob> --plain           # raw content only (no metadata)
  kb list --type notes --from 2026-02-01      # browse by type/date

  Score Interpretation: 0.8+ strong | 0.5-0.8 worth reading | <0.5 noise
  JSON output: snippets are plain text (no HTML). Table output adds <mark> highlighting.
  --full-chunks adds a "content" field with the complete chunk text for agent triage.
  --merge-chunks (with --dedupe --full-chunks) concatenates all matching chunks per document.

## People & projects contract
  kb me --json                       # your own profile (shortcut for person find me)
  kb person find "Name" --json       # compact profile (facts, metadata, breadcrumbs)
  kb person timeline "Name" --json   # chronological doc list
  kb person create "Name" --role "Role" --team "Team"
  kb person edit "Name" --role "New Role"
  kb person edit "Name" --meta "preferred_lang=French" --meta "timezone=CET"
  kb person edit "Name" --meta "timezone="   # remove a custom field
  kb person list --json              # all people
  kb project find "Name" --json      # compact profile (facts, metadata, breadcrumbs)
  kb project create "Name" --status "Active" --lead "Name"
  kb project edit "Name" --meta "priority=High"
  kb project list --json             # all projects
  kb entity stale --json               # entities not updated/mentioned in 30+ days
  kb entity stale --days 60 --json     # custom threshold
  kb entity stale --type person --json # filter by type

## Glossary contract
  kb glossary add "TERM" "expansion"
  kb glossary edit "TERM" "new expansion"  # update existing term
  kb glossary list --json

## Context contract
  kb context                         # compact entity index (for agents)
  kb context --human                 # markdown format (for humans)
  kb context --for "topic"           # filtered to a topic

## Index contract
  kb index status --json             # database health
  kb index run --no-embed            # text-only index (fast, no GPU)
  kb index run --cpu                 # full index with embeddings on CPU

## Sync contract
  kb sync                              # run all sync plugins
  kb sync granola --since 2026-01-01   # sync Granola meetings since date
  kb sync notion --since 2026-01-01    # sync Notion AI Meeting Notes since date
  Options: --dry-run | --force | --no-index

  Granola sync produces three files per meeting:
    .granola.notes.md        — human-written notes
    .granola.transcript.md   — full transcript
    .granola.ai-summary.md   — AI-generated summary (from Granola's summary panel)

## Granola contract
  kb granola view <calendar-uid>                                  # view notes (YAML header + markdown)
  kb granola view <calendar-uid> --summary                        # view AI summary
  kb granola view <calendar-uid> --transcript                     # view transcript (live from API)
  kb granola view <calendar-uid> --all                            # view notes + summary + transcript
  kb granola view <calendar-uid> --plain                          # raw markdown only (no header)

## Corrections contract
  # find-and-replace across memory
  kb correct "Quartz Indexer" --json                          # scan: list all occurrences (structured)
  kb correct "Quartz Indexer"                                 # scan: human-readable summary
  kb correct "Bram" --word-boundary --json               # scan: whole-word matches only
  kb correct "Quartz Indexer" --type transcript --json        # scan: filter by file type
  kb correct "Quartz Indexer" --scope "**/people/*" --json    # scan: filter by path glob
  kb correct "Quartz Indexer" "Datalux"                     # dry-run: preview replacements
  kb correct "Quartz Indexer" "Datalux" --apply             # apply: execute replacements
  kb correct "Quartz Indexer" "Datalux" --apply --json      # apply: structured result
  kb correct "Brahm" "Bram" --word-boundary --scope "meetings/2026/02/17/*" --apply
  kb correct "datalux" "Datalux" --ignore-case --apply  # case-insensitive replace

  JSON scan output includes title, date, attendees, category per match — agent-ready
  for disambiguation decisions (e.g. which "Bram" is in this meeting).
  Replacements are atomic (temp file → rename). Only .md files are modified.

## Python API contract

    from kb import KnowledgeBase

    kb = KnowledgeBase()                          # auto-discover config
    kb = KnowledgeBase(thread_safe=True)           # for multi-threaded apps (FastAPI etc.)
    kb.search("query")                             # -> SearchResponse
    kb.list_entities(entity_type="person")         # -> list[EntitySummary]
    kb.get_entity("name")                          # -> EntityDetail | None
    kb.find_entities("name")                       # -> list[EntitySummary]
    kb.get_entity_timeline("name")                 # -> list[TimelineEntry]
    kb.get_stale_entities(days=30)                 # -> list[dict]
    kb.context()                                   # -> ContextOutput
    kb.index()                                     # -> IndexResult
    kb.toggle_entity_pin("name")                   # -> EntityPinResult
    kb.toggle_document_pin("path")                 # -> DocumentPinResult
    kb.read_memory_file("notes/foo.md")            # -> str | None
    kb.list_memory_tree()                          # -> list[MemoryTreeNode]
    kb.delete_note("notes/2026-02-28-test.md")     # -> dict (removes file + index)
    kb.list_glossary_terms()                       # -> list[GlossaryEntry]
    kb.edit_glossary_term("TERM", "new expansion") # -> dict
    kb.delete_fact(fact_id)                        # -> dict (removes from DB + file)
    kb.edit_fact(fact_id, text="new text")          # -> dict (updates DB + file)
    kb.memory_similar("text", entity="Name")       # -> SimilarityResponse (#71)
    kb.close()                                     # release resources

## Global options contract
  --json | --format table|json|jsonl|csv | --fields f1,f2 | --jq expr
"""


class KbxGroup(click.Group):
    """Custom Group that adds init status and agent playbook to --help."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Override to inject init status before and agent playbook after help."""
        formatter.write(_get_init_status())
        formatter.write("\n\n")
        super().format_help(ctx, formatter)
        formatter.write(_AGENT_PLAYBOOK)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group(cls=KbxGroup)
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
@click.option(
    "--path",
    "path_filter",
    default=None,
    help=(
        "Scope results to a path prefix or glob (e.g. memory/meetings/, memory/*/2026/*). "
        "Applies to both FTS and vector search."
    ),
)
@click.option(
    "--fts-weight", default=1.0, type=float, help="FTS weight for RRF fusion (default 1.0)."
)
@click.option(
    "--vector-weight", default=1.0, type=float, help="Vector weight for RRF fusion (default 1.0)."
)
@click.option("--dedupe", is_flag=True, help="Keep only the best chunk per document.")
@click.option(
    "--snippet-chars", default=200, type=int, help="Max snippet length in characters (default 200)."
)
@click.option(
    "--full-chunks", is_flag=True, help="Include full chunk text as 'content' in JSON results."
)
@click.option(
    "--merge-chunks",
    is_flag=True,
    help="With --dedupe --full-chunks, concatenate all matching chunks per document.",
)
@click.option(
    "--include-overview",
    is_flag=True,
    help=(
        "Include the extractive L1 overview (paragraph-length summary, ~500 chars) "
        "on each result. Off by default to keep search output compact (#66 P4)."
    ),
)
@click.option(
    "--explain",
    is_flag=True,
    help=(
        "Attach scoring breakdown to each result (FTS/vector/fused/recency component "
        "scores, source pipeline) and diagnostic meta. Table format renders a readable "
        "per-result breakdown; pass --format json for the raw structured fields."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    help=(
        "With --explain, add a per-phase timing breakdown (FTS / vector / merge) to the "
        "meta header. No effect without --explain."
    ),
)
@click.option(
    "--why-not",
    "why_not",
    default=None,
    metavar="PATH",
    help=(
        "Explain why a specific document did or didn't surface for the query "
        "(ranked / below-cutoff / no-match / not-indexed / filtered). Implies --explain output."
    ),
)
@click.option(
    "--no-hotness",
    is_flag=True,
    help=(
        "Disable hotness blending (#67). By default, frequently-viewed docs surface "
        "higher; pass --no-hotness for cold/historical queries that should ignore "
        "access history."
    ),
)
@click.option(
    "--no-hierarchy",
    is_flag=True,
    help=(
        "Disable two-pass hierarchical search (#69) — Pass 1 entity match + Pass 2 "
        "doc constraint. With --no-hierarchy the search is purely flat across all "
        "documents regardless of entity relevance."
    ),
)
@click.option(
    "--hierarchy-alpha",
    default=0.5,
    type=float,
    help=(
        "When hierarchy is active, the blend weight on the doc score vs. the "
        "parent-entity score (0.0 = pure entity proximity, 1.0 = pure doc relevance)."
    ),
)
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
    path_filter: str | None,
    fts_weight: float,
    vector_weight: float,
    dedupe: bool,
    snippet_chars: int,
    full_chunks: bool,
    merge_chunks: bool,
    include_overview: bool,
    explain: bool,
    verbose: bool,
    why_not: str | None,
    no_hotness: bool,
    no_hierarchy: bool,
    hierarchy_alpha: float,
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
            if fmt == "table":
                print(f"Embedding backend: {embedder.backend_name}", file=sys.stderr)
        except Exception:
            click.echo("Warning: Could not load embedding model. Using FTS-only search.", err=True)
            fast = True

    db = _get_db()
    if why_not:
        explain = True  # why-not renders through the explain formatter
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
        path_filter=path_filter,
        sort_by=sort_by,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
        dedupe=dedupe,
        snippet_chars=snippet_chars,
        full_chunks=full_chunks,
        merge_chunks=merge_chunks,
        include_overview=include_overview,
        highlight=(fmt == "table"),
        explain=explain,
        verbose=verbose,
        why_not=why_not,
        hotness=not no_hotness,
        hierarchy=not no_hierarchy,
        hierarchy_alpha=hierarchy_alpha,
    )

    if explain and fmt == "table" and not fields and not jq_expr:
        from kb.explain import format_explain_text

        click.echo(format_explain_text(results))
        return

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
@click.option("--plain", is_flag=True, help="Output raw file content only.")
@output_options
def view(target: str, plain: bool, fmt: str, fields: list[str] | None, jq_expr: str | None) -> None:
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

    if plain:
        click.echo("\n\n".join(content_parts))
        return

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


def _find_document_by_target(conn: sqlite3.Connection, target: str) -> dict[str, Any] | None:
    """Resolve a document by path, content hash, title, glob, or substring.

    Raises SystemExit(2) for ambiguous matches (CLI-specific error handling).
    """
    from kb.crud import AmbiguousDocumentError, find_document_by_target

    try:
        return find_document_by_target(conn, target, strict=True)
    except AmbiguousDocumentError as e:
        click.echo(f"Ambiguous target. Matches: {e.matches}", err=True)
        raise SystemExit(2) from None


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


def _build_entity_result(db: Any, name: str) -> dict[str, Any]:
    """Build a compact entity result via KnowledgeBase.get_entity_profile()."""
    from kb.api import KnowledgeBase

    kb = KnowledgeBase._from_existing(db=db, project_root=Path("."))
    profile = kb.get_entity_profile(name)
    kb.close()
    if profile is None:
        return {"error": f"Entity not found: {name}"}
    return profile


def _resolve_me(name: str) -> str:
    """Resolve 'me' to the configured user name."""
    from kb.user_config import resolve_me

    result = resolve_me(name)
    if result is None:
        click.echo("'me' requires [user] name in kbx.toml or ~/.config/kbx/config.toml", err=True)
        raise SystemExit(1)
    return result


def _format_entity_profile(entity: dict[str, Any]) -> str:
    """Format a single entity as a human-readable profile."""
    lines: list[str] = []
    name = entity["name"]
    etype = entity["entity_type"]
    aliases = entity.get("aliases", [])

    # Header (filter src: prefixed aliases from display)
    display_aliases = [a for a in aliases if not a.startswith("src:")]
    alias_str = f"  aka {', '.join(display_aliases)}" if display_aliases else ""
    lines.append(f"{name} ({etype}){alias_str}")
    lines.append("")

    # Metadata (excluding sources — shown separately)
    meta = entity.get("metadata", {})
    if meta:
        for key, val in meta.items():
            if key == "sources":
                continue
            lines.append(f"  {key}: {val}")
        lines.append("")

    # Sources (separate section for readability)
    from kb.matching import extract_sources, format_source

    sources = extract_sources(meta)
    if sources:
        lines.append("Sources:")
        for src in sources:
            lines.append(format_source(src))
        lines.append("")

    # Facts
    facts = entity.get("facts", [])
    if facts:
        lines.append("Facts:")
        for f in facts:
            d = f.get("date") or "?"
            lines.append(f"  - [{d}] {f['text']}")
        lines.append("")

    # Stats + breadcrumbs
    doc_count = entity.get("document_count", 0)
    lines.append(f"{doc_count} linked documents")
    breadcrumbs = entity.get("breadcrumbs", {})
    for val in breadcrumbs.values():
        lines.append(f"  -> {val}")

    return "\n".join(lines)


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
        result: Any = _build_entity_result(db, matches[0]["name"])
    else:
        result = {
            "results": [_build_entity_result(db, m["name"]) for m in matches],
            "meta": {
                "total": len(matches),
                "note": f"Multiple entities match '{name}'. Use a more specific name.",
            },
        }

    # Custom human-readable format for table output
    if fmt == "table" and not fields and not jq_expr:
        if isinstance(result, dict) and "results" not in result:
            click.echo(_format_entity_profile(result))
        elif isinstance(result, dict) and "results" in result:
            for i, entity in enumerate(result["results"]):
                if i > 0:
                    click.echo("---")
                click.echo(_format_entity_profile(entity))
        return

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


def _autocommit_after_write(
    operation: str, target: str, *, no_commit: bool = False, file_count: int = 1
) -> None:
    """Auto-commit memory changes after a CLI write (issue #1). Best-effort, never fatal."""
    from kb.autocommit import maybe_auto_commit
    from kb.user_config import KbxConfig, find_config, load_config

    config_path = find_config()
    cfg = load_config(config_path) if config_path else KbxConfig()
    sha = maybe_auto_commit(
        _find_project_root(),
        operation=operation,
        target=target,
        config=cfg.writes,
        no_commit=no_commit,
        file_count=file_count,
    )
    if sha:
        click.echo(f"  auto-committed {sha[:8]} ({operation})", err=True)


def _commit_option(f: Callable[..., Any]) -> Callable[..., Any]:
    """Reusable --no-commit flag for write commands (issue #1)."""
    return click.option(
        "--no-commit",
        is_flag=True,
        help="Skip the git auto-commit even when [writes].auto_commit is enabled (#1).",
    )(f)


def _entity_create_impl(
    entity_type: str,
    name: str,
    metadata: dict[str, str],
    aliases: list[str],
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
    no_commit: bool = False,
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
    _autocommit_after_write(f"create-{entity_type}", name, no_commit=no_commit)
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _merge_meta_pairs(metadata: dict[str, str], meta_pairs: tuple[str, ...]) -> None:
    """Parse --meta 'key=value' pairs into the metadata dict.

    Empty value (e.g. 'key=') is preserved as '' to signal deletion.
    """
    for pair in meta_pairs:
        if "=" not in pair:
            click.echo(f"Invalid --meta format (expected key=value): {pair}", err=True)
            raise SystemExit(1)
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            click.echo(f"Invalid --meta format (empty key): {pair}", err=True)
            raise SystemExit(1)
        # Normalize to snake_case
        key = "_".join(key.lower().split())
        metadata[key] = value


def _parse_source_args(source_args: tuple[str, ...]) -> list[dict[str, str]]:
    """Parse multiple --source arguments into source dicts."""
    from kb.matching import parse_source_arg

    sources: list[dict[str, str]] = []
    for raw in source_args:
        try:
            sources.append(parse_source_arg(raw))
        except ValueError as e:
            click.echo(str(e), err=True)
            raise SystemExit(1) from None
    return sources


def _entity_edit_impl(
    name: str,
    metadata: dict[str, str],
    aliases: list[str],
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
    no_commit: bool = False,
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
    _autocommit_after_write(
        f"edit-{result.get('entity_type', 'entity')}", name, no_commit=no_commit
    )
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


def _entity_delete_impl(name: str, no_commit: bool = False) -> None:
    """Delete an entity via CRUD engine."""
    from kb.crud import EntityNotFoundError, delete_entity

    try:
        result = delete_entity(_get_db(), _find_project_root(), name)
    except EntityNotFoundError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    _autocommit_after_write(
        f"delete-{result.get('entity_type', 'entity')}", str(result["name"]), no_commit=no_commit
    )
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
@_commit_option
def person_create(
    name: str,
    role: str | None,
    email: str | None,
    team: str | None,
    aliases: tuple[str, ...],
    reports_to: str | None,
    company: str | None,
    no_commit: bool,
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
    _entity_create_impl(
        "person", name, metadata, list(aliases), fmt, fields, jq_expr, no_commit=no_commit
    )


@person.command("edit")
@click.argument("name")
@click.option("--role", default=None)
@click.option("--email", default=None)
@click.option("--team", default=None)
@click.option("--alias", "aliases", multiple=True)
@click.option("--reports-to", default=None)
@click.option("--company", default=None)
@click.option("--meta", "meta_pairs", multiple=True, help="Custom key=value metadata.")
@output_options
@_commit_option
def person_edit(
    name: str,
    role: str | None,
    email: str | None,
    team: str | None,
    aliases: tuple[str, ...],
    reports_to: str | None,
    company: str | None,
    meta_pairs: tuple[str, ...],
    no_commit: bool,
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
    _merge_meta_pairs(metadata, meta_pairs)
    _entity_edit_impl(name, metadata, list(aliases), fmt, fields, jq_expr, no_commit=no_commit)


@person.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this person?")
@_commit_option
def person_delete(name: str, no_commit: bool) -> None:
    """Delete a person (file + database)."""
    _entity_delete_impl(name, no_commit=no_commit)


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
@click.option(
    "--source", "source_args", multiple=True, help="External source (TYPE:URL or TYPE:key=val,...)."
)
@output_options
@_commit_option
def project_create(
    name: str,
    status: str | None,
    lead: str | None,
    codename: str | None,
    started: str | None,
    source_args: tuple[str, ...],
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Create a new project."""
    metadata: dict[str, Any] = {
        k: v for k, v in {"status": status, "lead": lead, "started": started}.items() if v
    }
    if source_args:
        metadata["sources"] = _parse_source_args(source_args)
    aliases = [codename] if codename else []
    _entity_create_impl(
        "project", name, metadata, aliases, fmt, fields, jq_expr, no_commit=no_commit
    )


@project.command("edit")
@click.argument("name")
@click.option("--status", default=None)
@click.option("--lead", default=None)
@click.option("--codename", default=None)
@click.option("--started", default=None, help="Start date (YYYY-MM).")
@click.option("--meta", "meta_pairs", multiple=True, help="Custom key=value metadata.")
@click.option(
    "--source",
    "source_args",
    multiple=True,
    help="Add source (TYPE:URL or TYPE:key=val,...). Appends.",
)
@click.option(
    "--remove-source", "remove_sources", multiple=True, help="Remove source by TYPE or TYPE:ID."
)
@output_options
@_commit_option
def project_edit(
    name: str,
    status: str | None,
    lead: str | None,
    codename: str | None,
    started: str | None,
    meta_pairs: tuple[str, ...],
    source_args: tuple[str, ...],
    remove_sources: tuple[str, ...],
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Edit a project's metadata."""
    metadata: dict[str, Any] = {
        k: v for k, v in {"status": status, "lead": lead, "started": started}.items() if v
    }
    _merge_meta_pairs(metadata, meta_pairs)
    if source_args or remove_sources:
        metadata["__source_add"] = _parse_source_args(source_args) if source_args else []
        metadata["__remove_sources"] = list(remove_sources) if remove_sources else []
    aliases = [codename] if codename else []
    _entity_edit_impl(name, metadata, aliases, fmt, fields, jq_expr, no_commit=no_commit)


@project.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this project?")
@_commit_option
def project_delete(name: str, no_commit: bool) -> None:
    """Delete a project (file + database)."""
    _entity_delete_impl(name, no_commit=no_commit)


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
# entity group
# ---------------------------------------------------------------------------


@cli.group()
def entity() -> None:
    """Entity commands — stale detection."""
    pass


@entity.command("stale")
@click.option("--days", default=30, type=click.IntRange(min=1), help="Freshness threshold in days.")
@click.option("--type", "entity_type", default=None, help="Filter by entity type (person/project).")
@output_options
def entity_stale(
    days: int,
    entity_type: str | None,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """List entities that haven't been updated or mentioned recently."""
    from kb.api import KnowledgeBase

    kb = KnowledgeBase(data_dir=_get_data_dir(), project_root=_find_project_root())
    results = kb.get_stale_entities(days=days, entity_type=entity_type)
    kb_output(
        {"results": results, "meta": {"count": len(results), "threshold_days": days}},
        fmt=fmt,
        fields=fields,
        jq_expr=jq_expr,
    )


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
@_commit_option
def glossary_add(
    term: str,
    expansion: str,
    section: str,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Add a term to the glossary."""
    from kb.glossary import add_term

    result = add_term(_find_project_root(), term, expansion, section=section)
    _autocommit_after_write("add-glossary", term, no_commit=no_commit)
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
@_commit_option
def glossary_delete(term: str, no_commit: bool) -> None:
    """Delete a term from the glossary."""
    from kb.glossary import delete_term

    try:
        delete_term(_find_project_root(), term)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    _autocommit_after_write("delete-glossary", term, no_commit=no_commit)
    click.echo(f"Deleted: {term}")


@glossary.command("edit")
@click.argument("term")
@click.argument("expansion")
@output_options
@_commit_option
def glossary_edit(
    term: str,
    expansion: str,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Update an existing glossary term's expansion."""
    from kb.glossary import edit_term

    try:
        result = edit_term(_find_project_root(), term, expansion)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    _autocommit_after_write("edit-glossary", term, no_commit=no_commit)
    kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)


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
# correct
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("term")
@click.argument("replacement", required=False, default=None)
@click.option("--apply", "apply_flag", is_flag=True, help="Apply corrections (default is dry-run).")
@click.option("--scope", default=None, help="Glob or path to limit scope (e.g. **/meetings/*).")
@click.option(
    "--type", "file_type", default=None, help="Filter by filename pattern (e.g. transcript)."
)
@click.option("--word-boundary", is_flag=True, help="Only match whole words.")
@click.option("--ignore-case", is_flag=True, help="Case-insensitive matching.")
@click.option(
    "--no-commit",
    is_flag=True,
    help="Skip the git auto-commit even when [writes].auto_commit is enabled (#1).",
)
@output_options
def correct(
    term: str,
    replacement: str | None,
    apply_flag: bool,
    scope: str | None,
    file_type: str | None,
    word_boundary: bool,
    ignore_case: bool,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Find and replace a term across all memory files.

    \b
    Scan mode (no REPLACEMENT):
      kb correct "Quartz Indexer"                         # list all occurrences
      kb correct "Quartz Indexer" --json                  # structured for agents
      kb correct "Bram" --word-boundary --json       # whole-word, agent-friendly

    \b
    Replace mode (with REPLACEMENT):
      kb correct "Quartz Indexer" "Datalux"             # dry-run preview
      kb correct "Quartz Indexer" "Datalux" --apply     # apply changes
      kb correct "Brahm" "Bram" --word-boundary --scope "meetings/2026/02/17/*" --apply

    \b
    With [writes].auto_commit enabled in kbx.toml, --apply makes one git commit of the
    change (audit trail + git revert). Pass --no-commit to skip it. (#1)
    """
    from kb.api import KnowledgeBase

    project_root = _find_project_root()
    kb = KnowledgeBase._from_existing(db=_get_db(), project_root=project_root)

    try:
        result: dict[str, Any] = kb.correct_term(
            term,
            replacement,
            apply=apply_flag,
            scope=scope,
            file_type=file_type,
            word_boundary=word_boundary,
            ignore_case=ignore_case,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None
    except OSError as e:
        # Batch write failed mid-apply; apply_corrections rolled the files back (#1 COW).
        click.echo(
            f"Error: correction failed and was rolled back — no files were changed ({e}).",
            err=True,
        )
        raise SystemExit(1) from None
    finally:
        kb.close()

    # Auto-commit applied changes (issue #1, CLI-only). Dry-run/scan writes nothing.
    if apply_flag and result.get("results"):
        from kb.autocommit import maybe_auto_commit
        from kb.user_config import KbxConfig, find_config, load_config

        _config_path = find_config()
        _cfg = load_config(_config_path) if _config_path else KbxConfig()
        _sha = maybe_auto_commit(
            project_root,
            operation="correct",
            target=f'"{term}" → "{replacement}"',
            config=_cfg.writes,
            no_commit=no_commit,
            file_count=int(result["meta"].get("files") or len(result["results"])),
        )
        if _sha:
            click.echo(f"  auto-committed {_sha[:8]}", err=True)

    meta = result["meta"]
    action = meta.get("action", "scan")

    if fmt in ("json", "jsonl"):
        kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)
        return

    # Human-readable output
    if not result["results"]:
        click.echo(f"No occurrences of '{term}' found.", err=True)
        return

    if action == "scan":
        total = meta.get("total_occurrences", 0)
        click.echo(
            f"Found {total} occurrences of '{term}' in {meta.get('files', 0)} files:\n",
            err=True,
        )
        for entry in result["results"]:
            header = entry["rel_path"]
            if entry.get("title"):
                header += f"  ({entry['title']}"
                if entry.get("date"):
                    header += f", {entry['date']}"
                header += ")"
            click.echo(f"  {entry['count']}x  {header}")
            if entry.get("attendees"):
                names = ", ".join(a["name"] for a in entry["attendees"])
                click.echo(f"       attendees: {names}")
            for line in (entry.get("sample_lines") or [])[:3]:
                click.echo(f"       | {line.strip()}")
    elif action == "dry_run":
        total = meta.get("total_occurrences", 0)
        click.echo(
            f"DRY RUN: Would replace {total} occurrences of "
            f"'{term}' → '{replacement}' in {meta.get('files', 0)} files:\n",
            err=True,
        )
        for entry in result["results"]:
            click.echo(f"  {entry['count']}x  {entry['rel_path']}")
        click.echo("\nAdd --apply to execute.", err=True)
    elif action == "applied":
        click.echo(
            f"Replaced {meta.get('occurrences_replaced', 0)} occurrences of "
            f"'{term}' → '{replacement}' in {meta.get('files_changed', 0)} files.",
            err=True,
        )
        for r in result["results"]:
            click.echo(f"  {r['path']}")


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

    meetings = click.prompt("Where are your meeting transcripts?", default="memory/meetings")
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
@_commit_option
def memory_add(
    text: str,
    entity_name: str | None,
    body: str | None,
    tags_str: str | None,
    pin_flag: bool,
    fact_date: str | None,
    no_commit: bool,
) -> None:
    """Record a fact about an entity, or create a searchable note.

    \b
    Decision tree:
      --entity only (no --body/--tags/--pin) → fact (appended to entity file)
      everything else                        → note file in memory/notes/
    """
    from kb.api import KnowledgeBase

    project_root = _find_project_root()
    kb = KnowledgeBase._from_existing(db=_get_db(), project_root=project_root)

    try:
        is_note = body is not None or tags_str is not None or pin_flag or entity_name is None

        if not is_note:
            assert entity_name is not None
            try:
                result = kb.add_fact(entity_name, text, date=fact_date)
                click.echo(f"Fact recorded for {result['entity']}.", err=True)
            except ValueError as e:
                click.echo(str(e), err=True)
                raise SystemExit(1) from None
            _autocommit_after_write("add-fact", str(result["entity"]), no_commit=no_commit)
            return

        # Read body from stdin if -
        if body == "-":
            body = sys.stdin.read()

        tags: list[str] = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        result = kb.add_note(
            text,
            body=body,
            tags=tags or None,
            pin=pin_flag,
            entity=entity_name,
            date=fact_date,
        )
        click.echo(f"Note created: {result['path']}", err=True)
        if pin_flag:
            click.echo("Pinned to context.", err=True)
        _autocommit_after_write("create-note", text, no_commit=no_commit)
    finally:
        kb.close()


@memory.command("list")
@click.option("--since", "since_days", default=None, type=int, help="Show facts from last N days.")
@output_options
def memory_list(
    since_days: int | None, fmt: str, fields: list[str] | None, jq_expr: str | None
) -> None:
    """List recorded facts."""
    from kb.api import KnowledgeBase

    project_root = _find_project_root()
    kb = KnowledgeBase._from_existing(db=_get_db(), project_root=project_root)
    try:
        facts = kb.list_facts(since_days=since_days)
    finally:
        kb.close()

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


@memory.command("similar")
@click.argument("text")
@click.option("--entity", "entity_name", default=None, help="Scope to facts for this entity.")
@click.option("--path", "path_arg", default=None, help="Scope to chunks of this document path.")
@click.option(
    "--threshold",
    default=0.85,
    type=float,
    show_default=True,
    help="Minimum cosine similarity (0.0-1.0).",
)
@click.option("--limit", default=5, type=int, show_default=True, help="Max matches to return.")
@output_options
def memory_similar_cmd(
    text: str,
    entity_name: str | None,
    path_arg: str | None,
    threshold: float,
    limit: int,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Find existing facts or content semantically similar to TEXT.

    \b
    Use --entity for entity-scoped fact dedup, --path for document-scoped chunk
    lookup, or neither for a global chunk search. Threshold + limit cap the
    returned matches. Pair with kbx-output JSON formats for programmatic use.

    \b
    Examples:
      kb memory similar "prefers async comms" --entity "Person A"
      kb memory similar "deployment rollback" --path memory/projects/foo.md
      kb memory similar "deployment rollback" --format json
    """
    from kb.api import KnowledgeBase

    kb = KnowledgeBase._from_existing(db=_get_db(), project_root=_find_project_root())
    try:
        resp = kb.memory_similar(
            text,
            entity=entity_name,
            path=path_arg,
            threshold=threshold,
            limit=limit,
        )
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    except RuntimeError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    finally:
        kb.close()

    kb_output(
        resp.model_dump(),
        fmt=fmt,
        fields=fields,
        jq_expr=jq_expr,
        columns=["score", "match_type", "entity_name", "text"],
    )


@memory.command("delete-fact")
@click.argument("entity_name")
@click.argument("fact_seq", type=int)
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
@output_options
@_commit_option
def memory_delete_fact(
    entity_name: str,
    fact_seq: int,
    force: bool,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Delete a fact by entity name and sequence number."""
    from kb.api import KnowledgeBase

    kb = KnowledgeBase._from_existing(db=_get_db(), project_root=_find_project_root())
    try:
        if not force:
            click.confirm(f"Delete fact #{fact_seq} on '{entity_name}'?", abort=True)

        result = kb.delete_fact(entity_name, fact_seq)
        if fmt == "table" and not fields and not jq_expr:
            click.echo(f"Deleted fact #{fact_seq} on '{entity_name}'.", err=True)
        else:
            kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)
        _autocommit_after_write("delete-fact", entity_name, no_commit=no_commit)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    finally:
        kb.close()


@memory.command("edit-fact")
@click.argument("entity_name")
@click.argument("fact_seq", type=int)
@click.option("--text", default=None, help="New fact text.")
@click.option("--date", "fact_date", default=None, help="New fact date (YYYY-MM-DD).")
@output_options
@_commit_option
def memory_edit_fact(
    entity_name: str,
    fact_seq: int,
    text: str | None,
    fact_date: str | None,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Edit a fact's text or date by entity name and sequence number."""
    from kb.api import KnowledgeBase

    if text is None and fact_date is None:
        click.echo("Specify --text and/or --date.", err=True)
        raise SystemExit(1)

    kb = KnowledgeBase._from_existing(db=_get_db(), project_root=_find_project_root())
    try:
        result = kb.edit_fact(entity_name, fact_seq, text=text, date=fact_date)
        if fmt == "table" and not fields and not jq_expr:
            click.echo(f"Updated fact #{fact_seq} on '{entity_name}'.", err=True)
        else:
            kb_output(result, fmt=fmt, fields=fields, jq_expr=jq_expr)
        _autocommit_after_write("edit-fact", entity_name, no_commit=no_commit)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    finally:
        kb.close()


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


@note.command("edit")
@click.argument("target")
@click.option("--body", default=None, help="Replace note body with this content.")
@click.option("--append", "append_text", default=None, help="Append this content to note body.")
@click.option(
    "--insert-under",
    "insert_under",
    default=None,
    help="Insert --body as a line under this heading (e.g. '## Open Items'). "
    "Creates the section if absent. Most-recent-first; idempotent on duplicate.",
)
@click.option("--tags", "tags_str", default=None, help="Comma-separated tags (replaces existing).")
@click.option("--pin/--unpin", "pin_flag", default=None, help="Pin or unpin note.")
@output_options
@_commit_option
def note_edit(
    target: str,
    body: str | None,
    append_text: str | None,
    insert_under: str | None,
    tags_str: str | None,
    pin_flag: bool | None,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Edit an existing memory note (body, tags, pin status)."""
    from kb.api import KnowledgeBase

    project_root = _find_project_root()
    db = _get_db()
    kb = KnowledgeBase._from_existing(db=db, project_root=project_root)

    try:
        tag_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
        result = kb.edit_note(
            target,
            body=body,
            append=append_text,
            tags=tag_list,
            pin=pin_flag,
            insert_under=insert_under,
        )

        # Build output with tags for display
        output_tags: list[str] = tag_list or []
        if not output_tags:
            conn = db.get_sqlite_conn()
            doc_row = conn.execute(
                "SELECT tags FROM documents WHERE path = ?", (result["path"],)
            ).fetchone()
            if doc_row and doc_row["tags"]:
                output_tags = json.loads(doc_row["tags"])

        result_data: dict[str, Any] = {
            "path": result["path"],
            "title": result["title"],
            "tags": output_tags,
            "pinned": result["pinned"],
        }
        if fmt == "table" and not fields and not jq_expr:
            click.echo(f"Updated: {result['path']}", err=True)
        else:
            kb_output(result_data, fmt=fmt, fields=fields, jq_expr=jq_expr)
        _autocommit_after_write("edit-note", str(result["path"]), no_commit=no_commit)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from None
    finally:
        kb.close()


@note.command("delete")
@click.argument("target")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
@output_options
@_commit_option
def note_delete(
    target: str,
    force: bool,
    no_commit: bool,
    fmt: str,
    fields: list[str] | None,
    jq_expr: str | None,
) -> None:
    """Delete a memory note (file + index entry)."""
    db = _get_db()
    conn = db.get_sqlite_conn()
    project_root = _find_project_root()

    doc = _find_document_by_target(conn, target)
    if doc is None:
        click.echo(f"Note not found: {target}", err=True)
        raise SystemExit(1)

    if doc["doc_type"] not in ("memory_note", "memory_doc"):
        click.echo(
            f"Not a memory note (doc_type={doc['doc_type']}). "
            "Only memory notes can be deleted with this command.",
            err=True,
        )
        raise SystemExit(1)

    if not force:
        click.confirm(f"Delete note '{doc['title']}'?", abort=True)

    doc_id = doc["id"]
    rel_path = doc["path"]

    # Delete file from disk
    file_path = project_root / rel_path
    if file_path.exists():
        file_path.unlink()

    # Clean DB
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.execute("DELETE FROM entity_mentions WHERE document_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()

    result_data: dict[str, Any] = {
        "path": rel_path,
        "title": doc["title"],
        "deleted": True,
    }
    if fmt == "table" and not fields and not jq_expr:
        click.echo(f"Deleted: {rel_path}", err=True)
    else:
        kb_output(result_data, fmt=fmt, fields=fields, jq_expr=jq_expr)
    _autocommit_after_write("delete-note", str(rel_path), no_commit=no_commit)


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


@cli.group(invoke_without_command=True)
@click.pass_context
def sync(ctx: click.Context) -> None:
    """Sync commands — pull data from external sources.

    Run without a subcommand to sync all sources.
    """
    if ctx.invoked_subcommand is None:
        # Run all sync plugins sequentially
        for name in ["granola", "notion"]:
            click.echo(f"\nSyncing {name}...", err=True)
            try:
                ctx.invoke(sync.commands[name])
            except Exception as e:
                click.echo(f"  {name} failed: {e}", err=True)
        click.echo("\nTip: Run `kb index run` to index new files.", err=True)


@sync.command("granola")
@click.option(
    "--since", default=None, help="Sync since date (YYYY-MM-DD or shorthand: 7d, 2w, 3m)."
)
@click.option("--dry-run", is_flag=True, help="Preview only, no file writes.")
@click.option("--force", is_flag=True, help="Overwrite files even if timestamps match.")
@click.option("--no-index", is_flag=True, help="Skip indexing after sync.")
@click.option(
    "--token-path",
    default=None,
    type=click.Path(exists=False),
    help="Legacy only: path to supabase.json (ignored unless --legacy is set).",
)
@click.option(
    "--legacy",
    is_flag=True,
    help="Use the deprecated internal-API path (reads supabase.json). "
    "Will be removed in a future release.",
)
def sync_granola(
    since: str | None,
    dry_run: bool,
    force: bool,
    no_index: bool,
    token_path: str | None,
    legacy: bool,
) -> None:
    """Sync meetings from Granola.

    Default path uses the public REST API at api.granola.ai/v1 with a Bearer key
    read from GRANOLA_API_KEY env or the macOS Keychain (service: granola_api_key).
    Pass --legacy to fall back to the internal-API path that reads supabase.json.
    """
    from kb.dateparse import resolve_since

    project_root = _find_project_root()
    data_dir = _get_data_dir()

    if legacy:
        from pathlib import Path as P

        from kb.sync.granola import sync_granola as do_legacy_sync

        click.echo(
            "WARN: --legacy path uses Granola's internal API and is deprecated. "
            "It will be removed in a future release. See docs/sync.md for migration.",
            err=True,
        )
        tp = P(token_path) if token_path else None
        result = do_legacy_sync(
            project_root=project_root,
            data_dir=data_dir,
            since=resolve_since(since),
            dry_run=dry_run,
            force=force,
            token_path=tp,
            on_progress=lambda msg: click.echo(msg, err=True),
        )
    else:
        from kb.sync.granola_public import sync_granola_public

        result = sync_granola_public(
            project_root=project_root,
            data_dir=data_dir,
            since=resolve_since(since),
            dry_run=dry_run,
            force=force,
            on_progress=lambda msg: click.echo(msg, err=True),
        )

    click.echo(
        f"Sync complete: {result['created']} created, {result['updated']} updated, "
        f"{result['skipped']} skipped (total: {result['total']}).",
        err=True,
    )

    if not dry_run and not no_index:
        click.echo("Tip: Run `kb index run` to index new files.", err=True)


@sync.command("notion")
@click.option(
    "--since", default=None, help="Sync since date (YYYY-MM-DD or shorthand: 7d, 2w, 3m)."
)
@click.option("--dry-run", is_flag=True, help="Preview only, no file writes.")
@click.option("--force", is_flag=True, help="Overwrite files even if timestamps match.")
@click.option("--no-index", is_flag=True, help="Skip indexing after sync.")
def sync_notion_cmd(since: str | None, dry_run: bool, force: bool, no_index: bool) -> None:
    """Sync meetings from Notion AI Meeting Notes."""
    from kb.dateparse import resolve_since
    from kb.sync.notion import sync_notion as do_sync
    from kb.user_config import find_config, load_config

    project_root = _find_project_root()
    data_dir = _get_data_dir()

    # Load space_id from config if available
    space_id: str | None = None
    config_path = find_config()
    if config_path is not None:
        cfg = load_config(config_path)
        space_id = cfg.sync.notion.space_id

    result = do_sync(
        project_root=project_root,
        data_dir=data_dir,
        since=resolve_since(since),
        dry_run=dry_run,
        force=force,
        space_id=space_id,
        on_progress=lambda msg: click.echo(msg, err=True),
    )

    click.echo(
        f"Sync complete: {result['created']} created, {result['updated']} updated, "
        f"{result['skipped']} skipped (total: {result['total']}).",
        err=True,
    )

    if not dry_run and not no_index:
        click.echo("Tip: Run `kb index run` to index new files.", err=True)


# ---------------------------------------------------------------------------
# Granola write commands
# ---------------------------------------------------------------------------


@cli.group()
def granola() -> None:
    """Granola integration — view notes on Granola meetings."""


@granola.command("view")
@click.argument("calendar_uid")
@click.option("--summary", "mode", flag_value="summary", help="Show AI summary instead of notes.")
@click.option(
    "--transcript", "mode", flag_value="transcript", help="Show transcript (live from API)."
)
@click.option("--all", "mode", flag_value="all", help="Show notes, AI summary, and transcript.")
@click.option("--plain", is_flag=True, help="Raw markdown only (no YAML header).")
def granola_view(calendar_uid: str, mode: str | None, plain: bool) -> None:
    """View notes on a Granola meeting document.

    CALENDAR_UID is the Google Calendar event ID or iCalUID.
    """
    from kb.sync.granola import (
        GranolaClient,
        extract_panel_markdown,
        transcript_to_markdown,
    )

    client = GranolaClient()
    click.echo(f"Searching for document matching '{calendar_uid}'...", err=True)
    doc = client.find_document(calendar_uid=calendar_uid)

    if not doc:
        raise click.ClickException(f"No Granola document found for UID '{calendar_uid}'.")

    notes_md = doc.get("notes_markdown") or ""
    summary_md = extract_panel_markdown(doc)

    # Fetch transcript only when needed (separate API call)
    transcript_md = ""
    if mode in ("transcript", "all"):
        click.echo("Fetching transcript...", err=True)
        segments = client.get_transcript(doc["id"])
        transcript_md = transcript_to_markdown(segments)

    if mode == "summary":
        body = summary_md or "(no AI summary)"
    elif mode == "transcript":
        body = transcript_md or "(no transcript)"
    elif mode == "all":
        parts = []
        parts.append(notes_md if notes_md.strip() else "(no notes)")
        if summary_md.strip():
            parts.append("---\n\n" + summary_md)
        if transcript_md.strip():
            parts.append("---\n\n" + transcript_md)
        body = "\n\n".join(parts)
    else:
        body = notes_md if notes_md.strip() else "(no notes)"

    if plain:
        click.echo(body)
        return

    # Build YAML header
    import yaml

    gcal = doc.get("google_calendar_event") or {}
    header: dict[str, object] = {
        "title": doc.get("title") or "Untitled",
        "date": (doc.get("created_at") or "")[:10],
        "granola_id": doc.get("id", ""),
        "calendar_uid": gcal.get("iCalUID") or gcal.get("id") or calendar_uid,
        "files": {
            "notes": bool(notes_md.strip()),
            "ai_summary": bool(summary_md.strip()),
            "transcript": bool(transcript_md.strip()),
        },
    }
    yaml_str: str = yaml.dump(header, default_flow_style=False, allow_unicode=True, sort_keys=False)
    click.echo("---")
    click.echo(yaml_str.rstrip())
    click.echo("---")
    click.echo()
    click.echo(body)


# ---------------------------------------------------------------------------
# access — clear hotness access history (#67)
# ---------------------------------------------------------------------------


@cli.group()
def access() -> None:
    """Hotness access-tracking management — reset history per #67."""


@access.command("reset")
@click.option(
    "--entity",
    "entity_name",
    default=None,
    help="Reset access history for a specific entity by name.",
)
@click.option(
    "--path",
    "doc_path",
    default=None,
    help="Reset access history for a specific document by path.",
)
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def access_reset(entity_name: str | None, doc_path: str | None, as_json: bool) -> None:
    """Clear hotness access counts and timestamps.

    With no flags: clears ALL documents and entities. With --entity or --path:
    clears only the matching record.
    """
    from kb.access import reset_document_access, reset_entity_access
    from kb.config import find_entity

    db = _get_db()
    conn = db.get_sqlite_conn()

    docs_cleared = 0
    entities_cleared = 0

    if entity_name is None and doc_path is None:
        docs_cleared = reset_document_access(conn)
        entities_cleared = reset_entity_access(conn)
    else:
        if entity_name:
            entity = find_entity(conn, entity_name)
            if entity is None:
                raise click.ClickException(f"Entity not found: {entity_name}")
            entities_cleared = reset_entity_access(conn, entity["id"])
        if doc_path:
            row = conn.execute("SELECT id FROM documents WHERE path = ?", (doc_path,)).fetchone()
            if row is None:
                raise click.ClickException(f"Document not found: {doc_path}")
            docs_cleared = reset_document_access(conn, row["id"])

    result = {"documents_cleared": docs_cleared, "entities_cleared": entities_cleared}
    if as_json:
        import json as _json

        click.echo(_json.dumps(result))
    else:
        click.echo(
            f"Reset hotness — documents: {docs_cleared}, entities: {entities_cleared}.",
            err=True,
        )
