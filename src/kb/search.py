"""Hybrid search pipeline — FTS5 + LanceDB vector + weighted RRF + recency."""

from __future__ import annotations

import html
import json
import math
import re
import sys
import time
from datetime import date
from typing import TYPE_CHECKING, Any

from kb.entities import strip_wikilinks
from kb.types import SearchMeta, SearchResponse, SearchResult

if TYPE_CHECKING:
    from kb.db import Database
    from kb.embeddings import Embedder

# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def _normalize_bm25_scores(results: list[dict[str, Any]]) -> None:
    """Min-max normalize raw BM25 scores in-place to [0, 1].

    FTS5 bm25() returns negative scores where lower (more negative) = more relevant.
    After normalization, 1.0 = best match, 0.0 = worst in the result set.
    Single results get score 1.0.
    """
    if not results:
        return

    raw_scores = [r["raw_bm25"] for r in results]
    min_raw = min(raw_scores)  # most relevant (most negative)
    max_raw = max(raw_scores)  # least relevant (closest to 0)

    if min_raw == max_raw:
        for r in results:
            r["bm25_score"] = 1.0
        return

    spread = max_raw - min_raw
    for r in results:
        # Invert: most negative (min_raw) → 1.0, least negative (max_raw) → 0.0
        r["bm25_score"] = (max_raw - r["raw_bm25"]) / spread


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank position."""
    return 1.0 / (k + rank)


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


def _recency_weight(doc_date: str | None, half_life_days: int = 90) -> float:
    """Exponential decay weight. Returns [0, 1]. Recent → ~1.0, old → ~0.

    None date returns 0.5 (neutral).
    """
    if doc_date is None:
        return 0.5

    try:
        d = date.fromisoformat(doc_date)
    except (ValueError, TypeError):
        return 0.5

    days_ago = (date.today() - d).days
    if days_ago < 0:
        return 1.0  # future date treated as very recent

    return math.exp(-math.log(2) * days_ago / half_life_days)


# ---------------------------------------------------------------------------
# Snippet helpers
# ---------------------------------------------------------------------------


def _strip_metadata_prefix(text: str) -> str:
    """Strip metadata prefix from chunk content.

    Handles both old format (``[Meeting: ...]\ncontent``) and new format
    (``Title\ncontent`` where title repeats the document title).
    """
    # Old format: [Meeting: ...], [Transcript: ...], [Summary: ...]
    stripped = re.sub(r"^\[(?:Meeting|Transcript|Summary):[^\]]*\]\n?", "", text)
    return stripped.strip()


def _make_snippet(content: str, max_chars: int = 200, query: str = "") -> str:
    """Extract a display snippet from chunk content.

    When *query* is provided, centers the snippet window around the first
    matching term so that ``_highlight_snippet`` will almost always have
    something to bold.  Falls back to the beginning of the content when no
    match is found.
    """
    clean = _strip_metadata_prefix(content)
    clean = strip_wikilinks(clean)
    if len(clean) <= max_chars:
        return clean

    # Try to center the window on the first query-term match
    start = 0
    if query:
        terms = _sanitize_fts_input(query).split()
        for term in terms:
            clean_term = term.rstrip("*")
            if not clean_term:
                continue
            m = re.search(rf"\b{re.escape(clean_term)}\b", clean, re.IGNORECASE)
            if m:
                # Place the match ~50 chars from the left edge of the window
                start = max(0, m.start() - 50)
                break

    window = clean[start : start + max_chars]

    # Trim to word boundary at both ends
    if start > 0:
        # Drop partial leading word
        first_space = window.find(" ")
        if first_space != -1 and first_space < 30:
            window = window[first_space + 1 :]

    # Drop partial trailing word
    if start + max_chars < len(clean):
        window = window.rsplit(" ", 1)[0] + "..."

    if start > 0:
        window = "..." + window

    return window


# ---------------------------------------------------------------------------
# Snippet highlighting
# ---------------------------------------------------------------------------


def _highlight_snippet(snippet: str, query: str) -> str:
    """HTML-escape snippet then wrap query terms in <mark> tags.

    HTML-escapes first (so content is safe), then applies word-boundary
    highlighting. Case-insensitive. Consumers render with ``{{ snippet | safe }}``.
    Uses ``<mark>`` (semantically correct for search highlighting, CSS-stylable).
    """
    escaped = html.escape(snippet)

    terms = _sanitize_fts_input(query).split()
    if not terms:
        return escaped

    for term in terms:
        clean_term = term.rstrip("*")
        if not clean_term:
            continue
        pattern = rf"\b({re.escape(clean_term)})\b"
        escaped = re.sub(pattern, r"<mark>\1</mark>", escaped, flags=re.IGNORECASE)

    return escaped


# ---------------------------------------------------------------------------
# Glossary-aware query expansion
# ---------------------------------------------------------------------------


def _expand_query_with_glossary(
    query: str,
    glossary: list[tuple[str, str, str]],
) -> tuple[str, dict[str, str]]:
    """Expand query terms using glossary entries.

    For each glossary entry ``(term, expansion, section)``, if the term appears
    as a whole word in the query (case-insensitive) append the expansion as an
    OR alternative, and vice-versa.  Uses word-boundary matching to avoid false
    positives (e.g. "EM" should not match inside "implementation").
    Returns ``(expanded_query, {original: expansion})``.
    """
    query_lower = query.lower()
    additions: list[str] = []
    expanded_terms: dict[str, str] = {}

    for term, expansion, _section in glossary:
        term_lower = term.lower()
        expansion_lower = expansion.lower()

        term_matches = bool(re.search(rf"\b{re.escape(term_lower)}\b", query_lower))
        expansion_matches = bool(re.search(rf"\b{re.escape(expansion_lower)}\b", query_lower))

        if term_matches and not expansion_matches:
            additions.append(expansion)
            expanded_terms[term] = expansion
        elif expansion_matches and not term_matches:
            additions.append(term)
            expanded_terms[expansion] = term

    if not additions:
        return query, {}

    expanded = query + " OR " + " OR ".join(f'"{a}"' for a in additions)
    return expanded, expanded_terms


# ---------------------------------------------------------------------------
# Entity-aware boosting
# ---------------------------------------------------------------------------

_ENTITY_BOOST = 1.15  # 15% multiplicative boost for entity match


def _apply_entity_boost(score: float, has_entity_match: bool) -> float:
    """Apply a multiplicative boost when a result's entities match the query."""
    if has_entity_match:
        return score * _ENTITY_BOOST
    return score


# ---------------------------------------------------------------------------
# Progress hooks
# ---------------------------------------------------------------------------


class SearchProgressHook:
    """Base class for search progress callbacks. Override methods as needed."""

    def on_fts_start(self) -> None:
        pass

    def on_fts_done(self, count: int) -> None:
        pass

    def on_vector_start(self) -> None:
        pass

    def on_vector_done(self, count: int) -> None:
        pass

    def on_fusion_done(self, count: int) -> None:
        pass


# ---------------------------------------------------------------------------
# FTS5 search
# ---------------------------------------------------------------------------


# FTS5 reserved operators that should be stripped from user input
_FTS5_OPERATORS = re.compile(r"\b(AND|OR|NOT|NEAR)\b", re.IGNORECASE)
_FTS5_SPECIAL = re.compile(r"[\^():\"]")
# Strip * that is NOT a valid FTS5 suffix wildcard (trailing after word chars)
_INVALID_STAR = re.compile(r"(?<!\w)\*|\*(?=\w)")


def _sanitize_fts_input(text: str) -> str:
    """Strip FTS5 operators and special characters from user input.

    Trailing ``*`` on words is preserved for FTS5 prefix matching (e.g. ``migr*``).
    """
    text = _FTS5_OPERATORS.sub(" ", text)
    text = _FTS5_SPECIAL.sub("", text)
    text = _INVALID_STAR.sub("", text)
    return " ".join(text.split())  # collapse whitespace


def _build_fts_query(query: str, extra_or_terms: list[str] | None = None) -> list[str]:
    """Build FTS5 query variants: phrase → proximity → AND → OR.

    Returns a list of FTS5 match expressions to try in order.
    ``extra_or_terms`` are appended as quoted phrases to the OR variant only
    (used for glossary expansion).
    """
    sanitized = _sanitize_fts_input(query)
    words = sanitized.split()

    if not words:
        return [query.strip()]

    if len(words) == 1:
        variants = [words[0]]
    else:
        phrase = '"' + " ".join(words) + '"'
        proximity = f"NEAR({' '.join(words)}, 10)"
        and_query = " ".join(words)  # implicit AND — all terms must appear in row
        or_query = " OR ".join(words)
        variants = [phrase, proximity, and_query, or_query]

    # Append glossary expansion terms to OR variant
    if extra_or_terms:
        sanitized_extras = [
            _sanitize_fts_input(t) for t in extra_or_terms if _sanitize_fts_input(t)
        ]
        if sanitized_extras:
            quoted = [f'"{t}"' for t in sanitized_extras]
            or_extra = variants[-1] + " OR " + " OR ".join(quoted)
            variants.append(or_extra)

    return variants


def _resolve_doc_ids_for_path(db: Database, path_filter: str | None) -> set[int] | None:
    """Resolve a `--path` pattern to the matching set of document IDs.

    Returns:
        - ``None`` if ``path_filter`` is None (no filter — caller should not constrain results)
        - A possibly-empty ``set[int]`` of matching document IDs otherwise

    Pattern handling:
        - Prefixes (no glob chars): ``memory/notes/`` → ``LIKE 'memory/notes/%'``
        - Simple globs: ``memory/*/2026/*`` → translated to SQL ``LIKE`` (``*`` → ``%``, ``?`` → ``_``)
        - Exact paths: matched literally.
    """
    if path_filter is None:
        return None

    if any(c in path_filter for c in "*?"):
        like_pattern = path_filter.replace("*", "%").replace("?", "_")
    elif path_filter.endswith("/"):
        like_pattern = path_filter + "%"
    else:
        like_pattern = path_filter

    conn = db.get_sqlite_conn()
    rows = conn.execute(
        "SELECT id FROM documents WHERE path LIKE ?",
        (like_pattern,),
    ).fetchall()
    return {row["id"] for row in rows}


def _fts_search(
    db: Database,
    query: str,
    limit: int = 50,
    extra_or_terms: list[str] | None = None,
    doc_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Run FTS5 search with phrase → proximity → OR, merging results across variants.

    Returns list of dicts with: chunk_id, document_id, content, heading, bm25_score (normalized).
    Accumulates results across all variants, keeping the best BM25 score per chunk.

    When ``doc_ids`` is an empty set, returns ``[]`` immediately. When it is a
    non-empty set, results are restricted to chunks whose ``document_id`` is in
    that set.
    """
    if doc_ids is not None and not doc_ids:
        return []

    conn = db.get_sqlite_conn()
    variants = _build_fts_query(query, extra_or_terms=extra_or_terms)

    # Accumulate best raw_bm25 per chunk across all variants
    merged: dict[int, dict[str, Any]] = {}  # chunk_id -> result dict

    if doc_ids is not None:
        doc_id_placeholders = ",".join("?" * len(doc_ids))
        doc_id_filter_sql = f" AND c.document_id IN ({doc_id_placeholders})"
        doc_id_params: tuple[int, ...] = tuple(sorted(doc_ids))
    else:
        doc_id_filter_sql = ""
        doc_id_params = ()

    for fts_expr in variants:
        try:
            rows = conn.execute(
                f"""
                SELECT
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.heading,
                    bm25(chunks_fts, 1.0, 2.0) as raw_bm25
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?{doc_id_filter_sql}
                ORDER BY bm25(chunks_fts, 1.0, 2.0)
                LIMIT ?
                """,
                (fts_expr, *doc_id_params, limit),
            ).fetchall()

            for row in rows:
                cid = row["chunk_id"]
                raw = row["raw_bm25"]
                if cid not in merged or raw < merged[cid]["raw_bm25"]:
                    # More negative = more relevant, so keep the lower value
                    merged[cid] = {
                        "chunk_id": cid,
                        "document_id": row["document_id"],
                        "content": row["content"],
                        "heading": row["heading"],
                        "raw_bm25": raw,
                    }

            if len(merged) >= limit:
                break
        except Exception:
            # FTS5 syntax error (e.g. NEAR not supported for some queries)
            continue

    if not merged:
        return []

    results = sorted(merged.values(), key=lambda r: r["raw_bm25"])[:limit]
    _normalize_bm25_scores(results)
    return results


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


def _vector_search(
    db: Database,
    embedder: Embedder,
    query: str,
    limit: int = 50,
    doc_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Run LanceDB vector search. Returns list of dicts with chunk_id, score.

    When ``doc_ids`` is an empty set, returns ``[]`` immediately without touching
    LanceDB. When it is a non-empty set, the LanceDB query is pre-filtered with
    a ``document_id IN (...)`` ``where`` clause so the ANN search only scores
    candidate chunks from the matching documents (mirrors FTS path filtering —
    issue #73).
    """
    if doc_ids is not None and not doc_ids:
        return []

    table = db.get_lance_table()
    if table is None:
        return []

    query_vec = embedder.embed_query(query)
    query_builder = table.search(query_vec[0].tolist())

    if doc_ids is not None:
        id_list = ",".join(str(did) for did in sorted(doc_ids))
        query_builder = query_builder.where(f"document_id IN ({id_list})")

    results = query_builder.limit(limit).to_list()

    return [
        {
            "chunk_id": int(r["chunk_id"]),
            "document_id": int(r["document_id"]),
            "score": 1.0 - float(r.get("_distance", 0)),  # cosine distance → similarity
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Strong signal fast path
# ---------------------------------------------------------------------------


def _is_strong_signal(fts_results: list[dict[str, Any]]) -> bool:
    """Check if FTS5 results are strong enough to skip vector search.

    With min-max normalization, top is always 1.0 when there are results.
    Strong signal requires few results (precise query) and a large gap
    between top and second result (clear winner, not a broad match).
    """
    if not fts_results:
        return False

    if len(fts_results) <= 2:
        return True  # very few hits = precise query

    top = fts_results[0]["bm25_score"]
    if top < 0.85:
        return False

    gap = top - fts_results[1]["bm25_score"]
    return bool(gap >= 0.15)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def _enrich_results(
    db: Database,
    chunk_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """Look up document metadata and entities for a set of chunk IDs.

    Returns {chunk_id: {title, path, date, doc_type, section, content, entities}}.
    """
    if not chunk_ids:
        return {}

    conn = db.get_sqlite_conn()
    placeholders = ",".join("?" * len(chunk_ids))

    rows = conn.execute(
        f"""
        SELECT
            c.id as chunk_id,
            c.chunk_index,
            c.heading,
            c.content,
            d.id as document_id,
            d.title,
            d.path,
            d.doc_date,
            d.doc_type,
            d.tags
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()

    # Batch entity lookup — collect all doc_ids, query once
    doc_ids = list({row["document_id"] for row in rows})
    entities_by_doc: dict[int, list[str]] = {did: [] for did in doc_ids}
    if doc_ids:
        ep = ",".join("?" * len(doc_ids))
        entity_rows = conn.execute(
            f"""SELECT DISTINCT em.document_id, e.name FROM entities e
                JOIN entity_mentions em ON e.id = em.entity_id
                WHERE em.document_id IN ({ep})""",
            doc_ids,
        ).fetchall()
        for er in entity_rows:
            entities_by_doc[er["document_id"]].append(er["name"])

    result = {}
    for row in rows:
        cid = row["chunk_id"]
        doc_id = row["document_id"]
        result[cid] = {
            "document_id": doc_id,
            "chunk_index": row["chunk_index"],
            "title": row["title"],
            "path": row["path"],
            "date": row["doc_date"],
            "doc_type": row["doc_type"],
            "section": row["heading"],
            "content": row["content"],
            "entities": entities_by_doc.get(doc_id, []),
            "tags": json.loads(row["tags"]) if row["tags"] else [],
        }

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    db: Database,
    embedder: Embedder | None,
    query: str,
    *,
    limit: int = 10,
    fast: bool = False,
    recency: float = 0.15,
    doc_type: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    tag: str | None = None,
    path_filter: str | None = None,
    sort_by: str = "score",
    fts_weight: float = 1.0,
    vector_weight: float = 1.0,
    dedupe: bool = False,
    snippet_chars: int = 200,
    full_chunks: bool = False,
    merge_chunks: bool = False,
    highlight: bool = False,
    hook: SearchProgressHook | None = None,
) -> SearchResponse:
    """Hybrid search: FTS5 + vector + weighted RRF + recency.

    Returns a SearchResponse Pydantic model with results and meta.
    """
    start_time = time.monotonic()
    hook = hook or SearchProgressHook()

    # Auto-extract date filters from query if not explicitly provided
    if from_date is None and to_date is None:
        from kb.dateparse import extract_date_filters

        query, from_date, to_date = extract_date_filters(query)

    # Glossary expansion — best-effort, does not block search
    expanded_terms: dict[str, str] = {}
    expansion_or_terms: list[str] | None = None
    try:
        from kb.config import find_project_root
        from kb.glossary import list_terms

        root = find_project_root()
        if root:
            glossary = [(e.term, e.expansion, e.section) for e in list_terms(root)]
            if glossary:
                _, expanded_terms = _expand_query_with_glossary(query, glossary)
                if expanded_terms:
                    expansion_or_terms = list(expanded_terms.values())
    except Exception:
        pass

    # Resolve --path filter to a set of document IDs (issue #73).
    # Done once so FTS and vector pipelines apply the same filter consistently.
    path_doc_ids = _resolve_doc_ids_for_path(db, path_filter)

    # Step 1: FTS5 search
    hook.on_fts_start()
    fts_results = _fts_search(
        db,
        query,
        limit=limit * 5,
        extra_or_terms=expansion_or_terms,
        doc_ids=path_doc_ids,
    )
    hook.on_fts_done(len(fts_results))

    # Step 2: Decide whether to do vector search
    vector_results: list[dict[str, Any]] = []
    used_fast_path = fast or _is_strong_signal(fts_results)

    if not used_fast_path and embedder is not None:
        # Check if LanceDB has data
        table = db.get_lance_table()
        if table is not None:
            hook.on_vector_start()
            vector_results = _vector_search(
                db, embedder, query, limit=limit * 5, doc_ids=path_doc_ids
            )
            hook.on_vector_done(len(vector_results))
        else:
            # FTS-only fallback — no vectors available
            print("Warning: No vector index found. Using FTS-only search.", file=sys.stderr)

    # Step 3: Fuse results
    if vector_results and fts_results:
        fused = _weighted_rrf(
            fts_results, vector_results, fts_weight=fts_weight, vector_weight=vector_weight
        )
    elif fts_results:
        # FTS-only: use normalized BM25 as score
        fused = {r["chunk_id"]: r["bm25_score"] for r in fts_results}
    elif vector_results:
        fused = {r["chunk_id"]: r["score"] for r in vector_results}
    else:
        fused = {}

    hook.on_fusion_done(len(fused))

    # Step 4: Apply recency weighting
    # We need document dates — look up chunk metadata
    all_chunk_ids = list(fused.keys())
    enriched = _enrich_results(db, all_chunk_ids)

    query_words_lower = set(query.lower().split())
    scored: list[tuple[int, float]] = []
    for chunk_id, base_score in fused.items():
        info = enriched.get(chunk_id)
        if info is None:
            continue

        # Apply doc_type filter
        if doc_type and info["doc_type"] != doc_type:
            continue

        # Apply date range filters
        if from_date and (not info["date"] or info["date"] < from_date):
            continue
        if to_date and (not info["date"] or info["date"] > to_date):
            continue

        # Apply tag filter
        if tag:
            filter_tags = [t.strip() for t in tag.split(",") if t.strip()]
            doc_tags: list[str] = info.get("tags", [])
            if not all(t in doc_tags for t in filter_tags):
                continue

        # Apply recency
        if recency > 0 and info["date"]:
            rw = _recency_weight(info["date"])
            final = (1 - recency) * base_score + recency * rw * base_score
        else:
            final = base_score

        # Entity boost: check if any entity name word matches a query word
        has_entity_match = False
        for entity_name in info["entities"]:
            entity_words = set(entity_name.lower().split())
            if entity_words & query_words_lower:
                has_entity_match = True
                break
        final = _apply_entity_boost(final, has_entity_match)

        scored.append((chunk_id, final))

    # Sort by score descending (initial ordering)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate: keep only the best chunk per document, count matching chunks
    doc_chunk_counts: dict[int, int] = {}
    doc_chunk_ids: dict[int, list[int]] = {}  # doc_id -> all matching chunk_ids
    if dedupe:
        for chunk_id, _score in scored:
            info = enriched[chunk_id]
            doc_id = info["document_id"]
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
            doc_chunk_ids.setdefault(doc_id, []).append(chunk_id)

        seen_docs: set[int] = set()
        deduped: list[tuple[int, float]] = []
        for chunk_id, score in scored:
            info = enriched[chunk_id]
            doc_id = info["document_id"]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                deduped.append((chunk_id, score))
        scored = deduped

    scored = scored[:limit]

    # Build output
    results: list[SearchResult] = []
    for chunk_id, score in scored:
        info = enriched[chunk_id]
        snippet = _make_snippet(info["content"], max_chars=snippet_chars, query=query)
        if highlight:
            snippet = _highlight_snippet(snippet, query)
        doc_id = info["document_id"]
        chunk_content: str | None = None
        if full_chunks:
            if merge_chunks and dedupe and doc_id in doc_chunk_ids:
                # Concatenate all matching chunks sorted by document position
                cids = doc_chunk_ids[doc_id]
                chunks_sorted = sorted(
                    (enriched[cid] for cid in cids),
                    key=lambda c: c["chunk_index"],
                )
                chunk_content = "\n\n".join(c["content"] for c in chunks_sorted)
            else:
                chunk_content = info["content"]
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                document_id=doc_id,
                title=info["title"],
                path=info["path"],
                date=info["date"],
                doc_type=info["doc_type"],
                score=round(score, 4),
                section=None if info["section"] == "__document__" else info["section"],
                snippet=snippet,
                content=chunk_content,
                entities=info["entities"],
                tags=info.get("tags", []),
                chunk_count=doc_chunk_counts.get(doc_id, 1),
            )
        )

    # Re-sort by date if requested (None dates go last)
    if sort_by == "date":
        results.sort(key=lambda r: r.date or "", reverse=True)

    elapsed_ms = round((time.monotonic() - start_time) * 1000, 1)

    return SearchResponse(
        results=results,
        meta=SearchMeta(
            query=query,
            total=len(results),
            limit=limit,
            sort_by=sort_by,
            execution_ms=elapsed_ms,
            expanded_terms=expanded_terms,
        ),
    )


# ---------------------------------------------------------------------------
# Weighted RRF fusion
# ---------------------------------------------------------------------------


def _weighted_rrf(
    fts_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = 60,
    fts_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> dict[int, float]:
    """Reciprocal Rank Fusion with configurable weighting for FTS and vector.

    Top rank bonus: +0.05 for rank 1, +0.02 for ranks 2-3.
    Chunks appearing in both lists accumulate scores from each.
    """
    scores: dict[int, float] = {}

    for rank, r in enumerate(fts_results, start=1):
        cid = r["chunk_id"]
        rrf = _rrf_score(rank, k) * fts_weight
        if rank == 1:
            rrf += 0.05
        elif rank <= 3:
            rrf += 0.02
        scores[cid] = scores.get(cid, 0) + rrf

    for rank, r in enumerate(vector_results, start=1):
        cid = r["chunk_id"]
        rrf = _rrf_score(rank, k) * vector_weight
        if rank == 1:
            rrf += 0.05
        elif rank <= 3:
            rrf += 0.02
        scores[cid] = scores.get(cid, 0) + rrf

    return scores
