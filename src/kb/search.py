"""Hybrid search pipeline — FTS5 + LanceDB vector + weighted RRF + recency."""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import date
from typing import TYPE_CHECKING, Any

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
    """Strip [Meeting: ...], [Transcript: ...], [Summary: ...] prefix from chunk content."""
    stripped = re.sub(r"^\[(?:Meeting|Transcript|Summary):[^\]]*\]\n?", "", text)
    return stripped.strip()


def _make_snippet(content: str, max_chars: int = 200) -> str:
    """Extract a display snippet from chunk content."""
    clean = _strip_metadata_prefix(content)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0] + "..."


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
_FTS5_SPECIAL = re.compile(r"[*^():\"]")


def _sanitize_fts_input(text: str) -> str:
    """Strip FTS5 operators and special characters from user input."""
    text = _FTS5_OPERATORS.sub(" ", text)
    text = _FTS5_SPECIAL.sub("", text)
    return " ".join(text.split())  # collapse whitespace


def _build_fts_query(query: str) -> list[str]:
    """Build FTS5 query variants: phrase, proximity, OR fallback.

    Returns a list of FTS5 match expressions to try in order.
    """
    sanitized = _sanitize_fts_input(query)
    words = sanitized.split()

    if not words:
        return [query.strip()]

    if len(words) == 1:
        return [words[0]]

    phrase = '"' + " ".join(words) + '"'
    proximity = f"NEAR({' '.join(words)}, 10)"
    or_query = " OR ".join(words)

    return [phrase, proximity, or_query]


def _fts_search(
    db: Database,
    query: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Run FTS5 search with phrase → proximity → OR, merging results across variants.

    Returns list of dicts with: chunk_id, document_id, content, heading, bm25_score (normalized).
    Accumulates results across all variants, keeping the best BM25 score per chunk.
    """
    conn = db.get_sqlite_conn()
    variants = _build_fts_query(query)

    # Accumulate best raw_bm25 per chunk across all variants
    merged: dict[int, dict[str, Any]] = {}  # chunk_id -> result dict

    for fts_expr in variants:
        try:
            rows = conn.execute(
                """
                SELECT
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.heading,
                    bm25(chunks_fts) as raw_bm25
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (fts_expr, limit),
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
) -> list[dict[str, Any]]:
    """Run LanceDB vector search. Returns list of dicts with chunk_id, score."""
    table = db.get_lance_table()
    if table is None:
        return []

    query_vec = embedder.embed_query(query)
    results = table.search(query_vec[0].tolist()).limit(limit).to_list()

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

    if len(fts_results) < 2:
        return True

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
    sort_by: str = "score",
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

    # Step 1: FTS5 search
    hook.on_fts_start()
    fts_results = _fts_search(db, query, limit=limit * 5)
    hook.on_fts_done(len(fts_results))

    # Step 2: Decide whether to do vector search
    vector_results: list[dict[str, Any]] = []
    used_fast_path = fast or _is_strong_signal(fts_results)

    if not used_fast_path and embedder is not None:
        # Check if LanceDB has data
        table = db.get_lance_table()
        if table is not None:
            hook.on_vector_start()
            vector_results = _vector_search(db, embedder, query, limit=limit * 5)
            hook.on_vector_done(len(vector_results))
        else:
            # FTS-only fallback — no vectors available
            print("Warning: No vector index found. Using FTS-only search.", file=sys.stderr)

    # Step 3: Fuse results
    if vector_results and fts_results:
        fused = _weighted_rrf(fts_results, vector_results)
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

        scored.append((chunk_id, final))

    # Sort by score descending (initial ordering)
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:limit]

    # Build output
    results: list[SearchResult] = []
    for chunk_id, score in scored:
        info = enriched[chunk_id]
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                document_id=info["document_id"],
                title=info["title"],
                path=info["path"],
                date=info["date"],
                doc_type=info["doc_type"],
                score=round(score, 4),
                section=None if info["section"] == "__document__" else info["section"],
                snippet=_make_snippet(info["content"]),
                entities=info["entities"],
                tags=info.get("tags", []),
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
        ),
    )


# ---------------------------------------------------------------------------
# Weighted RRF fusion
# ---------------------------------------------------------------------------


def _weighted_rrf(
    fts_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = 60,
) -> dict[int, float]:
    """Reciprocal Rank Fusion with equal weighting for FTS and vector.

    Both lists contribute equally. Top rank bonus: +0.05 for rank 1, +0.02 for ranks 2-3.
    Chunks appearing in both lists accumulate scores from each.
    """
    scores: dict[int, float] = {}

    for rank, r in enumerate(fts_results, start=1):
        cid = r["chunk_id"]
        rrf = _rrf_score(rank, k)
        if rank == 1:
            rrf += 0.05
        elif rank <= 3:
            rrf += 0.02
        scores[cid] = scores.get(cid, 0) + rrf

    for rank, r in enumerate(vector_results, start=1):
        cid = r["chunk_id"]
        rrf = _rrf_score(rank, k)
        if rank == 1:
            rrf += 0.05
        elif rank <= 3:
            rrf += 0.02
        scores[cid] = scores.get(cid, 0) + rrf

    return scores
