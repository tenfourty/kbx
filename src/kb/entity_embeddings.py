"""Entity-as-coherent-object vector embeddings (issue #96 / Prereq A for #69).

Each indexed person/project/team gets a single embedding built from a
deliberate concatenation of identifying signal — name, aliases, role, team,
top-N facts. The embedding lives in a separate LanceDB table (``entities``)
alongside the chunk embeddings, so #69's Pass 1 can do real semantic entity
match instead of falling back to name/alias FTS.

Phase 1 scope (this commit): profile-text builder, incremental embedding
during ``index_all``, and a ``search_entities()`` function used by #69.
CLI/MCP surfaces (``kbx entity search``, ``kb_entity_search`` tool) ship
later when #69 needs them user-facing.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kb.db import Database
    from kb.embeddings import Embedder

# Skip entities whose profile text is below this length — too thin to embed
# meaningfully (e.g. an entity with just a name and no role/facts).
MIN_PROFILE_CHARS = 20

# Cap facts included in the profile to keep the vector focused.
MAX_FACTS_IN_PROFILE = 5

# Cap profile text length to bound embedder cost. Qwen3 truncates at 8K chars
# anyway; this is a pre-truncation budget that prioritises the high-signal
# parts (name/aliases/role) over long fact tails.
MAX_PROFILE_CHARS = 2000


def build_entity_profile_text(
    name: str,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    facts: list[str] | None = None,
) -> str:
    """Concatenate entity identity signals into a single embedding input.

    The format is deliberately ``key | key | key`` — high-signal terms
    separated by pipes — so the embedder sees identity tokens without
    paragraph dilution. Empty/missing fields are skipped.

    Args:
        name: Canonical entity name (required).
        aliases: Known aliases (e.g. nicknames, abbreviations).
        metadata: Entity metadata dict; ``role`` and ``team`` are pulled out.
        facts: Recent fact texts (newest first). Capped at MAX_FACTS_IN_PROFILE.

    Returns:
        Concatenated profile text. May be empty if all inputs are empty.
    """
    parts: list[str] = [name]
    if aliases:
        parts.extend(a for a in aliases if a)
    if metadata:
        role = metadata.get("role")
        team = metadata.get("team")
        if role:
            parts.append(str(role))
        if team:
            parts.append(str(team))
    if facts:
        parts.extend(f for f in facts[:MAX_FACTS_IN_PROFILE] if f)

    text = " | ".join(p.strip() for p in parts if p and p.strip())
    if len(text) > MAX_PROFILE_CHARS:
        text = text[:MAX_PROFILE_CHARS]
    return text


def compute_profile_hash(text: str) -> str:
    """SHA-256 of the profile text — short hex for incremental skip-detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _gather_entity_rows(db: Database) -> list[dict[str, Any]]:
    """Collect entity rows + facts joined for profile building."""
    conn = db.get_sqlite_conn()
    rows = conn.execute(
        """
        SELECT id, name, entity_type, aliases, metadata
        FROM entities
        """
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        aliases = json.loads(row["aliases"]) if row["aliases"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        fact_rows = conn.execute(
            "SELECT fact_text FROM facts WHERE entity_id = ? "
            "ORDER BY fact_date DESC, seq DESC LIMIT ?",
            (row["id"], MAX_FACTS_IN_PROFILE),
        ).fetchall()
        facts = [fr["fact_text"] for fr in fact_rows]
        results.append(
            {
                "id": row["id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "aliases": aliases,
                "metadata": metadata,
                "facts": facts,
            }
        )
    return results


def embed_entities(
    db: Database,
    embedder: Embedder,
    *,
    full: bool = False,
    batch_size: int = 16,
) -> int:
    """Embed entity profiles into the LanceDB ``entities`` table.

    Args:
        db: kbx Database (SQLite + LanceDB).
        embedder: Embedder instance (Qwen3).
        full: When True, re-embed every entity. When False (default), skip
            entities whose ``profile_hash`` already matches the current
            profile text (incremental).
        batch_size: Embedder batch size.

    Returns:
        Number of entities newly embedded (or re-embedded).
    """
    entity_rows = _gather_entity_rows(db)

    # Build the work list: (entity_id, profile_text, profile_hash, entity_type)
    work: list[dict[str, Any]] = []
    for e in entity_rows:
        text = build_entity_profile_text(
            e["name"], aliases=e["aliases"], metadata=e["metadata"], facts=e["facts"]
        )
        if len(text) < MIN_PROFILE_CHARS:
            continue
        work.append(
            {
                "entity_id": int(e["id"]),
                "entity_type": e["entity_type"],
                "profile_text": text,
                "profile_hash": compute_profile_hash(text),
            }
        )

    if not work:
        return 0

    # Incremental: drop entries whose profile_hash matches the stored one.
    if not full:
        table = db.get_lance_entity_table()
        if table is not None:
            existing_hashes: dict[int, str] = {}
            for row in table.to_arrow().to_pylist():
                existing_hashes[int(row["entity_id"])] = row["profile_hash"]
            work = [
                w
                for w in work
                if existing_hashes.get(w["entity_id"]) != w["profile_hash"]
            ]
            if not work:
                return 0

    # Embed in batches
    texts = [w["profile_text"] for w in work]
    embeddings = embedder.embed(texts, batch_size=batch_size)

    rows_to_write: list[dict[str, Any]] = []
    for idx, w in enumerate(work):
        rows_to_write.append(
            {
                "entity_id": w["entity_id"],
                "embedding": embeddings[idx].tolist(),
                "entity_type": w["entity_type"],
                "profile_text": w["profile_text"],
                "profile_hash": w["profile_hash"],
            }
        )

    # Replace strategy: delete existing rows for these entity_ids, then insert.
    # Avoids duplicate-row pile-up across reindex cycles.
    table = db.ensure_lance_entity_table(rows_to_write)
    if table is not None:
        ids_to_replace = ",".join(str(w["entity_id"]) for w in work)
        try:
            table.delete(f"entity_id IN ({ids_to_replace})")
        except Exception:
            # Empty table on first run — delete may fail harmlessly; ensure_lance_entity_table
            # has already inserted via the data= path so no further write needed.
            return len(rows_to_write)
        table.add(rows_to_write)

    return len(rows_to_write)


def search_entities(
    db: Database,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 5,
    entity_type: str | None = None,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Semantic entity search — vector match over entity-profile embeddings.

    Args:
        db: kbx Database.
        embedder: Embedder for the query.
        query: Free-text query (e.g. "infrastructure lead").
        limit: Max results.
        entity_type: Optional filter (``person`` | ``project`` | ``team`` | ``glossary``).
        threshold: Minimum similarity in [0, 1] to include a result.

    Returns:
        List of dicts ``{entity_id, name, entity_type, score, profile_text}``,
        sorted by score descending. Empty list when no entity embeddings exist.
    """
    table = db.get_lance_entity_table()
    if table is None:
        return []

    query_vec = embedder.embed_query(query)
    builder = table.search(query_vec[0].tolist())
    if entity_type:
        builder = builder.where(f"entity_type = '{entity_type}'")

    raw = builder.limit(limit * 3).to_list()
    if not raw:
        return []

    conn = db.get_sqlite_conn()
    entity_ids = [int(r["entity_id"]) for r in raw]
    placeholders = ",".join("?" * len(entity_ids))
    name_rows = conn.execute(
        f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
        entity_ids,
    ).fetchall()
    name_by_id = {row["id"]: row["name"] for row in name_rows}

    results: list[dict[str, Any]] = []
    for r in raw:
        eid = int(r["entity_id"])
        # LanceDB returns cosine *distance* in `_distance`. Convert to similarity.
        score = max(0.0, min(1.0, 1.0 - float(r.get("_distance", 1.0))))
        if score < threshold:
            continue
        results.append(
            {
                "entity_id": eid,
                "name": name_by_id.get(eid, ""),
                "entity_type": r["entity_type"],
                "score": score,
                "profile_text": r["profile_text"],
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
