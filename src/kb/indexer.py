"""Indexing orchestrator — walks sources, chunks, embeds, stores, links entities."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kb.entities import (
    Entity,
    EntityMention,
    build_entity_patterns,
    find_entity_mentions,
    load_entities,
    seed_entities,
)
from kb.sources.meetings import walk_meetings
from kb.sources.memory import walk_memory
from kb.sync.granola import match_or_create_attendee
from kb.types import IndexResult as IndexResult

if TYPE_CHECKING:
    from kb.chunker import ParsedDocument
    from kb.db import Database
    from kb.embeddings import Embedder

# Summary embedding: for transcripts, take title + first N + last N tokens
SUMMARY_FIRST_CHARS = 3000  # ~750 tokens
SUMMARY_LAST_CHARS = 1500  # ~375 tokens


def _clear_all(db: Database, *, skip_vectors: bool = False) -> None:
    """Clear all indexed data for a full re-index.

    When skip_vectors=True, only clears SQLite tables (preserves LanceDB
    vectors that can't be recreated without embeddings).
    """
    conn = db.get_sqlite_conn()
    conn.execute("DELETE FROM document_attendees")
    conn.execute("DELETE FROM entity_mentions")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
    conn.commit()

    if not skip_vectors:
        lance_db = db.get_lance_db()
        if "chunks" in lance_db.list_tables().tables:
            lance_db.drop_table("chunks")
            db._lance_table = None


def _get_existing_hashes(db: Database) -> dict[str, str]:
    """Get path -> content_hash mapping for all existing documents."""
    conn = db.get_sqlite_conn()
    rows = conn.execute("SELECT path, content_hash FROM documents").fetchall()
    return {row["path"]: row["content_hash"] for row in rows}


def _delete_document(db: Database, path: str) -> None:
    """Delete a document and its chunks/mentions by path."""
    conn = db.get_sqlite_conn()
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
    if row:
        doc_id = row["id"]
        conn.execute("DELETE FROM document_attendees WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM entity_mentions WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def _flush_lance_batch(db: Database, batch: list[dict[str, Any]]) -> None:
    """Write a batch of rows to LanceDB."""
    if not batch:
        return

    table = db.get_lance_table()
    if table is None:
        db.ensure_lance_table(batch)
    else:
        import pyarrow as pa

        from kb.db import LANCE_SCHEMA

        table.add(pa.Table.from_pylist(batch, schema=LANCE_SCHEMA))


def _build_entities_info(
    mentions: list[EntityMention], entities: list[Entity]
) -> list[tuple[str, str, str]]:
    """Build entity info tuples for chunk prefixes from mentions.

    Returns list of (name, entity_type, extra) tuples.
    """
    entity_map = {e.id: e for e in entities}
    seen = set()
    info = []
    for mention in mentions:
        if mention.entity_id in seen:
            continue
        seen.add(mention.entity_id)
        e = entity_map.get(mention.entity_id)
        if e:
            extra = e.metadata.get("team", e.metadata.get("status", ""))
            info.append((e.name, e.entity_type, extra))
    return info


def _make_summary_text(
    doc: ParsedDocument,
    entities_info: list[tuple[str, str, str]] | None = None,
) -> str | None:
    """Create summary text for document-level embedding.

    - Notes (typically short): use entire raw_body
    - Transcripts (long): title + first 2K tokens + last 1K tokens
    - Memory files: already whole-file chunks, skip summary
    """
    if doc.doc_type in ("memory_person", "memory_project", "memory_context", "memory_glossary"):
        return None

    body = doc.raw_body
    if not body or not body.strip():
        return None

    # Build entity suffix for the summary prefix
    entity_suffix = ""
    if entities_info:
        parts = [f"{name} ({etype})" for name, etype, _ in entities_info]
        entity_suffix = " | Entities: " + ", ".join(parts)

    if doc.doc_type == "transcript" and len(body) > SUMMARY_FIRST_CHARS + SUMMARY_LAST_CHARS:
        # For long transcripts, take beginning + end
        return (
            f"[Summary: {doc.title} | Date: {doc.date or 'unknown'}{entity_suffix}]\n"
            + body[:SUMMARY_FIRST_CHARS]
            + "\n...\n"
            + body[-SUMMARY_LAST_CHARS:]
        )

    # Notes or short transcripts: use full body with prefix
    return f"[Summary: {doc.title} | Date: {doc.date or 'unknown'}{entity_suffix}]\n{body}"


def index_all(
    db: Database,
    embedder: Embedder | None = None,
    project_root: Path = Path("."),
    full: bool = False,
    batch_size: int | None = None,
    *,
    memory_only: bool = False,
    skip_seed: bool = False,
) -> IndexResult:
    """Index all sources. Incremental by default, full if requested.

    When embedder is None, runs text-only indexing: SQLite metadata, FTS5,
    entity linking, and attendee matching — but skips vector embeddings
    and LanceDB writes.

    Uses large-batch embedding for throughput: collects chunks across many
    documents, embeds them all at once, then stores results.
    batch_size overrides the default embed batch size (8 MPS, 16 CPU).
    """
    result = IndexResult()
    if embedder is None:
        result.embeddings_skipped = True

    if full:
        _clear_all(db, skip_vectors=(embedder is None))

    # Seed entities (skip during auto-reindex to avoid destructive upsert)
    if not skip_seed:
        seed_entities(db, project_root)
    entities = load_entities(db)
    entity_patterns = build_entity_patterns(entities)

    # Get existing hashes for incremental mode
    existing_hashes = _get_existing_hashes(db)

    # Collect all documents from all sources, filter to those needing indexing
    docs_to_index: list[ParsedDocument] = []
    if not memory_only:
        for doc in walk_meetings(project_root):
            if doc.path in existing_hashes and existing_hashes[doc.path] == doc.content_hash:
                result.documents_skipped += 1
                continue
            if doc.path in existing_hashes:
                _delete_document(db, doc.path)
            docs_to_index.append(doc)
    for doc in walk_memory(project_root):
        if doc.path in existing_hashes and existing_hashes[doc.path] == doc.content_hash:
            result.documents_skipped += 1
            continue
        if doc.path in existing_hashes:
            _delete_document(db, doc.path)
        docs_to_index.append(doc)
    del existing_hashes

    conn = db.get_sqlite_conn()

    # Flush interval: accumulate this many docs, then embed + write all at once.
    # With embedder: 50 docs balances GPU utilization vs memory.
    # Without embedder (--no-embed): no GPU memory pressure, so flush less
    # often to reduce commit overhead (~6 commits vs ~60 for 3000 docs).
    FLUSH_DOCS = 50 if embedder is not None else 500
    lance_batch: list[dict[str, Any]] = []

    # Accumulate: (doc_index, chunk_text) for batch embedding
    pending_texts: list[str] = []
    # Track which doc and chunk each text belongs to: (doc_idx_in_batch, chunk_local_idx)
    pending_map: list[tuple[int, int]] = []
    # Store per-doc SQLite results: doc_idx -> (doc_id, chunk_ids, entity_ids_json)
    doc_results: dict[int, tuple[int | None, list[int | None], str]] = {}

    def _flush_embeddings() -> None:
        """Embed all pending texts and write to LanceDB batch."""
        nonlocal pending_texts, pending_map

        if not pending_texts or embedder is None:
            pending_texts = []
            pending_map = []
            return

        embeddings = embedder.embed(pending_texts, batch_size=batch_size)

        for idx, (doc_idx, chunk_local_idx) in enumerate(pending_map):
            doc_id, chunk_ids, entity_ids_json = doc_results[doc_idx]
            doc = docs_to_index[doc_idx]
            chunk_id = chunk_ids[chunk_local_idx]
            assert chunk_id is not None  # lastrowid always set after INSERT
            assert doc_id is not None
            lance_batch.append(
                {
                    "chunk_id": int(chunk_id),
                    "embedding": embeddings[idx].tolist(),
                    "doc_type": doc.doc_type or "",
                    "doc_date": doc.date or "",
                    "tags": json.dumps(doc.tags),
                    "document_id": int(doc_id),
                    "entity_ids": entity_ids_json,
                }
            )

        # Release references immediately
        del embeddings
        pending_texts = []
        pending_map = []

    total = len(docs_to_index)

    for doc_idx, doc in enumerate(docs_to_index):
        try:
            # Insert document row
            cursor = conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.path,
                    doc.title,
                    doc.date,
                    doc.doc_type,
                    doc.source_system,
                    doc.source_id,
                    json.dumps(doc.tags),
                    doc.content_hash,
                    len(doc.chunks),
                    1 if doc.pinned else 0,
                ),
            )
            doc_id = cursor.lastrowid

            # Insert section-level chunks into SQLite (batched)
            chunk_data = [
                (doc_id, chunk.index, chunk.heading, chunk.content, chunk.metadata_prefix)
                for chunk in doc.chunks
            ]
            chunk_texts = [chunk.content for chunk in doc.chunks]
            if chunk_data:
                conn.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, heading, content, metadata_prefix) VALUES (?, ?, ?, ?, ?)",
                    chunk_data,
                )
            # Retrieve chunk IDs (single indexed query)
            chunk_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                    (doc_id,),
                ).fetchall()
            ]

            # Entity linking (before summary so we can include entity info in summary prefix)
            section_content = " ".join(chunk_texts)
            mentions = find_entity_mentions(
                doc.title, doc.tags, section_content, entities, cached_patterns=entity_patterns
            )
            entity_id_set = {m.entity_id for m in mentions}
            result.entities_linked += len(mentions)

            # Batch insert entity mentions
            if mentions:
                conn.executemany(
                    "INSERT OR IGNORE INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                    [(m.entity_id, doc_id, m.mention_type) for m in mentions],
                )

            # Process attendees (if any)
            if doc.attendees:
                for att in doc.attendees:
                    att_result = match_or_create_attendee(att, db, project_root)
                    if att_result is None:
                        continue
                    matched_by = (
                        "created"
                        if att_result.get("created")
                        else "email"
                        if att.get("email") and att_result.get("matched")
                        else "name"
                        if att_result.get("matched")
                        else None
                    )
                    conn.execute(
                        """INSERT OR IGNORE INTO document_attendees
                           (document_id, entity_id, name, email, matched_by)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            doc_id,
                            att_result["entity_id"],
                            att.get("name", ""),
                            att.get("email", ""),
                            matched_by,
                        ),
                    )
                    result.attendees_stored += 1
                    if att_result.get("created"):
                        result.entities_created += 1
                    if att_result.get("entity_id"):
                        conn.execute(
                            """INSERT OR IGNORE INTO entity_mentions
                               (entity_id, document_id, mention_type)
                               VALUES (?, ?, 'attendee')""",
                            (att_result["entity_id"], doc_id),
                        )
                        entity_id_set.add(att_result["entity_id"])

            # Update last_mentioned_at for all mentioned/attendee entities
            if entity_id_set and doc.date:
                for eid in entity_id_set:
                    conn.execute(
                        """UPDATE entities SET last_mentioned_at = MAX(
                            COALESCE(last_mentioned_at, ''), ?
                        ) WHERE id = ?""",
                        (doc.date, eid),
                    )

            entity_ids_json = json.dumps(sorted(entity_id_set))

            # Document-level summary chunk (with entity info in prefix)
            entities_info = _build_entities_info(mentions, entities)
            summary_text = _make_summary_text(doc, entities_info)
            if summary_text:
                summary_idx = len(doc.chunks)
                cursor = conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                    (doc_id, summary_idx, "__document__", summary_text),
                )
                chunk_ids.append(cursor.lastrowid)
                chunk_texts.append(summary_text)
                # Fix chunk_count to include the summary chunk (was off-by-one)
                conn.execute(
                    "UPDATE documents SET chunk_count = ? WHERE id = ?",
                    (len(chunk_ids), doc_id),
                )

            result.chunks_created += len(chunk_ids)

            # Store results for embedding phase
            doc_results[doc_idx] = (doc_id, chunk_ids, entity_ids_json)

            # Queue chunk texts for batch embedding (no intermediate flush)
            for local_idx, text in enumerate(chunk_texts):
                pending_texts.append(text)
                pending_map.append((doc_idx, local_idx))

            result.documents_indexed += 1

        except Exception as e:
            result.errors.append(f"{doc.path}: {e}")

        # Free heavy content from already-processed docs
        doc.raw_body = None
        doc.chunks = []

        # Every FLUSH_DOCS: embed all accumulated texts, commit, flush LanceDB
        if doc_idx % FLUSH_DOCS == (FLUSH_DOCS - 1):
            _flush_embeddings()
            conn.commit()
            if lance_batch:
                _flush_lance_batch(db, lance_batch)
                lance_batch.clear()
            doc_results.clear()
            # Reload entities if new ones were auto-created from attendees
            if result.entities_created > 0:
                entities = load_entities(db)
                entity_patterns = build_entity_patterns(entities)
            if embedder:
                embedder.release_gpu_memory()
            gc.collect()
            print(
                f"  [{doc_idx + 1}/{total}] documents processed...",
                file=sys.stderr,
                flush=True,
            )

    # Final flushes
    _flush_embeddings()
    conn.commit()

    if lance_batch:
        _flush_lance_batch(db, lance_batch)

    # Compact LanceDB: merge small files and prune old versions.
    # This is the equivalent of VACUUM — without it, append-heavy workloads
    # accumulate old versions and small fragment files on disk.
    if embedder is not None:
        lance_table = db.get_lance_table()
        if lance_table is not None:
            from datetime import timedelta

            try:
                lance_table.optimize(cleanup_older_than=timedelta(0))
                print("  LanceDB compacted.", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"  LanceDB compaction failed: {exc}", file=sys.stderr, flush=True)

    return result
