"""Pydantic strict models for kb data structures.

All models use ConfigDict(strict=True) which disables type coercion —
e.g., passing an int where a str is expected raises ValidationError.

Models that are mutated at runtime use frozen=False.
All other models use frozen=True for immutability.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class StrictFrozen(BaseModel):
    """Base for immutable strict models."""

    model_config = ConfigDict(strict=True, frozen=True)


class StrictMutable(BaseModel):
    """Base for mutable strict models (used where runtime mutation is needed)."""

    model_config = ConfigDict(strict=True)


# ---------------------------------------------------------------------------
# Entity models
# ---------------------------------------------------------------------------


class EntityData(StrictFrozen):
    """Parser output for entity seeding. Produced by _parse_person_file(), etc."""

    name: str
    entity_type: str
    aliases: list[str] = []
    metadata: dict[str, Any] = {}
    source_path: str | None = None
    pinned: bool = False


class Entity(StrictMutable):
    """Entity loaded from the database. Mutable for crud.py edit operations."""

    id: int
    name: str
    entity_type: str
    aliases: list[str] = []
    metadata: dict[str, Any] = {}
    source_path: str | None = None
    pinned: bool = False


class EntityMention(StrictFrozen):
    """A mention of an entity in a document."""

    entity_id: int
    mention_type: str  # "tagged", "participant", "title", "discussed", "attendee"


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------


class Chunk(StrictFrozen):
    """A single chunk of content from a document."""

    index: int
    heading: str | None
    content: str
    metadata_prefix: str = ""


class ParsedDocument(StrictMutable):
    """A parsed document ready for indexing. Mutable — indexer clears fields after processing."""

    path: str
    title: str
    date: str | None
    doc_type: str
    source_system: str
    source_id: str
    tags: list[str] = []
    content_hash: str = ""
    chunks: list[Chunk] = []
    raw_body: str | None = ""
    attendees: list[dict[str, str]] = []
    pinned: bool = False


class IndexResult(StrictMutable):
    """Statistics from an indexing run."""

    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    entities_linked: int = 0
    attendees_stored: int = 0
    entities_created: int = 0
    embeddings_skipped: bool = False
    errors: list[str] = []


# ---------------------------------------------------------------------------
# Search models
# ---------------------------------------------------------------------------


class TermMatch(StrictFrozen):
    """Which query term matched where, within a single result (issue #3 per-term detail).

    ``locations`` is a subset of ``["title", "body"]``; ``body_count`` is the number
    of whole-word occurrences in the matched chunk body (0 when matched in the title only).
    """

    term: str
    locations: list[str]
    body_count: int


class SearchExplain(StrictFrozen):
    """Per-result scoring breakdown — populated when ``explain=True``.

    Captures the component scores already computed during the search pipeline
    so callers can debug ranking without re-running the search. Phase 1 of
    issue #68 — covers the raw component data; human-readable formatting and
    'why not' diagnostics ship later. Issue #67 added the hotness fields.
    """

    fts_score: float | None  # normalized BM25 in [0,1], None if no FTS hit
    vector_score: float | None  # cosine similarity in [0,1], None if no vector hit
    fused_score: float  # score after RRF fusion (pre-recency, pre-entity-boost)
    recency_weight: float | None  # 0..1 (None when doc has no date or recency=0)
    entity_boost_applied: bool
    pre_hotness_score: float | None = None  # score before hotness blending (#67)
    hotness_score: float | None = None  # 0..1 hotness component (None when no access)
    access_count: int = 0  # times this doc was viewed/accessed (#67)
    parent_entity_score: float | None = None  # Pass 1 entity match score (#69)
    final_score: float  # post-everything — matches SearchResult.score
    source: str  # "fts_only" | "vector_only" | "both"
    fts_weight: float
    vector_weight: float
    recency: float
    matched_terms: list[TermMatch] = []  # query terms matched in this result (#3)


class SearchResult(StrictFrozen):
    """A single search result."""

    chunk_id: int
    document_id: int
    title: str
    path: str
    date: str | None
    doc_type: str
    score: float
    section: str | None
    snippet: str
    abstract: str | None = None  # extractive L0 abstract (issue #66 Phase 1)
    overview: str | None = None  # extractive L1 paragraph (issue #66 Phase 4)
    content: str | None = None  # full chunk text (populated when full_chunks=True)
    entities: list[str] = []
    tags: list[str] = []
    chunk_count: int = 1  # matching chunks from this doc (populated when dedupe=True)
    explain: SearchExplain | None = None  # populated when search(explain=True)


class TermHit(StrictFrozen):
    """How many documents contain a single query term (zero-result diagnostics, #3).

    The count is the raw FTS document frequency for the term across the whole
    corpus — it deliberately ignores date/tag/doc_type/path filters, so a term
    that exists but was filtered out of the results still reports a non-zero count.
    """

    term: str
    doc_count: int


class VectorNearMiss(StrictFrozen):
    """A closest-but-unsurfaced semantic match for a zero-result query (#3)."""

    title: str
    path: str
    similarity: float  # 1 - cosine distance; ~[0,1] for unit vectors, not hard-bounded


class ZeroResultDiagnostics(StrictFrozen):
    """Why a search returned nothing — populated only when explain=True and 0 results (#3).

    Scope: the zero-result slice of #3 (per-term FTS counts, vector near-misses,
    suggested reformulations). NOT the why-not mode or verbose mode.
    """

    term_hits: list[TermHit]
    vector_near_misses: list[VectorNearMiss]
    suggestions: list[str]


class PhaseTimings(StrictFrozen):
    """Per-phase search latency (ms) — populated when explain=True and verbose=True (#3)."""

    fts_ms: float
    vector_ms: float | None  # None on --fast (no vector phase)
    merge_ms: float  # scoring + fusion + dedup + result building


class WhyNotDiagnostics(StrictFrozen):
    """Why a specific document did/didn't surface for a query (#3 why-not mode)."""

    path: str
    # "not_indexed" | "ranked" | "below_cutoff" | "no_match" | "filtered_path"
    status: str
    detail: str
    rank: int | None = None  # 1-based score rank (ranked / below_cutoff)
    cutoff: int | None = None  # the limit, when below_cutoff


class SearchMeta(StrictFrozen):
    """Metadata about a search operation."""

    query: str
    total: int
    limit: int
    sort_by: str
    execution_ms: float
    expanded_terms: dict[str, str] = {}  # {original_term: expansion} for glossary UI

    # Explain-mode meta — populated only when search(explain=True). All optional
    # so the default JSON output stays unchanged for normal queries.
    search_mode: str | None = None  # "hybrid" | "fts_only" | "vector_only" | "fast"
    fts_variants_tried: list[str] = []
    fts_hits: int | None = None
    vector_hits: int | None = None
    both_hits: int | None = None  # chunks that appeared in both pipelines
    path_filter: str | None = None
    path_filter_doc_count: int | None = None  # doc count the path filter resolved to
    hierarchy_active: bool = False  # True when Pass 1 entity match drove Pass 2 (#69)
    pass1_entities: list[dict[str, Any]] = []  # Pass 1 entity hits (#69)
    hierarchy_alpha: float | None = None  # blend weight used in Pass 2 (#69)
    # Zero-result diagnostics (#3) — populated only when explain=True AND 0 results.
    zero_result_diagnostics: ZeroResultDiagnostics | None = None
    # Per-phase timing breakdown (#3) — populated only when explain=True AND verbose=True.
    timings: PhaseTimings | None = None
    # Why-not diagnostics (#3) — populated only when why_not=PATH is requested.
    why_not: WhyNotDiagnostics | None = None


class SearchResponse(StrictFrozen):
    """Complete search response."""

    results: list[SearchResult]
    meta: SearchMeta


# ---------------------------------------------------------------------------
# Context models
# ---------------------------------------------------------------------------


class DateRange(StrictFrozen):
    """Date range for context stats."""

    earliest: str | None
    latest: str | None


class ContextStats(StrictFrozen):
    """Stats included in context output."""

    documents: int
    entities: int
    date_range: DateRange


class ContextEntitySummary(StrictFrozen):
    """Abbreviated entity info in context output."""

    name: str
    entity_type: str
    mention_count: int


class ContextOutput(StrictFrozen):
    """Complete context output."""

    text: str
    stats: ContextStats
    entities: list[ContextEntitySummary]


# ---------------------------------------------------------------------------
# Glossary models
# ---------------------------------------------------------------------------


class GlossaryEntry(StrictFrozen):
    """A glossary term."""

    term: str
    expansion: str
    section: str


# ---------------------------------------------------------------------------
# CRUD models
# ---------------------------------------------------------------------------


class EntityTypeConfig(StrictFrozen):
    """Configuration for an entity type in the CRUD registry."""

    directory: str
    alias_label: str
    fields: list[tuple[str, str]]


# ---------------------------------------------------------------------------
# API response models (used by KnowledgeBase service class)
# ---------------------------------------------------------------------------


class EntityFact(StrictFrozen):
    """A structured fact about an entity."""

    seq: int | None = None
    text: str
    date: str | None


class EntitySummary(StrictFrozen):
    """Entity with mention count -- the common list/grid view."""

    id: int
    name: str
    entity_type: str
    aliases: list[str] = []
    metadata: dict[str, Any]
    mention_count: int
    pinned: bool


class EntityDetail(StrictFrozen):
    """Full entity record with facts and source path."""

    id: int
    name: str
    entity_type: str
    aliases: list[str]
    metadata: dict[str, Any]
    mention_count: int
    pinned: bool
    source_path: str | None
    facts: list[EntityFact]


class TimelineEntry(StrictFrozen):
    """A document mentioning an entity, ordered by date."""

    title: str
    date: str | None
    path: str
    doc_type: str | None = None
    mention_type: str | None = None


class EntityPinResult(StrictFrozen):
    """Result of toggling an entity's pin state."""

    name: str
    pinned: bool


class DocumentPinResult(StrictFrozen):
    """Result of toggling a document's pin state."""

    path: str
    pinned: bool


class MemoryTreeNode(StrictFrozen):
    """A file or directory in the memory/ tree."""

    name: str
    node_type: str  # "file" | "dir"
    path: str  # relative to memory/
    pinned: bool = False
    children: list[MemoryTreeNode] = []
    count: int = 0  # file count for dirs


# Resolve forward reference for self-referential children field.
MemoryTreeNode.model_rebuild()


# ---------------------------------------------------------------------------
# Context rendering helpers (entities enriched with mention counts)
# ---------------------------------------------------------------------------


class ContextEntity(StrictMutable):
    """Entity enriched with mention count for context rendering.

    Mutable because context.py filters and sorts these in place.
    """

    id: int
    name: str
    entity_type: str
    aliases: list[str] = []
    metadata: dict[str, Any] = {}
    source_path: str | None = None
    mention_count: int = 0
    pinned: bool = False
    updated_at: str | None = None
    last_mentioned_at: str | None = None


# ---------------------------------------------------------------------------
# Pinned document models (for context rendering)
# ---------------------------------------------------------------------------


class PinnedDocument(StrictFrozen):
    """A pinned document for context rendering."""

    path: str
    title: str
    headings: list[str] = []


# ---------------------------------------------------------------------------
# Similarity lookup (issue #71)
# ---------------------------------------------------------------------------


class SimilarityMatch(StrictFrozen):
    """A single match returned by ``kb.memory_similar()``."""

    text: str
    score: float
    match_type: str  # "fact" | "chunk"
    source_path: str | None = None
    entity_name: str | None = None
    fact_seq: int | None = None
    fact_date: str | None = None


class SimilarityResponse(StrictFrozen):
    """Result wrapper for ``kb.memory_similar()`` — matches plus quick-check flags."""

    query: str
    threshold: float
    matches: list[SimilarityMatch] = []
    has_similar: bool = False
    best_score: float = 0.0
    scope: dict[str, Any] = {}
