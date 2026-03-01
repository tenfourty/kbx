"""Pydantic strict models for kb data structures.

All models use ConfigDict(strict=True) which disables type coercion —
e.g., passing an int where a str is expected raises ValidationError.

Models that are mutated at runtime use frozen=False.
All other models use frozen=True for immutability.
"""

from __future__ import annotations

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
    metadata: dict[str, str] = {}
    source_path: str | None = None
    pinned: bool = False


class Entity(StrictMutable):
    """Entity loaded from the database. Mutable for crud.py edit operations."""

    id: int
    name: str
    entity_type: str
    aliases: list[str] = []
    metadata: dict[str, str] = {}
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
    entities: list[str] = []
    tags: list[str] = []
    chunk_count: int = 1  # matching chunks from this doc (populated when dedupe=True)


class SearchMeta(StrictFrozen):
    """Metadata about a search operation."""

    query: str
    total: int
    limit: int
    sort_by: str
    execution_ms: float
    expanded_terms: dict[str, str] = {}  # {original_term: expansion} for glossary UI


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

    id: int
    text: str
    date: str | None


class EntitySummary(StrictFrozen):
    """Entity with mention count -- the common list/grid view."""

    id: int
    name: str
    entity_type: str
    metadata: dict[str, str]
    mention_count: int
    pinned: bool


class EntityDetail(StrictFrozen):
    """Full entity record with facts and source path."""

    id: int
    name: str
    entity_type: str
    aliases: list[str]
    metadata: dict[str, str]
    mention_count: int
    pinned: bool
    source_path: str | None
    facts: list[EntityFact]


class TimelineEntry(StrictFrozen):
    """A document mentioning an entity, ordered by date."""

    title: str
    date: str | None
    path: str


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
    metadata: dict[str, str] = {}
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
