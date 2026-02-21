# Chunking Strategy

## Overview

Documents are split into chunks for embedding and search. Each chunk includes a metadata prefix that enriches the embedding with contextual information.

## Chunk Types

### Meeting Notes

**Strategy**: Split on `##` and `###` headers. Each section becomes one or more chunks.

| Parameter | Value |
|-----------|-------|
| MAX_SECTION_TOKENS | 2000 |
| Split method | Paragraph breaks within long sections |

Metadata prefix format:
```
[Meeting: Wren / Soren | Date: 2026-01-27 | Section: Performance Review | Entities: Soren Chen (person), Pipeline Improvements (project)]
```

Empty/trivial sections (e.g. "No personal notes available.") are skipped.

### Transcripts

**Strategy**: Sliding window over speaker turns.

| Parameter | Value |
|-----------|-------|
| TARGET_TRANSCRIPT_TOKENS | 1000 |
| TRANSCRIPT_OVERLAP_TOKENS | 100 |
| Speaker format | `**Speaker:** text` parsed into turns |

Speaker replacement: "System" label is replaced with the participant name parsed from the meeting title (e.g. "Wren / Soren" → "Soren" replaces "System").

Metadata prefix format:
```
[Transcript: Wren / Soren | Date: 2026-01-27 | Entities: ...]
```

### Memory Files

**Strategy**: Entire file as a single chunk (no splitting). Memory files are typically short (< 500 tokens).

Exception: `glossary.md` is chunked per table row, with the term as heading and `Term: Expansion. Notes` as content.

## Document-Level Summary Embeddings

One additional chunk per document with `heading = "__document__"`. Captures the document's overall topic for broad queries like "meetings about Rust migration".

| Doc Type | Summary Strategy |
|----------|-----------------|
| Notes | Full raw_body (typically < 1000 tokens) |
| Transcripts | First 3000 chars + last 1500 chars (~1125 tokens total) |
| Memory files | No summary (already whole-file chunks) |

Summary chunks include entity info in their prefix and compete naturally in search results via RRF.

## Entity-Enriched Prefixes

When entities are linked to a document during indexing, their names and types are appended to chunk prefixes. This helps the embedding model understand who and what a chunk is about.

Entity info is derived from the entity graph after linking (tags → participants → discussed), so entity linking runs before summary chunk creation.
