# kb — Ideas & Next Steps

Inspired by analysis of [qmd](https://github.com/tobi/qmd) (Tobi Lutke's local markdown search engine) and our own roadmap. These are improvements to adopt after the core phases (1-6) are complete.

## Embedding Model: Upgrade to Qwen3-Embedding-0.6B

### Model Comparison

| | **nomic-embed-text-v1.5** (current) | **embedding-gemma-300M** (qmd) | **Qwen3-Embedding-0.6B** (recommended) |
|---|---|---|---|
| Parameters | 137M | 300M | 600M |
| MTEB English (768d) | 62.28 | 69.67 | **70.70** |
| MMTEB Multilingual | not ranked | 61.15 | **64.33** |
| Context length | 8,192 tokens | 2,048 tokens | **32,768 tokens** |
| Dimensions | 768 (MRL to 64) | 768 (MRL to 128) | **32-1024** (MRL) |
| Model size | ~550MB (F32) | ~300MB (Q8) | ~600MB (Q8) |
| Languages | English-focused | 100+ | **100+** |
| Training data | ~5M contrastive pairs | ~320B tokens | Qwen3-0.6B base |
| Released | Feb 2024 | May 2025 | **Jun 2025** |
| sentence-transformers | Yes | Yes | **Yes** |
| Instruction-aware | Prefix only | Prefix only | **Full task instructions** (+1-5%) |

### Why Qwen3-Embedding-0.6B

- **Best MTEB English score** (70.70) of any model under 1B params — 8 points above nomic, 1 point above gemma
- **32K context** — can embed entire meeting notes files (largest is ~4K tokens) and even medium transcripts in a single pass
- **Instruction-aware** — custom task instructions give 1-5% improvement on domain-specific retrieval
- **Strong multilingual** — 64.33 MMTEB, trained on 100+ languages
- **Drop-in swap** — `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")`, same API as nomic

### French Language Considerations

The corpus is ~95% English with French names, phrases, and occasional French content. From MTEB-French benchmarks (May 2024), the top French retrieval models were BGE-M3 (0.81), multilingual-e5-large (0.76), sentence-camembert-large (0.72). The top 9 models were statistically equivalent (p=0.05). Qwen3-Embedding (June 2025, 100+ languages, MMTEB 64.33) should handle mixed English/French well — it's trained on far more multilingual data than any of the models in that benchmark.

### Upgrade Plan

**When:** After Phase 6, before CLAUDE.md reorg (Phase 7).

**Steps:**
1. Change `MODEL_NAME` in `kb/embeddings.py` to `Qwen/Qwen3-Embedding-0.6B`
2. Update `EMBEDDING_DIM` (consider 1024 for higher quality, or keep 768 for compatibility)
3. Run full re-embed: `kb index run --full`
4. Benchmark search quality on 10 test queries before/after

**Also adjust chunking strategy** (see below).

Sources: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Qwen3 blog](https://qwenlm.github.io/blog/qwen3-embedding/), [embedding-gemma-300m](https://huggingface.co/google/embeddinggemma-300m), [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [MTEB-French](https://arxiv.org/html/2405.20468v2)

---

## Chunking Strategy: Optimise for 32K Context

### Current Corpus Stats

| | Count | Avg size | Largest | Avg tokens (est.) |
|---|---|---|---|---|
| Meeting notes | 1,548 | 3.8 KB | 15.5 KB | ~960 |
| Transcripts | 1,548 | 73 KB | 378 KB | ~18,250 |
| Memory files | ~39 | varies | — | — |
| **Total** | **3,135** | — | — | — |

### Current Chunking (designed for nomic's 8K limit)

| Doc type | Strategy | Chunk size |
|---|---|---|
| Notes | Split on `##`/`###` headers | 500-800 tokens |
| Transcripts | Sliding window on speaker turns | 400-600 tokens, 50 overlap |
| Memory files | Whole file | Full file |
| Glossary | Each table row | Variable |

### What 32K Context Enables

With Qwen3's 32K context, we can embed much larger text spans. But **bigger chunks aren't automatically better** — the key is matching chunk granularity to query granularity:

#### 1. Document-Level Embeddings (new, high value)

**Add a "summary embedding" per document alongside chunk-level embeddings.** This is the biggest win.

- **Meeting notes** (avg ~960 tokens): Embed the **entire file** as one additional vector. 100% of notes fit within 32K. This catches broad queries like "meetings about Helix refactor" that might not match any single section header.
- **Transcripts** (avg ~18K tokens): Embed a **title + first 2K tokens + last 1K tokens** composite as one summary vector. Even the largest transcript (378KB / ~94K tokens) exceeds 32K, so full-doc embedding isn't possible for all transcripts, but a summary embedding still helps.
- **Memory files**: Already embedded as whole files. No change.

**Implementation:** In `kb/indexer.py`, after chunk-level embedding, create one additional "summary" chunk per document with `heading = "__document__"`. Store in the same LanceDB table. During search, these compete naturally via RRF — they'll surface for broad queries and lose to section-level chunks for specific queries.

```python
# In indexer.py, after processing chunks:
if doc.doc_type in ("notes", "transcript"):
    summary_text = _build_summary_text(doc)  # title + full content (notes) or title + head + tail (transcripts)
    summary_chunk = Chunk(index=len(doc.chunks), heading="__document__", content=summary_text)
    # Embed and store alongside regular chunks
```

**Effort:** Small. ~20 lines in indexer.py. High impact on recall for broad queries.

#### 2. Larger Transcript Windows (modest value)

Increase transcript sliding window from 400-600 tokens to **800-1200 tokens** with 100 token overlap. Rationale:
- More conversational context per chunk = better semantic embeddings
- Speaker turn boundaries are preserved (still split on turns, not mid-sentence)
- qmd uses 800 tokens with 15% overlap (120 tokens) — similar to what we'd adopt
- Reduces total chunk count (~30% fewer chunks), speeding up indexing and search

**Change in `kb/chunker.py`:**
```python
TARGET_TRANSCRIPT_TOKENS = 1000  # was 500
TRANSCRIPT_OVERLAP_TOKENS = 100  # was 50
```

**Effort:** Trivial. Two constant changes.

#### 3. Notes Chunking: Keep as-is (no change needed)

Header-based splitting is already the right strategy for notes. Sections are semantically coherent units. The avg section is well under 800 tokens. Increasing the max would just merge unrelated sections, hurting precision.

The one case where this matters: very long sections (>800 tokens) currently get split on paragraph breaks. With 32K context, we could raise `MAX_SECTION_TOKENS` from 800 to 2000 and avoid splitting long sections. But this is rare enough to not prioritise.

#### 4. Instruction-Aware Embeddings (new capability)

Qwen3-Embedding supports custom task instructions that improve retrieval by 1-5%. We can use different instructions for indexing vs searching:

**Indexing (document side):**
```python
model.encode(chunks, prompt_name="document")
# Or with custom instruction:
# "Represent this meeting note section for retrieval"
```

**Searching (query side):**
```python
model.encode([query], prompt_name="query")
# Or with custom instruction:
# "Find meeting notes, transcripts, and documents relevant to this query"
```

This replaces the current nomic-style `search_document:` / `search_query:` prefix system with the model's native instruction support.

**Effort:** Small. Change the `embed()` and `embed_query()` methods in `kb/embeddings.py`.

### Recommended Chunking Changes (ordered)

| Change | When | Effort | Impact |
|---|---|---|---|
| Document-level summary embeddings | Model upgrade | Small | High — fixes broad query recall |
| Larger transcript windows (1000 tokens) | Model upgrade | Trivial | Medium — better context per chunk |
| Instruction-aware embedding | Model upgrade | Small | Medium — 1-5% retrieval improvement |
| Raise `MAX_SECTION_TOKENS` to 2000 | Model upgrade | Trivial | Low — rare edge case |

All four changes should be done together when switching to Qwen3-Embedding-0.6B, since a full re-embed is needed anyway.

---

## Priority 1: Search Pipeline Improvements (Phase 4 enhancements)

### 1.1 Strong Signal Fast Path

**From qmd:** Before running vector search, probe FTS5 first. If the top BM25 result scores ≥ 0.85 with a ≥ 0.15 gap to the second result, skip the vector search entirely.

**Why it matters:** Most queries against this corpus are specific — "MFA implementation", "Wren / Soren January 27". FTS5 handles these perfectly. Skipping vector search saves ~200ms per query and avoids irrelevant semantic matches muddying the results.

**Implementation:**
```python
def search(query, ...):
    fts_results = search_fts(query, limit=20)
    top = fts_results[0].score if fts_results else 0
    second = fts_results[1].score if len(fts_results) > 1 else 0

    if top >= 0.85 and (top - second) >= 0.15:
        # Strong signal — FTS alone is sufficient
        return fts_results[:limit]

    # Otherwise, run full hybrid pipeline
    vec_results = search_vector(query, limit=20)
    return reciprocal_rank_fusion([fts_results, vec_results])
```

**Effort:** Small. Add to `kb/search.py` during Phase 4.

### 1.2 Weighted RRF Lists

**From qmd:** Not all ranked lists are equal. The original FTS and original vector search should get 2x weight in RRF vs expanded/auxiliary queries.

**Implementation:** Add a `weights` parameter to the RRF function:
```python
def reciprocal_rank_fusion(ranked_lists, weights=None, k=60):
    for list_idx, results in enumerate(ranked_lists):
        weight = weights[list_idx] if weights else 1.0
        for rank, result in enumerate(results):
            rrf_score += weight / (k + rank + 1)
```

Also add top-rank bonuses: rank 1 gets +0.05, ranks 2-3 get +0.02. This prevents the reranker from destroying high-confidence retrieval results.

**Effort:** Small. Part of the RRF implementation in Phase 4.

### 1.3 LLM Result Caching

**From qmd:** Cache embedding lookups and reranking scores in SQLite keyed by `(query, file, model, content_hash)`. Repeated or similar queries become near-instant.

**Implementation:** Add a `cache` table to `metadata.db`:
```sql
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
```

Cache key = SHA256 of `json.dumps({"op": "rerank", "query": query, "file": path, "chunk": chunk_text})`.

**Effort:** Small. Add to `kb/db.py` schema, use in search.

---

## Priority 2: Reranking (Post-Phase 4)

### 2.1 Cross-Encoder Reranking

**From qmd:** After RRF fusion, take the top 30-40 candidates and rerank using a cross-encoder model. qmd uses Qwen3-reranker-0.6B (~640MB GGUF).

**Critical insight from qmd:** Rerank **chunks**, not full documents. Sending full meeting transcripts to a reranker is O(tokens) — slow and wasteful. Instead:
1. Chunk each candidate document
2. Pick the best chunk per document (by keyword overlap with query)
3. Send only that chunk to the reranker

**Position-aware blending** (qmd's formula):
```python
def blend_scores(rrf_rank, reranker_score):
    if rrf_rank <= 3:
        rrf_weight = 0.75  # Trust retrieval for top results
    elif rrf_rank <= 10:
        rrf_weight = 0.60
    else:
        rrf_weight = 0.40  # Trust reranker for lower ranks

    rrf_score = 1.0 / rrf_rank
    return rrf_weight * rrf_score + (1 - rrf_weight) * reranker_score
```

This protects exact keyword matches in top positions while allowing the reranker to promote semantically relevant results from lower ranks.

**Model options for Python:**
- `sentence-transformers` cross-encoder (already in our deps): `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, fast)
- Qwen3-reranker via `llama-cpp-python` (matches qmd's approach, ~640MB)
- `flashrank` (lightweight, pip-installable reranker)

**Effort:** Medium. New module `kb/reranker.py`, optional dependency, `--rerank` flag on search.

### 2.2 Query Expansion

**From qmd:** Use a local LLM to generate alternative query phrasings before searching. qmd fine-tuned a 1.7B model for this. The expanded queries are typed:
- `lex` — keyword variant, routed to FTS only
- `vec` — semantic variant, routed to vector only
- `hyde` — hypothetical document, routed to vector only

**Why it matters:** For vague queries like "what was that discussion about team performance", expansion generates specific variants ("performance review", "team evaluation", "improvement plan") that hit both keyword and semantic matches.

**Pragmatic alternative:** We don't need a fine-tuned model. Options:
1. **Entity-aware expansion** — extract entity names from query, add aliases as search terms. "Linnea performance" → also search "Linnea Nguyen", "platform performance". This leverages our entity graph (which qmd doesn't have).
2. **Synonym expansion via embeddings** — embed the query, find nearest entity/term embeddings, add those as additional search terms.
3. **LLM expansion via API** — call Claude/OpenAI to generate 2-3 query variants. Slower but higher quality. Could cache aggressively.

**Effort:** Medium-Hard. Entity-aware expansion (option 1) is easiest and plays to kb's strengths.

---

## Priority 3: Operational Features

### 3.1 Index Health Monitoring

**From qmd:** Warn users when the index is stale or incomplete:
- "Warning: 45 documents (12%) need embeddings. Run `kb index run`."
- "Tip: Index last updated 14 days ago. Run `kb index run` to refresh."

**Implementation:** Add to `kb index status` output and optionally emit on every search.

**Effort:** Small. Use existing `documents` table timestamps.

### 3.2 Content-Hash Document IDs (docids)

**From qmd:** Each document gets a 6-character content hash ID (`#abc123`). These are stable, short, and useful for referencing specific documents in conversation:
```
qmd get "#abc123"
kb view #abc123
```

**Current kb design:** Uses auto-increment `id` from SQLite, which is fine for internal use but not memorable or stable across re-indexes.

**Implementation:** Add `docid` column (first 6 chars of `content_hash`) to `documents` table. Support `kb view #abc123` syntax.

**Effort:** Small. Schema addition + CLI parsing.

### 3.3 MCP Server Mode

**From qmd:** `qmd mcp` exposes search as MCP tools. Claude Desktop, Cursor, and other MCP clients can use it directly without Bash.

**When to add:** After Phase 6 when the CLI is stable. Doesn't replace the CLI — the CLI remains primary for context-window control. MCP is a convenience layer for tools that prefer tool-calling over Bash.

**Implementation:** `kb mcp` command using `mcp` Python package. Expose: `kb_search`, `kb_entity_find`, `kb_entity_timeline`, `kb_view`.

**Effort:** Medium. New command + MCP protocol handling. Consider as Phase 8.

---

## Priority 4: Longer-Term Ideas

### 4.1 Typed Query Routing

**From qmd:** Don't run every query through every search backend. If a query is clearly a keyword lookup ("GIM"), skip vector. If it's clearly semantic ("what was the discussion about improving deployment"), skip FTS.

**How qmd does it:** The query expansion model outputs typed variants (`lex`/`vec`/`hyde`), and each type is routed to the appropriate backend.

**Simpler approach for kb:** Use heuristics:
- Query is a known entity name → entity lookup, skip search
- Query has quotes → FTS only (exact match intent)
- Query is a question → vector + FTS hybrid
- Query is 1-2 words matching a glossary term → FTS only

**Effort:** Small-Medium. Heuristic routing in `kb/search.py`.

### 4.2 Similar Document Discovery

**From qmd:** `findSimilarFiles()` — given a document, find others with similar embeddings. Useful for "what other meetings covered this topic?"

**Implementation:** Embed the target document, query LanceDB for nearest vectors excluding the source document. Could surface as `kb similar <path>`.

**Effort:** Small. LanceDB already supports this.

### 4.3 Multi-Index Support

**From qmd:** Named indexes (`qmd --index work`, `qmd --index personal`). Useful if you want separate indexes for different contexts.

**Relevance for kb:** Low priority — the single project index is the use case. But could matter if kb is ever used for multiple projects.

### 4.4 Claude Code Plugin / Skill

Once `kb` is stable, create a Claude Code skill that teaches Claude when and how to use `kb` commands. This replaces the manual CLAUDE.md section with an auto-loaded skill.

```yaml
# skills/kb/SKILL.md
Use `kb` CLI for deep context lookups:
- People: `kb entity find "name" --json`
- Meetings: `kb search "topic" --json --fields title,date,snippet --limit 5`
- Timeline: `kb entity timeline "name" --from DATE --json`
```

**Effort:** Small. Skill definition only.

---

## Priority 5: Additional Ideas from qmd Source Code

Mined from a thorough read of qmd's `store.ts`, `qmd.ts` (CLI), `formatter.ts`, `mcp.ts`, and `collections.ts`.

### 5.1 Snippet Extraction with Line Context

**From qmd:** Search results include a smart snippet — not just the matching text, but a diff-style header showing where it is in the document:

```
@@ -42,3 @@ (41 before, 28 after)
Soren received 'needs improvement' rating...
```

The snippet is extracted by finding the line with the most query term overlap, then returning ±1-3 surrounding lines. The `@@ -start,count @@` header tells the AI exactly where in the document to look if it needs more context.

**Why it matters for kb:** Our current plan returns `snippet` as just a text fragment. Adding line position info means Claude can do `kb view <path> --offset 40 --limit 10` to expand around a match, rather than reading the whole file.

**Effort:** Small. Add line-position tracking to search results.

### 5.2 "Did You Mean?" Fuzzy File Matching

**From qmd:** When `qmd get <path>` fails to find a document, it runs Levenshtein distance matching against all indexed paths and suggests the closest matches:

```
Document not found: meeting-notes.md

Did you mean one of these?
  - meetings/2026/01/27/meeting-notes_f744ed8c.notes.md
  - meetings/2025/12/15/meeting-notes_a2b3c4d5.notes.md
```

**Implementation:** `findSimilarFiles()` computes edit distance against all paths, returns top-5 within distance ≤3.

**Effort:** Small. Add to `kb view` error handling.

### 5.3 Content-Addressable Storage (Separate Content Table)

**From qmd:** Documents and content are separated:
- `documents` table: path, title, hash, collection, timestamps, `active` flag
- `content` table: `hash → body` (content-addressable, deduped)

When a document is updated, the old content hash stays in the content table until cleanup. This means:
- Deduplication is free (identical files share one content row)
- Re-indexing only re-embeds changed content hashes
- Rollback is possible (deactivate document, old content still exists)

**Relevance for kb:** Our schema stores chunks inline. Content-addressable storage would help with incremental re-indexing — only embed new/changed content hashes. Consider for a future schema revision.

**Effort:** Medium. Schema change, affects indexer.

### 5.4 Database Cleanup Commands

**From qmd:** Dedicated maintenance operations:
- `cleanupOrphanedContent()` — remove content not referenced by any active document
- `cleanupOrphanedVectors()` — remove embeddings for deleted/changed documents
- `deleteInactiveDocuments()` — purge soft-deleted documents
- `vacuumDatabase()` — reclaim space

**Implementation for kb:**
```
kb index cleanup    # Remove orphaned chunks, vectors, and compact DB
kb index vacuum     # VACUUM the SQLite database
```

**Effort:** Small. SQL queries against existing schema.

### 5.5 Dynamic MCP Server Instructions

**From qmd:** The MCP server generates its `instructions` field dynamically at startup based on the actual index state:

```
QMD is your local search engine over 3096 markdown documents.

Collections (scope with `collection` parameter):
  - "meetings" (3096 docs) — Meeting notes and transcripts

Note: 45 documents need embedding. Run `qmd embed` to update.

Search:
  - `search` (~30ms) — keyword and exact phrase matching.
  - `vector_search` (~2s) — meaning-based.
  - `deep_search` (~10s) — auto-expands, reranks.
```

This means the AI knows what's available without making a single tool call. Includes collection descriptions, document counts, capability gaps, and a "which tool to use" escalation ladder.

**Relevance for kb MCP:** When we add MCP (Phase 8), generate instructions that include entity counts, source types, and date ranges. e.g. "67 entities (people, projects, companies). Meetings from 2024-06 to 2026-02."

**Effort:** Small. Part of MCP implementation.

### 5.6 Three-Tier Search Escalation

**From qmd:** The CLI and MCP expose three separate search commands with explicit latency budgets:
1. `search` (~30ms) — BM25 keyword only
2. `vsearch` (~2s) — vector only
3. `query` (~10s) — full hybrid + expansion + reranking

**Why this matters:** Users/agents can choose the right trade-off. Most queries are specific enough for keyword-only. Vague queries justify the 10s latency.

**For kb:** Currently planning a single `kb search` command. Consider adding `--mode fast|deep` or separate subcommands:
```
kb search "MFA implementation"           # Default: hybrid (vector + FTS + RRF)
kb search "MFA implementation" --fast    # FTS only, ~30ms
kb search "MFA implementation" --deep    # Full pipeline with reranking, ~10s
```

**Effort:** Small. Flag routing in `kb/search.py` and `kb/cli.py`.

### 5.7 Custom Update Commands per Source

**From qmd:** Collections can specify a bash command that runs before re-indexing:

```yaml
collections:
  meetings:
    path: ~/meetings
    update: "git pull origin main"
```

When you run `qmd update`, it executes the `update` command first, then re-indexes.

**For kb:** The `kb ingest` command already wraps `organise_granola.py` + index. But a generic `update` hook per source would let future sources (calendar sync, Slack export) define their own pre-index refresh. e.g.:
```python
# In sources/meetings.py
UPDATE_COMMAND = "python meetings/organise_granola.py"
```

**Effort:** Small. Config + subprocess call in indexer.

### 5.8 File Path Line References

**From qmd:** `qmd get "file.md:100"` starts output from line 100. `qmd get "file.md:100" --max-lines 20` returns lines 100-120. Also supports `--line-numbers` to prefix each line.

**For kb:** `kb view <path>` should support `kb view "path.md:42"` and `--offset`/`--limit` flags. Critical for the snippet → full-context workflow: search returns a snippet at line 42, Claude does `kb view "path.md:42" --limit 20` to get surrounding context.

**Effort:** Small. CLI argument parsing + file slicing.

### 5.9 Multi-Get with Glob Patterns

**From qmd:** `qmd multi-get "journals/2025-05*.md"` returns multiple documents matching a glob, with size limits to prevent token blowup (`--max-bytes 10KB` default, skip large files).

**For kb:** Add to `kb view`:
```
kb view "meetings/organised/2026/01/*" --format jsonl --fields title,date --max-size 10k
```

This enables "show me all meetings from January" without searching.

**Effort:** Small-Medium. Glob expansion + size filtering.

### 5.10 Progress Reporting (Terminal UX)

**From qmd:** During embedding and reranking, qmd shows:
- Terminal progress bar via OSC 9;4 escape sequences (shows in terminal tab title)
- ETA formatting ("2m 30s remaining")
- Cursor hide/show during progress
- Spinner states (indeterminate for reranking)

**For kb:** During `kb index run` (which can take minutes for 3000+ docs), show progress:
```
Indexing: [████████████████░░░░] 78% (2,412/3,096 docs) ETA: 45s
Embedding: [██████░░░░░░░░░░░░░░] 30% (912/3,096 chunks) ETA: 2m 10s
```

Use Python's `tqdm` (already common) or simple stderr progress.

**Effort:** Small. `tqdm` or manual progress bar in indexer.

### 5.11 WAL Mode and Pragmas

**From qmd:** Explicitly sets SQLite pragmas:
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
```

WAL (Write-Ahead Logging) allows concurrent reads during writes — important when search runs while indexing is in progress.

**For kb:** Check if `kb/db.py` sets WAL mode. If not, add it.

**Effort:** Trivial. One pragma.

### 5.12 XML Output Format

**From qmd:** Supports `--xml` output with proper escaping. Useful for Claude which handles XML well in its context window — XML-tagged search results are easier for the model to parse than raw JSON.

**For kb:** Consider adding `--format xml` alongside the existing json/jsonl/csv/table formats. Example:
```xml
<result docid="#abc123" score="0.94">
  <title>Wren / Soren</title>
  <date>2026-01-27</date>
  <snippet>Soren received 'needs improvement'...</snippet>
</result>
```

**Effort:** Small. New formatter in `kb/output.py`.

---

## Architecture Differences Summary

| Aspect | kb (ours) | qmd | Notes |
|---|---|---|---|
| **Entity graph** | First-class (people, projects, companies) | None (flat docs) | Our moat |
| **Source adapters** | Pluggable (`sources/meetings.py`, `sources/memory.py`) | Generic glob | Ours is better for structured data |
| **Chunking** | Header-aware + speaker-turn-aware | Fixed 800-token windows + 15% overlap | Ours produces more coherent chunks |
| **Vector store** | LanceDB (file-based, filtered search) | sqlite-vec (in SQLite) | LanceDB is stronger for filtered queries |
| **Keyword index** | SQLite FTS5 | SQLite FTS5 | Same |
| **Fusion** | RRF | RRF with weights + top-rank bonus | Adopt weights + bonus |
| **Query expansion** | None (planned: entity-aware) | Fine-tuned 1.7B model, typed variants | Their biggest investment |
| **Reranking** | None (planned) | Qwen3 cross-encoder on chunks | Adopt chunk-level approach |
| **Caching** | None (planned) | SQLite LLM cache | Adopt |
| **MCP** | Not planned (Phase 8?) | Built-in (stdio + HTTP) | Add later |
| **Language** | Python | TypeScript/Bun | Both fine for purpose |
| **Embedding model** | nomic-embed-text-v1.5 (768d) → Qwen3-Embedding-0.6B | embedding-gemma-300M (GGUF) | Qwen3 beats both; swap after Phase 6 |

---

## Implementation Order

### Quick wins (during or right after Phases 4-6)

These are small changes that improve the core experience:

1. **WAL mode pragma** — Trivial. Add `PRAGMA journal_mode = WAL` to `kb/db.py`. Enables concurrent reads during indexing.
2. **Strong signal fast path** — Phase 4. Skip vector search when FTS already has a high-confidence match.
3. **Weighted RRF + top-rank bonus** — Phase 4. Original lists get 2x weight; top-rank gets +0.05.
4. **Snippet extraction with line positions** — Phase 4. Return `line` and `@@ -start,count @@` headers in search results.
5. **Three-tier search** — Phase 4/5. Add `--fast` (FTS only) and `--deep` (with reranking) flags.
6. **File path line references** — Phase 5. `kb view "path.md:42" --limit 20`.
7. **Content-hash docids** — Phase 5. `kb view #abc123` shorthand.
8. **"Did you mean?" fuzzy matching** — Phase 5. Levenshtein on `kb view` not-found errors.
9. **Index health warnings** — Phase 5. Stale index, missing embeddings warnings.
10. **Progress bar during indexing** — Phase 5. `tqdm` or stderr progress for bulk index.

### Model upgrade (after Phase 6)

11. **Embedding model upgrade** — Swap to Qwen3-Embedding-0.6B + chunking adjustments + full re-embed (do all at once):
    - Change model to `Qwen/Qwen3-Embedding-0.6B` in `embeddings.py`
    - Add document-level summary embeddings in `indexer.py`
    - Increase transcript window to 1000 tokens in `chunker.py`
    - Add instruction-aware embedding in `embeddings.py`
    - Raise `MAX_SECTION_TOKENS` to 2000 in `chunker.py`
    - Run `kb index run --full` to re-embed everything

### Search quality improvements (after model upgrade)

12. **LLM result caching** — Cache reranking/expansion results in SQLite.
13. **Cross-encoder reranking** — New module `kb/reranker.py`, `--deep` flag uses it.
14. **Entity-aware query expansion** — Leverage entity graph for alias expansion.

### Operational improvements

15. **Database cleanup commands** — `kb index cleanup` (orphaned chunks/vectors), `kb index vacuum`.
16. **Multi-get with globs** — `kb view "meetings/organised/2026/01/*" --format jsonl`.
17. **XML output format** — `--format xml` for Claude-friendly structured output.
18. **Similar document discovery** — `kb similar <path>` via LanceDB nearest vectors.

### Platform expansion

19. **MCP server mode** — `kb mcp` with dynamic instructions (entity counts, date ranges, source types).
20. **Claude Code skill** — Skill definition teaching Claude when/how to use kb commands (see 6.1 below).
21. **Custom update hooks per source** — Pre-index commands (git pull, Slack sync, calendar fetch).

---

## Priority 6: Additional Ideas from qmd (SKILL.md, CLI, Config)

Mined from `skills/qmd/SKILL.md`, `example-index.yml`, `migrate-schema.ts`, and remaining `qmd.ts` CLI handlers.

### 6.1 Claude Code Skill with Score Interpretation Table

**From qmd's SKILL.md:** qmd ships a Claude Code skill that teaches the AI how to interpret search results and choose the right tool. The key innovation is a **score interpretation table** that prevents the AI from over-fetching or under-fetching:

| Score Range | Meaning | Action |
|---|---|---|
| 0.8-1.0 | Highly relevant | Use directly |
| 0.5-0.8 | Moderate relevance | Worth reading, may need context |
| 0.3-0.5 | Tangential | Only if nothing better |
| < 0.3 | Noise | Ignore |

And a **recommended workflow escalation**:
1. `kb index status` — understand what's available
2. `kb search "query" --fast` — try keyword first (~30ms)
3. `kb search "query"` — fall back to hybrid (~2s)
4. `kb search "query" --deep` — full pipeline with reranking (~10s)
5. `kb view <path>` — retrieve full document for deep analysis

**Why this matters:** Without this, Claude tends to always use the most expensive search and over-retrieve. The skill teaches it to escalate gradually.

**For kb skill (`skills/kb/SKILL.md`):**
```markdown
## Search Workflow
1. Check status: `kb index status --json`
2. Keyword search (fast): `kb search "query" --fast --json --limit 5`
3. Hybrid search (default): `kb search "query" --json --limit 10`
4. Entity lookup: `kb entity find "name" --json`
5. Full document: `kb view <path> --offset N --limit 20`

## Score Interpretation
- 0.8+: Highly relevant, use directly
- 0.5-0.8: Worth reading, may need surrounding context
- 0.3-0.5: Tangential, only if nothing better found
- <0.3: Noise, ignore

## Entity-Aware Patterns
- When asked about a person: `kb entity find "name"` → `kb entity timeline "name"`
- When asked about a project: `kb search "project name"` → filter by entity
- When asked "what happened last week": `kb search --from 2026-02-03 --to 2026-02-10`
```

**Effort:** Small. Write the SKILL.md file + register as Claude Code skill.

### 6.2 Global Context Injection

**From qmd's `example-index.yml`:** Collections can define a `global_context` string that gets injected into every search result:

```yaml
global_context: "If you see a relevant [[WikiWord]], search for that WikiWord"
```

This is a system-level hint that shapes how the AI uses search results.

**For kb:** Inject entity-aware context into search results:

```
Hint: Names in results may match known entities. Use `kb entity find "name"` for full context.
Known entities in these results: Soren Chen (person), Helix Refactor (project), Lattice Co (company).
```

This leverages kb's entity graph — after running `find_entity_mentions()` on results, surface the matched entities as navigation hints. The AI can then choose to dig deeper on specific entities.

**Implementation:** Add to the search result formatter, not the index.

**Effort:** Small. Post-search entity extraction + result header.

### 6.3 Schema Migration Framework

**From qmd's `migrate-schema.ts`:** A transactional schema migration pattern with rollback:

```python
def migrate_v1_to_v2(db):
    """Add docid column to documents table."""
    try:
        db.conn.execute("BEGIN")
        db.conn.execute("ALTER TABLE documents ADD COLUMN docid TEXT")
        db.conn.execute("UPDATE documents SET docid = substr(content_hash, 1, 6)")
        db.conn.execute("CREATE UNIQUE INDEX idx_docid ON documents(docid)")
        db.conn.execute("COMMIT")
        print("Migration complete: added docid column")
    except Exception as e:
        db.conn.execute("ROLLBACK")
        print(f"Migration failed: {e}")
        raise
```

**Why this matters:** kb will need schema migrations when adding features (docids, cache table, content-addressable storage). Having a migration framework from the start prevents "just delete and re-index" as the only upgrade path.

**Implementation:** Add `kb/migrations.py` with a `migrations` table tracking which migrations have run:

```sql
CREATE TABLE IF NOT EXISTS migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now')),
    description TEXT
);
```

Each migration is a function registered in order. `kb index run` checks and applies pending migrations before indexing.

**Effort:** Small-Medium. Worth doing before adding docids or cache table.

### 6.4 Context Check / Entity Audit Command

**From qmd's `context check`:** A command that proactively identifies gaps in the knowledge base:

```
$ qmd context check

Collections missing context:
  - "journals" (142 docs) — no description set
    Fix: qmd context add journals "Daily journal entries"

Paths with many documents but no context:
  - meetings/organised/2025/ (892 docs) — no path context
```

**For kb:** An entity audit command that identifies:
- People mentioned in meetings who don't have a `memory/people/*.md` profile
- Projects referenced in notes without a `memory/projects/*.md` file
- Entities with no meeting mentions (stale entities)
- Recent meetings with unrecognised names (potential new entities)

```
$ kb entity audit

Missing profiles (mentioned in 5+ meetings):
  - "Uma Garcia" — 12 mentions, no memory/people/uma.md
  - "Yuki Tanaka" — 8 mentions, no memory/people/yuki.md

Stale entities (no mentions in last 90 days):
  - "Lattice Co" (company) — last mentioned 2025-09-14

Unrecognised names in recent meetings (last 30 days):
  - "Pat" — mentioned in 3 meetings, not in entity list
```

**Effort:** Medium. Requires cross-referencing entity_mentions with memory files. High value for maintaining the knowledge base.

### 6.5 Fuzzy Document Path Resolution

**From qmd's `get` command:** When a path doesn't match exactly, qmd tries three resolution strategies in order:

1. **Exact match** — `kb view meetings/organised/2026/01/27/file.md`
2. **Suffix match** — `kb view file.md` matches any document ending in `file.md`
3. **Fuzzy Levenshtein** — `kb view flie.md` finds `file.md` (distance 1)

Plus the `#docid` shorthand for content-hash IDs.

**For kb:** Implement in `kb view`:

```python
def resolve_path(query, all_paths):
    # 1. Exact match
    if query in all_paths:
        return query

    # 2. Suffix match
    suffix_matches = [p for p in all_paths if p.endswith(query)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 3. Fuzzy match (Levenshtein distance ≤ 3)
    fuzzy = [(p, levenshtein(query, p.split("/")[-1])) for p in all_paths]
    close = sorted([(p, d) for p, d in fuzzy if d <= 3], key=lambda x: x[1])

    if close:
        return close[0][0]  # or prompt user to choose

    return None  # not found
```

**Effort:** Small. Combine with 3.2 (docids) and 5.2 (fuzzy matching) into a single path resolution module.

### 6.6 Update Flow with Source Freshness

**From qmd's `update` command:** The update flow checks source freshness before re-indexing:

1. Run per-source `update` commands (git pull, API fetch, etc.)
2. Scan for new/modified files (compare mtime or content hash)
3. Index only changed files
4. Report what changed: "Added 3 new documents, updated 1, removed 0"

**For kb:** The `kb ingest` command should follow this pattern:

```
$ kb ingest

[1/3] Checking for new Granola exports...
  Found: granola_transcripts_2026-02-10 (12 new files)
  Running: organise_granola.py

[2/3] Scanning for changes...
  New: 12 meeting files
  Modified: 0
  Unchanged: 3,084 (skipped)

[3/3] Indexing new documents...
  Embedded: 12 documents, 47 chunks
  Entity links: 23 new mentions

Done. Index: 3,096 documents, ~12,400 chunks, 67 entities.
```

**Effort:** Small. Already partially implemented — just needs the reporting and freshness check.

---

## Priority 7: Final Ideas from Full Source + Test Read

Mined from remaining `store.ts` (lines 2000-2913), `qmd.ts` (lines 800-2510), `llm.ts` (full), and all 6 test files.

### 7.1 FTS5 Phrase + Proximity Search (NEAR Queries)

**From qmd's `buildFTS5Query()`:** Instead of just splitting query terms and OR-ing them, build a multi-strategy FTS5 query:

1. **Exact phrase** — `"helix refactor"` (highest weight)
2. **Proximity (NEAR)** — `NEAR(helix refactor, 10)` — terms within 10 words of each other
3. **Individual terms** — `rust OR migration` (fallback)

The strategies rank phrase > proximity > individual, so exact matches always win.

**Why it matters:** Our current FTS5 usage is basic term matching. Phrase and proximity search dramatically improve precision for multi-word queries like "performance review" (should find the phrase, not docs about "performance" and "review" separately).

**Implementation:**
```python
def build_fts5_query(query: str) -> str:
    terms = query.split()
    if len(terms) >= 2:
        # Phrase match (highest priority via RANK)
        phrase = f'"{query}"'
        # Proximity match
        near = f'NEAR({" ".join(terms)}, 10)'
        # Individual terms
        individual = " OR ".join(terms)
        return f'{phrase} OR {near} OR {individual}'
    return query
```

**Effort:** Small. Change the FTS5 query builder in search.py. High impact on precision.

### 7.2 BM25 Score Normalization (Sigmoid Mapping)

**From qmd's `normalizeBM25()`:** SQLite's BM25 returns negative scores (typically -15 to -2, where more negative = more relevant). qmd normalizes these to [0, 1] using a sigmoid-like mapping so they're comparable with vector cosine similarity scores in RRF.

**Why it matters:** Without normalization, RRF fusion can't compare FTS and vector scores meaningfully. The strong signal fast path (1.1) needs a 0-1 score to apply the 0.85 threshold.

**Implementation:**
```python
def normalize_bm25(raw_score: float) -> float:
    """Map SQLite BM25 negative scores to [0, 1]."""
    # BM25 scores: -15 (very relevant) to 0 (irrelevant)
    # Sigmoid mapping: -15 → ~1.0, -2 → ~0.3, 0 → ~0.0
    return 1.0 / (1.0 + math.exp(0.5 * raw_score + 2))
```

**Effort:** Trivial. One function. Required for items 1.1 and 1.2 to work.

### 7.3 Search Progress Hooks (Pipeline Observability)

**From qmd's `SearchHooks` interface:** The search pipeline accepts optional callback hooks that the CLI wires to stderr output:

```python
@dataclass
class SearchHooks:
    on_strong_signal: Callable[[float], None] | None = None  # BM25 won
    on_expand: Callable[[str, list[str]], None] | None = None  # query expanded
    on_rerank_start: Callable[[], None] | None = None
    on_rerank_done: Callable[[], None] | None = None
```

**Why it matters:** Decouples search orchestration from UI. The search pipeline doesn't know about terminals, spinners, or progress bars — it just fires hooks. CLI, MCP, and tests can all wire different behaviors. Also useful for debugging: "why did this query return that?" becomes observable.

**Implementation:** Add optional `hooks` parameter to `search()` in `kb/search.py`. CLI wires them to stderr. MCP ignores them.

**Effort:** Small. Clean separation of concerns.

### 7.4 Soft-Delete / Deferred Document Deactivation

**From qmd's indexing pipeline:** During incremental indexing, qmd doesn't immediately delete documents that are missing from the filesystem. Instead:

1. Track all paths seen during the current index run
2. After indexing, mark unseen documents as `active = 0` (soft delete)
3. Orphaned content/vectors cleaned up later by `kb index cleanup`

**Why it matters:** If a meeting file is temporarily moved or renamed, the document isn't lost. Soft delete also enables undo ("I accidentally deleted a file, but the index still has it").

**Implementation:** Add `active` boolean column to documents table. `index_all()` tracks seen paths, deactivates unseen ones. Search filters on `active = 1`. Cleanup purges `active = 0` older than N days.

**Effort:** Small. Schema addition + indexer logic.

### 7.5 Search Quality Evaluation Framework

**From qmd's `eval.test.ts` + `eval-harness.ts`:** A structured evaluation framework that measures search quality across difficulty levels:

| Difficulty | Example query | BM25 target | Hybrid target |
|---|---|---|---|
| Easy | "MFA implementation" | Hit@3 ≥ 80% | Hit@3 ≥ 80% |
| Medium | "how to improve deployment" | Hit@3 ≥ 15% | Hit@3 ≥ 50% |
| Hard | "that discussion about team feedback" | Hit@3 ≥ 15% | Hit@3 ≥ 35% |
| Fusion | queries needing both signals | — | Should beat both individual methods |

**Why it matters:** Without an eval framework, search quality changes are "vibes-based". This gives quantitative before/after comparison when changing models, chunking, or fusion weights. Especially important when upgrading the embedding model.

**Implementation:**
```python
# kb/eval.py
@dataclass
class EvalQuery:
    query: str
    expected_path: str  # document that should appear
    difficulty: Literal["easy", "medium", "hard", "fusion"]

def evaluate(queries: list[EvalQuery], search_fn) -> EvalReport:
    """Run queries, compute Hit@1, Hit@3, Hit@5 by difficulty."""
```

Create `kb/tests/eval_queries.py` with ~20 curated queries across difficulty levels using real meeting data.

**Effort:** Medium. Requires curating test queries. Very high value for confidence in changes.

### 7.6 Graceful FTS-Only Fallback

**From qmd tests:** The system works with just BM25 when the vector table doesn't exist yet (e.g., first run before embeddings complete, or if embedding model isn't downloaded).

**Why it matters:** Bulk indexing takes minutes. Users should be able to search immediately after `kb index run` starts — FTS is populated instantly, vectors come later. Also useful for CI/testing where you don't want to download a 600MB model.

**Implementation:** In `search()`, catch LanceDB table-not-found errors gracefully and fall back to FTS-only results. Show a warning: "Vector search unavailable — using keyword search only. Run `kb index run` to enable hybrid search."

**Effort:** Trivial. Try/except around vector search call.

### 7.7 MCP Daemon Mode with PID Lifecycle

**From qmd's `mcp --daemon`:** The MCP server can run as a background daemon:

```
kb mcp --daemon --port 8181    # Start background server, write PID file
kb mcp stop                     # Kill daemon, clean PID file
kb mcp status                   # Check if daemon is running
```

Features:
- PID file at `~/.cache/kb/mcp.pid`
- Stale PID detection (process died but PID file remains)
- Prevents duplicate daemons
- HTTP transport (not just stdio) for multi-client access
- Health endpoint at `/health` returning `{status, uptime}`

**Why it matters:** stdio MCP requires a fresh process per client. A daemon with HTTP transport can serve Claude Desktop, Cursor, and custom scripts simultaneously.

**Effort:** Medium. Part of MCP implementation (item 21).

### 7.8 Line Range Syntax in View

**From qmd tests:** Beyond `path:42` (start at line 42), qmd supports full range syntax:

```
kb view "path.md:10-20"     # Lines 10-20
kb view "path.md:42"         # From line 42 (with default limit)
kb view "path.md" --lines 10-20  # Flag-based alternative
```

**Effort:** Trivial. Extend the line reference parser.

---

## Summary: Prioritised Backlog (30 items)

| # | Item | Phase | Effort | Impact |
|---|---|---|---|---|
| 1 | WAL mode pragma | Now | Trivial | Safety |
| 2 | BM25 score normalization (sigmoid) | 4 | Trivial | Required |
| 3 | Strong signal fast path | 4 | Small | High |
| 4 | FTS5 phrase + proximity search | 4 | Small | High |
| 5 | Weighted RRF + top-rank bonus | 4 | Small | High |
| 6 | Snippet extraction with line positions | 4 | Small | High |
| 7 | Three-tier search (`--fast`/`--deep`) | 4-5 | Small | Medium |
| 8 | Search progress hooks | 4 | Small | Architecture |
| 9 | Graceful FTS-only fallback | 4 | Trivial | UX |
| 10 | File path line references + ranges (`path:10-20`) | 5 | Small | Medium |
| 11 | Content-hash docids (`#abc123`) | 5 | Small | Medium |
| 12 | Fuzzy document path resolution | 5 | Small | Medium |
| 13 | Index health warnings | 5 | Small | Low |
| 14 | Progress bar during indexing | 5 | Small | UX |
| 15 | Soft-delete / deferred deactivation | 5 | Small | Safety |
| 16 | **Embedding model upgrade** (Qwen3-0.6B + chunking) | Post-6 | Medium | High |
| 17 | LLM result caching | Post-6 | Small | Medium |
| 18 | Cross-encoder reranking | Post-6 | Medium | High |
| 19 | Entity-aware query expansion | Post-6 | Medium | High |
| 20 | Search quality evaluation framework | Post-6 | Medium | Confidence |
| 21 | Schema migration framework | Post-6 | Small | Foundation |
| 22 | Database cleanup commands | Post-6 | Small | Ops |
| 23 | Multi-get with globs | Post-6 | Small | Convenience |
| 24 | XML output format | Post-6 | Small | AI UX |
| 25 | Global context injection in results | Post-6 | Small | AI UX |
| 26 | Entity audit command | Post-6 | Medium | Maintenance |
| 27 | MCP server mode (stdio + HTTP daemon) | 8 | Medium | Platform |
| 28 | Claude Code skill with score table | 8 | Small | AI UX |
| 29 | Custom update hooks per source | 8 | Small | Extensibility |
| 30 | Similar document discovery | 8 | Small | Discovery |

---

## Phase 4 Critical Path

Three items from this backlog are **required foundations** — don't skip them when building Phase 4 (search):

1. **BM25 score normalization (#2)** — Without this, RRF fusion can't compare FTS and vector scores. SQLite BM25 returns negative values (-15 to 0); vector cosine returns 0-1. The sigmoid mapping is trivial but everything downstream depends on it.

2. **FTS5 phrase + proximity queries (#4)** — Big precision win for small effort. Multi-word queries like "performance review" or "Helix refactor" should match the phrase, not just docs containing both words independently. `NEAR(term1 term2, 10)` handles this.

3. **Graceful FTS-only fallback (#9)** — Bulk indexing takes minutes. Users should be able to search immediately with FTS while embeddings are still being computed. Also essential for testing without downloading a 600MB model.

---

## Beyond qmd: Ideas from the Wider Landscape

Research across similar projects ([Semantra](https://github.com/freedmand/semantra), [Revery](https://github.com/thesephist/revery), [Khoj](https://github.com/khoj-ai/khoj), [Smart Connections](https://github.com/brianpetro/obsidian-smart-connections), [LightRAG](https://github.com/HKUDS/LightRAG)), academic techniques ([Late Chunking](https://arxiv.org/abs/2409.04701), [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), [HyDE](https://arxiv.org/abs/2212.10496), [RAPTOR](https://arxiv.org/abs/2401.18059)), and the [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) collection (34 techniques with implementations).

### 8.1 Late Chunking (Jina AI) — High Value

**Source:** [Jina AI research](https://jina.ai/news/late-chunking-in-long-context-embedding-models/), [paper](https://arxiv.org/abs/2409.04701)

**What it is:** Instead of "chunk then embed" (traditional), do "embed then chunk":
1. Pass the **entire document** through the embedding model to get token-level representations
2. **Then** split the token embeddings at chunk boundaries
3. Mean-pool each chunk's token embeddings

Because the transformer has seen the full document context before chunking, each chunk embedding captures cross-references, pronouns, and document-level context that traditional chunking loses.

**Why it matters for kb:** Meeting notes are full of anaphoric references ("He mentioned that the timeline was slipping" — who is "he"?). Late chunking resolves these because the full document context is available when embedding. Jina's paper shows **10-12% retrieval improvement** on documents with pronoun references.

**Compatibility:** Requires a long-context embedding model (Qwen3-Embedding-0.6B has 32K context — perfect). Can be implemented in ~30 lines of code by modifying the pooling step. Works with sentence-transformers.

**Implementation sketch:**
```python
# In kb/embeddings.py
def embed_late_chunked(model, full_text: str, chunk_boundaries: list[tuple[int, int]]) -> list[list[float]]:
    """Embed full document, then extract chunk embeddings from token representations."""
    # 1. Tokenize full document
    tokens = model.tokenize(full_text)
    # 2. Get token-level embeddings (before pooling)
    token_embeddings = model.encode(full_text, output_value="token_embeddings")
    # 3. For each chunk boundary, mean-pool the corresponding token embeddings
    chunk_embeddings = []
    for start_char, end_char in chunk_boundaries:
        start_tok, end_tok = char_to_token_range(tokens, start_char, end_char)
        chunk_emb = token_embeddings[start_tok:end_tok].mean(dim=0)
        chunk_embeddings.append(chunk_emb.tolist())
    return chunk_embeddings
```

**When:** Bundle with embedding model upgrade (#16). Requires Qwen3's 32K context.

**Effort:** Small-Medium. Modify `embed()` in embeddings.py. Need to pass chunk boundaries alongside text.

**Impact:** High. Direct improvement to chunk embedding quality with no model change.

### 8.2 Contextual Retrieval Lite: Entity-Enriched Chunk Prefixes

**Source:** [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (adapted for kb)

**What it is:** Anthropic's full contextual retrieval requires an LLM call per chunk (~12,000 calls for our corpus). Instead, use kb's **entity graph** to enrich the metadata prefix — achieving similar context enrichment without the cost:

```
[Meeting: Wren / Soren | Date: 2026-01-27 | Section: Performance Review
 Entities: Soren Chen (person/DevEfficiency), Pipeline Improvements (project)
 Related: 3 prior meetings with Soren in Jan 2026]
```

Compared to our current prefix (`[Meeting: title | Date: date | Section: heading]`), this adds entity context and meeting frequency — information that helps the embedding model understand *what* is being discussed, not just *where*.

**When:** Bundle with embedding model upgrade (#17) since a full re-embed is needed anyway.

**Effort:** Small. Modify chunk prefix generation in `chunker.py` + `indexer.py` to include entity data.

**Impact:** Medium-High. Leverages kb's entity graph (our differentiator) for embedding quality.

### 8.3 Recency-Weighted Scoring — High Value

**Source:** [Ragie](https://docs.ragie.ai/docs/retrievals-recency-bias), [research](https://arxiv.org/abs/2509.19376)

**What it is:** Blend similarity scores with a time decay function so recent documents rank higher by default:

```python
def recency_score(doc_date: date, half_life_days: int = 90) -> float:
    """Exponential decay: score halves every `half_life_days` days."""
    age_days = (date.today() - doc_date).days
    return 0.5 ** (age_days / half_life_days)

def blended_score(similarity: float, doc_date: date, recency_weight: float = 0.15) -> float:
    """Blend similarity with recency. Default: 85% similarity, 15% recency."""
    recency = recency_score(doc_date)
    return (1 - recency_weight) * similarity + recency_weight * recency
```

**Why it matters for kb:** "What's happening with the Helix refactor?" should surface the January 2026 meeting over the September 2025 one, even if the older meeting has slightly higher semantic similarity. Research shows a simple recency prior achieves **perfect accuracy on freshness queries**.

**Configuration:** The recency weight and half-life should be configurable:
```
kb search "Helix refactor"                    # Default: 15% recency weight
kb search "Helix refactor" --recency 0.3      # Heavier recency bias
kb search "Helix refactor" --recency 0        # Pure similarity (no recency)
```

**Effort:** Small. Add to the score blending step after RRF fusion.

**Impact:** High. Natural for a meeting-notes corpus where recency matters.

### 8.4 Entity-Enriched Graph Traversal (LightRAG-inspired) — Medium Value

**Source:** [LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025)

**What it is:** LightRAG builds a lightweight entity-relationship graph during indexing and uses it for dual-level retrieval:
- **Low-level:** Direct entity lookup ("find meetings with Soren")
- **High-level:** 1-hop graph traversal ("find meetings where topics related to Soren's projects are discussed")

**Why it matters for kb:** We already have the entity graph. Currently it's used for filtering (find docs mentioning entity X). Graph traversal would enable relational queries:

```
kb search "Soren" --related            # Meetings with Soren + meetings about his projects
kb entity connections "Soren"          # People, projects, companies connected to Soren
```

**Implementation:** Given entity "Soren Chen":
1. Find Soren's entity_mentions → meeting IDs
2. Find other entities co-occurring in those meetings → "Pipeline Improvements", "Linnea", "Kit K."
3. Use those related entities as additional search terms or filters

This is a 1-hop traversal on the entity co-occurrence graph. No external graph database needed — SQL joins on `entity_mentions` table.

**Effort:** Medium. SQL queries + search integration. High value for "who works with whom?" queries.

### 8.5 Connected Notes Discovery (Smart Connections-inspired)

**Source:** [Smart Connections](https://github.com/brianpetro/obsidian-smart-connections)

**What it is:** For the currently open document, automatically surface related documents:

```
kb related "meetings/organised/2026/01/27/wren-soren_f744ed8c.notes.md"

Related meetings:
  0.89  2026-01-13 Wren / Soren — Previous 1:1, same topics
  0.82  2026-01-27 Wren / Kit G — Same day, overlapping discussion
  0.78  2025-12-15 Soren / Platform Review — Related project

Related entities:
  Soren Chen (12 meetings), Pipeline Improvements (8 meetings)
```

This combines vector similarity with entity graph connections for richer "related" results.

**Effort:** Small. LanceDB nearest vectors + entity co-occurrence join.

---

## Triaged Out

Ideas considered and intentionally dropped:

| Idea | Source | Why dropped |
|---|---|---|
| **Query Arithmetic** (`--boost`/`--not`) | Semantra | Entity filtering + entity-aware expansion cover the same disambiguation use cases more naturally. Niche power-user feature with limited ROI. |
| **Proposition Chunking** | NirDiamant/RAG_Techniques | Requires LLM call per chunk during indexing (~12,000 calls). Our header-based chunking already produces semantically coherent sections. Solving a problem we don't have — meeting sections are naturally topical. |
| **Hierarchical Document Summaries (RAPTOR)** | RAPTOR paper | Over-engineered for a 3,000-doc personal corpus. Document-level embeddings (bundled with model upgrade) capture 80% of the benefit with 10% of the effort. Clustering + LLM summary generation is heavy machinery. |
| **Relevance Feedback / Learning from Search** | NirDiamant + Semantra | High infrastructure cost (feedback table, score adjustment, feedback UI) for marginal gain on a single-user knowledge base. Manual tuning is sufficient. |
| **Full LLM Contextual Retrieval** | Anthropic | ~12,000 LLM calls during indexing. Entity-enriched prefixes (8.2) achieve similar benefits using kb's entity graph — no LLM needed. |

---

## Triage Notes: Interactions & Dependencies

**Items that compete (do one, skip the other):**
- **Late Chunking (in #15)** wins over full LLM Contextual Retrieval — same problem (lost context when chunking), late chunking is ~30 lines of code with no LLM calls, contextual retrieval is ~12,000 LLM calls.
- **Three-tier search (#7, explicit `--fast`/`--deep`)** is the simple version of **Adaptive Query Classification**. Build three-tier first, then optionally add `--auto` mode with heuristic routing later as a refinement (not a separate backlog item).

**Dependency chain (order matters):**
1. **BM25 normalization (#2)** → required before Strong Signal (#3) and Weighted RRF (#4) — they need 0-1 scores
2. **Schema migration framework (#19)** → required before any schema changes: docids (#10), soft-delete (#14), cache table (#17)
3. **Search quality eval (#18)** → should come before model upgrade (#15) — need before/after measurements
4. **Model upgrade (#15)** bundles: Qwen3-0.6B + late chunking + entity-enriched prefixes + larger transcript windows + document-level embeddings + instruction-aware embedding — do it all at once since full re-embed is needed anyway

---

## CLI & MCP Design Patterns (Research)

Research into best-in-class CLI patterns for LLM-consumable tools and MCP server design for knowledge bases.

### Sources

- [Anthropic: Writing Tools for Agents](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/best-practices-and-examples)
- [Linearis](https://linearis.dev/) — Linear CLI designed for humans + LLMs (Thorsten Ball)
- [Heroku CLI Style Guide](https://devcenter.heroku.com/articles/cli-style-guide) — mature CLI conventions
- [OpenTofu `-json-into`](https://opentofu.org/docs/internals/machine-readable-ui/) — dual output streams
- [FastMCP](https://gofastmcp.com/) — Python MCP framework with decorator patterns
- [MCP specification](https://modelcontextprotocol.io/docs/concepts/architecture) — Tools, Resources, Prompts, Sampling
- ["Less is More" MCP design](https://blog.scottlogic.com/2025/04/05/less-is-more-designing-mcp-servers-that-actually-work.html) — workflow-based tool consolidation
- [Wren Willison's LLM CLI](https://llm.datasette.io/) — reference CLI for LLM tools

### CLI Patterns to Adopt

#### 1. Human Default, Machine Optional

Every command should produce readable human output by default, with `--json` for structured machine output. This is the universal pattern across Heroku, Linearis, and Willison's LLM tool.

```
kb search "Helix refactor"              # Human: formatted table
kb search "Helix refactor" --json       # Machine: JSON array, one object per result
kb search "Helix refactor" --jsonl      # Machine: newline-delimited JSON (for streaming)
```

**Already planned** in kb's Phase 5. The key insight from Heroku: `--json` should output to **stdout** while all human messaging (progress, warnings, hints) goes to **stderr**. This means `kb search "query" --json 2>/dev/null` always gives clean JSON, and `kb search "query" 2>&1` gives everything.

#### 2. `usage` Command for Agent Self-Documentation (from Linearis)

Linearis's most innovative pattern: a `usage` command that outputs a complete guide for AI agents:

```
$ kb usage

# kb — Personal Knowledge Base CLI
## Commands
- kb search <query> [--json] [--fast|--deep] [--limit N] [--recency WEIGHT]
  Search meeting notes, transcripts, and memory files.
  Returns: title, date, score, snippet, path

- kb entity find <name> [--json]
  Look up a person, project, or company.
  Returns: name, type, aliases, mention_count, related_entities

- kb view <path> [--offset N] [--limit N]
  Read a specific document. Supports path:LINE syntax.

## Score Interpretation
- 0.8+: Highly relevant, use directly
- 0.5-0.8: Worth reading, may need context
- <0.5: Tangential or noise

## Recommended Workflow
1. kb search "query" --fast --json --limit 5   (keyword, ~30ms)
2. kb search "query" --json --limit 10          (hybrid, ~2s)
3. kb view <path from results>                  (full document)
```

This is cheaper than MCP tool discovery (no protocol overhead) and works with any agent that has Bash access. Claude Code would call `kb usage` once at the start of a session to learn the tool surface.

**Impact:** Replaces the need for detailed CLAUDE.md instructions — the tool documents itself.

#### 3. Dual Output Streams (from OpenTofu)

OpenTofu's `-json-into <file>` writes machine-readable JSON to a file while still showing human-friendly output on the terminal. Useful for CI:

```
kb index run --json-into results.json    # Terminal: progress bar. File: structured log.
```

**Lower priority** for kb — the `--json` + stderr separation covers most use cases. But worth noting for future CI integration.

#### 4. Semantic Parameter Names (from Anthropic)

Anthropic's tool design guide emphasises naming parameters for clarity to LLMs:

- `query` not `q` — LLMs understand full words better
- `limit` not `n` or `count` — unambiguous
- `format` with enum `["json", "jsonl", "table", "xml"]` — bounded choices
- `recency_weight` not `rw` — self-documenting

Also: **always provide defaults**. Every optional parameter should have a documented default so the LLM never has to guess.

**Already good** in kb's CLI design. Just formalise this as a convention.

#### 5. Actionable Error Messages (from Anthropic + Heroku)

Errors should tell the agent **what to do**, not just what went wrong:

```
# Bad
Error: Index not found

# Good
Error: No index found. Run `kb index run` to create one.
Available commands: kb index run, kb index status
```

Anthropic's guide: include `available_actions` in error responses so the agent can self-correct without re-reading documentation.

```json
{
  "error": "No results found for 'Rust migrtion'",
  "suggestion": "Did you mean 'Helix refactor'?",
  "available_actions": ["kb search 'Helix refactor'", "kb search --fast 'Rust'"]
}
```

#### 6. Workflow Consolidation (from Anthropic + "Less is More")

Don't expose every internal function as a separate CLI command. Group related operations into workflows:

```
# Bad: 5 separate commands for one workflow
kb fts-search "query"
kb vector-search "query"
kb fuse-results
kb rerank-results
kb format-output

# Good: 1 command with mode flags
kb search "query"                # hybrid (default)
kb search "query" --fast         # FTS only
kb search "query" --deep         # hybrid + reranking
```

The "Less is More" article found that MCP servers with fewer, more powerful tools outperform granular ones — LLMs get confused choosing between 15 similar tools but handle 4-5 well-designed ones effectively.

**Target for kb:** 5-6 top-level commands max: `search`, `entity`, `view`, `index`, `mcp`, `usage`.

### MCP Design Patterns to Adopt

#### 1. Three MCP Primitives: Tools vs Resources vs Prompts

The MCP spec defines three distinct primitives. Most servers only implement Tools, but all three matter:

| Primitive | Control | Purpose | kb Example |
|---|---|---|---|
| **Tools** | Model-controlled | Actions with side effects or computation | `kb_search`, `kb_entity_find` |
| **Resources** | User/app-controlled | Read-only data, accessed by URI | `kb://status`, `kb://entity/soren` |
| **Prompts** | User-controlled | Reusable prompt templates | "Summarise recent meetings about {topic}" |

**Tools** are what the LLM calls autonomously. Keep these to 4-6 well-designed tools.

**Resources** are data the host application can expose to the LLM's context. They're read-only and identified by URIs:

```
kb://status                           → Index stats, entity counts, date ranges
kb://entity/soren-vance        → Full entity profile
kb://document/{docid}                 → Document content by docid
kb://recent?days=7                    → Documents from last 7 days
```

Resources are cheaper than tool calls — the host can pre-load them into context without the LLM making a decision.

**Prompts** are templates that users or host apps invoke explicitly:

```
Prompt: "weekly-summary"
Arguments: { week_start: "2026-02-03" }
Template: "Using kb, find all meetings from {week_start} to {week_end} and create a TGIF-style summary."
```

#### 2. Dynamic Instructions (from qmd + best practice)

Generate the MCP server's `instructions` field at startup from actual index state:

```
kb is a personal knowledge base over 3,096 documents (1,548 meeting notes, 1,548 transcripts, 39 memory files).
67 known entities: 45 people, 12 projects, 10 companies.
Date range: 2024-06 to 2026-02.

Tools (use in escalation order):
- kb_search (query, limit?, mode?) — keyword + vector hybrid search (~2s).
  Use mode="fast" for keyword-only (~30ms), mode="deep" for reranking (~10s).
- kb_entity_find (name) — look up person, project, or company.
- kb_entity_timeline (name, from?, to?) — chronological mentions of an entity.
- kb_view (path, offset?, limit?) — read a specific document.

Score interpretation: 0.8+ highly relevant, 0.5-0.8 worth reading, <0.5 noise.
```

This means the LLM knows what's available **before making any tool calls**. It won't waste a call on `kb_search` for an entity lookup, and it knows the corpus size/date range.

#### 3. URI Templates for Resources (from FastMCP)

FastMCP demonstrates URI templates with parameters that make resource discovery natural:

```python
@mcp.resource("kb://entity/{name}")
async def get_entity(name: str) -> str:
    """Return full profile for a known entity."""
    return entity_profile(name)

@mcp.resource("kb://meetings/{year}/{month}")
async def get_month_meetings(year: str, month: str) -> str:
    """List all meetings for a given month."""
    return list_meetings(year, month)
```

The URI structure acts as a natural API — `kb://entity/soren` is intuitive for both humans and LLMs.

#### 4. Workflow-Based Tools (from "Less is More")

Instead of one tool per search strategy, consolidate into workflow tools:

```python
@mcp.tool()
async def kb_search(
    query: str,
    mode: Literal["fast", "hybrid", "deep"] = "hybrid",
    limit: int = 10,
    recency_weight: float = 0.15,
    format: Literal["detailed", "concise"] = "concise"
) -> list[SearchResult]:
    """Search the knowledge base. Use mode='fast' for keyword-only, 'deep' for full reranking."""
```

Anthropic's guide recommends a `response_format` parameter (DETAILED vs CONCISE) so the agent can request less data when it just needs a quick check vs a deep analysis. For kb:

- `concise`: title, date, score, one-line snippet
- `detailed`: title, date, score, full snippet with line positions, entity mentions, related docs

#### 5. Progressive Discovery (from "Less is More")

Don't expose all tools at once. Start with core tools, reveal advanced ones as needed:

- **Always available:** `kb_search`, `kb_view`, `kb_entity_find`
- **On request:** `kb_entity_timeline`, `kb_related`
- **Admin:** `kb_index_status`, `kb_index_run`

In MCP, this is implemented by listing only core tools in the initial `tools/list` response, then adding more tools dynamically when the agent requests them or when the context warrants it.

**Simpler approach for kb:** Just list all 4-5 tools but use the `instructions` field to teach escalation order. With only 5 tools, progressive discovery adds complexity without benefit.

### Implications for kb

These patterns suggest the following concrete changes to kb's design:

1. **CLI commands (5-6 max):** `search`, `entity`, `view`, `index`, `usage`, `mcp`
2. **Output convention:** Human by default → `--json` for machine → stderr for progress/warnings
3. **`kb usage` command:** Self-documenting output for AI agents (replaces need for detailed CLAUDE.md)
4. **MCP tools (4-5 max):** `kb_search`, `kb_entity_find`, `kb_entity_timeline`, `kb_view`
5. **MCP resources:** `kb://status`, `kb://entity/{name}`, `kb://document/{docid}`, `kb://recent?days=N`
6. **Dynamic instructions:** Generated from index state at MCP startup
7. **Error messages:** Include `suggestion` and `available_actions` fields in JSON errors
8. **`response_format` parameter:** `concise` (default for MCP) vs `detailed` on search tools

These integrate naturally with existing backlog items #7b (three-tier search), #29 (MCP server), #30 (Claude Code skill), and #28 (XML output).

---

## Priority 9: `kb context` — Dynamic Context Layer (Replace Static CLAUDE.md)

### The Problem

CLAUDE.md is 139 lines of manually curated static knowledge — people tables, project tables, terms, tools, customers. This data already exists in `memory/` (22 people profiles, 10 project files, glossary, company context). Both drift apart. And all 139 lines load into every Claude session whether relevant or not.

Research says this is the wrong approach:
- **[HumanLayer](https://www.humanlayer.dev/blog/writing-a-good-claude-md):** CLAUDE.md should be <300 lines (ideally <60 at root). Organise as WHAT/WHY/HOW. Use progressive disclosure — point to separate docs, don't inline content. LLMs follow ~150-200 instructions consistently; our static tables consume that budget on reference data instead of behavioral instructions.
- **[Vercel AGENTS.md](https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals):** A compressed index (pipe-delimited, 80% smaller than full content) outperforms skills 100% vs 53%. Key insight: **eliminate decision points** — the agent already knows what exists, so it doesn't have to decide whether to look something up. Full content is retrieved on-demand.

### The Design

The AGENTS.md insight is that **kb already provides the retrieval mechanism**. We don't need a static docs index — we need a **compressed entity index** that tells the agent "here's what exists" and then kb handles on-demand retrieval. This is the same pattern Vercel uses, except our index is dynamic (generated from the entity graph) instead of hand-written.

Three layers, following HumanLayer's WHAT/WHY/HOW:

#### Layer 1: CLAUDE.md — Tiny (~25 lines)

Only behavioral instructions (HOW) and a pointer to kb:

```markdown
# Context
Run `kb context` at session start for background on me, my team, projects, and terms.
Use `kb search "topic" --json` for meeting history. Use `kb entity find "name" --json` for details.
Record important facts: `kb memory add "fact" --entity "name"`.
Full command reference: `kb usage`.

## Preferences
- Automation over tribal knowledge ("bake it into the process")
- Direct communication style, pushes for real fixes
- Simple, local-first systems (TASKS.md over heavy apps)
- Hypertransparency on incidents (model: Cloudflare)
- Deployment should be a "nonevent" — automated, non-blocking

## Scripts
### meetings/organise_granola.py
Run: `./meetings/organise_granola.py` (uses `uv run` shebang)
Flags: --dry-run, or pass a path to process a specific zip/directory
```

What moved out: People (30 lines), Terms (18 lines), Projects (14 lines), Customers (7 lines), Tools (12 lines) — all now generated by `kb context`.

What stays: Preferences (behavioral — HOW to work), script docs (workflow — HOW to run things), kb pointers (WHAT exists, WHERE to find it).

#### Layer 2: `kb context` — Compressed Entity Index

Following the AGENTS.md pattern — a compressed index that eliminates decision points without consuming context budget on full profiles:

```
$ kb context

# Wren Kasper — VP Engineering, Lattice Co
|67 entities, 3096 documents, meetings 2024-06 to 2026-02

## People (27)
|Kit(Head of Product) Kit K.(Director of Design) Linnea(Platform Lead)
|Rick(EM) Sam(Product) Ted(DevOps)
|Talia(Rust Lead) Soren(DevEff/CI) Mei(Data/ML)
|Henry(ExCom) Mike(Data) Leo(Eng Lead) Anders(Scanner)
|Nick(Engineer) Oscar(Growth) Pat(Platform) Quinn(Backend)
|Idris(Exec/Product) Ivan(SRE) Jack(Tech Lead)
|Ken V.(Security/SOC2) Uma(Infra) Vic(Principal Eng)
|Wendy(Recruiting) Xander(HR Ops) Yuki(Exec)
|Zara(Finance) Wren C.(Self-Hosted)
→ Details: `kb entity find "name" --json`

## Projects (12 active)
|Helix Refactor(Talia,~45%,3x goal) Beacon CLI(Anders,CLI strategy)
|Optimisation TF(Mei/Kit K.) Public Sources(Linnea,GitLab/NPM/PyPI)
|Tooling TF(Oscar) MFA(Ted,rollout ready)
|Atlas Pipeline(Mei,top priority) Local Dev Protection(Anders)
|Org Restructuring(smaller teams,Compass) Pipeline Improvements(Soren)
|Series B(Investor) ISO 27001 2025(Ken V.)
→ Details: `kb entity find "project" --json`

## Terms
|GIM=Internal Monorepo MR=Merge Request ExCom=Executive Committee
|TF=Task Force TGIF=Weekly update NHI=Non-Human Identity
|Code Red=Critical issue Gym=Staging(removing Q1) SEM=Staff/Manager
|EBR=Exec Business Review ARR=$18.3M target Compass=Career ladder
→ Full glossary: `kb view memory/glossary.md`

## Score Interpretation
|0.8+ highly relevant, use directly
|0.5-0.8 worth reading, may need context
|<0.5 tangential or noise
```

**~30 lines** that give the agent enough index to know everyone and everything by name. No decision points — if the agent sees "Soren" in conversation, it already knows he exists and is DevEff/CI, so it can decide whether to `kb entity find "Soren"` for full context. Compare to current CLAUDE.md: same information in fewer lines, plus it's auto-generated and always current.

**Flags:**

```
kb context                              # Compressed index (default, ~30 lines)
kb context --full                       # All profiles expanded (~300 lines)
kb context --for "Soren performance"     # Filtered: only relevant entities/projects
kb context --section people             # Single section
kb context --json                       # Structured output
```

The `--for` flag runs a lightweight entity search and only includes relevant entities. A session about "Helix refactor" would include Talia, Linnea, the Helix Refactor project — but not Idris, Series B, or customer accounts.

#### Layer 3: `kb memory` — Fact Recording

A way for Claude (or the user) to record facts that persist across sessions and are reflected in `kb context` output:

```
kb memory add "Promoted to Staff Engineer" --entity "Soren" --date 2026-01-19
kb memory add "Reached 50% detector coverage" --entity "Helix Refactor"
kb memory add "Team morale high after hackathon — consider quarterly"
kb memory list --since 30d
kb memory show "Soren"
```

**Storage:** Append dated entries to existing memory files (human-readable, version-controllable) + index in SQLite facts table for fast querying:

```markdown
# In memory/people/soren-vance.md, auto-appended:

## Timeline
- 2026-01-19: Promoted to Staff Engineer (source: promotion committee)
- 2026-02-10: Delegation improvements showing progress (source: 1:1)
```

Facts are written to markdown files (source of truth) and indexed in SQLite (fast lookup). `kb context` then generates its compressed index from the current state of these files, including recent facts.

This replaces Claude Code's separate auto-memory system (`.claude/projects/.../memory/MEMORY.md`) with a unified knowledge store. Instead of two memory systems drifting apart, everything goes through kb.

### MCP Integration

**Resources (read-only, pre-loadable):**
- `kb://context` — compressed entity index (like `kb context`)
- `kb://entity/{name}` — full entity profile + facts + recent mentions

**Tools (model-invoked):**
- `kb_remember(fact, entity?, date?)` — record a fact

**Dynamic instructions** (generated at startup from index state):
```
You are helping Wren Kasper, VP Engineering at Lattice Co.
67 entities, 3096 documents, meetings 2024-06 to 2026-02.
Use kb_remember to record important facts learned during conversation.
Load kb://context for background. Use kb_search for meeting lookups.
```

### Implementation Order

1. **`kb context` (read-only)** — Phase 5-6. Generate compressed index from memory files + entity graph. Highest value, no new data model needed.
2. **`kb memory add`** — Phase 6-7. Fact recording with append-to-file + SQLite indexing.
3. **`kb context --for`** — Phase 7. Filtered context using entity search.
4. **Shrink CLAUDE.md** — Phase 7. Once `kb context` is reliable, strip the knowledge tables.
5. **MCP resources + kb_remember** — Phase 8. Expose via MCP.
6. **SessionStart hook** — Phase 8. Auto-inject `kb context` at session start.

### Impact

| Component | Effort | Impact |
|---|---|---|
| `kb context` (compressed index) | Medium | **Very High** — eliminates CLAUDE.md drift, auto-current |
| `kb context --for` (filtered) | Small | High — reduces context waste on irrelevant entities |
| `kb memory add` (fact recording) | Small | Medium — knowledge accumulates naturally |
| CLAUDE.md shrink | Trivial | High — 139 → 25 lines, frees instruction budget |
| MCP resources | Small | Medium — works with Claude Desktop/Cursor |
| SessionStart hook | Trivial | High — zero-effort context loading |

This is the feature that turns kb from "a search tool" into "the knowledge layer for all AI workflows". Every Claude session starts with the right context automatically, the entity index eliminates decision points (per Vercel's findings), and facts accumulate across sessions instead of being lost.

---

## Priority 10: Meeting Intelligence — Leader-Specific Features

kb isn't a generic RAG tool — it's a meeting intelligence system for engineering leaders. These ideas lean into that by exploiting the structure already present in the corpus: speaker turns, recurring 1:1s, project threads across time, and the entity graph.

Sources: [Zep/Graphiti temporal KG](https://arxiv.org/html/2501.13956v1), [Haystack speaker diarization + RAG](https://haystack.deepset.ai/blog/level-up-rag-with-speaker-diarization), [AssemblyAI meeting intelligence platforms](https://www.assemblyai.com/blog/meeting-intelligence-platforms), [Instructor action item extraction](https://python.useinstructor.com/examples/action_items/), [AWS meeting summarization](https://aws.amazon.com/blogs/machine-learning/meeting-summarization-and-action-item-extraction-with-amazon-nova/), [Pinecone RAG access control](https://www.pinecone.io/learn/rag-access-control/).

### 10.1 Speaker-Attributed Search — High Value, Low Effort

**The opportunity:** Granola transcripts already have speaker turns (`**Me:**`, `**System:**`). The chunker already splits on speaker turns. But we don't index **who** said what — chunks just have text, with no `speaker` metadata.

**What this enables:**
```
kb search "platform performance" --speaker "Linnea"    # What did Linnea say?
kb search "deadline" --speaker "me"                    # What did I commit to?
```

**Implementation:**

1. During indexing, resolve speakers from the transcript:
   - `**Me:**` → "Wren Kasper" (always the user)
   - `**System:**` → extract from meeting title. "Wren x Soren" → System is "Soren"
   - Multi-person meetings: Granola sometimes labels speakers by name

2. Store `speaker` as chunk metadata in SQLite + LanceDB:
   ```sql
   ALTER TABLE chunks ADD COLUMN speaker TEXT;
   ```

3. In search, add optional `--speaker` filter that narrows results before scoring.

**Why this matters:** "What did Kit say about the Series B timeline?" is a natural query for an engineering leader. Without speaker attribution, you get all chunks mentioning Kit + Series B. With it, you get specifically what Kit said.

**Effort:** Small. The data is already in the transcripts. Changes: chunker (resolve speaker names), schema (add column), search (add filter). ~50 lines.

### 10.2 Temporal Fact Tracking (Zep/Graphiti-Inspired) — Medium Value, Medium Effort

**The insight from Zep:** Facts have validity periods. "Soren is on the Platform team" was true from 2024-06 to 2025-10. "Soren leads DevEfficiency" is true from 2025-10 onwards. Most knowledge systems treat facts as timeless — Zep tracks when facts become true and when they're superseded.

**Simplified version for kb:** We don't need Zep's full bitemporal model or Neo4j graph database. SQLite with temporal columns on the existing facts/entity system is enough:

```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    fact TEXT NOT NULL,
    valid_from TEXT,          -- When this became true
    valid_until TEXT,         -- When this stopped being true (NULL = still true)
    source_path TEXT,         -- Meeting where this was learned
    recorded_at TEXT DEFAULT (datetime('now'))
);
```

**What this enables:**
```
kb entity history "Soren"                      # Full timeline of facts
kb entity state "Soren" --at 2025-09          # What was true about Soren in Sep 2025?
kb search "Helix refactor" --changes         # What changed about this project over time?
```

**The key Zep insight we adopt:** When a new fact contradicts an old one, don't delete the old fact — set `valid_until` on the old one. "Soren is on Platform team" gets `valid_until = 2025-10-01` when we record "Soren leads DevEfficiency" with `valid_from = 2025-10-01`. History is preserved.

**Why this matters:** "Has Soren's situation improved since October?" requires understanding what was true then vs now. Without temporal tracking, you'd have to search all meetings and piece it together manually.

**Interaction with `kb memory add`:** When recording facts via `kb memory add "moved to DevEfficiency" --entity "Soren" --date 2025-10`, the system checks for conflicting active facts about Soren's team/role and auto-sets `valid_until` on the old one.

**Effort:** Medium. New table, fact conflict detection logic, timeline query builder. Worth doing after the entity graph is solid (Phase 6+).

### 10.3 Pre-Meeting Briefing — Very High Value

**The concept:** Before a 1:1 or meeting, generate a contextual briefing from kb:

```
$ kb brief "Soren"

# Briefing: Soren Chen
Last met: 2026-01-27 (14 days ago)
Total meetings: 23 | This quarter: 6

## Recent Context
- DevEfficiency/CI lead. Performance improvement trajectory.
- Jan 27: Discussed delegation improvements, showing progress.
- Jan 13: Promotion committee reviewed, promoted to Staff.

## Open Threads (topics discussed but not resolved)
- CI runner stability — mentioned in 3 of last 5 meetings
- Coaching pace — raised by Kit K. in Nov, still tracking

## Entity Connections
- Works closely with: Linnea (platform), Nick (feature flags)
- Projects: Pipeline Improvements, Optimisation TF

## Suggested Topics
- Follow up on delegation progress (14 days since last check-in)
- CI runner stability — any resolution?
```

**How it works:**
1. `kb entity find "Soren"` → get entity ID, metadata
2. `kb entity timeline "Soren" --limit 5` → last 5 meetings
3. For each meeting, extract key chunks (highest-scoring chunks mentioning Soren)
4. Identify "open threads" — topics that appear repeatedly but don't have resolution language ("decided", "resolved", "done")
5. Suggest follow-ups based on recency gaps

**Variations:**
```
kb brief "Soren"                       # Person briefing
kb brief "Helix Refactor"            # Project briefing (recent updates, decisions, blockers)
kb brief --upcoming                   # Brief for next calendar event (requires calendar integration)
kb brief --week                       # Weekly summary of all meetings (TGIF draft helper)
```

**Why this matters:** An engineering leader with ~10 recurring 1:1s can spend 2 minutes on `kb brief "name"` before each one to surface context that would otherwise require scrolling through old notes. This is the killer workflow feature.

**Effort:** Medium. Composition of existing search + entity timeline. The "open threads" detection is the hardest part — start with simple recency-based suggestions, add heuristic thread tracking later.

### 10.4 Decision & Commitment Extraction — Medium Value

**The concept:** During indexing, identify chunks that contain decisions, commitments, or action items:

```
"We decided to go with Dash0 for observability."       → decision
"I'll have the proposal ready by Friday."               → commitment (speaker: Me)
"Action item: Mei to set up SSO for Datastore."        → action item (owner: Mei)
"Next steps: schedule a follow-up with the Rust team."  → action item
```

**Implementation:** Two approaches, not mutually exclusive:

**Approach A: Heuristic tagging during indexing.**
Flag chunks matching patterns: "we decided", "action item", "next steps", "I'll", "by Friday/Monday/EOD", "agreed to", "let's do". Store as `chunk_type = "decision"` or `"commitment"` metadata.

**Approach B: Claude-assisted extraction via `kb memory add`.**
After each meeting, Claude (in the session that processes the notes) extracts decisions and records them: `kb memory add "Decided: use Dash0 for observability" --entity "Optimisation TF" --type decision`. No indexing heuristics needed — the AI does the extraction.

**Approach B is better** for kb because:
- It leverages Claude's understanding of context (heuristics miss nuance)
- It integrates with the `kb memory add` pipeline already designed
- It produces higher-quality extractions than regex
- It's a workflow choice, not infrastructure

**What this enables:**
```
kb decisions --entity "Helix Refactor" --since 90d
kb commitments --speaker "me" --since 7d          # What did I promise this week?
```

**Effort:** Small if using Approach B (just metadata on facts). Medium if using Approach A (indexing heuristics).

### 10.5 Sensitivity Tagging — Important for MCP Safety

**The problem:** Some meeting content is confidential — performance reviews, salary discussions, Series B details, customer contract terms. When kb is exposed via MCP (Phase 8), tools like Claude Desktop or Cursor could surface this content in contexts where it shouldn't appear (e.g., shared screens, logged sessions).

**Implementation:**

1. Tag documents with sensitivity levels during indexing:
   ```python
   SENSITIVITY_RULES = {
       "confidential": ["salary", "performance review", "termination", "series c", "valuation"],
       "internal": ["default"],   # Everything else
       "public": ["memory/glossary.md", "memory/context/company.md"]
   }
   ```

2. Add `sensitivity` column to documents table:
   ```sql
   ALTER TABLE documents ADD COLUMN sensitivity TEXT DEFAULT 'internal';
   ```

3. MCP server filters by sensitivity level:
   - Local CLI: no filtering (you're the only user)
   - MCP stdio (Claude Code): no filtering (local, your machine)
   - MCP HTTP (shared/remote): filter to `internal` + `public` only
   - Configurable via `kb mcp --sensitivity internal`

4. Entity-level sensitivity: mark entities like "Series B" or "Customer POC" as confidential. Any chunk mentioning them inherits the sensitivity tag.

**Why this matters:** Not urgent for the CLI-only phase, but **must be designed before MCP goes live**. Adding it after the fact means confidential content has already been exposed through the MCP surface.

**Effort:** Small for document-level tagging. Medium for entity-level inheritance. Should be designed alongside MCP (Phase 8).

### 10.6 Cross-Meeting Narrative Tracking

**The concept:** Surface the "story" of a topic across time:

```
$ kb narrative "Helix Refactor" --since 6m

# Helix Refactor — 6-Month Narrative (14 meetings)

## Sep 2025
- Engine rewrite kicked off. Talia leading. Target: 5x performance.
- Initial scope: 200 detectors of 600.

## Oct 2025
- First benchmarks: 3.2x improvement on converted detectors.
- Decision: prioritize high-frequency detectors first.

## Nov 2025
- 150/600 detectors converted. On track.
- Concern: integration testing bottleneck (raised by Linnea).

## Dec 2025
- 180/600 detectors. Slight delay from holidays.
- Decision: hire contractor for test harness.

## Jan 2026
- 200/600 detectors (~30%). Linnea raised platform dependency.
- Talia on leave Apr 2026 — succession plan needed.

## Feb 2026
- Status: ~30% complete, on trajectory for 50% by Q2.
- Open: Who covers Talia's leave?
```

**How it works:** This is a **composition feature** built on existing capabilities:
1. `kb entity timeline "Helix Refactor"` → all meetings mentioning it
2. Group by month
3. For each meeting, extract the highest-scoring chunk mentioning the entity
4. Optionally: use Claude to summarize each month's chunks into narrative bullets

**Effort:** Medium. The raw data is all there — this is mostly a formatter + optional LLM summarization pass.

---

## Final Backlog: 44 Items

### Phase 4 — Search (build these into the search pipeline)

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 1 | WAL mode pragma | Trivial | Safety | Add to `db.py` now |
| 2 | BM25 score normalization (sigmoid) | Trivial | **Required** | Everything downstream needs 0-1 scores |
| 3 | Strong signal fast path | Small | High | Skip vector when FTS is confident |
| 4 | FTS5 phrase + proximity search (NEAR) | Small | High | Big precision win for multi-word queries |
| 5 | Weighted RRF + top-rank bonus | Small | High | Original lists 2x weight; rank 1 gets +0.05 |
| 6 | Snippet extraction with line positions | Small | High | `@@ -start,count @@` headers for expand workflow |
| 7 | Recency-weighted scoring | Small | High | Exponential decay, 15% default weight, `--recency` flag |
| 8 | Search progress hooks | Small | Arch | Decouple pipeline from UI — hooks for CLI/MCP/tests |
| 9 | Graceful FTS-only fallback | Trivial | UX | Search works before vectors exist |

### Phase 5 — CLI & UX

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 7b | Three-tier search (`--fast`/`--deep`) | Small | Medium | User-controlled mode; `--auto` heuristics later |
| 10 | File path line references + ranges | Small | Medium | `kb view "path:10-20"` |
| 11 | Content-hash docids (`#abc123`) | Small | Medium | Stable short IDs; needs schema migration |
| 12 | Fuzzy document path resolution | Small | Medium | Exact → suffix → Levenshtein |
| 13 | Index health warnings | Small | Low | Stale index, missing embeddings |
| 14 | Progress bar during indexing | Small | UX | `tqdm` or stderr progress |
| 15 | Soft-delete / deferred deactivation | Small | Safety | `active` column, don't lose moved files |
| 15b | `kb usage` command (self-docs for agents) | Small | **High** | Linearis pattern — tool documents itself for AI. Replaces CLAUDE.md instructions. |
| 15c | Actionable error messages | Small | AI UX | Errors include `suggestion` + `available_actions` in JSON output |

### Model Upgrade (after Phase 6, do all at once)

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 16 | **Embedding model upgrade** | Medium | **High** | Qwen3-0.6B + late chunking + entity-enriched prefixes + document-level embeddings + larger transcript windows + instruction-aware embedding. Full re-embed. |

### Post-Model-Upgrade Improvements

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 17 | LLM result caching | Small | Medium | Cache reranking/expansion in SQLite |
| 18 | Search quality eval framework | Medium | **Confidence** | Hit@K by difficulty. Curate ~20 test queries. Run before/after model changes. |
| 19 | Schema migration framework | Small | Foundation | `migrations` table. Required before docids, cache, etc. |
| 20 | Cross-encoder reranking | Medium | High | Chunk-level reranking with position-aware blending |
| 21 | Entity-aware query expansion | Medium | High | Alias expansion using entity graph |
| 22 | Entity graph traversal (1-hop) | Medium | Discovery | "Who works with Soren?" via co-occurrence joins |
| 23 | Connected notes (`kb related`) | Small | Discovery | Vector similarity + entity overlap |
| 24 | Database cleanup commands | Small | Ops | Orphaned chunks/vectors, vacuum |
| 25 | Entity audit command | Medium | Maintenance | Find missing profiles, stale entities, new names |
| 26 | Global context injection | Small | AI UX | Entity hints in search result headers |
| 27 | Multi-get with globs | Small | Convenience | `kb view "meetings/2026/01/*"` |
| 28 | XML output format | Small | AI UX | `--format xml` for Claude |

### Context Layer (Phase 5-7) — Priority 9

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 33 | `kb context` (compressed entity index) | Medium | **Very High** | Auto-generated AGENTS.md-style index from entity graph. Eliminates decision points. |
| 34 | `kb memory add` (fact recording) | Small | Medium | Append facts to memory files + SQLite index. Replaces auto-memory. |
| 35 | `kb context --for` (filtered context) | Small | High | Only include entities relevant to current topic |
| 36 | Shrink CLAUDE.md (139 → ~25 lines) | Trivial | High | Move knowledge tables to kb context, keep only HOW + preferences |

### Meeting Intelligence (Phase 6-7) — Priority 10

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 39 | Speaker-attributed search | Small | **High** | Resolve `**System:**` → real name from title. `--speaker` filter. ~50 lines. |
| 40 | Pre-meeting briefing (`kb brief`) | Medium | **Very High** | Killer feature for leaders. Last meetings, open threads, entity connections. |
| 41 | Decision/commitment extraction | Small | Medium | Via `kb memory add --type decision`. Claude extracts, not heuristics. |
| 42 | Temporal fact tracking | Medium | Medium | Facts get `valid_from`/`valid_until`. Zep-inspired, SQL-only. |
| 43 | Cross-meeting narrative (`kb narrative`) | Medium | Medium | Topic story across time. Composition of entity timeline + formatter. |
| 44 | Sensitivity tagging | Small-Med | **Safety** | Document + entity level. Must be designed before MCP goes live. |

### Platform Expansion (Phase 8+)

| # | Item | Effort | Impact | Notes |
|---|---|---|---|---|
| 29 | MCP server (stdio + HTTP daemon) | Medium | Platform | 4-5 tools + URI resources + dynamic instructions. See CLI & MCP Design Patterns section. |
| 30 | Claude Code skill + score table | Small | AI UX | Teach Claude when/how to use kb; may be superseded by `kb usage` (#15b) |
| 31 | Custom update hooks per source | Small | Extensibility | Pre-index commands per source adapter |
| 32 | Update flow with source freshness | Small | UX | Report what changed during `kb ingest` |
| 37 | MCP resources (`kb://context`, `kb://entity/{name}`) | Small | Medium | Pre-loadable context for Claude Desktop/Cursor |
| 38 | SessionStart hook (auto-inject context) | Trivial | High | `kb context --compact --json` runs automatically at session start |
