# kb — Knowledge Base CLI

Give Claude Code persistent memory across sessions. `kb` indexes meeting transcripts, memory files, and entity records into a hybrid search engine so agents can find people, decisions, and context without manual file browsing. SQLite + LanceDB + FTS5 + Qwen3 embeddings.

IMPORTANT: Prefer reading docs/ and source files over guessing. The index below tells you where everything lives.

## Setup

```bash
uv sync --all-extras && uv run pre-commit install   # first time
uv run pytest -x -q --cov && uv run mypy src/       # verify
uv run kb --help                                      # CLI self-documentation (includes agent playbook)
```

## Common Commands

```bash
uv run pytest tests/test_search.py::test_name -x -v  # single test
uv run pytest -n auto -x -q --cov                     # parallel (xdist)
uv run ruff check src/ --fix                           # lint + autofix
uv run ruff format src/                                # format
uv run kb mcp                                          # start MCP server (stdio)
```

Pre-commit hooks enforce everything (ruff, mypy, bandit, pytest+cov). Trust the hooks.

**Quick CI check locally:**
```bash
make ci                          # mirror exact GitHub CI pipeline
make fix                         # auto-fix lint + format issues
./scripts/ci-local.sh            # same as make ci, with --fix flag available
./scripts/ci-local.sh --fix      # auto-fix then run full checks
```

## TDD Workflow

Every new feature must have:
1. **A happy-path integration test** — exercises the real SQLite + LanceDB layer using the test fixture DB (`KB_DATA_DIR` isolation). No mocking of internal modules — test the actual pipeline end-to-end.
2. **A happy-path E2E test** — a subprocess smoke test invoking `kbx <command>` as a real process and asserting on stdout/exit code (see `tests/test_cli.py` for the pattern).

**TDD order is mandatory:**
- Write integration and E2E tests first (red), then implement until they pass (green)
- When fixing a bug: write a test that reproduces the bug first, then fix the code
- Never write implementation code before a failing test exists

**Exceptions (must be explicitly noted in a comment, not silently skipped):**
- Pure infrastructure (migrations, NFC normalization, cache warming) — unit test only is acceptable
- MCP handler functions — handler unit tests are sufficient; do not test FastMCP transport wiring (that's FastMCP's responsibility)
- ML model behaviour (embedding quality, search tuning) — covered by `tests/eval_queries.py`, not pytest

Quick commands:
```bash
uv run pytest tests/test_search.py::test_name -x -v  # single test
uv run pytest -n auto -x -q --cov                     # parallel (xdist)
uv run pytest -m "not slow" -x -q                     # skip embedding model tests
```
Then: `uv run mypy src/` — clean before committing.

## Architecture

`sources/` (walk_*) → `chunker.py` (parse + chunk) → `indexer.py` (store + embed + link) → `search.py` (FTS5 + vector + RRF)

**Write-through principle:** Markdown files are the source of truth. All data writes go to flat files first; the DB is a derived index rebuilt from those files. Never write to the DB without a corresponding file.

**The boundary rule:** Sources yield `ParsedDocument`, indexer stores to SQLite + LanceDB, search returns `SearchResponse`. Types in `types.py` (Pydantic strict).

## Index

```
docs/|root: ./docs
|architecture.md — system design, data flow, component relationships
|chunking.md — markdown-aware chunking strategy (notes by ##, transcripts by ¶)
|cli.md — CLI commands reference
|context.md — compressed entity index for AI agents (~2K tokens)
|entities.md — entity system: seeding, regex linking, CRUD
|indexing.md — walk → chunk → embed → store pipeline
|integration.md — Granola ingest, import/export
|mcp.md — MCP server mode (stdio transport)
|output.md — render pipeline (table/json/jsonl/csv + fields + jq)
|search.md — FTS5 + vector + RRF fusion + recency weighting
|testing.md — test strategy, fixtures, markers

src/kb/|root: ./src/kb
|api.py — KnowledgeBase service class (public Python API — all write operations live here)
|cli.py — Click commands, all output via kb_output()
|config.py — project root detection, get_db() singleton
|types.py — Pydantic v2 strict models (ParsedDocument, SearchResult, Entity…)
|db.py — SQLite + LanceDB wrapper, migrations, NFC path normalization
|indexer.py — orchestrates: walk → chunk → embed → store → link entities
|search.py — hybrid search: FTS5 + vector + RRF fusion + recency weighting
|scoring.py — centralised score normalisation + compose_scores() for blended retrieval
|hotness.py — frequency x recency hotness score for #67 boost
|access.py — touch_document/touch_entity + reset helpers for hotness tracking
|entity_embeddings.py — entity-as-coherent-object vector embeddings for #69 Pass 1
|embeddings.py — Qwen3-Embedding-0.6B (MLX on Apple Silicon, PyTorch fallback)
|entities.py — entity seeding from memory/ + regex-based mention linking + source-ID linking
|matching.py — task-to-project matching (TypedDicts: TaskInput, ProjectInput) + sources extraction/formatting
|chunker.py — markdown-aware chunking (notes by ##, transcripts by ¶)
|abstracts.py — extractive L0 abstracts + L1 overviews (sentence/paragraph + title fallback, no LLM, #66 P1+P4)
|explain.py — human-readable formatter for `kbx search --explain` (#68 P2)
|context.py — compressed entity index for AI agents
|output.py — render pipeline (table/json/jsonl/csv + fields + jq)
|crud.py — entity CRUD with markdown file sync + find_document_by_target() shared resolver
|writeback.py — DB → markdown file sync (atomic writes)
|staleness.py — auto-reindex changed memory files on next search
|glossary.py — glossary term CRUD (memory/glossary.md)
|dateparse.py — natural date parsing ("since January", "last 7 days")
|mcp_server.py — MCP server mode (stdio transport)
|sources/{meetings,memory}.py — walk meetings/ and memory/ directories
|sync/granola_public.py — Granola public-API sync (default; Bearer key from Keychain/env)
|sync/granola.py — Granola internal-API sync (legacy, reachable via `kbx sync granola --legacy`)
```

## Conventions

- mypy strict, Pydantic v2 (`strict=True`; `StrictFrozen` for immutable, `StrictMutable` for models mutated at runtime), Python 3.10+
- Line length 100 (ruff enforced)
- Coverage minimum 90% — enforced by pre-commit
- **`kb --help` is the LLM API contract** — any CLI change MUST update `_AGENT_PLAYBOOK` in `cli.py`
- All file writes atomic (temp file → rename)
- NFC Unicode normalization on all paths (macOS compat)

## Claude Code Sandbox

The Claude Code sandbox injects SOCKS proxy env vars (`ALL_PROXY=socks5h://localhost:...`) that break `httpx` (requires `socksio` package). This affects any HTTP call — Granola sync, API calls, HuggingFace model downloads.

**How it's handled:**
- **CLI startup** (`cli.py`): Strips `ALL_PROXY`, `all_proxy`, `FTP_PROXY`, `ftp_proxy`, `GRPC_PROXY`, `grpc_proxy`, `RSYNC_PROXY` before any imports. Covers all CLI commands.
- **Tests** (`conftest.py`): Session-scoped autouse fixture strips the same vars + sets `HF_HUB_OFFLINE=1` so tests never hit the network.
- **Programmatic use**: If using `kb` as a library (e.g., `from kb.sync.granola import GranolaClient`), callers must strip proxy vars themselves — the CLI-level fix doesn't apply.

**Other sandbox notes:**
- DNS resolution can fail even for allowlisted hosts when running `uv run python3 -c "..."` — may need `dangerouslyDisableSandbox` for direct API calls outside the CLI entry point.
- GPG signing requires the passphrase to be pre-cached in `gpg-agent` from a regular terminal — `pinentry-mac` can't show a dialog from Claude Code subprocesses.
- Embedding model tests are skipped when the Qwen3 model isn't in the local HF cache (`~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/`).

## Claude Code Hooks

`.claude/settings.json` configures three hooks (matching gm's pattern):
- **PreToolUse** (`block-sensitive-files.sh`): Blocks Edit/Write to `.env*` and `uv.lock`
- **PostToolUse** (`post-edit-lint.sh`): Auto-runs `ruff check --fix` + `ruff format` on `.py` files after Edit/Write
- **PostToolUse** (`check-agent-playbook.sh`): When `src/kb/cli.py` is edited, emits a reminder to verify `_AGENT_PLAYBOOK` still reflects the change (commands, flags, JSON shapes). `kb --help` is the LLM API contract.

Scripts in `.claude/hooks/` read stdin JSON for `tool_input.file_path` and `cwd`.

## KnowledgeBase API (`api.py`)

All write operations go through `KnowledgeBase` methods. CLI and MCP are thin wrappers — zero duplicated business logic.

- **Construction**: `KnowledgeBase(project_root=..., data_dir=...)` for fresh DB, or `KnowledgeBase._from_existing(db=db, project_root=...)` to reuse an existing `Database` instance (no throwaway connection). `_from_existing` sets `_owns_db=False` so `close()` won't close the shared DB.
- **Entity profiles**: `get_entity_profile(name)` returns facts (with entity-scoped `seq` IDs), timestamps, doc count, CLI breadcrumbs, and MCP breadcrumbs (tool + params).
- **Facts**: `add_fact(entity, text, date)` assigns next `seq` per entity. `edit_fact(entity, seq, ...)` and `delete_fact(entity, seq)` address facts by `(entity_name, seq)`. Facts are entity-scoped (deterministic on DB rebuild from files).
- **Notes**: `add_note()`, `edit_note()`, `delete_note()`, `list_notes(tag, pinned_only, limit)`.
- **Documents**: `view_document(target)` (resolves by path/hash/title/glob), `list_documents(doc_type, from_date, to_date, limit, since_hours)`.
- **Listings**: `list_entities()`, `get_entity_timeline(name, from_date, to_date, doc_type, limit)`, `list_facts(since_days, entity)`, `get_stale_entities()`, `get_index_stats()`.
- **Corrections**: `correct_term(term, replacement, apply, scope, file_type, word_boundary, ignore_case)` — scan/dry-run/apply modes.
- **Write-through principle**: All fact/note/entity writes go to markdown files first, then DB. `_append_fact_to_file()` + `_set_pin_frontmatter()` handle file-level writes.
- **Facts round-trip**: `_seed_facts_from_files()` in `entities.py` parses `## Recent Facts` from entity markdown files during `seed_entities()`, inserting missing facts with next available `seq`. Ensures facts survive DB deletion + rebuild.

## MCP Server

- **31 tools** with full MCP tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`)
- **Handler pattern**: thin `handle_*` wrappers that create `KnowledgeBase._from_existing(db, project_root)` and delegate to API methods. MCP tool functions call `get_db()`/`find_project_root()` then pass to handlers.
- **Entity CRUD tools**: `kb_person_create`, `kb_person_edit`, `kb_project_create`, `kb_project_edit` — metadata via named params + `meta` string (semicolon-delimited `key=value` pairs).
- **Fact tools**: `kb_memory_add` (fact or note), `kb_memory_edit_fact(entity, fact_seq, ...)`, `kb_memory_delete_fact(entity, fact_seq)` — entity-scoped seq IDs.
- **Entity find**: `kb_person_find` / `kb_project_find` return facts with `seq` IDs + `mcp_breadcrumbs` (tool + params for agents) alongside CLI breadcrumbs.
- **`kb_usage`**: Structured JSON stats (docs, entities, facts, pinned, date_range, tool_count).
- **Error shape**: `{"error": str, "suggestion": str | null}` — all tools
- **List shape**: `{"results": [...], "meta": {"total": N, "limit": N}}` — all list tools. `total` is the true count, not capped by limit.
- **Tool annotations**: Defined as `_READ_ONLY`, `_MUTATING`, `_MUTATING_IDEMPOTENT`, `_DESTRUCTIVE` presets.
- **Document resolution**: Shared `crud.find_document_by_target()` — path, hash, title, glob, substring. CLI uses `strict=True` (raises `AmbiguousDocumentError`), MCP uses `strict=False` (returns None).
- **Pin/unpin writeback**: `kb_pin`/`kb_unpin` write `pinned: true/false` to YAML frontmatter for memory notes (survives reindex).
- **Tag filtering**: Only works on memory notes (`memory_note`, `memory_doc`). Meeting docs don't have tags — use `doc_type` filter instead.

## Granola Sync (Path D, since 2026-05-18)

- **Default path** = `kb.sync.granola_public` against `https://public-api.granola.ai/v1` (note: **not** `api.granola.ai`, which is the internal API used by the desktop app and `jlokos/granola`). Bearer auth via long-lived Personal API key.
- **Key resolution** (`_get_api_key`): `GRANOLA_API_KEY` env first, then `security find-generic-password -a $USER -s granola_api_key -w`. Never logged or included in exception text.
- **Endpoints used**: `GET /v1/notes` (cursor pagination, max `page_size=30`, server-side `updated_after`) + `GET /v1/notes/{id}?include=transcript`. Rate limit 5 req/sec sustained, 25 burst.
- **Schema rename when porting from legacy doc shape**: `calendar_event.calendar_event_id` → `iCalUID`; `organiser` → `organizer.email`; `scheduled_start_time` → `start.dateTime`. `_note_to_internal_doc` does this so existing `build_frontmatter` + `write_meeting` round-trip unchanged.
- **Transcript source enum**: public uses `["microphone", "speaker"]`, internal pipeline uses `["microphone", "system"]`. `_transform_transcript` maps `speaker` → `system`. The word "speaker" in the public schema means *system audio*, not a person; `speaker.diarization_label` (when present) is what carries the real per-speaker label, promoted to the `speaker` key for `transcript_to_markdown`.
- **No churn rule**: public-path files keep `source: granola-api` for parity with legacy writes; path provenance recorded via a separate `granola_api: public` frontmatter field. Renaming `source` would mark every legacy file as "updated" on first sync.
- **Field-coverage loss vs legacy**: public `User` is `{name, email}` only. Per-attendee `company` tag and mailing-list `group.members` expansion (mined from `details` by `build_frontmatter`) are unavailable. Existing legacy files keep whatever they captured.
- **Pagination**: `list_notes` raises `GranolaPublicAPIError` if (a) `hasMore` field absent from response (defensive against future rename), (b) cursor fails to advance (server-bug guard). Warns and breaks if `hasMore=true` with null cursor.
- **Timestamp comparisons** use `_parse_iso` → UTC-aware `datetime`; never compare ISO strings directly (Z vs ±HH:MM vs naive break lex order). `last_sync` state file stores full ISO; don't slice to `[:10]`.
- **State checkpointing**: every 10 notes + in `finally` block, so a mid-loop 429/crash keeps progress.
- **Legacy fallback**: `kbx sync granola --legacy` still works for one release window with a stderr deprecation banner; reads `~/Library/Application Support/Granola/supabase.json`. Granola 7.205.0+ stopped maintaining that file (credentials moved to Electron `safeStorage`-encrypted `supabase.json.enc` + `storage.dek`), so `--legacy` will hard-fail on current Granola versions — kept only for rollback safety.
- **Write path removed**: `kbx granola push`, `kbx granola edit`, the `kb_granola_edit` MCP tool, and `GranolaClient.update_document_notes`/`create_document` were removed in kbx 0.2.0 (see CHANGELOG). Legacy internal-API auth broke in Granola 7.205.0+, and the public API exposes no write endpoint. ProseMirror→Markdown rendering is still required for reads and is retained; Markdown→ProseMirror, the Yjs ydoc bridge (`sync/ydoc.py`), and the `pycrdt` dependency went with the write path.
- **Meeting frontmatter null guard**: `walk_meetings` (in `sources/meetings.py`) coerces `attendee.email: null` to `""` before constructing `ParsedDocument`. Required because cos-agent prep files often have group/room invitees without emails (`email: null` in YAML). Without this, Pydantic strict validation crashed the entire indexer (regression observed 2026-05-11).

## Gotchas

- **MPS memory** — embedding batches: 32 on MPS, 16 on CPU; texts truncated at 8K chars. `torch.mps.empty_cache()` between batches
- **Model cache** — ~1.1GB at `~/.config/kbx/model/`. First run downloads automatically
- **`KB_DATA_DIR`** — env var overrides default `~/.config/kbx/`. Tests use this to isolate
- **Incremental indexing** — skips unchanged files (by `content_hash`). Use `--full` to force
- **LanceDB lazy import** — only loaded for vector ops. `--no-embed` skips entirely
- **LanceDB `list_tables()`** — returns a `ListTablesResponse`, NOT a list. Check membership via `"name" in db.list_tables().tables`. `db.table_names()` is deprecated.
- **Entity list-value preservation** — the entity parser keeps YAML list values as Python lists (person + project parsers in `entities.py`); `types.py` metadata fields are `dict[str, Any]`. e.g. `task_keywords: [AI, agentic]` → `metadata['task_keywords'] == ['AI', 'agentic']`, not a stringified list.
- **Slow tests** — `@pytest.mark.slow` on ML model tests; pre-commit skips them (`-m "not slow"`). Run all with `uv run pytest`. The model-cache check (`conftest._model_cached`) looks in `get_data_dir()/model` (where kbx stores it), so these tests run locally; they skip on CI where the model is absent.
