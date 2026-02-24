# kb — Knowledge Base CLI

Give Claude Code persistent memory across sessions. `kb` indexes meeting transcripts, memory files, and entity records into a hybrid search engine so agents can find people, decisions, and context without manual file browsing. SQLite + LanceDB + FTS5 + Qwen3 embeddings.

IMPORTANT: Prefer reading docs/ and source files over guessing. The index below tells you where everything lives.

## Setup

```bash
uv sync --all-extras && uv run pre-commit install   # first time
uv run pytest -x -q --cov && uv run mypy src/       # verify
uv run kb usage                                       # CLI self-documentation
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

## TDD Workflow

1. Write failing test in `tests/`
2. Implement in `src/kb/`
3. `uv run pytest -x -q` — green, then `uv run mypy src/` — clean

## Architecture

`sources/` (walk_*) → `chunker.py` (parse + chunk) → `indexer.py` (store + embed + link) → `search.py` (FTS5 + vector + RRF)

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
|api.py — KnowledgeBase service class (public Python API for external consumers)
|cli.py — Click commands, all output via kb_output()
|config.py — project root detection, get_db() singleton
|types.py — Pydantic v2 strict models (ParsedDocument, SearchResult, Entity…)
|db.py — SQLite + LanceDB wrapper, migrations, NFC path normalization
|indexer.py — orchestrates: walk → chunk → embed → store → link entities
|search.py — hybrid search: FTS5 + vector + RRF fusion + recency weighting
|embeddings.py — Qwen3-Embedding-0.6B (MLX on Apple Silicon, PyTorch fallback)
|entities.py — entity seeding from memory/ + regex-based mention linking
|chunker.py — markdown-aware chunking (notes by ##, transcripts by ¶)
|context.py — compressed entity index for AI agents
|output.py — render pipeline (table/json/jsonl/csv + fields + jq)
|crud.py — entity CRUD with markdown file sync
|writeback.py — DB → markdown file sync (atomic writes)
|staleness.py — auto-reindex changed memory files on next search
|glossary.py — glossary term CRUD (memory/glossary.md)
|dateparse.py — natural date parsing ("since January", "last 7 days")
|mcp_server.py — MCP server mode (stdio transport)
|sources/{meetings,memory}.py — walk meetings/ and memory/ directories
|sync/granola.py — Granola API sync + attendee matching
```

## Conventions

- mypy strict, Pydantic v2 (`strict=True`; `StrictFrozen` for immutable, `StrictMutable` for models mutated at runtime), Python 3.10+
- Line length 100 (ruff enforced)
- Coverage minimum 90% — enforced by pre-commit
- **`kb usage` is the LLM API contract** — any CLI change MUST update `usage()` in `cli.py`
- All file writes atomic (temp file → rename)
- NFC Unicode normalization on all paths (macOS compat)

## Gotchas

- **MPS memory** — embedding batches: 32 on MPS, 16 on CPU; texts truncated at 8K chars. `torch.mps.empty_cache()` between batches
- **Model cache** — ~1.1GB at `~/.config/kbx/model/`. First run downloads automatically
- **`KB_DATA_DIR`** — env var overrides default `~/.config/kbx/`. Tests use this to isolate
- **Incremental indexing** — skips unchanged files (by `content_hash`). Use `--full` to force
- **LanceDB lazy import** — only loaded for vector ops. `--no-embed` skips entirely
- **Slow tests** — `@pytest.mark.slow` on ML model tests; pre-commit skips them (`-m "not slow"`). Run all with `uv run pytest`
