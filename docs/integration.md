# Integration & Polish (Phase 6)

## `kb ingest` — Granola Export Pipeline

Wraps the organise + index workflow into a single command:

```bash
kb ingest                        # auto-discover Granola exports, organise, then index
kb ingest path/to/export.zip     # specific export
kb ingest --dry-run              # preview only
kb ingest --skip-organise        # only run index, skip the organise step
```

### How it works

1. **Organise** (unless `--skip-organise`): Runs `meetings/organise_granola.py` as a subprocess from the project root. Passes any path arguments and `--dry-run` flag through.
2. **Index**: Loads the Embedder and calls `index_all()` directly. The Embedder is only loaded after organise completes (expensive ~600MB model).
3. All progress goes to stderr; structured output to stdout.

### Dry run

`--dry-run` passes through to `organise_granola.py` and skips the index step entirely.

## Schema Migrations

Migrations are defined in `kb/db.py` as a list of `(name, sql)` tuples:

```python
MIGRATIONS = [
    ("001_add_file_mtime", "ALTER TABLE documents ADD COLUMN file_mtime TEXT"),
]
```

### How they work

1. `Database.__init__()` calls `_apply_migrations()` after schema creation
2. The function checks the `migrations` table for already-applied migrations
3. New migrations are executed with `try/except` for idempotency (e.g. column already exists)
4. Each applied migration is recorded in the `migrations` table

### Adding a new migration

Append a tuple to the `MIGRATIONS` list:

```python
MIGRATIONS = [
    ("001_add_file_mtime", "ALTER TABLE documents ADD COLUMN file_mtime TEXT"),
    ("002_add_soft_delete", "ALTER TABLE documents ADD COLUMN active INTEGER DEFAULT 1"),
]
```

Migrations run automatically on next `Database()` instantiation. No manual steps needed.

### Idempotency

- `CREATE TABLE IF NOT EXISTS` for the migrations table itself
- `try/except` around each migration SQL (column may already exist)
- `INSERT OR IGNORE` for the migration record
- Creating the database twice produces identical results

## Search Quality Evaluation

### Standalone script

```bash
uv run python kb/tests/eval_queries.py      # from project root directory
```

Runs 20 test queries against the real indexed data (default `~/.config/kbx/`). Reports Hit@1, Hit@3, Hit@5 metrics. Exits 0 if all queries have at least Hit@5, exits 1 otherwise.

To test against a different database:

```bash
KB_DATA_DIR=/path/to/db uv run python kb/tests/eval_queries.py
```

### Pytest integration

`kb/tests/test_integration.py::TestSearchQualityEval` runs a subset of queries against the test fixture DB (not the real index). These tests verify the search pipeline works correctly with known data.

## `.gitignore`

The project root `.gitignore` excludes:

- `~/.config/kbx/` — database files, vectors, model cache (default location)
- `.venv/` — virtual environments
- `__pycache__/`, `*.pyc`, `*.pyo` — Python bytecode
- `.DS_Store` — macOS metadata

## Test Structure

| File | Tests | What |
|------|-------|------|
| `test_smoke.py` | 12 | Embedder, SQLite schema, LanceDB, output formatting |
| `test_entities.py` | 19 | Entity seeding, linking, YAML extraction |
| `test_indexing.py` | 25 | Chunking, parsing, indexer pipeline |
| `test_search.py` | 29 | BM25, RRF, recency, FTS, snippets, hooks |
| `test_cli.py` | 38 | All CLI commands via CliRunner |
| `test_integration.py` | 12 | Ingest, migrations, search quality, end-to-end |
| `test_context.py` | 17 | Context generation, filtering, glossary |
| `test_mcp.py` | 31 | MCP tools, resources, config helpers |
| **Total** | **183** | |

Run all tests:

```bash
cd kb && uv run pytest tests/ -v
```
