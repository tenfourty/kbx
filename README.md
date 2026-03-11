# kbx — Local Knowledge Base with Hybrid Search

Give your AI agents persistent memory. Index your markdown notes, meeting transcripts, and documentation into a hybrid search engine. Search with keywords or natural language. Everything runs locally — your data never leaves your machine.

kbx combines SQLite FTS5 full-text search with LanceDB vector search using Qwen3 embeddings — all on-device, with Apple Silicon acceleration via MLX.

You can read more about kbx's progress in the [CHANGELOG](CHANGELOG.md).

## Quick Start

```sh
# Install
pip install kbx                        # core CLI + FTS5 search
pip install "kbx[search]"              # + vector search (Qwen3 embeddings)
pip install "kbx[search,mlx]"          # + Apple Silicon acceleration

# Set up a knowledge base
kbx init                               # create kbx.toml in the current directory

# Index your markdown files
kbx index run                          # index everything under memory/
kbx index run --no-embed               # text-only index (fast, no model needed)

# Search
kbx search "quarterly planning"        # hybrid search (FTS5 + vector)
kbx search "quarterly planning" --fast # keyword-only (~instant, no model needed)
kbx search "MFA rollout" --json        # structured output for scripts

# Browse
kbx view "memory/notes/decisions.md"   # read a document
kbx view "#a1b2c3"                     # by content-hash prefix
kbx list --type notes --from 2026-01-01
```

### Using with AI Agents

kbx is built for agentic workflows. The `--json` output format, structured error responses, and built-in agent playbook make it a natural fit for AI assistants.

```sh
# Orient: get a compressed overview of all entities (~2K tokens)
kbx context

# Search with structured output
kbx search "authentication" --fast --json --limit 5

# Look up a person
kbx person find "Wren" --json

# Timeline of everything mentioning a project
kbx person timeline "Helix Refactor" --from 2026-01-01 --json

# Take notes that persist across sessions
kbx memory add "Decision: use Postgres" --tags decision,infra --pin
kbx memory add "Promoted to Staff" --entity "Soren"

# Pin important docs to the context window
kbx pin "memory/notes/priorities.md"
```

When you run `kbx --help`, it prints an agent playbook alongside the standard CLI help — a complete reference for AI agents to self-orient and use the knowledge base effectively.

### MCP Server

kbx exposes an MCP server for tighter integration with Claude Desktop, Claude Code, Cursor, and other MCP-compatible tools.

**Tools exposed:**
- `kb_search` — Hybrid or FTS-only search with date/tag filters
- `kb_person_find` — Entity lookup by name, alias, or partial match
- `kb_person_timeline` — Chronological document list for an entity
- `kb_view` — Retrieve a document by path, glob, or `#hash`
- `kb_context` — Compressed entity index for session orientation
- `kb_memory_add` — Create notes or record facts about entities
- `kb_pin` / `kb_unpin` — Pin documents to the context window
- `kb_usage` — Index status and usage instructions

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kbx": {
      "command": "kbx",
      "args": ["mcp"]
    }
  }
}
```

**Claude Code** (`.claude/settings.local.json`):

```json
{
  "mcpServers": {
    "kbx": {
      "command": "kbx",
      "args": ["mcp"],
      "type": "stdio"
    }
  }
}
```

See [MCP plugin docs](docs/plugins/mcp.md) for full tool parameter reference.

### Python API

Use kbx as a library in your own applications:

```python
from kb import KnowledgeBase

with KnowledgeBase(thread_safe=True) as kb:
    # Search
    results = kb.search("cloud migration")

    # Entities
    people = kb.list_entities(entity_type="person")
    alice = kb.get_entity("Wren")
    timeline = kb.get_entity_timeline("Wren")

    # Context
    ctx = kb.context()

    # Index
    kb.index()
```

The `KnowledgeBase` class manages the full lifecycle — DB connections, embedder, auto-reindexing of stale files. All methods return Pydantic models.

See [architecture docs](docs/architecture.md) for the full API surface.

## Architecture

```
Markdown files (source of truth)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   Source Adapters                     │
│  meetings.py — walk memory/meetings/YYYY/MM/DD/     │
│  memory.py   — walk memory/people/, projects/, ...  │
└────────────────────────┬────────────────────────────┘
                         │ ParsedDocument
                         ▼
┌─────────────────────────────────────────────────────┐
│                      Indexer                          │
│  chunk → embed → store → link entities               │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐    ┌─────────────────────────────┐
│     SQLite        │    │         LanceDB              │
│  docs, chunks,    │    │  Qwen3-Embedding-0.6B        │
│  FTS5, entities,  │    │  1024-dim vectors             │
│  facts, mentions  │    │  float32, instruction-aware   │
└──────────────────┘    └─────────────────────────────┘
           │                          │
           └────────────┬─────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Hybrid Search                       │
│  FTS5 (BM25) + Vector → RRF Fusion → Recency Weight │
└─────────────────────────────────────────────────────┘
```

**Write-through principle:** Markdown files are the source of truth. All data writes go to flat files first; the database is a derived index rebuilt from those files. The DB is disposable — delete it and re-index.

## Search

kbx supports two search modes:

| Mode | Flag | Speed | Method |
|------|------|-------|--------|
| Fast | `--fast` | ~instant | FTS5 keyword search only |
| Hybrid | *(default)* | ~2s | FTS5 + vector search + RRF fusion |

Hybrid search uses Reciprocal Rank Fusion (RRF) to combine keyword and semantic results, with a 90-day half-life recency weight. A strong-signal fast path skips vector search entirely when FTS5 produces a high-confidence match.

**Score interpretation:** 0.8+ strong | 0.5–0.8 worth reading | <0.5 noise

See [search docs](docs/search.md) for the full pipeline, score normalisation, and fusion strategy.

## Entity System

kbx automatically links people, projects, teams, and glossary terms to your documents:

```sh
kbx person find "Wren" --json        # profile + linked documents
kbx person timeline "Wren"           # chronological mentions
kbx person create "Soren" --role "SRE Lead" --team "Platform"
kbx project find "Helix Refactor"    # project profile + linked docs
kbx entity stale --days 30            # entities not mentioned recently
```

Entities are seeded from `memory/people/*.md` and `memory/projects/*.md` files, then linked to documents via five-tier matching: YAML tags → title participants → title substrings → source IDs → content name matching.

See [entity docs](docs/entities.md) for the full linking pipeline.

## Sync & Ingest

Pull meeting transcripts from external sources:

```sh
# Granola API sync
kbx sync granola --since 2026-01-01

# Notion AI Meeting Notes sync
kbx sync notion --since 2026-01-01

# Granola zip export ingest
kbx ingest export.zip

# View and edit synced meeting notes
kbx granola view <calendar-uid>
kbx granola edit <calendar-uid> --append "Action: follow up with Wren"
```

Sync is incremental — only new or updated meetings are fetched. Attendees are automatically matched to existing entities. See [Granola plugin docs](docs/plugins/granola.md) for configuration.

## Configuration

kbx looks for configuration in this order:

1. `$KBX_CONFIG` environment variable
2. `./kbx.toml` in the current directory (walk up from CWD)
3. `~/.config/kbx/config.toml`

Run `kbx init` to generate a starter config.

### Optional Extras

| Extra | What it adds |
|-------|-------------|
| `search` | LanceDB + sentence-transformers + NumPy for vector search |
| `mlx` | [MLX backend](docs/plugins/mlx.md) for faster embeddings on Apple Silicon |
| `mcp` | [MCP server](docs/plugins/mcp.md) for AI tool integration |
| `all` | Everything above plus test and dev dependencies |

Install with: `pip install "kbx[search,mlx,mcp]"`

Requires Python 3.10+.

## Data Storage

Index stored in the data directory (configurable via `kbx.toml` or `$KB_DATA_DIR`):

```
kbx-data/
├── metadata.db        # SQLite — documents, chunks, FTS5, entities, facts
└── vectors/           # LanceDB — Qwen3 embedding vectors (1024-dim)
```

The database is a derived index. Delete it and `kbx index run` to rebuild from your markdown files.

## Development

```sh
git clone https://github.com/tenfourty/kbx.git
cd kbx
uv sync --all-extras
uv run pre-commit install
uv run pytest -x -q --cov           # 1361 tests, 90%+ coverage
uv run mypy src/                     # strict mode
```

Quick CI check locally:

```sh
make ci                              # mirror exact GitHub CI pipeline
make fix                             # auto-fix lint + format issues
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [testing docs](docs/testing.md) for the test strategy.

## Documentation

| Doc | What it covers |
|-----|---------------|
| [Architecture](docs/architecture.md) | System design, data flow, module dependencies, Python API |
| [Search](docs/search.md) | FTS5 + vector + RRF fusion pipeline, score normalisation |
| [Entities](docs/entities.md) | Entity seeding, five-tier linking, disambiguation |
| [Indexing](docs/indexing.md) | Walk → chunk → embed → store pipeline |
| [Chunking](docs/chunking.md) | Markdown-aware chunking strategy |
| [CLI Reference](docs/cli.md) | All commands and options |
| [Output Formatting](docs/output.md) | JSON, table, CSV, JSONL, jq, field selection |
| [Context Layer](docs/context.md) | Compressed entity index for AI agents |
| [Testing](docs/testing.md) | Test strategy, fixtures, markers |
| [MCP Plugin](docs/plugins/mcp.md) | MCP server tools and resources |
| [MLX Plugin](docs/plugins/mlx.md) | Apple Silicon embedding acceleration |
| [Granola Plugin](docs/plugins/granola.md) | Meeting transcript sync (view, edit, push) |
| [Notion Plugin](docs/plugins/notion.md) | Notion AI Meeting Notes sync |
| [Integration](docs/integration.md) | Ingest, migrations, search quality |

## License

Apache-2.0
