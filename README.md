# kbx

Local knowledge base CLI with hybrid search over markdown files. Indexes
meeting transcripts, notes, and entity records into SQLite (FTS5) and LanceDB
(vector) for fast retrieval by humans and AI agents.

## Install

```bash
pip install kbx                      # core CLI + FTS5 search
pip install "kbx[search]"            # + vector search (Qwen3 embeddings)
pip install "kbx[search,mlx]"        # + Apple Silicon acceleration
```

Requires Python 3.10+.

## Quick Start

```bash
kbx init                   # create kbx.toml in the current directory
kbx index run              # index markdown files
kbx search "quarterly planning"      # hybrid search (FTS5 + vector)
kbx search "quarterly planning" --fast   # keyword-only (no model needed)
```

## Features

- **Full-text search** -- SQLite FTS5 with BM25 ranking and natural date filters
- **Vector search** -- Qwen3-Embedding-0.6B via sentence-transformers, fused with FTS5 using reciprocal rank fusion (RRF)
- **Entity linking** -- auto-links people, projects, and glossary terms to documents via regex matching
- **Entity CRUD** -- manage people, projects, and glossary terms from the CLI with markdown file sync
- **MCP server** -- stdio transport for integration with Claude, Cursor, and other AI tools
- **Granola sync** -- pull meeting transcripts from the Granola API or ingest local exports
- **Configurable** -- `kbx.toml` controls source directories, search behaviour, and extras
- **Incremental indexing** -- content-hash based; only re-indexes changed files

## Configuration

kbx looks for configuration in this order:

1. `$KBX_CONFIG` environment variable
2. `./kbx.toml` in the current directory
3. `~/.config/kbx/config.toml`

Run `kbx init` to generate a starter config file.

## Optional Extras

| Extra | What it adds |
|-------|-------------|
| `search` | LanceDB + sentence-transformers + NumPy for vector search |
| `mlx` | MLX backend for faster embeddings on Apple Silicon |
| `mcp` | MCP server for AI tool integration |
| `all` | Everything above plus test and dev dependencies |

Install with: `pip install "kbx[search,mlx,mcp]"`

## Development

```bash
git clone https://github.com/tenfourty/kbx.git
cd kbx
uv sync --all-extras
uv run pre-commit install
uv run pytest -x -q --cov
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0
