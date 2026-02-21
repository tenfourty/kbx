# MCP Server (Phase 8)

## Overview

The kb MCP server exposes knowledge base functionality as MCP tools and resources, allowing Claude Desktop, Claude Code, and other MCP-compatible clients to query the knowledge base directly.

The MCP server wraps existing kb functionality — no new search or entity logic. It uses `fast=True` (FTS-only) search by default to avoid loading the embedding model, making tool calls instant.

## Setup

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "kb": {
      "command": "kb",
      "args": ["mcp"]
    }
  }
}
```

### Claude Code (`.claude/settings.local.json`)

```json
{
  "mcpServers": {
    "kb": {
      "command": "kb",
      "args": ["mcp"],
      "type": "stdio"
    }
  }
}
```

### CLI

```bash
kb mcp    # Start MCP server on stdio transport
```

> **Note:** Requires `kb` to be installed globally via `uv tool install --editable .` from the project directory. If `kb` is not on PATH in your MCP client's environment, use the full path: `uv --directory "/path/to/project" run kb mcp`.

## MCP Tools

### `kb_search`

Search the knowledge base using FTS (fast, default) or hybrid search.

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `query` | string | required | Search query |
| `fast` | bool | true | FTS-only search (~instant). Set false for hybrid vector+FTS |
| `limit` | int | 5 | Max results |

Returns JSON with `results` array containing `title`, `path`, `date`, `score`, `snippet`.

### `kb_entity_find`

Look up an entity (person, project, team) by name.

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | required | Entity name, alias, or partial match |

Returns JSON with entity profile (`name`, `entity_type`, `aliases`, `metadata`) and linked `documents`.

### `kb_entity_timeline`

Chronological list of documents mentioning an entity.

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | required | Entity name |
| `from_date` | string | null | Start date (YYYY-MM-DD) |
| `to_date` | string | null | End date (YYYY-MM-DD) |

Returns JSON with `documents` array sorted by date ascending.

### `kb_view`

View a specific document by path or content-hash prefix.

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `target` | string | required | Document path, suffix, or `#hash` prefix |

Returns JSON with document metadata and all `chunks`.

### `kb_context`

Compressed entity index — overview of all people, projects, teams, and terms.

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `topic` | string | null | Filter to entities relevant to a topic |

Returns JSON with `text` (formatted index), `stats`, and `entities` array.

### `kb_usage`

Usage instructions and current index status (document/entity counts, date range).

Returns plain text with tool descriptions and recommended workflow.

## MCP Resources

### `kb://context`

Full compressed entity index. Use at session start for an overview of all entities.

### `kb://entity/{name}`

Entity profile and linked documents for a specific entity.

## Architecture

```
kb/config.py     — Shared helpers (find_project_root, get_db, find_entity)
kb/mcp_server.py — FastMCP server + handler functions
kb/cli.py        — CLI commands (imports from config.py)
```

### Handler Functions

Each MCP tool wraps a `handle_*` function that takes a `db` parameter:

```python
from kb.mcp_server import handle_kb_search

result = handle_kb_search(db, "MFA", fast=True, limit=5)
data = json.loads(result)
```

These functions are tested directly (no MCP transport in tests).

### config.py Refactor

Shared helpers extracted from cli.py into config.py:

- `find_project_root()` — Walks up from CWD to find project root
- `get_data_dir()` — Uses `KB_DATA_DIR` env var or auto-detects
- `get_db()` — Returns a Database instance
- `find_entity(conn, name)` — Case-insensitive entity lookup with alias/partial match

Both cli.py and mcp_server.py import from config.py.

## Differences from CLI

| Feature | CLI | MCP |
|---------|-----|-----|
| Search default | Hybrid (vector+FTS) | FTS-only (`fast=True`) |
| Embedding model | Loaded on demand | Never loaded |
| Output | Table/JSON/JSONL/CSV | JSON strings only |
| Transport | Terminal | stdio |
| Logging | stderr | stderr (never stdout) |
