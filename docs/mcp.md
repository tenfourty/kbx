# MCP Server

> **Note:** This page covers the MCP server's internal architecture. For setup instructions and tool parameter reference, see [plugins/mcp.md](plugins/mcp.md).

## Overview

The kbx MCP server exposes knowledge base functionality as MCP tools and resources, allowing Claude Desktop, Claude Code, and other MCP-compatible clients to query the knowledge base directly.

The MCP server wraps existing kbx functionality — no new search or entity logic. It uses `fast=True` (FTS-only) search by default to avoid loading the embedding model, making tool calls instant.

## Architecture

```
src/kb/config.py      — Shared helpers (find_project_root, get_db, find_entity)
src/kb/mcp_server.py  — FastMCP server + handler functions
src/kb/cli.py         — CLI commands (imports from config.py)
```

### Handler Functions

Each MCP tool wraps a `handle_*` function that takes a `db` parameter:

```python
from kb.mcp_server import handle_kb_search

result = handle_kb_search(db, "MFA", fast=True, limit=5)
data = json.loads(result)
```

These functions are tested directly (no MCP transport in tests).

### Shared Helpers (config.py)

Helpers used by both cli.py and mcp_server.py:

- `find_project_root()` — Walks up from CWD to find project root
- `get_data_dir()` — Uses `KB_DATA_DIR` env var or auto-detects
- `get_db()` — Returns a Database instance
- `find_entity(conn, name)` — Case-insensitive entity lookup with alias/partial match

## Differences from CLI

| Feature | CLI | MCP |
|---------|-----|-----|
| Search default | Hybrid (vector+FTS) | FTS-only (`fast=True`) |
| Embedding model | Loaded on demand | Never loaded |
| Output | Table/JSON/JSONL/CSV | JSON strings only |
| Transport | Terminal | stdio |
| Logging | stderr | stderr (never stdout) |
