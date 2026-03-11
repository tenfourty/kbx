# MCP Plugin

MCP (Model Context Protocol) is a standard for exposing tools and resources to AI assistants. The MCP plugin lets AI agents (Claude, etc.) query the knowledge base directly via a stdio-based server.

## Install

```bash
pip install "kbx[mcp]"
```

Depends on `mcp>=1.2` (FastMCP).

## Usage

```bash
kbx mcp
```

Starts the MCP server on stdio transport. Configure your AI tool to connect to this process.

## Available Tools

| Tool | Description |
|------|-------------|
| `kb_search` | Search the knowledge base. Supports `fast` (FTS-only, default) and hybrid (FTS + vector). Filter by date range, tag, sort by score or date. |
| `kb_person_find` | Look up a person by name, alias, or partial match. Returns profile and linked documents. |
| `kb_person_timeline` | Chronological list of documents mentioning a person. Optional date range filter. |
| `kb_view` | View a document by path, glob, or content-hash prefix (`#abc123`). Returns metadata and all chunks. |
| `kb_context` | Compressed entity index -- overview of people, projects, teams, terms. Supports `compact` (pipe-delimited) and `human` (markdown) formats. Optional topic filter. |
| `kb_usage` | Usage instructions and current index status (doc/entity/fact counts, date range). |
| `kb_memory_add` | Create a searchable note or record a fact about an entity. Supports tags, pinning, and entity linking. |
| `kb_pin` | Pin a document so it appears in `kb_context` output. |
| `kb_unpin` | Remove a document from context. |
| `kb_entity_stale` | List entities not mentioned in recent documents. Filter by days threshold. |

## Resources

| URI | Description |
|-----|-------------|
| `kb://context` | Compressed entity index (same as `kb_context()`) |
| `kb://person/{name}` | Person profile (same as `kb_person_find()`) |

## Tool Parameters

### `kb_search`

```
query: str           # search query (required)
fast: bool = True    # True = FTS only (instant), False = hybrid (FTS + vector, ~2s)
limit: int = 5       # max results (1-100)
from_date: str       # filter: YYYY-MM-DD
to_date: str         # filter: YYYY-MM-DD
tag: str             # filter by tag (comma-separated for AND)
sort_by: str         # "score" (default) or "date" (newest first)
```

### `kb_memory_add`

```
text: str            # note title or fact text (required)
body: str            # note body (markdown)
tags: str            # comma-separated tags
entity: str          # entity name (for facts or entity-linked notes)
pin: bool = False    # pin to context
date: str            # override date (YYYY-MM-DD)
```

## Error Handling

All tools return JSON. On error, the response includes an `"error"` key with a message. Errors are also logged to stderr.
