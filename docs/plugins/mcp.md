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

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kbx": {
      "command": "/Users/YOU/.local/bin/kbx",
      "args": ["mcp"]
    }
  }
}
```

> **Important:** Use the full path to the `kbx` binary. Claude Desktop launches with a minimal PATH that does not include `~/.local/bin/`. Find the path with `which kbx`.
>
> Typical locations:
> - **macOS / Linux** (`uv tool install`): `~/.local/bin/kbx`
> - **Windows** (`uv tool install`): `%APPDATA%\uv\tools\kbx\bin\kbx.exe`
> - **pip install**: check `which kbx` or `where kbx`

### Claude Code

`.claude/settings.local.json`:

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

Claude Code inherits your shell PATH, so the short name works here.

## Available Tools

| Tool | Description |
|------|-------------|
| `kb_search` | Search the knowledge base. Supports `fast` (FTS-only, default) and hybrid (FTS + vector). Filter by date range, tag, sort by score or date. |
| `kb_person_find` | Look up a person by name, alias, or partial match. Returns profile with facts, metadata, and breadcrumbs. |
| `kb_person_list` | List all known people with metadata. |
| `kb_person_timeline` | Chronological list of documents mentioning a person. Optional date range filter. |
| `kb_project_find` | Look up a project by name, alias, or partial match. Returns profile with facts, metadata, and breadcrumbs. |
| `kb_project_list` | List all known projects with metadata. |
| `kb_view` | View a document by path, glob, or content-hash prefix (`#abc123`). Returns metadata and all chunks. |
| `kb_list` | Browse documents by date and type. Filter by `doc_type`, `from_date`, `to_date`. Default limit 25. |
| `kb_context` | Compressed entity index — overview of people, projects, teams, terms. Supports `compact` (pipe-delimited) and `human` (markdown) formats. Optional topic filter. |
| `kb_usage` | Usage instructions and current index status (doc/entity/fact counts, date range). |
| `kb_index_status` | Database health: document counts by type, entity/fact counts, date range, last indexed timestamp. |
| `kb_memory_add` | Create a searchable note or record a fact about an entity. Supports tags, pinning, and entity linking. |
| `kb_memory_list` | List recorded facts, newest first. Optional `since_days` filter. |
| `kb_memory_delete_fact` | Delete a fact by ID. |
| `kb_memory_edit_fact` | Edit a fact's text or date. |
| `kb_note_list` | Browse memory notes. Filter by tag (AND), pinned_only. Default limit 25. |
| `kb_note_edit` | Edit a note's body, tags, or pin status. Supports body replacement, append, tag update, pin/unpin. |
| `kb_note_delete` | Delete a memory note (file + index entry). |
| `kb_pin` | Pin a document so it appears in `kb_context` output. |
| `kb_unpin` | Remove a document from context. |
| `kb_entity_stale` | List entities not mentioned in recent documents. Filter by days threshold. |
| `kb_glossary_list` | List all glossary terms (acronyms and jargon). |
| `kb_glossary_add` | Add a term to the glossary. Specify section (default: Acronyms). |
| `kb_glossary_edit` | Update an existing glossary term's expansion. |
| `kb_granola_view` | View meeting notes, AI summary, or transcript from Granola by calendar UID. |
| `kb_granola_edit` | Edit meeting notes in Granola (API). Replace body or append. |
| `kb_correct` | Find (and optionally replace) a term across memory files. Scan, dry-run, or apply modes. |

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

### `kb_list`

```
doc_type: str        # filter: "notes", "transcript", "memory_person", etc.
from_date: str       # filter: YYYY-MM-DD
to_date: str         # filter: YYYY-MM-DD
limit: int = 25      # max results (1-100)
```

### `kb_note_list`

```
tag: str             # comma-separated tags (AND filter)
pinned_only: bool    # only return pinned notes
limit: int = 25      # max results (1-100)
```

### `kb_note_edit`

```
target: str          # note path, title, or glob (required)
body: str            # replace note body entirely
append: str          # append to existing body
tags: str            # comma-separated tags (replaces existing)
pin: bool            # True to pin, False to unpin
```

### `kb_note_delete`

```
target: str          # note path, title, or glob (required)
```

### `kb_memory_list`

```
since_days: int      # only show facts from last N days
```

### `kb_memory_edit_fact`

```
fact_id: int         # fact ID (required)
text: str            # new fact text
date: str            # new date (YYYY-MM-DD)
```

### `kb_glossary_add`

```
term: str            # abbreviation or term (required)
expansion: str       # what it stands for (required)
section: str         # glossary section heading (default: "Acronyms")
```

### `kb_granola_view`

```
calendar_uid: str    # Google Calendar event ID or iCalUID (required)
mode: str            # "summary", "transcript", "all", or None (notes only)
```

### `kb_granola_edit`

```
calendar_uid: str    # Google Calendar event ID or iCalUID (required)
body: str            # replace notes entirely
append: str          # append to existing notes
```

### `kb_correct`

```
term: str            # term to find (required)
replacement: str     # replacement text (omit for scan-only)
apply: bool = False  # True to execute replacements, False for dry-run
scope: str           # glob pattern to limit search (e.g. "**/meetings/*")
file_type: str       # filter by file type
word_boundary: bool  # only match whole words
ignore_case: bool    # case-insensitive matching
```

### `kb_entity_stale`

```
days: int = 30       # threshold in days
entity_type: str     # "person" or "project"
```

## Error Handling

All tools return JSON. On error, the response includes an `"error"` key with a message. Errors are also logged to stderr.

## Troubleshooting

### Server fails to start / "No such file or directory"

Claude Desktop (and other GUI tools) launch with a minimal system PATH that does not include directories like `~/.local/bin/`. If the log shows:

```
Failed to spawn process: No such file or directory
```

**Fix:** Use the full absolute path to `kbx` in your config:

```bash
# Find the full path
which kbx
# e.g. /Users/you/.local/bin/kbx
```

Then update `claude_desktop_config.json` to use that path as the `"command"` value. See the [Usage](#usage) section above for the full config example.

**Claude Desktop log location:** `~/Library/Logs/Claude/mcp-server-kbx.log`
