# CLI Reference

## Overview

`kb/cli.py` implements the `kb` command-line tool using Click. All commands support structured output via `--format`, `--json`, `--fields`, and `--jq` options.

## Installation

```bash
# From the project directory:
uv tool install --editable .   # installs `kb` globally (editable — code changes take effect immediately)

# Or without installing:
uv run kb search "query"       # works from project root or kb/ directory
```

## Command Structure

```
kb
├── search <query>           # Hybrid semantic + keyword search
├── view <path|#docid>       # View a specific document
├── list                     # Browse documents by date/type
├── entity
│   ├── find <name>          # Entity profile + linked documents
│   ├── timeline <name>      # Chronological docs mentioning entity
│   └── list                 # All known entities
├── index
│   ├── run [paths...]       # Incremental index (or full with --full)
│   └── status               # DB health: counts, size, freshness
└── --help                   # Init status + commands + agent playbook
```

## Commands

### search

```bash
kb search "MFA implementation"              # default hybrid, table output
kb search "MFA implementation" --fast       # FTS only, ~instant
kb search "MFA implementation" --json       # JSON output
kb search "Rust" --limit 5 --recency 0.3   # tuned
kb search "Datastore" --format csv          # CSV
kb search "meeting" --jq '.results[0].title'
kb search "Linnea" --fields title,date,score
kb search "notes" --type notes              # filter by doc_type
```

Options:
- `--fast` — FTS-only search (no vector), ~instant
- `--deep` — Placeholder (prints "not yet implemented")
- `--limit N` — Max results (default: 10)
- `--recency FLOAT` — Recency weight 0-1 (default: 0.15)
- `--type TYPE` — Filter by doc_type

### view

```bash
kb view "memory/meetings/2026/01/27/Alice___Bob.notes.md"
kb view "Alice___Bob.notes.md"               # suffix match
kb view #abcdef                              # content-hash lookup (first 6+ chars)
kb view "path.md:42"                         # line range (future)
```

Resolution order: exact path → suffix match → fuzzy suggestion.

For `#abc123`: queries `content_hash LIKE 'abc123%'`.

### list

```bash
kb list                              # all documents, newest first
kb list --type notes --limit 20      # filter by type
kb list --from 2026-01-01 --to 2026-01-31
kb list --json
```

### entity find

```bash
kb entity find "Linnea"               # profile + linked docs
kb entity find "Linnea" --json
kb entity find "thomas"              # case-insensitive
```

Matching: exact name → alias → partial substring (case-insensitive).

### entity timeline

```bash
kb entity timeline "Linnea"           # chronological mentions
kb entity timeline "Linnea" --from 2026-01-01
```

### entity list

```bash
kb entity list                       # all entities
kb entity list --type person         # filter
kb entity list --json
```

### project create / edit

```bash
kb project create "My Project" --status Active --lead "Wren Smith"
kb project create "My Project" --source "slack:channel=C123,name=#my-channel"
kb project create "My Project" --source "linear:https://linear.app/project/123"

kb project edit "My Project" --source "jira:key=GG-1234"   # append source
kb project edit "My Project" --remove-source slack          # remove all slack sources
kb project edit "My Project" --remove-source "linear:PRJ-123"  # remove specific source
```

**Sources schema:** Projects support a `sources:` list in YAML frontmatter for linking to external systems. Each entry requires a `type` key; all other keys are free-form. Source IDs (`id`, `key`, `channel`) are automatically used for entity linking via `src:`-prefixed aliases.

### index run

```bash
kb index run                         # incremental index
kb index run --full                  # full re-index
```

### index status

```bash
kb index status                      # human-readable health
kb index status --json               # structured health info
```

Reports: document count by type, total chunks, entity count, vector count, last indexed timestamp, database file sizes.

## Output Options

Available on all data commands (search, view, list, entity, index status):

| Option | Description |
|--------|-------------|
| `--format table\|json\|jsonl\|csv` | Output format (default: table) |
| `--json` | Shortcut for `--format json` |
| `--fields title,date,score` | Select specific fields |
| `--jq '.results[0].title'` | jq expression filtering |

## Database Path

The database lives at `~/.config/kbx/` by default. This can be overridden via:
1. `data.dir` in `kbx.toml` config file
2. `KB_DATA_DIR` environment variable

Project root auto-detection walks up from CWD looking for `kbx.toml`, or a directory containing both `kbx/` and `meetings/` (or `memory/`).

## Error Handling

- Unknown path in `view` → suggestion with close matches
- Missing query in `search` → Click usage error
- Entity not found → "Entity not found: name" on stderr

Exit codes: 0 = success, 1 = not found/validation, 2 = ambiguous match.

## Testing

38 tests in `test_cli.py`:
- Search: 8 tests (fast/json, table, limit, csv, fields, deep placeholder, type filter, no query)
- View: 6 tests (path, hash, suffix, not found, content field, summary exclusion)
- List: 9 tests (json, type filter, limit, date range, default table, consistent schema)
- Entity: 10 tests (list json, type filter, find json, partial match, not found, timeline, ambiguous, JSON errors)
- Index: 1 test (status json)
- Usage: 2 tests (contains "kb search", contains "Score Interpretation")
- Error handling: 2 tests (bad search, bad path)
