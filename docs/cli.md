# CLI Reference

## Overview

`kbx` (also available as `kb`) is the command-line interface, implemented with Click. All commands support structured output via `--format`, `--json`, `--fields`, and `--jq` options.

## Installation

```bash
pip install kbx                        # from PyPI
uv tool install --editable .           # editable dev install (code changes take effect immediately)
```

## Command Structure

```
kbx
├── search <query>           # Hybrid semantic + keyword search
├── view <path|#docid|glob>  # View a specific document
├── list                     # Browse documents by date/type
├── context                  # Compressed entity index for AI agents
├── me                       # Your own profile (shortcut for person find me)
├── person
│   ├── find <name>          # Person profile + linked documents
│   ├── timeline <name>      # Chronological docs mentioning person
│   ├── list                 # All known people
│   ├── create <name>        # Create a new person
│   ├── edit <name>          # Edit a person's metadata
│   ├── delete <name>        # Delete a person
│   └── pin <name>           # Pin a person to context
├── project
│   ├── find <name>          # Project profile + linked documents
│   ├── list                 # All known projects
│   ├── create <name>        # Create a new project
│   ├── edit <name>          # Edit a project's metadata
│   └── delete <name>        # Delete a project
├── entity
│   ├── stale                # Entities not mentioned recently
│   ├── unlink <name> <doc>  # Suppress a false-positive entity↔doc match
│   └── relink <name> <doc>  # Undo a suppression
├── memory
│   ├── add <text>           # Create a note or record a fact
│   ├── list                 # List facts
│   ├── delete-fact <id>     # Delete a fact
│   └── edit-fact <id>       # Edit a fact
├── note
│   ├── list                 # Browse and filter notes
│   ├── edit <target>        # Edit a note's body/tags/pin
│   └── delete <target>      # Delete a note
├── pin <path|title|#hash>   # Pin a document to context
├── unpin <path|title>       # Remove from context
├── glossary
│   ├── add <term> <text>    # Add a glossary term
│   ├── list                 # List all terms
│   ├── edit <term> <text>   # Edit a term
│   └── delete <term>        # Delete a term
├── correct <term> [repl]    # Find and replace across memory files
├── sync
│   ├── granola              # Sync meetings from Granola API
│   └── notion               # Sync meetings from Notion
├── granola
│   ├── view <uid>           # View meeting notes/transcript/summary
│   ├── edit <uid>           # Edit meeting notes
│   └── push <uid>           # Push notes to a Granola document
├── ingest [paths]           # Organise Granola exports and index
├── index
│   ├── run [paths...]       # Incremental index (or full with --full)
│   └── status               # DB health: counts, size, freshness
├── init                     # Create kbx.toml config file
├── mcp                      # Start MCP server (stdio transport)
└── --help                   # Init status + commands + agent playbook
```

## Key Commands

### search

```bash
kbx search "MFA implementation"              # default hybrid, table output
kbx search "MFA implementation" --fast       # FTS only, ~instant
kbx search "MFA implementation" --json       # JSON output
kbx search "Rust" --limit 5 --recency 0.3   # tuned
kbx search "meeting" --type notes            # filter by doc_type
kbx search "topic" --from 2026-01-01 --to 2026-01-31  # date range
kbx search "topic" --tag infra --fast        # filter by tag
kbx search "topic" --sort date               # newest first
kbx search "topic" --dedupe                  # one result per document
kbx search "topic" --full-chunks             # include full chunk text
kbx search "topic" --fields title,date,score # select fields
kbx search "topic" --jq '.results[0].title'  # jq filtering
```

Options:
- `--fast` — FTS-only search (no vector), ~instant
- `--limit N` — Max results (default: 10)
- `--recency FLOAT` — Recency weight 0–1 (default: 0.15)
- `--type TYPE` — Filter by doc_type
- `--from` / `--to` — Date range filter
- `--tag TAG` — Filter by tag (comma-separated for AND)
- `--sort score|date` — Sort order
- `--dedupe` — One result per document
- `--full-chunks` — Include full chunk text in JSON output
- `--merge-chunks` — With `--dedupe --full-chunks`, concatenate all chunks per doc
- `--snippet-chars N` — Snippet length (default: 200)
- `--fts-weight` / `--vector-weight` — Boost FTS or vector results

### view

```bash
kbx view "memory/meetings/2026/01/27/abc_Wren_Soren.notes.md"
kbx view "Wren_Soren.notes.md"              # suffix match
kbx view "#abcdef"                          # content-hash lookup
kbx view "*Wren*"                          # glob pattern
kbx view "path.md" --plain                  # raw content only
kbx view "path.md" --json                   # structured with chunks
```

Resolution order: exact path → glob → suffix match → `#hash` lookup → fuzzy suggestion.

### context

```bash
kbx context                                 # compressed entity index
kbx context --json                          # structured output
kbx context --human                         # markdown format
kbx context --for "Helix refactor"          # filtered to relevant entities
```

### person / project

```bash
kbx person find "Wren" --json              # profile + linked docs
kbx person timeline "Wren" --from 2026-01-01
kbx person create "Soren" --role "SRE Lead" --team "Platform"
kbx person edit "Soren" --role "Staff SRE" --meta "timezone=CET"
kbx person pin "Wren"                      # pin to context
kbx person list --json

kbx project find "Helix Refactor" --json
kbx project create "New Project" --status Active --lead "Wren"
kbx project create "New Project" --source "slack:channel=C123,name=#project"
kbx project edit "New Project" --source "linear:https://linear.app/..."
kbx project list --json
```

### memory / note

```bash
kbx memory add "Decision: use Postgres" --body "Rationale..." --tags decision --pin
kbx memory add "Promoted to Staff" --entity "Soren"
kbx memory list --since 30 --json

kbx note list --tag decision --json
kbx note list --pinned
kbx note edit "title" --body "new content"
kbx note edit "title" --append "extra"
kbx note edit "title" --tags "a,b,c" --pin
kbx note delete "title"
```

### correct

```bash
kbx correct "Quartz Indexer" --json             # scan: list all occurrences
kbx correct "Quartz Indexer" "Datalux"        # dry-run: preview replacements
kbx correct "Quartz Indexer" "Datalux" --apply  # execute replacements
kbx correct "Bram" --word-boundary --json  # whole-word matches only
```

### sync

```bash
kbx sync granola --since 2026-01-01        # pull meetings from Granola API
kbx sync notion --since 2026-01-01         # pull meetings from Notion
kbx sync granola --dry-run                 # preview, no writes
kbx sync granola --force                   # overwrite existing files
kbx sync granola --no-index                # skip indexing after sync
```

### granola

```bash
kbx granola view <uid>                     # view meeting notes
kbx granola view <uid> --transcript        # show transcript
kbx granola view <uid> --summary           # show AI summary
kbx granola view <uid> --all               # notes + summary + transcript
kbx granola view <uid> --plain             # raw markdown (no YAML header)
```

Note: `kbx granola push` and `kbx granola edit` were removed in 0.2.0 — the legacy write API stopped working in Granola 7.205.0+ and the public API has no write endpoint. See `docs/plugins/granola.md`.

### ingest

```bash
kbx ingest export.zip                      # organise Granola export + index
kbx ingest --dry-run                       # preview only
kbx ingest --skip-organise                 # index only
```

### index

```bash
kbx index run                               # incremental index
kbx index run --full                        # full re-index
kbx index run --no-embed                    # text-only (no model needed)
kbx index status --json                     # database health
```

## Output Options

Available on all data commands:

| Option | Description |
|--------|-------------|
| `--format table\|json\|jsonl\|csv` | Output format (default: table) |
| `--json` | Shortcut for `--format json` |
| `--fields title,date,score` | Select specific fields |
| `--jq '.results[0].title'` | jq expression filtering |

## Database Path

The database lives at the configured data directory. Resolution order:
1. `data.dir` in `kbx.toml` config file
2. `$KB_DATA_DIR` environment variable
3. `~/.config/kbx/` (default)

Project root auto-detection walks up from CWD looking for `kbx.toml`.

## Error Handling

- Unknown path in `view` → suggestion with close matches
- Entity not found → message on stderr with suggestions
- JSON mode errors include `error`, `suggestion`, and `available_actions` fields

Exit codes: 0 = success, 1 = not found/validation, 2 = ambiguous match.
