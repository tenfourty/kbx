# Granola Plugin

Granola is a meeting transcript tool that records meetings and generates AI notes. The Granola plugin syncs meetings from the Granola API into `meetings/organised/` and indexes them into kb.

## How Sync Works

1. Authenticates via WorkOS tokens stored in Granola's local `supabase.json`
2. Lists all documents from the Granola API (paginated, 50 per request)
3. Filters by `--since` date (client-side; the API has no server-side filter)
4. For each document, fetches the transcript via a separate API call
5. Converts ProseMirror notes to Markdown, merges transcript segments
6. Builds YAML frontmatter (title, date, attendees, calendar event, tags)
7. Writes `.notes.md` and `.transcript.md` files to `meetings/organised/YYYY/MM/DD/`
8. Tracks sync state in `.granola_sync_state.json` for incremental runs

Existing files are skipped unless the remote `updated_at` is newer. The `--force` flag overwrites regardless.

Attendees are automatically matched to existing entities by email or name. Unmatched attendees get a new entity and `memory/people/{slug}.md` file.

## Configuration

| Variable | Purpose |
|----------|---------|
| `GRANOLA_WORKOS_CLIENT_ID` | WorkOS client ID for token refresh |

Token path defaults to `~/Library/Application Support/Granola/supabase.json` (macOS). Override with `--token-path`.

## CLI Commands

### `kb sync granola`

Pull meetings from the Granola API.

```
kb sync granola --since 2026-01-01      # sync since date
kb sync granola --dry-run               # preview, no writes
kb sync granola --force                 # overwrite existing files
kb sync granola --no-index              # skip indexing after sync
kb sync granola --token-path /path/to/supabase.json
```

Without `--since`, uses the last sync timestamp from state file (incremental).

### `kb ingest <paths>`

Organise Granola zip exports and index them.

```
kb ingest meetings/export.zip           # organise + index
kb ingest --dry-run                     # preview only
kb ingest --skip-organise               # index only (already organised)
```

This runs `meetings/organise_granola.py` to unpack exports into `meetings/organised/`, then indexes them.

## Rate Limiting

The sync processes documents in batches of 15 with a 0.5s delay between batches to avoid API rate limits.
