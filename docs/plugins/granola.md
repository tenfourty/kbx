# Granola Plugin

Granola is a meeting transcript tool that records meetings and generates AI notes. The Granola plugin syncs meetings from the Granola API into `memory/meetings/` and indexes them into kb.

## How Sync Works

1. Authenticates via WorkOS tokens stored in Granola's local `supabase.json`
2. Lists all documents from the Granola API (paginated, 50 per request)
3. Filters by `--since` date (client-side; the API has no server-side filter)
4. For each document, fetches the transcript via a separate API call
5. Converts ProseMirror notes to Markdown, merges transcript segments
6. Builds YAML frontmatter (title, date, attendees, calendar event, tags)
7. Writes `.notes.md` and `.transcript.md` files to `memory/meetings/YYYY/MM/DD/`
8. Tracks sync state in `.granola_sync_state.json` for incremental runs

Existing files are skipped unless the remote `updated_at` is newer. The `--force` flag overwrites regardless.

Attendees are automatically matched to existing entities by email or name. Unmatched attendees get a new entity and `memory/people/{slug}.md` file.

## Configuration

| Variable | Purpose |
|----------|---------|
| `GRANOLA_WORKOS_CLIENT_ID` | WorkOS client ID for token refresh |

Token path defaults to `~/Library/Application Support/Granola/supabase.json` (macOS). Override with `--token-path`.

## CLI Commands

### `kbx sync granola`

Pull meetings from the Granola API.

```
kbx sync granola --since 2026-01-01      # sync since date
kbx sync granola --dry-run               # preview, no writes
kbx sync granola --force                 # overwrite existing files
kbx sync granola --no-index              # skip indexing after sync
kbx sync granola --token-path /path/to/supabase.json
```

Without `--since`, uses the last sync timestamp from state file (incremental).

### `kbx ingest <paths>`

Organise Granola zip exports and index them.

```
kbx ingest meetings/export.zip           # organise + index
kbx ingest --dry-run                     # preview only
kbx ingest --skip-organise               # index only (already organised)
```

This runs `meetings/organise_granola.py` to unpack exports into `memory/meetings/`, then indexes them.

### `kbx granola view <calendar-uid>`

View synced meeting notes, AI summaries, or transcripts.

```
kbx granola view <uid>                   # view notes (default)
kbx granola view <uid> --summary         # show AI summary
kbx granola view <uid> --transcript      # show transcript
kbx granola view <uid> --all             # show notes + summary + transcript
kbx granola view <uid> --plain           # raw markdown (no YAML header)
```

### `kbx granola edit <calendar-uid>`

Edit or append to meeting notes.

```
kbx granola edit <uid> --body "markdown"       # full replace
kbx granola edit <uid> --body-file notes.md    # replace from file
kbx granola edit <uid> --append "extra notes"  # append to existing
```

### `kbx granola push <calendar-uid>`

Push prep notes back to Granola (writes to the document's notes field).

```
kbx granola push <uid> --notes "markdown"       # content to write
kbx granola push <uid> --notes-file prep.md     # write from file (strips YAML frontmatter)
kbx granola push <uid> --title "Meeting Title"  # title (for new docs)
```

Push is idempotent — skips if the first line is already present in existing notes.

## Rate Limiting

The sync processes documents in batches of 15 with a 0.5s delay between batches to avoid API rate limits.
