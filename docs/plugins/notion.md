# Notion Plugin

Notion AI Meeting Notes sync pulls meeting transcripts and AI-generated notes from Notion into `memory/meetings/`.

## How Sync Works

1. Authenticates via Notion API token (environment variable or config)
2. Lists meeting pages from Notion (paginated, batches of 10)
3. Filters by `--since` date
4. Fetches transcript and notes content for each meeting
5. Builds YAML frontmatter (title, date, attendees, calendar event, tags)
6. Writes `.notion.notes.md` and `.notion.transcript.md` files to `memory/meetings/YYYY/MM/DD/`
7. Tracks sync state for incremental runs

Existing files are skipped unless the remote content is newer. The `--force` flag overwrites regardless.

## CLI Commands

### `kbx sync notion`

Pull meetings from Notion AI Meeting Notes.

```
kbx sync notion --since 2026-01-01      # sync since date
kbx sync notion --dry-run               # preview, no writes
kbx sync notion --force                 # overwrite existing files
kbx sync notion --no-index              # skip indexing after sync
```

Without `--since`, uses the last sync timestamp (incremental).

## Rate Limiting

Notion sync uses stricter rate limiting than Granola:

- Batch size: 10 documents per batch
- Batch delay: 0.3s between batches
- Per-request throttle: 0.3s minimum between API calls
- Automatic retry on 429 (rate limit) with exponential backoff (10s, 20s, 40s, 80s, 160s)

## Output Files

Each meeting produces two files:

| Suffix | Content |
|--------|---------|
| `.notion.notes.md` | AI summary (preferred) or manual notes |
| `.notion.transcript.md` | Full meeting transcript |

Files follow the standard naming convention: `{uid_prefix}_{Title}.notion.notes.md`
