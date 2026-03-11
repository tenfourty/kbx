# Context Layer

## `kbx context` — Compressed Entity Index

Generates a ~30-line compressed index of all entities (people, projects, teams) and key terms. Designed to give AI agents enough context to know everyone and everything by name without consuming the full instruction budget.

### Usage

```bash
kbx context                           # full compressed index (plain text)
kbx context --json                    # structured JSON output
kbx context --for "Rust migration"    # filtered to relevant entities only
```

### Plain text output format

```
# Knowledge Base Context
|67 entities, 3096 documents, 2024-06 to 2026-02

## People (27)
|Linnea(Platform Lead) Kit(Head of Product) Talia(Helix Refactor Lead)
→ Details: `kbx entity find "name" --json`

## Projects (12)
|Helix Refactor(Talia,In Progress) Beacon CLI(Anders,Planning)
→ Details: `kbx entity find "project" --json`

## Teams (8)
|Platform(PLAT) Engine(ENG) Security(SEC)

## Terms
|GIM=Internal Monorepo MR=Merge Request
→ Full glossary: `kbx view memory/glossary.md`
```

### JSON output

```json
{
  "text": "# Knowledge Base Context\n...",
  "stats": {
    "documents": 3096,
    "entities": 67,
    "date_range": {"earliest": "2024-06-01", "latest": "2026-02-14"}
  },
  "entities": [
    {"name": "Linnea Nguyen", "entity_type": "person", "mention_count": 45},
    {"name": "Helix Refactor", "entity_type": "project", "mention_count": 32}
  ]
}
```

### Topic filtering (`--for`)

When a topic is provided, the command:
1. Runs an FTS search for the topic
2. Collects entity IDs mentioned in matching documents
3. Filters the output to only those entities
4. Adjusts the header: `# Context: Rust migration`

This is useful for scoping context to a specific conversation topic.

## `kbx memory add` — Fact Recording

Records facts about entities, appending to both the SQLite database and the entity's markdown source file.

### Usage

```bash
kbx memory add "Promoted to Staff Engineer" --entity "Soren" --date 2026-01-19
kbx memory add "Led Platform TF kickoff" --entity "Linnea"
```

### How it works

1. Finds the entity by name (case-insensitive, partial match)
2. Inserts a row into the `facts` SQLite table
3. If the entity has a `source_path` (e.g. `memory/people/bob-chen.md`), appends the fact under a `## Recent Facts` section in that file

### File format

Facts are appended as markdown list items:

```markdown
## Recent Facts
- [2026-01-19] Promoted to Staff Engineer
- [2026-02-01] Led Platform TF kickoff
```

## `kbx memory list` — List Facts

```bash
kbx memory list                     # all facts
kbx memory list --since 30          # last 30 days
kbx memory list --json              # structured output
```

### Schema

The `facts` table is created via migration `002_create_facts_table`:

```sql
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER REFERENCES entities(id),
    fact_text TEXT NOT NULL,
    fact_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
```

## How Context Works

`kbx context` dynamically generates a compressed entity index from the indexed database:

- **Pinned people**: Sorted by recency with freshness indicators
- **Key people**: Unpinned entities with significant metadata or facts, recency-weighted
- **Projects**: Active projects with lead and status (excludes completed)
- **Teams**: Only teams with mentions
- **Terms**: Parsed from `memory/glossary.md`
- **Standard industry terms** (CI, CD, SaaS, API, etc.) are filtered out to reduce noise

The memory files (`memory/people/`, `memory/projects/`, etc.) remain the source of truth for entity seeding. `kbx context` reads from the indexed database, not directly from files.
