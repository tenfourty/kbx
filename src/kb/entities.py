"""Entity seeding and linking for the knowledge base."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from kb.types import Entity as Entity
from kb.types import EntityData
from kb.types import EntityMention as EntityMention

if TYPE_CHECKING:
    from kb.db import Database


def _parse_field(text: str, field_name: str) -> str | None:
    """Extract a **Field:** value from markdown text."""
    pattern = rf"\*\*{re.escape(field_name)}:\*\*\s*(.+)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def _parse_all_fields(text: str) -> dict[str, str]:
    """Extract all **Key:** Value lines from the header block of markdown text.

    Returns a dict mapping snake_case keys to their values.
    Only parses lines before the first ## heading (the header block).
    """
    # Only look at the header (before first ## heading)
    header = text
    match = re.search(r"^## ", text, re.MULTILINE)
    if match:
        header = text[: match.start()]

    fields: dict[str, str] = {}
    for m in re.finditer(r"\*\*(.+?):\*\*\s*(.+)", header):
        label = m.group(1).strip()
        value = m.group(2).strip()
        # Convert Title Case label to snake_case key
        key = "_".join(label.lower().split())
        fields[key] = value
    return fields


def _parse_title(text: str) -> str | None:
    """Extract the # Title from markdown."""
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _stem_to_first_name(name: str) -> str | None:
    """Extract first name from a full name."""
    parts = name.split()
    if len(parts) >= 2:
        return parts[0]
    return None


# Fields that are handled specially (not metadata) for person files
_PERSON_SPECIAL_FIELDS = frozenset({"also_known_as", "pinned"})

# Fields that are handled specially (not metadata) for project files
_PROJECT_SPECIAL_FIELDS = frozenset({"codename/also_called", "pinned"})


def _parse_person_file(path: Path) -> EntityData:
    """Parse a memory/people/*.md file into entity data.

    Picks up all **Key:** Value lines in the header block, not just hardcoded
    fields. Special fields (Also known as, Pinned) are handled separately.
    """
    text = path.read_text(encoding="utf-8")
    name = _parse_title(text) or path.stem.replace("-", " ").title()

    also_known = _parse_field(text, "Also known as")

    aliases: list[str] = []
    if also_known:
        aliases.extend(a.strip() for a in also_known.split(","))

    first_name = _stem_to_first_name(name)
    if first_name and first_name not in aliases:
        aliases.append(first_name)

    file_stem = path.stem
    if file_stem not in aliases:
        aliases.append(file_stem)

    pinned_str = _parse_field(text, "Pinned")
    is_pinned = bool(pinned_str and pinned_str.lower() in ("true", "yes", "1"))

    # Parse all **Key:** Value fields — includes both known and custom fields
    all_fields = _parse_all_fields(text)
    metadata: dict[str, str] = {
        k: v for k, v in all_fields.items() if k not in _PERSON_SPECIAL_FIELDS
    }

    return EntityData(
        name=name,
        entity_type="person",
        aliases=aliases,
        metadata=metadata,
        source_path=str(Path("memory/people") / path.name),
        pinned=is_pinned,
    )


def _parse_project_file(path: Path) -> EntityData:
    """Parse a memory/projects/*.md file into entity data.

    Picks up all **Key:** Value lines in the header block, not just hardcoded
    fields. Special fields (Codename/Also called, Pinned) are handled separately.
    """
    text = path.read_text(encoding="utf-8")
    name = _parse_title(text) or path.stem.replace("-", " ").title()

    codename = _parse_field(text, "Codename/Also called")

    pinned_str = _parse_field(text, "Pinned")
    is_pinned = bool(pinned_str and pinned_str.lower() in ("true", "yes", "1"))

    aliases: list[str] = []
    if codename:
        aliases.extend(a.strip() for a in codename.split(","))

    file_stem = path.stem
    if file_stem not in aliases:
        aliases.append(file_stem)

    # Parse all **Key:** Value fields — includes both known and custom fields
    all_fields = _parse_all_fields(text)
    metadata: dict[str, str] = {
        k: v for k, v in all_fields.items() if k not in _PROJECT_SPECIAL_FIELDS
    }

    return EntityData(
        name=name,
        entity_type="project",
        aliases=aliases,
        metadata=metadata,
        source_path=str(Path("memory/projects") / path.name),
        pinned=is_pinned,
    )


def _parse_teams_from_company(path: Path) -> list[EntityData]:
    """Parse teams from memory/context/company.md."""
    text = path.read_text(encoding="utf-8")
    teams: list[EntityData] = []

    # Find the "Teams (from Linear)" section
    in_teams = False
    for line in text.splitlines():
        if "Teams (from Linear)" in line:
            in_teams = True
            continue
        if in_teams:
            if line.startswith("##"):
                break
            m = re.match(r"^-\s+\*\*(.+?)\*\*(?:\s+\((\w+)\))?\s*—\s*(.+)", line)
            if m:
                team_name = m.group(1).strip()
                abbreviation = m.group(2)
                description = m.group(3).strip()

                aliases: list[str] = []
                if abbreviation:
                    aliases.append(abbreviation)

                teams.append(
                    EntityData(
                        name=team_name,
                        entity_type="team",
                        aliases=aliases,
                        metadata={"description": description},
                        source_path="memory/context/company.md",
                    )
                )

    return teams


def _get_claude_md_alias_map(path: Path) -> dict[str, str]:
    """Build a mapping from CLAUDE.md short names to full names.

    Returns e.g. {"Kit K.": "Kit Larsen", "Talia": "Talia Ström", ...}
    """
    text = path.read_text(encoding="utf-8")
    alias_map = {}

    in_people = False
    for line in text.splitlines():
        if "| Who | Role |" in line:
            in_people = True
            continue
        if in_people:
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                break
            m = re.match(r"\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|", line)
            if m:
                short_name = m.group(1).strip()
                desc = m.group(2).strip()

                # Extract full name
                full_name_match = re.match(
                    r"([A-ZÀ-ÿ][\w\-'À-ÿ]+(?:\s+[A-ZÀ-ÿ][\w\-'À-ÿ]+)+)", desc
                )
                if full_name_match:
                    full_name = full_name_match.group(1)
                    alias_map[short_name] = full_name

    return alias_map


def _add_claude_md_extra_people(
    claude_md_path: Path,
    existing_names: set[str],
) -> list[EntityData]:
    """Add people from CLAUDE.md who don't exist in memory/people/."""
    alias_map = _get_claude_md_alias_map(claude_md_path)
    extra: list[EntityData] = []

    for short_name, full_name in alias_map.items():
        if full_name not in existing_names:
            first_name = _stem_to_first_name(full_name)
            aliases = [short_name]
            if first_name and first_name != short_name and first_name not in aliases:
                aliases.append(first_name)

            extra.append(
                EntityData(
                    name=full_name,
                    entity_type="person",
                    aliases=aliases,
                    metadata={},
                    source_path="CLAUDE.md",
                )
            )
            existing_names.add(full_name)

    return extra


def _merge_claude_md_aliases(
    entities: list[EntityData],
    claude_md_path: Path,
) -> list[EntityData]:
    """Merge additional aliases from CLAUDE.md into existing person entities.

    Returns a new list with updated EntityData objects (EntityData is frozen).
    """
    alias_map = _get_claude_md_alias_map(claude_md_path)

    # Build reverse map: full_name -> short_name(s)
    name_to_shorts: dict[str, list[str]] = {}
    for short, full in alias_map.items():
        name_to_shorts.setdefault(full, []).append(short)

    result: list[EntityData] = []
    for entity in entities:
        if entity.entity_type != "person":
            result.append(entity)
            continue
        shorts = name_to_shorts.get(entity.name, [])
        new_aliases = list(entity.aliases)
        changed = False
        for short in shorts:
            if short not in new_aliases:
                new_aliases.append(short)
                changed = True
        if changed:
            result.append(entity.model_copy(update={"aliases": new_aliases}))
        else:
            result.append(entity)
    return result


def seed_entities(db: Database, project_root: Path) -> int:
    """Seed entities from memory files and CLAUDE.md.

    Non-destructive upsert: updates existing entities in place (preserving IDs
    and entity_mentions), inserts new ones, removes stale ones.
    Returns the number of entities after seeding.
    """
    conn = db.get_sqlite_conn()

    all_entities: list[EntityData] = []

    # Source A: People from memory/people/
    people_dir = project_root / "memory" / "people"
    if people_dir.exists():
        for f in sorted(people_dir.glob("*.md")):
            all_entities.append(_parse_person_file(f))

    # Source B: Projects from memory/projects/
    projects_dir = project_root / "memory" / "projects"
    if projects_dir.exists():
        for f in sorted(projects_dir.glob("*.md")):
            all_entities.append(_parse_project_file(f))

    # Source C: Company entity
    all_entities.append(
        EntityData(
            name="Lattice Co",
            entity_type="company",
            aliases=["AC"],
            metadata={},
            source_path="memory/context/company.md",
        )
    )

    # Source D: Teams from company.md
    company_md = project_root / "memory" / "context" / "company.md"
    if company_md.exists():
        all_entities.extend(_parse_teams_from_company(company_md))

    # Cross-reference CLAUDE.md for extra aliases on existing people
    claude_md = project_root / "CLAUDE.md"
    if claude_md.exists():
        all_entities = _merge_claude_md_aliases(all_entities, claude_md)

        # Add people from CLAUDE.md not in memory/people/
        existing_names = {e.name for e in all_entities if e.entity_type == "person"}
        extra_people = _add_claude_md_extra_people(claude_md, existing_names)
        all_entities.extend(extra_people)

    # Load existing entities: name -> id
    existing = {
        row["name"]: row["id"] for row in conn.execute("SELECT id, name FROM entities").fetchall()
    }

    seen_names: set[str] = set()
    for entity in all_entities:
        name = entity.name
        seen_names.add(name)
        aliases_json = json.dumps(entity.aliases)
        metadata_json = json.dumps(entity.metadata)
        source_path = entity.source_path
        source_pinned = entity.pinned

        if name in existing:
            # Update in place — keeps ID stable, preserves entity_mentions
            # Only set pinned=1 from source; never clear a dynamic pin
            if source_pinned:
                conn.execute(
                    "UPDATE entities SET entity_type=?, aliases=?, metadata=?, source_path=?, pinned=1 WHERE id=?",
                    (
                        entity.entity_type,
                        aliases_json,
                        metadata_json,
                        source_path,
                        existing[name],
                    ),
                )
            else:
                conn.execute(
                    "UPDATE entities SET entity_type=?, aliases=?, metadata=?, source_path=? WHERE id=?",
                    (
                        entity.entity_type,
                        aliases_json,
                        metadata_json,
                        source_path,
                        existing[name],
                    ),
                )
        else:
            pinned_val = 1 if source_pinned else 0
            conn.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
                (name, entity.entity_type, aliases_json, metadata_json, source_path, pinned_val),
            )

    # Remove entities no longer in source files (cascade deletes their mentions)
    for name, eid in existing.items():
        if name not in seen_names:
            conn.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (eid,))
            conn.execute("DELETE FROM entities WHERE id = ?", (eid,))

    conn.commit()

    return len(all_entities)


def load_entities(db: Database) -> list[Entity]:
    """Load all entities from the database."""
    conn = db.get_sqlite_conn()
    rows = conn.execute(
        "SELECT id, name, entity_type, aliases, metadata, source_path, pinned FROM entities"
    ).fetchall()

    entities = []
    for row in rows:
        aliases = json.loads(row["aliases"]) if row["aliases"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        entities.append(
            Entity(
                id=row["id"],
                name=row["name"],
                entity_type=row["entity_type"],
                aliases=aliases,
                metadata=metadata,
                source_path=row["source_path"],
                pinned=bool(row["pinned"]),
            )
        )
    return entities


def load_entity_by_id(db: Database, entity_id: int) -> Entity | None:
    """Load a single entity by its database ID."""
    conn = db.get_sqlite_conn()
    row = conn.execute(
        "SELECT id, name, entity_type, aliases, metadata, source_path, pinned FROM entities WHERE id = ?",
        (entity_id,),
    ).fetchone()
    if not row:
        return None
    return Entity(
        id=row["id"],
        name=row["name"],
        entity_type=row["entity_type"],
        aliases=json.loads(row["aliases"]) if row["aliases"] else [],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        source_path=row["source_path"],
        pinned=bool(row["pinned"]),
    )


# Common English words that happen to be entity names — require capitalized form
_COMMON_WORDS = frozenset(
    {
        "product",
        "data",
        "design",
        "staff",
        "security",
        "core",
        "sources",
        "quality",
        "applied",
        "front",
        "incidents",
        "machine",
    }
)


def _boundary_pattern(escaped: str, name: str, flags: int = 0) -> re.Pattern[str]:
    """Build a word-boundary regex that handles names ending with non-word chars.

    Standard \\b doesn't work after non-word chars like periods (e.g. "Anders M.").
    Uses lookahead for the trailing boundary when the name ends with a non-word char.
    """
    # Trailing boundary: use lookahead if name ends with non-word char
    if name[-1].isalnum() or name[-1] == "_":
        return re.compile(rf"\b{escaped}\b", flags)
    return re.compile(rf"\b{escaped}(?=\s|$|[^\w])", flags)


def _build_name_patterns(entity: Entity) -> list[tuple[re.Pattern[str], int]]:
    """Build regex patterns for matching entity names/aliases in content.

    Returns (pattern, priority) tuples. Higher priority = longer match = preferred.

    Matching rules:
    - Full names (2+ words): case-insensitive word-boundary match
    - Short uppercase abbreviations (<=4 chars, all caps): case-sensitive
    - Single common English words (Product, Data, etc.): case-sensitive
    - Very short single names (<=3 chars, 1 word): skipped for content matching
      (still matched via tag/title matching which is separate)
    - File-stem aliases (contain hyphens, all lowercase): skipped
    """
    patterns = []
    all_names = [entity.name, *entity.aliases]

    for name in all_names:
        # Skip very short names (< 2 chars) and file stems with hyphens
        if len(name) < 2:
            continue
        if "-" in name and name == name.lower():
            continue

        escaped = re.escape(name)
        word_count = len(name.split())

        # Short uppercase abbreviations (IT, GG, SRC, SEC) — case-sensitive
        if name.isupper() and len(name) <= 4:
            patterns.append((_boundary_pattern(escaped, name), len(name)))
            continue

        # Multi-word names — case-insensitive (e.g. "Soren Vance", "Helix Refactor")
        if word_count >= 2:
            patterns.append((_boundary_pattern(escaped, name, re.IGNORECASE), len(name)))
            continue

        # Single-word names below here
        # Skip very short first-name aliases for content matching (too ambiguous)
        # e.g. "Ed", "Jo", "Al" — these match via tag/title only
        if word_count == 1 and len(name) <= 3 and name[0].isupper() and name[1:].islower():
            continue

        # Common English words used as team names — case-sensitive only
        if name.lower() in _COMMON_WORDS:
            patterns.append((_boundary_pattern(escaped, name), len(name)))
            continue

        # Everything else — case-insensitive
        patterns.append((_boundary_pattern(escaped, name, re.IGNORECASE), len(name)))

    return patterns


def build_entity_patterns(
    entities: list[Entity],
) -> list[tuple[re.Pattern[str], Entity]]:
    """Pre-compile content-matching patterns for all entities.

    Call once per entity set, then pass the result to find_entity_mentions()
    to avoid recompiling ~1000 regex patterns per document.
    Returns patterns sorted by priority (longest match first).
    """
    all_patterns: list[tuple[re.Pattern[str], int, Entity]] = []
    for entity in entities:
        for pattern, priority in _build_name_patterns(entity):
            all_patterns.append((pattern, priority, entity))

    # Sort by priority descending (longer matches first)
    all_patterns.sort(key=lambda x: x[1], reverse=True)

    return [(p, e) for p, _pri, e in all_patterns]


def find_entity_mentions(
    title: str,
    tags: list[str],
    content: str,
    entities: list[Entity],
    *,
    cached_patterns: list[tuple[re.Pattern[str], Entity]] | None = None,
) -> list[EntityMention]:
    """Find entity mentions in a document's metadata and content.

    Matching rules:
    1. Tags → match tag to entity name/aliases (mention_type = "tagged")
    2. Title parsing → split on separators, match participants (mention_type = "participant")
    3. Title substring → word-boundary match names/aliases in title (mention_type = "title")
    4. Content name matching → scan for known names (mention_type = "discussed")

    Pass cached_patterns (from build_entity_patterns()) to avoid recompiling
    regex patterns on every call. If None, patterns are built on the fly.

    Disambiguation: prefer longer alias matches. If ambiguous (e.g. "Anders"),
    match all possible entities.
    """
    mentions: list[EntityMention] = []
    seen: set[tuple[int, str]] = set()

    def _add(entity_id: int, mention_type: str) -> None:
        key = (entity_id, mention_type)
        if key not in seen:
            seen.add(key)
            mentions.append(EntityMention(entity_id=entity_id, mention_type=mention_type))

    # 1. Tag matching
    for tag in tags:
        tag_lower = tag.lower().strip()
        if not tag_lower:
            continue
        for entity in entities:
            all_names = [entity.name.lower()] + [a.lower() for a in entity.aliases]
            if tag_lower in all_names:
                _add(entity.id, "tagged")
                continue
            # Partial match: tag is a first name or short form
            for name in all_names:
                if tag_lower == name:
                    _add(entity.id, "tagged")
                    break

    # 2. Title participant parsing
    # Split on common separators: " / ", " x ", " & ", " vs "
    parts = re.split(r"\s+/\s+|\s+x\s+|\s+&\s+|\s+vs\s+", title, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part_lower = part.lower()
        for entity in entities:
            all_names = [entity.name.lower()] + [a.lower() for a in entity.aliases]
            if part_lower in all_names or any(
                re.search(rf"\b{re.escape(n)}\b", part_lower)
                for n in all_names
                if len(n) >= 4  # skip very short names for substring matching
            ):
                _add(entity.id, "participant")

    # 3. Title substring matching — catch names embedded in the title
    # e.g. "Anders Sync Notes", "Wren 1:1", "Helix Refactor Review"
    title_lower = title.lower()
    for entity in entities:
        all_names = [entity.name, *list(entity.aliases)]
        for name in all_names:
            # Skip very short names and file-stem aliases
            if len(name) <= 3:
                continue
            if "-" in name and name == name.lower():
                continue
            if re.search(rf"\b{re.escape(name)}\b", title_lower, re.IGNORECASE):
                _add(entity.id, "title")
                break  # one match per entity is enough

    # 4. Content name matching with disambiguation
    if cached_patterns is None:
        cached_patterns = build_entity_patterns(entities)

    for pattern, entity in cached_patterns:
        if pattern.search(content):
            _add(entity.id, "discussed")

    return mentions
