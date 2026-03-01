"""Generic CRUD engine for kb entities."""

from __future__ import annotations

import json as _json
import re
import unicodedata
from typing import TYPE_CHECKING, Any

import yaml

from kb.entities import seed_entities
from kb.types import EntityTypeConfig
from kb.writeback import _write_atomic

if TYPE_CHECKING:
    from pathlib import Path

    from kb.db import Database


def _snake_to_title(key: str) -> str:
    """Convert snake_case metadata key to Title Case label.

    Examples: 'preferred_lang' -> 'Preferred Lang', 'timezone' -> 'Timezone'
    """
    return " ".join(word.capitalize() for word in key.split("_"))


def _title_to_snake(label: str) -> str:
    """Convert Title Case label to snake_case metadata key.

    Examples: 'Preferred Lang' -> 'preferred_lang', 'Reports to' -> 'reports_to'
    """
    return "_".join(label.lower().split())


class EntityExistsError(Exception):
    """Raised when creating an entity that already exists."""


class EntityNotFoundError(Exception):
    """Raised when an entity is not found."""


# ---------------------------------------------------------------------------
# Type registry: maps entity_type -> (directory, field_order, alias_field_name)
# ---------------------------------------------------------------------------

_TYPE_REGISTRY: dict[str, EntityTypeConfig] = {
    "person": EntityTypeConfig(
        directory="memory/people",
        alias_label="Also known as",
        fields=[
            ("Email", "email"),
            ("Role", "role"),
            ("Team", "team"),
            ("Reports to", "reports_to"),
            ("Company", "company"),
        ],
    ),
    "project": EntityTypeConfig(
        directory="memory/projects",
        alias_label="Codename/Also called",
        fields=[
            ("Status", "status"),
            ("Started", "started"),
            ("Lead", "lead"),
        ],
    ),
}


def get_type_registry() -> dict[str, EntityTypeConfig]:
    """Return a copy of the type registry for introspection."""
    return dict(_TYPE_REGISTRY)


_ENTITY_REF_FIELDS = frozenset({"team", "reports_to", "lead", "company"})


def _name_to_slug(name: str) -> str:
    """Convert 'Jane Doe' to 'jane-doe' for filenames.

    Strips accents via NFKD decomposition so filenames are ASCII-only.
    Entity names in the DB and markdown headings keep their original accents.
    """
    # Strip accents: NFKD decompose, then drop combining marks
    slug = unicodedata.normalize("NFKD", name)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = slug.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)  # remove non-word chars except hyphens
    slug = re.sub(r"[\s]+", "-", slug)  # spaces to hyphens
    return slug


def _build_markdown(
    name: str,
    entity_type: str,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> str:
    """Build a markdown file for an entity with YAML frontmatter."""
    metadata = metadata or {}
    aliases = aliases or []
    reg = _TYPE_REGISTRY[entity_type]

    fm: dict[str, object] = {}
    if aliases:
        fm["aliases"] = aliases

    registered_keys = {key for _label, key in reg.fields}
    for _label, key in reg.fields:
        value = metadata.get(key)
        if value:
            if key in _ENTITY_REF_FIELDS:
                fm[key] = f"[[{value}]]" if "[[" not in value else value
            else:
                fm[key] = value

    for key, value in metadata.items():
        if key not in registered_keys and value:
            fm[key] = value

    if not fm:
        return f"# {name}\n"

    yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n# {name}\n"


def create_entity(
    db: Database,
    project_root: Path,
    entity_type: str,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new entity: write markdown file + seed into SQLite.

    Raises EntityExistsError if file already exists.
    """
    if entity_type not in _TYPE_REGISTRY:
        raise ValueError(f"Unknown entity type: {entity_type}. Valid: {list(_TYPE_REGISTRY)}")

    reg = _TYPE_REGISTRY[entity_type]
    slug = _name_to_slug(name)
    file_path = project_root / reg.directory / f"{slug}.md"

    if file_path.exists():
        raise EntityExistsError(f"{entity_type.title()} '{name}' already exists at {file_path}")

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = _build_markdown(name, entity_type, metadata, aliases)
    _write_atomic(file_path, content)

    # Sync to SQLite
    seed_entities(db, project_root)

    return {
        "name": name,
        "entity_type": entity_type,
        "path": str(file_path.relative_to(project_root)),
    }


def edit_entity(
    db: Database,
    project_root: Path,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Edit an existing entity's metadata. Preserves freeform content.

    Raises EntityNotFoundError if entity not in DB.
    """
    from kb.config import find_entity
    from kb.entities import load_entity_by_id
    from kb.writeback import write_entity_to_file

    conn = db.get_sqlite_conn()
    row = find_entity(conn, name)
    if row is None:
        raise EntityNotFoundError(f"Entity not found: {name}")

    entity = load_entity_by_id(db, row["id"])
    if entity is None:
        raise EntityNotFoundError(f"Entity not found: {name}")

    # Merge metadata updates (empty string = remove key)
    if metadata:
        for key, value in metadata.items():
            if value == "":
                entity.metadata.pop(key, None)
            else:
                entity.metadata[key] = value

    # Merge alias updates
    if aliases:
        for alias in aliases:
            if alias not in entity.aliases:
                entity.aliases.append(alias)

    # Update DB first
    from datetime import date

    conn.execute(
        "UPDATE entities SET aliases=?, metadata=?, updated_at=? WHERE id=?",
        (
            _json.dumps(entity.aliases),
            _json.dumps(entity.metadata),
            date.today().isoformat(),
            entity.id,
        ),
    )
    conn.commit()

    # Write back to file (preserves freeform content)
    write_entity_to_file(db, entity.id, project_root)

    # Re-seed to ensure consistency
    seed_entities(db, project_root)

    return {"name": entity.name, "entity_type": entity.entity_type, "updated": True}


def delete_entity(
    db: Database,
    project_root: Path,
    name: str,
) -> dict[str, Any]:
    """Delete an entity: remove file + clean SQLite.

    Raises EntityNotFoundError if entity not in DB.
    """
    from kb.config import find_entity

    conn = db.get_sqlite_conn()
    row = find_entity(conn, name)
    if row is None:
        raise EntityNotFoundError(f"Entity not found: {name}")

    entity_id = row["id"]
    source_path = row["source_path"]

    # Remove file if it exists
    if source_path:
        file_path = project_root / source_path
        if file_path.exists():
            file_path.unlink()

    # Clean SQLite
    conn.execute("DELETE FROM facts WHERE entity_id = ?", (entity_id,))
    conn.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (entity_id,))
    conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
    conn.commit()

    return {"name": row["name"], "entity_type": row["entity_type"], "deleted": True}
