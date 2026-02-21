"""Write entity metadata from DB back to source markdown files.

Preserves freeform content (## sections) while rewriting the structured
header fields (# Title, **Field:** value lines).
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from kb.entities import Entity, load_entity_by_id

if TYPE_CHECKING:
    from pathlib import Path

    from kb.db import Database

# Person fields in output order (field_name, metadata_key)
_PERSON_FIELDS = [
    ("Also known as", None),  # special: from aliases list
    ("Email", "email"),
    ("Role", "role"),
    ("Team", "team"),
    ("Reports to", "reports_to"),
    ("Company", "company"),
    ("Pinned", None),  # special: from entity.pinned
]

# Project fields in output order
_PROJECT_FIELDS = [
    ("Codename/Also called", None),  # special: from aliases list
    ("Status", "status"),
    ("Started", "started"),
    ("Lead", "lead"),
    ("Pinned", None),
]


def _split_header_and_body(content: str) -> tuple[str, str]:
    """Split file content into structured header and freeform body.

    The header is everything before the first ## heading.
    The body is the ## heading and everything after it.
    Returns (header, body) where body may be empty.
    """
    # Find first ## heading (not inside the title # line)
    match = re.search(r"^## ", content, re.MULTILINE)
    if match:
        return content[: match.start()], content[match.start() :]
    return content, ""


def _derive_aliases(entity: Entity) -> list[str]:
    """Get the non-derived aliases for an entity.

    Excludes:
    - First-name-only aliases (single word matching first word of entity name)
    - File-stem aliases (contain hyphens and all lowercase)
    """
    name_parts = entity.name.split()
    first_name = name_parts[0] if len(name_parts) >= 2 else None
    result = []
    for alias in entity.aliases:
        # Skip file stems (lowercase with hyphens)
        if "-" in alias and alias == alias.lower():
            continue
        # Skip first-name auto-alias
        if first_name and alias == first_name:
            continue
        result.append(alias)
    return result


def _rebuild_person_header(entity: Entity) -> str:
    """Rebuild the structured header for a person entity file."""
    lines = [f"# {entity.name}", ""]

    aliases = _derive_aliases(entity)

    for field_name, meta_key in _PERSON_FIELDS:
        if field_name == "Also known as":
            if aliases:
                lines.append(f"**Also known as:** {', '.join(aliases)}")
        elif field_name == "Pinned":
            if entity.pinned:
                lines.append("**Pinned:** true")
        else:
            value = entity.metadata.get(meta_key) if meta_key else None
            if value:
                lines.append(f"**{field_name}:** {value}")

    lines.append("")
    return "\n".join(lines)


def _rebuild_project_header(entity: Entity) -> str:
    """Rebuild the structured header for a project entity file."""
    lines = [f"# {entity.name}", ""]

    aliases = _derive_aliases(entity)

    for field_name, meta_key in _PROJECT_FIELDS:
        if field_name == "Codename/Also called":
            if aliases:
                lines.append(f"**Codename/Also called:** {', '.join(aliases)}")
        elif field_name == "Pinned":
            if entity.pinned:
                lines.append("**Pinned:** true")
        else:
            value = entity.metadata.get(meta_key) if meta_key else None
            if value:
                lines.append(f"**{field_name}:** {value}")

    lines.append("")
    return "\n".join(lines)


def _write_atomic(path: Path, content: str) -> None:
    """Atomic write: write to temp file then rename."""
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def write_entity_to_file(
    db: Database,
    entity_id: int,
    project_root: Path,
) -> bool:
    """Write entity metadata from DB back to its source file.

    Preserves all freeform content (## sections and below).
    Returns True if the file was written, False if skipped.
    """
    entity = load_entity_by_id(db, entity_id)
    if not entity or not entity.source_path:
        return False

    # Only write back to memory/people/ and memory/projects/ files
    if not (
        entity.source_path.startswith("memory/people/")
        or entity.source_path.startswith("memory/projects/")
    ):
        return False

    file_path = project_root / entity.source_path
    if not file_path.exists():
        return False

    # Read existing content and split header from body
    existing = file_path.read_text(encoding="utf-8")
    _header, body = _split_header_and_body(existing)

    # Rebuild header based on entity type
    if entity.entity_type == "person":
        new_header = _rebuild_person_header(entity)
    elif entity.entity_type == "project":
        new_header = _rebuild_project_header(entity)
    else:
        return False

    new_content = new_header + body

    # Only write if content actually changed
    if new_content == existing:
        return False

    _write_atomic(file_path, new_content)
    return True
