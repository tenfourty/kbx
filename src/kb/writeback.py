"""Write entity metadata from DB back to source markdown files.

Preserves freeform content (## sections) while rewriting the structured
YAML frontmatter header.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import yaml

from kb.entities import Entity, load_entity_by_id

if TYPE_CHECKING:
    from pathlib import Path

    from kb.db import Database


# Person YAML fields in output order: (key, is_entity_ref)
_PERSON_YAML_FIELDS = [
    ("email", False),
    ("role", False),
    ("team", True),
    ("reports_to", True),
    ("company", True),
    ("pcm_base", False),
    ("pcm_phase", False),
]

# Project YAML fields in output order
_PROJECT_YAML_FIELDS = [
    ("status", False),
    ("started", False),
    ("lead", True),
]


def _split_header_and_body(content: str) -> tuple[str, str]:
    """Split file content into structured header and freeform body.

    Handles both YAML frontmatter (---...---) and legacy (**Key:** value) formats.
    The header is everything before the first ## heading.
    The body is the ## heading and everything after it.
    Returns (header, body) where body may be empty.
    """
    # Check for YAML frontmatter first
    fm_match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    if fm_match:
        after_fm = content[fm_match.end() :]
        body_match = re.search(r"^## ", after_fm, re.MULTILINE)
        if body_match:
            header_end = fm_match.end() + body_match.start()
            return content[:header_end], content[header_end:]
        return content, ""

    # Legacy format: split at first ##
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
        # Skip source-ID aliases (src:C08HJC8MWQN)
        if alias.startswith("src:"):
            continue
        result.append(alias)
    return result


def _add_wikilinks_to_entity_ref(value: str) -> str:
    """Add [[wikilinks]] to entity-reference field values.

    If value already contains [[...]], return as-is.
    Otherwise, wrap the entire value in [[ ]].
    """
    if "[[" in value:
        return value
    return f"[[{value}]]"


def _rebuild_person_header(entity: Entity) -> str:
    """Rebuild the YAML frontmatter header for a person entity file."""
    aliases = _derive_aliases(entity)
    fm: dict[str, object] = {}
    if aliases:
        fm["aliases"] = aliases

    registered_keys: set[str] = set()
    for key, is_ref in _PERSON_YAML_FIELDS:
        registered_keys.add(key)
        value = entity.metadata.get(key)
        if value:
            fm[key] = _add_wikilinks_to_entity_ref(value) if is_ref else value

    if entity.pinned:
        fm["pinned"] = True

    # Custom fields (not in registry)
    for key, value in entity.metadata.items():
        if key not in registered_keys and value:
            fm[key] = value

    yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n# {entity.name}\n\n"


def _rebuild_project_header(entity: Entity) -> str:
    """Rebuild the YAML frontmatter header for a project entity file."""
    aliases = _derive_aliases(entity)
    fm: dict[str, object] = {}
    if aliases:
        fm["aliases"] = aliases

    registered_keys: set[str] = set()
    for key, is_ref in _PROJECT_YAML_FIELDS:
        registered_keys.add(key)
        value = entity.metadata.get(key)
        if value:
            fm[key] = _add_wikilinks_to_entity_ref(value) if is_ref else value

    if entity.pinned:
        fm["pinned"] = True

    # Custom fields (not in registry)
    for key, value in entity.metadata.items():
        if key not in registered_keys and value:
            fm[key] = value

    yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n# {entity.name}\n\n"


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
