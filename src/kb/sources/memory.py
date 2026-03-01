"""Memory source adapter — walks memory/ and yields ParsedDocuments."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from kb.chunker import Chunk, ParsedDocument, content_hash, parse_frontmatter
from kb.db import normalize_path
from kb.entities import strip_wikilinks

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _extract_title(text: str) -> str | None:
    """Extract title from YAML frontmatter or # heading."""
    fm = parse_frontmatter(text)
    if fm.get("title"):
        return str(fm["title"])
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_glossary(text: str, rel_path: str) -> ParsedDocument:
    """Parse glossary.md into one chunk per table row."""
    title = _extract_title(text) or "Glossary"
    chunks: list[Chunk] = []
    idx = 0

    for line in text.splitlines():
        # Match table data rows (skip header and separator rows)
        if not line.startswith("|"):
            continue
        if line.startswith("|---") or line.startswith("| ---"):
            continue

        # Parse table columns
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last from split
        cells = [c for c in cells if c]

        if len(cells) < 2:
            continue

        # Skip header rows (bold or known headers)
        if cells[0] in ("Term", "Nickname", "**Term**", "**Nickname**"):
            continue

        term = cells[0]
        expansion = cells[1] if len(cells) > 1 else ""
        notes = cells[2] if len(cells) > 2 else ""

        chunk_text = f"{term}: {expansion}"
        if notes:
            chunk_text += f". {notes}"

        chunks.append(Chunk(index=idx, heading=term, content=chunk_text))
        idx += 1

    return ParsedDocument(
        path=rel_path,
        title=title,
        date=None,
        doc_type="memory_glossary",
        source_system="memory",
        source_id=rel_path,
        tags=[],
        content_hash=content_hash(text),
        chunks=chunks,
        raw_body=text,
    )


def _parse_memory_file(
    path: Path,
    rel_path: str,
    doc_type: str,
) -> ParsedDocument:
    """Parse a memory file as a single chunk."""
    text = path.read_text(encoding="utf-8")
    title = _extract_title(text) or path.stem.replace("-", " ").title()

    # Extract date, tags, and pinned from YAML frontmatter if present
    fm = parse_frontmatter(text)
    date: str | None = None
    tags: list[str] = []
    if fm.get("date"):
        date = str(fm["date"])
    if fm.get("tags") and isinstance(fm["tags"], list):
        tags = [str(t) for t in fm["tags"]]
    pinned = bool(fm.get("pinned", False))

    chunk_content = strip_wikilinks(text.strip())
    chunks = [Chunk(index=0, heading=None, content=chunk_content)] if chunk_content else []

    return ParsedDocument(
        path=rel_path,
        title=title,
        date=date,
        doc_type=doc_type,
        source_system="memory",
        source_id=rel_path,
        tags=tags,
        content_hash=content_hash(text),
        chunks=chunks,
        raw_body=text,
        pinned=pinned,
    )


_SUBDIR_DOC_TYPES: dict[str, str] = {
    "people": "memory_person",
    "projects": "memory_project",
    "context": "memory_context",
    "notes": "memory_note",
}
"""Known subdirectory → doc_type mapping. Anything else gets 'memory_doc'."""


def walk_memory(project_root: Path) -> Iterator[ParsedDocument]:
    """Yield ParsedDocument for each memory file.

    Known subdirectories get specific doc_types:
    - memory/people/*.md → memory_person
    - memory/projects/*.md → memory_project
    - memory/context/*.md → memory_context
    - memory/notes/*.md → memory_note
    - memory/glossary.md → memory_glossary (chunk per table row)

    All other memory/**/*.md files → memory_doc (catch-all).
    """
    memory_dir = project_root / "memory"
    if not memory_dir.exists():
        return

    # Track yielded paths to avoid duplicates between specific + catch-all passes
    yielded: set[str] = set()

    # Glossary (special chunking)
    glossary = memory_dir / "glossary.md"
    if glossary.exists():
        text = glossary.read_text(encoding="utf-8")
        rel_path = normalize_path(str(glossary.relative_to(project_root)))
        yielded.add(rel_path)
        yield _parse_glossary(text, rel_path)

    # Known subdirectories with specific doc_types
    for subdir_name, doc_type in _SUBDIR_DOC_TYPES.items():
        subdir = memory_dir / subdir_name
        if not subdir.exists():
            continue
        for f in sorted(subdir.glob("*.md")):
            if f.name == "glossary.md":
                continue  # already handled above
            rel_path = normalize_path(str(f.relative_to(project_root)))
            yielded.add(rel_path)
            yield _parse_memory_file(f, rel_path, doc_type)

    # Catch-all: any other memory/**/*.md not yet yielded.
    # Skip memory/meetings/ — those are handled by the meetings source adapter.
    meetings_subdir = memory_dir / "meetings"
    for f in sorted(memory_dir.rglob("*.md")):
        if f.is_relative_to(meetings_subdir):
            continue
        rel_path = normalize_path(str(f.relative_to(project_root)))
        if rel_path in yielded:
            continue
        yielded.add(rel_path)
        yield _parse_memory_file(f, rel_path, "memory_doc")
