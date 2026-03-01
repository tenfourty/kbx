"""Meeting source adapter — walks a meetings directory and yields ParsedDocuments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kb.chunker import (
    ParsedDocument,
    _strip_frontmatter,
    chunk_notes,
    chunk_transcript,
    content_hash,
    parse_frontmatter,
)
from kb.db import normalize_path

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def walk_meetings(
    project_root: Path, *, meetings_dir: Path | None = None
) -> Iterator[ParsedDocument]:
    """Yield ParsedDocument for each meeting file under meetings_dir.

    Args:
        project_root: Root of the project (used for relative path calculation).
        meetings_dir: Resolved meetings directory. Defaults to project_root / "meetings".

    Parses .notes.md and .transcript.md files with YAML frontmatter.
    """
    if meetings_dir is None:
        meetings_dir = project_root / "memory" / "meetings"
    if not meetings_dir.exists():
        return

    for md_file in sorted(meetings_dir.rglob("*.md")):
        # Only process .notes.md and .transcript.md files
        name = md_file.name
        if not (name.endswith(".notes.md") or name.endswith(".transcript.md")):
            continue

        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        fm = parse_frontmatter(text)
        if not fm or not fm.get("title"):
            continue

        title = fm["title"]
        date = fm.get("date")
        doc_type = fm.get("type", "notes")
        granola_id = fm.get("granola_id", "")
        notion_page_id = fm.get("notion_page_id", "")
        tags = fm.get("tags", [])
        attendees = fm.get("attendees", [])

        # Detect source system from frontmatter or filename
        source_field = fm.get("source", "")
        if source_field and "notion" in source_field:
            source_system = "notion"
        else:
            source_system = "granola"
        source_id = str(granola_id or notion_page_id or "")

        body = _strip_frontmatter(text)

        if doc_type == "transcript":
            chunks = chunk_transcript(body, title, date)
        else:
            chunks = chunk_notes(body, title, date)

        # Skip documents with no chunks
        if not chunks:
            continue

        rel_path = normalize_path(str(md_file.relative_to(project_root)))

        yield ParsedDocument(
            path=rel_path,
            title=title,
            date=date,
            doc_type=doc_type,
            source_system=source_system,
            source_id=source_id,
            tags=tags,
            content_hash=content_hash(text),
            chunks=chunks,
            raw_body=body,
            attendees=attendees,
        )
