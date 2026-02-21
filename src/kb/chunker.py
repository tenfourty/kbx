"""Markdown-aware chunking with YAML frontmatter parsing."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import yaml

from kb.types import Chunk as Chunk
from kb.types import ParsedDocument as ParsedDocument

# Token estimation: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4
MAX_SECTION_TOKENS = 2000
TARGET_TRANSCRIPT_TOKENS = 1000
TRANSCRIPT_OVERLAP_TOKENS = 100


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def content_hash(text: str) -> str:
    """SHA-256 hash of file content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from markdown text.

    Returns dict with keys: title, date, type, granola_id, tags, source,
    plus enriched fields: attendees, calendar_event, granola_summary,
    granola_updated_at. Missing keys default to None or empty list.
    """
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}

    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}

    if not isinstance(fm, dict):
        return {}

    # Normalize date to string
    date_val = fm.get("date")
    if date_val is not None:
        date_val = str(date_val)

    # Normalize granola_summary (strip trailing whitespace from block scalars)
    summary = fm.get("granola_summary")
    if isinstance(summary, str):
        summary = summary.strip()

    return {
        "title": fm.get("title"),
        "date": date_val,
        "type": fm.get("type"),
        "granola_id": fm.get("granola_id"),
        "tags": fm.get("tags", []) or [],
        "source": fm.get("source"),
        "attendees": fm.get("attendees", []) or [],
        "calendar_event": fm.get("calendar_event"),
        "granola_summary": summary or None,
        "granola_updated_at": fm.get("granola_updated_at"),
        "pinned": bool(fm.get("pinned", False)),
    }


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from markdown text."""
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)


def _split_on_headers(body: str) -> list[tuple[str | None, str]]:
    """Split markdown body into (heading, content) pairs on ## and ### headers."""
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines(keepends=True):
        header_match = re.match(r"^(#{2,3})\s+(.+)", line)
        if header_match:
            # Save previous section
            section_text = "".join(current_lines).strip()
            if section_text:
                sections.append((current_heading, section_text))
            current_heading = header_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    section_text = "".join(current_lines).strip()
    if section_text:
        sections.append((current_heading, section_text))

    return sections


def _split_long_section(text: str, max_tokens: int) -> list[str]:
    """Split a long section on paragraph breaks to fit within max_tokens."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    result: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current and current_len + para_tokens > max_tokens:
            result.append("\n\n".join(current))
            current = [para]
            current_len = para_tokens
        else:
            current.append(para)
            current_len += para_tokens

    if current:
        result.append("\n\n".join(current))

    return result


def _build_entity_suffix(entities_info: list[tuple[str, str, str]] | None) -> str:
    """Build entity suffix for chunk prefix.

    entities_info: list of (name, entity_type, extra_info) tuples.
    Returns e.g. ' | Entities: Soren Vance (person), Pipeline Improvements (project)'
    """
    if not entities_info:
        return ""
    parts = []
    for name, etype, _extra in entities_info:
        parts.append(f"{name} ({etype})")
    return " | Entities: " + ", ".join(parts)


def chunk_notes(
    body: str,
    title: str,
    date: str | None,
    entities_info: list[tuple[str, str, str]] | None = None,
) -> list[Chunk]:
    """Chunk meeting notes by splitting on ## / ### headers.

    Each section becomes one or more chunks. Sections exceeding MAX_SECTION_TOKENS
    are split on paragraph breaks.
    """
    sections = _split_on_headers(body)
    chunks: list[Chunk] = []
    idx = 0
    entity_suffix = _build_entity_suffix(entities_info)

    for heading, text in sections:
        # Skip empty or trivial sections
        clean = text.strip()
        if not clean or clean in ("No personal notes available.", "---"):
            continue

        date_str = date or "unknown"
        parts = _split_long_section(clean, MAX_SECTION_TOKENS)

        for part in parts:
            heading_label = heading or "General"
            prefix = (
                f"[Meeting: {title} | Date: {date_str} | Section: {heading_label}{entity_suffix}]"
            )
            chunk_content = f"{prefix}\n{part}"
            chunks.append(Chunk(index=idx, heading=heading, content=chunk_content))
            idx += 1

    # If no sections found, treat entire body as one chunk
    if not chunks:
        clean = body.strip()
        if clean:
            prefix = (
                f"[Meeting: {title} | Date: {date or 'unknown'} | Section: General{entity_suffix}]"
            )
            chunks.append(Chunk(index=0, heading=None, content=f"{prefix}\n{clean}"))

    return chunks


def _parse_speaker_turns(body: str) -> list[tuple[str, str]]:
    """Parse speaker turns from transcript text.

    Returns list of (speaker_label, text) tuples.
    """
    turns: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        m = re.match(r"^\*\*(.+?):\*\*\s*(.*)", line)
        if m:
            if current_speaker is not None:
                turns.append((current_speaker, " ".join(current_lines).strip()))
            current_speaker = m.group(1)
            text = m.group(2).strip()
            current_lines = [text] if text else []
        elif line.strip() and current_speaker is not None:
            current_lines.append(line.strip())

    if current_speaker is not None:
        turns.append((current_speaker, " ".join(current_lines).strip()))

    return turns


def _replace_system_speaker(turns: list[tuple[str, str]], title: str) -> list[tuple[str, str]]:
    """Replace 'System' speaker label with participant name from title."""
    # Parse participants from title: "Wren / Soren" → ["Wren", "Soren"]
    parts = re.split(r"\s+/\s+|\s+x\s+|\s+&\s+|\s+vs\s+", title, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    # The non-"Me" participants
    other_names = [p for p in parts if p.lower() not in ("me",)]

    if len(other_names) == 1:
        replacement = other_names[0]
    elif len(other_names) > 1:
        replacement = "Other"
    else:
        replacement = "System"

    return [(replacement if speaker == "System" else speaker, text) for speaker, text in turns]


def chunk_transcript(
    body: str,
    title: str,
    date: str | None,
    entities_info: list[tuple[str, str, str]] | None = None,
) -> list[Chunk]:
    """Chunk transcript using sliding window on speaker turns.

    Groups turns into windows of ~800-1200 tokens with ~100 token overlap.
    """
    turns = _parse_speaker_turns(body)
    turns = _replace_system_speaker(turns, title)
    entity_suffix = _build_entity_suffix(entities_info)

    if not turns:
        # Fallback: treat as plain text
        clean = body.strip()
        if clean:
            prefix = f"[Transcript: {title} | Date: {date or 'unknown'}{entity_suffix}]"
            return [Chunk(index=0, heading=None, content=f"{prefix}\n{clean}")]
        return []

    # Build turn texts with speaker labels
    turn_texts = [f"{speaker}: {text}" for speaker, text in turns if text]
    if not turn_texts:
        return []

    target_chars = TARGET_TRANSCRIPT_TOKENS * CHARS_PER_TOKEN
    overlap_chars = TRANSCRIPT_OVERLAP_TOKENS * CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    idx = 0
    start = 0
    date_str = date or "unknown"

    while start < len(turn_texts):
        # Accumulate turns until target size
        window: list[str] = []
        window_chars = 0
        end = start

        while end < len(turn_texts) and window_chars < target_chars:
            window.append(turn_texts[end])
            window_chars += len(turn_texts[end]) + 1  # +1 for newline
            end += 1

        prefix = f"[Transcript: {title} | Date: {date_str}{entity_suffix}]"
        chunk_text = f"{prefix}\n" + "\n".join(window)
        chunks.append(Chunk(index=idx, heading=None, content=chunk_text))
        idx += 1

        if end >= len(turn_texts):
            break

        # Move start forward, keeping overlap
        overlap_start = end
        overlap_acc = 0
        while overlap_start > start and overlap_acc < overlap_chars:
            overlap_start -= 1
            overlap_acc += len(turn_texts[overlap_start]) + 1

        start = overlap_start if overlap_start > start else end

    return chunks
