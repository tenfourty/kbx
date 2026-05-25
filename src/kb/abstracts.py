"""Extractive L0 abstracts for indexed documents (issue #66 Phase 1).

Generates a one-sentence summary at index time using cheap pure-Python
heuristics — no LLM. The abstract is stored on ``documents.abstract`` and
surfaced via search results so agents can triage candidates at ~100 tokens
each rather than loading full chunks.

Strategy:
    1. Strip YAML frontmatter and leading markdown headings.
    2. Take the first sentence terminated by ``.``, ``!`` or ``?``.
    3. Fall back to the document title if no sentence is found.
    4. Return ``None`` when both are unavailable.

Phase 2 (LLM-generated L0/L1) plugs in later via the same ``documents.abstract``
column with a marker indicating the source (``"extractive"`` vs ``"llm"``).
"""

from __future__ import annotations

import re

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_FORMAT_RE = re.compile(r"[*_`]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Sentence terminator: ., ! or ? followed by whitespace or end-of-string.
# Negative lookbehind avoids splitting on common abbreviations + single-letter
# initials (e.g. "Mr.", "Dr.", "St.", "vs.", "A.", "Q.") which would otherwise
# truncate sentences mid-name.
_ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "St", "vs", "etc", "Inc", "Co", "Ltd")
_ABBREV_RE = re.compile(r"(?<!\b" + r")(?<!\b".join(_ABBREVIATIONS) + r")(?<!\b[A-Z])([.!?])(\s|$)")


def _strip_markdown(text: str) -> str:
    """Strip wikilinks, markdown links, and inline formatting (``*``, ``_``, `` ` ``)."""
    text = _WIKILINK_RE.sub(r"\1", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _INLINE_FORMAT_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_abstract(
    content: str,
    title: str | None = None,
    max_chars: int = 200,
    min_chars: int = 5,
) -> str | None:
    """Extract a one-sentence L0 abstract from document content.

    Args:
        content: Raw markdown content (may include YAML frontmatter + headings).
        title: Optional document title used as fallback when no sentence found.
        max_chars: Cap sentence length. Longer sentences are truncated with ``…``.
        min_chars: Sentences shorter than this are rejected (falls back to title).

    Returns:
        The cleaned abstract string, or ``None`` if neither content nor title
        yields anything usable.
    """
    # Drop YAML frontmatter and leading headings; both are noise for an L0.
    body = _FRONTMATTER_RE.sub("", content, count=1) if content else ""
    body = _HEADING_RE.sub("", body)

    # Scan paragraphs in order, taking the first sentence that meets length bounds.
    for paragraph in (p.strip() for p in body.split("\n\n")):
        if not paragraph:
            continue
        flat = _strip_markdown(paragraph)
        if not flat:
            continue

        match = _ABBREV_RE.search(flat)
        if match:
            sentence = flat[: match.end(1)].strip()
        else:
            sentence = flat.strip()

        if len(sentence) >= min_chars:
            if len(sentence) > max_chars:
                sentence = sentence[: max_chars - 1].rstrip() + "…"
            return sentence

    if title and title.strip():
        title_clean = title.strip()
        if len(title_clean) > max_chars:
            title_clean = title_clean[: max_chars - 1].rstrip() + "…"
        return title_clean

    return None
