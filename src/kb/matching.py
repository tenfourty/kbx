"""Task-to-project matching logic.

Pure functions for linking external tasks to kbx projects. No DB or IO
dependencies — operates on TypedDicts and strings only.

Two-tier algorithm:
  Tier 1: Explicit ``project: <Name>`` line in task description (1:1, first wins).
  Tier 2: Word-boundary title matching against name + aliases + task_keywords.

Short keywords (< min_keyword_len chars) use ``\\b`` word-boundary regex;
longer patterns use fast substring matching.
"""

from __future__ import annotations

import ast
import re
from typing import Any, TypedDict


class TaskInput(TypedDict, total=False):
    """Minimal task shape for matching. kbx's own type — no gm dependency.

    Required: ``title``.
    Optional: ``description`` (used for Tier 1 explicit project links).
    All other keys are passed through in results but ignored by matching.
    """

    title: str
    description: str | None


class ProjectInput(TypedDict, total=False):
    """Project shape for matching. Typically built from EntitySummary fields.

    Required: ``name``.
    Optional: ``aliases``, ``metadata`` (may contain ``task_keywords``).
    """

    name: str
    aliases: list[str]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Tier 1: explicit project link in task descriptions
# ---------------------------------------------------------------------------

# Convention: one ``project: <Name>`` line per task description (1:1).
_PROJECT_LINE_RE = re.compile(r"^project:\s*(.+)", re.IGNORECASE | re.MULTILINE)


def extract_project_link(description: str | None) -> str | None:
    """Extract the explicit project name from a ``project: <Name>`` line.

    Returns lowercased project name, or ``None`` if no match.
    Convention: 1:1 task-to-project. If multiple lines exist, first wins.
    """
    if not description:
        return None
    m = _PROJECT_LINE_RE.search(description)
    if m:
        val = m.group(1).strip()
        if val:
            return val.lower()
    return None


# ---------------------------------------------------------------------------
# Keyword parsing
# ---------------------------------------------------------------------------


def _parse_keywords(raw: str | list[str] | Any) -> list[str]:
    """Parse task_keywords from metadata.

    Handles multiple formats:
    - Already a list: ``['AI', 'agentic']`` — returned directly (lowered)
    - Stringified Python list: ``"['AI', 'agentic']"`` — parsed with ast.literal_eval
    - Comma-separated string: ``"infra, deploy"`` — split on commas
    """
    if isinstance(raw, list):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    if not isinstance(raw, str):
        return []
    stripped = raw.strip()
    if not stripped:
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(item).strip().lower() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
    return [kw.strip().lower() for kw in stripped.split(",") if kw.strip()]


# ---------------------------------------------------------------------------
# Pattern building
# ---------------------------------------------------------------------------


def _build_match_patterns(
    name: str,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    min_keyword_len: int = 4,
) -> list[tuple[str, bool]]:
    """Build ``(lowered_pattern, needs_word_boundary)`` tuples.

    Short patterns (< *min_keyword_len* chars) get word-boundary enforcement
    to prevent false positives like ``"AI"`` matching ``"Railway"``.

    Returns deduplicated patterns (preserving order).
    """
    raw_patterns: list[str] = []
    if name:
        raw_patterns.append(name.lower())
    if aliases:
        raw_patterns.extend(a.lower() for a in aliases if a)
    if metadata:
        kw_raw = metadata.get("task_keywords")
        if kw_raw:
            raw_patterns.extend(_parse_keywords(kw_raw))

    seen: set[str] = set()
    result: list[tuple[str, bool]] = []
    for p in raw_patterns:
        if p and p not in seen:
            seen.add(p)
            result.append((p, len(p) < min_keyword_len))
    return result


def _title_matches(title_lower: str, patterns: list[tuple[str, bool]]) -> bool:
    """Check if a task title matches any project pattern.

    Short patterns use word-boundary regex; long patterns use substring.
    """
    for pattern, needs_boundary in patterns:
        if needs_boundary:
            if re.search(rf"\b{re.escape(pattern)}\b", title_lower):
                return True
        else:
            if pattern in title_lower:
                return True
    return False


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------


def match_tasks_to_projects(
    projects: list[ProjectInput],
    tasks: list[TaskInput],
    *,
    min_keyword_len: int = 4,
) -> dict[str, list[TaskInput]]:
    """Match tasks to projects via two-tier linking.

    Parameters
    ----------
    projects:
        List of ``ProjectInput`` dicts with ``name``, ``aliases``, ``metadata``.
    tasks:
        List of ``TaskInput`` dicts with ``title`` and optional ``description``.
        Any extra keys are preserved in results but ignored by matching.
    min_keyword_len:
        Keywords shorter than this use word-boundary matching (``\\b``).
        Keywords >= this length use substring matching.
        Default 4 eliminates false positives for "AI", "NHI", "ODS", etc.

    Returns
    -------
    dict mapping project name to list of matched ``TaskInput`` dicts.
    Projects with zero matches have empty lists.

    Matching Algorithm
    ------------------
    **Tier 1 — Explicit link (highest priority):**
    Parse ``project: <Name>`` line from task description. If found, link to
    that project only — skip Tier 2. 1:1 enforced (first match wins).
    Case-insensitive match against project names.

    **Tier 2 — Title matching (fallback):**
    For tasks without explicit links, match task title against project
    name + aliases + task_keywords. Match mode depends on pattern length:
    - len >= min_keyword_len: substring match
    - len < min_keyword_len: word-boundary match (``\\bpattern\\b``)

    A task can match multiple projects (via Tier 2).
    A project can match multiple tasks.
    """
    # Build name lookup: lowered project name → canonical name
    name_lookup: dict[str, str] = {}
    for project in projects:
        pname = project.get("name", "")
        if pname:
            name_lookup[pname.lower()] = pname

    result: dict[str, list[TaskInput]] = {project.get("name", ""): [] for project in projects}

    # Pre-build patterns per project for Tier 2
    project_patterns: list[tuple[str, list[tuple[str, bool]]]] = []
    for project in projects:
        pname = project.get("name", "")
        if not pname:
            continue
        patterns = _build_match_patterns(
            pname,
            aliases=project.get("aliases"),
            metadata=project.get("metadata"),
            min_keyword_len=min_keyword_len,
        )
        if patterns:
            project_patterns.append((pname, patterns))

    # Separate tasks into Tier 1 (explicitly linked) and Tier 2 (fallback)
    fallback_tasks: list[TaskInput] = []
    for task in tasks:
        link = extract_project_link(task.get("description"))
        if link is not None:
            canonical = name_lookup.get(link)
            if canonical is not None:
                result[canonical].append(task)
            # Skip Tier 2 even if project name didn't match — explicit link means
            # the user intended a specific project; don't guess.
        else:
            fallback_tasks.append(task)

    # Tier 2: title matching for remaining tasks
    for task in fallback_tasks:
        title_lower = task.get("title", "").lower()
        if not title_lower:
            continue
        for pname, patterns in project_patterns:
            if _title_matches(title_lower, patterns):
                result[pname].append(task)

    return result
