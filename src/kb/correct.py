"""Find-and-replace corrections across memory files.

Scans memory/ for literal string occurrences and applies replacements
with atomic file writes. Designed for fixing STT garbles (e.g. Quartz Indexer →
Datalux) and entity name corrections (e.g. Bram → Bram).
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from kb.types import StrictFrozen

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CorrectionMatch(StrictFrozen):
    """A file containing one or more occurrences of the search term."""

    rel_path: str
    count: int
    sample_lines: list[str]
    search_term: str  # original term used in scan()


class CorrectionResult(StrictFrozen):
    """Summary of corrections applied."""

    files_changed: int
    occurrences_replaced: int
    changed_paths: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_pattern(
    term: str,
    *,
    ignore_case: bool = False,
    word_boundary: bool = False,
) -> re.Pattern[str]:
    """Build a compiled regex for the literal search term."""
    escaped = re.escape(term)
    if word_boundary:
        escaped = rf"\b{escaped}\b"
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(escaped, flags)


def _match_scope(rel_path: str, scope: str | None) -> bool:
    """Check whether a relative path matches the scope filter."""
    if scope is None:
        return True
    if rel_path == scope:
        return True
    # fnmatch treats * as matching any chars including /
    if fnmatch.fnmatch(rel_path, scope):
        return True
    # Handle **/ prefix: strip it for root-level matches
    # (e.g. **/people/* should match people/eric.md)
    if scope.startswith("**/"):
        return fnmatch.fnmatch(rel_path, scope[3:])
    return False


def _match_file_type(rel_path: str, file_type: str | None) -> bool:
    """Check whether a relative path matches the file_type filter."""
    if file_type is None:
        return True
    return file_type in Path(rel_path).name


def _extract_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from a markdown file.

    Returns parsed dict, or empty dict if no valid frontmatter found.
    """
    if not content.startswith("---\n"):
        return {}
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}
    yaml_str = content[4:end]
    try:
        import yaml

        data = yaml.safe_load(yaml_str)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _categorize_path(rel_path: str) -> str:
    """Categorize a file by its location in the memory tree."""
    if rel_path.startswith("meetings/"):
        return "meeting"
    if rel_path.startswith("people/"):
        return "entity"
    if rel_path.startswith("projects/"):
        return "entity"
    if rel_path.startswith("notes/"):
        return "note"
    return "other"


def _atomic_write(file_path: Path, content: str) -> None:
    """Write content to file atomically via temp file + rename."""
    fd, tmp = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, str(file_path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def scan(
    memory_root: Path,
    term: str,
    *,
    ignore_case: bool = False,
    word_boundary: bool = False,
    scope: str | None = None,
    file_type: str | None = None,
) -> list[CorrectionMatch]:
    """Scan memory files for occurrences of a literal term.

    Args:
        memory_root: Path to the memory/ directory.
        term: Literal string to search for.
        ignore_case: Match case-insensitively.
        word_boundary: Only match whole words.
        scope: Glob or exact relative path to limit search.
        file_type: Filter files by substring in filename (e.g. "transcript").

    Returns:
        List of CorrectionMatch objects, one per file with matches.
    """
    pattern = _build_pattern(term, ignore_case=ignore_case, word_boundary=word_boundary)
    matches: list[CorrectionMatch] = []

    for dirpath, _dirnames, filenames in os.walk(memory_root):
        for fname in sorted(filenames):
            if not fname.endswith(".md"):
                continue
            full_path = Path(dirpath) / fname
            rel_path = unicodedata.normalize("NFC", str(full_path.relative_to(memory_root)))

            if not _match_scope(rel_path, scope):
                continue
            if not _match_file_type(rel_path, file_type):
                continue

            try:
                content = full_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            found = pattern.findall(content)
            if not found:
                continue

            sample_lines = [line for line in content.splitlines() if pattern.search(line)]

            matches.append(
                CorrectionMatch(
                    rel_path=rel_path,
                    count=len(found),
                    sample_lines=sample_lines,
                    search_term=term,
                )
            )

    return matches


# ---------------------------------------------------------------------------
# Enrichment (for agent workflows)
# ---------------------------------------------------------------------------


def enrich_matches(
    memory_root: Path,
    matches: list[CorrectionMatch],
) -> list[dict[str, Any]]:
    """Enrich CorrectionMatch objects with frontmatter metadata.

    Extracts title, date, attendees from YAML frontmatter for each matched
    file. Designed for agent consumption — provides context needed to make
    disambiguation decisions (e.g. which "Bram" is in this meeting).

    Args:
        memory_root: Path to the memory/ directory.
        matches: CorrectionMatch objects from scan().

    Returns:
        List of dicts with match fields + title, date, attendees, category.
    """
    enriched: list[dict[str, Any]] = []

    for match in matches:
        full_path = memory_root / match.rel_path
        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""

        fm = _extract_frontmatter(content)
        category = _categorize_path(match.rel_path)

        attendees: list[dict[str, str]] = []
        if category == "meeting":
            raw_attendees = fm.get("attendees", [])
            if isinstance(raw_attendees, list):
                for a in raw_attendees:
                    if isinstance(a, dict) and "name" in a:
                        entry: dict[str, str] = {"name": a["name"]}
                        if "email" in a:
                            entry["email"] = a["email"]
                        attendees.append(entry)

        enriched.append(
            {
                "rel_path": match.rel_path,
                "count": match.count,
                "sample_lines": match.sample_lines,
                "search_term": match.search_term,
                "title": fm.get("title", ""),
                "date": str(fm["date"]) if "date" in fm else "",
                "attendees": attendees,
                "category": category,
            }
        )

    return enriched


# ---------------------------------------------------------------------------
# Applicator
# ---------------------------------------------------------------------------


def apply_corrections(
    memory_root: Path,
    matches: list[CorrectionMatch],
    new_term: str,
    *,
    ignore_case: bool = False,
    word_boundary: bool = False,
    log_path: Path | None = None,
) -> CorrectionResult:
    """Apply find-and-replace corrections to matched files.

    Uses atomic writes (temp file → os.replace) to avoid partial writes.

    Args:
        memory_root: Path to the memory/ directory.
        matches: CorrectionMatch objects from scan().
        new_term: Replacement string.
        ignore_case: Replace case-insensitively.
        word_boundary: Only replace whole-word matches.
        log_path: Optional path to append audit log entries.

    Returns:
        CorrectionResult with counts of changes made.
    """
    if not matches:
        return CorrectionResult(
            files_changed=0,
            occurrences_replaced=0,
            changed_paths=[],
        )

    files_changed = 0
    total_replaced = 0
    changed_paths: list[str] = []

    for match in matches:
        full_path = memory_root / match.rel_path
        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        pattern = _build_pattern(
            match.search_term,
            ignore_case=ignore_case,
            word_boundary=word_boundary,
        )

        new_content, count = pattern.subn(lambda _: new_term, content)
        if count > 0:
            _atomic_write(full_path, new_content)
            files_changed += 1
            total_replaced += count
            changed_paths.append(match.rel_path)

    result = CorrectionResult(
        files_changed=files_changed,
        occurrences_replaced=total_replaced,
        changed_paths=changed_paths,
    )

    if log_path is not None and files_changed > 0:
        _write_audit_log(log_path, matches[0].search_term, new_term, result)

    return result


def _write_audit_log(
    log_path: Path,
    old_term: str,
    new_term: str,
    result: CorrectionResult,
) -> None:
    """Append an audit log entry for a correction run."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    entry = (
        f"[{ts}] '{old_term}' → '{new_term}' | "
        f"{result.files_changed} files, {result.occurrences_replaced} replacements | "
        f"paths: {', '.join(result.changed_paths)}\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
