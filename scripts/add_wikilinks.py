#!/usr/bin/env python3
"""Add [[wikilinks]] for known entity names in memory file body text.

Usage:
    python scripts/add_wikilinks.py [--dry-run] [--project-root PATH] [--db-path PATH]
"""

import argparse
import contextlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path


def _load_entity_names(db_path: Path) -> list[tuple[str, list[str]]]:
    """Load entity names and aliases from kbx database.

    Returns list of (name, [aliases]) tuples, sorted by name length descending
    (longer names first to avoid partial matches).
    """
    db_file = db_path / "metadata.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT name, aliases FROM entities").fetchall()
    conn.close()

    entities = []
    for row in rows:
        name = row["name"]
        aliases = json.loads(row["aliases"]) if row["aliases"] else []
        entities.append((name, aliases))

    # Sort by name length descending (longer first)
    entities.sort(key=lambda x: len(x[0]), reverse=True)
    return entities


def _build_patterns(
    entities: list[tuple[str, list[str]]],
) -> list[tuple[re.Pattern[str], str]]:
    """Build regex patterns for entity names.

    Only includes multi-word names (2+ words) to avoid ambiguous matches.
    Returns (pattern, replacement_name) tuples.
    """
    patterns = []
    seen: set[str] = set()
    for name, aliases in entities:
        all_names = [name, *aliases]
        for n in all_names:
            # Only multi-word names (2+ words, at least 5 chars)
            words = n.split()
            if len(words) < 2 or len(n) < 5:
                continue
            # Skip file-stem aliases (lowercase with hyphens)
            if "-" in n and n == n.lower():
                continue
            # Skip duplicates (case-insensitive)
            key = n.lower()
            if key in seen:
                continue
            seen.add(key)

            escaped = re.escape(n)
            # Negative lookbehind for [[ and lookahead for ]] to avoid double-wrapping
            pattern = re.compile(
                rf"(?<!\[\[)\b{escaped}\b(?!\]\])",
                re.IGNORECASE,
            )
            patterns.append((pattern, n))

    return patterns


def _split_frontmatter_and_rest(text: str) -> tuple[str, str]:
    """Split YAML frontmatter from the rest of the file."""
    m = re.match(r"^(---\s*\n.*?\n---\s*\n)", text, re.DOTALL)
    if m:
        return m.group(1), text[m.end() :]
    return "", text


def _add_wikilinks_to_text(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    """Add [[wikilinks]] to body text, skipping code blocks and title."""
    frontmatter, rest = _split_frontmatter_and_rest(text)

    if not rest.strip():
        return text

    # Split rest into lines, process each
    lines = rest.split("\n")
    in_code_block = False
    result_lines = []

    for line in lines:
        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result_lines.append(line)
            continue

        if in_code_block:
            result_lines.append(line)
            continue

        # Skip the # Title line
        if re.match(r"^#\s+", line):
            result_lines.append(line)
            continue

        # Apply patterns to this line
        for pattern, name in patterns:
            line = pattern.sub(f"[[{name}]]", line)

        result_lines.append(line)

    return frontmatter + "\n".join(result_lines)


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".md.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Add [[wikilinks]] to memory files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Project root (Control Tower directory)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to kbx-data directory (default: project-root/kbx-data)",
    )
    args = parser.parse_args()

    root = args.project_root
    db_path = args.db_path or root / "kbx-data"

    if not (db_path / "metadata.db").exists():
        print(f"Error: Database not found at {db_path / 'metadata.db'}")
        return

    # Load entities and build patterns
    entities = _load_entity_names(db_path)
    patterns = _build_patterns(entities)
    print(f"Loaded {len(entities)} entities, {len(patterns)} linkable name patterns")

    # Scan memory files
    converted = 0
    skipped = 0

    for subdir in ["memory/people", "memory/projects", "memory/notes", "memory/context"]:
        dir_path = root / subdir
        if not dir_path.exists():
            continue
        for path in sorted(dir_path.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            new_text = _add_wikilinks_to_text(text, patterns)
            if new_text == text:
                skipped += 1
                continue
            if args.dry_run:
                print(f"  WOULD link: {path.relative_to(root)}")
            else:
                _atomic_write(path, new_text)
                print(f"  Linked: {path.relative_to(root)}")
            converted += 1

    print(f"\nDone: {converted} files updated, {skipped} unchanged")


if __name__ == "__main__":
    main()
