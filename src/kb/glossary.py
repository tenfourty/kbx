"""Glossary CRUD for memory/glossary.md."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from kb.types import GlossaryEntry

if TYPE_CHECKING:
    from pathlib import Path


def _read_glossary(project_root: Path) -> str:
    path = project_root / "memory" / "glossary.md"
    if not path.exists():
        return "# Glossary\n"
    return path.read_text(encoding="utf-8")


def _write_glossary(project_root: Path, content: str) -> None:
    path = project_root / "memory" / "glossary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def add_term(
    project_root: Path,
    term: str,
    expansion: str,
    *,
    section: str = "Acronyms",
) -> dict[str, Any]:
    """Add a term to glossary.md under the given section."""
    content = _read_glossary(project_root)
    new_row = f"| {term} | {expansion} |"

    # Find the section
    section_pattern = rf"^## {re.escape(section)}\s*$"
    match = re.search(section_pattern, content, re.MULTILINE)

    if match:
        # Find the end of the table in this section (next ## or EOF)
        rest = content[match.end() :]
        next_section = re.search(r"^## ", rest, re.MULTILINE)
        if next_section:
            insert_pos = match.end() + next_section.start()
            content = content[:insert_pos] + new_row + "\n" + content[insert_pos:]
        else:
            content = content.rstrip("\n") + "\n" + new_row + "\n"
    else:
        # Create the section at the end
        content = (
            content.rstrip("\n")
            + f"\n\n## {section}\n\n| Term | Expansion |\n|------|----------|\n{new_row}\n"
        )

    _write_glossary(project_root, content)
    return {"term": term, "expansion": expansion, "section": section}


def delete_term(project_root: Path, term: str) -> dict[str, Any]:
    """Delete a term from glossary.md."""
    content = _read_glossary(project_root)
    # Match the table row for this term
    pattern = rf"^\| *{re.escape(term)} *\|.*\|.*$\n?"
    new_content, count = re.subn(pattern, "", content, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"Term not found: {term}")
    _write_glossary(project_root, new_content)
    return {"term": term, "deleted": True}


def list_terms(project_root: Path) -> list[GlossaryEntry]:
    """List all glossary terms."""
    from kb.context import _parse_glossary_terms

    raw = _parse_glossary_terms(project_root)
    return [GlossaryEntry(term=t, expansion=e, section=s) for t, e, s in raw]
