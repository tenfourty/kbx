#!/usr/bin/env python3
"""Migrate person/project files from **Key:** Value to YAML frontmatter."""

import argparse
import contextlib
import os
import re
import tempfile
from pathlib import Path

import yaml

_ENTITY_REF_FIELDS = {"team", "reports_to", "lead", "company"}


def _parse_title(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _has_yaml_frontmatter(text: str) -> bool:
    return bool(re.match(r"^---\s*\n", text))


def _parse_old_header(text: str) -> tuple[dict[str, str], str, str]:
    """Returns (fields_dict, title, body_from_first_##)."""
    match = re.search(r"^## ", text, re.MULTILINE)
    if match:
        header_text = text[: match.start()]
        body = text[match.start() :]
    else:
        header_text = text
        body = ""
    title = _parse_title(header_text) or "Untitled"
    fields: dict[str, str] = {}
    for m in re.finditer(r"\*\*(.+?):\*\*\s*(.+)", header_text):
        label = m.group(1).strip()
        value = m.group(2).strip()
        key = "_".join(label.lower().split())
        fields[key] = value
    return fields, title, body


def _add_wikilink(value: str) -> str:
    if "[[" in value:
        return value
    # Only wrap simple values (no complex punctuation)
    return f"[[{value}]]"


def _convert_person(text: str) -> str:
    if _has_yaml_frontmatter(text):
        return text
    fields, title, body = _parse_old_header(text)
    fm: dict[str, object] = {}

    aka = fields.pop("also_known_as", None)
    if aka:
        fm["aliases"] = [a.strip() for a in aka.split(",")]

    for key in ["email", "role", "team", "reports_to", "company", "pcm_base", "pcm_phase"]:
        val = fields.pop(key, None)
        if val:
            fm[key] = _add_wikilink(val) if key in _ENTITY_REF_FIELDS else val

    pinned = fields.pop("pinned", None)
    if pinned and pinned.lower() in ("true", "yes", "1"):
        fm["pinned"] = True

    for key, val in fields.items():
        if val:
            fm[key] = val

    if not fm:
        return text  # nothing to convert

    yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n# {title}\n\n{body}"


def _convert_project(text: str) -> str:
    if _has_yaml_frontmatter(text):
        return text
    fields, title, body = _parse_old_header(text)
    fm: dict[str, object] = {}

    codename = fields.pop("codename/also_called", None)
    if codename:
        fm["aliases"] = [a.strip() for a in codename.split(",")]

    for key in ["status", "started", "lead"]:
        val = fields.pop(key, None)
        if val:
            fm[key] = _add_wikilink(val) if key in _ENTITY_REF_FIELDS else val

    pinned = fields.pop("pinned", None)
    if pinned and pinned.lower() in ("true", "yes", "1"):
        fm["pinned"] = True

    for key, val in fields.items():
        if val:
            fm[key] = val

    if not fm:
        return text

    yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n# {title}\n\n{body}"


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
    parser = argparse.ArgumentParser(description="Migrate entity files to YAML frontmatter")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Project root (Control Tower directory)",
    )
    args = parser.parse_args()

    root = args.project_root
    people_dir = root / "memory" / "people"
    projects_dir = root / "memory" / "projects"

    converted = skipped = 0

    if people_dir.exists():
        for path in sorted(people_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            new_text = _convert_person(text)
            if new_text == text:
                skipped += 1
                continue
            if args.dry_run:
                print(f"  WOULD convert: {path.name}")
            else:
                _atomic_write(path, new_text)
                print(f"  Converted: {path.name}")
            converted += 1

    if projects_dir.exists():
        for path in sorted(projects_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            new_text = _convert_project(text)
            if new_text == text:
                skipped += 1
                continue
            if args.dry_run:
                print(f"  WOULD convert: {path.name}")
            else:
                _atomic_write(path, new_text)
                print(f"  Converted: {path.name}")
            converted += 1

    print(f"\nDone: {converted} converted, {skipped} skipped")


if __name__ == "__main__":
    main()
