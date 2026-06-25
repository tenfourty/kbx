"""Persistent entity↔document link suppressions (kbx #35 / #36).

Entity↔document links are re-derived on every reindex, and Granola sync regenerates
meeting files — so a suppression ("this entity is *not* in this document") cannot live
in document frontmatter. It lives in a sidecar file at
``memory/.kbx/entity-suppressions.json``, loaded by the indexer and honoured when
deriving entity mentions.

Shape: ``{"<doc rel-path>": ["<entity name>", ...], ...}``; entity names match
case-insensitively. JSON (not TOML) because it is machine-written and JSON has a stdlib
writer.
"""

from __future__ import annotations

import json
from pathlib import Path

_REL = Path("memory") / ".kbx" / "entity-suppressions.json"


def _store_path(project_root: Path) -> Path:
    return project_root / _REL


def _load_raw(project_root: Path) -> dict[str, list[str]]:
    p = _store_path(project_root)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for doc, ents in data.items():
        if isinstance(doc, str) and isinstance(ents, list):
            out[doc] = [e for e in ents if isinstance(e, str)]
    return out


def _save_raw(project_root: Path, data: dict[str, list[str]]) -> None:
    p = _store_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def load_suppressions(project_root: Path) -> dict[str, set[str]]:
    """Return ``{doc_rel_path: {lowercased entity name, ...}}`` for the indexer."""
    return {doc: {e.lower() for e in ents} for doc, ents in _load_raw(project_root).items()}


def add_suppression(project_root: Path, document: str, entity: str) -> None:
    """Suppress the entity↔document link (idempotent, case-insensitive)."""
    data = _load_raw(project_root)
    ents = data.setdefault(document, [])
    if all(e.lower() != entity.lower() for e in ents):
        ents.append(entity)
    _save_raw(project_root, data)


def remove_suppression(project_root: Path, document: str, entity: str) -> bool:
    """Remove a suppression (case-insensitive). Returns True if one was removed."""
    data = _load_raw(project_root)
    ents = data.get(document, [])
    kept = [e for e in ents if e.lower() != entity.lower()]
    if len(kept) == len(ents):
        return False
    if kept:
        data[document] = kept
    else:
        data.pop(document, None)
    _save_raw(project_root, data)
    return True
