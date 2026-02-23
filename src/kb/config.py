"""Shared configuration and helpers for kb CLI and MCP server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from kb.db import Database

if TYPE_CHECKING:
    import sqlite3


def find_project_root() -> Path:
    """Find project root using config file or legacy detection.

    Search order:
      1. kbx.toml config file (via find_config) — project root is its parent
      2. Legacy fallback: walk up from CWD looking for kb/ + meetings/ or memory/
      3. Fall back to CWD if nothing found
    """
    from kb.user_config import find_config

    config_path = find_config()
    if config_path is not None:
        return config_path.parent

    # Legacy fallback: walk up looking for kbx/ + meetings/ or memory/
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        if (d / "kbx").is_dir() and ((d / "meetings").is_dir() or (d / "memory").is_dir()):
            return d
    return cwd


def get_data_dir() -> Path:
    """Get the database directory using config or env var or default.

    Priority:
      1. Config file (via user_config) — respects KB_DATA_DIR, data.dir, default
      2. KB_DATA_DIR env var (legacy, relative resolved against project root)
      3. ~/.config/kbx/ (default)

    Always returns an absolute path.
    """
    from kb.user_config import find_config, load_config, resolve_data_dir

    config_path = find_config()
    if config_path is not None:
        cfg = load_config(config_path)
        return resolve_data_dir(cfg, config_path)

    # Legacy: check KB_DATA_DIR env var
    env = os.environ.get("KB_DATA_DIR")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = find_project_root() / p
        return p.resolve()

    return (Path.home() / ".config" / "kbx").resolve()


_db_instance: Database | None = None
_db_data_dir: Path | None = None


def get_db() -> Database:
    """Get or create the Database singleton. Resets if data dir changes."""
    global _db_instance, _db_data_dir
    data_dir = get_data_dir()
    if _db_instance is None or _db_data_dir != data_dir:
        if _db_instance is not None:
            _db_instance.close()
        _db_instance = Database(data_dir)
        _db_data_dir = data_dir
    return _db_instance


def find_entity(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    """Find entity by name/alias (case-insensitive, partial match).

    Returns the first match, or None. For ambiguity-aware lookups, use find_entities().
    """
    results = find_entities(conn, name)
    return results[0] if results else None


def find_entities(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    """Find all matching entities by name/alias (case-insensitive, partial match).

    Returns matches in priority order: exact name > exact alias > partial name > partial alias.
    Multiple results indicate ambiguity (e.g. "Anders" matches two people).
    """
    name_lower = name.lower()

    # Exact name match (case-insensitive)
    rows = conn.execute("SELECT * FROM entities WHERE LOWER(name) = ?", (name_lower,)).fetchall()
    if rows:
        return list(rows)

    # Exact alias match — collect ALL matches (e.g. "Anders" is alias for both Erics)
    all_entities = conn.execute("SELECT * FROM entities").fetchall()
    alias_matches = []
    for e in all_entities:
        aliases = json.loads(e["aliases"]) if e["aliases"] else []
        for alias in aliases:
            if alias.lower() == name_lower:
                alias_matches.append(e)
                break
    if alias_matches:
        return alias_matches

    # Partial name match (substring) — collect ALL matches
    partial_matches = []
    seen_ids: set[int] = set()
    for e in all_entities:
        if name_lower in e["name"].lower() and e["id"] not in seen_ids:
            partial_matches.append(e)
            seen_ids.add(e["id"])
            continue
        aliases = json.loads(e["aliases"]) if e["aliases"] else []
        for alias in aliases:
            if name_lower in alias.lower() and e["id"] not in seen_ids:
                partial_matches.append(e)
                seen_ids.add(e["id"])
                break
    if partial_matches:
        return partial_matches

    return []
