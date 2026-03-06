"""Config file discovery and loading for kbx.

Search order for config file:
  1. $KBX_CONFIG env var (explicit override)
  2. ./kbx.toml (project-local, walk up from CWD)
  3. $XDG_CONFIG_HOME/kbx/config.toml (~/.config/kbx/config.toml)

Data directory resolution:
  1. $KB_DATA_DIR env var (backwards compat)
  2. data.dir from config file (relative to config file location)
  3. ~/.config/kbx/ (default)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel


class SourcesConfig(BaseModel):
    """Source directory configuration."""

    meetings: str = "memory/meetings"
    memory: str = "memory"


class DataConfig(BaseModel):
    """Data directory configuration."""

    dir: str | None = None  # None = auto-detect via XDG


class UserConfig(BaseModel):
    """User identity configuration."""

    name: str | None = None  # e.g. "Idris Kalmar" — resolved when name == "me"


class NotionSyncConfig(BaseModel):
    """Notion sync configuration."""

    space_id: str | None = None  # Notion workspace ID (overrides auto-detection)


class GranolaSyncConfig(BaseModel):
    """Granola sync configuration."""

    token_path: str | None = None  # Path to supabase.json (overrides default)


class SyncConfig(BaseModel):
    """Sync plugin configuration."""

    notion: NotionSyncConfig = NotionSyncConfig()
    granola: GranolaSyncConfig = GranolaSyncConfig()


class KbxConfig(BaseModel):
    """Top-level kbx configuration."""

    sources: SourcesConfig = SourcesConfig()
    data: DataConfig = DataConfig()
    user: UserConfig = UserConfig()
    sync: SyncConfig = SyncConfig()


def find_config(cwd: Path | None = None) -> Path | None:
    """Find the config file using the search order.

    Returns the path to the config file, or None if not found.
    """
    # 1. Explicit env var
    env = os.environ.get("KBX_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        return None

    # 2. Local kbx.toml (walk up from cwd)
    start = cwd or Path.cwd()
    for d in [start, *start.parents]:
        candidate = d / "kbx.toml"
        if candidate.is_file():
            return candidate

    # 3. XDG config home
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    xdg_config = Path(xdg) / "kbx" / "config.toml"
    if xdg_config.is_file():
        return xdg_config

    return None


def load_config(config_path: Path) -> KbxConfig:
    """Load config from a TOML file, merging with defaults."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return KbxConfig(**data)


def resolve_data_dir(cfg: KbxConfig, config_path: Path | None) -> Path:
    """Resolve the data directory from config + env vars + default.

    Priority:
      1. $KB_DATA_DIR env var (backwards compat)
      2. data.dir from config file (relative to config file location)
      3. ~/.config/kbx/ (default)
    """
    # 1. Env var (backwards compat)
    env = os.environ.get("KB_DATA_DIR")
    if env:
        return Path(env).resolve()

    # 2. Config file
    if cfg.data.dir is not None:
        p = Path(cfg.data.dir)
        if not p.is_absolute() and config_path is not None:
            p = config_path.parent / p
        return p.resolve()

    # 3. Default
    return (Path.home() / ".config" / "kbx").resolve()


def resolve_source_dir(cfg: KbxConfig, key: str, config_path: Path | None) -> Path:
    """Resolve a source directory (meetings or memory) from config.

    Relative paths are resolved against the config file's parent directory.
    """
    raw: str = getattr(cfg.sources, key)
    p = Path(raw)
    if not p.is_absolute() and config_path is not None:
        p = config_path.parent / p
    return p.resolve()
