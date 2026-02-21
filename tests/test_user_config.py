"""Tests for config file discovery and loading."""

from pathlib import Path


def test_default_config():
    """Default config has sensible defaults for source paths and data dir."""
    from kb.user_config import KbxConfig

    cfg = KbxConfig()
    assert cfg.sources.meetings == "meetings/organised"
    assert cfg.sources.memory == "memory"
    assert cfg.data.dir is None  # None = auto-detect


def test_find_config_local(tmp_path):
    """Config file in current directory takes priority."""
    from kb.user_config import find_config

    (tmp_path / "kbx.toml").write_text('[sources]\nmeetings = "my-meetings"\n')
    result = find_config(cwd=tmp_path)
    assert result is not None
    assert result == tmp_path / "kbx.toml"


def test_find_config_walks_up(tmp_path):
    """Config file found by walking up from CWD."""
    from kb.user_config import find_config

    (tmp_path / "kbx.toml").write_text('[sources]\nmeetings = "my-meetings"\n')
    child = tmp_path / "sub" / "dir"
    child.mkdir(parents=True)
    result = find_config(cwd=child)
    assert result is not None
    assert result == tmp_path / "kbx.toml"


def test_find_config_xdg(tmp_path, monkeypatch):
    """Falls back to XDG config dir."""
    from kb.user_config import find_config

    config_dir = tmp_path / "config" / "kbx"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[sources]\nmemory = "notes"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # No local kbx.toml, so XDG should be found
    result = find_config(cwd=tmp_path / "some" / "other" / "dir")
    assert result is not None
    assert "config/kbx/config.toml" in str(result)


def test_find_config_env_override(tmp_path, monkeypatch):
    """KBX_CONFIG env var overrides all other discovery."""
    from kb.user_config import find_config

    config_file = tmp_path / "custom.toml"
    config_file.write_text('[sources]\nmeetings = "custom-meetings"\n')
    monkeypatch.setenv("KBX_CONFIG", str(config_file))
    result = find_config(cwd=Path("/nonexistent"))
    assert result == config_file


def test_find_config_env_override_missing_file(tmp_path, monkeypatch):
    """KBX_CONFIG pointing to non-existent file returns None."""
    from kb.user_config import find_config

    monkeypatch.setenv("KBX_CONFIG", str(tmp_path / "nonexistent.toml"))
    result = find_config(cwd=tmp_path)
    assert result is None


def test_find_config_none_when_no_config(tmp_path, monkeypatch):
    """Returns None when no config file exists anywhere."""
    from kb.user_config import find_config

    # Ensure no XDG config exists
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty_config"))
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    result = find_config(cwd=tmp_path)
    assert result is None


def test_load_config_merges_with_defaults(tmp_path):
    """Loading a partial config merges with defaults."""
    from kb.user_config import load_config

    (tmp_path / "kbx.toml").write_text('[sources]\nmeetings = "my-meetings"\n')
    cfg = load_config(tmp_path / "kbx.toml")
    assert cfg.sources.meetings == "my-meetings"
    # Non-overridden defaults persist
    assert cfg.sources.memory == "memory"


def test_load_config_full(tmp_path):
    """Loading a full config overrides all defaults."""
    from kb.user_config import load_config

    (tmp_path / "kbx.toml").write_text(
        '[sources]\nmeetings = "m"\nmemory = "n"\n\n[data]\ndir = ".db"\n'
    )
    cfg = load_config(tmp_path / "kbx.toml")
    assert cfg.sources.meetings == "m"
    assert cfg.sources.memory == "n"
    assert cfg.data.dir == ".db"


def test_resolve_data_dir_from_config(tmp_path):
    """data.dir in config resolves relative to config file location."""
    from kb.user_config import load_config, resolve_data_dir

    config_file = tmp_path / "kbx.toml"
    config_file.write_text('[data]\ndir = ".kbx"\n')
    cfg = load_config(config_file)
    data_dir = resolve_data_dir(cfg, config_file)
    assert data_dir == (tmp_path / ".kbx").resolve()


def test_resolve_data_dir_absolute_in_config(tmp_path):
    """Absolute data.dir in config used as-is."""
    from kb.user_config import load_config, resolve_data_dir

    abs_dir = str(tmp_path / "absolute" / "path")
    config_file = tmp_path / "kbx.toml"
    config_file.write_text(f'[data]\ndir = "{abs_dir}"\n')
    cfg = load_config(config_file)
    data_dir = resolve_data_dir(cfg, config_file)
    assert data_dir == Path(abs_dir).resolve()


def test_resolve_data_dir_env_override(tmp_path, monkeypatch):
    """KB_DATA_DIR env var takes precedence over config."""
    from kb.user_config import load_config, resolve_data_dir

    config_file = tmp_path / "kbx.toml"
    config_file.write_text('[data]\ndir = ".kbx"\n')
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "env-data"))
    cfg = load_config(config_file)
    data_dir = resolve_data_dir(cfg, config_file)
    assert data_dir == (tmp_path / "env-data").resolve()


def test_resolve_data_dir_default_xdg(tmp_path, monkeypatch):
    """Without config, data dir defaults to XDG data home."""
    from kb.user_config import KbxConfig, resolve_data_dir

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("KB_DATA_DIR", raising=False)
    cfg = KbxConfig()
    data_dir = resolve_data_dir(cfg, config_path=None)
    assert data_dir == tmp_path / "data" / "kbx"


def test_resolve_source_dir_relative(tmp_path):
    """Source dirs resolve relative to config file location."""
    from kb.user_config import load_config, resolve_source_dir

    config_file = tmp_path / "kbx.toml"
    config_file.write_text('[sources]\nmeetings = "my-meetings"\nmemory = "my-memory"\n')
    cfg = load_config(config_file)
    meetings_dir = resolve_source_dir(cfg, "meetings", config_file)
    memory_dir = resolve_source_dir(cfg, "memory", config_file)
    assert meetings_dir == (tmp_path / "my-meetings").resolve()
    assert memory_dir == (tmp_path / "my-memory").resolve()


def test_resolve_source_dir_no_config():
    """Source dirs without config resolve against CWD."""
    from kb.user_config import KbxConfig, resolve_source_dir

    cfg = KbxConfig()
    meetings_dir = resolve_source_dir(cfg, "meetings", config_path=None)
    # Without config_path, path is resolved from CWD
    assert meetings_dir == Path("meetings/organised").resolve()


# ---------------------------------------------------------------------------
# Integration tests: config.py uses user_config
# ---------------------------------------------------------------------------


def test_find_project_root_with_config(tmp_path, monkeypatch):
    """find_project_root() returns config parent when kbx.toml exists."""
    from kb.config import find_project_root

    config_file = tmp_path / "kbx.toml"
    config_file.write_text('[sources]\nmeetings = "meetings"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    assert find_project_root() == tmp_path


def test_find_project_root_legacy_fallback(tmp_path, monkeypatch):
    """find_project_root() falls back to legacy detection when no config."""
    from kb.config import find_project_root

    # Create legacy layout
    (tmp_path / "kb").mkdir()
    (tmp_path / "memory").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    assert find_project_root() == tmp_path


def test_find_project_root_cwd_fallback(tmp_path, monkeypatch):
    """find_project_root() falls back to CWD when nothing matches."""
    from kb.config import find_project_root

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    assert find_project_root() == tmp_path


def test_get_data_dir_with_config(tmp_path, monkeypatch):
    """get_data_dir() resolves data.dir from kbx.toml."""
    from kb.config import get_data_dir

    config_file = tmp_path / "kbx.toml"
    config_file.write_text('[data]\ndir = ".kbx-data"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    monkeypatch.delenv("KB_DATA_DIR", raising=False)
    assert get_data_dir() == (tmp_path / ".kbx-data").resolve()


def test_get_data_dir_legacy(tmp_path, monkeypatch):
    """get_data_dir() uses legacy kb/data when no config."""
    from kb.config import get_data_dir

    (tmp_path / "kb").mkdir()
    (tmp_path / "memory").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    monkeypatch.delenv("KB_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    assert get_data_dir() == tmp_path / "kb" / "data"


def test_get_data_dir_env_override_legacy(tmp_path, monkeypatch):
    """get_data_dir() respects KB_DATA_DIR in legacy mode."""
    from kb.config import get_data_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KBX_CONFIG", raising=False)
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    assert get_data_dir() == (tmp_path / "custom-data").resolve()


# ---------------------------------------------------------------------------
# CLI init command tests
# ---------------------------------------------------------------------------


def test_init_creates_local_config(tmp_path, monkeypatch):
    """kb init creates kbx.toml in current directory."""
    from click.testing import CliRunner

    from kb.cli import cli

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="meetings/organised\nmemory\n.kbx\n")
    assert result.exit_code == 0
    assert (tmp_path / "kbx.toml").exists()
    content = (tmp_path / "kbx.toml").read_text()
    assert "[sources]" in content
    assert "meetings/organised" in content
    assert "[data]" in content


def test_init_global_creates_xdg_config(tmp_path, monkeypatch):
    """kb init --global creates config in XDG config dir."""
    from click.testing import CliRunner

    from kb.cli import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--global"], input="meetings\nmemory\n.kbx\n")
    assert result.exit_code == 0
    assert (tmp_path / "config" / "kbx" / "config.toml").exists()


def test_init_does_not_overwrite(tmp_path, monkeypatch):
    """kb init refuses to overwrite existing config."""
    from click.testing import CliRunner

    from kb.cli import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "kbx.toml").write_text("[sources]\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "already exists" in result.output
