"""Shared fixtures for kb tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Offline-only: never hit huggingface.co during tests.
#
# HF_HUB_OFFLINE=1 tells huggingface_hub to only use locally cached files.
# SOCKS proxy vars are stripped to prevent httpx from trying a SOCKS
# transport (socksio isn't installed).
# ---------------------------------------------------------------------------
_PROXY_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
    "GRPC_PROXY",
    "grpc_proxy",
    "RSYNC_PROXY",
)


@pytest.fixture(autouse=True, scope="session")
def _offline_and_no_socks_proxy():
    saved = {k: os.environ.pop(k) for k in _PROXY_VARS if k in os.environ}
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    yield
    os.environ.update(saved)
    if old_offline is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = old_offline


def _model_cached() -> bool:
    """Return True if Qwen3-Embedding-0.6B is in the local HF cache."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    return (hf_cache / "models--Qwen--Qwen3-Embedding-0.6B").is_dir()


requires_model = pytest.mark.skipif(
    not _model_cached(),
    reason="Qwen3-Embedding-0.6B not in local HF cache (run `kbx index run` once to download)",
)
"""Decorator for tests that need the real embedding model cached locally."""


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_db():
    """Empty test database in a temporary directory."""
    from kb.db import Database

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        yield db, Path(tmpdir)
        db.close()


def invoke_cli(runner: Any, args: list[str], db_path: Any) -> Any:
    """Invoke the CLI with KB_DATA_DIR env var pointing to test db."""
    from kb.cli import cli

    env = {"KB_DATA_DIR": str(db_path)}
    return runner.invoke(cli, args, env=env, catch_exceptions=False)
