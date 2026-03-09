"""Shared fixtures for kb tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Strip SOCKS proxy env vars that break huggingface_hub's httpx client.
# Embedding tests use locally-cached models and never need network access.
# ---------------------------------------------------------------------------
_PROXY_VARS = ("ALL_PROXY", "all_proxy", "FTP_PROXY", "ftp_proxy",
               "GRPC_PROXY", "grpc_proxy", "RSYNC_PROXY")


@pytest.fixture(autouse=True, scope="session")
def _strip_socks_proxy():
    saved = {k: os.environ.pop(k) for k in _PROXY_VARS if k in os.environ}
    yield
    os.environ.update(saved)


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
