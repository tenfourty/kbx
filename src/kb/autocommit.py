"""Auto-commit memory writes to git (issue #1, CLI-only).

Off by default. When ``WritesConfig.auto_commit`` is set and the project is a git
repo, ``maybe_auto_commit`` makes a single commit per CLI write command, scoped to
the ``memory/`` directory. CLI-only by construction: only ``cli.py`` calls this —
MCP writes never do, so the unattended sync pipeline stays out of git history.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kb.user_config import WritesConfig


class _SafeDict(dict[str, object]):
    """format_map mapping that renders unknown placeholders as empty strings."""

    def __missing__(self, key: str) -> str:
        return ""


def _is_git_repo(root: Path) -> bool:
    try:
        subprocess.run(  # nosec B607 — git is a standard system utility; fixed args, no shell
            ["git", "-C", str(root), "rev-parse", "--git-dir"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _format_message(fmt: str, operation: str, target: str, file_count: int) -> str:
    mapping = _SafeDict(
        operation=operation,
        target=target,
        file_count=file_count,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command="",
    )
    return fmt.format_map(mapping).strip()


def maybe_auto_commit(
    project_root: Path,
    *,
    operation: str,
    target: str,
    config: WritesConfig,
    no_commit: bool,
    file_count: int = 1,
) -> str | None:
    """Commit ``memory/`` changes after a CLI write, when enabled. Returns the SHA or None.

    No-ops (returns None) when auto-commit is off, ``--no-commit`` is set, the project
    is not a git repo, or nothing under ``memory/`` is staged. Never raises on git
    trouble — a failed auto-commit must not fail the user's write.
    """
    if not config.auto_commit or no_commit:
        return None
    if not _is_git_repo(project_root):
        return None

    root = str(project_root)
    memory = str(project_root / "memory")
    try:
        # Stage only memory/ — never sweep unrelated working-tree edits into the commit.
        subprocess.run(  # nosec B607
            ["git", "-C", root, "add", "--", memory], capture_output=True, check=True
        )
        # Nothing staged under memory/? (returncode 0 == no diff) → nothing to commit.
        if (
            subprocess.run(  # nosec B607
                ["git", "-C", root, "diff", "--cached", "--quiet", "--", memory]
            ).returncode
            == 0
        ):
            return None
        message = _format_message(config.auto_commit_message_format, operation, target, file_count)
        subprocess.run(  # nosec B607
            [
                "git",
                "-C",
                root,
                "commit",
                "--author",
                config.auto_commit_author,
                "-m",
                message,
                "--",
                memory,
            ],
            capture_output=True,
            check=True,
        )
        return subprocess.run(  # nosec B607
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
