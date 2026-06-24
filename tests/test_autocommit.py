"""Tests for auto-commit on CLI writes (issue #1)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from kb.user_config import WritesConfig


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    _git(["init"], root)
    _git(["config", "user.email", "t@example.com"], root)
    _git(["config", "user.name", "Test"], root)
    mem = root / "memory" / "notes"
    mem.mkdir(parents=True)
    (mem / "n.md").write_text("initial\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-m", "initial"], root)


def _head_subject(root: Path) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=root, capture_output=True, text=True
    )
    return out.stdout.strip()


def _commit_count(root: Path) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return int(out.stdout.strip())


class TestMaybeAutoCommit:
    def test_commits_when_enabled(self, tmp_path: Path):
        from kb.autocommit import maybe_auto_commit

        _init_repo(tmp_path)
        (tmp_path / "memory" / "notes" / "n.md").write_text("changed\n", encoding="utf-8")
        sha = maybe_auto_commit(
            tmp_path,
            operation="add-fact",
            target="Person A",
            config=WritesConfig(auto_commit=True),
            no_commit=False,
        )
        assert sha is not None
        assert _commit_count(tmp_path) == 2
        assert _head_subject(tmp_path) == "kbx: add-fact Person A"

    def test_no_commit_when_disabled(self, tmp_path: Path):
        from kb.autocommit import maybe_auto_commit

        _init_repo(tmp_path)
        (tmp_path / "memory" / "notes" / "n.md").write_text("changed\n", encoding="utf-8")
        sha = maybe_auto_commit(
            tmp_path,
            operation="add-fact",
            target="Person A",
            config=WritesConfig(auto_commit=False),
            no_commit=False,
        )
        assert sha is None
        assert _commit_count(tmp_path) == 1

    def test_no_commit_when_no_commit_flag(self, tmp_path: Path):
        from kb.autocommit import maybe_auto_commit

        _init_repo(tmp_path)
        (tmp_path / "memory" / "notes" / "n.md").write_text("changed\n", encoding="utf-8")
        sha = maybe_auto_commit(
            tmp_path,
            operation="add-fact",
            target="P",
            config=WritesConfig(auto_commit=True),
            no_commit=True,
        )
        assert sha is None
        assert _commit_count(tmp_path) == 1

    def test_no_commit_when_not_a_git_repo(self, tmp_path: Path):
        from kb.autocommit import maybe_auto_commit

        (tmp_path / "memory").mkdir()
        sha = maybe_auto_commit(
            tmp_path,
            operation="add-fact",
            target="P",
            config=WritesConfig(auto_commit=True),
            no_commit=False,
        )
        assert sha is None  # graceful — no exception when not a repo

    def test_returns_none_when_no_changes(self, tmp_path: Path):
        from kb.autocommit import maybe_auto_commit

        _init_repo(tmp_path)
        sha = maybe_auto_commit(
            tmp_path,
            operation="add-fact",
            target="P",
            config=WritesConfig(auto_commit=True),
            no_commit=False,
        )
        assert sha is None
        assert _commit_count(tmp_path) == 1

    def test_only_stages_memory_dir(self, tmp_path: Path):
        """A dirty file outside memory/ must not be swept into the auto-commit."""
        from kb.autocommit import maybe_auto_commit

        _init_repo(tmp_path)
        (tmp_path / "memory" / "notes" / "n.md").write_text("changed\n", encoding="utf-8")
        (tmp_path / "unrelated.txt").write_text("dirty\n", encoding="utf-8")
        sha = maybe_auto_commit(
            tmp_path,
            operation="add-fact",
            target="P",
            config=WritesConfig(auto_commit=True),
            no_commit=False,
        )
        assert sha is not None
        # unrelated.txt stays untracked/unstaged after the commit
        status = subprocess.run(
            ["git", "status", "--porcelain", "unrelated.txt"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        ).stdout
        assert "unrelated.txt" in status  # still dirty, not committed
