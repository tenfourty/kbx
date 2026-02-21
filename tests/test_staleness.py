"""Tests for staleness detection and auto-reindex."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from kb.db import Database


@pytest.fixture
def stale_env():
    """Create a project with indexed memory files, then yield for modification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "memory" / "people").mkdir(parents=True)
        (root / "memory" / "projects").mkdir(parents=True)
        person = root / "memory" / "people" / "jane-doe.md"
        person.write_text("# Jane Doe\n\n**Role:** Engineer\n")

        db = Database(root / "data")

        # Run initial index
        from kb.indexer import index_all

        index_all(db, None, root)

        yield db, root, person
        db.close()


class TestStalenessDetection:
    def test_detects_modified_file(self, stale_env):
        from kb.staleness import find_stale_sources

        db, root, person = stale_env
        # Modify the file after indexing with mtime in the future
        person.write_text("# Jane Doe\n\n**Role:** Staff Engineer\n")
        future_time = time.time() + 2
        os.utime(person, (future_time, future_time))
        stale = find_stale_sources(db, root)
        assert len(stale) > 0

    def test_no_stale_when_unchanged(self, stale_env):
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        stale = find_stale_sources(db, root)
        assert len(stale) == 0

    def test_ignores_new_file(self, stale_env):
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        # New files not yet indexed are ignored (need full `kb index run`)
        new_person = root / "memory" / "people" / "new-person.md"
        new_person.write_text("# New Person\n")
        stale = find_stale_sources(db, root)
        assert not any("new-person" in s for s in stale)

    def test_auto_reindex_updates_data(self, stale_env):
        from kb.staleness import auto_reindex_if_stale

        db, root, person = stale_env
        person.write_text("# Jane Doe\n\n**Role:** Staff Engineer\n")
        future_time = time.time() + 2
        os.utime(person, (future_time, future_time))
        reindexed = auto_reindex_if_stale(db, root)
        assert reindexed > 0

    def test_auto_reindex_returns_zero_when_clean(self, stale_env):
        from kb.staleness import auto_reindex_if_stale

        db, root, _person = stale_env
        reindexed = auto_reindex_if_stale(db, root)
        assert reindexed == 0
