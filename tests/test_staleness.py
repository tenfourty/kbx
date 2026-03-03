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

    def test_detects_stale_notes(self, stale_env):
        """find_stale_sources watches memory/notes/ directory."""
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        # Create and index a note
        notes_dir = root / "memory" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        note = notes_dir / "2026-02-28-test.md"
        note.write_text("---\ntitle: Test Note\ndate: 2026-02-28\n---\nOriginal\n")

        from kb.indexer import index_all

        index_all(db, None, root, memory_only=True, skip_seed=True)

        # Modify the note
        note.write_text("---\ntitle: Test Note\ndate: 2026-02-28\n---\nModified\n")
        future_time = time.time() + 2
        os.utime(note, (future_time, future_time))

        stale = find_stale_sources(db, root)
        assert any("notes" in s for s in stale)

    def test_detects_stale_in_new_subdir(self, stale_env):
        """find_stale_sources detects changes in arbitrary memory subdirectories."""
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        # Create a new subdir (e.g. journal/) and index a file
        journal_dir = root / "memory" / "journal"
        journal_dir.mkdir(parents=True)
        entry = journal_dir / "2026-03-03-daily.md"
        entry.write_text("---\ntitle: Daily Journal\ndate: 2026-03-03\n---\nOriginal\n")

        from kb.indexer import index_all

        index_all(db, None, root, memory_only=True, skip_seed=True)

        # Modify the file
        entry.write_text("---\ntitle: Daily Journal\ndate: 2026-03-03\n---\nModified\n")
        future_time = time.time() + 2
        os.utime(entry, (future_time, future_time))

        stale = find_stale_sources(db, root)
        assert any("journal" in s for s in stale)

    def test_detects_stale_in_nested_subdir(self, stale_env):
        """find_stale_sources walks nested subdirectories recursively."""
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        # Nested path: memory/journal/daily/
        nested_dir = root / "memory" / "journal" / "daily"
        nested_dir.mkdir(parents=True)
        entry = nested_dir / "2026-03-03.md"
        entry.write_text("---\ntitle: Nested Entry\ndate: 2026-03-03\n---\nOriginal\n")

        from kb.indexer import index_all

        index_all(db, None, root, memory_only=True, skip_seed=True)

        # Modify the file
        entry.write_text("---\ntitle: Nested Entry\ndate: 2026-03-03\n---\nModified\n")
        future_time = time.time() + 2
        os.utime(entry, (future_time, future_time))

        stale = find_stale_sources(db, root)
        assert any("journal/daily" in s for s in stale)

    def test_excludes_meetings_from_staleness(self, stale_env):
        """find_stale_sources does NOT check memory/meetings/ files."""
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        # Create a meeting file and index it
        meeting_dir = root / "memory" / "meetings" / "2026" / "03" / "03"
        meeting_dir.mkdir(parents=True)
        meeting = meeting_dir / "abc12345_Test.granola.notes.md"
        meeting.write_text(
            "---\ntitle: Test Meeting\ndate: 2026-03-03\ntype: notes\n"
            "granola_id: abc12345\n---\n\n## Notes\n\nSome content.\n"
        )

        from kb.indexer import index_all

        index_all(db, None, root, skip_seed=True)

        # Modify the meeting file
        meeting.write_text(
            "---\ntitle: Test Meeting\ndate: 2026-03-03\ntype: notes\n"
            "granola_id: abc12345\n---\n\n## Notes\n\nModified content.\n"
        )
        future_time = time.time() + 2
        os.utime(meeting, (future_time, future_time))

        stale = find_stale_sources(db, root)
        assert not any("meetings" in s for s in stale)

    def test_detects_stale_glossary(self, stale_env):
        """find_stale_sources detects changes to memory/glossary.md."""
        from kb.staleness import find_stale_sources

        db, root, _person = stale_env
        glossary = root / "memory" / "glossary.md"
        glossary.write_text(
            "# Glossary\n\n| Term | Expansion |\n|---|---|\n| API | Application |\n"
        )

        from kb.indexer import index_all

        index_all(db, None, root, memory_only=True, skip_seed=True)

        # Modify the glossary
        glossary.write_text(
            "# Glossary\n\n| Term | Expansion |\n|---|---|\n| API | App Interface |\n"
        )
        future_time = time.time() + 2
        os.utime(glossary, (future_time, future_time))

        stale = find_stale_sources(db, root)
        assert any("glossary" in s for s in stale)
