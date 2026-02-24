"""Tests for the KnowledgeBase API."""

from __future__ import annotations

import pytest


@pytest.fixture
def project_root(tmp_path):
    """Create a minimal project root with memory/ directory."""
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "glossary.md").write_text("# Glossary\n")
    return tmp_path


@pytest.fixture
def kb_instance(tmp_path, project_root):
    """Create a KnowledgeBase instance with isolated data dir."""
    from kb.api import KnowledgeBase

    data_dir = tmp_path / "data"
    kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
    yield kb
    kb.close()


class TestLifecycle:
    """Constructor, close, context manager."""

    def test_constructor_creates_db(self, tmp_path, project_root):
        from kb.api import KnowledgeBase

        data_dir = tmp_path / "data"
        kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
        assert (data_dir / "metadata.db").exists()
        kb.close()

    def test_context_manager(self, tmp_path, project_root):
        from kb.api import KnowledgeBase

        data_dir = tmp_path / "data"
        with KnowledgeBase(project_root=project_root, data_dir=data_dir) as kb:
            assert (data_dir / "metadata.db").exists()
        # After exit, db should be closed (no error on double-close)
        kb.close()

    def test_thread_safe_mode(self, tmp_path, project_root):
        from kb.api import KnowledgeBase

        data_dir = tmp_path / "data"
        kb = KnowledgeBase(project_root=project_root, data_dir=data_dir, thread_safe=True)
        conn = kb._db.get_sqlite_conn()
        # Verify WAL mode is enabled
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        kb.close()

    def test_import_from_package(self):
        from kb import KnowledgeBase

        assert KnowledgeBase is not None

    def test_count_documents_empty(self, kb_instance):
        assert kb_instance.count_documents() == 0
