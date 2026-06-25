"""Tests for the entity↔document suppression store (kbx #35)."""

from __future__ import annotations

from pathlib import Path


class TestSuppressionStore:
    def test_load_missing_returns_empty(self, tmp_path: Path):
        from kb.suppressions import load_suppressions

        assert load_suppressions(tmp_path) == {}

    def test_add_then_load_normalized(self, tmp_path: Path):
        from kb.suppressions import add_suppression, load_suppressions

        add_suppression(tmp_path, "memory/meetings/x.md", "Jérémy Cotineau")
        s = load_suppressions(tmp_path)
        assert "memory/meetings/x.md" in s
        # entity names are normalized to lowercase for matching
        assert "jérémy cotineau" in s["memory/meetings/x.md"]

    def test_add_is_idempotent(self, tmp_path: Path):
        from kb.suppressions import add_suppression, load_suppressions

        add_suppression(tmp_path, "d.md", "Anders")
        add_suppression(tmp_path, "d.md", "Anders")
        assert load_suppressions(tmp_path)["d.md"] == {"anders"}

    def test_remove_case_insensitive(self, tmp_path: Path):
        from kb.suppressions import add_suppression, load_suppressions, remove_suppression

        add_suppression(tmp_path, "d.md", "Anders")
        assert remove_suppression(tmp_path, "d.md", "anders") is True
        assert load_suppressions(tmp_path) == {}

    def test_remove_missing_returns_false(self, tmp_path: Path):
        from kb.suppressions import remove_suppression

        assert remove_suppression(tmp_path, "d.md", "Nobody") is False

    def test_corrupt_file_is_ignored(self, tmp_path: Path):
        from kb.suppressions import _store_path, load_suppressions

        p = _store_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ not json", encoding="utf-8")
        assert load_suppressions(tmp_path) == {}

    def test_multiple_entities_one_doc(self, tmp_path: Path):
        from kb.suppressions import add_suppression, load_suppressions

        add_suppression(tmp_path, "d.md", "Anders")
        add_suppression(tmp_path, "d.md", "Kit")
        assert load_suppressions(tmp_path)["d.md"] == {"anders", "kit"}
