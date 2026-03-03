"""Tests for kbx correct — find-and-replace across memory files."""

from __future__ import annotations

from pathlib import Path

import pytest

from kb.correct import apply_corrections, enrich_matches, scan


@pytest.fixture
def memory_tree(tmp_path: Path) -> Path:
    """Create a small memory/ tree with known content for correction tests."""
    memory = tmp_path / "memory"

    # meetings
    meetings_dir = memory / "meetings" / "2026" / "02" / "16"
    meetings_dir.mkdir(parents=True)
    (meetings_dir / "Platform_Stability.granola.transcript.md").write_text(
        "---\ntitle: Platform Stability\ndate: '2026-02-16'\n"
        "attendees:\n- name: Idris Kalmar\n  email: idris@example.com\n"
        "- name: Pavel Panfilov\n  email: pavel@example.com\n---\n"
        "We need to migrate from Quartz Indexer to the new stack.\n"
        "The Quartz Indexer dashboard has been unreliable.\n",
        encoding="utf-8",
    )
    (meetings_dir / "Platform_Stability.granola.ai-summary.md").write_text(
        "---\ntitle: Platform Stability\ndate: '2026-02-16'\n---\n"
        "## Summary\n- Quartz Indexer migration discussed\n- corelogix alerts need fixing\n",
        encoding="utf-8",
    )

    # meeting with Bram/Bram ambiguity
    arno_dir = memory / "meetings" / "2026" / "02" / "17"
    arno_dir.mkdir(parents=True)
    (arno_dir / "Monthly_Jeremy___Arnault.notion.transcript.md").write_text(
        "---\ntitle: Monthly Jeremy & Bram\ndate: '2026-02-17'\n"
        "attendees:\n- name: Idris Kalmar\n  email: idris@example.com\n"
        "- name: Bram Chazareix\n  email: arnault@example.com\n---\n"
        "Hey Bram, how are you?\n"
        "Bram is one of the top performers.\n",
        encoding="utf-8",
    )

    # notes
    notes_dir = memory / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "initiatives.md").write_text(
        "---\ntitle: Active Initiatives\ntags: [initiatives]\n---\n"
        "# Initiatives\n- Coralogix migration on track\n",
        encoding="utf-8",
    )

    # people entity
    people_dir = memory / "people"
    people_dir.mkdir(parents=True)
    (people_dir / "soren-vance.md").write_text(
        "---\nemail: eric@example.com\nrole: CEO\n---\n"
        "# Soren Vance\n\nDiscussed Quartz Indexer vendor evaluation.\n",
        encoding="utf-8",
    )

    # project entity
    projects_dir = memory / "projects"
    projects_dir.mkdir(parents=True)
    (projects_dir / "observability-migration.md").write_text(
        "---\nname: Observability Migration\n---\n"
        "# Observability Migration\n\n"
        "Migrating from Quartz Indexer to Coralogix.\n"
        "Granola garbles Coralogix as Quartz Indexer or Chorologix.\n",
        encoding="utf-8",
    )

    # glossary
    (memory / "glossary.md").write_text(
        "# Glossary\n\nCoralogix=Observability platform\n",
        encoding="utf-8",
    )

    return memory


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


class TestScan:
    def test_scan_finds_all_occurrences(self, memory_tree: Path):
        """scan() returns all files containing the term."""
        matches = scan(memory_tree, "Quartz Indexer")
        assert len(matches) > 0
        # Should find in transcript, summary, people, projects
        paths = {m.rel_path for m in matches}
        assert any("Platform_Stability.granola.transcript.md" in p for p in paths)
        assert any("soren-vance.md" in p for p in paths)
        assert any("observability-migration.md" in p for p in paths)

    def test_scan_counts_occurrences(self, memory_tree: Path):
        """Each match should have correct occurrence count."""
        matches = scan(memory_tree, "Quartz Indexer")
        transcript_match = [m for m in matches if "transcript.md" in m.rel_path]
        assert transcript_match
        assert transcript_match[0].count == 2  # two occurrences in transcript

    def test_scan_case_sensitive_by_default(self, memory_tree: Path):
        """Default scan is case-sensitive."""
        matches = scan(memory_tree, "Quartz Indexer")
        # The summary has "corelogix" (lowercase) which shouldn't match
        summary_match = [m for m in matches if "ai-summary.md" in m.rel_path]
        assert summary_match
        assert summary_match[0].count == 1  # only "Quartz Indexer", not "corelogix"

    def test_scan_ignore_case(self, memory_tree: Path):
        """ignore_case=True matches all case variants."""
        matches = scan(memory_tree, "Quartz Indexer", ignore_case=True)
        summary_match = [m for m in matches if "ai-summary.md" in m.rel_path]
        assert summary_match
        assert summary_match[0].count == 2  # "Quartz Indexer" + "corelogix"

    def test_scan_word_boundary(self, memory_tree: Path):
        """word_boundary=True avoids substring matches."""
        matches = scan(memory_tree, "Bram", word_boundary=True)
        arno_match = [m for m in matches if "Bram" in m.rel_path]
        assert arno_match
        assert arno_match[0].count == 2  # "Bram" appears twice as whole word

    def test_scan_scope_glob(self, memory_tree: Path):
        """scope limits scan to matching files."""
        matches = scan(memory_tree, "Quartz Indexer", scope="**/people/*")
        assert len(matches) == 1
        assert "soren-vance.md" in matches[0].rel_path

    def test_scan_scope_specific_file(self, memory_tree: Path):
        """scope can target a specific relative path."""
        matches = scan(
            memory_tree,
            "Quartz Indexer",
            scope="projects/observability-migration.md",
        )
        assert len(matches) == 1
        assert "observability-migration.md" in matches[0].rel_path

    def test_scan_file_type_filter(self, memory_tree: Path):
        """file_type filters by file suffix pattern."""
        matches = scan(memory_tree, "Quartz Indexer", file_type="transcript")
        assert all("transcript.md" in m.rel_path for m in matches)

    def test_scan_no_matches(self, memory_tree: Path):
        """scan() returns empty list when no matches found."""
        matches = scan(memory_tree, "ZZZnonexistent")
        assert matches == []

    def test_scan_returns_sample_lines(self, memory_tree: Path):
        """Each match includes sample context lines."""
        matches = scan(memory_tree, "Quartz Indexer")
        for m in matches:
            assert len(m.sample_lines) > 0
            assert any("Quartz Indexer" in line for line in m.sample_lines)


# ---------------------------------------------------------------------------
# Apply tests
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_replaces_in_files(self, memory_tree: Path):
        """apply_corrections() modifies files on disk."""
        matches = scan(memory_tree, "Quartz Indexer")
        result = apply_corrections(memory_tree, matches, "Coralogix")
        assert result.files_changed > 0
        assert result.occurrences_replaced > 0

        # Verify file content changed
        transcript = (
            memory_tree
            / "meetings"
            / "2026"
            / "02"
            / "16"
            / "Platform_Stability.granola.transcript.md"
        )
        content = transcript.read_text(encoding="utf-8")
        assert "Quartz Indexer" not in content
        assert "Coralogix" in content

    def test_apply_preserves_frontmatter(self, memory_tree: Path):
        """apply_corrections() doesn't corrupt YAML frontmatter."""
        matches = scan(memory_tree, "Quartz Indexer")
        apply_corrections(memory_tree, matches, "Coralogix")

        transcript = (
            memory_tree
            / "meetings"
            / "2026"
            / "02"
            / "16"
            / "Platform_Stability.granola.transcript.md"
        )
        content = transcript.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "title: Platform Stability" in content

    def test_apply_ignore_case_preserves_replacement(self, memory_tree: Path):
        """With ignore_case, all variants are replaced with the exact new_term."""
        matches = scan(memory_tree, "Quartz Indexer", ignore_case=True)
        apply_corrections(
            memory_tree,
            matches,
            "Coralogix",
            ignore_case=True,
        )

        summary = (
            memory_tree
            / "meetings"
            / "2026"
            / "02"
            / "16"
            / "Platform_Stability.granola.ai-summary.md"
        )
        content = summary.read_text(encoding="utf-8")
        assert "Quartz Indexer" not in content
        assert "corelogix" not in content
        assert content.count("Coralogix") >= 2

    def test_apply_word_boundary(self, memory_tree: Path):
        """word_boundary replacement only affects whole-word matches."""
        matches = scan(memory_tree, "Bram", word_boundary=True)
        result = apply_corrections(
            memory_tree,
            matches,
            "Bram",
            word_boundary=True,
        )
        assert result.occurrences_replaced > 0

        arno_file = (
            memory_tree
            / "meetings"
            / "2026"
            / "02"
            / "17"
            / "Monthly_Jeremy___Arnault.notion.transcript.md"
        )
        content = arno_file.read_text(encoding="utf-8")
        assert "Hey Bram" in content
        assert "Bram is one of the top" in content

    def test_apply_returns_accurate_counts(self, memory_tree: Path):
        """CorrectionResult has correct file and occurrence counts."""
        matches = scan(memory_tree, "Quartz Indexer")
        total_occurrences = sum(m.count for m in matches)
        result = apply_corrections(memory_tree, matches, "Coralogix")
        assert result.files_changed == len(matches)
        assert result.occurrences_replaced == total_occurrences

    def test_apply_no_matches_is_noop(self, memory_tree: Path):
        """apply_corrections() with empty matches changes nothing."""
        result = apply_corrections(memory_tree, [], "Coralogix")
        assert result.files_changed == 0
        assert result.occurrences_replaced == 0

    def test_apply_logs_changed_files(self, memory_tree: Path):
        """CorrectionResult includes list of changed file paths."""
        matches = scan(memory_tree, "Quartz Indexer")
        result = apply_corrections(memory_tree, matches, "Coralogix")
        assert len(result.changed_paths) == result.files_changed
        assert all(isinstance(p, str) for p in result.changed_paths)


# ---------------------------------------------------------------------------
# CorrectionMatch model tests
# ---------------------------------------------------------------------------


class TestCorrectionMatch:
    def test_match_fields(self, memory_tree: Path):
        """CorrectionMatch has required fields."""
        matches = scan(memory_tree, "Quartz Indexer")
        m = matches[0]
        assert isinstance(m.rel_path, str)
        assert isinstance(m.count, int)
        assert isinstance(m.sample_lines, list)
        assert m.count > 0


# ---------------------------------------------------------------------------
# Enrichment tests (frontmatter metadata for agent workflows)
# ---------------------------------------------------------------------------


class TestEnrichMatches:
    def test_enrich_adds_title(self, memory_tree: Path):
        """enrich_matches() extracts title from frontmatter."""
        matches = scan(memory_tree, "Quartz Indexer")
        enriched = enrich_matches(memory_tree, matches)
        transcript = [e for e in enriched if "transcript.md" in e["rel_path"]]
        assert transcript
        assert transcript[0]["title"] == "Platform Stability"

    def test_enrich_adds_date(self, memory_tree: Path):
        """enrich_matches() extracts date from frontmatter."""
        matches = scan(memory_tree, "Quartz Indexer")
        enriched = enrich_matches(memory_tree, matches)
        transcript = [e for e in enriched if "transcript.md" in e["rel_path"]]
        assert transcript
        assert transcript[0]["date"] == "2026-02-16"

    def test_enrich_adds_attendees(self, memory_tree: Path):
        """enrich_matches() extracts attendees from meeting frontmatter."""
        matches = scan(memory_tree, "Bram", word_boundary=True)
        enriched = enrich_matches(memory_tree, matches)
        arno = [e for e in enriched if "Bram" in e["rel_path"]]
        assert arno
        attendees = arno[0]["attendees"]
        assert len(attendees) == 2
        names = [a["name"] for a in attendees]
        assert "Bram Chazareix" in names

    def test_enrich_no_attendees_for_non_meetings(self, memory_tree: Path):
        """Non-meeting files have empty attendees list."""
        matches = scan(memory_tree, "Quartz Indexer")
        enriched = enrich_matches(memory_tree, matches)
        people_match = [e for e in enriched if "soren-vance.md" in e["rel_path"]]
        assert people_match
        assert people_match[0]["attendees"] == []

    def test_enrich_preserves_match_fields(self, memory_tree: Path):
        """Enriched dicts include all original CorrectionMatch fields."""
        matches = scan(memory_tree, "Quartz Indexer")
        enriched = enrich_matches(memory_tree, matches)
        for e in enriched:
            assert "rel_path" in e
            assert "count" in e
            assert "sample_lines" in e
            assert "search_term" in e

    def test_enrich_includes_file_category(self, memory_tree: Path):
        """Enriched dicts include a category field (meeting/note/entity/other)."""
        matches = scan(memory_tree, "Quartz Indexer")
        enriched = enrich_matches(memory_tree, matches)
        categories = {e["category"] for e in enriched}
        assert "meeting" in categories
        assert "entity" in categories
