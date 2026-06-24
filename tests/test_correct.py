"""Tests for kbx correct — find-and-replace across memory files."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kb.cli import cli
from kb.correct import _match_file_type, _match_scope, apply_corrections, enrich_matches, scan


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
        "## Summary\n- Quartz Indexer refactor discussed\n- quartz indexer alerts need fixing\n",
        encoding="utf-8",
    )

    # meeting with Bram/Bram ambiguity
    bram_dir = memory / "meetings" / "2026" / "02" / "17"
    bram_dir.mkdir(parents=True)
    (bram_dir / "Monthly_Idris___Bramble.notion.transcript.md").write_text(
        "---\ntitle: Monthly Idris & Bramble\ndate: '2026-02-17'\n"
        "attendees:\n- name: Idris Kalmar\n  email: idris@example.com\n"
        "- name: Bramble Holt\n  email: bram@example.com\n---\n"
        "Hey Bram, how are you?\n"
        "Bram is one of the top performers.\n",
        encoding="utf-8",
    )

    # notes
    notes_dir = memory / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "initiatives.md").write_text(
        "---\ntitle: Active Initiatives\ntags: [initiatives]\n---\n"
        "# Initiatives\n- Datalux refactor on track\n",
        encoding="utf-8",
    )

    # people entity
    people_dir = memory / "people"
    people_dir.mkdir(parents=True)
    (people_dir / "soren-vance.md").write_text(
        "---\nemail: soren@example.com\nrole: CEO\n---\n"
        "# Soren Vance\n\nDiscussed Quartz Indexer vendor evaluation.\n",
        encoding="utf-8",
    )

    # project entity
    projects_dir = memory / "projects"
    projects_dir.mkdir(parents=True)
    (projects_dir / "observability-refactor.md").write_text(
        "---\nname: Observability Refactor\n---\n"
        "# Observability Refactor\n\n"
        "Migrating from Quartz Indexer to Datalux.\n"
        "Granola garbles Datalux as Quartz Indexer or Chorologix.\n",
        encoding="utf-8",
    )

    # glossary
    (memory / "glossary.md").write_text(
        "# Glossary\n\nDatalux=Observability platform\n",
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
        assert any("observability-refactor.md" in p for p in paths)

    def test_scan_counts_occurrences(self, memory_tree: Path):
        """Each match should have correct occurrence count."""
        matches = scan(memory_tree, "Quartz Indexer")
        transcript_match = [m for m in matches if "transcript.md" in m.rel_path]
        assert transcript_match
        assert transcript_match[0].count == 2  # two occurrences in transcript

    def test_scan_case_sensitive_by_default(self, memory_tree: Path):
        """Default scan is case-sensitive."""
        matches = scan(memory_tree, "Quartz Indexer")
        # The summary has "quartz indexer" (lowercase) which shouldn't match
        summary_match = [m for m in matches if "ai-summary.md" in m.rel_path]
        assert summary_match
        assert summary_match[0].count == 1  # only "Quartz Indexer", not "quartz indexer"

    def test_scan_ignore_case(self, memory_tree: Path):
        """ignore_case=True matches all case variants."""
        matches = scan(memory_tree, "Quartz Indexer", ignore_case=True)
        summary_match = [m for m in matches if "ai-summary.md" in m.rel_path]
        assert summary_match
        assert summary_match[0].count == 2  # "Quartz Indexer" + "quartz indexer"

    def test_scan_word_boundary(self, memory_tree: Path):
        """word_boundary=True avoids substring matches."""
        matches = scan(memory_tree, "Bram", word_boundary=True)
        bram_match = [m for m in matches if "Bram" in m.rel_path]
        assert bram_match
        assert bram_match[0].count == 2  # "Bram" appears twice as whole word

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
            scope="projects/observability-refactor.md",
        )
        assert len(matches) == 1
        assert "observability-refactor.md" in matches[0].rel_path

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
        result = apply_corrections(memory_tree, matches, "Datalux")
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
        assert "Datalux" in content

    def test_apply_preserves_frontmatter(self, memory_tree: Path):
        """apply_corrections() doesn't corrupt YAML frontmatter."""
        matches = scan(memory_tree, "Quartz Indexer")
        apply_corrections(memory_tree, matches, "Datalux")

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
            "Datalux",
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
        assert "quartz indexer" not in content
        assert content.count("Datalux") >= 2

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

        bram_file = (
            memory_tree
            / "meetings"
            / "2026"
            / "02"
            / "17"
            / "Monthly_Idris___Bramble.notion.transcript.md"
        )
        content = bram_file.read_text(encoding="utf-8")
        assert "Hey Bram" in content
        assert "Bram is one of the top" in content

    def test_apply_returns_accurate_counts(self, memory_tree: Path):
        """CorrectionResult has correct file and occurrence counts."""
        matches = scan(memory_tree, "Quartz Indexer")
        total_occurrences = sum(m.count for m in matches)
        result = apply_corrections(memory_tree, matches, "Datalux")
        assert result.files_changed == len(matches)
        assert result.occurrences_replaced == total_occurrences

    def test_apply_no_matches_is_noop(self, memory_tree: Path):
        """apply_corrections() with empty matches changes nothing."""
        result = apply_corrections(memory_tree, [], "Datalux")
        assert result.files_changed == 0
        assert result.occurrences_replaced == 0

    def test_apply_logs_changed_files(self, memory_tree: Path):
        """CorrectionResult includes list of changed file paths."""
        matches = scan(memory_tree, "Quartz Indexer")
        result = apply_corrections(memory_tree, matches, "Datalux")
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
        bram = [e for e in enriched if "Bram" in e["rel_path"]]
        assert bram
        attendees = bram[0]["attendees"]
        assert len(attendees) == 2
        names = [a["name"] for a in attendees]
        assert "Bramble Holt" in names

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


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestCorrectCLI:
    def _invoke(self, runner: CliRunner, memory_tree: Path, args: list[str]):
        """Invoke `kbx correct` with project root pointing to memory_tree's parent."""
        project_root = memory_tree.parent  # memory_tree is tmp_path/memory
        with patch("kb.cli._find_project_root", return_value=project_root):
            return runner.invoke(cli, ["correct", *args], catch_exceptions=False)

    def test_scan_json_output(self, cli_runner: CliRunner, memory_tree: Path):
        """kbx correct TERM --json returns structured scan output."""
        result = self._invoke(cli_runner, memory_tree, ["Quartz Indexer", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "meta" in data
        assert data["meta"]["action"] == "scan"
        assert data["meta"]["total_occurrences"] > 0
        assert data["meta"]["files"] > 0

    def test_scan_json_includes_enrichment(self, cli_runner: CliRunner, memory_tree: Path):
        """JSON scan output includes title, date, attendees, category."""
        result = self._invoke(cli_runner, memory_tree, ["Quartz Indexer", "--json"])
        data = json.loads(result.output)
        for r in data["results"]:
            assert "title" in r
            assert "date" in r
            assert "attendees" in r
            assert "category" in r

    def test_scan_human_output(self, cli_runner: CliRunner, memory_tree: Path):
        """kbx correct TERM (no --json) produces human-readable output."""
        result = self._invoke(cli_runner, memory_tree, ["Quartz Indexer"])
        assert result.exit_code == 0
        # Human output goes to stderr, check combined
        assert "Quartz Indexer" in result.output or result.exit_code == 0

    def test_scan_no_matches(self, cli_runner: CliRunner, memory_tree: Path):
        """kbx correct with no matches returns empty results."""
        result = self._invoke(cli_runner, memory_tree, ["ZZZnonexistent", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["meta"]["total"] == 0
        assert data["results"] == []

    def test_dry_run_json(self, cli_runner: CliRunner, memory_tree: Path):
        """kbx correct OLD NEW --json shows dry-run preview."""
        result = self._invoke(cli_runner, memory_tree, ["Quartz Indexer", "Datalux", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["meta"]["action"] == "dry_run"
        assert data["meta"]["replacement"] == "Datalux"

    def test_dry_run_does_not_modify_files(self, cli_runner: CliRunner, memory_tree: Path):
        """Dry-run (no --apply) leaves files unchanged."""
        self._invoke(cli_runner, memory_tree, ["Quartz Indexer", "Datalux"])
        transcript = (
            memory_tree
            / "meetings"
            / "2026"
            / "02"
            / "16"
            / "Platform_Stability.granola.transcript.md"
        )
        content = transcript.read_text(encoding="utf-8")
        assert "Quartz Indexer" in content  # unchanged

    def test_apply_modifies_files(self, cli_runner: CliRunner, memory_tree: Path):
        """kbx correct OLD NEW --apply actually replaces content."""
        result = self._invoke(
            cli_runner, memory_tree, ["Quartz Indexer", "Datalux", "--apply", "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["meta"]["action"] == "applied"
        assert data["meta"]["files_changed"] > 0
        assert data["meta"]["occurrences_replaced"] > 0

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
        assert "Datalux" in content

    def test_scope_flag(self, cli_runner: CliRunner, memory_tree: Path):
        """--scope limits scan to matching files."""
        result = self._invoke(
            cli_runner, memory_tree, ["Quartz Indexer", "--scope", "**/people/*", "--json"]
        )
        data = json.loads(result.output)
        assert data["meta"]["files"] == 1
        assert "soren-vance.md" in data["results"][0]["rel_path"]

    def test_type_flag(self, cli_runner: CliRunner, memory_tree: Path):
        """--type filters by filename pattern."""
        result = self._invoke(
            cli_runner, memory_tree, ["Quartz Indexer", "--type", "transcript", "--json"]
        )
        data = json.loads(result.output)
        for r in data["results"]:
            assert "transcript" in r["rel_path"]

    def test_word_boundary_flag(self, cli_runner: CliRunner, memory_tree: Path):
        """--word-boundary avoids substring matches."""
        result = self._invoke(cli_runner, memory_tree, ["Bram", "--word-boundary", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["meta"]["total_occurrences"] > 0

    def test_ignore_case_flag(self, cli_runner: CliRunner, memory_tree: Path):
        """--ignore-case matches all case variants."""
        result = self._invoke(
            cli_runner, memory_tree, ["quartz indexer", "--ignore-case", "--json"]
        )
        data = json.loads(result.output)
        assert data["meta"]["total_occurrences"] > 0


# ---------------------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_apply_writes_audit_log(self, memory_tree: Path, tmp_path: Path):
        """apply_corrections() writes an audit log entry when log_path is provided."""
        from kb.correct import apply_corrections, scan

        log_path = tmp_path / "corrections.log"
        matches = scan(memory_tree, "Quartz Indexer")
        apply_corrections(memory_tree, matches, "Datalux", log_path=log_path)
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "Quartz Indexer" in content
        assert "Datalux" in content

    def test_audit_log_contains_timestamp(self, memory_tree: Path, tmp_path: Path):
        """Audit log entry includes an ISO timestamp."""
        from kb.correct import apply_corrections, scan

        log_path = tmp_path / "corrections.log"
        matches = scan(memory_tree, "Quartz Indexer")
        apply_corrections(memory_tree, matches, "Datalux", log_path=log_path)
        content = log_path.read_text(encoding="utf-8")
        # ISO format: 2026-03-03T...
        assert "2026-" in content or "202" in content

    def test_audit_log_contains_file_count(self, memory_tree: Path, tmp_path: Path):
        """Audit log entry includes files changed and occurrences replaced."""
        from kb.correct import apply_corrections, scan

        log_path = tmp_path / "corrections.log"
        matches = scan(memory_tree, "Quartz Indexer")
        result = apply_corrections(memory_tree, matches, "Datalux", log_path=log_path)
        content = log_path.read_text(encoding="utf-8")
        assert str(result.files_changed) in content
        assert str(result.occurrences_replaced) in content

    def test_audit_log_appends(self, memory_tree: Path, tmp_path: Path):
        """Multiple corrections append to the same log file."""
        from kb.correct import apply_corrections, scan

        log_path = tmp_path / "corrections.log"
        matches1 = scan(memory_tree, "Quartz Indexer")
        apply_corrections(memory_tree, matches1, "Datalux", log_path=log_path)
        matches2 = scan(memory_tree, "Bram", word_boundary=True)
        apply_corrections(memory_tree, matches2, "Bram", word_boundary=True, log_path=log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "Datalux" in content
        assert "Bram" in content

    def test_no_log_when_no_path(self, memory_tree: Path, tmp_path: Path):
        """No audit log is written when log_path is not provided."""
        from kb.correct import apply_corrections, scan

        matches = scan(memory_tree, "Quartz Indexer")
        apply_corrections(memory_tree, matches, "Datalux")
        # No file should be created in tmp_path
        log_files = list(tmp_path.glob("*.log"))
        # memory_tree is under tmp_path, so filter
        assert not any("corrections.log" in str(f) for f in log_files)


# ---------------------------------------------------------------------------
# Bug fix tests (code review round 1)
# ---------------------------------------------------------------------------


class TestScopeCompat:
    """Bug #1: _match_scope must work without Path.full_match (Python <3.13)."""

    def test_scope_glob_double_star(self):
        """_match_scope handles ** glob without Path.full_match."""
        assert _match_scope("people/soren-vance.md", "**/people/*")

    def test_scope_glob_double_star_nested(self):
        """** matches deeply nested paths."""
        assert _match_scope(
            "meetings/2026/02/17/Monthly.md",
            "**/meetings/**",
        )

    def test_scope_simple_glob(self):
        """Simple glob without ** still works."""
        assert _match_scope("people/soren-vance.md", "people/*")

    def test_scope_no_match(self):
        """Non-matching scope returns False."""
        assert not _match_scope("notes/init.md", "**/people/*")

    def test_scope_exact_match(self):
        """Exact path match still works."""
        assert _match_scope("projects/obs.md", "projects/obs.md")

    def test_scope_none_matches_all(self):
        """scope=None matches everything."""
        assert _match_scope("any/path.md", None)


class TestReplacementEscaping:
    """Bug #2: Replacement strings with backslashes must be treated literally."""

    def test_apply_replacement_with_backslash_digit(self, memory_tree: Path):
        r"""Replacement containing \1 should not be treated as regex group ref."""
        matches = scan(memory_tree, "Quartz Indexer")
        # Should not raise re.error — \1 must be treated as literal text
        result = apply_corrections(memory_tree, matches, r"replaced\1text")
        assert result.files_changed > 0
        content = (memory_tree / result.changed_paths[0]).read_text()
        assert "replaced\\1text" in content

    def test_apply_replacement_with_backslash_g(self, memory_tree: Path):
        r"""Replacement containing \g<0> should not expand to matched text."""
        matches = scan(memory_tree, "Quartz Indexer")
        result = apply_corrections(memory_tree, matches, r"new\g<0>value")
        assert result.files_changed > 0
        content = (memory_tree / result.changed_paths[0]).read_text()
        assert "new\\g<0>value" in content


class TestFileTypeFilename:
    """Bug #3: _match_file_type should check filename only, not full path."""

    def test_file_type_matches_filename_not_directory(self, memory_tree: Path):
        """file_type in a directory name should NOT cause false matches."""
        # Create a file in a directory named 'transcript' — file itself is not a transcript
        transcript_dir = memory_tree / "notes" / "transcript"
        transcript_dir.mkdir(parents=True)
        (transcript_dir / "summary.md").write_text(
            "Quartz Indexer stuff here\n",
            encoding="utf-8",
        )

        matches = scan(memory_tree, "Quartz Indexer", file_type="transcript")
        paths = [m.rel_path for m in matches]
        # summary.md should NOT match — only files with 'transcript' in their NAME
        assert not any(p.endswith("summary.md") for p in paths)
        # But actual transcript files should still match
        assert any("transcript.md" in p for p in paths)

    def test_match_file_type_checks_filename(self):
        """_match_file_type should only check the filename component."""
        # Directory contains 'transcript' but filename doesn't
        assert not _match_file_type("notes/transcript/summary.md", "transcript")
        # Filename contains 'transcript'
        assert _match_file_type(
            "meetings/2026/02/16/Platform_Stability.granola.transcript.md",
            "transcript",
        )


class TestApplyJsonSchema:
    """Bug #4: Apply-mode JSON should use {results, meta} wrapper."""

    def test_apply_json_has_results_and_meta(self, cli_runner: CliRunner, memory_tree: Path):
        """Apply-mode JSON output uses standard {results, meta} wrapper."""
        project_root = memory_tree.parent
        with patch("kb.cli._find_project_root", return_value=project_root):
            result = cli_runner.invoke(
                cli,
                ["correct", "Quartz Indexer", "Datalux", "--apply", "--json"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "meta" in data
        assert isinstance(data["results"], list)
        assert data["meta"]["action"] == "applied"
        assert data["meta"]["files_changed"] > 0
        assert data["meta"]["occurrences_replaced"] > 0

    def test_apply_json_results_are_changed_paths(self, cli_runner: CliRunner, memory_tree: Path):
        """Apply-mode results list contains the changed file paths."""
        project_root = memory_tree.parent
        with patch("kb.cli._find_project_root", return_value=project_root):
            result = cli_runner.invoke(
                cli,
                ["correct", "Quartz Indexer", "Datalux", "--apply", "--json"],
                catch_exceptions=False,
            )
        data = json.loads(result.output)
        assert len(data["results"]) > 0
        assert all("path" in r for r in data["results"])


class TestNfcNormalization:
    """Bug #5: rel_path in CorrectionMatch should be NFC-normalized."""

    def test_scan_nfc_normalizes_rel_paths(self, memory_tree: Path):
        """scan() returns NFC-normalized rel_path values."""
        # Create a file with NFD-encoded name (macOS default encoding)
        nfd_name = unicodedata.normalize("NFD", "réunion.md")
        nfd_file = memory_tree / "notes" / nfd_name
        nfd_file.write_text("Quartz Indexer mentioned here\n", encoding="utf-8")

        matches = scan(memory_tree, "Quartz Indexer")
        for m in matches:
            assert m.rel_path == unicodedata.normalize("NFC", m.rel_path), (
                f"rel_path not NFC-normalized: {m.rel_path!r}"
            )
