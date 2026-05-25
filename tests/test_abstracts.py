"""Tests for kb.abstracts — extractive L0 abstract generation (issue #66 P1)."""

from __future__ import annotations

from kb.abstracts import extract_abstract


class TestExtractAbstract:
    def test_first_sentence_of_paragraph(self):
        """Picks the first sentence in the first body paragraph."""
        content = "First sentence here. Second sentence ignored. Third too."
        assert extract_abstract(content) == "First sentence here."

    def test_strips_yaml_frontmatter(self):
        """YAML frontmatter is discarded before sentence detection."""
        content = "---\ntitle: Notes\ndate: 2026-05-24\n---\n\nActual first sentence."
        assert extract_abstract(content) == "Actual first sentence."

    def test_strips_leading_headings(self):
        """Leading markdown headings are skipped."""
        content = "## Overview\n\nMigration is on track for Q3."
        assert extract_abstract(content) == "Migration is on track for Q3."

    def test_question_mark_terminates(self):
        """`?` is a valid sentence terminator."""
        content = "Should we ship this week? Probably yes."
        assert extract_abstract(content) == "Should we ship this week?"

    def test_exclamation_terminates(self):
        """`!` is a valid sentence terminator."""
        content = "Ship it! That decision was easy."
        assert extract_abstract(content) == "Ship it!"

    def test_no_terminator_takes_full_paragraph(self):
        """Paragraph without a sentence terminator returns the whole paragraph."""
        content = "Notes about deployment"
        assert extract_abstract(content) == "Notes about deployment"

    def test_falls_back_to_title_when_empty_content(self):
        """Empty content returns the title (when supplied)."""
        assert extract_abstract("", title="Sprint Planning") == "Sprint Planning"

    def test_falls_back_to_title_when_all_frontmatter(self):
        """Document with only YAML frontmatter falls back to the title."""
        content = "---\ntitle: Sprint Planning\n---\n"
        assert extract_abstract(content, title="Sprint Planning") == "Sprint Planning"

    def test_returns_none_when_no_content_or_title(self):
        """Empty content and empty/None title returns None."""
        assert extract_abstract("") is None
        assert extract_abstract("", title=None) is None
        assert extract_abstract("", title="   ") is None

    def test_caps_at_max_chars(self):
        """Long sentences are truncated with an ellipsis."""
        long_sentence = "x " * 200 + "."
        result = extract_abstract(long_sentence, max_chars=50)
        assert result is not None
        assert len(result) == 50
        assert result.endswith("…")

    def test_min_chars_skips_tiny_sentence(self):
        """Sentences shorter than min_chars are skipped in favour of the next paragraph."""
        content = "Hi.\n\nThe real first sentence is much longer."
        # "Hi." is 3 chars, below default min_chars=5 — should skip to next paragraph.
        assert extract_abstract(content) == "The real first sentence is much longer."

    def test_strips_wikilinks(self):
        """`[[Wiki Link]]` is replaced with its display text."""
        content = "Reports to [[Idris Kalmar]] who leads platform."
        assert extract_abstract(content) == "Reports to Idris Kalmar who leads platform."

    def test_strips_markdown_links(self):
        """`[text](url)` is replaced with just `text`."""
        content = "See [the design doc](https://example.com) for details."
        assert extract_abstract(content) == "See the design doc for details."

    def test_strips_inline_formatting(self):
        """Bold/italic/code markdown is stripped from the abstract."""
        content = "We **must** ship by *Friday* with `--enable-x`."
        assert extract_abstract(content) == "We must ship by Friday with --enable-x."

    def test_abbreviation_does_not_split(self):
        """`Mr.`, `Dr.`, etc. do not prematurely terminate the sentence."""
        content = "Dr. Smith and Mr. Jones reviewed the plan."
        assert extract_abstract(content) == "Dr. Smith and Mr. Jones reviewed the plan."

    def test_single_letter_initial_does_not_split(self):
        """Single-letter initials like `J. Smith` do not split a sentence."""
        content = "Q. asked the team to ship. Then the team did."
        # The first sentence-terminator after a single letter (`Q.`) is rejected;
        # the sentence continues to the next `.`.
        result = extract_abstract(content)
        assert result is not None
        assert "Q. asked the team to ship" in result

    def test_collapses_whitespace(self):
        """Newlines and runs of whitespace inside a paragraph collapse to single spaces."""
        content = "First   sentence\nspans   multiple    lines."
        assert extract_abstract(content) == "First sentence spans multiple lines."

    def test_title_truncated_when_used_as_fallback(self):
        """Title fallback is truncated to max_chars."""
        result = extract_abstract("", title="x" * 500, max_chars=20)
        assert result is not None
        assert len(result) == 20
        assert result.endswith("…")
