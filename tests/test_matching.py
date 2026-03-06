"""Tests for kb.matching — task-to-project matching logic."""

from __future__ import annotations

import pytest

from kb.matching import (
    TaskInput,
    _build_match_patterns,
    _parse_keywords,
    _title_matches,
    extract_project_link,
    match_tasks_to_projects,
)

# ---------------------------------------------------------------------------
# extract_project_link
# ---------------------------------------------------------------------------


class TestExtractProjectLink:
    def test_basic(self):
        assert extract_project_link("project: AI Adoption") == "ai adoption"

    def test_first_wins(self):
        desc = "project: AI Adoption\nproject: Helix Refactor"
        assert extract_project_link(desc) == "ai adoption"

    def test_case_insensitive(self):
        assert extract_project_link("Project: Foo") == "foo"
        assert extract_project_link("PROJECT: Bar") == "bar"

    def test_empty(self):
        assert extract_project_link(None) is None
        assert extract_project_link("") is None

    def test_no_match(self):
        assert extract_project_link("no project here") is None

    def test_strips_whitespace(self):
        assert extract_project_link("project:   AI Adoption  ") == "ai adoption"

    def test_multiline_finds_first(self):
        desc = "Some context\nproject: Foo Bar\nMore text\nproject: Baz"
        assert extract_project_link(desc) == "foo bar"

    def test_blank_value_returns_none(self):
        assert extract_project_link("project:   ") is None


# ---------------------------------------------------------------------------
# _parse_keywords
# ---------------------------------------------------------------------------


class TestParseKeywords:
    def test_list(self):
        assert _parse_keywords(["AI", "claude"]) == ["ai", "claude"]

    def test_stringified_list(self):
        assert _parse_keywords("['AI', 'claude']") == ["ai", "claude"]

    def test_csv(self):
        assert _parse_keywords("AI, claude") == ["ai", "claude"]

    def test_empty_string(self):
        assert _parse_keywords("") == []

    def test_none(self):
        assert _parse_keywords(None) == []

    def test_non_string_non_list(self):
        assert _parse_keywords(42) == []

    def test_strips_items(self):
        assert _parse_keywords([" AI ", " claude "]) == ["ai", "claude"]

    def test_filters_empty_items(self):
        assert _parse_keywords(["AI", "", "  "]) == ["ai"]


# ---------------------------------------------------------------------------
# _build_match_patterns
# ---------------------------------------------------------------------------


class TestBuildMatchPatterns:
    def test_name_only(self):
        patterns = _build_match_patterns("AI Adoption")
        assert ("ai adoption", False) in patterns

    def test_with_aliases(self):
        patterns = _build_match_patterns("AI Adoption", aliases=["TF Agentic AI"])
        lowered = [p for p, _ in patterns]
        assert "tf agentic ai" in lowered

    def test_with_keywords(self):
        patterns = _build_match_patterns(
            "AI Adoption", metadata={"task_keywords": ["claude", "agentic"]}
        )
        lowered = [p for p, _ in patterns]
        assert "claude" in lowered
        assert "agentic" in lowered

    def test_deduplication(self):
        patterns = _build_match_patterns(
            "Foo", aliases=["foo", "Foo"], metadata={"task_keywords": ["foo"]}
        )
        assert len(patterns) == 1

    def test_short_keyword_needs_boundary(self):
        patterns = _build_match_patterns("X", metadata={"task_keywords": ["AI"]})
        # "x" is 1 char, "ai" is 2 chars — both < 4 → need boundary
        for _pattern, needs_boundary in patterns:
            assert needs_boundary is True

    def test_long_keyword_no_boundary(self):
        patterns = _build_match_patterns("Foo", metadata={"task_keywords": ["claude"]})
        claude_entry = next(p for p in patterns if p[0] == "claude")
        assert claude_entry[1] is False

    def test_min_keyword_len_configurable(self):
        patterns = _build_match_patterns(
            "X", metadata={"task_keywords": ["AWS"]}, min_keyword_len=3
        )
        aws_entry = next(p for p in patterns if p[0] == "aws")
        assert aws_entry[1] is False  # len("aws") == 3, not < 3

    def test_empty_patterns_skipped(self):
        patterns = _build_match_patterns("", aliases=["", None], metadata={})
        assert len(patterns) == 0


# ---------------------------------------------------------------------------
# _title_matches
# ---------------------------------------------------------------------------


class TestTitleMatches:
    def test_substring_match(self):
        patterns = [("claude", False)]
        assert _title_matches("review claude setup", patterns) is True

    def test_substring_no_match(self):
        patterns = [("claude", False)]
        assert _title_matches("review cursor setup", patterns) is False

    def test_word_boundary_match(self):
        patterns = [("ai", True)]
        assert _title_matches("review ai tools", patterns) is True

    def test_word_boundary_no_false_positive(self):
        patterns = [("ai", True)]
        assert _title_matches("railway deployment", patterns) is False
        assert _title_matches("waiting for approval", patterns) is False
        assert _title_matches("email setup", patterns) is False
        assert _title_matches("kaizen retrospective", patterns) is False

    def test_word_boundary_aws(self):
        patterns = [("aws", True)]
        assert _title_matches("update aws config", patterns) is True
        assert _title_matches("outlaws of the marsh", patterns) is False
        assert _title_matches("claws of the cat", patterns) is False

    def test_word_boundary_ods(self):
        patterns = [("ods", True)]
        assert _title_matches("review ods data", patterns) is True
        assert _title_matches("methods overview", patterns) is False
        assert _title_matches("periods analysis", patterns) is False

    def test_word_boundary_mfa(self):
        patterns = [("mfa", True)]
        assert _title_matches("mfa rollout plan", patterns) is True
        assert _title_matches("review mfa implementation", patterns) is True


# ---------------------------------------------------------------------------
# match_tasks_to_projects
# ---------------------------------------------------------------------------


class TestMatchTasksToProjects:
    @pytest.fixture()
    def projects(self):
        return [
            {
                "name": "AI Adoption",
                "aliases": ["TF Agentic AI"],
                "metadata": {"task_keywords": ["AI", "claude"]},
            },
            {"name": "Helix Refactor", "aliases": [], "metadata": {}},
            {"name": "Security Policies", "aliases": [], "metadata": {"task_keywords": ["SOC2"]}},
        ]

    def test_tier1_explicit_link(self, projects):
        tasks: list[TaskInput] = [
            {"title": "Some unrelated title", "description": "project: AI Adoption"},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 1
        assert len(result["Helix Refactor"]) == 0

    def test_tier1_skips_tier2(self, projects):
        """Task with explicit link should NOT also match via title."""
        tasks: list[TaskInput] = [
            {"title": "Review AI tools", "description": "project: Helix Refactor"},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["Helix Refactor"]) == 1
        assert len(result["AI Adoption"]) == 0  # title has "AI" but explicit link wins

    def test_tier1_case_insensitive(self, projects):
        tasks: list[TaskInput] = [
            {"title": "task", "description": "project: ai adoption"},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 1

    def test_tier1_unknown_project_no_fallback(self, projects):
        """If project: line names unknown project, don't fall through to Tier 2."""
        tasks: list[TaskInput] = [
            {"title": "Review AI tools", "description": "project: Nonexistent Project"},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 0

    def test_tier2_name_substring(self, projects):
        tasks: list[TaskInput] = [
            {"title": "Helix Refactor planning", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["Helix Refactor"]) == 1

    def test_tier2_alias_match(self, projects):
        tasks: list[TaskInput] = [
            {"title": "TF Agentic AI meeting prep", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 1

    def test_tier2_keyword_match(self, projects):
        tasks: list[TaskInput] = [
            {"title": "SOC2 audit preparation", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["Security Policies"]) == 1

    def test_tier2_word_boundary_short_keyword(self, projects):
        """'AI' keyword should NOT match 'Railway' via substring."""
        tasks: list[TaskInput] = [
            {"title": "Railway deployment review", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 0

    def test_tier2_word_boundary_short_keyword_matches(self, projects):
        """'AI' keyword SHOULD match 'Review AI tools' via word boundary."""
        tasks: list[TaskInput] = [
            {"title": "Review AI tools", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 1

    def test_tier2_long_keyword_substring(self, projects):
        """'claude' (6 chars) should match via substring."""
        tasks: list[TaskInput] = [
            {"title": "Review claude setup", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["AI Adoption"]) == 1

    def test_multiple_projects_match(self, projects):
        """A task can match multiple projects via Tier 2."""
        tasks: list[TaskInput] = [
            {"title": "Helix Refactor and SOC2 audit", "description": None},
        ]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["Helix Refactor"]) == 1
        assert len(result["Security Policies"]) == 1

    def test_empty_tasks(self, projects):
        result = match_tasks_to_projects(projects, [])
        assert all(len(v) == 0 for v in result.values())
        assert len(result) == 3

    def test_empty_projects(self):
        tasks: list[TaskInput] = [{"title": "Some task", "description": None}]
        result = match_tasks_to_projects([], tasks)
        assert result == {}

    def test_no_description_key(self, projects):
        """Tasks without 'description' key should work (Tier 2 only)."""
        tasks: list[TaskInput] = [{"title": "Helix Refactor status"}]
        result = match_tasks_to_projects(projects, tasks)
        assert len(result["Helix Refactor"]) == 1

    def test_min_keyword_len_configurable(self):
        projects = [
            {"name": "Foo", "aliases": [], "metadata": {"task_keywords": ["AWS"]}},
        ]
        tasks: list[TaskInput] = [{"title": "outlaws", "description": None}]
        # Default min_keyword_len=4: "aws" uses word boundary → no match
        result4 = match_tasks_to_projects(projects, tasks, min_keyword_len=4)
        assert len(result4["Foo"]) == 0
        # min_keyword_len=2: "aws" uses substring → matches "outlaws"
        result2 = match_tasks_to_projects(projects, tasks, min_keyword_len=2)
        assert len(result2["Foo"]) == 1

    def test_extra_keys_preserved(self, projects):
        """Extra keys on task dicts should be preserved in results."""
        tasks = [
            {
                "title": "Helix Refactor status",
                "description": None,
                "id": "abc123",
                "tags": ["Active"],
            },
        ]
        result = match_tasks_to_projects(projects, tasks)
        matched = result["Helix Refactor"]
        assert len(matched) == 1
        assert matched[0]["id"] == "abc123"
        assert matched[0]["tags"] == ["Active"]
