"""Tests for kb context layer — Phase 7: context generation, memory commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import invoke_cli

from kb.db import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def context_db():
    """Create a test database with entities and documents for context tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()

        # Documents with dates
        docs = [
            (
                "meetings/2026/01/27/mfa_review.notes.md",
                "MFA Implementation Review",
                "2026-01-27",
                "notes",
                "granola",
                "id1",
                "[]",
                "aaa111222333",
                2,
            ),
            (
                "meetings/2026/01/20/rust_status.notes.md",
                "Helix Refactor Status",
                "2026-01-20",
                "notes",
                "granola",
                "id2",
                "[]",
                "bbb444555666",
                2,
            ),
            (
                "meetings/2026/02/01/quartz-indexer.notes.md",
                "Atlas Pipeline Plan",
                "2026-02-01",
                "notes",
                "granola",
                "id3",
                "[]",
                "ccc777888999",
                1,
            ),
            (
                "memory/people/eve.md",
                "Talia Ström",
                None,
                "memory_person",
                "memory",
                None,
                "[]",
                "ddd000111222",
                1,
            ),
        ]
        for path, title, date, dtype, src, sid, tags, chash, cc in docs:
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (path, title, date, dtype, src, sid, tags, chash, cc),
            )

        chunks = [
            (1, 0, "Overview", "MFA implementation using TOTP with Okta integration."),
            (1, 1, "Rollout", "Rollout to all employees by end of February."),
            (2, 0, "Status", "Rust migration at 30% completion."),
            (2, 1, "Performance", "Rust engine is 5x faster than Python."),
            (3, 0, "Plan", "Datastore SSO integration and Grafana dashboard migration."),
            (4, 0, None, "Talia Ström is the Infrastructure/Platform lead."),
        ]
        for doc_id, idx, heading, content in chunks:
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (doc_id, idx, heading, content),
            )

        # Entities with metadata
        entities = [
            (
                "Talia Ström",
                "person",
                '["Talia"]',
                '{"role": "Engineering Leader", "team": "Platform", "reports_to": "CTO"}',
                "memory/people/eve.md",
            ),
            (
                "Linnea Holm",
                "person",
                '["Linnea"]',
                '{"role": "Helix Refactor Lead", "team": "Engine", "reports_to": "Talia Ström"}',
                "memory/people/camille.md",
            ),
            (
                "Soren Vance",
                "person",
                '["Soren"]',
                '{"role": "Platform/DevEfficiency"}',
                "memory/people/thomas-beaumont.md",
            ),
            (
                "Helix Refactor",
                "project",
                '["helix-refactor"]',
                '{"status": "In Progress", "lead": "Linnea"}',
                None,
            ),
            (
                "AC Beacon CLI",
                "project",
                '["beacon-cli"]',
                '{"status": "Planning", "lead": "Mei"}',
                None,
            ),
            (
                "Platform",
                "team",
                '["PLAT"]',
                '{"description": "Infrastructure and platform"}',
                "memory/context/company.md",
            ),
        ]
        for name, etype, aliases, meta, src in entities:
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                (name, etype, aliases, meta, src),
            )

        # Entity mentions (Talia mentioned in 3 docs, Linnea in 1)
        mentions = [
            (1, 1, "discussed"),  # Talia in MFA review
            (1, 2, "discussed"),  # Talia in Rust status
            (1, 4, "discussed"),  # Talia in her own memory file
            (2, 2, "discussed"),  # Linnea in Rust status
            (4, 2, "discussed"),  # Helix Refactor project in Rust status doc
        ]
        for eid, did, mtype in mentions:
            conn.execute(
                "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
                (eid, did, mtype),
            )

        conn.commit()
        yield db, Path(tmpdir)
        db.close()


@pytest.fixture
def project_root_with_glossary(tmp_path):
    """Create a minimal project root with a glossary file."""
    glossary = tmp_path / "memory" / "glossary.md"
    glossary.parent.mkdir(parents=True)
    glossary.write_text(
        "# Glossary\n\n## Acronyms & Abbreviations\n\n"
        "| Term | Expansion | Notes |\n"
        "|------|-----------|-------|\n"
        "| AC | Lattice Co | Company name |\n"
        "| GIM | Lattice Co Internal Monorepo | ward-runs-app |\n"
        "| MR | Merge Request | GitLab |\n"
        "| ExCom | Executive Committee | Meets Wednesdays |\n"
        "| TF | Task Force | Cross-cutting |\n"
    )
    return tmp_path


@pytest.fixture
def project_root_with_full_glossary(tmp_path):
    """Project root with glossary having multiple table sections."""
    glossary = tmp_path / "memory" / "glossary.md"
    glossary.parent.mkdir(parents=True)
    glossary.write_text(
        "# Glossary\n\n## Acronyms\n\n"
        "| Term | Expansion | Notes |\n|------|-----------|-------|\n"
        + "".join(f"| TERM{i} | Expansion {i} | |\n" for i in range(20))
        + "\n## Internal Language\n\n"
        "| Term | Meaning |\n|------|--------|\n"
        "| Wards | Main GIM repo |\n"
        "| TGIF | Friday update |\n"
        "\n## Nicknames\n\n"
        "| Nickname | Full name |\n|----------|----------|\n"
        "| Anders M. | Soren Vance |\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Step 1: generate_context() function
# ---------------------------------------------------------------------------


class TestGenerateContext:
    def test_returns_context_output_with_text_and_stats(
        self, context_db, project_root_with_glossary
    ):
        """generate_context returns a ContextOutput with text, stats, entities."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert hasattr(result, "text")
        assert hasattr(result, "stats")
        assert hasattr(result, "entities")

    def test_text_contains_people_section(self, context_db, project_root_with_glossary):
        """Generated text should include a People section."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert "[People:" in result.text
        assert "Talia" in result.text

    def test_text_contains_projects_section(self, context_db, project_root_with_glossary):
        """Generated text should include a Projects section."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert "[Projects:" in result.text
        assert "Helix Refactor" in result.text

    def test_text_contains_terms_section(self, context_db, project_root_with_glossary):
        """Generated text should include GG Acronyms section from glossary."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert "[GG Acronyms]" in result.text
        assert "AC=Lattice Co" in result.text

    def test_stats_counts(self, context_db, project_root_with_glossary):
        """Stats should contain correct counts."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert result.stats.documents == 4
        assert result.stats.entities == 6

    def test_people_sorted_by_mention_count(self, context_db, project_root_with_glossary):
        """People should be sorted by mention count (most referenced first)."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        people = [e for e in result.entities if e.entity_type == "person"]
        # Talia has 3 mentions, Linnea has 1, Soren has 0
        assert people[0].name == "Talia Ström"

    def test_topic_filter(self, context_db, project_root_with_glossary):
        """--for topic should filter entities to only relevant ones."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary, topic="migration")
        text = result.text
        # Should have a context header
        assert "Cloud" in text or "[KB:" in text
        # Entities should be filtered — only those mentioned in migration-related docs
        entity_names = [e.name for e in result.entities]
        # Linnea and Talia should be present (mentioned in Helix Refactor status doc)
        # Soren should NOT be present (no mentions in migration docs)
        assert "Talia Ström" in entity_names
        assert "Soren Vance" not in entity_names


# ---------------------------------------------------------------------------
# Step 1b: kb context CLI command
# ---------------------------------------------------------------------------


class TestContextCommand:
    def test_context_plain_text(self, runner, context_db, project_root_with_glossary):
        """kb context should output compact format by default."""
        _, db_path = context_db
        with patch("kb.cli._find_project_root", return_value=project_root_with_glossary):
            result = invoke_cli(runner, ["context"], db_path)
        assert result.exit_code == 0
        assert "[People:" in result.output

    def test_context_json(self, runner, context_db, project_root_with_glossary):
        """kb context --json should output structured JSON."""
        _, db_path = context_db
        with patch("kb.cli._find_project_root", return_value=project_root_with_glossary):
            result = invoke_cli(runner, ["context", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "text" in data
        assert "stats" in data
        assert "entities" in data

    def test_context_for_topic(self, runner, context_db, project_root_with_glossary):
        """kb context --for 'topic' should filter entities."""
        _, db_path = context_db
        with patch("kb.cli._find_project_root", return_value=project_root_with_glossary):
            result = invoke_cli(runner, ["context", "--for", "Cloud migration"], db_path)
        assert result.exit_code == 0
        assert "Cloud" in result.output


# ---------------------------------------------------------------------------
# Step 2: kb memory add / list
# ---------------------------------------------------------------------------


class TestMemoryAdd:
    def test_memory_add_creates_fact(self, runner, context_db):
        """kb memory add should create a fact in the database."""
        _, db_path = context_db
        result = invoke_cli(
            runner,
            [
                "memory",
                "add",
                "Promoted to Staff Engineer",
                "--entity",
                "Talia",
                "--date",
                "2026-01-19",
            ],
            db_path,
        )
        assert result.exit_code == 0

        # Verify fact in database
        db = Database(Path(db_path))
        conn = db.get_sqlite_conn()
        facts = conn.execute("SELECT * FROM facts").fetchall()
        assert len(facts) == 1
        assert facts[0]["fact_text"] == "Promoted to Staff Engineer"
        assert facts[0]["fact_date"] == "2026-01-19"
        db.close()

    def test_memory_add_appends_to_file(self, runner, context_db):
        """kb memory add should append to the entity's source file."""
        _, db_path = context_db
        # Create the source file
        source_dir = Path(db_path).parent / "memory" / "people"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / "eve.md"
        source_file.write_text("# Talia Ström\n\n**Role:** Engineering Leader\n")

        # Update entity source_path to use the temp dir-relative path
        db = Database(Path(db_path))
        conn = db.get_sqlite_conn()
        conn.execute(
            "UPDATE entities SET source_path = ? WHERE name = 'Talia Ström'",
            (str(source_file),),
        )
        conn.commit()
        db.close()

        result = invoke_cli(
            runner,
            ["memory", "add", "Led Platform TF", "--entity", "Talia", "--date", "2026-02-01"],
            db_path,
        )
        assert result.exit_code == 0
        content = source_file.read_text()
        assert "Led Platform TF" in content

    def test_memory_add_entity_not_found(self, runner, context_db):
        """kb memory add with unknown entity should fail gracefully."""
        _, db_path = context_db
        result = invoke_cli(
            runner,
            ["memory", "add", "Some fact", "--entity", "NonexistentPerson"],
            db_path,
        )
        assert result.exit_code != 0


class TestMemoryList:
    def test_memory_list_empty(self, runner, context_db):
        """kb memory list on empty facts table should succeed."""
        _, db_path = context_db
        result = invoke_cli(runner, ["memory", "list"], db_path)
        assert result.exit_code == 0

    def test_memory_list_shows_facts(self, runner, context_db):
        """kb memory list should show previously added facts."""
        _, db_path = context_db
        # Add a fact first
        invoke_cli(
            runner,
            ["memory", "add", "Completed Rust review", "--entity", "Talia", "--date", "2026-02-10"],
            db_path,
        )
        result = invoke_cli(runner, ["memory", "list", "--json"], db_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "results" in data
        assert len(data["results"]) >= 1
        assert data["results"][0]["fact_text"] == "Completed Rust review"

    def test_memory_list_since_filter(self, runner, context_db):
        """kb memory list --since should filter by days."""
        _, db_path = context_db
        invoke_cli(
            runner,
            ["memory", "add", "Recent fact", "--entity", "Talia", "--date", "2026-02-14"],
            db_path,
        )
        result = invoke_cli(runner, ["memory", "list", "--since", "7", "--json"], db_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Step 3: Migration for facts table
# ---------------------------------------------------------------------------


class TestAppendFactToFile:
    def test_fact_inserted_under_header_not_at_eof(self, tmp_path):
        """Fact should go under ## Recent Facts, not after a following section."""
        md = tmp_path / "person.md"
        md.write_text(
            "# Person\n\n## Recent Facts\n- [2026-01-01] Old fact\n\n## Notes\nSome notes.\n"
        )
        from kb.cli import _append_fact_to_file

        _append_fact_to_file(md, "New fact", "2026-02-01")
        content = md.read_text()
        # New fact should appear before ## Notes
        notes_pos = content.index("## Notes")
        fact_pos = content.index("[2026-02-01] New fact")
        assert fact_pos < notes_pos

    def test_fact_appended_when_no_following_section(self, tmp_path):
        """When ## Recent Facts is the last section, fact appends at end (existing behavior)."""
        md = tmp_path / "person.md"
        md.write_text("# Person\n\n## Recent Facts\n- [2026-01-01] Old fact\n")
        from kb.cli import _append_fact_to_file

        _append_fact_to_file(md, "New fact", "2026-02-01")
        content = md.read_text()
        assert "[2026-02-01] New fact" in content
        assert content.index("[2026-02-01]") > content.index("[2026-01-01]")

    def test_fact_creates_section_when_missing(self, tmp_path):
        """When no ## Recent Facts exists, creates the section."""
        md = tmp_path / "person.md"
        md.write_text("# Person\n\nSome content.\n")
        from kb.cli import _append_fact_to_file

        _append_fact_to_file(md, "First fact", "2026-02-01")
        content = md.read_text()
        assert "## Recent Facts" in content
        assert "[2026-02-01] First fact" in content


class TestContextGlossaryDisplay:
    """Context output should filter terms and exclude nicknames."""

    def test_standard_terms_filtered(self, context_db, tmp_path):
        """Standard industry terms should be filtered from acronyms."""
        from kb.context import generate_context

        glossary = tmp_path / "memory" / "glossary.md"
        glossary.parent.mkdir(parents=True)
        glossary.write_text(
            "# Glossary\n\n## Acronyms\n\n"
            "| Term | Expansion |\n|------|-----------|"
            "\n| AC | Lattice Co |\n"
            "\n| CI | Continuous Integration |\n"
        )

        db, _ = context_db
        result = generate_context(db, tmp_path)
        text = result.text
        assert "AC=Lattice Co" in text
        assert "CI=" not in text  # CI is a standard industry term

    def test_nicknames_excluded(self, context_db, tmp_path):
        """Nicknames section (Full name column) should be excluded from context terms."""
        from kb.context import generate_context

        glossary = tmp_path / "memory" / "glossary.md"
        glossary.parent.mkdir(parents=True)
        glossary.write_text(
            "# Glossary\n\n## Acronyms\n\n"
            "| Term | Expansion |\n|------|-----------|"
            "\n| AC | Lattice Co |\n"
            "\n## Nicknames\n\n"
            "| Nickname | Full name |\n|----------|----------|\n"
            "| Anders M. | Soren Vance |\n"
            "| Talia | Talia Ström |\n"
        )

        db, _ = context_db
        result = generate_context(db, tmp_path)
        text = result.text
        # GG should be present (it's an acronym)
        assert "AC=Lattice Co" in text
        # Nicknames should NOT be in Terms section (they're redundant with People)
        assert "Anders M.=Soren Vance" not in text
        assert "Talia=Talia Ström" not in text


class TestGlossaryParsing:
    def test_returns_more_than_15_terms(self, project_root_with_full_glossary):
        """Should return all terms, not capped at 15."""
        from kb.context import _parse_glossary_terms

        terms = _parse_glossary_terms(project_root_with_full_glossary)
        assert len(terms) > 15  # 20 acronyms + 2 internal + 1 nickname = 23

    def test_includes_internal_language(self, project_root_with_full_glossary):
        """Should parse Internal Language table (column header 'Meaning')."""
        from kb.context import _parse_glossary_terms

        terms = _parse_glossary_terms(project_root_with_full_glossary)
        term_names = [t for t, _e, _s in terms]
        assert "Wards" in term_names
        assert "TGIF" in term_names

    def test_includes_nicknames(self, project_root_with_full_glossary):
        """Should parse Nicknames table (column header 'Full name')."""
        from kb.context import _parse_glossary_terms

        terms = _parse_glossary_terms(project_root_with_full_glossary)
        term_names = [t for t, _e, _s in terms]
        assert "Anders M." in term_names


class TestContextSmartFiltering:
    def test_people_zero_mentions_excluded(self, context_db, project_root_with_glossary):
        """People with 0 mentions should be excluded from context output."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        # Soren has 0 mentions in the fixture — should not appear
        assert "Soren" not in result.text

    def test_pinned_entity_always_shown(self, context_db, project_root_with_glossary):
        """Pinned entity with 0 mentions appears in output."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        # Pin Soren (who has 0 mentions)
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Soren Vance'")
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        assert "Soren" in result.text

    def test_completed_projects_excluded(self, context_db, project_root_with_glossary):
        """Projects with status 'Completed' excluded from output."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        # Add a Completed project
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
            ("Old Project", "project", "[]", '{"status": "Completed"}'),
        )
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        assert "Old Project" not in result.text

    def test_zero_mention_teams_excluded(self, context_db, project_root_with_glossary):
        """Teams with 0 mentions excluded from output."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        # Add a team with 0 mentions
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
            ("Ghost Team", "team", "[]", '{"description": "no mentions"}'),
        )
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        assert "Ghost Team" not in result.text
        # Platform team also has 0 mentions in fixture — should be excluded
        assert "Platform" not in result.text

    def test_topic_filter_skips_smart_filtering(self, context_db, project_root_with_glossary):
        """Topic mode should NOT apply mention-count filters."""
        from kb.context import generate_context

        db, _ = context_db
        # In topic mode, even 0-mention entities should appear if relevant to the topic
        result = generate_context(db, project_root_with_glossary, topic="Cloud migration")
        # This should still include entities found via topic search, regardless of mention count
        entity_names = [e.name for e in result.entities]
        # The topic search finds entities via FTS/name matching, not mention count
        assert len(entity_names) > 0


class TestContextTermFiltering:
    @pytest.fixture
    def glossary_with_standard_terms(self, tmp_path):
        """Glossary with standard industry terms and project-specific terms."""
        glossary = tmp_path / "memory" / "glossary.md"
        glossary.parent.mkdir(parents=True)
        glossary.write_text(
            "# Glossary\n\n"
            "## Acronyms & Abbreviations\n\n"
            "| Term | Expansion | Notes |\n|------|-----------|-------|\n"
            "| AC | Lattice Co | Company name |\n"
            "| CI | Continuous Integration | GitLab CI |\n"
            "| SaaS | Software as a Service | Cloud |\n"
            "| NHI | Non-Human Identity | Team |\n"
            "| ExCom | Executive Committee | Leadership |\n"
            "\n## Internal Language\n\n"
            "| Term | Meaning |\n|------|--------|\n"
            "| Wards | Main GIM repo |\n"
            "| TGIF | Friday update |\n"
            "| Code Red | Critical issue |\n"
        )
        return tmp_path

    def test_standard_industry_terms_excluded(self, context_db, glossary_with_standard_terms):
        """Standard terms like CI, SaaS should not appear in context."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, glossary_with_standard_terms)
        text = result.text
        # CI and SaaS are standard industry terms — should be excluded
        assert "CI=" not in text
        assert "SaaS=" not in text
        # AC is project-specific — should be included
        assert "AC=Lattice Co" in text

    def test_gg_acronyms_shown_with_expansion(self, context_db, glossary_with_standard_terms):
        """Project-specific acronyms should show term=expansion format."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, glossary_with_standard_terms)
        text = result.text
        assert "AC=Lattice Co" in text
        assert "NHI=Non-Human Identity" in text

    def test_internal_terms_names_only(self, context_db, glossary_with_standard_terms):
        """Internal Language terms should appear as names without =expansion."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, glossary_with_standard_terms)
        text = result.text
        # Internal terms should appear without expansion
        assert "Wards" in text
        assert "Wards=" not in text
        assert "TGIF" in text
        assert "TGIF=" not in text


class TestGlossaryParsingSections:
    def test_returns_section_name(self, project_root_with_full_glossary):
        """Each term should include its section name as third element."""
        from kb.context import _parse_glossary_terms

        terms = _parse_glossary_terms(project_root_with_full_glossary)
        assert len(terms) > 0
        assert len(terms[0]) == 3  # (term, expansion, section)

    def test_acronyms_section_tagged(self, project_root_with_full_glossary):
        """Terms from Acronyms section should have 'Acronyms' in section name."""
        from kb.context import _parse_glossary_terms

        terms = _parse_glossary_terms(project_root_with_full_glossary)
        acronym_terms = [t for t in terms if "Acronym" in t[2]]
        assert len(acronym_terms) > 0
        # TERM0 through TERM19 should be in Acronyms
        assert any(t[0].startswith("TERM") for t in acronym_terms)

    def test_internal_section_tagged(self, project_root_with_full_glossary):
        """Terms from Internal Language should have 'Internal' in section name."""
        from kb.context import _parse_glossary_terms

        terms = _parse_glossary_terms(project_root_with_full_glossary)
        internal_terms = [t for t in terms if "Internal" in t[2]]
        assert len(internal_terms) > 0
        term_names = [t[0] for t in internal_terms]
        assert "Wards" in term_names


class TestSanitizeFTSDedup:
    def test_context_uses_search_sanitize(self):
        """context module should import _sanitize_fts_input from search, not define its own."""
        import kb.context as ctx
        import kb.search as srch

        assert ctx._sanitize_fts_input is srch._sanitize_fts_input


class TestContextFormat:
    def test_compact_format_default(self, context_db, project_root_with_glossary):
        """Default format should use [Section] headers and | separators."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        text = result.text
        assert "[People:" in text
        assert "|" in text
        assert "##" not in text

    def test_human_format_uses_markdown(self, context_db, project_root_with_glossary):
        """Human format should use ## headers."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary, fmt="human")
        text = result.text
        assert "## Key People" in text

    def test_compact_pinned_marker(self, context_db, project_root_with_glossary):
        """Pinned entities should show star in compact format."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        # Pin Talia
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Talia Ström'")
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        text = result.text
        assert "\u2605" in text  # ★

    def test_commands_section_present(self, context_db, project_root_with_glossary):
        """Compact format should include [Commands] section."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        text = result.text
        assert "[Commands]" in text
        assert "kb search" in text
        assert "kb pin" in text


class TestFactsMigration:
    def test_facts_table_exists(self):
        """Fresh DB should have a facts table after migrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert "facts" in tables
            db.close()

    def test_facts_migration_recorded(self):
        """Facts migration should be recorded in migrations table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            rows = conn.execute("SELECT name FROM migrations").fetchall()
            names = [r["name"] for r in rows]
            assert "002_create_facts_table" in names
            db.close()


# ---------------------------------------------------------------------------
# Pydantic model tests (Task 11)
# ---------------------------------------------------------------------------


class TestContextPydantic:
    """Verify context.py returns Pydantic models instead of raw dicts."""

    def test_generate_context_returns_context_output(self, context_db, project_root_with_glossary):
        from kb.context import generate_context
        from kb.types import ContextOutput

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert isinstance(result, ContextOutput)

    def test_context_output_has_text(self, context_db, project_root_with_glossary):
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert isinstance(result.text, str)
        assert "[People:" in result.text

    def test_context_output_stats(self, context_db, project_root_with_glossary):
        from kb.context import generate_context
        from kb.types import ContextStats, DateRange

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert isinstance(result.stats, ContextStats)
        assert isinstance(result.stats.date_range, DateRange)
        assert result.stats.documents == 4
        assert result.stats.entities == 6

    def test_context_output_entities_are_summaries(self, context_db, project_root_with_glossary):
        from kb.context import generate_context
        from kb.types import ContextEntitySummary

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert len(result.entities) > 0
        for e in result.entities:
            assert isinstance(e, ContextEntitySummary)
            assert isinstance(e.name, str)
            assert isinstance(e.entity_type, str)
            assert isinstance(e.mention_count, int)

    def test_context_output_model_dump(self, context_db, project_root_with_glossary):
        """model_dump() should produce a dict matching the old dict format."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        d = result.model_dump()
        assert "text" in d
        assert "stats" in d
        assert "entities" in d
        assert isinstance(d["stats"]["date_range"]["earliest"], (str, type(None)))

    def test_context_output_frozen(self, context_db, project_root_with_glossary):
        """ContextOutput should be immutable."""
        from pydantic import ValidationError

        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        with pytest.raises(ValidationError):
            result.text = "mutated"  # type: ignore[misc]

    def test_topic_filter_returns_context_output(self, context_db, project_root_with_glossary):
        from kb.context import generate_context
        from kb.types import ContextOutput

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary, topic="Cloud migration")
        assert isinstance(result, ContextOutput)

    def test_human_format_returns_context_output(self, context_db, project_root_with_glossary):
        from kb.context import generate_context
        from kb.types import ContextOutput

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary, fmt="human")
        assert isinstance(result, ContextOutput)
        assert "## Key People" in result.text


class TestPeopleTierHelpers:
    """Tests for _get_fact_counts and _is_key_person."""

    def test_get_fact_counts_empty(self, context_db):
        """No facts -> empty dict."""
        from kb.context import _get_fact_counts

        db, _ = context_db
        conn = db.get_sqlite_conn()
        result = _get_fact_counts(conn)
        assert result == {}

    def test_get_fact_counts_with_facts(self, context_db):
        """Facts present -> entity_id: count mapping."""
        from kb.context import _get_fact_counts

        db, _ = context_db
        conn = db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (1, 'Promoted', '2026-01-15')"
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (1, 'Led TF', '2026-02-01')"
        )
        conn.commit()
        result = _get_fact_counts(conn)
        assert result == {1: 2}

    def test_is_key_person_two_fields(self, context_db):
        """Person with 2+ metadata fields (role+team) is key."""
        from kb.context import _is_key_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=1,
            name="Talia",
            entity_type="person",
            aliases=[],
            metadata={"role": "Lead", "team": "Platform"},
            source_path=None,
            mention_count=3,
            pinned=False,
        )
        assert _is_key_person(entity, {}) is True

    def test_is_key_person_one_field_with_facts(self, context_db):
        """Person with 1 metadata field but facts is key."""
        from kb.context import _is_key_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=99,
            name="Soren",
            entity_type="person",
            aliases=[],
            metadata={"role": "Engineer"},
            source_path=None,
            mention_count=0,
            pinned=False,
        )
        assert _is_key_person(entity, {99: 1}) is True

    def test_is_key_person_one_field_no_facts(self, context_db):
        """Person with 1 metadata field and no facts is NOT key."""
        from kb.context import _is_key_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=99,
            name="Soren",
            entity_type="person",
            aliases=[],
            metadata={"role": "Engineer"},
            source_path=None,
            mention_count=5,
            pinned=False,
        )
        assert _is_key_person(entity, {}) is False

    def test_is_key_person_empty_metadata(self, context_db):
        """Person with no metadata and no facts is NOT key."""
        from kb.context import _is_key_person
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=99,
            name="Nobody",
            entity_type="person",
            aliases=[],
            metadata={},
            source_path=None,
            mention_count=10,
            pinned=False,
        )
        assert _is_key_person(entity, {}) is False


class TestPersonFormatters:
    """Tests for _format_person_pinned and _format_person_key."""

    def test_format_pinned_full_metadata(self):
        """Pinned person shows role, team, reports_to, and mention count."""
        from kb.context import _format_person_pinned
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=1,
            name="Talia Ström",
            entity_type="person",
            aliases=["Talia"],
            metadata={
                "role": "Engineering Leader",
                "team": "Platform",
                "reports_to": "CTO",
            },
            source_path=None,
            mention_count=3,
            pinned=True,
        )
        result = _format_person_pinned(entity)
        assert result == "Talia\u2605(Engineering Leader|team:Platform|\u2192CTO,3m) ○"

    def test_format_pinned_no_reports_to(self):
        """Pinned person without reports_to omits the arrow."""
        from kb.context import _format_person_pinned
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=1,
            name="Soren Vance",
            entity_type="person",
            aliases=["Soren"],
            metadata={"role": "Head of Engineering"},
            source_path=None,
            mention_count=1265,
            pinned=True,
        )
        result = _format_person_pinned(entity)
        assert result == "Soren\u2605(Head of Engineering,1265m) ○"

    def test_format_pinned_long_role_truncated(self):
        """Pinned person with long role gets truncated at 20 chars."""
        from kb.context import _format_person_pinned
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=1,
            name="Anders Holt",
            entity_type="person",
            aliases=["Anders"],
            metadata={
                "role": "Engineering Director \u2014 Engine Team",
                "team": "Engine",
            },
            source_path=None,
            mention_count=100,
            pinned=True,
        )
        result = _format_person_pinned(entity)
        assert "Engineering Direc..." in result
        assert "team:Engine" in result

    def test_format_key_role_and_reports_to(self):
        """Key person shows role and reports_to (no team)."""
        from kb.context import _format_person_key
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=2,
            name="Idris Falk",
            entity_type="person",
            aliases=["Idris"],
            metadata={
                "role": "Eng Director",
                "team": "Core",
                "reports_to": "Jeremy",
            },
            source_path=None,
            mention_count=1066,
            pinned=False,
        )
        result = _format_person_key(entity)
        assert result == "Idris(Eng Director|\u2192Jeremy,1066m) ○"

    def test_format_key_no_reports_to(self):
        """Key person without reports_to shows just role."""
        from kb.context import _format_person_key
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=2,
            name="Linnea Holm",
            entity_type="person",
            aliases=["Linnea"],
            metadata={"role": "Helix Refactor Lead", "team": "Engine"},
            source_path=None,
            mention_count=1,
            pinned=False,
        )
        result = _format_person_key(entity)
        assert result == "Linnea(Helix Refactor Lead,1m) ○"


class TestPeopleTierSplitting:
    """People are split into pinned and key tiers in full context mode."""

    def test_pinned_person_in_output(self, context_db, project_root_with_glossary):
        """Pinned person always appears regardless of mention count."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Soren Vance'")
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        assert "Soren" in result.text
        assert "[People:pinned]" in result.text

    def test_key_person_in_output(self, context_db, project_root_with_glossary):
        """Key person (2+ metadata fields, not pinned) appears in key section."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert "Talia" in result.text
        assert "[People:key]" in result.text

    def test_non_key_person_excluded(self, context_db, project_root_with_glossary):
        """Person with <2 metadata fields and no facts is excluded."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        assert "Soren" not in result.text

    def test_fact_makes_person_key(self, context_db, project_root_with_glossary):
        """Person with 1 metadata field but facts becomes key."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date)"
            " VALUES (3, 'Led migration', '2026-02-01')"
        )
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        assert "Soren" in result.text

    def test_topic_mode_unchanged(self, context_db, project_root_with_glossary):
        """Topic mode uses flat [People:N by mentions], never pinned/key tiers."""
        from kb.context import generate_context

        db, _ = context_db
        # Use "Talia" as topic so name-matching strategy finds the person entity
        result = generate_context(db, project_root_with_glossary, topic="Talia")
        # Topic mode should use the flat "N by mentions" format
        assert "[People:" in result.text
        assert "by mentions" in result.text
        # And never show tier headers
        assert "[People:pinned]" not in result.text
        assert "[People:key]" not in result.text


class TestPeopleTierRendering:
    """Integration tests for pinned/key people rendering in both formats."""

    def test_compact_pinned_shows_full_metadata(self, context_db, project_root_with_glossary):
        """Pinned person in compact format shows role, team, reports_to."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Talia Ström'")
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        text = result.text
        assert "[People:pinned]" in text
        assert "Talia\u2605(" in text
        assert "team:Platform" in text
        assert "\u2192CTO" in text

    def test_compact_key_shows_reports_to(self, context_db, project_root_with_glossary):
        """Key person in compact format shows role and reports_to."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        text = result.text
        assert "[People:key]" in text
        assert "\u2192Talia Ström" in text

    def test_compact_key_no_team(self, context_db, project_root_with_glossary):
        """Key person in compact format does NOT show team."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary)
        text = result.text
        assert "team:Engine" not in text

    def test_human_pinned_section(self, context_db, project_root_with_glossary):
        """Human format has ## Pinned People section."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Talia Ström'")
        conn.commit()
        result = generate_context(db, project_root_with_glossary, fmt="human")
        text = result.text
        assert "## Pinned People" in text
        assert "## Key People" in text

    def test_human_key_section(self, context_db, project_root_with_glossary):
        """Human format has ## Key People section with key people."""
        from kb.context import generate_context

        db, _ = context_db
        result = generate_context(db, project_root_with_glossary, fmt="human")
        text = result.text
        assert "## Key People" in text
        assert "Talia" in text

    def test_no_people_sections_when_empty(self, context_db, project_root_with_glossary):
        """No people sections if no pinned or key people exist."""
        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        conn.execute("DELETE FROM entities WHERE entity_type = 'person'")
        conn.commit()
        result = generate_context(db, project_root_with_glossary)
        assert "[People:" not in result.text


# ---------------------------------------------------------------------------
# Freshness indicators (Task 5)
# ---------------------------------------------------------------------------


class TestFreshnessIndicator:
    def test_freshness_indicator_fresh(self):
        """Entity updated within 30 days shows ● indicator."""
        from datetime import date

        from kb.context import _freshness_indicator
        from kb.types import ContextEntity

        today = date.today().isoformat()
        entity = ContextEntity(
            id=1,
            name="Wren",
            entity_type="person",
            mention_count=10,
            last_mentioned_at=today,
        )
        result = _freshness_indicator(entity)
        assert result == " ●"

    def test_freshness_indicator_aging(self):
        """Entity updated 31-90 days ago shows ◐Nd indicator."""
        from datetime import date, timedelta

        from kb.context import _freshness_indicator
        from kb.types import ContextEntity

        d = (date.today() - timedelta(days=60)).isoformat()
        entity = ContextEntity(
            id=1,
            name="Soren",
            entity_type="person",
            mention_count=5,
            last_mentioned_at=d,
        )
        result = _freshness_indicator(entity)
        assert result == " ◐60d"

    def test_freshness_indicator_stale(self):
        """Entity updated >90 days ago shows ○Nd indicator."""
        from datetime import date, timedelta

        from kb.context import _freshness_indicator
        from kb.types import ContextEntity

        d = (date.today() - timedelta(days=120)).isoformat()
        entity = ContextEntity(
            id=1,
            name="Linnea",
            entity_type="person",
            mention_count=3,
            last_mentioned_at=d,
        )
        result = _freshness_indicator(entity)
        assert result == " ○120d"

    def test_freshness_indicator_no_dates(self):
        """Entity with no timestamps shows ○ indicator."""
        from kb.context import _freshness_indicator
        from kb.types import ContextEntity

        entity = ContextEntity(
            id=1,
            name="Unknown",
            entity_type="person",
            mention_count=0,
        )
        result = _freshness_indicator(entity)
        assert result == " ○"

    def test_freshness_uses_max_of_both_timestamps(self):
        """Freshness uses the more recent of updated_at and last_mentioned_at."""
        from datetime import date

        from kb.context import _freshness_indicator
        from kb.types import ContextEntity

        today = date.today().isoformat()
        entity = ContextEntity(
            id=1,
            name="Kit",
            entity_type="person",
            mention_count=5,
            updated_at="2020-01-01",  # very old
            last_mentioned_at=today,  # fresh
        )
        result = _freshness_indicator(entity)
        assert result == " ●"


class TestRecencySortKey:
    def test_recent_entity_ranks_higher(self):
        """Entity with recent activity + fewer mentions beats stale entity with many mentions."""
        from datetime import date, timedelta

        from kb.context import _recency_sort_key
        from kb.types import ContextEntity

        recent = ContextEntity(
            id=1,
            name="Recent",
            entity_type="person",
            mention_count=5,
            last_mentioned_at=date.today().isoformat(),
        )
        stale = ContextEntity(
            id=2,
            name="Stale",
            entity_type="person",
            mention_count=50,
            last_mentioned_at=(date.today() - timedelta(days=365)).isoformat(),
        )
        assert _recency_sort_key(recent) > _recency_sort_key(stale)


class TestContextFreshnessIntegration:
    def test_context_output_contains_freshness_indicators(
        self, context_db, project_root_with_glossary
    ):
        """Context output includes freshness indicators for entities."""
        from datetime import date

        from kb.context import generate_context

        db, _ = context_db
        conn = db.get_sqlite_conn()
        # Pin Talia and set a recent last_mentioned_at
        conn.execute(
            "UPDATE entities SET pinned = 1, last_mentioned_at = ? WHERE name = 'Talia Ström'",
            (date.today().isoformat(),),
        )
        conn.commit()

        result = generate_context(db, project_root_with_glossary)
        text = result.text
        # Talia is pinned and fresh — should have ● indicator
        assert "●" in text
