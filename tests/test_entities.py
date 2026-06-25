"""Tests for entity seeding and linking."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db():
    """Create a temporary database."""
    from kb.db import Database

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        yield db
        db.close()


@pytest.fixture
def project_root():
    """Return the real project root."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def entity_root(tmp_path):
    """Synthetic project root with people, projects, company.md for entity tests."""
    mem = tmp_path / "memory"
    # People
    people = mem / "people"
    people.mkdir(parents=True)
    (people / "david-marchand.md").write_text(
        "# Soren Vance\n\n"
        "**Also known as:** Anders M., Anders\n"
        "**Email:** david@example.com\n"
        "**Role:** CEO\n"
        "**Team:** ExCom\n"
    )
    (people / "wren-kasper.md").write_text(
        "# Wren Kasper\n\n**Also known as:** Wren\n**Role:** Engineer\n**Team:** Platform\n"
    )
    # Projects
    projects = mem / "projects"
    projects.mkdir(parents=True)
    (projects / "helix-refactor.md").write_text(
        "# Helix Refactor\n\n"
        "**Codename/Also called:** Core Detection Engine Rewrite\n"
        "**Status:** Active\n"
        "**Lead:** Linnea Aalto\n"
    )
    # Context
    context = mem / "context"
    context.mkdir(parents=True)
    (context / "company.md").write_text(
        "# Company Context\n\n"
        "## Teams (from Linear)\n\n"
        "- **Secrets Detection** (SCRT) — Detects hardcoded secrets\n"
        "- **NHI Governance** — Non-human identity management\n"
        "- **Front End** (FE) — UI and web app\n"
        "- **SRE** — Site reliability engineering\n"
        "- **DevEfficiency** (DEFF) — Developer tooling\n"
        "\n## Other Section\n\nSome other content.\n"
    )
    return tmp_path


class TestPersonParsing:
    def test_parse_person_file(self, tmp_path):
        """Test parsing a sample person memory file."""
        from kb.entities import _parse_person_file

        person_file = tmp_path / "test-person.md"
        person_file.write_text(
            "# Jane Doe\n\n"
            "**Also known as:** JD, Jane D.\n"
            "**Role:** Engineering Lead\n"
            "**Team:** Core\n"
            "**Reports to:** CTO\n"
        )

        result = _parse_person_file(person_file)
        assert result.name == "Jane Doe"
        assert result.entity_type == "person"
        assert "JD" in result.aliases
        assert "Jane D." in result.aliases
        assert "Jane" in result.aliases
        assert "test-person" in result.aliases
        assert result.metadata["role"] == "Engineering Lead"
        assert result.metadata["team"] == "Core"
        assert result.metadata["reports_to"] == "CTO"

    def test_parse_real_person_file(self, entity_root):
        """Test parsing a person file with AKA, email, and metadata."""
        from kb.entities import _parse_person_file

        path = entity_root / "memory" / "people" / "david-marchand.md"

        result = _parse_person_file(path)
        assert result.name == "Soren Vance"
        assert result.entity_type == "person"
        assert "Anders M." in result.aliases
        assert "Anders" in result.aliases
        assert "david-marchand" in result.aliases
        assert result.metadata["role"] == "CEO"


class TestProjectParsing:
    def test_parse_project_file(self, tmp_path):
        """Test parsing a sample project memory file."""
        from kb.entities import _parse_project_file

        project_file = tmp_path / "test-project.md"
        project_file.write_text(
            "# Test Project\n\n"
            "**Codename/Also called:** Project X\n"
            "**Status:** Active\n"
            "**Started:** Q1 2026\n"
            "**Lead:** Jane Doe\n"
        )

        result = _parse_project_file(project_file)
        assert result.name == "Test Project"
        assert result.entity_type == "project"
        assert "Project X" in result.aliases
        assert "test-project" in result.aliases
        assert result.metadata["status"] == "Active"
        assert result.metadata["lead"] == "Jane Doe"

    def test_parse_real_project_file(self, entity_root):
        """Test parsing a project file with codename and metadata."""
        from kb.entities import _parse_project_file

        path = entity_root / "memory" / "projects" / "helix-refactor.md"

        result = _parse_project_file(path)
        assert result.name == "Helix Refactor"
        assert "Core Detection Engine Rewrite" in result.aliases
        assert result.metadata["status"] == "Active"
        assert "Linnea" in result.metadata["lead"]


class TestTeamExtraction:
    def test_parse_teams_from_company(self, entity_root):
        """Test team extraction from company.md."""
        from kb.entities import _parse_teams_from_company

        path = entity_root / "memory" / "context" / "company.md"

        teams = _parse_teams_from_company(path)
        assert len(teams) == 5

        team_names = {t.name for t in teams}
        assert "Secrets Detection" in team_names
        assert "NHI Governance" in team_names
        assert "Front End" in team_names
        assert "SRE" in team_names
        assert "DevEfficiency" in team_names

        # Check abbreviation aliases
        scrt = next(t for t in teams if t.name == "Secrets Detection")
        assert "SCRT" in scrt.aliases

    def test_team_entity_type(self, entity_root):
        """All teams should have entity_type 'team'."""
        from kb.entities import _parse_teams_from_company

        path = entity_root / "memory" / "context" / "company.md"

        teams = _parse_teams_from_company(path)
        for team in teams:
            assert team.entity_type == "team"


class TestEntityLinking:
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for linking tests."""
        from kb.entities import Entity

        return [
            Entity(id=1, name="Wren Kasper", entity_type="person", aliases=["Wren", "wren-kasper"]),
            Entity(
                id=2,
                name="Soren Vance",
                entity_type="person",
                aliases=["Soren", "thomas-beaumont"],
            ),
            Entity(
                id=3,
                name="Anders Holt",
                entity_type="person",
                aliases=["Anders M.", "Anders", "david-marchand"],
            ),
            Entity(
                id=4,
                name="Kit Larsen",
                entity_type="person",
                aliases=["Kit K.", "Anders", "dave-kowalski"],
            ),
            Entity(
                id=5,
                name="Helix Refactor",
                entity_type="project",
                aliases=["Core Detection Engine Rewrite", "helix-refactor"],
            ),
            Entity(id=6, name="Lattice Co", entity_type="company", aliases=["AC"]),
        ]

    def test_tag_matching(self, sample_entities):
        """Tags should match entity names/aliases."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Some meeting",
            tags=["Soren"],
            content="Nothing relevant here.",
            entities=sample_entities,
        )

        tagged = [m for m in mentions if m.mention_type == "tagged"]
        assert len(tagged) == 1
        assert tagged[0].entity_id == 2  # Soren Vance

    def test_title_participant_parsing(self, sample_entities):
        """Title separators should identify participants."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Wren / Soren",
            tags=[],
            content="Short content.",
            entities=sample_entities,
        )

        participants = [m for m in mentions if m.mention_type == "participant"]
        participant_ids = {m.entity_id for m in participants}
        assert 1 in participant_ids  # Wren Kasper
        assert 2 in participant_ids  # Soren Vance

    def test_content_name_matching(self, sample_entities):
        """Content should match entity names with word boundaries."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="We discussed the Helix Refactor progress and AC strategy.",
            entities=sample_entities,
        )

        discussed = [m for m in mentions if m.mention_type == "discussed"]
        discussed_ids = {m.entity_id for m in discussed}
        assert 5 in discussed_ids  # Helix Refactor
        assert 6 in discussed_ids  # Lattice Co (matched via "AC" alias)

    def test_content_word_boundary(self, sample_entities):
        """Word boundaries should prevent false matches."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="We went to Charleston for the conference.",
            entities=sample_entities,
        )

        discussed = [m for m in mentions if m.mention_type == "discussed"]
        # "Soren" should NOT match "Charleston"
        charles_mentions = [m for m in discussed if m.entity_id == 2]
        assert len(charles_mentions) == 0

    def test_short_first_name_matched_in_content(self, sample_entities):
        """An *unambiguous* first name >3 chars should match in content (threshold rule).

        ("Soren" is a 5-char single name owned by exactly one entity; ambiguous bare
        first names are gated separately — see TestFirstNameDisambiguation, #36.)
        """
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="Soren presented the quarterly results.",
            entities=sample_entities,
        )

        discussed = [m for m in mentions if m.mention_type == "discussed"]
        assert 2 in {m.entity_id for m in discussed}  # Soren Vance (unambiguous)

    def test_suppressed_ids_are_not_linked(self, sample_entities):
        """find_entity_mentions skips entity ids in suppressed_ids (#35)."""
        from kb.entities import find_entity_mentions

        # Use a full name so the match is robust to #36's bare-name gating.
        base = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="Soren Vance presented the quarterly results.",
            entities=sample_entities,
        )
        base_ids = {m.entity_id for m in base}
        assert base_ids, "expected at least one match to suppress"
        target = next(iter(base_ids))
        suppressed = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="Soren Vance presented the quarterly results.",
            entities=sample_entities,
            suppressed_ids={target},
        )
        assert target not in {m.entity_id for m in suppressed}

    def test_very_short_name_skipped_in_content(self, sample_entities):
        """Very short single names (<=3 chars) should NOT match in content."""
        from kb.entities import Entity, find_entity_mentions

        entities_with_short = [
            *sample_entities,
            Entity(id=10, name="Ed Wilson", entity_type="person", aliases=["Ed", "ed-wilson"]),
        ]

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="Ed presented the quarterly results.",
            entities=entities_with_short,
        )

        discussed = [m for m in mentions if m.mention_type == "discussed"]
        ed_ids = {m.entity_id for m in discussed if m.entity_id == 10}
        # "Ed" (2 chars) is <= 3 — should NOT match in content
        assert len(ed_ids) == 0

    def test_short_first_name_matches_via_tag(self, sample_entities):
        """Short first names should still match via tag matching."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=["Anders"],
            content="Nothing relevant here.",
            entities=sample_entities,
        )

        tagged = [m for m in mentions if m.mention_type == "tagged"]
        david_ids = {m.entity_id for m in tagged if m.entity_id in (3, 4)}
        # Both Davids should match via tag
        assert 3 in david_ids
        assert 4 in david_ids

    def test_title_substring_matching(self, sample_entities):
        """An unambiguous name appearing as a substring in the title matches as 'title'.

        ("Wren" is owned by one entity; ambiguous bare names in titles are gated — #36.)
        """
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Wren Sync Notes",
            tags=[],
            content="Short content with no names.",
            entities=sample_entities,
        )

        title_mentions = [m for m in mentions if m.mention_type == "title"]
        assert 1 in {m.entity_id for m in title_mentions}  # Wren Kasper (unambiguous)

    def test_title_substring_skips_short_names(self, sample_entities):
        """Title substring matching should skip very short names (<=3 chars)."""
        from kb.entities import Entity, find_entity_mentions

        entities_with_short = [
            *sample_entities,
            Entity(id=10, name="Ed Wilson", entity_type="person", aliases=["Ed", "ed-wilson"]),
        ]

        mentions = find_entity_mentions(
            title="Ed Weekly Standup",
            tags=[],
            content="Short content.",
            entities=entities_with_short,
        )

        title_mentions = [m for m in mentions if m.mention_type == "title"]
        ed_ids = {m.entity_id for m in title_mentions if m.entity_id == 10}
        # "Ed" (2 chars) is too short for title substring matching
        assert len(ed_ids) == 0

    def test_title_substring_full_name(self, sample_entities):
        """Full names in titles should match via title substring."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Wren Kasper 1:1",
            tags=[],
            content="Short content.",
            entities=sample_entities,
        )

        title_mentions = [m for m in mentions if m.mention_type == "title"]
        assert 1 in {m.entity_id for m in title_mentions}

    def test_disambiguation_david_m(self, sample_entities):
        """Specific 'Anders M.' should only match Soren Vance."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="Anders M. presented the quarterly results.",
            entities=sample_entities,
        )

        discussed = [m for m in mentions if m.mention_type == "discussed"]
        # "Anders M." should match Soren Vance specifically
        assert any(m.entity_id == 3 for m in discussed)

    def test_combined_mention_types(self, sample_entities):
        """A document can have multiple mention types for different entities."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Wren / Soren",
            tags=["Soren"],
            content="We discussed the Helix Refactor progress.",
            entities=sample_entities,
        )

        types = {m.mention_type for m in mentions}
        assert "participant" in types
        assert "tagged" in types
        assert "discussed" in types


class TestFirstNameDisambiguation:
    """Bare ambiguous first names must not auto-link without corroboration (#36)."""

    def _alex_pair(self):
        from kb.entities import Entity

        return [
            Entity(id=101, name="Alexandre Dupont", entity_type="person", aliases=["Alexandre"]),
            Entity(id=102, name="Alexandre Martin", entity_type="person", aliases=["Alexandre"]),
        ]

    def test_build_first_name_index_flags_ambiguous(self):
        from kb.entities import build_first_name_index

        owners = build_first_name_index(self._alex_pair())
        assert owners.get("alexandre") == {101, 102}

    def test_build_first_name_index_folds_accents(self):
        """An accented name and its ASCII near-twin collide in the ambiguity index."""
        from kb.entities import Entity, build_first_name_index

        ents = [
            Entity(id=201, name="Jérémy Cotineau", entity_type="person", aliases=["Jérémy"]),
            Entity(id=202, name="Jeremy Brown", entity_type="person", aliases=["Jeremy"]),
        ]
        owners = build_first_name_index(ents)
        assert owners.get("jeremy") == {201, 202}

    def test_ambiguous_bare_first_name_not_linked_without_context(self):
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Weekly sync",
            tags=[],
            content="Alexandre opened the meeting and walked through the roadmap.",
            entities=self._alex_pair(),
        )
        discussed = {m.entity_id for m in mentions if m.mention_type == "discussed"}
        assert discussed == set(), "ambiguous bare first name should not auto-link"

    def test_ambiguous_bare_name_linked_when_full_name_corroborates(self):
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Weekly sync",
            tags=[],
            content="Alexandre Dupont opened. Later Alexandre summarised the actions.",
            entities=self._alex_pair(),
        )
        discussed = {m.entity_id for m in mentions if m.mention_type == "discussed"}
        assert 101 in discussed, "full-name match should corroborate this Alexandre"
        assert 102 not in discussed, "the other Alexandre stays unlinked"

    def test_ambiguous_bare_name_linked_when_attendee_corroborates(self):
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Weekly sync",
            tags=[],
            content="Alexandre opened the meeting and walked through the roadmap.",
            entities=self._alex_pair(),
            attendees=["Alexandre Dupont"],
        )
        discussed = {m.entity_id for m in mentions if m.mention_type == "discussed"}
        assert 101 in discussed, "attendee should corroborate this Alexandre"
        assert 102 not in discussed

    def test_accent_twin_gates_an_otherwise_unambiguous_match(self):
        """A bare ASCII 'Jeremy' that could be the accented 'Jérémy' is gated."""
        from kb.entities import Entity, find_entity_mentions

        ents = [
            Entity(id=201, name="Jérémy Cotineau", entity_type="person", aliases=["Jérémy"]),
            Entity(id=202, name="Jeremy Brown", entity_type="person", aliases=["Jeremy"]),
        ]
        mentions = find_entity_mentions(
            title="Review",
            tags=[],
            content="Jeremy attended the review and gave feedback.",
            entities=ents,
        )
        discussed = {m.entity_id for m in mentions if m.mention_type == "discussed"}
        assert discussed == set(), "accent-fold collision should gate the bare match"

    def test_ambiguous_bare_name_in_title_gated(self):
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Alexandre Sync Notes",
            tags=[],
            content="Short content with no names.",
            entities=self._alex_pair(),
        )
        title_ids = {m.entity_id for m in mentions if m.mention_type == "title"}
        assert title_ids == set(), "ambiguous bare first name in title should be gated"

    def test_unambiguous_first_name_still_links(self):
        from kb.entities import Entity, find_entity_mentions

        ents = [
            Entity(id=301, name="Soren Vance", entity_type="person", aliases=["Soren"]),
        ]
        mentions = find_entity_mentions(
            title="Weekly sync",
            tags=[],
            content="Soren walked the team through the new design.",
            entities=ents,
        )
        discussed = {m.entity_id for m in mentions if m.mention_type == "discussed"}
        assert 301 in discussed, "an unambiguous first name must still auto-link"


class TestSeedEntitiesNonDestructive:
    """seed_entities() must be non-destructive: upsert entities, preserve mentions."""

    def _make_project_root(self, tmp_path):
        """Create a minimal project root with one person file."""
        people = tmp_path / "memory" / "people"
        people.mkdir(parents=True)
        (people / "jane-doe.md").write_text("# Jane Doe\n\n**Role:** Engineer\n**Team:** Core\n")
        return tmp_path

    def test_seed_preserves_entity_mentions(self, tmp_path):
        """seed_entities should NOT delete entity_mentions for unchanged entities."""
        from kb.db import Database
        from kb.entities import seed_entities

        root = self._make_project_root(tmp_path)
        db = Database(tmp_path / "data")
        conn = db.get_sqlite_conn()

        # First seed
        seed_entities(db, root)

        # Get entity ID for Jane Doe
        row = conn.execute("SELECT id FROM entities WHERE name = 'Jane Doe'").fetchone()
        assert row is not None
        jane_id = row["id"]

        # Manually insert a fake document and entity mention
        conn.execute(
            "INSERT INTO documents (path, title, content_hash, chunk_count) VALUES (?, ?, ?, ?)",
            ("fake/doc.md", "Fake Doc", "abc123", 1),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
            (jane_id, doc_id, "discussed"),
        )
        conn.commit()

        # Re-seed — mentions should survive
        seed_entities(db, root)

        mentions = conn.execute(
            "SELECT * FROM entity_mentions WHERE entity_id = ? AND document_id = ?",
            (jane_id, doc_id),
        ).fetchall()
        assert len(mentions) == 1, "entity_mentions should survive re-seed"
        db.close()

    def test_seed_updates_entity_fields(self, tmp_path):
        """Existing entity fields (aliases, metadata) should be updated on re-seed."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        root = self._make_project_root(tmp_path)
        db = Database(tmp_path / "data")

        # First seed
        seed_entities(db, root)
        entities = load_entities(db)
        jane = next(e for e in entities if e.name == "Jane Doe")
        original_id = jane.id
        assert jane.metadata.get("role") == "Engineer"

        # Modify the source file — change role
        (root / "memory" / "people" / "jane-doe.md").write_text(
            "# Jane Doe\n\n**Role:** Staff Engineer\n**Team:** Platform\n"
        )

        # Re-seed
        seed_entities(db, root)
        entities = load_entities(db)
        jane = next(e for e in entities if e.name == "Jane Doe")
        assert jane.id == original_id, "Entity ID should be stable across re-seeds"
        assert jane.metadata.get("role") == "Staff Engineer"
        assert jane.metadata.get("team") == "Platform"
        db.close()

    def test_seed_adds_new_entities(self, tmp_path):
        """New entities from source files should be inserted on re-seed."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        root = self._make_project_root(tmp_path)
        db = Database(tmp_path / "data")

        seed_entities(db, root)
        count_before = len(load_entities(db))

        # Add a new person file
        (root / "memory" / "people" / "soren-vance.md").write_text(
            "# Soren Vance\n\n**Role:** Designer\n"
        )

        seed_entities(db, root)
        count_after = len(load_entities(db))
        assert count_after == count_before + 1

        entities = load_entities(db)
        names = {e.name for e in entities}
        assert "Soren Vance" in names
        db.close()

    def test_seed_removes_deleted_entities(self, tmp_path):
        """Entities no longer in source files should be deleted (with their mentions)."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        root = self._make_project_root(tmp_path)
        db = Database(tmp_path / "data")
        conn = db.get_sqlite_conn()

        seed_entities(db, root)
        jane = next(e for e in load_entities(db) if e.name == "Jane Doe")

        # Add a mention for Jane
        conn.execute(
            "INSERT INTO documents (path, title, content_hash, chunk_count) VALUES (?, ?, ?, ?)",
            ("fake/doc.md", "Fake", "abc123", 1),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (?, ?, ?)",
            (jane.id, doc_id, "discussed"),
        )
        conn.commit()

        # Delete the person file
        (root / "memory" / "people" / "jane-doe.md").unlink()

        # Re-seed
        seed_entities(db, root)

        entities = load_entities(db)
        names = {e.name for e in entities}
        assert "Jane Doe" not in names

        # Mentions should also be gone
        mentions = conn.execute(
            "SELECT * FROM entity_mentions WHERE entity_id = ?", (jane.id,)
        ).fetchall()
        assert len(mentions) == 0
        db.close()

    def test_entity_ids_stable_across_reseeds(self, tmp_path):
        """Entity IDs should not change when re-seeding unchanged entities."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        root = self._make_project_root(tmp_path)
        db = Database(tmp_path / "data")

        seed_entities(db, root)
        ids_before = {e.name: e.id for e in load_entities(db)}

        seed_entities(db, root)
        ids_after = {e.name: e.id for e in load_entities(db)}

        for name in ids_before:
            assert ids_before[name] == ids_after[name], f"ID changed for {name}"
        db.close()


class TestFullSeeding:
    def test_seed_entities(self, tmp_db, entity_root):
        """Test full seeding against synthetic memory files."""
        from kb.entities import seed_entities

        n = seed_entities(tmp_db, entity_root)
        # 2 people + 1 project + 1 company (Lattice Co) + 5 teams = 9
        assert n >= 9, f"Expected >= 9 entities, got {n}"

    def test_seed_correct_people_count(self, tmp_db, entity_root):
        """Seeding should create people from memory files."""
        from kb.entities import load_entities, seed_entities

        seed_entities(tmp_db, entity_root)
        entities = load_entities(tmp_db)

        people = [e for e in entities if e.entity_type == "person"]
        assert len(people) >= 2

    def test_seed_correct_project_count(self, tmp_db, entity_root):
        """Seeding should create projects from memory files."""
        from kb.entities import load_entities, seed_entities

        seed_entities(tmp_db, entity_root)
        entities = load_entities(tmp_db)

        projects = [e for e in entities if e.entity_type == "project"]
        assert len(projects) >= 1

    def test_seed_idempotent(self, tmp_db, entity_root):
        """Seeding should be idempotent — re-running gives same count."""
        from kb.entities import seed_entities

        n1 = seed_entities(tmp_db, entity_root)
        n2 = seed_entities(tmp_db, entity_root)
        assert n1 == n2

    def test_seed_has_acme_company(self, tmp_db, entity_root):
        """Seeding should create an Lattice Co company entity."""
        from kb.entities import load_entities, seed_entities

        seed_entities(tmp_db, entity_root)
        entities = load_entities(tmp_db)

        companies = [e for e in entities if e.entity_type == "company"]
        assert len(companies) >= 1
        co = next(e for e in companies if e.name == "Lattice Co")
        assert "AC" in co.aliases

    def test_seed_has_teams(self, tmp_db, entity_root):
        """Seeding should create team entities."""
        from kb.entities import load_entities, seed_entities

        seed_entities(tmp_db, entity_root)
        entities = load_entities(tmp_db)

        teams = [e for e in entities if e.entity_type == "team"]
        assert len(teams) == 5


class TestPinnedMigration:
    def test_pinned_column_exists(self):
        """Fresh DB should have pinned column on entities table."""
        from kb.db import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            columns = [r[1] for r in conn.execute("PRAGMA table_info(entities)").fetchall()]
            assert "pinned" in columns
            db.close()

    def test_pinned_defaults_to_zero(self):
        """New entities should have pinned=0 by default."""
        from kb.db import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir))
            conn = db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata) VALUES (?, ?, ?, ?)",
                ("Test Person", "person", "[]", "{}"),
            )
            conn.commit()
            row = conn.execute("SELECT pinned FROM entities WHERE name = 'Test Person'").fetchone()
            assert row["pinned"] == 0
            db.close()


class TestPinnedParsing:
    def test_parse_person_pinned_true(self, tmp_path):
        """Person file with **Pinned:** true should have pinned=True."""
        from kb.entities import _parse_person_file

        person_file = tmp_path / "pinned-person.md"
        person_file.write_text("# Pinned Person\n\n**Role:** CEO\n**Pinned:** true\n")
        result = _parse_person_file(person_file)
        assert result.pinned is True

    def test_parse_person_no_pinned(self, tmp_path):
        """Person file without Pinned field should default to pinned=False."""
        from kb.entities import _parse_person_file

        person_file = tmp_path / "normal-person.md"
        person_file.write_text("# Normal Person\n\n**Role:** Engineer\n")
        result = _parse_person_file(person_file)
        assert result.pinned is False

    def test_parse_project_pinned(self, tmp_path):
        """Project file with **Pinned:** true should have pinned=True."""
        from kb.entities import _parse_project_file

        project_file = tmp_path / "pinned-project.md"
        project_file.write_text("# Pinned Project\n\n**Status:** Active\n**Pinned:** true\n")
        result = _parse_project_file(project_file)
        assert result.pinned is True

    def test_seed_sets_pinned_column(self, tmp_path):
        """seed_entities should set pinned=1 for entities with **Pinned:** true in source."""
        from kb.db import Database
        from kb.entities import seed_entities

        people = tmp_path / "memory" / "people"
        people.mkdir(parents=True)
        (people / "pinned-ceo.md").write_text("# Pinned CEO\n\n**Role:** CEO\n**Pinned:** true\n")
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT pinned FROM entities WHERE name = 'Pinned CEO'").fetchone()
        assert row["pinned"] == 1
        db.close()

    def test_seed_preserves_dynamic_pin(self, tmp_path):
        """Re-seeding should NOT clear a dynamically set pinned=1."""
        from kb.db import Database
        from kb.entities import seed_entities

        people = tmp_path / "memory" / "people"
        people.mkdir(parents=True)
        (people / "jane-doe.md").write_text("# Jane Doe\n\n**Role:** Engineer\n")
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        conn = db.get_sqlite_conn()
        # Dynamically pin Jane
        conn.execute("UPDATE entities SET pinned = 1 WHERE name = 'Jane Doe'")
        conn.commit()
        # Re-seed — pinned should survive
        seed_entities(db, tmp_path)
        row = conn.execute("SELECT pinned FROM entities WHERE name = 'Jane Doe'").fetchone()
        assert row["pinned"] == 1
        db.close()

    def test_seed_sets_updated_at_on_new_entities(self, tmp_path):
        """New entities inserted by seed_entities should have updated_at set to today."""
        from datetime import date

        from kb.db import Database
        from kb.entities import seed_entities

        people = tmp_path / "memory" / "people"
        people.mkdir(parents=True)
        (people / "jane-doe.md").write_text("# Jane Doe\n\n**Role:** Engineer\n")
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT updated_at FROM entities WHERE name = 'Jane Doe'").fetchone()
        assert row["updated_at"] == date.today().isoformat()
        db.close()

    def test_entity_dataclass_has_pinned(self):
        """Entity dataclass should have a pinned field."""
        from kb.entities import Entity

        e = Entity(id=1, name="Test", entity_type="person")
        assert hasattr(e, "pinned")
        assert e.pinned is False

    def test_load_entities_reads_pinned(self, tmp_path):
        """load_entities should read the pinned column."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        people = tmp_path / "memory" / "people"
        people.mkdir(parents=True)
        (people / "pinned-ceo.md").write_text("# Pinned CEO\n\n**Role:** CEO\n**Pinned:** true\n")
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        entities = load_entities(db)
        ceo = next(e for e in entities if e.name == "Pinned CEO")
        assert ceo.pinned is True
        db.close()


class TestPydanticModels:
    def test_entity_is_pydantic(self):
        from pydantic import BaseModel

        from kb.entities import Entity

        assert issubclass(Entity, BaseModel)

    def test_entity_mention_is_pydantic(self):
        from pydantic import BaseModel

        from kb.entities import EntityMention

        assert issubclass(EntityMention, BaseModel)

    def test_load_entities_returns_pydantic(self, tmp_db, project_root):
        from pydantic import BaseModel

        from kb.entities import load_entities, seed_entities

        seed_entities(tmp_db, project_root)
        entities = load_entities(tmp_db)
        if entities:
            assert isinstance(entities[0], BaseModel)


class TestParserPydanticOutput:
    def test_parse_person_returns_entity_data(self, tmp_path):
        from kb.entities import _parse_person_file
        from kb.types import EntityData

        person_file = tmp_path / "test.md"
        person_file.write_text("# Test Person\n\n**Role:** Engineer\n")
        result = _parse_person_file(person_file)
        assert isinstance(result, EntityData)
        assert result.name == "Test Person"
        assert result.entity_type == "person"

    def test_parse_project_returns_entity_data(self, tmp_path):
        from kb.entities import _parse_project_file
        from kb.types import EntityData

        proj_file = tmp_path / "test-project.md"
        proj_file.write_text("# Test Project\n\n**Status:** Active\n")
        result = _parse_project_file(proj_file)
        assert isinstance(result, EntityData)
        assert result.name == "Test Project"
        assert result.entity_type == "project"


class TestYamlFrontmatterParsing:
    """Parse person/project files with YAML frontmatter."""

    def test_parse_person_yaml_frontmatter(self, tmp_path):
        from kb.entities import _parse_person_file

        f = tmp_path / "jane-doe.md"
        f.write_text(
            "---\n"
            "aliases: [JD, Jane D.]\n"
            "email: jane@example.com\n"
            "role: Engineering Lead\n"
            'team: "[[Core]]"\n'
            'reports_to: "[[CTO]]"\n'
            "pinned: true\n"
            "---\n"
            "# Jane Doe\n\n"
            "## Notes\n\nSome notes here.\n"
        )
        result = _parse_person_file(f)
        assert result.name == "Jane Doe"
        assert result.entity_type == "person"
        assert "JD" in result.aliases
        assert "Jane D." in result.aliases
        assert "Jane" in result.aliases  # auto first-name
        assert "jane-doe" in result.aliases  # auto file-stem
        assert result.metadata["email"] == "jane@example.com"
        assert result.metadata["role"] == "Engineering Lead"
        assert result.metadata["team"] == "Core"  # wikilinks stripped
        assert result.metadata["reports_to"] == "CTO"  # wikilinks stripped
        assert result.pinned is True

    def test_parse_person_yaml_no_aliases(self, tmp_path):
        from kb.entities import _parse_person_file

        f = tmp_path / "solo.md"
        f.write_text("---\nemail: solo@example.com\nrole: Dev\n---\n# Solo Dev\n")
        result = _parse_person_file(f)
        assert result.name == "Solo Dev"
        assert "Solo" in result.aliases
        assert "solo" in result.aliases

    def test_parse_person_old_format_still_works(self, tmp_path):
        from kb.entities import _parse_person_file

        f = tmp_path / "old-style.md"
        f.write_text(
            "# Old Style\n\n**Also known as:** OS\n**Role:** Legacy Dev\n**Team:** Platform\n"
        )
        result = _parse_person_file(f)
        assert result.name == "Old Style"
        assert "OS" in result.aliases
        assert result.metadata["role"] == "Legacy Dev"
        assert result.metadata["team"] == "Platform"

    def test_parse_project_yaml_frontmatter(self, tmp_path):
        from kb.entities import _parse_project_file

        f = tmp_path / "my-project.md"
        f.write_text(
            "---\n"
            "aliases: [MP, MyProj]\n"
            "status: Active\n"
            "started: January 2026\n"
            'lead: "[[Wren Kasper]] — Tech Lead"\n'
            "---\n"
            "# My Project\n\n"
            "## Overview\n\nProject details.\n"
        )
        result = _parse_project_file(f)
        assert result.name == "My Project"
        assert result.entity_type == "project"
        assert "MP" in result.aliases
        assert "MyProj" in result.aliases
        assert "my-project" in result.aliases
        assert result.metadata["status"] == "Active"
        assert result.metadata["started"] == "January 2026"
        assert result.metadata["lead"] == "Wren Kasper — Tech Lead"

    def test_parse_project_old_format_still_works(self, tmp_path):
        from kb.entities import _parse_project_file

        f = tmp_path / "old-proj.md"
        f.write_text("# Old Project\n\n**Codename/Also called:** OP\n**Status:** Done\n")
        result = _parse_project_file(f)
        assert result.name == "Old Project"
        assert "OP" in result.aliases
        assert result.metadata["status"] == "Done"

    def test_yaml_custom_fields_preserved(self, tmp_path):
        from kb.entities import _parse_person_file

        f = tmp_path / "custom.md"
        f.write_text("---\nrole: Dev\npreferred_lang: Python\n---\n# Custom Dev\n")
        result = _parse_person_file(f)
        assert result.metadata["preferred_lang"] == "Python"

    def test_yaml_pcm_fields(self, tmp_path):
        from kb.entities import _parse_person_file

        f = tmp_path / "pcm.md"
        f.write_text(
            "---\n"
            "role: Dev\n"
            'pcm_base: "Persister (Persévérant)"\n'
            'pcm_phase: "Thinker (Analyseur)"\n'
            "---\n"
            "# PCM Person\n"
        )
        result = _parse_person_file(f)
        assert result.metadata["pcm_base"] == "Persister (Persévérant)"
        assert result.metadata["pcm_phase"] == "Thinker (Analyseur)"


class TestStripWikilinks:
    def test_strip_simple(self):
        from kb.entities import strip_wikilinks

        assert strip_wikilinks("[[Foo]]") == "Foo"

    def test_strip_multiple(self):
        from kb.entities import strip_wikilinks

        assert strip_wikilinks("[[Foo]] and [[Bar]]") == "Foo and Bar"

    def test_strip_in_sentence(self):
        from kb.entities import strip_wikilinks

        assert (
            strip_wikilinks("Reports to [[Idris Kalmar]] (CTO)") == "Reports to Idris Kalmar (CTO)"
        )

    def test_no_wikilinks(self):
        from kb.entities import strip_wikilinks

        assert strip_wikilinks("plain text") == "plain text"

    def test_empty_string(self):
        from kb.entities import strip_wikilinks

        assert strip_wikilinks("") == ""

    def test_nested_brackets(self):
        from kb.entities import strip_wikilinks

        # Edge case: only strips [[...]], not single brackets
        assert strip_wikilinks("[not a link]") == "[not a link]"


class TestSourceIdEntityLinking:
    """Tests for source ID extraction, alias augmentation, and entity linking."""

    def test_source_ids_added_as_aliases(self, tmp_path):
        """Projects with sources should get src:-prefixed aliases after seeding."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        projects = tmp_path / "memory" / "projects"
        projects.mkdir(parents=True)
        (projects / "ai-adoption.md").write_text(
            "---\n"
            "status: Active\n"
            "sources:\n"
            "- type: slack\n"
            "  channel: C08HJC8MWQN\n"
            '  name: "#proj-agentic"\n'
            "- type: linear\n"
            "  id: PRJ-123\n"
            "---\n"
            "# AI Adoption\n"
        )
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        entities = load_entities(db)
        project = next(e for e in entities if e.name == "AI Adoption")
        assert "src:C08HJC8MWQN" in project.aliases
        assert "src:PRJ-123" in project.aliases
        db.close()

    def test_short_source_ids_ignored(self, tmp_path):
        """Source IDs shorter than 3 chars should not become aliases."""
        from kb.db import Database
        from kb.entities import load_entities, seed_entities

        projects = tmp_path / "memory" / "projects"
        projects.mkdir(parents=True)
        (projects / "short-id.md").write_text(
            "---\nstatus: Active\nsources:\n- type: custom\n  id: AB\n---\n# Short ID Project\n"
        )
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        entities = load_entities(db)
        project = next(e for e in entities if e.name == "Short ID Project")
        assert not any(a.startswith("src:") for a in project.aliases)
        db.close()

    def test_source_id_entity_linking(self, tmp_path):
        """Document mentioning a source ID should link to the project."""
        from kb.db import Database
        from kb.entities import find_entity_mentions, load_entities, seed_entities

        projects = tmp_path / "memory" / "projects"
        projects.mkdir(parents=True)
        (projects / "my-project.md").write_text(
            "---\n"
            "status: Active\n"
            "sources:\n"
            "- type: slack\n"
            "  channel: C08HJC8MWQN\n"
            "---\n"
            "# My Project\n"
        )
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        entities = load_entities(db)

        # Simulate a document that mentions the Slack channel ID
        mentions = find_entity_mentions(
            title="Weekly sync",
            tags=[],
            content="Discussion in C08HJC8MWQN about progress.",
            entities=entities,
        )
        project = next(e for e in entities if e.name == "My Project")
        project_mentions = [m for m in mentions if m.entity_id == project.id]
        assert len(project_mentions) >= 1
        db.close()

    def test_source_ref_mention_type(self, tmp_path):
        """Source ID match should produce mention_type='source_ref'."""
        from kb.db import Database
        from kb.entities import find_entity_mentions, load_entities, seed_entities

        projects = tmp_path / "memory" / "projects"
        projects.mkdir(parents=True)
        (projects / "linked-proj.md").write_text(
            "---\nstatus: Active\nsources:\n- type: linear\n  id: PRJ-456\n---\n# Linked Project\n"
        )
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        entities = load_entities(db)

        mentions = find_entity_mentions(
            title="Some meeting",
            tags=[],
            content="Reviewing PRJ-456 status and updates.",
            entities=entities,
        )
        project = next(e for e in entities if e.name == "Linked Project")
        source_ref_mentions = [
            m for m in mentions if m.entity_id == project.id and m.mention_type == "source_ref"
        ]
        assert len(source_ref_mentions) == 1

    def test_source_ids_skipped_in_display(self, tmp_path):
        """src:-prefixed aliases should be filtered from _derive_aliases()."""
        from kb.entities import Entity
        from kb.writeback import _derive_aliases

        entity = Entity(
            id=1,
            name="My Project",
            entity_type="project",
            aliases=["MP", "src:C08HJC8MWQN", "src:PRJ-123", "my-project"],
            metadata={},
            source_path="memory/projects/my-project.md",
        )
        derived = _derive_aliases(entity)
        assert "MP" in derived
        assert "src:C08HJC8MWQN" not in derived
        assert "src:PRJ-123" not in derived
        # File stems also filtered
        assert "my-project" not in derived

    def test_source_id_no_false_positive_in_title(self, tmp_path):
        """Source IDs in content should not accidentally match via title tier."""
        from kb.db import Database
        from kb.entities import find_entity_mentions, load_entities, seed_entities

        projects = tmp_path / "memory" / "projects"
        projects.mkdir(parents=True)
        (projects / "proj-x.md").write_text(
            "---\n"
            "status: Active\n"
            "sources:\n"
            "- type: slack\n"
            "  channel: CABCDEF1234\n"
            "---\n"
            "# Project X\n"
        )
        db = Database(tmp_path / "data")
        seed_entities(db, tmp_path)
        entities = load_entities(db)

        # Content mentions channel but title is unrelated
        mentions = find_entity_mentions(
            title="Unrelated meeting",
            tags=[],
            content="Messages in CABCDEF1234 were reviewed.",
            entities=entities,
        )
        project = next(e for e in entities if e.name == "Project X")
        source_mentions = [
            m for m in mentions if m.entity_id == project.id and m.mention_type == "source_ref"
        ]
        assert len(source_mentions) == 1
        db.close()
