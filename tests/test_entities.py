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
    (people / "alice-smith.md").write_text(
        "# Wren Smith\n\n**Also known as:** Wren\n**Role:** Engineer\n**Team:** Platform\n"
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
            Entity(id=1, name="Wren Kasper", entity_type="person", aliases=["Wren", "alice-reed"]),
            Entity(
                id=2,
                name="Soren Vance",
                entity_type="person",
                aliases=["Soren", "thomas-beaumont"],
            ),
            Entity(
                id=3,
                name="Soren Vance",
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
        """First names >3 chars should match in content (threshold lowered from 6 to 3)."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Random meeting",
            tags=[],
            content="Anders presented the quarterly results.",
            entities=sample_entities,
        )

        discussed = [m for m in mentions if m.mention_type == "discussed"]
        david_ids = {m.entity_id for m in discussed if m.entity_id in (3, 4)}
        # "Anders" (5 chars) should now match — threshold lowered to 3
        assert len(david_ids) >= 1

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
        """Entity names appearing as substrings in the title should match as 'title' type."""
        from kb.entities import find_entity_mentions

        mentions = find_entity_mentions(
            title="Anders Sync Notes",
            tags=[],
            content="Short content with no names.",
            entities=sample_entities,
        )

        title_mentions = [m for m in mentions if m.mention_type == "title"]
        title_ids = {m.entity_id for m in title_mentions}
        # Both Davids should match via title substring
        assert 3 in title_ids  # Soren Vance
        assert 4 in title_ids  # Kit Larsen (alias "Anders")

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
        (root / "memory" / "people" / "bob-smith.md").write_text(
            "# Soren Smith\n\n**Role:** Designer\n"
        )

        seed_entities(db, root)
        count_after = len(load_entities(db))
        assert count_after == count_before + 1

        entities = load_entities(db)
        names = {e.name for e in entities}
        assert "Soren Smith" in names
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
