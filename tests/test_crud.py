"""Tests for the generic CRUD engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kb.db import Database


@pytest.fixture
def crud_env():
    """Create a temp project root with memory dirs and a test DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "memory" / "people").mkdir(parents=True)
        (root / "memory" / "projects").mkdir(parents=True)
        db = Database(root / "data")
        yield db, root
        db.close()


class TestTypeRegistry:
    def test_type_registry_uses_pydantic(self):
        from kb.crud import _TYPE_REGISTRY
        from kb.types import EntityTypeConfig

        assert len(_TYPE_REGISTRY) > 0
        for key, value in _TYPE_REGISTRY.items():
            assert isinstance(value, EntityTypeConfig), (
                f"_TYPE_REGISTRY[{key!r}] is {type(value).__name__}, expected EntityTypeConfig"
            )

    def test_entity_type_config_fields(self):
        from kb.crud import _TYPE_REGISTRY

        person = _TYPE_REGISTRY["person"]
        assert person.directory == "memory/people"
        assert person.alias_label == "Also known as"
        assert len(person.fields) > 0

        project = _TYPE_REGISTRY["project"]
        assert project.directory == "memory/projects"
        assert project.alias_label == "Codename/Also called"


class TestPersonCreate:
    def test_create_person_writes_file(self, crud_env):
        from kb.crud import create_entity

        db, root = crud_env
        result = create_entity(
            db,
            root,
            "person",
            "Jane Doe",
            metadata={"role": "Engineer", "email": "jane@example.com", "team": "Core"},
        )
        assert result["name"] == "Jane Doe"
        assert (root / "memory" / "people" / "jane-doe.md").exists()
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "# Jane Doe" in content
        assert "role: Engineer" in content
        assert "email: jane@example.com" in content

    def test_create_person_seeds_sqlite(self, crud_env):
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = 'Jane Doe'").fetchone()
        assert row is not None
        assert row["entity_type"] == "person"

    def test_create_person_with_aliases(self, crud_env):
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", aliases=["JD", "Jane D."])
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "JD" in content
        assert "Jane D." in content

    def test_create_duplicate_raises(self, crud_env):
        from kb.crud import EntityExistsError, create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe")
        with pytest.raises(EntityExistsError):
            create_entity(db, root, "person", "Jane Doe")

    def test_create_entity_sets_updated_at(self, crud_env):
        """New entity via create_entity should have updated_at set to today."""
        from datetime import date

        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT updated_at FROM entities WHERE name = 'Jane Doe'").fetchone()
        assert row["updated_at"] == date.today().isoformat()


class TestProjectCreate:
    def test_create_project_writes_file(self, crud_env):
        from kb.crud import create_entity

        db, root = crud_env
        result = create_entity(
            db,
            root,
            "project",
            "AI Adoption",
            metadata={"status": "Active", "lead": "Wren Kasper"},
        )
        assert result["name"] == "AI Adoption"
        path = root / "memory" / "projects" / "ai-adoption.md"
        assert path.exists()
        content = path.read_text()
        assert "# AI Adoption" in content
        assert "status: Active" in content
        assert "[[Wren Kasper]]" in content


class TestEntityEdit:
    def test_edit_updates_metadata(self, crud_env):
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"role": "Staff Engineer"})
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "role: Staff Engineer" in content

    def test_edit_preserves_freeform(self, crud_env):
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        # Manually add freeform content
        path = root / "memory" / "people" / "jane-doe.md"
        content = path.read_text() + "\n## Notes\nSome important notes here.\n"
        path.write_text(content)
        edit_entity(db, root, "Jane Doe", metadata={"role": "Staff Engineer"})
        content = path.read_text()
        assert "## Notes" in content
        assert "Some important notes" in content
        assert "role: Staff Engineer" in content

    def test_edit_not_found_raises(self, crud_env):
        from kb.crud import EntityNotFoundError, edit_entity

        db, root = crud_env
        with pytest.raises(EntityNotFoundError):
            edit_entity(db, root, "Nobody", metadata={"role": "CEO"})

    def test_edit_entity_sets_updated_at(self, crud_env):
        """Editing an entity sets updated_at to today's date."""
        from datetime import date

        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Wren", metadata={"role": "Engineer"})
        edit_entity(db, root, "Wren", metadata={"role": "VP Product"})

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT updated_at FROM entities WHERE name = 'Wren'").fetchone()
        assert row["updated_at"] is not None
        assert row["updated_at"] == date.today().isoformat()


class TestCustomMetadata:
    """Tests for custom metadata via --meta flag (Issue #15)."""

    def test_edit_sets_custom_metadata(self, crud_env):
        """Setting custom metadata via edit_entity stores it in DB and file."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": "French"})

        # Check file
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "preferred_lang: French" in content

        # Check DB
        conn = db.get_sqlite_conn()
        import json

        row = conn.execute("SELECT metadata FROM entities WHERE name = 'Jane Doe'").fetchone()
        meta = json.loads(row["metadata"])
        assert meta["preferred_lang"] == "French"
        assert meta["role"] == "Engineer"  # original still there

    def test_edit_multiple_custom_fields(self, crud_env):
        """Multiple custom metadata fields can be set in one call."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(
            db,
            root,
            "Jane Doe",
            metadata={"preferred_lang": "French", "timezone": "CET"},
        )
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "preferred_lang: French" in content
        assert "timezone: CET" in content

    def test_edit_remove_custom_field(self, crud_env):
        """Setting a custom field to empty string removes it."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": "French"})

        # Verify it's there
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "preferred_lang: French" in content

        # Remove it
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": ""})
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "preferred_lang:" not in content

        # Verify DB
        import json

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'Jane Doe'").fetchone()
        meta = json.loads(row["metadata"])
        assert "preferred_lang" not in meta

    def test_custom_metadata_in_find_json(self, crud_env):
        """Custom metadata appears in entity metadata dict."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": "French"})

        from kb.entities import load_entity_by_id

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT id FROM entities WHERE name = 'Jane Doe'").fetchone()
        entity = load_entity_by_id(db, row["id"])
        assert entity is not None
        assert entity.metadata["preferred_lang"] == "French"
        assert entity.metadata["role"] == "Engineer"

    def test_parser_picks_up_arbitrary_fields(self, crud_env):
        """Parser reads any **Key:** Value line from person files."""
        from kb.entities import _parse_person_file

        _db, root = crud_env
        path = root / "memory" / "people" / "jane-doe.md"
        path.write_text(
            "# Jane Doe\n\n**Role:** Engineer\n**Preferred Lang:** French\n**Timezone:** CET\n"
        )
        data = _parse_person_file(path)
        assert data.metadata["role"] == "Engineer"
        assert data.metadata["preferred_lang"] == "French"
        assert data.metadata["timezone"] == "CET"

    def test_parser_picks_up_arbitrary_project_fields(self, crud_env):
        """Parser reads any **Key:** Value line from project files."""
        from kb.entities import _parse_project_file

        _db, root = crud_env
        path = root / "memory" / "projects" / "my-project.md"
        path.write_text(
            "# My Project\n\n**Status:** Active\n**Priority:** High\n**Budget:** 100k\n"
        )
        data = _parse_project_file(path)
        assert data.metadata["status"] == "Active"
        assert data.metadata["priority"] == "High"
        assert data.metadata["budget"] == "100k"

    def test_custom_metadata_on_project_edit(self, crud_env):
        """Custom metadata works on projects too."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "project", "AI Adoption", metadata={"status": "Active"})
        edit_entity(db, root, "AI Adoption", metadata={"priority": "High"})

        content = (root / "memory" / "projects" / "ai-adoption.md").read_text()
        assert "priority: High" in content
        assert "status: Active" in content

    def test_custom_metadata_roundtrip(self, crud_env):
        """Custom metadata survives create -> edit -> re-seed cycle."""
        from kb.crud import create_entity, edit_entity
        from kb.entities import seed_entities

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": "French"})

        # Re-seed from files (simulates restart)
        seed_entities(db, root)

        import json

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'Jane Doe'").fetchone()
        meta = json.loads(row["metadata"])
        assert meta["role"] == "Engineer"
        assert meta["preferred_lang"] == "French"

    def test_custom_key_normalization(self, crud_env):
        """Custom keys use snake_case in metadata dict, Title Case in file."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe")
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": "French"})

        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        # File should have snake_case key in YAML frontmatter
        assert "preferred_lang: French" in content

        # DB should have snake_case
        import json

        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'Jane Doe'").fetchone()
        meta = json.loads(row["metadata"])
        assert "preferred_lang" in meta

    def test_create_with_custom_metadata(self, crud_env):
        """Custom metadata can be set during entity creation."""
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(
            db,
            root,
            "person",
            "Jane Doe",
            metadata={"role": "Engineer", "preferred_lang": "French"},
        )
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "role: Engineer" in content
        assert "preferred_lang: French" in content


class TestNameToSlugAccentStripping:
    """Tests for ASCII-only file slugs (Issue #18)."""

    def test_slug_strips_accents(self):
        from kb.crud import _name_to_slug

        assert _name_to_slug("Nathanaël Soulat") == "nathanael-soulat"

    def test_slug_strips_multiple_accents(self):
        from kb.crud import _name_to_slug

        assert _name_to_slug("Aurélien Gâteau") == "aurelien-gateau"

    def test_slug_no_accents_unchanged(self):
        from kb.crud import _name_to_slug

        assert _name_to_slug("Jane Doe") == "jane-doe"

    def test_slug_german_umlauts(self):
        from kb.crud import _name_to_slug

        assert _name_to_slug("Müller Straße") == "muller-strae"

    def test_slug_cedilla(self):
        from kb.crud import _name_to_slug

        assert _name_to_slug("François Garçon") == "francois-garcon"

    def test_create_accented_person_ascii_filename(self, crud_env):
        """person create with accented name creates ASCII filename but keeps accented heading."""
        from kb.crud import create_entity

        db, root = crud_env
        result = create_entity(db, root, "person", "Nathanaël Soulat")
        # File should have ASCII-only name
        assert (root / "memory" / "people" / "nathanael-soulat.md").exists()
        # Heading should keep the accent
        content = (root / "memory" / "people" / "nathanael-soulat.md").read_text()
        assert "# Nathanaël Soulat" in content
        # Result path should use ASCII slug
        assert result["path"] == "memory/people/nathanael-soulat.md"

    def test_create_accented_person_seeds_sqlite_with_accented_name(self, crud_env):
        """DB entity name keeps accents even though filename is ASCII."""
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Nathanaël Soulat")
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = 'Nathanaël Soulat'").fetchone()
        assert row is not None
        assert row["entity_type"] == "person"
        # source_path should use ASCII slug
        assert row["source_path"] == "memory/people/nathanael-soulat.md"


class TestAccentInsensitiveFind:
    """Tests for accent-insensitive entity lookup (Issue #18)."""

    def test_find_entity_accent_insensitive(self, crud_env):
        """find_entity matches 'Nathanael' to stored 'Nathanaël'."""
        from kb.config import find_entity
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Nathanaël Soulat")
        conn = db.get_sqlite_conn()
        row = find_entity(conn, "Nathanael Soulat")
        assert row is not None
        assert row["name"] == "Nathanaël Soulat"

    def test_find_entity_accent_insensitive_partial(self, crud_env):
        """find_entity matches 'Aurelien' to stored 'Aurélien'."""
        from kb.config import find_entity
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Aurélien Gâteau", metadata={"role": "Engineer"})
        conn = db.get_sqlite_conn()
        row = find_entity(conn, "Aurelien")
        assert row is not None
        assert row["name"] == "Aurélien Gâteau"

    def test_find_entity_exact_accented_still_works(self, crud_env):
        """Exact accented name still works (fast path)."""
        from kb.config import find_entity
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Nathanaël Soulat")
        conn = db.get_sqlite_conn()
        row = find_entity(conn, "Nathanaël Soulat")
        assert row is not None
        assert row["name"] == "Nathanaël Soulat"

    def test_find_entity_case_and_accent_insensitive(self, crud_env):
        """Lookup is both case-insensitive AND accent-insensitive."""
        from kb.config import find_entity
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Nathanaël Soulat")
        conn = db.get_sqlite_conn()
        row = find_entity(conn, "nathanael soulat")
        assert row is not None
        assert row["name"] == "Nathanaël Soulat"

    def test_find_entities_accent_insensitive_returns_all(self, crud_env):
        """find_entities returns all accent-insensitive matches."""
        from kb.config import find_entities
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Nathanaël Soulat")
        conn = db.get_sqlite_conn()
        rows = find_entities(conn, "Nathanael Soulat")
        assert len(rows) == 1
        assert rows[0]["name"] == "Nathanaël Soulat"

    def test_find_entity_accent_insensitive_alias(self, crud_env):
        """Accent-insensitive match works on aliases too."""
        from kb.config import find_entity
        from kb.crud import create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", aliases=["Héloïse"])
        conn = db.get_sqlite_conn()
        row = find_entity(conn, "Heloise")
        assert row is not None
        assert row["name"] == "Jane Doe"


class TestYamlFrontmatterCreate:
    """New entities should be created with YAML frontmatter."""

    def test_build_markdown_person_yaml(self):
        from kb.crud import _build_markdown

        md = _build_markdown(
            "Jane Doe",
            "person",
            metadata={"email": "j@x.com", "role": "Dev", "team": "Core"},
            aliases=["JD"],
        )
        assert md.startswith("---\n")
        assert "aliases:" in md
        assert "JD" in md
        assert "email: j@x.com" in md
        assert "role: Dev" in md
        assert "[[Core]]" in md  # team is entity-ref field
        assert "# Jane Doe" in md

    def test_build_markdown_project_yaml(self):
        from kb.crud import _build_markdown

        md = _build_markdown(
            "Big Project",
            "project",
            metadata={"status": "Active", "lead": "Wren"},
            aliases=["BP"],
        )
        assert md.startswith("---\n")
        assert "status: Active" in md
        assert "[[Wren]]" in md  # lead is entity-ref field
        assert "# Big Project" in md

    def test_build_markdown_no_aliases(self):
        from kb.crud import _build_markdown

        md = _build_markdown("Solo", "person", metadata={"role": "Dev"})
        assert "aliases" not in md
        assert "role: Dev" in md
        assert "# Solo" in md

    def test_build_markdown_empty_metadata(self):
        from kb.crud import _build_markdown

        md = _build_markdown("Empty", "person")
        assert md == "# Empty\n"  # no frontmatter if nothing to write


class TestEntityDelete:
    def test_delete_removes_file_and_db(self, crud_env):
        from kb.crud import create_entity, delete_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe")
        delete_entity(db, root, "Jane Doe")
        assert not (root / "memory" / "people" / "jane-doe.md").exists()
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = 'Jane Doe'").fetchone()
        assert row is None

    def test_delete_not_found_raises(self, crud_env):
        from kb.crud import EntityNotFoundError, delete_entity

        db, root = crud_env
        with pytest.raises(EntityNotFoundError):
            delete_entity(db, root, "Nobody")


class TestSourcesManagement:
    """Tests for sources: schema on project entities."""

    def test_create_project_with_sources(self, crud_env):
        from kb.crud import create_entity

        db, root = crud_env
        sources = [
            {"type": "linear", "id": "PRJ-123", "url": "https://linear.app/project/123"},
            {"type": "slack", "channel": "C08HJC8MWQN"},
        ]
        create_entity(
            db,
            root,
            "project",
            "AI Adoption",
            metadata={"status": "Active", "sources": sources},
        )
        content = (root / "memory" / "projects" / "ai-adoption.md").read_text()
        assert "sources:" in content
        assert "type: linear" in content
        assert "PRJ-123" in content
        assert "C08HJC8MWQN" in content

    def test_sources_roundtrip(self, crud_env):
        """Sources survive create -> re-seed -> read cycle."""
        import json

        from kb.crud import create_entity
        from kb.entities import seed_entities

        db, root = crud_env
        sources = [{"type": "linear", "id": "PRJ-123"}]
        create_entity(
            db,
            root,
            "project",
            "AI Adoption",
            metadata={"status": "Active", "sources": sources},
        )
        # Re-seed from disk
        seed_entities(db, root)
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'AI Adoption'").fetchone()
        meta = json.loads(row["metadata"])
        assert isinstance(meta["sources"], list)
        assert meta["sources"][0]["type"] == "linear"
        assert meta["sources"][0]["id"] == "PRJ-123"

    def test_edit_add_source(self, crud_env):
        """--source appends to existing sources via __source_add."""
        import json

        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(
            db,
            root,
            "project",
            "AI Adoption",
            metadata={"sources": [{"type": "linear", "id": "PRJ-123"}]},
        )
        edit_entity(
            db,
            root,
            "AI Adoption",
            metadata={
                "__source_add": [{"type": "slack", "channel": "C123"}],
                "__remove_sources": [],
            },
        )
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'AI Adoption'").fetchone()
        meta = json.loads(row["metadata"])
        assert len(meta["sources"]) == 2
        assert meta["sources"][1]["type"] == "slack"

    def test_edit_remove_source_by_type(self, crud_env):
        """--remove-source TYPE removes all entries of that type."""
        import json

        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        sources = [
            {"type": "linear", "id": "PRJ-123"},
            {"type": "slack", "channel": "C123"},
        ]
        create_entity(
            db,
            root,
            "project",
            "AI Adoption",
            metadata={"sources": sources},
        )
        edit_entity(
            db,
            root,
            "AI Adoption",
            metadata={"__source_add": [], "__remove_sources": ["linear"]},
        )
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'AI Adoption'").fetchone()
        meta = json.loads(row["metadata"])
        assert len(meta["sources"]) == 1
        assert meta["sources"][0]["type"] == "slack"

    def test_edit_remove_source_by_type_and_id(self, crud_env):
        """--remove-source TYPE:ID removes only the matching entry."""
        import json

        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        sources = [
            {"type": "linear", "id": "PRJ-123"},
            {"type": "linear", "id": "PRJ-456"},
        ]
        create_entity(
            db,
            root,
            "project",
            "AI Adoption",
            metadata={"sources": sources},
        )
        edit_entity(
            db,
            root,
            "AI Adoption",
            metadata={"__source_add": [], "__remove_sources": ["linear:PRJ-123"]},
        )
        conn = db.get_sqlite_conn()
        row = conn.execute("SELECT metadata FROM entities WHERE name = 'AI Adoption'").fetchone()
        meta = json.loads(row["metadata"])
        assert len(meta["sources"]) == 1
        assert meta["sources"][0]["id"] == "PRJ-456"
