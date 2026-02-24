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
        assert "**Role:** Engineer" in content
        assert "**Email:** jane@example.com" in content

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
        assert "**Also known as:** JD, Jane D." in content

    def test_create_duplicate_raises(self, crud_env):
        from kb.crud import EntityExistsError, create_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe")
        with pytest.raises(EntityExistsError):
            create_entity(db, root, "person", "Jane Doe")


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
        assert "**Status:** Active" in content
        assert "**Lead:** Wren Kasper" in content


class TestEntityEdit:
    def test_edit_updates_metadata(self, crud_env):
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"role": "Staff Engineer"})
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "**Role:** Staff Engineer" in content

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
        assert "**Role:** Staff Engineer" in content

    def test_edit_not_found_raises(self, crud_env):
        from kb.crud import EntityNotFoundError, edit_entity

        db, root = crud_env
        with pytest.raises(EntityNotFoundError):
            edit_entity(db, root, "Nobody", metadata={"role": "CEO"})


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
        assert "**Preferred Lang:** French" in content

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
        assert "**Preferred Lang:** French" in content
        assert "**Timezone:** CET" in content

    def test_edit_remove_custom_field(self, crud_env):
        """Setting a custom field to empty string removes it."""
        from kb.crud import create_entity, edit_entity

        db, root = crud_env
        create_entity(db, root, "person", "Jane Doe", metadata={"role": "Engineer"})
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": "French"})

        # Verify it's there
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "**Preferred Lang:** French" in content

        # Remove it
        edit_entity(db, root, "Jane Doe", metadata={"preferred_lang": ""})
        content = (root / "memory" / "people" / "jane-doe.md").read_text()
        assert "**Preferred Lang:**" not in content

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
        assert "**Priority:** High" in content
        assert "**Status:** Active" in content

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
        # File should have Title Case
        assert "**Preferred Lang:** French" in content

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
        assert "**Role:** Engineer" in content
        assert "**Preferred Lang:** French" in content


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
