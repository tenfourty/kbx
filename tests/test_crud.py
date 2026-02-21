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
