"""Tests for entity writeback (DB -> file sync)."""

import json
import tempfile
from pathlib import Path

import pytest

from kb.db import Database
from kb.entities import Entity, _parse_person_file, _parse_project_file, load_entity_by_id
from kb.writeback import (
    _derive_aliases,
    _rebuild_person_header,
    _rebuild_project_header,
    _split_header_and_body,
    write_entity_to_file,
)


@pytest.fixture
def tmp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        yield db
        db.close()


def _make_entity(**kwargs) -> Entity:
    """Helper to build an Entity with defaults."""
    defaults = {
        "id": 1,
        "name": "Jane Doe",
        "entity_type": "person",
        "aliases": ["JD", "Jane D.", "Jane", "jane-doe"],
        "metadata": {"role": "Engineer", "team": "Core"},
        "source_path": "memory/people/jane-doe.md",
        "pinned": False,
    }
    defaults.update(kwargs)
    return Entity(**defaults)


class TestSplitHeaderAndBody:
    def test_splits_at_first_h2(self):
        content = "# Title\n\n**Role:** Eng\n\n## Notes\nSome notes\n"
        header, body = _split_header_and_body(content)
        assert header == "# Title\n\n**Role:** Eng\n\n"
        assert body == "## Notes\nSome notes\n"

    def test_no_h2_returns_all_as_header(self):
        content = "# Title\n\n**Role:** Eng\n"
        header, body = _split_header_and_body(content)
        assert header == content
        assert body == ""

    def test_multiple_h2_splits_at_first(self):
        content = "# Title\n\n## First\nA\n## Second\nB\n"
        header, body = _split_header_and_body(content)
        assert header == "# Title\n\n"
        assert body == "## First\nA\n## Second\nB\n"


class TestDeriveAliases:
    def test_excludes_first_name_and_file_stem(self):
        entity = _make_entity(
            aliases=["JD", "Jane D.", "Jane", "jane-doe"],
        )
        result = _derive_aliases(entity)
        assert "JD" in result
        assert "Jane D." in result
        assert "Jane" not in result  # first name excluded
        assert "jane-doe" not in result  # file stem excluded

    def test_keeps_all_when_no_derivable(self):
        entity = _make_entity(
            name="A. B. Smith",
            aliases=["ABS", "A.B."],
        )
        result = _derive_aliases(entity)
        assert result == ["ABS", "A.B."]


class TestRebuildPersonHeader:
    def test_basic_person(self):
        entity = _make_entity()
        header = _rebuild_person_header(entity)
        assert header.startswith("---\n")
        assert "---\n# Jane Doe\n" in header
        assert "aliases:" in header
        assert "- JD" in header
        assert "- Jane D." in header
        assert "role: Engineer" in header
        assert "team: '[[Core]]'" in header
        assert "pinned" not in header

    def test_with_email_and_company(self):
        entity = _make_entity(
            metadata={"email": "jane@example.com", "role": "Eng", "company": "Lattice"},
        )
        header = _rebuild_person_header(entity)
        assert "email: jane@example.com" in header
        assert "company: '[[Lattice]]'" in header

    def test_pinned_true(self):
        entity = _make_entity(pinned=True)
        header = _rebuild_person_header(entity)
        assert "pinned: true" in header

    def test_omits_empty_fields(self):
        entity = _make_entity(metadata={}, aliases=[])
        header = _rebuild_person_header(entity)
        assert "aliases:" not in header
        assert "role:" not in header


class TestRebuildProjectHeader:
    def test_basic_project(self):
        entity = Entity(
            id=2,
            name="My Project",
            entity_type="project",
            aliases=["MP", "my-project"],
            metadata={"status": "Active", "lead": "Jane"},
            source_path="memory/projects/my-project.md",
            pinned=False,
        )
        header = _rebuild_project_header(entity)
        assert header.startswith("---\n")
        assert "---\n# My Project\n" in header
        assert "- MP" in header
        assert "status: Active" in header
        assert "lead: '[[Jane]]'" in header


class TestLoadEntityById:
    def test_load_existing(self, tmp_db):
        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "Test Person",
                "person",
                json.dumps(["TP"]),
                json.dumps({"role": "Dev"}),
                "memory/people/test.md",
                0,
            ),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        entity = load_entity_by_id(tmp_db, eid)
        assert entity is not None
        assert entity.name == "Test Person"
        assert entity.aliases == ["TP"]
        assert entity.metadata["role"] == "Dev"

    def test_load_missing(self, tmp_db):
        assert load_entity_by_id(tmp_db, 99999) is None


class TestWriteEntityToFile:
    def test_roundtrip_person(self, tmp_db, tmp_path):
        """Parse a person file, write to DB, write back, parse again — should be identical."""
        person_file = tmp_path / "memory" / "people" / "jane-doe.md"
        person_file.parent.mkdir(parents=True)
        person_file.write_text(
            "# Jane Doe\n\n"
            "**Also known as:** JD, Jane D.\n"
            "**Email:** jane@example.com\n"
            "**Role:** Engineering Lead\n"
            "**Team:** Core\n"
            "**Reports to:** CTO\n\n"
            "## Notes\nSome important notes.\n"
        )

        # Parse and insert into DB
        parsed = _parse_person_file(person_file)
        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            (
                parsed.name,
                parsed.entity_type,
                json.dumps(parsed.aliases),
                json.dumps(parsed.metadata),
                parsed.source_path,
                0,
            ),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Write back
        result = write_entity_to_file(tmp_db, eid, tmp_path)
        assert result is True

        # Parse again
        reparsed = _parse_person_file(person_file)
        assert reparsed.name == parsed.name
        assert reparsed.metadata["role"] == parsed.metadata["role"]
        assert reparsed.metadata["email"] == "jane@example.com"

        # Body preserved
        content = person_file.read_text()
        assert "## Notes\nSome important notes.\n" in content

    def test_roundtrip_project(self, tmp_db, tmp_path):
        """Parse a project file, write to DB, write back, parse again."""
        project_file = tmp_path / "memory" / "projects" / "my-proj.md"
        project_file.parent.mkdir(parents=True)
        project_file.write_text(
            "# My Project\n\n"
            "**Codename/Also called:** MP\n"
            "**Status:** Active\n"
            "**Lead:** Jane\n\n"
            "## Details\nProject details here.\n"
        )

        parsed = _parse_project_file(project_file)
        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            (
                parsed.name,
                parsed.entity_type,
                json.dumps(parsed.aliases),
                json.dumps(parsed.metadata),
                parsed.source_path,
                0,
            ),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        result = write_entity_to_file(tmp_db, eid, tmp_path)
        assert result is True

        content = project_file.read_text()
        assert "## Details\nProject details here.\n" in content

    def test_preserves_freeform_content(self, tmp_db, tmp_path):
        """Write-back should preserve ## sections exactly."""
        person_file = tmp_path / "memory" / "people" / "test.md"
        person_file.parent.mkdir(parents=True)
        person_file.write_text(
            "# Test Person\n\n"
            "**Role:** Dev\n\n"
            "## Communication\n"
            "- Weekly syncs\n"
            "- Slack active\n\n"
            "## Notes\n"
            "Important stuff here.\n"
        )

        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "Test Person",
                "person",
                json.dumps([]),
                json.dumps({"role": "Dev", "email": "test@example.com"}),
                "memory/people/test.md",
                0,
            ),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        write_entity_to_file(tmp_db, eid, tmp_path)

        content = person_file.read_text()
        # New email field added (now in YAML frontmatter)
        assert "email: test@example.com" in content
        # Freeform preserved
        assert "## Communication\n- Weekly syncs\n- Slack active\n" in content
        assert "## Notes\nImportant stuff here.\n" in content

    def test_skips_nonexistent_file(self, tmp_db, tmp_path):
        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            ("Ghost", "person", "[]", "{}", "memory/people/ghost.md", 0),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        assert write_entity_to_file(tmp_db, eid, tmp_path) is False

    def test_skips_non_memory_source(self, tmp_db, tmp_path):
        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            ("CLAUDE Entity", "person", "[]", "{}", "CLAUDE.md", 0),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        assert write_entity_to_file(tmp_db, eid, tmp_path) is False

    def test_no_write_when_unchanged(self, tmp_db, tmp_path):
        """If header matches DB state, file should not be rewritten."""
        person_file = tmp_path / "memory" / "people" / "same.md"
        person_file.parent.mkdir(parents=True)
        # Must match the canonical YAML format produced by _rebuild_person_header
        person_file.write_text("---\nrole: Dev\n---\n# Same Person\n\n")

        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "Same Person",
                "person",
                json.dumps([]),
                json.dumps({"role": "Dev"}),
                "memory/people/same.md",
                0,
            ),
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        assert write_entity_to_file(tmp_db, eid, tmp_path) is False


class TestYamlFrontmatterWriteback:
    """Writeback should emit YAML frontmatter."""

    def test_person_yaml_header(self):
        entity = _make_entity(
            metadata={
                "email": "jane@x.com",
                "role": "Lead",
                "team": "Core",
                "reports_to": "CTO",
            },
        )
        header = _rebuild_person_header(entity)
        assert header.startswith("---\n")
        assert "---\n# Jane Doe" in header
        assert "aliases:" in header
        assert "JD" in header
        assert "jane-doe" not in header  # file stem excluded by _derive_aliases
        # "Jane" (first name only) excluded from aliases by _derive_aliases
        # but "Jane D." is kept — check aliases list doesn't have bare "Jane"
        assert "- Jane\n" not in header
        assert "email: jane@x.com" in header
        assert "role: Lead" in header
        assert "team: '[[Core]]'" in header
        assert "reports_to: '[[CTO]]'" in header
        assert "pinned" not in header  # not pinned

    def test_person_yaml_header_pinned(self):
        entity = _make_entity(
            name="Boss",
            aliases=["boss"],
            metadata={"role": "CEO"},
            source_path="memory/people/boss.md",
            pinned=True,
        )
        header = _rebuild_person_header(entity)
        assert "pinned: true" in header

    def test_project_yaml_header(self):
        entity = Entity(
            id=1,
            name="My Project",
            entity_type="project",
            aliases=["MP", "my-project"],
            metadata={"status": "Active", "lead": "Wren Smith"},
            source_path="memory/projects/my-project.md",
            pinned=False,
        )
        header = _rebuild_project_header(entity)
        assert header.startswith("---\n")
        assert "---\n# My Project" in header
        assert "MP" in header
        assert "my-project" not in header  # file stem excluded
        assert "status: Active" in header
        assert "[[Wren Smith]]" in header

    def test_split_header_body_yaml(self):
        content = "---\nrole: Dev\n---\n# Name\n\n## Notes\n\nBody here.\n"
        header, body = _split_header_and_body(content)
        assert "---" in header
        assert body.startswith("## Notes")

    def test_split_header_body_legacy(self):
        content = "# Name\n\n**Role:** Dev\n\n## Notes\n\nBody here.\n"
        header, body = _split_header_and_body(content)
        assert "# Name" in header
        assert body.startswith("## Notes")

    def test_person_yaml_custom_fields(self):
        """Custom metadata fields not in the registry should still appear."""
        entity = _make_entity(
            metadata={"role": "Dev", "timezone": "Europe/Paris"},
            aliases=[],
        )
        header = _rebuild_person_header(entity)
        assert "timezone: Europe/Paris" in header

    def test_entity_ref_already_wikilinked(self):
        """If a ref field already has [[wikilinks]], don't double-wrap."""
        entity = _make_entity(
            metadata={"team": "[[Core]]"},
            aliases=[],
        )
        header = _rebuild_person_header(entity)
        assert "[[Core]]" in header
        assert "[[[[" not in header  # no double-wrapping


class TestParsePersonEmailCompany:
    """Test that _parse_person_file now reads Email and Company fields."""

    def test_parses_email(self, tmp_path):
        f = tmp_path / "person.md"
        f.write_text("# Jane Doe\n\n**Email:** jane@example.com\n**Role:** Dev\n")
        result = _parse_person_file(f)
        assert result.metadata["email"] == "jane@example.com"

    def test_parses_company(self, tmp_path):
        f = tmp_path / "person.md"
        f.write_text("# Jane Doe\n\n**Company:** Lattice Co\n")
        result = _parse_person_file(f)
        assert result.metadata["company"] == "Lattice Co"

    def test_parses_both(self, tmp_path):
        f = tmp_path / "person.md"
        f.write_text("# Jane Doe\n\n**Email:** jane@acme.com\n**Company:** Lattice\n**Role:** CTO\n")
        result = _parse_person_file(f)
        assert result.metadata["email"] == "jane@acme.com"
        assert result.metadata["company"] == "Lattice"
        assert result.metadata["role"] == "CTO"


class TestDocumentAttendeesMigration:
    """Test that the document_attendees table is created."""

    def test_table_exists(self, tmp_db):
        conn = tmp_db.get_sqlite_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_attendees'"
        ).fetchone()
        assert row is not None

    def test_insert_and_query(self, tmp_db):
        conn = tmp_db.get_sqlite_conn()
        # Need a document to reference
        conn.execute(
            "INSERT INTO documents (path, title, content_hash, chunk_count) VALUES (?, ?, ?, ?)",
            ("test.md", "Test", "abc123", 0),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO document_attendees (document_id, name, email, matched_by) VALUES (?, ?, ?, ?)",
            (doc_id, "Jane Doe", "jane@example.com", "name"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT name, email FROM document_attendees WHERE document_id = ?", (doc_id,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "Jane Doe"
        assert rows[0]["email"] == "jane@example.com"

    def test_unique_constraint(self, tmp_db):
        conn = tmp_db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO documents (path, title, content_hash, chunk_count) VALUES (?, ?, ?, ?)",
            ("test.md", "Test", "abc123", 0),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO document_attendees (document_id, name, email) VALUES (?, ?, ?)",
            (doc_id, "Jane", "jane@example.com"),
        )
        # Same doc + email should be ignored with INSERT OR IGNORE
        conn.execute(
            "INSERT OR IGNORE INTO document_attendees (document_id, name, email) VALUES (?, ?, ?)",
            (doc_id, "Jane Doe", "jane@example.com"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT * FROM document_attendees WHERE document_id = ?", (doc_id,)
        ).fetchall()
        assert len(rows) == 1
