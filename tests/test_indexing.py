"""Tests for chunking, source adapters, and indexing."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Return the real project root."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def tmp_db():
    """Create a temporary database."""
    from kb.db import Database

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        yield db
        db.close()


class TestFrontmatter:
    def test_parse_frontmatter(self):
        from kb.chunker import parse_frontmatter

        text = (
            "---\ntitle: Test\ndate: 2026-01-27\ntype: notes\n"
            "granola_id: abc123\ntags:\n  - Soren\n---\n# Body"
        )
        fm = parse_frontmatter(text)
        assert fm["title"] == "Test"
        assert fm["date"] == "2026-01-27"
        assert fm["type"] == "notes"
        assert fm["granola_id"] == "abc123"
        assert fm["tags"] == ["Soren"]

    def test_parse_frontmatter_no_tags(self):
        from kb.chunker import parse_frontmatter

        text = "---\ntitle: Meeting\ndate: 2026-01-01\ntype: notes\n---\n# Content"
        fm = parse_frontmatter(text)
        assert fm["title"] == "Meeting"
        assert fm["tags"] == []

    def test_parse_frontmatter_missing(self):
        from kb.chunker import parse_frontmatter

        fm = parse_frontmatter("# Just a heading\nSome content.")
        assert fm == {}

    def test_parse_frontmatter_invalid_yaml(self):
        from kb.chunker import parse_frontmatter

        text = "---\n: [invalid\nyaml: {{\n---\nBody"
        fm = parse_frontmatter(text)
        assert fm == {}

    def test_parse_frontmatter_non_dict(self):
        from kb.chunker import parse_frontmatter

        text = "---\n- just\n- a\n- list\n---\nBody"
        fm = parse_frontmatter(text)
        assert fm == {}

    def test_parse_frontmatter_notion_page_id(self):
        from kb.chunker import parse_frontmatter

        text = (
            "---\ntitle: Notion Page\ndate: 2026-02-01\ntype: notes\n"
            "source: notion-api\nnotion_page_id: abc-def-123\n---\n# Body"
        )
        fm = parse_frontmatter(text)
        assert fm["notion_page_id"] == "abc-def-123"
        assert fm["calendar_uid"] is None

    def test_parse_frontmatter_calendar_uid(self):
        from kb.chunker import parse_frontmatter

        text = (
            "---\ntitle: Meeting\ndate: 2026-02-01\ntype: notes\n"
            "granola_id: abc123\ncalendar_uid: cal-uid-567890\n---\n# Body"
        )
        fm = parse_frontmatter(text)
        assert fm["calendar_uid"] == "cal-uid-567890"
        assert fm["notion_page_id"] is None


class TestNotesChunking:
    def test_splits_on_headers(self):
        from kb.chunker import chunk_notes

        body = (
            "# Main Title\n\n"
            "## Section One\n\nContent for section one.\n\n"
            "## Section Two\n\nContent for section two.\n\n"
            "### Subsection\n\nSubsection content.\n"
        )
        chunks = chunk_notes(body, "Test Meeting", "2026-01-27")
        assert len(chunks) >= 3

        # Check metadata prefix stored separately
        assert "[Meeting: Test Meeting | Date: 2026-01-27" in chunks[0].metadata_prefix
        assert not chunks[0].content.startswith("[Meeting:")

        # Check headings
        headings = [c.heading for c in chunks]
        assert "Section One" in headings
        assert "Section Two" in headings
        assert "Subsection" in headings

    def test_skips_empty_sections(self):
        from kb.chunker import chunk_notes

        body = "## My notes\n\nNo personal notes available.\n\n---\n\n## Real Content\n\nActual stuff here.\n"
        chunks = chunk_notes(body, "Test", "2026-01-27")
        # Only "Real Content" should produce a chunk
        assert len(chunks) >= 1
        assert any("Actual stuff here" in c.content for c in chunks)

    def test_long_section_split(self):
        from kb.chunker import chunk_notes

        # Create a section exceeding MAX_SECTION_TOKENS (2000 tokens = ~8000 chars)
        long_para1 = "This is a long paragraph. " * 200  # ~5200 chars
        long_para2 = "Another long paragraph. " * 200  # ~4800 chars
        body = f"## Long Section\n\n{long_para1}\n\n{long_para2}\n"
        chunks = chunk_notes(body, "Test", "2026-01-27")
        assert len(chunks) >= 2  # Should be split

    def test_entity_enriched_prefix(self):
        from kb.chunker import chunk_notes

        entities_info = [
            ("Soren Vance", "person", "DevEfficiency"),
            ("Pipeline Improvements", "project", ""),
        ]
        body = "## Performance Review\n\nCharles received feedback.\n"
        chunks = chunk_notes(body, "Wren / Soren", "2026-01-27", entities_info=entities_info)
        assert len(chunks) >= 1
        assert (
            "Entities: Soren Vance (person), Pipeline Improvements (project)"
            in chunks[0].metadata_prefix
        )

    def test_real_notes_file(self, project_root):
        from kb.chunker import _strip_frontmatter, chunk_notes, parse_frontmatter

        path = (
            project_root
            / "memory"
            / "meetings"
            / "2026"
            / "01"
            / "27"
            / "f744ed8c_User___Charles.granola.notes.md"
        )
        if not path.exists():
            pytest.skip("Real meeting files not available")

        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        body = _strip_frontmatter(text)
        chunks = chunk_notes(body, fm["title"], fm["date"])
        assert len(chunks) >= 3  # Multiple sections


class TestTranscriptChunking:
    def test_speaker_turn_parsing(self):
        from kb.chunker import chunk_transcript

        body = (
            "## Transcript\n\n"
            "**Me:** Hello there.\n\n"
            "**System:** Hi, how are you?\n\n"
            "**Me:** Good, thanks.\n\n"
            "**System:** Great to hear.\n"
        )
        chunks = chunk_transcript(body, "Me / Soren", "2026-01-27")
        assert len(chunks) >= 1
        # System should be replaced with Soren
        assert "Soren:" in chunks[0].content
        assert "System:" not in chunks[0].content

    def test_speaker_replacement_with_title(self):
        from kb.chunker import _replace_system_speaker

        turns = [("Me", "Hello"), ("System", "Hi there")]
        replaced = _replace_system_speaker(turns, "Me / Soren")
        assert replaced[1][0] == "Soren"

    def test_transcript_metadata_prefix(self):
        from kb.chunker import chunk_transcript

        body = "**Me:** Hello.\n\n**System:** Hi.\n"
        chunks = chunk_transcript(body, "Wren / Soren", "2026-01-27")
        assert len(chunks) >= 1
        assert "[Transcript: Wren / Soren | Date: 2026-01-27]" in chunks[0].metadata_prefix
        assert not chunks[0].content.startswith("[Transcript:")

    def test_sliding_window(self):
        from kb.chunker import chunk_transcript

        # Build a long transcript (many turns)
        turns = []
        for i in range(100):
            speaker = "Me" if i % 2 == 0 else "System"
            turns.append(
                f"**{speaker}:** This is turn number {i} with some extra text to make it longer and fill up space in the chunk window."
            )

        body = "\n\n".join(turns)
        chunks = chunk_transcript(body, "Wren / Soren", "2026-01-27")
        assert len(chunks) >= 2  # Should produce multiple windows

    def test_entity_enriched_prefix(self):
        from kb.chunker import chunk_transcript

        entities_info = [
            ("Linnea Holm", "person", "Product"),
        ]
        body = "**Me:** Hello.\n\n**System:** Hi there.\n"
        chunks = chunk_transcript(body, "Me / Soren", "2026-01-27", entities_info=entities_info)
        assert len(chunks) >= 1
        assert "Entities: Linnea Holm (person)" in chunks[0].metadata_prefix

    def test_no_headers_single_chunk_fallback(self):
        """Body with no headers should produce a single chunk."""
        from kb.chunker import chunk_notes

        body = "Just some text with no markdown headers at all.\nAnother line."
        chunks = chunk_notes(body, "Plain Meeting", "2026-01-27")
        assert len(chunks) == 1
        assert "Just some text" in chunks[0].content
        assert "[Meeting: Plain Meeting" in chunks[0].metadata_prefix

    def test_no_turns_transcript_fallback(self):
        """Transcript with no speaker turns falls back to plain text chunk."""
        from kb.chunker import chunk_transcript

        body = "Just some plain text with no **Speaker:** patterns."
        chunks = chunk_transcript(body, "Wren / Soren", "2026-01-27")
        assert len(chunks) == 1
        assert "[Transcript: Wren / Soren" in chunks[0].metadata_prefix
        assert "Just some plain text" in chunks[0].content

    def test_empty_transcript_returns_empty(self):
        """Empty transcript body should return no chunks."""
        from kb.chunker import chunk_transcript

        chunks = chunk_transcript("", "Wren / Soren", "2026-01-27")
        assert chunks == []

    def test_empty_turn_texts_returns_empty(self):
        """Speaker turns with no actual text should return empty."""
        from kb.chunker import chunk_transcript

        body = "**Me:** \n\n**System:** \n"
        chunks = chunk_transcript(body, "Wren / Soren", "2026-01-27")
        assert chunks == []

    def test_system_speaker_multiple_participants(self):
        """Multiple non-Me participants: System becomes 'Other'."""
        from kb.chunker import _replace_system_speaker

        turns = [("Me", "Hello"), ("System", "Hi")]
        replaced = _replace_system_speaker(turns, "Me / Soren / Soren")
        assert replaced[1][0] == "Other"

    def test_system_speaker_no_other_participants(self):
        """No identifiable other participants: System stays 'System'."""
        from kb.chunker import _replace_system_speaker

        turns = [("Me", "Hello"), ("System", "Hi")]
        replaced = _replace_system_speaker(turns, "Me")
        assert replaced[1][0] == "System"

    def test_multiline_speaker_turn(self):
        """Continuation lines should be joined into the current speaker's turn."""
        from kb.chunker import _parse_speaker_turns

        body = "**Me:** First line.\nSecond line of my turn.\n\n**System:** Reply.\n"
        turns = _parse_speaker_turns(body)
        assert len(turns) == 2
        assert "First line." in turns[0][1]
        assert "Second line of my turn." in turns[0][1]


@pytest.fixture
def memory_root(tmp_path):
    """Synthetic memory files for fast walker tests."""
    mem = tmp_path / "memory"
    # People
    people = mem / "people"
    people.mkdir(parents=True)
    (people / "alice.md").write_text("# Wren Smith\n\n**Role:** Engineer\n**Team:** Platform\n")
    (people / "bob.md").write_text("# Soren Jones\n\n**Role:** Manager\n**Team:** Product\n")
    # Projects
    projects = mem / "projects"
    projects.mkdir(parents=True)
    (projects / "alpha.md").write_text("# Project Alpha\n\n**Status:** Active\n**Lead:** Wren\n")
    # Context
    context = mem / "context"
    context.mkdir(parents=True)
    (context / "company.md").write_text("# Company Context\n\nWe are a tech company.\n")
    # Notes
    notes = mem / "notes"
    notes.mkdir(parents=True)
    (notes / "note1.md").write_text(
        "---\ntitle: Important Note\ndate: 2026-01-15\ntags:\n  - priority\n---\n\nSome content.\n"
    )
    # Glossary
    (mem / "glossary.md").write_text(
        "# Glossary\n\n"
        "| Term | Expansion | Notes |\n"
        "| --- | --- | --- |\n"
        "| API | Application Programming Interface | Common |\n"
        "| CLI | Command Line Interface | |\n"
        "| MFA | Multi-Factor Authentication | Security |\n"
    )
    # Catch-all: a file outside known subdirs
    decisions = mem / "decisions"
    decisions.mkdir(parents=True)
    (decisions / "dec-001.md").write_text("# Decision 001\n\nWe decided to use Python.\n")
    return tmp_path


class TestMemoryParsing:
    def test_person_single_chunk(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        people = [d for d in docs if d.doc_type == "memory_person"]

        assert len(people) == 2

        for person in people:
            assert len(person.chunks) == 1
            assert person.source_system == "memory"

    def test_project_single_chunk(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        projects = [d for d in docs if d.doc_type == "memory_project"]

        assert len(projects) == 1
        for proj in projects:
            assert len(proj.chunks) == 1

    def test_context_files(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        context = [d for d in docs if d.doc_type == "memory_context"]

        assert len(context) == 1

    def test_note_files(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        notes = [d for d in docs if d.doc_type == "memory_note"]

        assert len(notes) == 1
        assert notes[0].tags == ["priority"]

    def test_catchall_doc_type(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        catchall = [d for d in docs if d.doc_type == "memory_doc"]

        assert len(catchall) == 1
        assert "Decision 001" in catchall[0].chunks[0].content


class TestGlossaryParsing:
    def test_glossary_row_per_chunk(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        glossary = [d for d in docs if d.doc_type == "memory_glossary"]

        assert len(glossary) == 1
        doc = glossary[0]
        assert len(doc.chunks) == 3  # API, CLI, MFA

        # Check chunk format
        first = doc.chunks[0]
        assert ":" in first.content  # "TERM: expansion" format


@pytest.fixture
def meetings_root(tmp_path):
    """Synthetic meeting files for fast walker tests."""
    base = tmp_path / "memory" / "meetings" / "2026" / "01" / "15"
    base.mkdir(parents=True)
    for i, (name, tag) in enumerate(
        [
            ("aabb0001_User___Bob", "Soren"),
            ("aabb0002_User___Charles", "Soren"),
            ("aabb0003_Team_Standup", "Engineering"),
        ]
    ):
        (base / f"{name}.granola.notes.md").write_text(
            f"---\ntitle: Meeting {i + 1}\ndate: 2026-01-15\ntype: notes\n"
            f"granola_id: aabb000{i + 1}\ntags:\n  - {tag}\n---\n\n"
            f"## Discussion\n\nContent for meeting {i + 1}.\n"
        )
    for i, name in enumerate(
        [
            "aabb0001_User___Bob",
            "aabb0002_User___Charles",
        ]
    ):
        (base / f"{name}.granola.transcript.md").write_text(
            f"---\ntitle: Transcript {i + 1}\ndate: 2026-01-15\ntype: transcript\n"
            f"granola_id: aabb000{i + 1}\n---\n\n"
            f"**Me:** Hello.\n\n**System:** Hi there.\n"
        )
    # Add prep and debrief files
    (base / "aabb0001_User___Bob.prep.md").write_text(
        "---\ntitle: 'Prep: User / Soren'\ndate: 2026-01-15\ntype: prep\n"
        "source: cos-agent\ncalendar_uid: aabb0001\n---\n\n"
        "## Agenda\n\nDiscuss project updates.\n"
    )
    (base / "aabb0001_User___Bob.debrief.md").write_text(
        "---\ntitle: 'Debrief: User / Soren'\ndate: 2026-01-15\ntype: debrief\n"
        "source: cos-agent\ncalendar_uid: aabb0001\n---\n\n"
        "## Action Items\n\n- Follow up on project timeline.\n"
    )
    return tmp_path


class TestMeetingWalker:
    def test_walk_meetings_count(self, meetings_root):
        from kb.sources.meetings import walk_meetings

        docs = list(walk_meetings(meetings_root))
        notes = [d for d in docs if d.doc_type == "notes"]
        transcripts = [d for d in docs if d.doc_type == "transcript"]
        preps = [d for d in docs if d.doc_type == "prep"]
        debriefs = [d for d in docs if d.doc_type == "debrief"]

        assert len(notes) == 3
        assert len(transcripts) == 2
        assert len(preps) == 1
        assert len(debriefs) == 1

    def test_walk_meetings_has_chunks(self, meetings_root):
        from kb.sources.meetings import walk_meetings

        docs = list(walk_meetings(meetings_root))
        total_chunks = sum(len(d.chunks) for d in docs)
        assert total_chunks >= 7  # 5 original + at least 1 per prep/debrief

    def test_meetings_have_raw_body(self, meetings_root):
        from kb.sources.meetings import walk_meetings

        docs = list(walk_meetings(meetings_root))
        for doc in docs:
            assert doc.raw_body, f"Document {doc.path} missing raw_body"

    def test_prep_file_indexed(self, meetings_root):
        """Prep files (.prep.md) are picked up by the meetings walker."""
        from kb.sources.meetings import walk_meetings

        docs = list(walk_meetings(meetings_root))
        preps = [d for d in docs if d.doc_type == "prep"]
        assert len(preps) == 1
        assert "Prep:" in preps[0].title
        assert preps[0].path.endswith(".prep.md")

    def test_debrief_file_indexed(self, meetings_root):
        """Debrief files (.debrief.md) are picked up by the meetings walker."""
        from kb.sources.meetings import walk_meetings

        docs = list(walk_meetings(meetings_root))
        debriefs = [d for d in docs if d.doc_type == "debrief"]
        assert len(debriefs) == 1
        assert "Debrief:" in debriefs[0].title
        assert debriefs[0].path.endswith(".debrief.md")

    def test_prep_debrief_source_system(self, meetings_root):
        """Prep/debrief files with source: cos-agent get that as source_system."""
        from kb.sources.meetings import walk_meetings

        docs = list(walk_meetings(meetings_root))
        agent_docs = [d for d in docs if d.doc_type in ("prep", "debrief")]
        assert len(agent_docs) == 2
        for doc in agent_docs:
            assert doc.source_system == "cos-agent"


class TestMemoryWalker:
    def test_walk_memory_count(self, memory_root):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(memory_root))
        # 2 people + 1 project + 1 context + 1 note + 1 glossary + 1 catch-all = 7
        assert len(docs) == 7

    def test_walk_memory_empty_dir(self, tmp_path):
        from kb.sources.memory import walk_memory

        docs = list(walk_memory(tmp_path))
        assert len(docs) == 0


class TestDocumentLevelSummary:
    def test_notes_get_summary_chunk(self):
        from kb.chunker import ParsedDocument
        from kb.indexer import _make_summary_text

        doc = ParsedDocument(
            path="test.notes.md",
            title="Test Meeting",
            date="2026-01-27",
            doc_type="notes",
            source_system="granola",
            source_id="abc123",
            raw_body="## Section\n\nSome meeting content here.",
        )
        summary = _make_summary_text(doc)
        assert summary is not None
        assert "[Summary: Test Meeting | Date: 2026-01-27]" in summary
        assert "Some meeting content" in summary

    def test_transcript_long_gets_truncated_summary(self):
        from kb.chunker import ParsedDocument
        from kb.indexer import _make_summary_text

        # Create a very long transcript body
        long_body = "Speaker turn content. " * 10000  # ~220K chars

        doc = ParsedDocument(
            path="test.transcript.md",
            title="Long Meeting",
            date="2026-02-01",
            doc_type="transcript",
            source_system="granola",
            source_id="def456",
            raw_body=long_body,
        )
        summary = _make_summary_text(doc)
        assert summary is not None
        assert "[Summary: Long Meeting | Date: 2026-02-01]" in summary
        # Should be truncated (not the full body)
        assert len(summary) < len(long_body)

    def test_memory_files_no_summary(self):
        from kb.chunker import ParsedDocument
        from kb.indexer import _make_summary_text

        for doc_type in ("memory_person", "memory_project", "memory_context", "memory_glossary"):
            doc = ParsedDocument(
                path="test.md",
                title="Test",
                date=None,
                doc_type=doc_type,
                source_system="memory",
                source_id="test",
                raw_body="Some content",
            )
            assert _make_summary_text(doc) is None

    def test_summary_includes_entity_info(self):
        from kb.chunker import ParsedDocument
        from kb.indexer import _make_summary_text

        entities_info = [
            ("Soren Vance", "person", "DevEfficiency"),
            ("Pipeline Improvements", "project", ""),
        ]
        doc = ParsedDocument(
            path="test.notes.md",
            title="Wren / Soren",
            date="2026-01-27",
            doc_type="notes",
            source_system="granola",
            source_id="abc123",
            raw_body="## Review\n\nContent here.",
        )
        summary = _make_summary_text(doc, entities_info)
        assert summary is not None
        assert "Entities: Soren Vance (person), Pipeline Improvements (project)" in summary


@pytest.mark.slow
class TestIndexer:
    def test_incremental_skip(self, tmp_db):
        """Test that unchanged documents are skipped on re-index."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        # Create a minimal project root with one test file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            # Create minimal memory file
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test-person.md").write_text("# Test Person\n\n**Role:** Test\n")

            embedder = Embedder()
            r1 = index_all(tmp_db, embedder, tmp_root, full=True)
            assert r1.documents_indexed >= 1

            r2 = index_all(tmp_db, embedder, tmp_root, full=False)
            assert r2.documents_skipped >= 1
            assert r2.documents_indexed == 0

    def test_full_reindex(self, tmp_db):
        """Test full re-index clears and recreates."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            embedder = Embedder()
            r1 = index_all(tmp_db, embedder, tmp_root, full=True)
            assert r1.documents_indexed >= 1

            r2 = index_all(tmp_db, embedder, tmp_root, full=True)
            assert r2.documents_indexed >= 1
            assert r2.documents_skipped == 0

    def test_entity_linking_during_index(self, tmp_db):
        """Test that entity mentions are created during indexing."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            # Create a memory person
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "jane-doe.md").write_text(
                "# Jane Doe\n\n**Also known as:** Jane\n**Role:** Engineer\n"
            )

            # Create a meeting that mentions Jane
            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "01"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-01-01\ntype: notes\n"
                "granola_id: aabbccdd\ntags:\n  - Jane\n---\n\n"
                "## Discussion\n\nJane Doe presented the quarterly results.\n"
            )

            embedder = Embedder()
            result = index_all(tmp_db, embedder, tmp_root, full=True)
            assert result.entities_linked >= 1

            # Verify mentions exist in DB
            conn = tmp_db.get_sqlite_conn()
            mentions = conn.execute("SELECT COUNT(*) as cnt FROM entity_mentions").fetchone()
            assert mentions["cnt"] >= 1

    def test_summary_chunks_created(self, tmp_db):
        """Test that document-level summary chunks are created for notes."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "01"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-01-01\ntype: notes\n"
                "granola_id: aabbccdd\n---\n\n"
                "## Section A\n\nContent A.\n\n"
                "## Section B\n\nContent B.\n"
            )

            embedder = Embedder()
            index_all(tmp_db, embedder, tmp_root, full=True)

            # Verify __document__ chunk exists
            conn = tmp_db.get_sqlite_conn()
            summary = conn.execute("SELECT * FROM chunks WHERE heading = '__document__'").fetchall()
            assert len(summary) >= 1
            assert "[Summary: Test Meeting" in summary[0]["content"]

    def test_chunk_count_includes_summary(self, tmp_db):
        """Test that documents.chunk_count includes the __document__ summary chunk."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "01"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-01-01\ntype: notes\n"
                "granola_id: aabbccdd\n---\n\n"
                "## Section A\n\nContent A.\n\n"
                "## Section B\n\nContent B.\n"
            )

            embedder = Embedder()
            index_all(tmp_db, embedder, tmp_root, full=True)

            conn = tmp_db.get_sqlite_conn()
            doc = conn.execute("SELECT chunk_count FROM documents").fetchone()
            actual = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()
            assert doc["chunk_count"] == actual["c"], (
                f"chunk_count ({doc['chunk_count']}) should match actual chunks ({actual['c']})"
            )

    def test_memory_files_no_summary_chunks(self, tmp_db):
        """Test that memory files do NOT get summary chunks."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test Person\n\n**Role:** Engineer\n")

            embedder = Embedder()
            index_all(tmp_db, embedder, tmp_root, full=True)

            conn = tmp_db.get_sqlite_conn()
            summaries = conn.execute(
                "SELECT * FROM chunks WHERE heading = '__document__'"
            ).fetchall()
            assert len(summaries) == 0


class TestIndexerWithMockEmbedder:
    """Tests for index_all() with a mocked embedder to cover embedding flush + LanceDB paths."""

    def test_mock_embedder_writes_lance(self, tmp_db):
        """index_all with mock embedder covers _flush_embeddings and LanceDB write."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.indexer import index_all

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(5, 1024).astype(np.float32)
        mock_embedder.release_gpu_memory = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "01"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-01-01\ntype: notes\n"
                "granola_id: aabbccdd\n---\n\n"
                "## Section A\n\nContent A.\n\n"
                "## Section B\n\nContent B.\n"
            )

            result = index_all(tmp_db, mock_embedder, tmp_root, full=True)

            assert result.documents_indexed >= 2
            assert result.embeddings_skipped is False
            mock_embedder.embed.assert_called()

            # LanceDB should have vectors
            table = tmp_db.get_lance_table()
            assert table is not None

    def test_mock_embedder_incremental_update(self, tmp_db):
        """Changed docs get re-indexed with mock embedder (covers _delete_document + re-embed)."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.indexer import index_all

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(5, 1024).astype(np.float32)
        mock_embedder.release_gpu_memory = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            r1 = index_all(tmp_db, mock_embedder, tmp_root, full=True)
            assert r1.documents_indexed >= 1

            # Modify the file to trigger re-index
            (people_dir / "test.md").write_text("# Test Updated\n\n**Role:** Senior\n")
            r2 = index_all(tmp_db, mock_embedder, tmp_root, full=False)
            assert r2.documents_indexed >= 1  # changed file re-indexed


class TestNoEmbedIndexing:
    """Tests for embedding-optional indexing (embedder=None)."""

    def test_no_embed_populates_sqlite(self, tmp_db):
        """Index with embedder=None populates documents, chunks, and FTS."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test-person.md").write_text("# Test Person\n\n**Role:** Engineer\n")

            result = index_all(tmp_db, None, tmp_root, full=True)
            assert result.documents_indexed >= 1

            conn = tmp_db.get_sqlite_conn()
            docs = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()
            assert docs["c"] >= 1
            chunks = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()
            assert chunks["c"] >= 1
            # FTS should also be populated
            fts = conn.execute(
                "SELECT COUNT(*) as c FROM chunks_fts WHERE chunks_fts MATCH 'Engineer'"
            ).fetchone()
            assert fts["c"] >= 1

    def test_no_embed_skips_lance(self, tmp_db):
        """Index with embedder=None creates no LanceDB table."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            index_all(tmp_db, None, tmp_root, full=True)

            # LanceDB table should not exist
            table = tmp_db.get_lance_table()
            assert table is None

    def test_no_embed_entity_linking(self, tmp_db):
        """Entity mentions still created without embeddings."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "jane-doe.md").write_text(
                "# Jane Doe\n\n**Also known as:** Jane\n**Role:** Engineer\n"
            )

            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "01"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-01-01\ntype: notes\n"
                "granola_id: aabbccdd\ntags:\n  - Jane\n---\n\n"
                "## Discussion\n\nJane Doe presented the quarterly results.\n"
            )

            result = index_all(tmp_db, None, tmp_root, full=True)
            assert result.entities_linked >= 1

            conn = tmp_db.get_sqlite_conn()
            mentions = conn.execute("SELECT COUNT(*) as cnt FROM entity_mentions").fetchone()
            assert mentions["cnt"] >= 1

    def test_no_embed_attendees(self, tmp_db):
        """Attendees still processed without embeddings."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "jane-doe.md").write_text(
                "# Jane Doe\n\n**Email:** jane@example.com\n**Role:** Engineer\n"
            )

            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "01"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-01-01\ntype: notes\n"
                "granola_id: aabbccdd\ntags:\n  - Jane\n"
                "attendees:\n"
                "  - name: Jane Doe\n"
                "    email: jane@example.com\n"
                "---\n\n"
                "## Discussion\n\nJane presented results.\n"
            )

            result = index_all(tmp_db, None, tmp_root, full=True)
            assert result.attendees_stored >= 1

            conn = tmp_db.get_sqlite_conn()
            attendees = conn.execute("SELECT COUNT(*) as c FROM document_attendees").fetchone()
            assert attendees["c"] >= 1

    def test_no_embed_incremental(self, tmp_db):
        """Unchanged docs skipped on second run without embeddings."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            r1 = index_all(tmp_db, None, tmp_root, full=True)
            assert r1.documents_indexed >= 1

            r2 = index_all(tmp_db, None, tmp_root, full=False)
            assert r2.documents_skipped >= 1
            assert r2.documents_indexed == 0

    def test_no_embed_full_preserves_vectors(self, tmp_db):
        """Full re-index with no embed clears SQLite but doesn't drop LanceDB."""
        import pyarrow as pa

        from kb.db import LANCE_SCHEMA
        from kb.indexer import index_all

        # Manually create a LanceDB table with dummy data
        lance_db = tmp_db.get_lance_db()
        dummy = [
            {
                "chunk_id": 999,
                "embedding": [0.0] * 1024,
                "doc_type": "test",
                "doc_date": "2026-01-01",
                "tags": "[]",
                "document_id": 999,
                "entity_ids": "[]",
            }
        ]
        lance_db.create_table("chunks", data=pa.Table.from_pylist(dummy, schema=LANCE_SCHEMA))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            # Full re-index with no embedder should NOT drop LanceDB
            index_all(tmp_db, None, tmp_root, full=True)

            # LanceDB table should still exist
            assert "chunks" in lance_db.list_tables().tables

    def test_no_embed_result_fields(self, tmp_db):
        """IndexResult reports embeddings_skipped=True when embedder is None."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            result = index_all(tmp_db, None, tmp_root, full=True)
            assert result.embeddings_skipped is True

    @pytest.mark.slow
    def test_no_embed_with_embedder_has_skipped_false(self, tmp_db):
        """IndexResult reports embeddings_skipped=False when embedder is provided."""
        from kb.embeddings import Embedder
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")

            embedder = Embedder()
            result = index_all(tmp_db, embedder, tmp_root, full=True)
            assert result.embeddings_skipped is False

    def test_indexer_updates_last_mentioned_at(self, tmp_db):
        """Indexing a document updates entity.last_mentioned_at to the document date."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            # Create entity via memory file
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "jane-doe.md").write_text(
                "# Jane Doe\n\n**Also known as:** Jane\n**Role:** Engineer\n"
            )

            # Create meeting that mentions Jane with a specific date
            meetings_dir = tmp_root / "memory" / "meetings" / "2026" / "02" / "15"
            meetings_dir.mkdir(parents=True)
            (meetings_dir / "aabbccdd_Test_Meeting.granola.notes.md").write_text(
                "---\ntitle: Test Meeting\ndate: 2026-02-15\ntype: notes\n"
                "granola_id: aabbccdd\ntags:\n  - Jane\n---\n\n"
                "## Discussion\n\nJane Doe presented quarterly results.\n"
            )

            index_all(tmp_db, None, tmp_root, full=True)

            conn = tmp_db.get_sqlite_conn()
            row = conn.execute(
                "SELECT last_mentioned_at FROM entities WHERE name = 'Jane Doe'"
            ).fetchone()
            assert row is not None
            assert row["last_mentioned_at"] == "2026-02-15"


class TestResetIndex:
    """Tests for _reset_index with skip_vectors option."""

    def test_reset_with_skip_vectors(self, tmp_db):
        """_reset_index with skip_vectors=True clears SQLite but leaves LanceDB."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.indexer import _clear_all, index_all

        # First, index something with a mock embedder to populate LanceDB
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(5, 1024).astype(np.float32)
        mock_embedder.release_gpu_memory = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")
            index_all(tmp_db, mock_embedder, tmp_root, full=True)

        # Now reset with skip_vectors — SQLite should be empty, LanceDB intact
        _clear_all(tmp_db, skip_vectors=True)
        conn = tmp_db.get_sqlite_conn()
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert docs == 0

    def test_reset_without_skip_vectors(self, tmp_db):
        """_reset_index without skip_vectors clears everything including LanceDB."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.indexer import _clear_all, index_all

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(5, 1024).astype(np.float32)
        mock_embedder.release_gpu_memory = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            people_dir = tmp_root / "memory" / "people"
            people_dir.mkdir(parents=True)
            (people_dir / "test.md").write_text("# Test\n\n**Role:** Test\n")
            index_all(tmp_db, mock_embedder, tmp_root, full=True)

        _clear_all(tmp_db, skip_vectors=False)
        conn = tmp_db.get_sqlite_conn()
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert docs == 0
        # LanceDB table should be gone
        lance_db = tmp_db.get_lance_db()
        assert "chunks" not in lance_db.list_tables().tables


class TestFlushLanceBatch:
    def test_empty_batch_is_noop(self, tmp_db):
        """Flushing an empty batch should do nothing."""
        from kb.indexer import _flush_lance_batch

        # Should not raise
        _flush_lance_batch(tmp_db, [])
        # No table should have been created
        assert tmp_db.get_lance_table() is None


class TestIndexingDeleteAndReindex:
    def test_changed_meeting_triggers_delete_and_reindex(self, tmp_db):
        """When a meeting file changes, the old version is deleted and re-indexed."""
        from kb.indexer import index_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            meet_dir = tmp_root / "memory" / "meetings" / "2026" / "01" / "15"
            meet_dir.mkdir(parents=True)
            (meet_dir / "aabb0001_User___Bob.granola.notes.md").write_text(
                "---\ntitle: Meeting 1\ndate: 2026-01-15\ntype: notes\n"
                "granola_id: aabb0001\ntags:\n  - Soren\n---\n\n## Discussion\n\nOriginal.\n"
            )

            r1 = index_all(tmp_db, None, tmp_root, full=False)
            assert r1.documents_indexed == 1

            # Change the file — triggers _delete_document + re-index
            (meet_dir / "aabb0001_User___Bob.granola.notes.md").write_text(
                "---\ntitle: Meeting 1\ndate: 2026-01-15\ntype: notes\n"
                "granola_id: aabb0001\ntags:\n  - Soren\n---\n\n## Discussion\n\nUpdated content.\n"
            )

            r2 = index_all(tmp_db, None, tmp_root, full=False)
            assert r2.documents_indexed == 1  # re-indexed
            assert r2.documents_skipped == 0  # hash changed, not skipped


class TestMakeSummaryChunkEdgeCases:
    def test_empty_body_returns_none(self):
        """A document with empty body should not produce a summary."""
        from unittest.mock import MagicMock

        from kb.indexer import _make_summary_text

        doc = MagicMock()
        doc.doc_type = "notes"
        doc.raw_body = ""
        assert _make_summary_text(doc) is None

    def test_none_body_returns_none(self):
        """A document with None body should not produce a summary."""
        from unittest.mock import MagicMock

        from kb.indexer import _make_summary_text

        doc = MagicMock()
        doc.doc_type = "notes"
        doc.raw_body = None
        assert _make_summary_text(doc) is None


class TestImportChain:
    """Tests for lazy import chain — no unnecessary dependencies."""

    def test_indexer_importable_without_sentence_transformers(self):
        """kb.indexer can be imported without sentence_transformers."""
        import importlib
        import sys

        # Temporarily remove sentence_transformers from sys.modules to test
        # the TYPE_CHECKING guard. We can't truly uninstall it, but we verify
        # the import path doesn't eagerly load it.
        saved = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            # Force reimport of indexer
            if "kb.indexer" in sys.modules:
                del sys.modules["kb.indexer"]
            if "kb.embeddings" in sys.modules:
                del sys.modules["kb.embeddings"]
            importlib.import_module("kb.indexer")
        finally:
            if saved is not None:
                sys.modules["sentence_transformers"] = saved
            elif "sentence_transformers" in sys.modules:
                del sys.modules["sentence_transformers"]
            # Restore original modules
            if "kb.indexer" in sys.modules:
                del sys.modules["kb.indexer"]
            if "kb.embeddings" in sys.modules:
                del sys.modules["kb.embeddings"]


# ---------------------------------------------------------------------------
# Wikilink stripping in memory chunks
# ---------------------------------------------------------------------------


class TestWikilinksInChunks:
    def test_memory_chunks_strip_wikilinks(self):
        """Memory file chunks should not contain [[wikilinks]]."""
        from kb.sources.memory import _parse_memory_file

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Test\n\nReports to [[Idris Kalmar]] on [[Core]] team.\n")
            f.flush()
            doc = _parse_memory_file(Path(f.name), "memory/people/test.md", "memory_person")

        assert doc.chunks
        for chunk in doc.chunks:
            assert "[[" not in chunk.content
            assert "Idris Kalmar" in chunk.content
