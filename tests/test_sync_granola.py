"""Tests for Granola API sync — client, transform, write, CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def supabase_json(tmp_dir):
    """Write a mock supabase.json and return its path."""
    path = tmp_dir / "supabase.json"
    data = {
        "workos_tokens": {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
        }
    }
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def client(supabase_json):
    """Create a GranolaClient with mock tokens."""
    from kb.sync.granola import GranolaClient

    return GranolaClient(token_path=supabase_json)


# ---------------------------------------------------------------------------
# Phase 1: API Client & Auth
# ---------------------------------------------------------------------------


class TestTokenReading:
    def test_read_tokens_from_supabase_json(self, supabase_json):
        """Reads access/refresh tokens from a mock supabase.json."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)
        tokens = client._read_tokens()
        assert tokens["access_token"] == "test-access-token"
        assert tokens["refresh_token"] == "test-refresh-token"

    def test_read_tokens_missing_file(self, tmp_dir):
        """Clear error when Granola not installed (file missing)."""
        from kb.sync.granola import GranolaClient

        missing = tmp_dir / "nonexistent" / "supabase.json"
        with pytest.raises(FileNotFoundError, match=r"supabase\.json"):
            GranolaClient(token_path=missing)._read_tokens()

    def test_read_tokens_bad_json(self, tmp_dir):
        """Handles corrupt supabase.json gracefully."""
        from kb.sync.granola import GranolaClient

        bad = tmp_dir / "supabase.json"
        bad.write_text("not json{{{")
        with pytest.raises(ValueError, match=r"(?i)parse|json|corrupt"):
            GranolaClient(token_path=bad)._read_tokens()


class TestTokenRefresh:
    def test_refresh_token_rotation(self, supabase_json):
        """Sends refresh token, gets new access+refresh, writes back atomically."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
        }

        with patch("httpx.post", return_value=mock_response) as mock_post:
            client._refresh_tokens()

        # Verify the refresh request was made with old refresh token
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "test-refresh-token" in str(call_kwargs)

        # Verify tokens written back to file
        updated = json.loads(supabase_json.read_text())
        assert updated["workos_tokens"]["access_token"] == "new-access"
        assert updated["workos_tokens"]["refresh_token"] == "new-refresh"

    def test_refresh_token_single_use(self, supabase_json):
        """After rotation, old refresh token should not be in the file."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
        }

        with patch("httpx.post", return_value=mock_response):
            client._refresh_tokens()

        updated = json.loads(supabase_json.read_text())
        assert updated["workos_tokens"]["refresh_token"] == "rotated-refresh"
        assert "test-refresh-token" not in json.dumps(updated)


class TestAPIRequests:
    def test_list_documents(self, client):
        """Mock HTTP: returns paginated document list."""
        mock_docs = [
            {
                "id": "doc-1",
                "title": "Weekly 1:1",
                "created_at": "2026-02-10T15:00:00Z",
                "updated_at": "2026-02-10T16:00:00Z",
            },
            {
                "id": "doc-2",
                "title": "Team Standup",
                "created_at": "2026-02-11T09:00:00Z",
                "updated_at": "2026-02-11T10:00:00Z",
            },
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"docs": mock_docs}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", return_value=mock_response):
            docs = client.list_documents()

        assert len(docs) == 2
        assert docs[0]["id"] == "doc-1"
        assert docs[1]["title"] == "Team Standup"

    def test_list_documents_since(self, client):
        """Filters documents by date (client-side since API doesn't support it)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "docs": [
                {
                    "id": "doc-1",
                    "title": "Old Meeting",
                    "created_at": "2026-02-01T09:00:00Z",
                    "updated_at": "2026-02-01T10:00:00Z",
                },
                {
                    "id": "doc-2",
                    "title": "Team Standup",
                    "created_at": "2026-02-11T09:00:00Z",
                    "updated_at": "2026-02-11T10:00:00Z",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", return_value=mock_response):
            docs = client.list_documents(since="2026-02-11")

        # Only doc-2 should survive the client-side filter
        assert len(docs) == 1
        assert docs[0]["id"] == "doc-2"

    def test_get_transcript(self, client):
        """Mock HTTP: returns transcript segments with timestamps."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transcript": [
                {
                    "speaker": "Wren Kasper",
                    "text": "Let's discuss the roadmap.",
                    "start": 0.0,
                    "end": 3.5,
                },
                {
                    "speaker": "Soren Vance",
                    "text": "Sure, I've prepared some notes.",
                    "start": 3.5,
                    "end": 7.0,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", return_value=mock_response):
            transcript = client.get_transcript("doc-1")

        assert len(transcript) == 2
        assert transcript[0]["speaker"] == "Wren Kasper"
        assert transcript[1]["text"] == "Sure, I've prepared some notes."

    def test_get_metadata(self, client):
        """Mock HTTP: returns attendees, calendar event."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "people": [
                {"name": "Wren Kasper", "email": "wren@example.com"},
                {"name": "Soren Vance", "email": "soren@example.com"},
            ],
            "calendar_event": {
                "title": "Weekly 1:1",
                "organizer_email": "wren@example.com",
                "start_time": "2026-02-10T15:00:00Z",
                "end_time": "2026-02-10T15:30:00Z",
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", return_value=mock_response):
            metadata = client.get_metadata("doc-1")

        assert len(metadata["people"]) == 2
        assert metadata["people"][0]["email"] == "wren@example.com"
        assert metadata["calendar_event"]["title"] == "Weekly 1:1"


# ---------------------------------------------------------------------------
# Phase 2: Transform & Write
# ---------------------------------------------------------------------------


class TestProseMirrorToMarkdown:
    def test_headings(self):
        """Converts heading nodes to markdown."""
        from kb.sync.granola import prosemirror_to_markdown

        nodes = [
            {
                "type": "heading",
                "attrs": {"level": 1},
                "content": [{"type": "text", "text": "Meeting Notes"}],
            },
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Some content here."}],
            },
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Action Items"}],
            },
        ]
        md = prosemirror_to_markdown(nodes)
        assert "# Meeting Notes" in md
        assert "Some content here." in md
        assert "## Action Items" in md

    def test_lists(self):
        """Converts bullet/ordered lists."""
        from kb.sync.granola import prosemirror_to_markdown

        nodes = [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Item one"}],
                            },
                        ],
                    },
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Item two"}],
                            },
                        ],
                    },
                ],
            },
            {
                "type": "orderedList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "First"}]},
                        ],
                    },
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "Second"}]},
                        ],
                    },
                ],
            },
        ]
        md = prosemirror_to_markdown(nodes)
        assert "- Item one" in md
        assert "- Item two" in md
        assert "1. First" in md
        assert "2. Second" in md

    def test_nested(self):
        """Handles nested structures (list inside list)."""
        from kb.sync.granola import prosemirror_to_markdown

        nodes = [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "Parent"}]},
                            {
                                "type": "bulletList",
                                "content": [
                                    {
                                        "type": "listItem",
                                        "content": [
                                            {
                                                "type": "paragraph",
                                                "content": [{"type": "text", "text": "Child"}],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ]
        md = prosemirror_to_markdown(nodes)
        assert "- Parent" in md
        assert "  - Child" in md

    def test_text_marks(self):
        """Handles bold, italic, code marks on text."""
        from kb.sync.granola import prosemirror_to_markdown

        nodes = [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "This is "},
                    {"type": "text", "text": "bold", "marks": [{"type": "bold"}]},
                    {"type": "text", "text": " and "},
                    {"type": "text", "text": "italic", "marks": [{"type": "italic"}]},
                    {"type": "text", "text": " and "},
                    {"type": "text", "text": "code", "marks": [{"type": "code"}]},
                ],
            },
        ]
        md = prosemirror_to_markdown(nodes)
        assert "**bold**" in md
        assert "*italic*" in md
        assert "`code`" in md


class TestTranscriptToMarkdown:
    def test_segments_to_markdown(self):
        """Speaker + text, merges consecutive same-speaker segments."""
        from kb.sync.granola import transcript_to_markdown

        segments = [
            {"speaker": "Wren Kasper", "text": "Hello everyone.", "start": 0.0, "end": 2.0},
            {"speaker": "Wren Kasper", "text": "Let's begin.", "start": 2.0, "end": 4.0},
            {"speaker": "Soren Vance", "text": "Sounds good.", "start": 4.0, "end": 6.0},
        ]
        md = transcript_to_markdown(segments)
        # Consecutive same-speaker should be merged
        assert md.count("**Wren Kasper**") == 1
        assert "Hello everyone." in md
        assert "Let's begin." in md
        assert "**Soren Vance**" in md


class TestBuildFrontmatter:
    def test_build_frontmatter(self):
        """Enriched YAML with attendees, calendar, summary."""
        from kb.sync.granola import build_frontmatter

        doc = {
            "id": "abc12345",
            "title": "Wren / Soren",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        metadata = {
            "people": [
                {"name": "Wren Kasper", "email": "wren@example.com"},
                {"name": "Soren Vance", "email": "soren@example.com"},
            ],
            "calendar_event": {
                "title": "Weekly 1:1",
                "organizer_email": "wren@example.com",
                "start_time": "2026-01-27T15:00:00Z",
                "end_time": "2026-01-27T15:30:00Z",
            },
        }
        summary = "Discussed performance review and CI pipeline."

        fm = build_frontmatter(doc, metadata, summary)
        assert fm["title"] == "Wren / Soren"
        assert fm["date"] == "2026-01-27"
        assert fm["granola_id"] == "abc12345"
        assert fm["granola_updated_at"] == "2026-01-27T16:30:00Z"
        assert len(fm["attendees"]) == 2
        assert fm["attendees"][0]["email"] == "wren@example.com"
        assert fm["calendar_event"]["title"] == "Weekly 1:1"
        assert fm["granola_summary"] == "Discussed performance review and CI pipeline."
        # Tags derived from attendee names
        assert "Soren" in fm["tags"] or "Soren Vance" in fm["tags"]

    def test_build_frontmatter_minimal(self):
        """Graceful when optional fields are missing."""
        from kb.sync.granola import build_frontmatter

        doc = {
            "id": "def67890",
            "title": "Quick Chat",
            "created_at": "2026-02-01T10:00:00Z",
            "updated_at": "2026-02-01T10:30:00Z",
        }
        metadata = {"people": [], "calendar_event": None}

        fm = build_frontmatter(doc, metadata, summary=None)
        assert fm["title"] == "Quick Chat"
        assert fm["date"] == "2026-02-01"
        assert fm["granola_id"] == "def67890"
        assert fm["attendees"] == []
        assert "calendar_event" not in fm or fm["calendar_event"] is None
        assert "granola_summary" not in fm or fm["granola_summary"] is None

    def test_calendar_uid_from_icaluid(self):
        """calendar_uid is extracted from google_calendar_event.iCalUID."""
        from kb.sync.granola import build_frontmatter

        doc = {
            "id": "doc-1",
            "title": "Test",
            "created_at": "2026-02-23T10:00:00Z",
            "updated_at": "2026-02-23T10:30:00Z",
            "google_calendar_event": {
                "summary": "Test",
                "iCalUID": "abc123@google.com",
                "organizer": {"email": "a@example.com"},
                "start": {"dateTime": "2026-02-23T10:00:00+01:00"},
                "end": {"dateTime": "2026-02-23T10:30:00+01:00"},
            },
        }
        fm = build_frontmatter(doc, {})
        assert fm["calendar_uid"] == "abc123@google.com"

    def test_calendar_uid_absent_without_icaluid(self):
        """calendar_uid is not set when iCalUID is missing."""
        from kb.sync.granola import build_frontmatter

        doc = {
            "id": "doc-1",
            "title": "Test",
            "created_at": "2026-02-23T10:00:00Z",
            "updated_at": "2026-02-23T10:30:00Z",
        }
        fm = build_frontmatter(doc, {})
        assert "calendar_uid" not in fm

    def test_date_from_calendar_event_start(self):
        """Date is derived from calendar event start, not created_at."""
        from kb.sync.granola import build_frontmatter

        doc = {
            "id": "doc-1",
            "title": "Test",
            "created_at": "2026-03-04T22:00:00Z",
            "updated_at": "2026-03-05T10:30:00Z",
            "google_calendar_event": {
                "summary": "Test",
                "iCalUID": "abc123@google.com",
                "organizer": {"email": "a@example.com"},
                "start": {"dateTime": "2026-03-05T10:00:00+01:00"},
                "end": {"dateTime": "2026-03-05T10:30:00+01:00"},
            },
        }
        fm = build_frontmatter(doc, {})
        # Should use calendar start (Mar 5), NOT created_at (Mar 4)
        assert fm["date"] == "2026-03-05"

    def test_date_falls_back_to_created_at(self):
        """Date falls back to created_at when no calendar event."""
        from kb.sync.granola import build_frontmatter

        doc = {
            "id": "doc-1",
            "title": "Test",
            "created_at": "2026-03-04T22:00:00Z",
            "updated_at": "2026-03-05T10:30:00Z",
        }
        fm = build_frontmatter(doc, {})
        assert fm["date"] == "2026-03-04"


class TestNormaliseCalendarUid:
    def test_strips_google_com(self):
        from kb.sync.granola import _normalise_calendar_uid

        assert _normalise_calendar_uid("abc123@google.com") == "abc123"

    def test_bare_uid_unchanged(self):
        from kb.sync.granola import _normalise_calendar_uid

        assert _normalise_calendar_uid("abc123") == "abc123"

    def test_strips_recurring_timestamp(self):
        from kb.sync.granola import _normalise_calendar_uid

        assert (
            _normalise_calendar_uid("cg3f4m6mmiq5ebpbaetrlbj0v1_20260305T083000Z")
            == "cg3f4m6mmiq5ebpbaetrlbj0v1"
        )

    def test_strips_recurring_r_prefix_and_google_com(self):
        from kb.sync.granola import _normalise_calendar_uid

        assert (
            _normalise_calendar_uid("cg3f4m6mmiq5ebpbaetrlbj0v1_R20260108T083000@google.com")
            == "cg3f4m6mmiq5ebpbaetrlbj0v1"
        )

    def test_both_variants_match(self):
        """Two UIDs for the same recurring event normalise to the same value."""
        from kb.sync.granola import _normalise_calendar_uid

        a = _normalise_calendar_uid("mfvcv93lenon8i0ckjtokt4o88_20260305T123000Z")
        b = _normalise_calendar_uid("mfvcv93lenon8i0ckjtokt4o88_R20260108T083000@google.com")
        assert a == b == "mfvcv93lenon8i0ckjtokt4o88"

    def test_simple_with_and_without_google_com(self):
        from kb.sync.granola import _normalise_calendar_uid

        a = _normalise_calendar_uid("7jap9887fcnj5ru93b6ulntsa5")
        b = _normalise_calendar_uid("7jap9887fcnj5ru93b6ulntsa5@google.com")
        assert a == b


class TestClassifyTranscript:
    def test_assemblyai_returns_none(self):
        """AssemblyAI transcript (has speaker field) → primary (None)."""
        from kb.sync.granola import classify_transcript

        segments = [
            {"speaker": "Speaker A", "text": "Hello"},
            {"speaker": "Speaker B", "text": "Hi there"},
        ]
        assert classify_transcript(segments) is None

    def test_mic_returns_mic(self):
        """Microphone recording (source field, no speaker) → 'mic'."""
        from kb.sync.granola import classify_transcript

        segments = [
            {"source": "microphone", "text": "Hello"},
            {"source": "system", "text": "Hi there"},
        ]
        assert classify_transcript(segments) == "mic"

    def test_empty_returns_mic(self):
        """Empty segments → secondary (mic) — never primary."""
        from kb.sync.granola import classify_transcript

        assert classify_transcript([]) == "mic"

    def test_mixed_prefers_assemblyai(self):
        """If any segment has speaker field, treat as AssemblyAI."""
        from kb.sync.granola import classify_transcript

        segments = [
            {"source": "microphone", "text": "Hello"},
            {"speaker": "Speaker A", "text": "Hi there"},
        ]
        assert classify_transcript(segments) is None


class TestFindExistingById:
    def test_find_old_pattern(self, tmp_dir):
        """Finds files using old naming pattern: {Title}_{id_prefix}.notes.md."""
        from kb.sync.granola import _find_existing_by_id

        day_dir = tmp_dir / "day"
        day_dir.mkdir()
        (day_dir / "Wren___Soren_aabb1122.notes.md").write_text("old")
        result = _find_existing_by_id(day_dir, "aabb1122")
        assert result == "Wren___Soren_aabb1122"

    def test_find_new_pattern(self, tmp_dir):
        """Finds files using new naming pattern: {id_prefix}_{Title}.granola.notes.md."""
        from kb.sync.granola import _find_existing_by_id

        day_dir = tmp_dir / "day"
        day_dir.mkdir()
        (day_dir / "aabb1122_Wren___Soren.granola.notes.md").write_text("new")
        result = _find_existing_by_id(day_dir, "aabb1122")
        assert result == "aabb1122_Wren___Soren"

    def test_prefers_new_over_old(self, tmp_dir):
        """When both patterns exist, prefers the new one."""
        from kb.sync.granola import _find_existing_by_id

        day_dir = tmp_dir / "day"
        day_dir.mkdir()
        (day_dir / "Wren___Soren_aabb1122.notes.md").write_text("old")
        (day_dir / "aabb1122_Wren___Soren.granola.notes.md").write_text("new")
        result = _find_existing_by_id(day_dir, "aabb1122")
        # New pattern takes precedence
        assert result == "aabb1122_Wren___Soren"

    def test_not_found(self, tmp_dir):
        """Returns None when no matching file exists."""
        from kb.sync.granola import _find_existing_by_id

        day_dir = tmp_dir / "day"
        day_dir.mkdir()
        (day_dir / "unrelated_file.notes.md").write_text("x")
        assert _find_existing_by_id(day_dir, "aabb1122") is None

    def test_missing_dir(self, tmp_dir):
        """Returns None when directory doesn't exist."""
        from kb.sync.granola import _find_existing_by_id

        assert _find_existing_by_id(tmp_dir / "nonexistent", "aabb1122") is None


class TestSanitizeTitle:
    def test_emoji_stripped(self):
        """Emoji prefixes are stripped so titles are consistent."""
        from kb.sync.granola import _sanitize_title

        assert _sanitize_title("🗓️ Weekly Delivery Sync") == _sanitize_title("Weekly Delivery Sync")

    def test_compound_emoji_stripped(self):
        """Compound emojis (ZWJ sequences) are fully stripped."""
        from kb.sync.granola import _sanitize_title

        assert _sanitize_title("👩‍⚕️ Visite médicale") == _sanitize_title("Visite médicale")

    def test_preserves_accented_chars(self):
        """Accented characters (é, ö, etc.) are preserved."""
        from kb.sync.granola import _sanitize_title

        result = _sanitize_title("Grégory / Linnea")
        assert "Grégory" in result
        assert "Linnea" in result

    def test_slash_separator(self):
        """Slash becomes triple underscore."""
        from kb.sync.granola import _sanitize_title

        assert _sanitize_title("Wren / Soren") == "Wren___Soren"


class TestMakeFilename:
    def test_new_pattern(self):
        """New pattern: {uid_prefix}_{Sanitized_Title}."""
        from kb.sync.granola import _make_filename

        result = _make_filename("Wren / Soren", "aabb112233445566")
        assert result == "aabb1122_Wren___Soren"

    def test_with_calendar_uid(self):
        """Uses calendar_uid when provided."""
        from kb.sync.granola import _make_filename

        result = _make_filename("Weekly 1:1", "aabb1122", calendar_uid="cal-uid-12345678")
        assert result.startswith("cal-uid-")
        assert "aabb1122" not in result

    def test_emoji_title_matches_plain(self):
        """Emoji and non-emoji titles produce the same filename."""
        from kb.sync.granola import _make_filename

        emoji_result = _make_filename("🗓️ Weekly Sync", "aabb1122")
        plain_result = _make_filename("Weekly Sync", "aabb1122")
        assert emoji_result == plain_result

    def test_without_calendar_uid(self):
        """Falls back to granola_id when no calendar_uid."""
        from kb.sync.granola import _make_filename

        result = _make_filename("Quick Chat", "def67890abcdef12")
        assert result.startswith("def67890")


class TestWriteMeeting:
    def test_write_meeting_file(self, tmp_dir):
        """Writes meeting to correct path with new naming pattern."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Wren / Soren",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        notes_md = "# Meeting Notes\n\nSome content."
        transcript_md = "**Wren Kasper**: Hello."
        frontmatter = {
            "title": "Wren / Soren",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": ["Soren"],
            "attendees": [],
        }

        result = write_meeting(doc, frontmatter, notes_md, transcript_md, tmp_dir)

        assert result["notes_path"].exists()
        assert result["transcript_path"].exists()
        assert "2026/01/27" in str(result["notes_path"])
        # New pattern: {uid_prefix}_{Title}.granola.notes.md
        assert result["notes_path"].name.startswith("aabb1122")
        assert result["notes_path"].name.endswith(".granola.notes.md")
        assert result["transcript_path"].name.endswith(".granola.transcript.md")

        content = result["notes_path"].read_text()
        assert "title: Wren / Soren" in content
        assert "# Meeting Notes" in content

    def test_write_meeting_file_with_calendar_uid(self, tmp_dir):
        """Uses calendar_uid for filename prefix when available."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Weekly 1:1",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        frontmatter = {
            "title": "Weekly 1:1",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "calendar_uid": "cal-uid-12345678",
            "tags": [],
            "attendees": [],
        }

        result = write_meeting(doc, frontmatter, "# Notes", "transcript", tmp_dir)
        # Should use calendar_uid prefix, not granola_id
        assert result["notes_path"].name.startswith("cal-uid-")
        assert "aabb1122" not in result["notes_path"].name

    def test_write_meeting_file_update(self, tmp_dir):
        """Overwrites when updated_at is newer."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Wren / Soren",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        frontmatter = {
            "title": "Wren / Soren",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": [],
            "attendees": [],
        }

        # Write initial version
        write_meeting(doc, frontmatter, "# Original", "transcript", tmp_dir)

        # Write updated version with newer timestamp
        doc_updated = {**doc, "updated_at": "2026-01-27T18:00:00Z"}
        fm_updated = {**frontmatter, "granola_updated_at": "2026-01-27T18:00:00Z"}
        result = write_meeting(doc_updated, fm_updated, "# Updated", "transcript v2", tmp_dir)

        assert result["status"] == "updated"
        content = result["notes_path"].read_text()
        assert "# Updated" in content

    def test_write_meeting_file_skip_unchanged(self, tmp_dir):
        """Skips when content is identical."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Wren / Soren",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        frontmatter = {
            "title": "Wren / Soren",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": [],
            "attendees": [],
        }

        # Write initial version
        write_meeting(doc, frontmatter, "# Same Content", "transcript", tmp_dir)

        # Try to write same content again (same updated_at)
        result = write_meeting(doc, frontmatter, "# Same Content", "transcript", tmp_dir)
        assert result["status"] == "skipped"

    def test_write_meeting_renames_old_pattern(self, tmp_dir):
        """Old-pattern files get renamed to new pattern during write."""
        from kb.sync.granola import write_meeting

        # Create old-pattern files manually
        day_dir = tmp_dir / "memory" / "meetings" / "2026" / "01" / "27"
        day_dir.mkdir(parents=True)
        old_notes = day_dir / "Wren___Soren_aabb1122.notes.md"
        old_transcript = day_dir / "Wren___Soren_aabb1122.transcript.md"
        old_notes.write_text(
            "---\ntitle: Wren / Soren\ndate: 2026-01-27\ntype: notes\n"
            "granola_id: aabb1122\ngranola_updated_at: '2026-01-27T15:00:00Z'\n---\n\n# Old\n"
        )
        old_transcript.write_text(
            "---\ntitle: Wren / Soren\ndate: 2026-01-27\ntype: transcript\n"
            "granola_id: aabb1122\n---\n\nOld transcript.\n"
        )

        doc = {
            "id": "aabb1122",
            "title": "Wren / Soren",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T18:00:00Z",
        }
        frontmatter = {
            "title": "Wren / Soren",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T18:00:00Z",
            "tags": [],
            "attendees": [],
        }

        result = write_meeting(doc, frontmatter, "# Updated", "New transcript", tmp_dir)

        # Old files should be gone
        assert not old_notes.exists()
        assert not old_transcript.exists()
        # New files should exist with .granola. pattern
        assert result["notes_path"].name.endswith(".granola.notes.md")
        assert result["transcript_path"].name.endswith(".granola.transcript.md")
        assert result["status"] == "updated"

    def test_write_meeting_mic_variant(self, tmp_dir):
        """variant='mic' produces .granola-mic. suffixed files."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Weekly 1:1",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        frontmatter = {
            "title": "Weekly 1:1",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": [],
            "attendees": [],
        }
        result = write_meeting(doc, frontmatter, "# Notes", "transcript", tmp_dir, variant="mic")
        assert result["notes_path"].name.endswith(".granola-mic.notes.md")
        assert result["transcript_path"].name.endswith(".granola-mic.transcript.md")
        assert result["notes_path"].exists()
        assert result["transcript_path"].exists()

    def test_write_meeting_both_variants(self, tmp_dir):
        """Primary and mic variant coexist in the same directory."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Weekly 1:1",
            "created_at": "2026-01-27T15:00:00Z",
            "updated_at": "2026-01-27T16:30:00Z",
        }
        fm = {
            "title": "Weekly 1:1",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": [],
            "attendees": [],
        }
        # Write primary (assemblyai)
        primary = write_meeting(doc, fm, "# AssemblyAI notes", "Speaker A: Hi", tmp_dir)
        # Write mic variant
        mic = write_meeting(doc, fm, "# Mic notes", "Me: Hi", tmp_dir, variant="mic")

        assert primary["notes_path"].exists()
        assert mic["notes_path"].exists()
        assert primary["notes_path"] != mic["notes_path"]
        assert ".granola.notes.md" in primary["notes_path"].name
        assert ".granola-mic.notes.md" in mic["notes_path"].name


class TestExtractNotesContent:
    def test_extracts_from_notes_field(self):
        """extract_notes_content reads from doc.notes.content."""
        from kb.sync.granola import extract_notes_content

        doc = {
            "notes": {"type": "doc", "content": [{"type": "paragraph"}]},
            "last_viewed_panel": {"content": {"type": "doc", "content": [{"type": "heading"}]}},
        }
        result = extract_notes_content(doc)
        assert result == [{"type": "paragraph"}]

    def test_ignores_panel(self):
        """extract_notes_content does NOT fall back to last_viewed_panel."""
        from kb.sync.granola import extract_notes_content

        doc = {
            "notes": None,
            "last_viewed_panel": {"content": {"type": "doc", "content": [{"type": "heading"}]}},
        }
        assert extract_notes_content(doc) == []

    def test_returns_empty_for_no_notes(self):
        """extract_notes_content returns [] when no notes field."""
        from kb.sync.granola import extract_notes_content

        assert extract_notes_content({}) == []

    def test_handles_notes_content_as_list(self):
        """extract_notes_content handles notes.content as a plain list."""
        from kb.sync.granola import extract_notes_content

        doc = {"notes": {"content": [{"type": "paragraph"}]}}
        assert extract_notes_content(doc) == [{"type": "paragraph"}]


class TestExtractPanelMarkdown:
    def test_extracts_panel_content(self):
        """extract_panel_markdown renders ProseMirror from last_viewed_panel."""
        from kb.sync.granola import extract_panel_markdown

        doc = {
            "last_viewed_panel": {
                "content": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "heading",
                            "attrs": {"level": 3},
                            "content": [{"type": "text", "text": "Summary"}],
                        },
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Key points here."}],
                        },
                    ],
                }
            }
        }
        md = extract_panel_markdown(doc)
        assert "### Summary" in md
        assert "Key points here." in md

    def test_returns_empty_for_no_panel(self):
        """extract_panel_markdown returns '' when no panel."""
        from kb.sync.granola import extract_panel_markdown

        assert extract_panel_markdown({}) == ""
        assert extract_panel_markdown({"last_viewed_panel": None}) == ""

    def test_returns_empty_for_empty_panel(self):
        """extract_panel_markdown returns '' for panel with no content."""
        from kb.sync.granola import extract_panel_markdown

        doc = {"last_viewed_panel": {"content": {"type": "doc", "content": []}}}
        assert extract_panel_markdown(doc) == ""

    def test_handles_content_as_list(self):
        """extract_panel_markdown handles content as a plain list of nodes."""
        from kb.sync.granola import extract_panel_markdown

        doc = {
            "last_viewed_panel": {
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Direct list."}],
                    }
                ]
            }
        }
        md = extract_panel_markdown(doc)
        assert "Direct list." in md


class TestWriteMeetingSummary:
    def test_writes_summary_file(self, tmp_dir):
        """write_meeting creates .granola.ai-summary.md when summary_md provided."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Standup",
            "created_at": "2026-01-27T15:00:00Z",
        }
        fm = {
            "title": "Standup",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": [],
            "attendees": [],
        }

        result = write_meeting(
            doc, fm, "# Notes", "transcript", tmp_dir, summary_md="### AI Summary\n\nKey points."
        )

        assert result["summary_path"] is not None
        assert result["summary_path"].exists()
        assert result["summary_path"].name.endswith(".granola.ai-summary.md")
        content = result["summary_path"].read_text()
        assert "type: summary" in content
        assert "### AI Summary" in content
        assert "Key points." in content

    def test_no_summary_file_when_empty(self, tmp_dir):
        """write_meeting does NOT create summary file when summary_md is empty."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Standup",
            "created_at": "2026-01-27T15:00:00Z",
        }
        fm = {
            "title": "Standup",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:30:00Z",
            "tags": [],
            "attendees": [],
        }

        result = write_meeting(doc, fm, "# Notes", "transcript", tmp_dir, summary_md="")

        assert result["summary_path"] is None
        # Only notes and transcript should exist
        assert result["notes_path"].exists()
        assert result["transcript_path"].exists()
        summary_files = list(tmp_dir.rglob("*.granola.ai-summary.md"))
        assert len(summary_files) == 0

    def test_summary_file_updated_on_resync(self, tmp_dir):
        """write_meeting updates summary file on resync with newer timestamp."""
        from kb.sync.granola import write_meeting

        doc = {
            "id": "aabb1122",
            "title": "Standup",
            "created_at": "2026-01-27T15:00:00Z",
        }
        fm_old = {
            "title": "Standup",
            "date": "2026-01-27",
            "type": "notes",
            "granola_id": "aabb1122",
            "granola_updated_at": "2026-01-27T16:00:00Z",
            "tags": [],
            "attendees": [],
        }
        fm_new = {**fm_old, "granola_updated_at": "2026-01-27T17:00:00Z"}

        # First write
        write_meeting(doc, fm_old, "# Notes", "transcript", tmp_dir, summary_md="### V1")

        # Second write with newer timestamp
        result = write_meeting(doc, fm_new, "# Notes", "transcript", tmp_dir, summary_md="### V2")

        assert result["status"] == "updated"
        assert result["summary_path"] is not None
        content = result["summary_path"].read_text()
        assert "### V2" in content


# ---------------------------------------------------------------------------
# Phase 3: CLI Integration
# ---------------------------------------------------------------------------


class TestSyncCLI:
    def test_sync_command_exists(self):
        """kb sync granola --help should work."""
        from click.testing import CliRunner

        from kb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["sync", "granola", "--help"])
        assert result.exit_code == 0
        assert "granola" in result.output.lower() or "sync" in result.output.lower()

    def test_sync_command_group(self):
        """kb sync --help lists granola subcommand."""
        from click.testing import CliRunner

        from kb.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["sync", "--help"])
        assert result.exit_code == 0
        assert "granola" in result.output

    def test_sync_dry_run(self, tmp_dir):
        """--dry-run reports but writes nothing."""
        from click.testing import CliRunner

        from kb.cli import cli

        mock_docs = [
            {
                "id": "dry-run-doc",
                "title": "Test Meeting",
                "created_at": "2026-02-10T15:00:00Z",
                "updated_at": "2026-02-10T16:00:00Z",
                "last_viewed_panel": {"doc": {"content": []}},
            },
        ]

        with (
            patch("kb.sync.granola.GranolaClient") as MockClient,
            patch("kb.cli._find_project_root", return_value=tmp_dir),
        ):
            instance = MockClient.return_value
            instance.list_documents.return_value = mock_docs
            instance.get_transcript.return_value = []
            instance.get_metadata.return_value = {"people": [], "calendar_event": None}

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["sync", "granola", "--legacy", "--dry-run"],
                env={"KB_DATA_DIR": str(tmp_dir / "data")},
            )

        assert result.exit_code == 0
        meetings_dir = tmp_dir / "memory" / "meetings"
        assert not meetings_dir.exists() or not list(meetings_dir.rglob("*.md"))

    def test_sync_since_date(self):
        """--since 2026-02-01 filters documents."""
        from click.testing import CliRunner

        from kb.cli import cli

        with patch("kb.sync.granola.GranolaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_documents.return_value = []

            runner = CliRunner()
            result = runner.invoke(
                cli, ["sync", "granola", "--legacy", "--since", "2026-02-01", "--dry-run"]
            )

        assert result.exit_code == 0
        instance.list_documents.assert_called_once_with(since="2026-02-01")

    def test_sync_state_saved(self, tmp_dir):
        """After sync, state file has last sync timestamp."""
        from kb.sync.granola import load_sync_state, save_sync_state

        state_path = tmp_dir / ".granola_sync_state.json"
        save_sync_state(state_path, last_sync="2026-02-10T16:00:00Z", docs_synced=5)

        state = load_sync_state(state_path)
        assert state["last_sync"] == "2026-02-10T16:00:00Z"
        assert state["docs_synced"] == 5

    def test_sync_state_missing(self, tmp_dir):
        """Missing state file returns empty state."""
        from kb.sync.granola import load_sync_state

        state_path = tmp_dir / ".granola_sync_state.json"
        state = load_sync_state(state_path)
        assert state == {}


# ---------------------------------------------------------------------------
# Phase 4: Entity Auto-Creation
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db():
    """Create a temporary database."""
    from kb.db import Database

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        yield db
        db.close()


class TestEntityAutoCreation:
    def _seed_entity(
        self, conn, name, entity_type="person", aliases=None, metadata=None, source_path=None
    ):
        """Helper to insert a test entity."""
        import json as _json

        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
            (
                name,
                entity_type,
                _json.dumps(aliases or []),
                _json.dumps(metadata or {}),
                source_path,
            ),
        )
        conn.commit()

    def test_match_attendee_by_email(self, tmp_db):
        """Known email maps to existing entity."""
        from kb.sync.granola import match_or_create_attendee

        conn = tmp_db.get_sqlite_conn()
        self._seed_entity(conn, "Wren Kasper", metadata={"email": "wren@example.com"})

        attendee = {"name": "Wren Kasper", "email": "wren@example.com"}
        result = match_or_create_attendee(attendee, tmp_db, project_root=None)

        assert result is not None
        assert result["name"] == "Wren Kasper"
        assert result["matched"]

    def test_match_attendee_by_name(self, tmp_db):
        """Known name maps to existing entity."""
        from kb.sync.granola import match_or_create_attendee

        conn = tmp_db.get_sqlite_conn()
        self._seed_entity(conn, "Soren Vance", aliases=["Soren"])

        attendee = {"name": "Soren Vance", "email": "soren@example.com"}
        result = match_or_create_attendee(attendee, tmp_db, project_root=None)

        assert result is not None
        assert result["name"] == "Soren Vance"
        assert result["matched"]

    def test_create_entity_for_unknown(self, tmp_db, tmp_dir):
        """Unknown attendee gets a new person entity + memory file."""
        from kb.sync.granola import match_or_create_attendee

        attendee = {"name": "New Person", "email": "new@example.com"}
        result = match_or_create_attendee(attendee, tmp_db, project_root=tmp_dir)

        assert result is not None
        assert result["name"] == "New Person"
        assert not result["matched"]
        assert result["created"]

        # Verify entity was created in DB
        conn = tmp_db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = ?", ("New Person",)).fetchone()
        assert row is not None
        assert row["entity_type"] == "person"

        # Verify memory file was created
        memory_file = tmp_dir / "memory" / "people" / "new-person.md"
        assert memory_file.exists()
        content = memory_file.read_text()
        assert "New Person" in content
        assert "new@example.com" in content

    def test_no_duplicate_entity_creation(self, tmp_db, tmp_dir):
        """Same attendee synced twice creates only one entity."""
        from kb.sync.granola import match_or_create_attendee

        attendee = {"name": "New Person", "email": "new@example.com"}

        # First call creates
        match_or_create_attendee(attendee, tmp_db, project_root=tmp_dir)
        # Second call matches existing
        result = match_or_create_attendee(attendee, tmp_db, project_root=tmp_dir)

        assert result["matched"]

        conn = tmp_db.get_sqlite_conn()
        rows = conn.execute("SELECT * FROM entities WHERE name = ?", ("New Person",)).fetchall()
        assert len(rows) == 1

    def test_entity_metadata_from_attendee(self, tmp_db, tmp_dir):
        """Email, company stored in entity metadata."""
        from kb.sync.granola import match_or_create_attendee

        attendee = {"name": "Jane Doe", "email": "jane@acme.com", "company": "Lattice Co"}
        match_or_create_attendee(attendee, tmp_db, project_root=tmp_dir)

        conn = tmp_db.get_sqlite_conn()
        row = conn.execute("SELECT * FROM entities WHERE name = ?", ("Jane Doe",)).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["email"] == "jane@acme.com"
        assert meta["company"] == "Lattice Co"

    def test_create_attendee_sets_updated_at(self, tmp_db, tmp_dir):
        """Auto-created attendee entity should have updated_at set to today."""
        from datetime import date

        from kb.sync.granola import match_or_create_attendee

        attendee = {"name": "Fresh Person", "email": "fresh@example.com"}
        match_or_create_attendee(attendee, tmp_db, project_root=tmp_dir)

        conn = tmp_db.get_sqlite_conn()
        row = conn.execute(
            "SELECT updated_at FROM entities WHERE name = ?", ("Fresh Person",)
        ).fetchone()
        assert row["updated_at"] == date.today().isoformat()


# ---------------------------------------------------------------------------
# Phase 5: Pipeline Updates
# ---------------------------------------------------------------------------


class TestEnrichedFrontmatter:
    def test_enriched_frontmatter_attendees_parsed(self):
        """parse_frontmatter reads the attendees list."""
        from kb.chunker import parse_frontmatter

        text = """---
title: Wren / Soren
date: 2026-01-27
type: notes
granola_id: abc12345
granola_updated_at: '2026-01-27T16:30:00Z'
tags:
  - Soren
attendees:
  - name: Wren Kasper
    email: wren@example.com
  - name: Soren Vance
    email: soren@example.com
calendar_event:
  title: Weekly 1:1
  organiser: wren@example.com
  scheduled_start: '2026-01-27T15:00:00Z'
  scheduled_end: '2026-01-27T15:30:00Z'
granola_summary: |
  Discussed performance review feedback.
---
# Meeting Notes
"""
        fm = parse_frontmatter(text)
        assert fm["title"] == "Wren / Soren"
        assert len(fm["attendees"]) == 2
        assert fm["attendees"][0]["email"] == "wren@example.com"
        assert fm["calendar_event"]["title"] == "Weekly 1:1"
        assert "performance review" in fm["granola_summary"]
        assert fm["granola_updated_at"] == "2026-01-27T16:30:00Z"

    def test_backward_compat_old_frontmatter(self):
        """Old files without attendees still work."""
        from kb.chunker import parse_frontmatter

        text = """---
title: Old Meeting
date: 2025-06-15
type: notes
granola_id: old12345
tags:
  - Soren
---
# Meeting Notes
"""
        fm = parse_frontmatter(text)
        assert fm["title"] == "Old Meeting"
        assert fm["attendees"] == []
        assert fm["calendar_event"] is None
        assert fm["granola_summary"] is None
        assert fm["granola_updated_at"] is None

    def test_walk_meetings_enriched_frontmatter(self, tmp_dir):
        """walk_meetings populates attendees on ParsedDocument."""
        from kb.sources.meetings import walk_meetings

        # Create a meeting file with enriched frontmatter
        day_dir = tmp_dir / "memory" / "meetings" / "2026" / "01" / "27"
        day_dir.mkdir(parents=True)
        notes = day_dir / "abc12345_Wren___Soren.granola.notes.md"
        notes.write_text("""---
title: Wren / Soren
date: 2026-01-27
type: notes
granola_id: abc12345
tags:
  - Soren
attendees:
  - name: Wren Kasper
    email: wren@example.com
  - name: Soren Vance
    email: soren@example.com
---
## Discussion

Talked about performance review.
""")

        docs = list(walk_meetings(tmp_dir))
        assert len(docs) == 1
        assert docs[0].title == "Wren / Soren"
        # Attendees should be available on the parsed document
        assert len(docs[0].attendees) == 2
        assert docs[0].attendees[0]["email"] == "wren@example.com"

    def test_granola_summary_in_view(self, tmp_dir):
        """kb view --json includes granola_summary field when present."""
        # This tests the frontmatter parsing — the actual view command uses
        # the chunks from the DB. We test that the field is preserved through
        # parse_frontmatter so it can be stored and displayed.
        from kb.chunker import parse_frontmatter

        text = """---
title: Meeting with Summary
date: 2026-02-10
type: notes
granola_id: sum12345
granola_summary: |
  AI-generated summary of the meeting.
---
# Notes
"""
        fm = parse_frontmatter(text)
        assert fm["granola_summary"] is not None
        assert "AI-generated summary" in fm["granola_summary"]


# ---------------------------------------------------------------------------
# Phase 6: Token write atomic
# ---------------------------------------------------------------------------


class TestTokenWriteAtomic:
    def test_preserves_string_encoded_tokens(self, tmp_dir):
        """_write_tokens_atomic preserves double-encoded (string) workos_tokens format."""
        token_path = tmp_dir / "supabase.json"
        inner = json.dumps({"access_token": "old", "refresh_token": "old-ref"})
        token_path.write_text(json.dumps({"workos_tokens": inner, "other": "data"}))

        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path)
        client._write_tokens_atomic("new-access", "new-refresh")

        saved = json.loads(token_path.read_text())
        # workos_tokens should remain as a string
        assert isinstance(saved["workos_tokens"], str)
        inner_parsed = json.loads(saved["workos_tokens"])
        assert inner_parsed["access_token"] == "new-access"
        assert inner_parsed["refresh_token"] == "new-refresh"
        # Other keys preserved
        assert saved["other"] == "data"

    def test_preserves_dict_encoded_tokens(self, tmp_dir):
        """_write_tokens_atomic preserves dict-format workos_tokens."""
        token_path = tmp_dir / "supabase.json"
        token_path.write_text(
            json.dumps({"workos_tokens": {"access_token": "old", "refresh_token": "old-ref"}})
        )

        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path)
        client._write_tokens_atomic("new-access", "new-refresh")

        saved = json.loads(token_path.read_text())
        # workos_tokens should remain as dict
        assert isinstance(saved["workos_tokens"], dict)
        assert saved["workos_tokens"]["access_token"] == "new-access"
        assert saved["workos_tokens"]["refresh_token"] == "new-refresh"


# ---------------------------------------------------------------------------
# Phase 7: Request retry on 401
# ---------------------------------------------------------------------------


class TestRequestRetryOn401:
    def test_retries_on_401_then_succeeds(self, supabase_json):
        """_request retries once on 401 with refreshed tokens."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        # First call returns 401, second succeeds
        mock_401 = MagicMock()
        mock_401.status_code = 401

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.raise_for_status = MagicMock()
        mock_200.json.return_value = {"result": "ok"}

        call_count = [0]

        def mock_request(method, url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_401
            return mock_200

        with (
            patch("httpx.request", side_effect=mock_request),
            patch.object(client, "_refresh_tokens") as mock_refresh,
        ):
            result = client._request("GET", "/v1/test")

        # Should have called refresh once
        mock_refresh.assert_called_once()
        # Result should be the success response
        assert result.json() == {"result": "ok"}

    def test_no_retry_when_retry_false(self, supabase_json):
        """_request with retry=False raises on 401 without retrying."""
        import httpx

        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        mock_401 = MagicMock(spec=httpx.Response)
        mock_401.status_code = 401
        mock_401.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_401
        )

        with patch("httpx.request", return_value=mock_401), pytest.raises(httpx.HTTPStatusError):
            client._request("GET", "/v1/test", retry=False)


# ---------------------------------------------------------------------------
# Phase 8: Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_list_documents_paginates(self, supabase_json):
        """list_documents fetches multiple pages until empty response."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        page1_docs = [{"id": f"doc-{i}", "title": f"Doc {i}"} for i in range(50)]
        page2_docs = [{"id": f"doc-{i}", "title": f"Doc {i}"} for i in range(50, 75)]

        page1_response = MagicMock()
        page1_response.status_code = 200
        page1_response.json.return_value = {"docs": page1_docs}
        page1_response.raise_for_status = MagicMock()

        page2_response = MagicMock()
        page2_response.status_code = 200
        page2_response.json.return_value = {"docs": page2_docs}
        page2_response.raise_for_status = MagicMock()

        call_count = [0]

        def mock_request(method, path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return page1_response
            return page2_response

        with patch.object(client, "_request", side_effect=mock_request), patch("time.sleep"):
            docs = client.list_documents()

        # 50 from page1 + 25 from page2
        assert len(docs) == 75
        # Two API calls made
        assert call_count[0] == 2

    def test_list_documents_single_page(self, supabase_json):
        """list_documents handles single-page result (fewer than limit)."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        page_docs = [{"id": "doc-1", "title": "Only Doc"}]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"docs": page_docs}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", return_value=mock_response):
            docs = client.list_documents()

        assert len(docs) == 1
        assert docs[0]["id"] == "doc-1"

    def test_list_documents_empty(self, supabase_json):
        """list_documents returns empty list for no documents."""
        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"docs": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_request", return_value=mock_response):
            docs = client.list_documents()

        assert docs == []

    def test_list_documents_raises_on_unsupported_client(self, supabase_json):
        """Granola rejecting our client (200 + ``{message: ...}``, no ``docs``) must raise."""
        import pytest

        from kb.sync.granola import GranolaAPIError, GranolaClient

        client = GranolaClient(token_path=supabase_json)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Unsupported client"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(client, "_request", return_value=mock_response),
            pytest.raises(GranolaAPIError, match="Unsupported client"),
        ):
            client.list_documents()


class TestGetUserId:
    def test_returns_user_id(self, client):
        """get_user_id extracts user_id from the first document."""
        docs = [{"id": "doc-1", "user_id": "user-abc-123"}]

        with patch.object(client, "list_documents", return_value=docs):
            uid = client.get_user_id()

        assert uid == "user-abc-123"

    def test_skips_docs_without_user_id(self, client):
        """get_user_id skips documents missing user_id."""
        docs = [
            {"id": "doc-1"},
            {"id": "doc-2", "user_id": "user-xyz-789"},
        ]

        with patch.object(client, "list_documents", return_value=docs):
            uid = client.get_user_id()

        assert uid == "user-xyz-789"

    def test_raises_on_empty(self, client):
        """get_user_id raises ValueError when no documents have user_id."""
        with (
            patch.object(client, "list_documents", return_value=[]),
            pytest.raises(ValueError, match="Could not determine user ID"),
        ):
            client.get_user_id()


class TestFindDocument:
    def test_find_by_title(self, client):
        """find_document matches by title substring (case-insensitive)."""
        docs = [
            {"id": "doc-1", "title": "Weekly Standup"},
            {"id": "doc-2", "title": "1:1 with Wren"},
        ]
        with (
            patch.object(client, "list_documents", return_value=docs),
            patch("time.sleep"),
        ):
            result = client.find_document(title="standup")

        assert result is not None
        assert result["id"] == "doc-1"

    def test_find_by_calendar_uid(self, client):
        """find_document matches by calendar event UID."""
        docs = [
            {
                "id": "doc-1",
                "title": "Meeting",
                "google_calendar_event": {"iCalUID": "abc123@google.com"},
            },
        ]
        with (
            patch.object(client, "list_documents", return_value=docs),
            patch("time.sleep"),
        ):
            result = client.find_document(calendar_uid="abc123")

        assert result is not None
        assert result["id"] == "doc-1"

    def test_returns_none_when_not_found(self, client):
        """find_document returns None when no match."""
        with (
            patch.object(client, "list_documents", return_value=[]),
            patch("time.sleep"),
        ):
            result = client.find_document(title="nonexistent")

        assert result is None

    def test_calendar_uid_takes_priority(self, client):
        """Calendar UID match is checked before title match."""
        docs = [
            {
                "id": "doc-1",
                "title": "Standup",
                "google_calendar_event": {"iCalUID": "uid-match@google.com"},
            },
            {"id": "doc-2", "title": "Standup"},
        ]
        with (
            patch.object(client, "list_documents", return_value=docs),
            patch("time.sleep"),
        ):
            result = client.find_document(title="standup", calendar_uid="uid-match")

        assert result is not None
        assert result["id"] == "doc-1"

    def test_find_recurring_event_by_instance_id(self, client):
        """find_document matches recurring event via id field when iCalUID uses base date."""
        docs = [
            {
                "id": "doc-1",
                "title": "Platform Stability - weekly sync",
                "google_calendar_event": {
                    # iCalUID uses recurring base date format — won't substring-match
                    "iCalUID": "abc123_R20260223T103000@google.com",
                    # id uses the specific instance date — should match
                    "id": "abc123_20260309T103000Z",
                },
            },
        ]
        with (
            patch.object(client, "list_documents", return_value=docs),
            patch("time.sleep"),
        ):
            result = client.find_document(calendar_uid="abc123_20260309T103000Z")

        assert result is not None
        assert result["id"] == "doc-1"

    def test_find_recurring_event_no_false_positive_across_instances(self, client):
        """find_document does not match a different instance of the same recurring event."""
        docs = [
            {
                "id": "doc-march-2",
                "title": "Weekly Sync",
                "google_calendar_event": {
                    "iCalUID": "abc123_R20260223T103000@google.com",
                    "id": "abc123_20260302T103000Z",
                },
            },
        ]
        with (
            patch.object(client, "list_documents", return_value=docs),
            patch("time.sleep"),
        ):
            # Search for March 9 instance — should NOT match March 2 doc
            result = client.find_document(calendar_uid="abc123_20260309T103000Z")

        assert result is None

    def test_find_by_icaluid_still_works(self, client):
        """find_document still matches when iCalUID contains the search UID."""
        docs = [
            {
                "id": "doc-1",
                "title": "One-off Meeting",
                "google_calendar_event": {
                    "iCalUID": "simple123@google.com",
                },
            },
        ]
        with (
            patch.object(client, "list_documents", return_value=docs),
            patch("time.sleep"),
        ):
            result = client.find_document(calendar_uid="simple123")

        assert result is not None
        assert result["id"] == "doc-1"


class TestRequestProxyStripping:
    """_request should strip SOCKS proxy env vars before making HTTP calls."""

    def test_request_strips_socks_proxy_vars(self, supabase_json):
        """_request strips ALL_PROXY and related vars so httpx doesn't try SOCKS."""
        import os

        from kb.sync.granola import GranolaClient

        client = GranolaClient(token_path=supabase_json)

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.raise_for_status = MagicMock()

        captured_env = {}

        def mock_request(method, url, **kwargs):
            # Capture the proxy env vars at the moment httpx.request is called
            for var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
                captured_env[var] = os.environ.get(var)
            return mock_200

        # Set SOCKS proxy vars as the sandbox would
        os.environ["ALL_PROXY"] = "socks5h://localhost:59537"
        os.environ["all_proxy"] = "socks5h://localhost:59537"  # noqa: SIM112
        try:
            with patch("httpx.request", side_effect=mock_request):
                client._request("GET", "/v1/test")

            # Proxy vars should have been stripped before the httpx call
            assert captured_env["ALL_PROXY"] is None
            assert captured_env["all_proxy"] is None
        finally:
            os.environ.pop("ALL_PROXY", None)
            os.environ.pop("all_proxy", None)


class TestGranolaViewCLI:
    """Tests for ``kbx granola view``."""

    def _make_doc(self, **overrides):
        doc = {
            "id": "doc-1",
            "title": "Weekly Standup",
            "created_at": "2026-03-03T10:00:00.000Z",
            "notes_markdown": "## Notes\n\n- Item 1\n- Item 2",
            "google_calendar_event": {"id": "uid-abc_20260303T100000Z"},
            "last_viewed_panel": {
                "content": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "heading",
                            "attrs": {"level": 2, "id": "h1"},
                            "content": [{"type": "text", "text": "Summary"}],
                        },
                        {
                            "type": "paragraph",
                            "attrs": {"id": "p1"},
                            "content": [{"type": "text", "text": "Key decisions were made."}],
                        },
                    ],
                }
            },
        }
        doc.update(overrides)
        return doc

    def test_view_basic(self):
        """granola view shows YAML header + notes markdown."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()

        with patch("kb.sync.granola.GranolaClient", return_value=mock_client):
            result = runner.invoke(cli, ["granola", "view", "uid-abc_20260303T100000Z"])

        assert result.exit_code == 0
        assert "title: Weekly Standup" in result.output
        assert "date: '2026-03-03'" in result.output or "date: 2026-03-03" in result.output
        assert "## Notes" in result.output
        assert "- Item 1" in result.output

    def test_view_plain(self):
        """granola view --plain outputs raw markdown without YAML header."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()

        with patch("kb.sync.granola.GranolaClient", return_value=mock_client):
            result = runner.invoke(cli, ["granola", "view", "uid-abc_20260303T100000Z", "--plain"])

        assert result.exit_code == 0
        assert "---" not in result.output
        assert "## Notes" in result.output

    def test_view_summary(self):
        """granola view --summary shows AI summary instead of notes."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()

        with patch("kb.sync.granola.GranolaClient", return_value=mock_client):
            result = runner.invoke(
                cli, ["granola", "view", "uid-abc_20260303T100000Z", "--summary"]
            )

        assert result.exit_code == 0
        assert "Key decisions were made." in result.output
        # Should NOT contain the notes
        assert "- Item 1" not in result.output

    def test_view_all(self):
        """granola view --all shows notes + summary + transcript."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()
        mock_client.get_transcript.return_value = [
            {"text": "Transcript line.", "start_timestamp": 0.0, "end_timestamp": 2.0},
        ]

        with (
            patch("kb.sync.granola.GranolaClient", return_value=mock_client),
            patch(
                "kb.sync.granola.transcript_to_markdown",
                return_value="Transcript line.",
            ),
        ):
            result = runner.invoke(cli, ["granola", "view", "uid-abc_20260303T100000Z", "--all"])

        assert result.exit_code == 0
        assert "## Notes" in result.output
        assert "Key decisions were made." in result.output
        assert "Transcript line." in result.output

    def test_view_transcript(self):
        """granola view --transcript fetches and shows transcript."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()
        mock_client.get_transcript.return_value = [
            {"text": "Hello everyone.", "start_timestamp": 0.0, "end_timestamp": 2.0},
            {"text": "Let's start.", "start_timestamp": 2.0, "end_timestamp": 4.0},
        ]

        with (
            patch("kb.sync.granola.GranolaClient", return_value=mock_client),
            patch(
                "kb.sync.granola.transcript_to_markdown",
                return_value="Hello everyone.\n\nLet's start.",
            ),
        ):
            result = runner.invoke(
                cli, ["granola", "view", "uid-abc_20260303T100000Z", "--transcript"]
            )

        assert result.exit_code == 0
        assert "Hello everyone." in result.output
        assert "Let's start." in result.output
        # Should NOT contain notes
        assert "- Item 1" not in result.output

    def test_view_no_transcript(self):
        """granola view --transcript shows '(no transcript)' when empty."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()
        mock_client.get_transcript.return_value = []

        with (
            patch("kb.sync.granola.GranolaClient", return_value=mock_client),
            patch("kb.sync.granola.transcript_to_markdown", return_value=""),
        ):
            result = runner.invoke(cli, ["granola", "view", "uid-abc", "--transcript"])

        assert result.exit_code == 0
        assert "(no transcript)" in result.output

    def test_view_not_found(self):
        """granola view errors when no doc found."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = None

        with patch("kb.sync.granola.GranolaClient", return_value=mock_client):
            result = runner.invoke(cli, ["granola", "view", "missing-uid"])

        assert result.exit_code != 0
        assert "No Granola document found" in result.output

    def test_view_no_notes(self):
        """granola view shows '(no notes)' when doc has no notes."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc(notes_markdown="")

        with patch("kb.sync.granola.GranolaClient", return_value=mock_client):
            result = runner.invoke(cli, ["granola", "view", "uid-abc"])

        assert result.exit_code == 0
        assert "(no notes)" in result.output

    def test_view_yaml_has_files_field(self):
        """granola view YAML header includes files availability."""
        from kb.cli import cli

        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.find_document.return_value = self._make_doc()

        with patch("kb.sync.granola.GranolaClient", return_value=mock_client):
            result = runner.invoke(cli, ["granola", "view", "uid-abc"])

        assert result.exit_code == 0
        assert "notes: true" in result.output
        assert "ai_summary: true" in result.output
