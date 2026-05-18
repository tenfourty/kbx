"""Tests for the Granola public-API sync path (kb.sync.granola_public)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def project_root(tmp_dir):
    (tmp_dir / "memory" / "meetings").mkdir(parents=True)
    return tmp_dir


@pytest.fixture
def data_dir(tmp_dir):
    d = tmp_dir / "data"
    d.mkdir()
    return d


@pytest.fixture
def sample_note_summary():
    return {
        "id": "not_abc123def456",
        "object": "note",
        "title": "Quarterly yoghurt budget review",
        "owner": {"name": "Idris Kalmar", "email": "idris@example.com"},
        "created_at": "2026-05-15T15:30:00Z",
        "updated_at": "2026-05-15T16:45:00Z",
    }


@pytest.fixture
def sample_note_full(sample_note_summary):
    return {
        **sample_note_summary,
        "web_url": "https://notes.granola.ai/d/abc",
        "calendar_event": {
            "event_title": "Quarterly yoghurt budget review",
            "invitees": [],
            "organiser": "idris@example.com",
            "calendar_event_id": "evt_xyz789_20260515T153000Z",
            "scheduled_start_time": "2026-05-15T15:30:00Z",
            "scheduled_end_time": "2026-05-15T16:30:00Z",
        },
        "attendees": [
            {"name": "Idris Kalmar", "email": "idris@example.com"},
            {"name": "Wren Kasper", "email": "alice@example.com"},
        ],
        "folder_membership": [],
        "summary_text": "Spent $100k, made $150k.",
        "summary_markdown": "## Summary\n\nSpent $100k, made $150k.\n",
        "transcript": [
            {
                "speaker": {"source": "microphone"},
                "text": "Welcome everyone.",
                "start_time": "2026-05-15T15:30:00Z",
                "end_time": "2026-05-15T15:30:05Z",
            },
            {
                "speaker": {"source": "speaker", "diarization_label": "Speaker A"},
                "text": "Thanks for joining.",
                "start_time": "2026-05-15T15:30:05Z",
                "end_time": "2026-05-15T15:30:10Z",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


class TestGetAPIKey:
    def test_reads_from_env_var(self, monkeypatch):
        from kb.sync.granola_public import _get_api_key

        monkeypatch.setenv("GRANOLA_API_KEY", "grn_env_value")
        assert _get_api_key() == "grn_env_value"

    def test_env_var_takes_precedence_over_keychain(self, monkeypatch):
        from kb.sync.granola_public import _get_api_key

        monkeypatch.setenv("GRANOLA_API_KEY", "grn_env_value")
        with patch("subprocess.run") as mock_run:
            assert _get_api_key() == "grn_env_value"
            mock_run.assert_not_called()

    def test_reads_from_keychain_when_env_missing(self, monkeypatch):
        from kb.sync.granola_public import _get_api_key

        monkeypatch.delenv("GRANOLA_API_KEY", raising=False)
        monkeypatch.setenv("USER", "testuser")
        mock_result = MagicMock(stdout="grn_keychain_value\n")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert _get_api_key() == "grn_keychain_value"
            args = mock_run.call_args[0][0]
            assert args[0] == "security"
            assert "testuser" in args
            assert "granola_api_key" in args

    def test_raises_when_keychain_missing(self, monkeypatch):
        from kb.sync.granola_public import GranolaPublicAPIError, _get_api_key

        monkeypatch.delenv("GRANOLA_API_KEY", raising=False)
        monkeypatch.setenv("USER", "testuser")
        err = subprocess.CalledProcessError(44, ["security"])
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(GranolaPublicAPIError, match="not found in Keychain"):
                _get_api_key()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TestGranolaPublicClient:
    def test_list_notes_paginates_until_no_more(self):
        from kb.sync.granola_public import GranolaPublicClient

        client = GranolaPublicClient(api_key="grn_test")
        responses = [
            {"notes": [{"id": "n1"}, {"id": "n2"}], "hasMore": True, "cursor": "c1"},
            {"notes": [{"id": "n3"}], "hasMore": False, "cursor": None},
        ]
        with patch.object(client, "_request", side_effect=responses) as mock_req:
            notes = list(client.list_notes(updated_after="2026-05-01T00:00:00Z"))

        assert [n["id"] for n in notes] == ["n1", "n2", "n3"]
        assert mock_req.call_count == 2
        first_call_params = mock_req.call_args_list[0].kwargs["params"]
        assert first_call_params["updated_after"] == "2026-05-01T00:00:00Z"
        assert first_call_params["page_size"] == 30
        second_call_params = mock_req.call_args_list[1].kwargs["params"]
        assert second_call_params["cursor"] == "c1"

    def test_list_notes_clamps_page_size_to_max(self):
        from kb.sync.granola_public import GranolaPublicClient

        client = GranolaPublicClient(api_key="grn_test")
        with patch.object(
            client, "_request", return_value={"notes": [], "hasMore": False}
        ) as mock_req:
            list(client.list_notes(page_size=500))

        assert mock_req.call_args.kwargs["params"]["page_size"] == 30

    def test_get_note_includes_transcript_by_default(self):
        from kb.sync.granola_public import GranolaPublicClient

        client = GranolaPublicClient(api_key="grn_test")
        with patch.object(
            client, "_request", return_value={"id": "not_abc"}
        ) as mock_req:
            client.get_note("not_abc")

        assert mock_req.call_args.args == ("GET", "/notes/not_abc")
        assert mock_req.call_args.kwargs["params"] == {"include": "transcript"}

    def test_request_raises_on_401(self):
        from kb.sync.granola_public import GranolaPublicAPIError, GranolaPublicClient

        client = GranolaPublicClient(api_key="grn_bad")
        mock_response = MagicMock(status_code=401)
        with patch("httpx.request", return_value=mock_response):
            with pytest.raises(GranolaPublicAPIError, match="401"):
                client._request("GET", "/notes")

    def test_request_raises_on_429(self):
        from kb.sync.granola_public import GranolaPublicAPIError, GranolaPublicClient

        client = GranolaPublicClient(api_key="grn_test")
        mock_response = MagicMock(status_code=429)
        with patch("httpx.request", return_value=mock_response):
            with pytest.raises(GranolaPublicAPIError, match="429"):
                client._request("GET", "/notes")


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


class TestNoteToInternalDoc:
    def test_maps_top_level_fields(self, sample_note_full):
        from kb.sync.granola_public import _note_to_internal_doc

        doc = _note_to_internal_doc(sample_note_full)
        assert doc["id"] == "not_abc123def456"
        assert doc["title"] == "Quarterly yoghurt budget review"
        assert doc["created_at"] == "2026-05-15T15:30:00Z"
        assert doc["updated_at"] == "2026-05-15T16:45:00Z"
        assert doc["notes_markdown"].startswith("## Summary")

    def test_maps_calendar_event_to_google_calendar_event(self, sample_note_full):
        from kb.sync.granola_public import _note_to_internal_doc

        doc = _note_to_internal_doc(sample_note_full)
        gcal = doc["google_calendar_event"]
        assert gcal["iCalUID"] == "evt_xyz789_20260515T153000Z"
        assert gcal["organizer"]["email"] == "idris@example.com"
        assert gcal["start"]["dateTime"] == "2026-05-15T15:30:00Z"
        assert gcal["end"]["dateTime"] == "2026-05-15T16:30:00Z"
        assert gcal["summary"] == "Quarterly yoghurt budget review"

    def test_maps_owner_to_creator_and_includes_attendees(self, sample_note_full):
        from kb.sync.granola_public import _note_to_internal_doc

        doc = _note_to_internal_doc(sample_note_full)
        assert doc["people"]["creator"]["email"] == "idris@example.com"
        emails = [a["email"] for a in doc["people"]["attendees"]]
        assert "alice@example.com" in emails

    def test_handles_missing_calendar_event(self, sample_note_summary):
        from kb.sync.granola_public import _note_to_internal_doc

        doc = _note_to_internal_doc({**sample_note_summary, "transcript": []})
        assert "google_calendar_event" not in doc

    def test_uses_untitled_when_title_null(self, sample_note_full):
        from kb.sync.granola_public import _note_to_internal_doc

        doc = _note_to_internal_doc({**sample_note_full, "title": None})
        assert doc["title"] == "Untitled"


class TestTransformTranscript:
    def test_diarized_segment_uses_speaker_label(self):
        from kb.sync.granola_public import _transform_transcript

        segments = [
            {
                "speaker": {"source": "speaker", "diarization_label": "Speaker A"},
                "text": "Hello.",
                "start_time": "2026-05-15T15:30:00Z",
                "end_time": "2026-05-15T15:30:05Z",
            }
        ]
        result = _transform_transcript(segments)
        assert result[0]["speaker"] == "Speaker A"
        assert "source" not in result[0]

    def test_non_diarized_microphone_maps_to_internal_microphone(self):
        from kb.sync.granola_public import _transform_transcript

        segments = [{"speaker": {"source": "microphone"}, "text": "Yo."}]
        result = _transform_transcript(segments)
        assert result[0]["source"] == "microphone"

    def test_non_diarized_speaker_maps_to_internal_system(self):
        """Public-API 'speaker' (system audio) → internal 'system' label."""
        from kb.sync.granola_public import _transform_transcript

        segments = [{"speaker": {"source": "speaker"}, "text": "Hi."}]
        result = _transform_transcript(segments)
        assert result[0]["source"] == "system"

    def test_renders_to_markdown_with_internal_labels(self):
        from kb.sync.granola import transcript_to_markdown
        from kb.sync.granola_public import _transform_transcript

        segments = [
            {"speaker": {"source": "microphone"}, "text": "Hi."},
            {"speaker": {"source": "speaker"}, "text": "Hello."},
        ]
        md = transcript_to_markdown(_transform_transcript(segments))
        assert "**Me**" in md
        assert "**System**" in md


# ---------------------------------------------------------------------------
# End-to-end sync
# ---------------------------------------------------------------------------


class TestSyncGranolaPublic:
    def test_creates_files_with_expected_frontmatter(
        self, project_root, data_dir, sample_note_summary, sample_note_full
    ):
        from kb.sync.granola_public import sync_granola_public

        with patch("kb.sync.granola_public.GranolaPublicClient") as MockClient:
            mock = MockClient.return_value
            mock.list_notes.return_value = iter([sample_note_summary])
            mock.get_note.return_value = sample_note_full

            result = sync_granola_public(
                project_root=project_root,
                data_dir=data_dir,
                since="2026-05-01",
                api_key="grn_test",
            )

        assert result["created"] == 1
        assert result["total"] == 1

        meetings = list((project_root / "memory" / "meetings").rglob("*.granola.notes.md"))
        assert len(meetings) == 1
        notes_path = meetings[0]
        content = notes_path.read_text()
        assert "source: granola-public-api" in content
        assert "granola_id: not_abc123def456" in content
        assert "calendar_uid: evt_xyz789_20260515T153000Z" in content
        # Date directory derives from calendar_event start, not created_at
        assert "/2026/05/15/" in str(notes_path)

    def test_writes_transcript_with_internal_labels(
        self, project_root, data_dir, sample_note_summary, sample_note_full
    ):
        from kb.sync.granola_public import sync_granola_public

        with patch("kb.sync.granola_public.GranolaPublicClient") as MockClient:
            mock = MockClient.return_value
            mock.list_notes.return_value = iter([sample_note_summary])
            mock.get_note.return_value = sample_note_full

            sync_granola_public(
                project_root=project_root, data_dir=data_dir, api_key="grn_test"
            )

        transcript_path = next(
            (project_root / "memory" / "meetings").rglob("*.granola.transcript.md")
        )
        content = transcript_path.read_text()
        # First segment is mic-only → 'Me'; second is diarized → 'Speaker A'
        assert "**Me**" in content
        assert "**Speaker A**" in content

    def test_dry_run_does_not_write_files(
        self, project_root, data_dir, sample_note_summary, sample_note_full
    ):
        from kb.sync.granola_public import sync_granola_public

        with patch("kb.sync.granola_public.GranolaPublicClient") as MockClient:
            mock = MockClient.return_value
            mock.list_notes.return_value = iter([sample_note_summary])
            mock.get_note.return_value = sample_note_full

            result = sync_granola_public(
                project_root=project_root,
                data_dir=data_dir,
                api_key="grn_test",
                dry_run=True,
            )

        assert result.get("dry_run") is True
        assert result["created"] == 0
        meetings = list((project_root / "memory" / "meetings").rglob("*.md"))
        assert meetings == []

    def test_promotes_bare_date_since_to_iso(
        self, project_root, data_dir, sample_note_summary, sample_note_full
    ):
        from kb.sync.granola_public import sync_granola_public

        with patch("kb.sync.granola_public.GranolaPublicClient") as MockClient:
            mock = MockClient.return_value
            mock.list_notes.return_value = iter([])
            sync_granola_public(
                project_root=project_root,
                data_dir=data_dir,
                since="2026-05-01",
                api_key="grn_test",
            )

        mock.list_notes.assert_called_once()
        assert mock.list_notes.call_args.kwargs["updated_after"] == "2026-05-01T00:00:00Z"
