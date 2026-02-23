"""Tests for Notion sync — token reading and API client."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from kb.sync.notion import NotionClient


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestTokenReading:
    """Phase 1: Token decryption and reading."""

    def test_token_from_env_var(self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """NOTION_TOKEN_V2 env var takes priority over cookie store."""
        from kb.sync.notion import NotionClient

        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token-from-env")
        client = NotionClient(cookie_path=tmp_dir / "nonexistent")
        assert client._get_token() == "test-token-from-env"

    def test_missing_cookie_file_raises(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing cookie DB raises FileNotFoundError."""
        from kb.sync.notion import NotionClient

        monkeypatch.delenv("NOTION_TOKEN_V2", raising=False)
        client = NotionClient(cookie_path=tmp_dir / "missing.db")
        with pytest.raises(FileNotFoundError, match="Notion cookie"):
            client._get_token()

    def test_missing_token_in_cookie_db_raises(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cookie DB without token_v2 row raises ValueError."""
        from kb.sync.notion import NotionClient

        monkeypatch.delenv("NOTION_TOKEN_V2", raising=False)
        # Create empty cookies DB with correct schema
        db_path = tmp_dir / "Cookies"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE cookies (name TEXT, host_key TEXT, encrypted_value BLOB, value TEXT)"
        )
        conn.commit()
        conn.close()

        client = NotionClient(cookie_path=db_path)
        with pytest.raises(ValueError, match="token_v2 not found"):
            client._get_token()

    def test_decrypt_cookie_v10(self) -> None:
        """Decrypt a v10-prefixed AES-128-CBC cookie value."""
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        from kb.sync.notion import _decrypt_electron_cookie

        # Create a known encrypted value using the same algorithm
        password = b"test-safe-storage-key"
        key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)
        iv = b" " * 16
        plaintext = b"v03:eyJhbGciOiJ0ZXN0In0"

        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        encrypted = b"v10" + ciphertext

        result = _decrypt_electron_cookie(encrypted, password)
        assert result == "v03:eyJhbGciOiJ0ZXN0In0"

    def test_decrypt_cookie_bad_prefix(self) -> None:
        """Non-v10 prefix raises ValueError."""
        from kb.sync.notion import _decrypt_electron_cookie

        with pytest.raises(ValueError, match="Unsupported cookie encryption"):
            _decrypt_electron_cookie(b"v11baddata", b"password")

    def test_decrypt_cookie_no_token_in_decrypted(self) -> None:
        """Decrypted value without v0 prefix raises ValueError."""
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        from kb.sync.notion import _decrypt_electron_cookie

        password = b"test-key"
        key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)
        iv = b" " * 16
        # Plaintext that does NOT contain "v0"
        plaintext = b"no-token-here-at-all"

        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        encrypted = b"v10" + ciphertext

        with pytest.raises(ValueError, match="Could not find token"):
            _decrypt_electron_cookie(encrypted, password)

    def test_token_caching(self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token is cached after first retrieval."""
        from kb.sync.notion import NotionClient

        monkeypatch.setenv("NOTION_TOKEN_V2", "cached-token")
        client = NotionClient(cookie_path=tmp_dir / "nonexistent")

        # First call reads from env
        assert client._get_token() == "cached-token"

        # Remove env var — should still return cached value
        monkeypatch.delenv("NOTION_TOKEN_V2")
        assert client._get_token() == "cached-token"

    def test_cookie_db_decrypt_with_keychain(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full flow: cookie DB + keychain password = decrypted token."""
        monkeypatch.delenv("NOTION_TOKEN_V2", raising=False)

        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        from kb.sync.notion import NotionClient

        # Build an encrypted cookie value
        password = b"mock-keychain-password"
        key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)
        iv = b" " * 16
        plaintext = b"v02:some-notion-token-value"

        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        encrypted = b"v10" + ciphertext

        # Create cookie DB with the encrypted value
        db_path = tmp_dir / "Cookies"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE cookies (name TEXT, host_key TEXT, encrypted_value BLOB, value TEXT)"
        )
        conn.execute(
            "INSERT INTO cookies (name, host_key, encrypted_value, value) VALUES (?, ?, ?, ?)",
            ("token_v2", ".www.notion.so", encrypted, ""),
        )
        conn.commit()
        conn.close()

        # Mock the keychain call to return our test password
        with patch("kb.sync.notion._get_keychain_password", return_value=password):
            client = NotionClient(cookie_path=db_path)
            token = client._get_token()

        assert token == "v02:some-notion-token-value"


class TestAPIClient:
    """Phase 2: API request methods."""

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> NotionClient:
        """Create a NotionClient with mock token."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import NotionClient

        return NotionClient()

    def test_search_pages(self, client: NotionClient) -> None:
        """Search returns page results with pagination."""
        mock_response = {
            "results": [{"id": "page-1"}, {"id": "page-2"}],
            "total": 2,
            "recordMap": {"block": {}},
        }
        with patch.object(client, "_request", return_value=mock_response) as mock_req:
            results, _record_map = client.search_pages(space_id="space-1", limit=10)
            assert len(results) == 2
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0][0] == "/search"

    def test_load_page_chunk(self, client: NotionClient) -> None:
        """loadPageChunk returns block record map."""
        mock_response = {
            "recordMap": {
                "block": {
                    "page-1": {"value": {"type": "page", "content": ["trans-1"]}},
                    "trans-1": {"value": {"type": "transcription", "format": {}}},
                }
            }
        }
        with patch.object(client, "_request", return_value=mock_response):
            blocks = client.load_page_chunk("page-1", limit=5)
            assert "page-1" in blocks
            assert blocks["trans-1"]["value"]["type"] == "transcription"

    def test_sync_record_values(self, client: NotionClient) -> None:
        """syncRecordValues batch-fetches records."""
        mock_response = {
            "recordMap": {
                "notion_user": {
                    "user-1": {"value": {"name": "Wren", "email": "alice@example.com"}}
                }
            }
        }
        with patch.object(client, "_request", return_value=mock_response):
            users = client.get_users(["user-1"])
            assert users["user-1"]["name"] == "Wren"

    def test_get_space_id(self, client: NotionClient) -> None:
        """get_space_id extracts workspace ID from /getSpaces response."""
        mock_response = {"user-1": {"space_view": {"sv-1": {"value": {"space_id": "space-abc"}}}}}
        with patch.object(client, "_request", return_value=mock_response):
            assert client.get_space_id() == "space-abc"

    def test_get_space_id_no_workspace_raises(self, client: NotionClient) -> None:
        """get_space_id raises ValueError when no workspace found."""
        with (
            patch.object(client, "_request", return_value={}),
            pytest.raises(ValueError, match="No workspace found"),
        ):
            client.get_space_id()

    def test_get_current_user_id(self, client: NotionClient) -> None:
        """get_current_user_id extracts first user key from /getSpaces."""
        with patch.object(client, "_request", return_value={"user-42": {}}):
            assert client.get_current_user_id() == "user-42"


class TestContentExtraction:
    """Phase 3: Extract transcript, notes, and summary from blocks."""

    def test_transcript_to_markdown_with_speakers(self) -> None:
        """Transcript segments produce speaker-attributed markdown."""
        from kb.sync.notion import transcript_to_markdown

        blocks = {
            "seg-1": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["Hello, how are you?"]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker0"}},
                    "parent_id": "transcript-block",
                }
            },
            "seg-2": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["I'm good, thanks."]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker1"}},
                    "parent_id": "transcript-block",
                }
            },
        }
        child_order = ["seg-1", "seg-2"]

        result = transcript_to_markdown(blocks, child_order)
        assert "**Speaker 1**: Hello, how are you?" in result
        assert "**Speaker 2**: I'm good, thanks." in result

    def test_transcript_merges_consecutive_same_speaker(self) -> None:
        """Consecutive segments from the same speaker are merged."""
        from kb.sync.notion import transcript_to_markdown

        blocks = {
            "seg-1": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["First part."]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker0"}},
                    "parent_id": "t",
                }
            },
            "seg-2": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["Second part."]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker0"}},
                    "parent_id": "t",
                }
            },
        }
        result = transcript_to_markdown(blocks, ["seg-1", "seg-2"])
        assert result.count("**Speaker 1**") == 1
        assert "First part. Second part." in result

    def test_notes_blocks_to_markdown(self) -> None:
        """Notes child blocks produce markdown text."""
        from kb.sync.notion import notes_to_markdown

        blocks = {
            "n1": {
                "value": {
                    "type": "sub_sub_header",
                    "properties": {"title": [["Action Items"]]},
                    "parent_id": "notes-block",
                }
            },
            "n2": {
                "value": {
                    "type": "to_do",
                    "properties": {"title": [["Follow up with team"]], "checked": [["Yes"]]},
                    "parent_id": "notes-block",
                }
            },
        }
        result = notes_to_markdown(blocks, ["n1", "n2"])
        assert "### Action Items" in result
        assert "- [x] Follow up with team" in result

    def test_extract_text_from_title_property(self) -> None:
        """Notion's title property array is flattened to text."""
        from kb.sync.notion import _extract_text

        # Simple text
        assert _extract_text([["Hello world"]]) == "Hello world"
        # Multi-part with formatting markers
        assert _extract_text([["Hello "], ["world"]]) == "Hello world"
        # Empty
        assert _extract_text([]) == ""


class TestFrontmatterAndWrite:
    """Phase 4: Frontmatter building and file writing."""

    def test_build_frontmatter(self) -> None:
        """Frontmatter includes all required fields."""
        from kb.sync.notion import build_frontmatter

        fm = build_frontmatter(
            title="Weekly Sync",
            page_id="page-123",
            last_edited_time=1771881551056,
            calendar_event={
                "uid": "abc123@google.com",
                "startTime": "2026-02-23T10:00:00.000+01:00",
                "endTime": "2026-02-23T10:30:00.000+01:00",
            },
            attendees=[
                {"name": "Wren Smith", "email": "alice@example.com"},
                {"name": "Soren Jones", "email": "bob@example.com"},
            ],
        )
        assert fm["title"] == "Weekly Sync"
        assert fm["source"] == "notion-api"
        assert fm["notion_page_id"] == "page-123"
        assert fm["calendar_uid"] == "abc123@google.com"
        assert len(fm["attendees"]) == 2
        assert fm["calendar_event"]["scheduled_start"] == "2026-02-23T10:00:00.000+01:00"
        assert "Wren" in fm["tags"]

    def test_make_filename_with_calendar_uid(self) -> None:
        """Filename uses calendar UID prefix when available."""
        from kb.sync.notion import make_filename

        result = make_filename("Weekly Sync", calendar_uid="abc123def@google.com", source="notion")
        assert result == "abc123de_Weekly_Sync"

    def test_make_filename_without_calendar_uid(self) -> None:
        """Filename falls back to source ID prefix."""
        from kb.sync.notion import make_filename

        result = make_filename("Weekly Sync", source_id="page-123-456-789", source="notion")
        assert result == "page-123_Weekly_Sync"

    def test_write_meeting_creates_files(self, tmp_dir: Path) -> None:
        """write_meeting creates notes and transcript files."""
        from kb.sync.notion import write_meeting

        result = write_meeting(
            frontmatter={"title": "Test", "date": "2026-02-23", "source": "notion-api"},
            notes_md="## Notes\nSome notes",
            transcript_md="**Speaker 1**: Hello",
            base_name="abc12345_Test",
            source="notion",
            project_root=tmp_dir,
        )
        assert result["status"] == "created"
        assert result["notes_path"].exists()
        assert result["transcript_path"].exists()
        assert ".notion.notes.md" in result["notes_path"].name
        assert ".notion.transcript.md" in result["transcript_path"].name


class TestSyncOrchestration:
    """Phase 5: High-level sync pipeline."""

    def test_sync_state_persistence(self, tmp_dir: Path) -> None:
        """Sync state is saved and loaded correctly."""
        from kb.sync.notion import load_sync_state, save_sync_state

        state_path = tmp_dir / ".notion_sync_state.json"
        save_sync_state(state_path, last_sync="2026-02-23T10:00:00Z", docs_synced=5)

        state = load_sync_state(state_path)
        assert state["last_sync"] == "2026-02-23T10:00:00Z"
        assert state["docs_synced"] == 5

    def test_load_sync_state_missing_file(self, tmp_dir: Path) -> None:
        """Missing state file returns empty dict."""
        from kb.sync.notion import load_sync_state

        state = load_sync_state(tmp_dir / "missing.json")
        assert state == {}

    def test_sync_notion_dry_run(self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry run reports counts without writing files."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import sync_notion

        with patch("kb.sync.notion.NotionClient") as MockClient:
            client = MockClient.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"
            client.search_pages.return_value = (
                [{"id": "page-1"}],
                {
                    "page-1": {
                        "value": {
                            "type": "page",
                            "content": ["trans-1"],
                            "properties": {"title": [["Test Meeting"]]},
                        }
                    }
                },
            )

            result = sync_notion(
                project_root=tmp_dir,
                data_dir=tmp_dir,
                dry_run=True,
            )
            assert result["dry_run"] is True
            assert result["total"] >= 1

    def test_sync_notion_full_pipeline(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full sync writes meeting files and saves state."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import sync_notion

        page_id = "page-abc-123"
        transcript_id = "transcript-1"
        notes_id = "notes-1"

        # Block map returned by load_page_chunk for the page
        page_blocks = {
            page_id: {
                "value": {
                    "type": "page",
                    "content": ["trans-block"],
                    "properties": {"title": [["Weekly Sync"]]},
                }
            },
            "trans-block": {
                "value": {
                    "type": "transcription",
                    "last_edited_time": 1708700400000,
                    "format": {
                        "transcription_state": {"state": "idle"},
                        "transcription_calendar_event": {
                            "uid": "cal-uid-1234@google.com",
                            "startTime": "2026-02-23T10:00:00.000+01:00",
                            "endTime": "2026-02-23T10:30:00.000+01:00",
                            "attendeeIds": ["user-a", "user-b"],
                        },
                        "transcription_transcript_id": transcript_id,
                        "transcription_notes_id": notes_id,
                    },
                }
            },
        }

        # Transcript blocks
        transcript_blocks = {
            transcript_id: {
                "value": {
                    "type": "text",
                    "content": ["seg-1", "seg-2"],
                }
            },
            "seg-1": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["Hello everyone."]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker0"}},
                }
            },
            "seg-2": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["Hi, let's begin."]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker1"}},
                }
            },
        }

        # Notes blocks
        notes_blocks = {
            notes_id: {
                "value": {
                    "type": "text",
                    "content": ["note-1"],
                }
            },
            "note-1": {
                "value": {
                    "type": "sub_sub_header",
                    "properties": {"title": [["Action Items"]]},
                }
            },
        }

        with patch("kb.sync.notion.NotionClient") as MockClient:
            client = MockClient.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"
            client.search_pages.return_value = ([{"id": page_id}], {})

            def _load_page_chunk(pid: str, limit: int = 5) -> dict:
                if pid == page_id:
                    return page_blocks
                return {}

            def _load_block_children(bid: str, limit: int = 200) -> dict:
                if bid == transcript_id:
                    return transcript_blocks
                if bid == notes_id:
                    return notes_blocks
                return {}

            client.load_page_chunk.side_effect = _load_page_chunk
            client.load_block_children.side_effect = _load_block_children
            client.get_users.return_value = {
                "user-a": {"name": "Wren Smith", "email": "alice@example.com"},
                "user-b": {"name": "Soren Jones", "email": "bob@example.com"},
            }

            result = sync_notion(
                project_root=tmp_dir,
                data_dir=tmp_dir,
            )

        assert result["total"] == 1
        assert result["created"] == 1
        assert result["updated"] == 0
        assert result["skipped"] == 0
        assert "dry_run" not in result

        # Verify state file was written
        state_path = tmp_dir / ".notion_sync_state.json"
        assert state_path.exists()
        import json

        state = json.loads(state_path.read_text())
        assert state["docs_synced"] == 1
        assert "last_sync" in state

        # Verify meeting files were written
        import glob

        notes_files = glob.glob(
            str(tmp_dir / "meetings" / "organised" / "**/*.notes.md"), recursive=True
        )
        transcript_files = glob.glob(
            str(tmp_dir / "meetings" / "organised" / "**/*.transcript.md"), recursive=True
        )
        assert len(notes_files) == 1
        assert len(transcript_files) == 1

        # Verify content
        notes_content = Path(notes_files[0]).read_text()
        assert "Weekly Sync" in notes_content
        assert "### Action Items" in notes_content

        transcript_content = Path(transcript_files[0]).read_text()
        assert "Speaker 1" in transcript_content
        assert "Speaker 2" in transcript_content

    def test_sync_notion_no_transcription_skips_page(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pages without transcription blocks are skipped."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import sync_notion

        page_blocks = {
            "page-1": {
                "value": {
                    "type": "page",
                    "content": [],
                    "properties": {"title": [["No Transcript"]]},
                }
            },
        }

        with patch("kb.sync.notion.NotionClient") as MockClient:
            client = MockClient.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"
            client.search_pages.return_value = ([{"id": "page-1"}], {})
            client.load_page_chunk.return_value = page_blocks

            result = sync_notion(
                project_root=tmp_dir,
                data_dir=tmp_dir,
            )

        assert result["total"] == 0
        assert result["created"] == 0

    def test_sync_notion_uses_saved_state(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sync uses last_sync from saved state when no --since provided."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import save_sync_state, sync_notion

        # Pre-seed state
        state_path = tmp_dir / ".notion_sync_state.json"
        save_sync_state(state_path, last_sync="2026-02-20T10:00:00Z")

        with patch("kb.sync.notion.NotionClient") as MockClient:
            client = MockClient.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"
            client.search_pages.return_value = ([], {})

            sync_notion(
                project_root=tmp_dir,
                data_dir=tmp_dir,
            )

            # Verify search_pages was called with the saved since date
            call_kwargs = client.search_pages.call_args[1]
            assert call_kwargs["since"] == "2026-02-20"

    def test_sync_notion_with_on_progress(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """on_progress callback receives log messages."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import sync_notion

        messages: list[str] = []

        with patch("kb.sync.notion.NotionClient") as MockClient:
            client = MockClient.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"
            client.search_pages.return_value = ([], {})

            sync_notion(
                project_root=tmp_dir,
                data_dir=tmp_dir,
                on_progress=messages.append,
            )

        assert any("User: user-1" in m for m in messages)
        assert any("(full sync)" in m for m in messages)

    def test_load_sync_state_corrupted_file(self, tmp_dir: Path) -> None:
        """Corrupted state file returns empty dict."""
        from kb.sync.notion import load_sync_state

        state_path = tmp_dir / ".notion_sync_state.json"
        state_path.write_text("not valid json {{{", encoding="utf-8")
        assert load_sync_state(state_path) == {}

    def test_sync_notion_update_and_skip(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second sync with force=True updates; without force skips (same timestamp)."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import sync_notion

        page_id = "page-update-1"
        transcript_id = "transcript-u1"

        page_blocks = {
            page_id: {
                "value": {
                    "type": "page",
                    "content": ["trans-block"],
                    "properties": {"title": [["Update Test"]]},
                }
            },
            "trans-block": {
                "value": {
                    "type": "transcription",
                    "last_edited_time": 1708700400000,
                    "format": {
                        "transcription_state": {"state": "idle"},
                        "transcription_calendar_event": {},
                        "transcription_transcript_id": transcript_id,
                    },
                }
            },
        }

        transcript_blocks = {
            transcript_id: {"value": {"type": "text", "content": ["seg-1"]}},
            "seg-1": {
                "value": {
                    "type": "text",
                    "properties": {"title": [["Hello."]]},
                    "format": {"transcript_metadata": {"speaker_name": "speaker0"}},
                }
            },
        }

        def _setup_mock(mock_cls: Any) -> None:
            client = mock_cls.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"
            client.search_pages.return_value = ([{"id": page_id}], {})
            client.load_page_chunk.return_value = page_blocks
            client.load_block_children.return_value = transcript_blocks
            client.get_users.return_value = {}

        # First run — creates
        with patch("kb.sync.notion.NotionClient") as MockClient:
            _setup_mock(MockClient)
            r1 = sync_notion(project_root=tmp_dir, data_dir=tmp_dir)
        assert r1["created"] == 1

        # Second run with force — updates
        with patch("kb.sync.notion.NotionClient") as MockClient:
            _setup_mock(MockClient)
            r2 = sync_notion(project_root=tmp_dir, data_dir=tmp_dir, force=True)
        assert r2["updated"] == 1

        # Third run without force — skips (same timestamp)
        with patch("kb.sync.notion.NotionClient") as MockClient:
            _setup_mock(MockClient)
            r3 = sync_notion(project_root=tmp_dir, data_dir=tmp_dir)
        assert r3["skipped"] == 1


class TestEndToEnd:
    """Phase 7: Full pipeline integration test."""

    def test_full_sync_writes_files_and_state(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full sync pipeline writes notes and transcript files with correct content."""
        monkeypatch.setenv("NOTION_TOKEN_V2", "test-token")
        from kb.sync.notion import sync_notion

        # Mock the NotionClient entirely
        with patch("kb.sync.notion.NotionClient") as MockClient:
            client = MockClient.return_value
            client.get_current_user_id.return_value = "user-1"
            client.get_space_id.return_value = "space-1"

            # search_pages returns one page
            client.search_pages.return_value = (
                [{"id": "page-abc"}],
                {},
            )

            # load_page_chunk returns page with transcription block
            client.load_page_chunk.return_value = {
                "page-abc": {
                    "value": {
                        "type": "page",
                        "content": ["trans-1"],
                        "properties": {"title": [["Weekly Team Sync"]]},
                    }
                },
                "trans-1": {
                    "value": {
                        "type": "transcription",
                        "format": {
                            "transcription_state": {"state": "idle"},
                            "transcription_transcript_id": "t-block",
                            "transcription_summary_id": "s-block",
                            "transcription_notes_id": "n-block",
                            "transcription_calendar_event": {
                                "uid": "cal12345@google.com",
                                "startTime": "2026-02-23T10:00:00+01:00",
                                "endTime": "2026-02-23T10:30:00+01:00",
                                "attendeeIds": ["user-a", "user-b"],
                            },
                        },
                        "last_edited_time": 1771881551056,
                    }
                },
            }

            # get_users resolves attendees
            client.get_users.return_value = {
                "user-a": {"name": "Wren Smith", "email": "alice@example.com"},
                "user-b": {"name": "Soren Jones", "email": "bob@example.com"},
            }

            # load_block_children for transcript and notes
            def mock_load_children(block_id: str, limit: int = 200) -> dict[str, Any]:
                if block_id == "t-block":
                    return {
                        "t-block": {"value": {"type": "text", "content": ["seg-1", "seg-2"]}},
                        "seg-1": {
                            "value": {
                                "type": "text",
                                "properties": {"title": [["Hello everyone"]]},
                                "format": {"transcript_metadata": {"speaker_name": "speaker0"}},
                            }
                        },
                        "seg-2": {
                            "value": {
                                "type": "text",
                                "properties": {"title": [["Thanks for joining"]]},
                                "format": {"transcript_metadata": {"speaker_name": "speaker1"}},
                            }
                        },
                    }
                elif block_id == "n-block":
                    return {
                        "n-block": {"value": {"type": "text", "content": ["n1"]}},
                        "n1": {
                            "value": {
                                "type": "bulleted_list",
                                "properties": {"title": [["Review action items"]]},
                            }
                        },
                    }
                return {}

            client.load_block_children.side_effect = mock_load_children

            # Disable time.sleep for speed
            with patch("kb.sync.notion.time.sleep"):
                result = sync_notion(
                    project_root=tmp_dir,
                    data_dir=tmp_dir,
                )

        # Verify result counts
        assert result["created"] == 1
        assert result["total"] == 1
        assert result["skipped"] == 0

        # Verify files exist (date from last_edited_time epoch → 2026-02-23 UTC)
        md_files = list(tmp_dir.rglob("*.notion.notes.md"))
        assert len(md_files) == 1, f"Expected 1 notes file, found: {md_files}"

        transcript_files = list(tmp_dir.rglob("*.notion.transcript.md"))
        assert len(transcript_files) == 1, f"Expected 1 transcript file, found: {transcript_files}"

        # Verify notes content
        notes_content = md_files[0].read_text()
        assert "title: Weekly Team Sync" in notes_content
        assert "source: notion-api" in notes_content
        assert "calendar_uid: cal12345@google.com" in notes_content
        assert "alice@example.com" in notes_content

        # Verify transcript content
        transcript_content = transcript_files[0].read_text()
        assert "**Speaker 1**: Hello everyone" in transcript_content
        assert "**Speaker 2**: Thanks for joining" in transcript_content
        assert "type: transcript" in transcript_content

        # Verify sync state was saved
        state_file = tmp_dir / ".notion_sync_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["docs_synced"] == 1
        assert "last_sync" in state

        # Verify filename convention: starts with calendar UID prefix
        notes_name = md_files[0].name
        assert notes_name.startswith("cal12345")
        assert ".notion.notes.md" in notes_name


class TestCLI:
    """Phase 6: CLI integration."""

    def test_sync_notion_help(self) -> None:
        """kb sync notion --help works."""
        from click.testing import CliRunner

        from kb.cli import cli

        result = CliRunner().invoke(cli, ["sync", "notion", "--help"])
        assert result.exit_code == 0
        assert "--since" in result.output
        assert "--dry-run" in result.output

    def test_sync_all_help(self) -> None:
        """kb sync --help shows all subcommands."""
        from click.testing import CliRunner

        from kb.cli import cli

        result = CliRunner().invoke(cli, ["sync", "--help"])
        assert result.exit_code == 0
        assert "granola" in result.output
        assert "notion" in result.output
