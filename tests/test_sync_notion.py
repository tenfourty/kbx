"""Tests for Notion sync — token reading and API client."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


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
