"""Notion internal API sync — client, transform, and write.

Pulls meeting transcripts from Notion's AI Meeting Notes via the
internal /api/v3/ API, produces enriched markdown files, and
optionally triggers indexing.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTION_API_BASE = "https://www.notion.so/api/v3"

# Default cookie path on macOS (Notion Electron app)
DEFAULT_COOKIE_PATH = (
    Path.home() / "Library" / "Application Support" / "Notion" / "Partitions" / "notion" / "Cookies"
)

KEYCHAIN_SERVICE = "Notion Safe Storage"

# Rate limiting
BATCH_SIZE = 15
BATCH_DELAY = 0.5


# ---------------------------------------------------------------------------
# Cookie decryption
# ---------------------------------------------------------------------------


def _get_keychain_password(service: str = KEYCHAIN_SERVICE) -> bytes:
    """Read the Electron safe storage password from macOS Keychain."""
    result = subprocess.run(  # nosec B607 — macOS system utility, not user input
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"Keychain entry '{service}' not found. Is the Notion desktop app installed?"
        )
    return result.stdout.strip().encode("utf-8")


def _decrypt_electron_cookie(encrypted_value: bytes, password: bytes) -> str:
    """Decrypt an Electron v10-prefixed AES-128-CBC cookie.

    Electron on macOS uses:
    - PBKDF2(SHA1, password, 'saltysalt', 1003 iters, 16 byte key)
    - AES-128-CBC with IV of 16 spaces
    - PKCS7 padding
    - 'v10' prefix on the ciphertext
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if not encrypted_value.startswith(b"v10"):
        raise ValueError(f"Unsupported cookie encryption version: {encrypted_value[:3]!r}")

    key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)
    iv = b" " * 16
    ciphertext = encrypted_value[3:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(ciphertext) + decryptor.finalize()

    # Strip PKCS7 padding
    pad_len = decrypted[-1]
    if 1 <= pad_len <= 16 and all(b == pad_len for b in decrypted[-pad_len:]):
        decrypted = decrypted[:-pad_len]

    # The raw bytes may have a non-UTF8 prefix before the actual token.
    # Find the token start (URL-encoded "v0" prefix).
    text = decrypted.decode("latin-1")
    idx = text.find("v0")
    if idx < 0:
        raise ValueError("Could not find token in decrypted cookie value")

    return urllib.parse.unquote(text[idx:])


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class NotionClient:
    """Client for Notion's internal API."""

    def __init__(self, cookie_path: Path | None = None) -> None:
        self._cookie_path = cookie_path or DEFAULT_COOKIE_PATH
        self._token: str | None = None

    def _get_token(self) -> str:
        """Get the token_v2, preferring env var over cookie store."""
        if self._token:
            return self._token

        # 1. Try env var
        env_token = os.environ.get("NOTION_TOKEN_V2")
        if env_token:
            self._token = env_token
            return self._token

        # 2. Decrypt from cookie store
        if not self._cookie_path.exists():
            raise FileNotFoundError(
                f"Notion cookie store not found at {self._cookie_path}. "
                "Is the Notion desktop app installed?"
            )

        conn = sqlite3.connect(str(self._cookie_path))
        try:
            row = conn.execute(
                "SELECT encrypted_value FROM cookies "
                "WHERE name='token_v2' AND host_key='.www.notion.so'"
            ).fetchone()
        finally:
            conn.close()

        if not row:
            raise ValueError("token_v2 not found in Notion cookie store")

        password = _get_keychain_password()
        self._token = _decrypt_electron_cookie(row[0], password)
        return self._token

    def _request(
        self,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> Any:
        """Make an authenticated API request to Notion's internal API."""
        import httpx

        token = self._get_token()
        response = httpx.request(
            "POST",
            f"{NOTION_API_BASE}{path}",
            json=json_data or {},
            headers={"Content-Type": "application/json"},
            cookies={"token_v2": token},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
