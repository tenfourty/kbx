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
from typing import Any, cast

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

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def search_pages(
        self,
        space_id: str,
        since: str | None = None,
        created_by: str | None = None,
        limit: int = 100,
        pagination_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search for pages in a workspace.

        Returns (results_list, block_record_map).
        """
        filters: dict[str, Any] = {
            "isDeletedOnly": False,
            "excludeTemplates": True,
            "navigableBlockContentOnly": True,
            "requireEditPermissions": False,
            "ancestors": [],
            "createdBy": [created_by] if created_by else [],
            "editedBy": [],
            "lastEditedTime": {},
            "createdTime": {},
        }
        if since:
            filters["createdTime"] = {
                "type": "after",
                "value": {"type": "exact", "value": since},
            }

        payload: dict[str, Any] = {
            "type": "BlocksInSpace",
            "query": "",
            "spaceId": space_id,
            "limit": limit,
            "filters": filters,
            "sort": {"field": "created", "direction": "desc"},
            "source": "quick_find_input_change",
        }
        if pagination_token:
            payload["paginationToken"] = pagination_token

        data = self._request("/search", json_data=payload)
        results = data.get("results", [])
        record_map = data.get("recordMap", {}).get("block", {})
        return results, record_map

    def load_page_chunk(self, page_id: str, limit: int = 5) -> dict[str, Any]:
        """Load a page's blocks. Returns block record map."""
        data = self._request(
            "/loadPageChunk",
            json_data={
                "pageId": page_id,
                "limit": limit,
                "cursor": {"stack": []},
                "chunkNumber": 0,
                "verticalColumns": False,
            },
        )
        return cast("dict[str, Any]", data.get("recordMap", {}).get("block", {}))

    def load_block_children(self, block_id: str, limit: int = 200) -> dict[str, Any]:
        """Load a block's children (e.g. transcript segments).

        Returns block record map.
        """
        return self.load_page_chunk(block_id, limit=limit)

    def get_users(self, user_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-resolve Notion user IDs to name/email dicts."""
        record_requests = [
            {"pointer": {"table": "notion_user", "id": uid}, "version": -1} for uid in user_ids
        ]
        data = self._request("/syncRecordValues", json_data={"requests": record_requests})
        users = data.get("recordMap", {}).get("notion_user", {})
        return {uid: record["value"] for uid, record in users.items() if "value" in record}

    def _get_spaces_data(self) -> dict[str, Any]:
        """Fetch /getSpaces response, cached for the lifetime of this client."""
        if not hasattr(self, "_spaces_cache"):
            self._spaces_cache: dict[str, Any] = self._request("/getSpaces", json_data={})
        return self._spaces_cache

    def get_space_id(self) -> str:
        """Get the current user's primary space ID."""
        data = self._get_spaces_data()
        # Response is keyed by user ID, each containing space data
        for user_data in data.values():
            space_views = user_data.get("space_view", {})
            for sv in space_views.values():
                space_id = sv.get("value", {}).get("space_id")
                if space_id:
                    return cast("str", space_id)
        raise ValueError("No workspace found for current user")

    def get_current_user_id(self) -> str:
        """Get the current authenticated user's ID."""
        data = self._get_spaces_data()
        user_ids = list(data.keys())
        if not user_ids:
            raise ValueError("No user found in API response")
        return user_ids[0]
