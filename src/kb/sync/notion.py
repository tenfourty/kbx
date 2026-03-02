"""Notion internal API sync — client, transform, and write.

Pulls meeting transcripts from Notion's AI Meeting Notes via the
internal /api/v3/ API, produces enriched markdown files, and
optionally triggers indexing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone
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
BATCH_SIZE = 10
BATCH_DELAY = 0.3
# Minimum interval between API requests (global rate limit)
MIN_REQUEST_INTERVAL = 0.3


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
        self._last_request_time: float = 0.0

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
        *,
        max_retries: int = 5,
    ) -> Any:
        """Make an authenticated API request to Notion's internal API.

        Retries on 429 (rate limit) with exponential backoff.
        """
        import httpx

        token = self._get_token()
        for attempt in range(max_retries + 1):
            # Global rate limiter: space requests MIN_REQUEST_INTERVAL apart
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.monotonic()

            response = httpx.request(
                "POST",
                f"{NOTION_API_BASE}{path}",
                json=json_data or {},
                headers={"Content-Type": "application/json"},
                cookies={"token_v2": token},
                timeout=30.0,
            )
            if response.status_code == 429 and attempt < max_retries:
                wait = 10 * 2**attempt  # 10, 20, 40, 80, 160 seconds
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        # unreachable, but keeps mypy happy
        raise RuntimeError("Exhausted retries")

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
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Search for pages in a workspace.

        Returns (results_list, block_record_map, next_pagination_token).
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
            filters["lastEditedTime"] = {
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
        next_token = data.get("paginationToken")
        # API returns the token as an int timestamp; stringify for consistency
        return results, record_map, str(next_token) if next_token is not None else None

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


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

_SPEAKER_LABEL_RE = re.compile(r"^speaker(\d+)$")


def _extract_text(title_prop: list[Any]) -> str:
    """Extract plain text from Notion's title property array.

    Notion stores text as: [["Hello "], ["world", [["b"]]]]
    We extract just the text parts (index 0 of each sub-array).
    """
    parts: list[str] = []
    for part in title_prop:
        if isinstance(part, list) and part and isinstance(part[0], str):
            parts.append(part[0])
    return "".join(parts)


def transcript_to_markdown(blocks: dict[str, Any], child_order: list[str]) -> str:
    """Convert transcript blocks to speaker-attributed markdown.

    Speaker names like 'speaker0' become 'Speaker 1', 'speaker1' -> 'Speaker 2', etc.
    Consecutive segments from the same speaker are merged.
    """
    merged: list[dict[str, str]] = []
    for child_id in child_order:
        record = blocks.get(child_id, {})
        val = record.get("value", {})
        if val.get("type") != "text":
            continue

        props = val.get("properties", {})
        text = _extract_text(props.get("title", [])).strip()
        if not text:
            continue

        fmt = val.get("format", {})
        raw_speaker = fmt.get("transcript_metadata", {}).get("speaker_name", "unknown")
        m = _SPEAKER_LABEL_RE.match(raw_speaker)
        speaker = f"Speaker {int(m.group(1)) + 1}" if m else raw_speaker

        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": speaker, "text": text})

    return "\n\n".join(f"**{seg['speaker']}**: {seg['text']}" for seg in merged)


def notes_to_markdown(blocks: dict[str, Any], child_order: list[str]) -> str:
    """Convert Notion notes blocks to markdown.

    Handles: sub_sub_header (###), to_do, bulleted_list, text, header_4 (####).
    """
    lines: list[str] = []
    for child_id in child_order:
        record = blocks.get(child_id, {})
        val = record.get("value", {})
        btype = val.get("type", "")
        props = val.get("properties", {})
        text = _extract_text(props.get("title", []))

        if btype == "sub_sub_header":
            lines.append(f"### {text}")
        elif btype == "header_4":
            lines.append(f"#### {text}")
        elif btype == "to_do":
            checked = _extract_text(props.get("checked", [])) == "Yes"
            marker = "[x]" if checked else "[ ]"
            lines.append(f"- {marker} {text}")
        elif btype == "bulleted_list":
            lines.append(f"- {text}")
        elif btype == "numbered_list":
            lines.append(f"1. {text}")
        elif btype == "text" and text:
            lines.append(text)

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def build_frontmatter(
    title: str,
    page_id: str,
    last_edited_time: int,
    calendar_event: dict[str, Any] | None = None,
    attendees: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build YAML frontmatter dict from Notion meeting data."""
    dt = datetime.fromtimestamp(last_edited_time / 1000, tz=timezone.utc)
    date = dt.strftime("%Y-%m-%d")

    tags: list[str] = []
    for a in attendees or []:
        name = a.get("name", "")
        if name:
            parts = name.split()
            tags.append(parts[0] if len(parts) >= 2 else name)

    fm: dict[str, Any] = {
        "title": title,
        "date": date,
        "type": "notes",
        "source": "notion-api",
        "notion_page_id": page_id,
        "notion_updated_at": dt.isoformat(),
        "tags": tags,
        "attendees": attendees or [],
    }

    if calendar_event:
        fm["calendar_uid"] = calendar_event.get("uid", "")
        fm["calendar_event"] = {
            "title": title,
            "scheduled_start": calendar_event.get("startTime", ""),
            "scheduled_end": calendar_event.get("endTime", ""),
        }
    else:
        fm["calendar_event"] = None

    return fm


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------


def _strip_emojis(text: str) -> str:
    """Strip emoji characters, variation selectors, and ZWJ from text."""
    import unicodedata

    return "".join(
        c for c in text if unicodedata.category(c) != "So" and c not in ("\u200d", "\ufe0f")
    )


def _sanitize_title(title: str) -> str:
    """Sanitize a title for use in filenames.

    Strips emojis, replaces separators (/) with ___, other non-alnum with _,
    collapses runs. Shared convention across Granola and Notion sync.
    """
    stripped = _strip_emojis(title)
    sanitized = re.sub(r"\s*/\s*", "___", stripped)
    sanitized = re.sub(r"[^\w\-]", "_", sanitized)
    sanitized = sanitized.replace("___", "\x00")
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.replace("\x00", "___")
    return sanitized.strip("_")


def make_filename(
    title: str,
    calendar_uid: str | None = None,
    source_id: str | None = None,
    source: str = "notion",
) -> str:
    """Create a filename base from title and UID.

    Format: {uid_prefix}_{Sanitized_Title}
    uid_prefix: first 8 chars of calendar_uid, or first 8 of source_id.
    """
    if calendar_uid:
        prefix = calendar_uid[:8]
    elif source_id:
        prefix = source_id[:8]
    else:
        prefix = "unknown"

    sanitized = _sanitize_title(title)
    return f"{prefix}_{sanitized}"


# ---------------------------------------------------------------------------
# Write meeting files
# ---------------------------------------------------------------------------


def _frontmatter_to_yaml(fm: dict[str, Any]) -> str:
    """Serialize frontmatter dict to YAML string."""
    import yaml

    clean = {k: v for k, v in fm.items() if v is not None}
    yaml_str: str = yaml.dump(clean, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return "---\n" + yaml_str + "---"


def write_meeting(
    frontmatter: dict[str, Any],
    notes_md: str,
    transcript_md: str,
    base_name: str,
    source: str,
    project_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Write meeting files to organised directory.

    Returns dict with notes_path, transcript_path, status.
    """
    date = frontmatter.get("date", "")
    parts = date.split("-")
    if len(parts) == 3:
        year, month, day = parts
    else:
        year, month, day = "unknown", "00", "00"

    out_dir = project_root / "memory" / "meetings" / year / month / day

    notes_path = out_dir / f"{base_name}.{source}.notes.md"
    transcript_path = out_dir / f"{base_name}.{source}.transcript.md"

    # Check if we should skip
    if notes_path.exists() and not force:
        existing = notes_path.read_text(encoding="utf-8")
        existing_updated = _extract_notion_updated_at(existing)
        new_updated = frontmatter.get("notion_updated_at", "")
        if existing_updated and new_updated and new_updated <= existing_updated:
            return {
                "notes_path": notes_path,
                "transcript_path": transcript_path,
                "status": "skipped",
            }

    status = "updated" if notes_path.exists() else "created"

    # Build content
    notes_fm = _frontmatter_to_yaml(frontmatter)
    notes_content = f"{notes_fm}\n\n{notes_md}"

    transcript_fm_dict = {**frontmatter, "type": "transcript"}
    transcript_fm = _frontmatter_to_yaml(transcript_fm_dict)
    transcript_content = f"{transcript_fm}\n\n{transcript_md}"

    # Write files
    _write_file(notes_path, notes_content)
    _write_file(transcript_path, transcript_content)

    return {"notes_path": notes_path, "transcript_path": transcript_path, "status": status}


def _extract_notion_updated_at(content: str) -> str | None:
    """Extract notion_updated_at from frontmatter."""
    m = re.search(r"notion_updated_at:\s*['\"]?([^'\"\n]+)", content)
    return m.group(1).strip() if m else None


def _write_file(path: Path, content: str) -> None:
    """Write content to file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Sync state persistence
# ---------------------------------------------------------------------------


def save_sync_state(path: Path, **kwargs: Any) -> None:
    """Save sync state to a JSON file."""
    state: dict[str, Any] = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    state.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_sync_state(path: Path) -> dict[str, Any]:
    """Load sync state from a JSON file."""
    if not path.exists():
        return {}
    try:
        return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# High-level sync orchestration
# ---------------------------------------------------------------------------


def sync_notion(
    project_root: Path,
    data_dir: Path,
    since: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the full Notion sync pipeline.

    Returns dict with keys: total, created, updated, skipped, dry_run (if applicable).
    """
    import sys

    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        else:
            print(msg, file=sys.stderr)

    client = NotionClient()

    # Get workspace and user info
    user_id = client.get_current_user_id()
    space_id = client.get_space_id()
    _log(f"User: {user_id}, Space: {space_id}")

    # Load sync state
    state_path = data_dir / ".notion_sync_state.json"
    state = load_sync_state(state_path)

    effective_since = since
    if not effective_since and state.get("last_sync"):
        effective_since = state["last_sync"][:10]

    _log(f"Searching pages{' since ' + effective_since if effective_since else ' (full sync)'}...")

    # Discover all pages — collect recordMaps from search to avoid extra API calls
    all_pages: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    cached_blocks: dict[str, Any] = {}
    pagination_token: str | None = None
    while True:
        results, record_map, next_token = client.search_pages(
            space_id=space_id,
            since=effective_since,
            created_by=user_id,
            pagination_token=pagination_token,
        )
        for page in results:
            pid = page["id"]
            if pid not in seen_page_ids:
                seen_page_ids.add(pid)
                all_pages.append(page)
        cached_blocks.update(record_map)

        # Pagination ends when: no token returned, no results, or token repeats
        if not next_token or not results or next_token == pagination_token:
            break
        pagination_token = next_token
        time.sleep(BATCH_DELAY)

    # Client-side date filter — Notion's internal search API ignores date filters,
    # so we filter locally using last_edited_time from the recordMap.
    if effective_since:
        since_ts = int(
            datetime.strptime(effective_since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            * 1000
        )
        before_count = len(all_pages)
        all_pages = [
            p
            for p in all_pages
            if cached_blocks.get(p["id"], {}).get("value", {}).get("last_edited_time", since_ts)
            >= since_ts
        ]
        if len(all_pages) < before_count:
            _log(
                f"Filtered to {len(all_pages)} pages edited since {effective_since} (was {before_count})."
            )

    _log(f"Found {len(all_pages)} pages. Checking for meeting transcripts...")

    if dry_run:
        _log("Dry run — not writing files.")
        return {
            "total": len(all_pages),
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "dry_run": True,
        }

    # Process each page — find transcription blocks.
    # Use cached recordMap from search when children are available; else API.
    all_attendee_ids: set[str] = set()
    meetings: list[dict[str, Any]] = []
    total_pages = len(all_pages)
    api_calls = 0
    for page_idx, page_result in enumerate(all_pages):
        page_id = page_result["id"]

        # Try cached blocks if page AND its children are in the cache
        blocks: dict[str, Any] | None = None
        if page_id in cached_blocks:
            page_val = cached_blocks[page_id].get("value", {})
            content_ids = page_val.get("content", [])
            if content_ids and all(cid in cached_blocks for cid in content_ids):
                blocks = {page_id: cached_blocks[page_id]}
                for cid in content_ids:
                    blocks[cid] = cached_blocks[cid]

        if blocks is None:
            api_calls += 1
            if api_calls == 1 or api_calls % 20 == 0 or page_idx == total_pages - 1:
                _log(f"  Scanning page {page_idx + 1}/{total_pages}...")
            blocks = client.load_page_chunk(page_id, limit=5)

        # Find transcription block
        transcription = None
        for block_data in blocks.values():
            val = block_data.get("value", {})
            if val.get("type") == "transcription":
                state_info = val.get("format", {}).get("transcription_state", {})
                if state_info.get("state") == "idle":
                    transcription = val
                break

        if not transcription:
            continue

        fmt = transcription.get("format", {})
        cal = fmt.get("transcription_calendar_event", {})
        attendee_ids = cal.get("attendeeIds", [])
        all_attendee_ids.update(attendee_ids)

        # Get page title
        page_block = blocks.get(page_id, {}).get("value", {})
        title_raw = page_block.get("properties", {}).get("title", [])
        title = _extract_text(title_raw).replace("\u2023", "").strip()

        meetings.append(
            {
                "page_id": page_id,
                "title": title or "Untitled",
                "transcription": transcription,
                "last_edited_time": transcription.get("last_edited_time", 0),
            }
        )

    _log(f"Found {len(meetings)} meetings with transcripts.")

    # Batch-resolve all attendee IDs
    user_map: dict[str, dict[str, str]] = {}
    if all_attendee_ids:
        raw_users = client.get_users(list(all_attendee_ids))
        for uid, udata in raw_users.items():
            user_map[uid] = {
                "name": udata.get("name", ""),
                "email": udata.get("email", ""),
            }

    created = 0
    updated = 0
    skipped = 0

    for i, meeting in enumerate(meetings):
        title = meeting["title"]

        fmt = meeting["transcription"].get("format", {})
        cal = fmt.get("transcription_calendar_event", {})
        attendee_ids = cal.get("attendeeIds", [])

        attendees = [user_map.get(uid, {"name": "Unknown", "email": ""}) for uid in attendee_ids]

        fm = build_frontmatter(
            title=title,
            page_id=meeting["page_id"],
            last_edited_time=meeting["last_edited_time"],
            calendar_event=cal if cal.get("uid") else None,
            attendees=attendees,
        )

        base_name = make_filename(
            title=title,
            calendar_uid=cal.get("uid"),
            source_id=meeting["page_id"],
        )

        # Early skip: check if file exists and is up-to-date before fetching content
        if not force:
            date = fm.get("date", "")
            parts = date.split("-")
            if len(parts) == 3:
                year, month, day = parts
            else:
                year, month, day = "unknown", "00", "00"
            notes_path = project_root / "memory" / "meetings" / year / month / day
            notes_path = notes_path / f"{base_name}.notion.notes.md"
            if notes_path.exists():
                existing = notes_path.read_text(encoding="utf-8")
                existing_updated = _extract_notion_updated_at(existing)
                new_updated = fm.get("notion_updated_at", "")
                if existing_updated and new_updated and new_updated <= existing_updated:
                    skipped += 1
                    continue

        _log(f"[{i + 1}/{len(meetings)}] {title}")

        # Fetch transcript segments
        transcript_id = fmt.get("transcription_transcript_id", "")
        transcript_md = ""
        if transcript_id:
            transcript_blocks = client.load_block_children(transcript_id, limit=500)
            parent_block = transcript_blocks.get(transcript_id, {}).get("value", {})
            child_order = parent_block.get("content", [])
            transcript_md = transcript_to_markdown(transcript_blocks, child_order)

        # Fetch AI summary (preferred) and manual notes (fallback)
        summary_id = fmt.get("transcription_summary_id", "")
        summary_md = ""
        if summary_id:
            summary_blocks = client.load_block_children(summary_id, limit=200)
            parent_block = summary_blocks.get(summary_id, {}).get("value", {})
            child_order = parent_block.get("content", [])
            summary_md = notes_to_markdown(summary_blocks, child_order)

        notes_id = fmt.get("transcription_notes_id", "")
        manual_notes_md = ""
        if notes_id:
            notes_blocks = client.load_block_children(notes_id, limit=200)
            parent_block = notes_blocks.get(notes_id, {}).get("value", {})
            child_order = parent_block.get("content", [])
            manual_notes_md = notes_to_markdown(notes_blocks, child_order)

        # Use AI summary as primary body; fall back to manual notes if empty
        notes_md = summary_md if summary_md.strip() else manual_notes_md

        result = write_meeting(
            frontmatter=fm,
            notes_md=notes_md,
            transcript_md=transcript_md,
            base_name=base_name,
            source="notion",
            project_root=project_root,
            force=force,
        )

        if result["status"] == "created":
            created += 1
        elif result["status"] == "updated":
            updated += 1
        else:
            skipped += 1

        # Throttle: each meeting needs 1-3 API calls (transcript + summary + notes)
        time.sleep(BATCH_DELAY)

    # Save sync state
    save_sync_state(
        state_path,
        last_sync=datetime.now().isoformat(),
        docs_synced=len(meetings),
        last_run=datetime.now().isoformat(),
    )

    return {
        "total": len(meetings),
        "created": created,
        "updated": updated,
        "skipped": skipped,
    }
