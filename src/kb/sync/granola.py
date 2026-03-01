"""Granola API sync — client, transform, and write.

Pulls meetings directly from Granola's internal API, produces enriched
markdown files, and optionally triggers indexing.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

# httpx is imported lazily inside GranolaClient methods that use it,
# so modules that only import match_or_create_attendee() don't need httpx.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRANOLA_API_BASE = "https://api.granola.ai"
WORKOS_AUTH_URL = "https://api.workos.com/user_management/authenticate"
USER_AGENT = "Granola/5.354.0"

# Default token path on macOS
DEFAULT_TOKEN_PATH = Path.home() / "Library" / "Application Support" / "Granola" / "supabase.json"

# Rate limiting: batch size and delay between requests
BATCH_SIZE = 15
BATCH_DELAY = 0.5  # seconds between batches


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class GranolaClient:
    """Client for Granola's internal API."""

    def __init__(self, token_path: Path | None = None) -> None:
        self._token_path = token_path or DEFAULT_TOKEN_PATH
        self._access_token: str | None = None
        self._refresh_token_value: str | None = None

    def _read_tokens(self) -> dict[str, str]:
        """Read access and refresh tokens from supabase.json.

        Returns dict with 'access_token' and 'refresh_token' keys.
        Raises FileNotFoundError if file doesn't exist.
        Raises ValueError if file is not valid JSON or missing expected keys.
        """
        if not self._token_path.exists():
            raise FileNotFoundError(
                f"Granola supabase.json not found at {self._token_path}. Is Granola installed?"
            )

        text = self._token_path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse supabase.json: {e}") from e

        workos_raw = data.get("workos_tokens", {})
        # workos_tokens may be a JSON string (double-encoded) or a dict
        if isinstance(workos_raw, str):
            try:
                workos = json.loads(workos_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse workos_tokens: {e}") from e
        else:
            workos = workos_raw

        access = workos.get("access_token")
        refresh = workos.get("refresh_token")

        if not access or not refresh:
            raise ValueError("supabase.json missing workos_tokens.access_token or refresh_token")

        self._access_token = access
        self._refresh_token_value = refresh
        return {"access_token": access, "refresh_token": refresh}

    def _refresh_tokens(self) -> None:
        """Refresh tokens using WorkOS API with file locking for atomicity.

        Single-use refresh tokens require atomic read-refresh-write to prevent
        race conditions if multiple processes sync concurrently.
        """
        import httpx

        # Ensure we have current tokens
        if not self._refresh_token_value:
            self._read_tokens()

        # Call WorkOS to get new tokens
        response = httpx.post(
            WORKOS_AUTH_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token_value,
                "client_id": os.environ.get("GRANOLA_WORKOS_CLIENT_ID", ""),
            },
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        new_tokens = response.json()

        new_access = new_tokens["access_token"]
        new_refresh = new_tokens["refresh_token"]

        # Atomic write-back with file locking
        self._write_tokens_atomic(new_access, new_refresh)

        self._access_token = new_access
        self._refresh_token_value = new_refresh

    def _write_tokens_atomic(self, access_token: str, refresh_token: str) -> None:
        """Write tokens back to supabase.json with file locking."""
        # Read current file content
        data = json.loads(self._token_path.read_text(encoding="utf-8"))

        workos_raw = data.get("workos_tokens", {})
        if isinstance(workos_raw, str):
            workos = json.loads(workos_raw)
        else:
            workos = workos_raw

        workos["access_token"] = access_token
        workos["refresh_token"] = refresh_token

        # Preserve original encoding format (string vs dict)
        if isinstance(data.get("workos_tokens"), str):
            data["workos_tokens"] = json.dumps(workos)
        else:
            data["workos_tokens"] = workos

        # Write atomically: write to temp, then rename
        tmp_path = self._token_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        tmp_path.rename(self._token_path)

    def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        retry: bool = True,
    ) -> Any:
        """Make an authenticated API request. Retries once on 401 with token refresh."""
        import httpx

        if not self._access_token:
            self._read_tokens()

        response = httpx.request(
            method,
            f"{GRANOLA_API_BASE}{path}",
            json=json_data,
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        if response.status_code == 401 and retry:
            self._refresh_tokens()
            return self._request(method, path, json_data=json_data, retry=False)

        response.raise_for_status()
        return response

    def list_documents(self, since: str | None = None) -> list[dict[str, Any]]:
        """List all Granola documents, optionally filtered by date.

        Args:
            since: ISO date string (YYYY-MM-DD) to filter documents created/updated after.

        Returns list of document dicts with id, title, created_at, updated_at, etc.
        The API doesn't support server-side date filtering, so we filter client-side.
        """
        all_docs: list[dict[str, Any]] = []
        offset = 0
        limit = 50

        while True:
            payload: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "include_last_viewed_panel": True,
            }

            response = self._request("POST", "/v2/get-documents", json_data=payload)
            data = response.json()
            docs = data.get("docs", [])

            if not docs:
                break

            all_docs.extend(docs)
            offset += limit

            # If we got fewer than limit, we've reached the end
            if len(docs) < limit:
                break

            time.sleep(BATCH_DELAY)

        # Client-side date filtering (API doesn't support it)
        if since:
            since_dt = f"{since}T00:00:00Z"
            all_docs = [
                d for d in all_docs if (d.get("updated_at") or d.get("created_at", "")) >= since_dt
            ]

        return all_docs

    def get_transcript(self, doc_id: str) -> list[dict[str, Any]]:
        """Get transcript segments for a document.

        Returns list of dicts with text, start_timestamp, end_timestamp, source.
        Note: Granola API returns a list directly (not wrapped in a dict).
        There is no speaker attribution — segments have 'source' (microphone/system).
        """
        response = self._request(
            "POST",
            "/v1/get-document-transcript",
            json_data={"document_id": doc_id},
        )
        data = response.json()
        # API returns a list directly
        if isinstance(data, list):
            return data
        # Fallback for wrapped response
        return cast("list[dict[Any, Any]]", data.get("transcript", []))

    def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get document metadata (attendees, calendar event, etc.).

        Note: Most useful data (people, calendar) is already on the document
        from list_documents. This endpoint is for additional metadata.
        """
        response = self._request(
            "POST",
            "/v1/get-document-metadata",
            json_data={"document_id": doc_id},
        )
        return cast("dict[Any, Any]", response.json())


# ---------------------------------------------------------------------------
# ProseMirror → Markdown
# ---------------------------------------------------------------------------


def prosemirror_to_markdown(nodes: list[dict[str, Any]], indent: int = 0) -> str:
    """Convert ProseMirror document nodes to Markdown.

    Handles: headings, paragraphs, bullet lists, ordered lists, list items,
    text with marks (bold, italic, code), blockquotes, code blocks.
    """
    parts: list[str] = []

    for node in nodes:
        node_type = node.get("type", "")

        if node_type == "heading":
            level = node.get("attrs", {}).get("level", 1)
            text = _render_inline(node.get("content", []))
            parts.append(f"{'#' * level} {text}")

        elif node_type == "paragraph":
            text = _render_inline(node.get("content", []))
            parts.append(text)

        elif node_type == "bulletList":
            items = node.get("content", [])
            for item in items:
                item_parts = _render_list_item(item, indent, bullet="- ")
                parts.append(item_parts)

        elif node_type == "orderedList":
            items = node.get("content", [])
            for i, item in enumerate(items, 1):
                item_parts = _render_list_item(item, indent, bullet=f"{i}. ")
                parts.append(item_parts)

        elif node_type == "blockquote":
            inner = prosemirror_to_markdown(node.get("content", []))
            quoted = "\n".join(f"> {line}" for line in inner.splitlines())
            parts.append(quoted)

        elif node_type == "codeBlock":
            lang = node.get("attrs", {}).get("language", "")
            text = _render_inline(node.get("content", []))
            parts.append(f"```{lang}\n{text}\n```")

        elif node_type == "horizontalRule":
            parts.append("---")

        elif node_type == "text":
            parts.append(_render_text_node(node))

    return "\n\n".join(parts)


def _render_list_item(item: dict[str, Any], indent: int, bullet: str) -> str:
    """Render a list item, handling nested content."""
    prefix = "  " * indent
    result_parts: list[str] = []

    for child in item.get("content", []):
        child_type = child.get("type", "")

        if child_type == "paragraph":
            text = _render_inline(child.get("content", []))
            result_parts.append(f"{prefix}{bullet}{text}")

        elif child_type == "bulletList":
            nested = child.get("content", [])
            for nested_item in nested:
                nested_text = _render_list_item(nested_item, indent + 1, bullet="- ")
                result_parts.append(nested_text)

        elif child_type == "orderedList":
            nested = child.get("content", [])
            for i, nested_item in enumerate(nested, 1):
                nested_text = _render_list_item(nested_item, indent + 1, bullet=f"{i}. ")
                result_parts.append(nested_text)

    return "\n".join(result_parts)


def _render_inline(content: list[dict[str, Any]]) -> str:
    """Render inline content nodes (text, marks) to a string."""
    parts = []
    for node in content:
        if node.get("type") == "text":
            parts.append(_render_text_node(node))
        elif node.get("type") == "hardBreak":
            parts.append("\n")
    return "".join(parts)


def _render_text_node(node: dict[str, Any]) -> str:
    """Render a text node with optional marks (bold, italic, code)."""
    text: str = node.get("text", "")
    marks = node.get("marks", [])

    for mark in marks:
        mark_type = mark.get("type", "")
        if mark_type == "bold":
            text = f"**{text}**"
        elif mark_type == "italic":
            text = f"*{text}*"
        elif mark_type == "code":
            text = f"`{text}`"
        elif mark_type == "link":
            href = mark.get("attrs", {}).get("href", "")
            text = f"[{text}]({href})"

    return text


# ---------------------------------------------------------------------------
# Transcript → Markdown
# ---------------------------------------------------------------------------


def transcript_to_markdown(segments: list[dict[str, Any]]) -> str:
    """Convert transcript segments to markdown.

    Handles two formats:
    - With 'speaker' field: speaker-attributed (test mocks, legacy)
    - With 'source' field (Granola API): 'microphone' → Me, 'system' → System

    Merges consecutive segments from the same speaker/source.
    """
    if not segments:
        return ""

    # Map Granola API source values to human-readable speaker labels
    SOURCE_LABELS = {"microphone": "Me", "system": "System"}

    merged: list[dict[str, Any]] = []
    for seg in segments:
        # Prefer explicit 'speaker' field (tests), fall back to 'source' mapping
        speaker = seg.get("speaker") or SOURCE_LABELS.get(
            seg.get("source", ""), seg.get("source", "Unknown")
        )
        text = seg.get("text", "").strip()
        if not text:
            continue

        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": speaker, "text": text})

    lines = []
    for seg in merged:
        lines.append(f"**{seg['speaker']}**: {seg['text']}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def _name_from_email(email: str) -> str:
    """Derive a human-readable name from an email address.

    'alice.reed@example.com' → 'Wren Kasper'
    """
    local = email.split("@")[0]
    parts = re.split(r"[.\-_]", local)
    return " ".join(p.capitalize() for p in parts if p)


def build_frontmatter(
    doc: dict[str, Any],
    metadata: dict[str, Any],
    summary: str | None = None,
) -> dict[str, Any]:
    """Build enriched YAML frontmatter dict from Granola API data.

    Handles two data sources:
    - doc['people']['attendees']: attendee list from the document itself
    - doc['google_calendar_event']: Google Calendar event data
    - metadata: from /v1/get-document-metadata (fallback for attendees)
    - summary: extracted from notes content if present

    Also handles the simpler mock format from tests (metadata with 'people' list).
    """
    created = doc.get("created_at", "")
    date = created[:10] if created else ""

    # Extract attendees — try doc.people.attendees first (real API),
    # then metadata.people (test mocks), then metadata.attendees
    people_data = doc.get("people")
    if isinstance(people_data, dict):
        raw_attendees = people_data.get("attendees", [])
    elif isinstance(metadata.get("people"), list):
        raw_attendees = metadata["people"]
    else:
        raw_attendees = metadata.get("attendees", [])

    # Include creator as first attendee
    creator = None
    if isinstance(people_data, dict) and people_data.get("creator"):
        c = people_data["creator"]
        creator = {"name": c.get("name", ""), "email": c.get("email", "")}

    # Build attendee list, filtering out group emails and calendar rooms
    attendees = []
    seen_emails: set[str] = set()

    # Add creator first if available
    if creator and creator.get("email"):
        seen_emails.add(creator["email"].lower())
        attendees.append(creator)

    for p in raw_attendees or []:
        email = p.get("email", "")

        # Skip calendar resources (room bookings)
        if email and "@resource.calendar.google.com" in email:
            continue

        # Skip duplicates
        if email and email.lower() in seen_emails:
            continue

        # Skip group emails (mailing lists)
        details = p.get("details", {}) or {}
        group = details.get("group")
        if group and group.get("members"):
            # Expand group members into individual attendees
            for member in group["members"]:
                m_name = member.get("name", "")
                m_email = member.get("email", "")
                if m_email and m_email.lower() not in seen_emails:
                    seen_emails.add(m_email.lower())
                    attendees.append(
                        {"name": m_name or _name_from_email(m_email), "email": m_email}
                    )
            continue

        if not email:
            continue

        # Derive name: from details.person, from field, or from email
        name = p.get("name", "")
        if not name:
            person_details = details.get("person", {})
            name_info = person_details.get("name", {})
            name = name_info.get("fullName", "") if isinstance(name_info, dict) else ""
        if not name:
            name = _name_from_email(email)

        seen_emails.add(email.lower())
        attendee: dict[str, str] = {"name": name, "email": email}
        # Extract company from details if available
        company = details.get("company", {})
        if isinstance(company, dict) and company.get("name"):
            attendee["company"] = company["name"]
        # Extract company from employment in person details
        elif details.get("person", {}).get("employment", {}).get("name"):
            attendee["company"] = details["person"]["employment"]["name"]
        attendees.append(attendee)

    # Extract tags from attendee names (use first name if multi-word)
    tags = []
    for a in attendees:
        name = a.get("name", "")
        if name:
            parts = name.split()
            if len(parts) >= 2:
                tags.append(parts[0])
            else:
                tags.append(name)

    fm: dict[str, Any] = {
        "title": doc.get("title") or "Untitled",
        "date": date,
        "type": "notes",
        "source": "granola-api",
        "granola_id": doc.get("id", ""),
        "granola_updated_at": doc.get("updated_at", ""),
        "tags": tags,
        "attendees": attendees,
    }

    # Calendar event — from Google Calendar data on doc
    gcal = doc.get("google_calendar_event")
    if gcal:
        start = gcal.get("start", {})
        end = gcal.get("end", {})
        fm["calendar_event"] = {
            "title": gcal.get("summary", ""),
            "organiser": (gcal.get("organizer") or {}).get("email", ""),
            "scheduled_start": start.get("dateTime", ""),
            "scheduled_end": end.get("dateTime", ""),
        }
        # Extract calendar UID for stable file naming
        cal_uid = gcal.get("iCalUID") or gcal.get("id") or ""
        if cal_uid:
            fm["calendar_uid"] = cal_uid
    elif metadata.get("calendar_event"):
        # Fallback for test mocks
        cal = metadata["calendar_event"]
        fm["calendar_event"] = {
            "title": cal.get("title", ""),
            "organiser": cal.get("organizer_email", ""),
            "scheduled_start": cal.get("start_time", ""),
            "scheduled_end": cal.get("end_time", ""),
        }
    else:
        fm["calendar_event"] = None

    if summary:
        fm["granola_summary"] = summary
    else:
        fm["granola_summary"] = None

    return fm


# ---------------------------------------------------------------------------
# Write meeting files
# ---------------------------------------------------------------------------


def _frontmatter_to_yaml(fm: dict[str, Any]) -> str:
    """Serialize frontmatter dict to YAML string (between --- markers)."""
    import yaml

    # Filter out None values for cleaner output
    clean = {k: v for k, v in fm.items() if v is not None}
    yaml_str: str = yaml.dump(clean, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return "---\n" + yaml_str + "---"


def _find_existing_by_id(out_dir: Path, id_prefix: str) -> str | None:
    """Find an existing file's base name by granola_id prefix in a directory.

    Scans .notes.md files for matching id_prefix in both old and new patterns:
    - Old: {Title}_{id_prefix}.notes.md
    - New: {id_prefix}_{Title}.granola.notes.md

    Returns the base name (without .notes.md or .granola.notes.md) if found,
    else None.
    """
    if not out_dir.exists():
        return None
    # New pattern: {id_prefix}_{Title}.granola.notes.md
    new_prefix = f"{id_prefix}_"
    for f in out_dir.iterdir():
        if f.name.startswith(new_prefix) and f.name.endswith(".granola.notes.md"):
            return f.name[: -len(".granola.notes.md")]
    # Old pattern: {Title}_{id_prefix}.notes.md
    old_suffix = f"_{id_prefix}.notes.md"
    for f in out_dir.iterdir():
        if f.name.endswith(old_suffix):
            return f.name[: -len(".notes.md")]
    return None


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
    # Strip emojis first so "🗓️ Title" and "Title" produce the same result
    stripped = _strip_emojis(title)
    sanitized = re.sub(r"\s*/\s*", "___", stripped)
    sanitized = re.sub(r"[^\w\-]", "_", sanitized)
    # Collapse runs of underscores to single, but preserve ___ (slash separator)
    # Strategy: protect ___ → collapse → restore
    sanitized = sanitized.replace("___", "\x00")
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.replace("\x00", "___")
    return sanitized.strip("_")


def _make_filename(title: str, granola_id: str, calendar_uid: str | None = None) -> str:
    """Create a filename from title and a UID prefix.

    Pattern: {uid_prefix}_{Sanitized_Title}
    uid_prefix is first 8 chars of calendar_uid (preferred) or granola_id.
    """
    sanitized = _sanitize_title(title)
    uid = calendar_uid if calendar_uid else granola_id
    uid_prefix = uid[:8] if len(uid) >= 8 else uid
    return f"{uid_prefix}_{sanitized}"


def write_meeting(
    doc: dict[str, Any],
    frontmatter: dict[str, Any],
    notes_md: str,
    transcript_md: str,
    project_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Write meeting files to the organised directory.

    Returns dict with:
        notes_path: Path to written notes file
        transcript_path: Path to written transcript file
        status: "created", "updated", or "skipped"
    """
    date = frontmatter.get("date", "")
    if not date:
        created = doc.get("created_at", "")
        date = created[:10] if created else "unknown"

    # Build output directory: memory/meetings/YYYY/MM/DD/
    parts = date.split("-")
    if len(parts) == 3:
        year, month, day = parts
    else:
        year, month, day = "unknown", "00", "00"

    out_dir = project_root / "memory" / "meetings" / year / month / day
    granola_id = doc.get("id", "unknown")
    id_prefix = granola_id[:8] if len(granola_id) >= 8 else granola_id

    # Try to find existing files by granola_id prefix in the target directory.
    # This handles naming differences (emojis, special chars) between old imports
    # and new sync-generated filenames.
    existing_base = _find_existing_by_id(out_dir, id_prefix)
    calendar_uid = frontmatter.get("calendar_uid")
    new_base = _make_filename(
        frontmatter.get("title") or "Untitled", granola_id, calendar_uid=calendar_uid
    )

    # Determine if existing file uses old pattern and needs renaming
    if existing_base:
        old_notes = out_dir / f"{existing_base}.notes.md"
        old_transcript = out_dir / f"{existing_base}.transcript.md"
        new_notes_check = out_dir / f"{existing_base}.granola.notes.md"
        if old_notes.exists() and not new_notes_check.exists():
            # Old pattern — rename to new pattern
            target_notes = out_dir / f"{new_base}.granola.notes.md"
            target_transcript = out_dir / f"{new_base}.granola.transcript.md"
            old_notes.rename(target_notes)
            if old_transcript.exists():
                old_transcript.rename(target_transcript)
            base_name = new_base
        else:
            base_name = existing_base
    else:
        base_name = new_base

    notes_path = out_dir / f"{base_name}.granola.notes.md"
    transcript_path = out_dir / f"{base_name}.granola.transcript.md"

    # Build full file content
    fm_yaml = _frontmatter_to_yaml(frontmatter)
    notes_content = f"{fm_yaml}\n\n{notes_md}"

    transcript_fm = {**frontmatter, "type": "transcript"}
    transcript_fm_yaml = _frontmatter_to_yaml(transcript_fm)
    transcript_content = f"{transcript_fm_yaml}\n\n{transcript_md}"

    # Check if files already exist
    if notes_path.exists() and not force:
        existing = notes_path.read_text(encoding="utf-8")
        # Compare updated_at timestamps
        existing_updated = _extract_granola_updated_at(existing)
        new_updated = frontmatter.get("granola_updated_at", "")

        if existing_updated and new_updated and new_updated <= existing_updated:
            return {
                "notes_path": notes_path,
                "transcript_path": transcript_path,
                "status": "skipped",
            }
        else:
            # Update — write new content
            _write_file(notes_path, notes_content)
            _write_file(transcript_path, transcript_content)
            return {
                "notes_path": notes_path,
                "transcript_path": transcript_path,
                "status": "updated",
            }

    # Create new files
    _write_file(notes_path, notes_content)
    _write_file(transcript_path, transcript_content)
    return {
        "notes_path": notes_path,
        "transcript_path": transcript_path,
        "status": "created",
    }


def _extract_granola_updated_at(content: str) -> str | None:
    """Extract granola_updated_at from frontmatter in file content."""
    m = re.search(r"granola_updated_at:\s*(.+)", content)
    if not m:
        return None
    val = m.group(1).strip()
    # Strip YAML quotes if present
    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
        val = val[1:-1]
    return val


def _should_skip_early(doc: dict[str, Any], project_root: Path) -> bool:
    """Check if a doc can be skipped without fetching its transcript.

    Compares doc['updated_at'] against the stored granola_updated_at
    in the existing notes file. Returns True if we can safely skip.
    """
    created = doc.get("created_at", "")
    date = created[:10] if created else ""
    if not date:
        return False

    parts = date.split("-")
    if len(parts) != 3:
        return False
    year, month, day = parts

    out_dir = project_root / "memory" / "meetings" / year / month / day
    granola_id = doc.get("id", "unknown")
    id_prefix = granola_id[:8] if len(granola_id) >= 8 else granola_id

    existing_base = _find_existing_by_id(out_dir, id_prefix)
    if not existing_base:
        return False

    # Check both new and old extension patterns
    notes_path = out_dir / f"{existing_base}.granola.notes.md"
    if not notes_path.exists():
        notes_path = out_dir / f"{existing_base}.notes.md"
    if not notes_path.exists():
        return False

    existing_updated = _extract_granola_updated_at(notes_path.read_text(encoding="utf-8"))
    new_updated = doc.get("updated_at", "")

    return bool(existing_updated and new_updated and new_updated <= existing_updated)


def _write_file(path: Path, content: str) -> None:
    """Write content to file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Sync state persistence
# ---------------------------------------------------------------------------


def save_sync_state(path: Path, **kwargs: Any) -> None:
    """Save sync state to a JSON file."""
    state = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    state.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_sync_state(path: Path) -> dict[str, Any]:
    """Load sync state from a JSON file. Returns empty dict if missing."""
    if not path.exists():
        return {}
    try:
        return cast("dict[Any, Any]", json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Entity auto-creation
# ---------------------------------------------------------------------------


def match_or_create_attendee(
    attendee: dict[str, Any],
    db: Any,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    """Match an attendee to an existing entity, or auto-create a new one.

    Matching order:
    1. Email match against entity metadata
    2. Name match against entity name/aliases

    If no match and project_root is provided, creates:
    - A new entity in the DB
    - A memory/people/{slug}.md file

    Returns dict with: name, entity_id, matched (bool), created (bool)
    """
    conn = db.get_sqlite_conn()
    name = attendee.get("name", "")
    email = attendee.get("email", "")

    # 1. Try email match
    if email:
        rows = conn.execute("SELECT * FROM entities WHERE entity_type = 'person'").fetchall()
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if meta.get("email", "").lower() == email.lower():
                return {
                    "name": row["name"],
                    "entity_id": row["id"],
                    "matched": True,
                    "created": False,
                }

    # 2. Try name match
    if name:
        from kb.config import find_entity

        entity_row = find_entity(conn, name)
        if entity_row is not None:
            # Learn email if entity doesn't have one yet
            if email:
                meta = json.loads(entity_row["metadata"]) if entity_row["metadata"] else {}
                if not meta.get("email"):
                    meta["email"] = email
                    conn.execute(
                        "UPDATE entities SET metadata = ? WHERE id = ?",
                        (json.dumps(meta), entity_row["id"]),
                    )
                    conn.commit()
                    # Write back to source file
                    if project_root:
                        from kb.writeback import write_entity_to_file

                        write_entity_to_file(db, entity_row["id"], project_root)
            return {
                "name": entity_row["name"],
                "entity_id": entity_row["id"],
                "matched": True,
                "created": False,
            }

    # 3. Auto-create
    if not name:
        return None

    metadata = {}
    if email:
        metadata["email"] = email
    if attendee.get("company"):
        metadata["company"] = attendee["company"]

    # Derive aliases
    aliases = []
    first_name = name.split()[0] if len(name.split()) >= 2 else None
    if first_name:
        aliases.append(first_name)

    slug = re.sub(r"[^\w]+", "-", name.lower()).strip("-")

    from datetime import date

    conn.execute(
        "INSERT INTO entities (name, entity_type, aliases, metadata, source_path, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            name,
            "person",
            json.dumps(aliases),
            json.dumps(metadata),
            f"memory/people/{slug}.md",
            date.today().isoformat(),
        ),
    )
    conn.commit()

    entity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Create memory file if project_root provided
    if project_root:
        _create_person_memory_file(project_root, name, email, attendee.get("company"), slug)

    return {"name": name, "entity_id": entity_id, "matched": False, "created": True}


def _create_person_memory_file(
    project_root: Path,
    name: str,
    email: str | None,
    company: str | None,
    slug: str,
) -> None:
    """Create a memory/people/{slug}.md file for an auto-discovered attendee."""
    people_dir = project_root / "memory" / "people"
    people_dir.mkdir(parents=True, exist_ok=True)

    path = people_dir / f"{slug}.md"
    if path.exists():
        return

    lines = [f"# {name}", ""]
    if email:
        lines.append(f"**Email:** {email}")
    if company:
        lines.append(f"**Company:** {company}")
    lines.append("")
    lines.append("*Auto-created from Granola meeting attendee.*")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Extract notes content from Granola document
# ---------------------------------------------------------------------------


def extract_notes_content(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ProseMirror content nodes from a Granola document.

    Tries multiple locations:
    1. doc['last_viewed_panel']['content'] (real API)
    2. doc['notes']['content'] (alternative location)
    3. doc['last_viewed_panel']['doc']['content'] (test mocks)
    """
    # Real API: panel.content — may be a list of nodes or a ProseMirror doc wrapper
    panel = doc.get("last_viewed_panel") or {}
    content = panel.get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, dict) and content.get("type") == "doc":
        return cast("list[dict[Any, Any]]", content.get("content", []))

    # Alternative: notes field
    notes = doc.get("notes") or {}
    if isinstance(notes, dict):
        content = notes.get("content")
        if isinstance(content, list):
            return content
        if isinstance(content, dict) and content.get("type") == "doc":
            return cast("list[dict[Any, Any]]", content.get("content", []))

    # Test mock fallback: panel.doc.content
    doc_content = panel.get("doc") or {}
    return cast("list[dict[Any, Any]]", doc_content.get("content", []))


def extract_summary(doc: dict[str, Any]) -> str | None:
    """Extract AI summary text from a Granola document.

    Looks in the last_viewed_panel for summary-related content.
    """
    panel = doc.get("last_viewed_panel") or {}
    doc_content = panel.get("doc") or {}
    content = doc_content.get("content", [])

    # Look for a summary section — typically after a heading called "Summary"
    in_summary = False
    summary_parts = []
    for node in content:
        if node.get("type") == "heading":
            text = _render_inline(node.get("content", []))
            if "summary" in text.lower():
                in_summary = True
                continue
            elif in_summary:
                break
        elif in_summary and node.get("type") == "paragraph":
            text = _render_inline(node.get("content", []))
            if text.strip():
                summary_parts.append(text)

    return "\n".join(summary_parts) if summary_parts else None


# ---------------------------------------------------------------------------
# High-level sync orchestration
# ---------------------------------------------------------------------------


def sync_granola(
    project_root: Path,
    data_dir: Path,
    since: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    token_path: Path | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the full Granola sync pipeline.

    Args:
        force: Overwrite existing files even if timestamps match.

    Returns summary dict with counts of created/updated/skipped docs.
    """
    import sys

    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        else:
            print(msg, file=sys.stderr)

    client = GranolaClient(token_path=token_path)

    # Load sync state for incremental sync
    state_path = data_dir / ".granola_sync_state.json"
    state = load_sync_state(state_path)

    effective_since = since
    if not effective_since and state.get("last_sync"):
        effective_since = state["last_sync"][:10]

    _log(
        f"Fetching documents{' since ' + effective_since if effective_since else ' (full sync)'}..."
    )
    docs = client.list_documents(since=effective_since)
    _log(f"Found {len(docs)} documents.")

    if dry_run:
        _log("Dry run — not writing files.")
        return {"total": len(docs), "created": 0, "updated": 0, "skipped": 0, "dry_run": True}

    created = 0
    updated = 0
    skipped = 0

    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        title = doc.get("title") or "Untitled"

        _log(f"[{i + 1}/{len(docs)}] {title}")

        # Early skip: check local timestamp before fetching transcript
        if not force and _should_skip_early(doc, project_root):
            skipped += 1
            continue

        # Fetch transcript (separate API call needed)
        transcript_segments = client.get_transcript(doc_id)

        # Extract notes content — prefer notes_markdown, fall back to ProseMirror
        notes_md = doc.get("notes_markdown") or ""
        if not notes_md.strip():
            pm_nodes = extract_notes_content(doc)
            notes_md = prosemirror_to_markdown(pm_nodes)

        transcript_md = transcript_to_markdown(transcript_segments)
        summary = extract_summary(doc)

        # Build frontmatter — people/calendar data is already on the doc,
        # so we pass an empty metadata dict (no extra API call needed)
        fm = build_frontmatter(doc, {}, summary)

        # Write files
        result = write_meeting(doc, fm, notes_md, transcript_md, project_root, force=force)

        if result["status"] == "created":
            created += 1
        elif result["status"] == "updated":
            updated += 1
        else:
            skipped += 1

        # Rate limiting between documents
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    # Save sync state
    last_updated = (
        max((d.get("updated_at", "") for d in docs), default="")
        if docs
        else state.get("last_sync", "")
    )

    save_sync_state(
        state_path,
        last_sync=last_updated,
        docs_synced=len(docs),
        last_run=datetime.now().isoformat(),
    )

    return {
        "total": len(docs),
        "created": created,
        "updated": updated,
        "skipped": skipped,
    }
