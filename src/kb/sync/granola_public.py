"""Granola public-API sync — Bearer-key client + writeback into existing pipeline.

Replaces the internal `/v2/get-documents` path (see `kb.sync.granola`) which broke in
May 2026 when Granola 7.205.0 migrated credentials to encrypted local storage. This
module talks to the documented public API at ``https://api.granola.ai/v1`` using a
long-lived Bearer key — no token refresh, no plaintext file scraping.

Output layout (filenames, frontmatter shape, transcript markdown) is identical to the
internal-API path so existing meeting files round-trip and downstream consumers (kbx
indexer, prep/debrief tooling) need no changes.

Known field-coverage loss vs the legacy internal API:
  * Attendees: public ``User`` is ``{name, email}`` only. The legacy ``people.attendees``
    carried a ``details`` blob that ``build_frontmatter`` mined for company name
    (``details.company.name`` and ``details.person.employment.name``) and for mailing-list
    group expansion (``details.group.members``). Files synced via the public API will not
    carry the per-attendee ``company`` tag nor expand group invitees. Existing legacy
    files retain whatever they captured at the time of their last sync.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from kb.sync.granola import (
    build_frontmatter,
    load_sync_state,
    save_sync_state,
    transcript_to_markdown,
    write_meeting,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://public-api.granola.ai/v1"
KEYCHAIN_SERVICE = "granola_api_key"
ENV_VAR = "GRANOLA_API_KEY"
MAX_PAGE_SIZE = 30  # public API hard limit
REQUEST_DELAY = 0.25  # seconds — keeps us well under the 5 req/sec sustained limit


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GranolaPublicAPIError(RuntimeError):
    """Raised when the public Granola API rejects a request or no key is available."""


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp to a UTC-aware datetime.

    Accepts ``Z``-suffixed, offset-aware, and naive (assumed UTC) strings.
    Returns ``None`` on empty or unparseable input — callers use that as
    "treat as oldest possible" without crashing the sync.
    """
    if not ts:
        return None
    try:
        # ``fromisoformat`` accepts offsets natively from Python 3.11+, and
        # tolerates the ``Z`` suffix from 3.11 — we normalise it for safety.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _redact(key: str) -> str:
    """Return a key prefix safe for log/error messages (first 8 chars)."""
    return (key[:8] + "…") if len(key) > 8 else "<short>"


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Resolve the Granola API key from env or macOS Keychain.

    Order: ``GRANOLA_API_KEY`` environment variable, then
    ``security find-generic-password -a $USER -s granola_api_key -w``.
    """
    env_key = os.environ.get(ENV_VAR)
    if env_key and env_key.strip():
        return env_key.strip()

    user = os.environ.get("USER", "")
    if not user:
        raise GranolaPublicAPIError(
            f"Granola API key not found. Set {ENV_VAR} or store it in macOS Keychain "
            f"as service '{KEYCHAIN_SERVICE}'."
        )

    try:
        result = subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
            ["security", "find-generic-password", "-a", user, "-s", KEYCHAIN_SERVICE, "-w"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as e:
        raise GranolaPublicAPIError(
            f"`security` CLI not available; set {ENV_VAR} env var instead."
        ) from e
    except subprocess.CalledProcessError as e:
        raise GranolaPublicAPIError(
            f"Granola API key not found in Keychain (service '{KEYCHAIN_SERVICE}', "
            f"account '{user}'). Create it in Granola desktop: Settings → Connectors "
            f"→ API keys → Personal API key, then store with `security add-generic-password "
            f"-a $USER -s {KEYCHAIN_SERVICE} -w <key>`. Alternatively set {ENV_VAR}."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise GranolaPublicAPIError("Timed out reading API key from Keychain.") from e

    key = result.stdout.strip()
    if not key:
        raise GranolaPublicAPIError(
            f"Keychain entry for '{KEYCHAIN_SERVICE}' returned an empty value."
        )
    return key


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GranolaPublicClient:
    """Thin client for Granola's public REST API.

    Auth: ``Authorization: Bearer <key>``. Key sourced from ``api_key`` arg,
    else env var, else macOS Keychain — see :func:`_get_api_key`.

    Holds a persistent ``httpx.Client`` so the list+get-per-note pattern reuses
    the underlying HTTP/2 connection instead of re-handshaking per request. Use
    as a context manager or call :meth:`close` when finished.
    """

    def __init__(self, api_key: str | None = None, base_url: str = API_BASE) -> None:
        self._api_key = api_key or _get_api_key()
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GranolaPublicClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = self._http.request(method, path, params=params)
        if response.status_code == 401:
            raise GranolaPublicAPIError(
                "Granola public API rejected the key (401 Unauthorized). "
                f"Regenerate via Granola desktop Settings → Connectors → API keys, "
                f"then update the {KEYCHAIN_SERVICE} Keychain entry."
            )
        if response.status_code == 403:
            raise GranolaPublicAPIError(
                "Granola public API forbidden (403). The key is valid but lacks scope "
                "for this resource — check your plan tier (Personal API key requires "
                "Business or Enterprise) and whether the workspace owner has revoked access."
            )
        if response.status_code == 429:
            raise GranolaPublicAPIError(
                "Granola public API rate limit (429). Sustained limit is 5 req/sec; "
                "consider increasing REQUEST_DELAY."
            )
        if response.status_code == 404:
            raise GranolaPublicAPIError(f"Granola public API: not found ({path}).")
        response.raise_for_status()
        return dict(response.json())

    def list_notes(
        self,
        updated_after: str | None = None,
        page_size: int = MAX_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """Yield every NoteSummary updated after ``updated_after`` (inclusive).

        Pagination is cursor-based; advances until ``hasMore`` is false. Raises
        ``GranolaPublicAPIError`` if the cursor fails to advance (defends against
        a server-side bug returning the same cursor) or if the response is missing
        the ``hasMore`` field entirely (defends against a future field rename).
        """
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": min(page_size, MAX_PAGE_SIZE)}
            if updated_after:
                params["updated_after"] = updated_after
            if cursor:
                params["cursor"] = cursor

            data = self._request("GET", "/notes", params=params)
            yield from data.get("notes", [])

            if "hasMore" not in data:
                raise GranolaPublicAPIError(
                    "Granola /v1/notes response missing 'hasMore' field "
                    f"(keys={sorted(data.keys())}). The API contract may have changed; "
                    "review the public OpenAPI spec and adjust list_notes accordingly."
                )

            if not data["hasMore"]:
                break

            next_cursor = data.get("cursor")
            if not next_cursor:
                # hasMore=true but no cursor — server bug or end-of-data signalling
                # via cursor=null. Treat as end-of-data, log a warning so it surfaces.
                print(
                    "WARN: Granola /v1/notes returned hasMore=true with no cursor; "
                    "treating as end-of-data.",
                    file=sys.stderr,
                )
                break
            if next_cursor == cursor:
                raise GranolaPublicAPIError(
                    f"Granola /v1/notes cursor failed to advance ({next_cursor!r}); "
                    "aborting to avoid an infinite loop. Retry the sync; if it persists, "
                    "report to Granola support."
                )
            cursor = next_cursor
            time.sleep(REQUEST_DELAY)

    def get_note(self, note_id: str, include_transcript: bool = True) -> dict[str, Any]:
        """Fetch a single Note by id, optionally including the transcript array."""
        params = {"include": "transcript"} if include_transcript else None
        return self._request("GET", f"/notes/{note_id}", params=params)


# ---------------------------------------------------------------------------
# Public-API → internal-doc shape transformation
# ---------------------------------------------------------------------------

# Public Speaker.source enum is ('microphone', 'speaker'); internal pipeline expects
# ('microphone', 'system'). The internal-API path used 'system' to mean system audio.
_SOURCE_PUBLIC_TO_INTERNAL = {"microphone": "microphone", "speaker": "system"}


def _note_to_internal_doc(note: dict[str, Any]) -> dict[str, Any]:
    """Map a public-API Note to the internal-doc shape consumed by build_frontmatter
    and write_meeting.
    """
    cal = note.get("calendar_event") or {}
    attendees = note.get("attendees") or []
    owner = note.get("owner") or {}

    doc: dict[str, Any] = {
        "id": note.get("id", ""),
        "title": note.get("title") or "Untitled",
        "created_at": note.get("created_at", ""),
        "updated_at": note.get("updated_at", ""),
        "notes_markdown": note.get("summary_markdown") or note.get("summary_text") or "",
        "people": {
            "creator": owner if owner.get("email") else None,
            "attendees": attendees,
        },
    }

    if cal.get("calendar_event_id"):
        doc["google_calendar_event"] = {
            "summary": cal.get("event_title") or "",
            "organizer": {"email": cal.get("organiser") or ""},
            "start": {"dateTime": cal.get("scheduled_start_time") or ""},
            "end": {"dateTime": cal.get("scheduled_end_time") or ""},
            "iCalUID": cal.get("calendar_event_id") or "",
        }

    return doc


def _transform_transcript(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map public-API transcript segments to the internal shape.

    Diarized segments (those with ``speaker.diarization_label``) carry a ``speaker``
    key so :func:`transcript_to_markdown` renders them as labelled lines. Non-diarized
    segments fall back to a ``source`` key that maps to the existing 'Me'/'System'
    labels.
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        speaker = seg.get("speaker") or {}
        label = (speaker.get("diarization_label") or "").strip()
        source = speaker.get("source") or ""
        text = seg.get("text", "")
        entry: dict[str, Any] = {
            "text": text,
            "start_timestamp": seg.get("start_time", ""),
            "end_timestamp": seg.get("end_time", ""),
        }
        if label:
            entry["speaker"] = label
        else:
            entry["source"] = _SOURCE_PUBLIC_TO_INTERNAL.get(source, source)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


def sync_granola_public(
    project_root: Path,
    data_dir: Path,
    since: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    api_key: str | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the public-API Granola sync.

    Returns a summary dict ``{total, created, updated, skipped[, dry_run]}`` matching
    the legacy :func:`kb.sync.granola.sync_granola` shape.
    """

    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        else:
            print(msg, file=sys.stderr)

    client = GranolaPublicClient(api_key=api_key)

    state_path = data_dir / ".granola_sync_state.json"
    state = load_sync_state(state_path)

    # ``since`` may arrive as a bare date (YYYY-MM-DD) or full ISO. ``last_sync``
    # is always a full ISO timestamp (we save it that way below); use it verbatim
    # rather than slicing the date prefix, so the next run picks up exactly where
    # the previous one stopped and we don't re-pull a whole day's notes.
    effective_since = since or state.get("last_sync") or None
    updated_after: str | None = None
    if effective_since:
        updated_after = (
            effective_since if "T" in effective_since else f"{effective_since}T00:00:00Z"
        )

    _log(
        f"Fetching notes{' updated since ' + effective_since if effective_since else ' (full sync)'}..."
    )

    # Iterate lazily — full sync may yield hundreds of summaries.
    note_iter = client.list_notes(updated_after=updated_after)

    if dry_run:
        summaries = list(note_iter)
        _log(f"Found {len(summaries)} notes.")
        _log("Dry run — not writing files.")
        return {
            "total": len(summaries),
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "dry_run": True,
        }

    created = 0
    updated = 0
    skipped = 0
    total = 0
    latest_dt = _parse_iso(state.get("last_sync", "") or "")

    state_save_interval = 10  # checkpoint every N notes so a mid-loop crash keeps progress

    def _persist_state() -> None:
        if latest_dt is not None:
            save_sync_state(
                state_path,
                last_sync=latest_dt.isoformat().replace("+00:00", "Z"),
            )

    try:
        for summary in note_iter:
            total += 1
            note_id = summary["id"]
            title = summary.get("title") or "Untitled"
            _log(f"[{total}] {title}")

            note = client.get_note(note_id, include_transcript=True)
            doc = _note_to_internal_doc(note)

            notes_md = note.get("summary_markdown") or note.get("summary_text") or ""
            transcript_segments = _transform_transcript(note.get("transcript") or [])
            transcript_md = transcript_to_markdown(transcript_segments)

            fm = build_frontmatter(doc, {}, summary=note.get("summary_text"))
            # Keep ``source: granola-api`` for parity with legacy-written files so a
            # one-off churn doesn't rewrite every meeting. Provenance of the API
            # path is recorded in a separate field that older readers ignore.
            fm["source"] = "granola-api"
            fm["granola_api"] = "public"

            result = write_meeting(
                doc,
                fm,
                notes_md,
                transcript_md,
                project_root,
                force=force,
                summary_md=notes_md,
                variant=None,
            )

            if result["status"] == "created":
                created += 1
            elif result["status"] == "updated":
                updated += 1
            else:
                skipped += 1

            note_dt = _parse_iso(note.get("updated_at", ""))
            if note_dt is not None and (latest_dt is None or note_dt > latest_dt):
                latest_dt = note_dt

            if total % state_save_interval == 0:
                _persist_state()

            time.sleep(REQUEST_DELAY)
    finally:
        _persist_state()

    return {
        "total": total,
        "created": created,
        "updated": updated,
        "skipped": skipped,
    }
