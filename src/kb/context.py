"""Context layer — compressed entity index for AI agents."""

from __future__ import annotations

import contextlib
import json
import math
import re
from datetime import date
from typing import TYPE_CHECKING, Any

from kb.search import _sanitize_fts_input
from kb.types import (
    ContextEntity,
    ContextEntitySummary,
    ContextOutput,
    ContextStats,
    DateRange,
    PinnedDocument,
)

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

    from kb.db import Database

# Standard industry terms — excluded from compact context to reduce noise
_STANDARD_INDUSTRY_TERMS = frozenset(
    {
        "CI",
        "CD",
        "SaaS",
        "API",
        "SDK",
        "CLI",
        "REST",
        "HTTP",
        "HTTPS",
        "DNS",
        "SSH",
        "TLS",
        "SSL",
        "JWT",
        "OAuth",
        "OIDC",
        "SAML",
        "JSON",
        "YAML",
        "XML",
        "CSV",
        "SQL",
        "NoSQL",
        "ORM",
        "AWS",
        "GCP",
        "Azure",
        "S3",
        "EC2",
        "ECS",
        "EKS",
        "GKE",
        "K8s",
        "Docker",
        "Terraform",
        "Ansible",
        "PR",
        "CI/CD",
        "DevOps",
        "SRE",
        "SLA",
        "SLO",
        "SLI",
        "RBAC",
        "SSO",
        "MFA",
        "LDAP",
        "IAM",
        "VPN",
        "SOC2",
        "SOC 2",
        "GDPR",
        "HIPAA",
        "PCI DSS",
        "ISO",
        "OKRs",
        "KPIs",
        "ROI",
        "ARR",
        "MRR",
        "POC",
        "MVP",
        "RCA",
        "RFP",
        "SOP",
        "CPU",
        "GPU",
        "RAM",
        "SSD",
        "HDD",
        "IP",
        "TCP",
        "UDP",
        "SMTP",
        "IMAP",
        "IDE",
        "VSCode",
        "Git",
        "GitHub",
        "GitLab",
        "On-prem",
        "SIEM",
        "WAF",
        "VCS",
        "FP",
        "GPL",
        "LGPL",
        "MPL",
        "EPL",
        "ESG",
        "FUSE",
        "DORA",
        "PSR",
        "EM",
        "EBR",
        "MDM",
        "MR",
    }
)


def _get_entity_mention_counts(conn: sqlite3.Connection) -> dict[int, int]:
    """Count entity mentions per entity."""
    rows = conn.execute(
        "SELECT entity_id, COUNT(*) as cnt FROM entity_mentions GROUP BY entity_id"
    ).fetchall()
    return {r["entity_id"]: r["cnt"] for r in rows}


_DETAIL_KEYS = ("role", "team", "company", "reports_to")


def _get_fact_counts(conn: sqlite3.Connection) -> dict[int, int]:
    """Count facts per entity."""
    rows = conn.execute(
        "SELECT entity_id, COUNT(*) as cnt FROM facts GROUP BY entity_id"
    ).fetchall()
    return {r["entity_id"]: r["cnt"] for r in rows}


def _is_key_person(entity: ContextEntity, fact_counts: dict[int, int]) -> bool:
    """Return True if a person has meaningful detail (2+ metadata fields or facts).

    Key people are unpinned entities with substantive metadata -- enough to be
    useful in compressed context output.
    """
    meta = entity.metadata
    field_count = sum(1 for k in _DETAIL_KEYS if meta.get(k))
    facts = fact_counts.get(entity.id, 0)
    return field_count >= 2 or facts > 0


def _get_date_range(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    """Get earliest and latest doc_date."""
    row = conn.execute(
        "SELECT MIN(doc_date) as earliest, MAX(doc_date) as latest FROM documents WHERE doc_date IS NOT NULL"
    ).fetchone()
    return row["earliest"], row["latest"]


def _format_person(entity: ContextEntity) -> str:
    """Format a person entity for the compressed display."""
    meta = entity.metadata
    aliases = entity.aliases

    # Pick shortest recognisable name (alias or first name)
    short_name = _short_name(entity.name, aliases)

    # Build role/team tag
    parts: list[str] = []
    if meta.get("role"):
        # Abbreviate long roles
        role = _truncate_role(meta["role"], max_len=25)
        parts.append(role)
    elif meta.get("team"):
        parts.append(meta["team"])

    if parts:
        return f"{short_name}({'/'.join(parts)})"
    return str(short_name)


def _format_project(entity: ContextEntity) -> str:
    """Format a project entity for the compressed display."""
    meta = entity.metadata
    parts: list[str] = []
    if meta.get("lead"):
        parts.append(meta["lead"])
    if meta.get("status"):
        parts.append(meta["status"])
    name = entity.name
    if parts:
        return f"{name}({','.join(parts)})"
    return name


def _format_team(entity: ContextEntity) -> str:
    """Format a team entity for the compressed display."""
    aliases = entity.aliases
    abbrev = next((a for a in aliases if a.isupper() and len(a) <= 5), None)
    name = entity.name
    if abbrev:
        return f"{name}({abbrev})"
    return name


def _parse_glossary_terms(
    project_root: Path, *, exclude_nicknames: bool = False
) -> list[tuple[str, str, str]]:
    """Parse term=expansion pairs from memory/glossary.md.

    Returns (term, expansion, section_name) 3-tuples.

    If *exclude_nicknames* is True, skips any table section whose header row
    contains a "Full name" column (these are nickname→person mappings that
    duplicate the People section in context output).
    """
    glossary = project_root / "memory" / "glossary.md"
    if not glossary.exists():
        return []

    skip_terms = {"term", "nickname", "---", "------"}
    skip_expansions = {"expansion", "meaning", "full name", "notes", "---", "---------"}

    terms = []
    in_nickname_section = False
    current_section = ""
    text = glossary.read_text(encoding="utf-8")
    for line in text.splitlines():
        # Detect section headers (## headings reset nickname flag)
        if line.startswith("##"):
            in_nickname_section = False
            current_section = line.lstrip("#").strip()
            continue

        # Detect table header rows that mark a nickname section
        if exclude_nicknames and re.match(r"^\|.*\bFull name\b.*\|", line, re.IGNORECASE):
            in_nickname_section = True
            continue

        if in_nickname_section:
            continue

        m = re.match(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|", line)
        if m:
            term = m.group(1).strip()
            expansion = m.group(2).strip()
            if term.lower() in skip_terms:
                continue
            if expansion.lower() in skip_expansions:
                continue
            # Skip separator rows
            if all(c in "-| " for c in term):
                continue
            terms.append((term, expansion, current_section))

    return terms


def _get_topic_entity_ids(db: Database, topic: str) -> set[int]:
    """Get entity IDs relevant to a topic by searching documents and entity names.

    Two strategies combined:
    1. Direct name match: entities whose name/aliases match the topic text
    2. FTS search: entities mentioned in documents matching the topic
    """
    conn = db.get_sqlite_conn()
    entity_ids: set[int] = set()

    # Strategy 1: Direct entity name/alias matching against topic
    topic_lower = topic.lower()
    entity_rows = conn.execute("SELECT id, name, aliases FROM entities").fetchall()
    for er in entity_rows:
        names = [er["name"].lower()]
        if er["aliases"]:
            names.extend(a.lower() for a in json.loads(er["aliases"]))
        for name in names:
            if len(name) >= 3 and (name in topic_lower or topic_lower in name):
                entity_ids.add(er["id"])
                break

    # Strategy 2: FTS search for topic in document content
    safe_topic = _sanitize_fts_input(topic)
    if safe_topic:
        rows: list[Any] = []
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT c.document_id
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                LIMIT 50
                """,
                (safe_topic,),
            ).fetchall()
        except Exception:
            words = safe_topic.split()
            if len(words) > 1:
                or_query = " OR ".join(words)
                with contextlib.suppress(Exception):
                    rows = conn.execute(
                        """
                        SELECT DISTINCT c.document_id
                        FROM chunks_fts
                        JOIN chunks c ON c.id = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        LIMIT 50
                        """,
                        (or_query,),
                    ).fetchall()

        if rows:
            doc_ids = [r["document_id"] for r in rows]
            placeholders = ",".join("?" * len(doc_ids))
            mention_rows = conn.execute(
                f"SELECT DISTINCT entity_id FROM entity_mentions WHERE document_id IN ({placeholders})",
                doc_ids,
            ).fetchall()
            entity_ids.update(r["entity_id"] for r in mention_rows)

    return entity_ids


def _pack_items(items: list[str], sep: str = "|") -> list[str]:
    """Pack items into lines of ~80 chars, separated by |."""
    lines = []
    current = ""
    for item in items:
        candidate = f"{current}{sep}{item}" if current else item
        if len(candidate) + 1 > 80 and current:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _format_person_compact(entity: ContextEntity) -> str:
    """Format a person for compact display, with star for pinned and mention count."""
    meta = entity.metadata
    aliases = entity.aliases

    # Pick shortest recognisable name
    short_name = _short_name(entity.name, aliases)

    pinned = "\u2605" if entity.pinned else ""

    parts: list[str] = []
    if meta.get("role"):
        role = _truncate_role(meta["role"])
        parts.append(role)

    mc = entity.mention_count
    parts.append(f"{mc}m")

    return f"{short_name}{pinned}({','.join(parts)})"


def _short_name(name: str, aliases: list[str]) -> str:
    """Pick shortest recognisable name from aliases."""
    short = name
    for a in aliases:
        if 2 <= len(a) < len(short) and not ("-" in a and a == a.lower()):
            short = a
    return short


def _truncate_role(role: str, max_len: int = 20) -> str:
    """Truncate a role string if longer than max_len."""
    if len(role) > max_len:
        return role[: max_len - 3] + "..."
    return role


def _freshness_indicator(entity: ContextEntity) -> str:
    """Return freshness badge: ● (fresh ≤30d), ◐Nd (aging 31-90d), ○Nd (stale >90d)."""
    most_recent = max(
        entity.updated_at or "",
        entity.last_mentioned_at or "",
    )
    if not most_recent:
        return " ○"
    try:
        d = date.fromisoformat(most_recent)
    except (ValueError, TypeError):
        return " ○"
    age = (date.today() - d).days
    if age <= 30:
        return " ●"
    elif age <= 90:
        return f" ◐{age}d"
    else:
        return f" ○{age}d"


def _recency_sort_key(entity: ContextEntity) -> float:
    """Sort key: mention_count * recency_weight. Higher = more prominent."""
    most_recent = max(
        entity.updated_at or "",
        entity.last_mentioned_at or "",
    )
    if not most_recent:
        weight = 0.5
    else:
        try:
            d = date.fromisoformat(most_recent)
            age = (date.today() - d).days
            weight = math.exp(-math.log(2) * age / 90) if age >= 0 else 1.0
        except (ValueError, TypeError):
            weight = 0.5
    return entity.mention_count * weight


def _format_person_pinned(entity: ContextEntity) -> str:
    """Format a pinned person: Name\u2605(Role|team:Team|\u2192ReportsTo,Nm)."""
    meta = entity.metadata
    short = _short_name(entity.name, entity.aliases)

    parts: list[str] = []
    if meta.get("role"):
        parts.append(_truncate_role(meta["role"]))
    if meta.get("team"):
        parts.append(f"team:{meta['team']}")
    if meta.get("reports_to"):
        parts.append(f"\u2192{meta['reports_to']}")

    # Join metadata with |, then append mention count with ,
    fresh = _freshness_indicator(entity)
    if parts:
        label = "|".join(parts)
        return f"{short}\u2605({label},{entity.mention_count}m){fresh}"
    return f"{short}\u2605({entity.mention_count}m){fresh}"


def _format_person_key(entity: ContextEntity) -> str:
    """Format a key person: Name(Role|\u2192ReportsTo,Nm)."""
    meta = entity.metadata
    short = _short_name(entity.name, entity.aliases)

    parts: list[str] = []
    if meta.get("role"):
        parts.append(_truncate_role(meta["role"]))
    if meta.get("reports_to"):
        parts.append(f"\u2192{meta['reports_to']}")

    # Join role+reports_to with |, then append mention count with ,
    fresh = _freshness_indicator(entity)
    if parts:
        label = "|".join(parts)
        return f"{short}({label},{entity.mention_count}m){fresh}"
    return f"{short}({entity.mention_count}m){fresh}"


def _get_pinned_documents(conn: sqlite3.Connection) -> list[PinnedDocument]:
    """Load pinned documents with their section headings."""
    rows = conn.execute(
        "SELECT id, path, title FROM documents WHERE pinned = 1 ORDER BY title"
    ).fetchall()
    pinned: list[PinnedDocument] = []
    for r in rows:
        # Extract section headings from chunks
        heading_rows = conn.execute(
            "SELECT DISTINCT heading FROM chunks WHERE document_id = ? AND heading IS NOT NULL AND heading != '__document__' ORDER BY chunk_index",
            (r["id"],),
        ).fetchall()
        headings = [str(h["heading"]) for h in heading_rows]
        pinned.append(
            PinnedDocument(
                path=str(r["path"]),
                title=str(r["title"] or r["path"].rsplit("/", 1)[-1]),
                headings=headings,
            )
        )
    return pinned


def _render_compact(
    *,
    topic: str | None,
    entity_count: int,
    doc_count: int,
    earliest: str | None,
    latest: str | None,
    pinned_docs: list[PinnedDocument],
    pinned_people: list[ContextEntity],
    key_people: list[ContextEntity],
    topic_people: list[ContextEntity],
    projects: list[ContextEntity],
    teams: list[ContextEntity],
    gg_acronyms: list[tuple[str, str, str]],
    internal_terms: list[tuple[str, str, str]],
) -> str:
    """Render compact pipe-delimited format for AI context injection."""
    lines: list[str] = []

    # Header
    date_range = f"{earliest}\u2192{latest}" if earliest and latest else ""
    lines.append(f"[KB:{entity_count} entities|{doc_count} docs|{date_range}]")

    # Pinned documents (most important context, shown first)
    if pinned_docs:
        lines.append(f"[Pinned:{len(pinned_docs)}]")
        for pd in pinned_docs:
            headings_str = f" ({' | '.join(pd.headings)})" if pd.headings else ""
            lines.append(f'  {pd.path} "{pd.title}"{headings_str}')

    # People (topic mode: flat list; full mode: pinned + key tiers)
    if topic_people:
        lines.append(f"[People:{len(topic_people)} by mentions]")
        person_strs = [_format_person_compact(p) for p in topic_people]
        lines.extend(_pack_items(person_strs))
        lines.append('\u2192kb entity find "name" --json')
    else:
        if pinned_people:
            lines.append("[People:pinned]")
            person_strs = [_format_person_pinned(p) for p in pinned_people]
            lines.extend(person_strs)  # one per line for readability
        if key_people:
            lines.append("[People:key]")
            person_strs = [_format_person_key(p) for p in key_people]
            lines.extend(_pack_items(person_strs))
        if pinned_people or key_people:
            lines.append('\u2192kb entity find "name" --json')

    # Projects
    if projects:
        lines.append(f"[Projects:{len(projects)} active]")
        proj_strs = [_format_project(p) + ("\u2605" if p.pinned else "") for p in projects]
        lines.extend(_pack_items(proj_strs))
        lines.append('\u2192kb entity find "project" --json')

    # Teams
    if teams:
        lines.append(f"[Teams:{len(teams)}]")
        team_strs = [_format_team(t) + ("\u2605" if t.pinned else "") for t in teams]
        lines.extend(_pack_items(team_strs))

    # GG Acronyms (only in full context)
    if not topic and gg_acronyms:
        lines.append("[GG Acronyms]")
        acro_strs = [
            f"{t}={e.split('|')[0].strip()}" if "|" in e else f"{t}={e}" for t, e, _s in gg_acronyms
        ]
        lines.extend(_pack_items(acro_strs))

    # GG Jargon (only in full context)
    if not topic and internal_terms:
        lines.append("[GG Jargon]")
        jargon_strs = [t for t, _e, _s in internal_terms]
        lines.extend(_pack_items(jargon_strs))

    if not topic and (gg_acronyms or internal_terms):
        lines.append("\u2192kb view memory/glossary.md")

    # Commands section
    lines.append("[Commands]")
    lines.append('search: kb search "query" --fast --json|entity: kb entity find "name" --json')
    lines.append('timeline: kb entity timeline "name" --json|doc: kb view <path> --json')
    lines.append('note: kb memory add "title" --body "..." --tags t1,t2 --pin')
    lines.append("pin: kb pin <path-or-title>|unpin: kb unpin <path-or-title>")

    return "\n".join(lines)


def _render_human(
    *,
    topic: str | None,
    entity_count: int,
    doc_count: int,
    earliest: str | None,
    latest: str | None,
    pinned_docs: list[PinnedDocument],
    pinned_people: list[ContextEntity],
    key_people: list[ContextEntity],
    topic_people: list[ContextEntity],
    projects: list[ContextEntity],
    teams: list[ContextEntity],
    gg_acronyms: list[tuple[str, str, str]],
    internal_terms: list[tuple[str, str, str]],
) -> str:
    """Render human-readable markdown format."""
    lines: list[str] = []

    if topic:
        lines.append(f"# Context: {topic}")
    else:
        lines.append("# Knowledge Base Context")
    lines.append(
        f"|{entity_count} entities, {doc_count} documents"
        + (f", {earliest} to {latest}" if earliest and latest else "")
    )
    lines.append("")

    # Pinned documents
    if pinned_docs:
        lines.append(f"## Pinned Documents ({len(pinned_docs)})")
        for pd in pinned_docs:
            headings_str = f"  Sections: {', '.join(pd.headings)}" if pd.headings else ""
            lines.append(f"- **{pd.title}** \u2014 `{pd.path}`")
            if headings_str:
                lines.append(headings_str)
        lines.append("")

    # People sections (topic mode: flat; full mode: pinned + key tiers)
    if topic_people:
        lines.append(f"## People ({len(topic_people)})")
        person_strs = [_format_person(p) for p in topic_people]
        current_line = "|"
        for ps in person_strs:
            if len(current_line) + len(ps) + 1 > 80 and current_line != "|":
                lines.append(current_line)
                current_line = "|"
            current_line += ps + " "
        if current_line.strip("|").strip():
            lines.append(current_line)
        lines.append('\u2192 Details: `kb entity find "name" --json`')
        lines.append("")
    else:
        if pinned_people:
            lines.append(f"## Pinned People ({len(pinned_people)})")
            person_strs = [_format_person_pinned(p) for p in pinned_people]
            for ps in person_strs:
                lines.append(ps)
            lines.append("")
        if key_people:
            lines.append(f"## Key People ({len(key_people)})")
            person_strs = [_format_person_key(p) for p in key_people]
            lines.extend(_pack_items(person_strs))
            lines.append("")
        if pinned_people or key_people:
            lines.append('\u2192 Details: `kb entity find "name" --json`')
            lines.append("")

    # Projects section
    if projects:
        lines.append(f"## Projects ({len(projects)})")
        proj_strs = [_format_project(p) for p in projects]
        current_line = "|"
        for ps in proj_strs:
            if len(current_line) + len(ps) + 1 > 80 and current_line != "|":
                lines.append(current_line)
                current_line = "|"
            current_line += ps + " "
        if current_line.strip("|").strip():
            lines.append(current_line)
        lines.append('\u2192 Details: `kb entity find "project" --json`')
        lines.append("")

    # Teams section
    if teams:
        lines.append(f"## Teams ({len(teams)})")
        team_strs = [_format_team(t) for t in teams]
        lines.append("|" + " ".join(team_strs))
        lines.append("")

    # Terms sections (only in full context, not topic-filtered)
    if not topic:
        if gg_acronyms:
            lines.append("## GG Acronyms")
            acro_strs = [
                f"{t}={e.split('|')[0].strip()}" if "|" in e else f"{t}={e}"
                for t, e, _s in gg_acronyms
            ]
            current_line = "|"
            for ts in acro_strs:
                if len(current_line) + len(ts) + 1 > 80 and current_line != "|":
                    lines.append(current_line)
                    current_line = "|"
                current_line += ts + " "
            if current_line.strip("|").strip():
                lines.append(current_line)
            lines.append("")

        if internal_terms:
            lines.append("## GG Jargon")
            jargon_strs = [t for t, _e, _s in internal_terms]
            current_line = "|"
            for ts in jargon_strs:
                if len(current_line) + len(ts) + 1 > 80 and current_line != "|":
                    lines.append(current_line)
                    current_line = "|"
                current_line += ts + " "
            if current_line.strip("|").strip():
                lines.append(current_line)

        if gg_acronyms or internal_terms:
            lines.append("\u2192 Full glossary: `kb view memory/glossary.md`")

    return "\n".join(lines)


def generate_context(
    db: Database,
    project_root: Path,
    topic: str | None = None,
    fmt: str = "compact",
) -> ContextOutput:
    """Generate a compressed entity index for AI agents.

    Args:
        fmt: "compact" (default, pipe-delimited) or "human" (markdown)

    Returns a ContextOutput with text, stats, and entity summaries.
    """
    conn = db.get_sqlite_conn()

    # Gather stats
    doc_count: int = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()["cnt"]
    entity_count: int = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
    earliest, latest = _get_date_range(conn)
    mention_counts = _get_entity_mention_counts(conn)

    # Load all entities (including freshness columns)
    rows = conn.execute(
        "SELECT id, name, entity_type, aliases, metadata, source_path, pinned,"
        " updated_at, last_mentioned_at"
        " FROM entities ORDER BY entity_type, name"
    ).fetchall()

    all_entities: list[ContextEntity] = []
    for r in rows:
        raw_aliases = json.loads(r["aliases"]) if r["aliases"] else []
        raw_metadata = json.loads(r["metadata"]) if r["metadata"] else {}
        # Ensure aliases are list[str] and metadata keys are str for strict Pydantic
        aliases = [str(a) for a in raw_aliases]
        metadata = {str(k): v for k, v in raw_metadata.items()}
        all_entities.append(
            ContextEntity(
                id=int(r["id"]),
                name=str(r["name"]),
                entity_type=str(r["entity_type"]),
                aliases=aliases,
                metadata=metadata,
                source_path=str(r["source_path"]) if r["source_path"] is not None else None,
                mention_count=mention_counts.get(r["id"], 0),
                pinned=bool(r["pinned"]),
                updated_at=str(r["updated_at"]) if r["updated_at"] else None,
                last_mentioned_at=str(r["last_mentioned_at"]) if r["last_mentioned_at"] else None,
            )
        )

    # Filter by topic if provided
    if topic:
        relevant_ids = _get_topic_entity_ids(db, topic)
        filtered = [e for e in all_entities if e.id in relevant_ids]
    else:
        filtered = all_entities

    # Group by type with smart filtering (only in full context, not topic mode)
    pinned_people: list[ContextEntity] = []
    key_people: list[ContextEntity] = []
    if topic:
        people = sorted(
            [e for e in filtered if e.entity_type == "person"],
            key=lambda e: e.mention_count,
            reverse=True,
        )
        projects = [e for e in filtered if e.entity_type == "project"]
        teams = [e for e in filtered if e.entity_type == "team"]
    else:
        # People: split into pinned and key tiers (drop "others")
        all_people = [e for e in filtered if e.entity_type == "person"]
        fact_counts = _get_fact_counts(conn)
        pinned_people = sorted(
            [e for e in all_people if e.pinned],
            key=_recency_sort_key,
            reverse=True,
        )
        key_people = sorted(
            [e for e in all_people if not e.pinned and _is_key_person(e, fact_counts)],
            key=_recency_sort_key,
            reverse=True,
        )
        people = pinned_people + key_people  # for entity summaries

        # Projects: drop completed
        projects = [
            e
            for e in filtered
            if e.entity_type == "project"
            and not e.metadata.get("status", "").lower().startswith("completed")
        ]

        # Teams: drop 0-mention, sort by mentions
        teams = sorted(
            [e for e in filtered if e.entity_type == "team" and e.mention_count > 0],
            key=lambda e: e.mention_count,
            reverse=True,
        )

    # Parse glossary terms (excluding nicknames — redundant with People section)
    all_terms = _parse_glossary_terms(project_root, exclude_nicknames=True)

    # Split by section and filter
    acronym_terms = [
        (t, e, s) for t, e, s in all_terms if "acronym" in s.lower() or "abbreviat" in s.lower()
    ]
    internal_terms = [(t, e, s) for t, e, s in all_terms if "internal" in s.lower()]
    # Filter out standard industry terms from acronyms
    gg_acronyms = [(t, e, s) for t, e, s in acronym_terms if t not in _STANDARD_INDUSTRY_TERMS]

    # Load pinned documents
    pinned_docs = _get_pinned_documents(conn)

    # Render text based on format
    render_kwargs: dict[str, Any] = dict(
        topic=topic,
        entity_count=entity_count,
        doc_count=doc_count,
        earliest=earliest,
        latest=latest,
        pinned_docs=pinned_docs,
        pinned_people=pinned_people if not topic else [],
        key_people=key_people if not topic else [],
        topic_people=people if topic else [],
        projects=projects,
        teams=teams,
        gg_acronyms=gg_acronyms,
        internal_terms=internal_terms,
    )

    if fmt == "human":
        text = _render_human(**render_kwargs)
    else:
        text = _render_compact(**render_kwargs)

    stats = ContextStats(
        documents=doc_count,
        entities=entity_count,
        date_range=DateRange(earliest=earliest, latest=latest),
    )

    # Sort filtered entities by mention count (most referenced first)
    filtered_sorted = sorted(filtered, key=lambda e: e.mention_count, reverse=True)

    return ContextOutput(
        text=text,
        stats=stats,
        entities=[
            ContextEntitySummary(
                name=e.name,
                entity_type=e.entity_type,
                mention_count=e.mention_count,
            )
            for e in filtered_sorted
        ],
    )
