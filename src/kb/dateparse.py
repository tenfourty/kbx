"""Extract date filters from natural-language search queries."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Month + year: "January 2026", "Jan 2026"
_MONTH_YEAR = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE)

# ISO month: "2026-01"
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b")

# Relative: "last week", "last N days/weeks/months"
_LAST_N = re.compile(r"\blast\s+(\d+)\s+(day|days|week|weeks|month|months)\b", re.IGNORECASE)
_LAST_WEEK = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_PAST_N = re.compile(r"\bpast\s+(\d+)\s+(day|days|week|weeks|month|months)\b", re.IGNORECASE)
_THIS_YEAR = re.compile(r"\bthis\s+year\b", re.IGNORECASE)
_THIS_MONTH = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY = re.compile(r"\btoday\b", re.IGNORECASE)


def _month_range(year: int, month: int) -> tuple[str, str]:
    """Return (first_day, last_day) ISO strings for a month."""
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


def _months_ago(n: int) -> str:
    today = date.today()
    month = today.month - n
    year = today.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(today.day, calendar.monthrange(year, month)[1])
    return date(year, month, day).isoformat()


def extract_date_filters(query: str) -> tuple[str, str | None, str | None]:
    """Extract date intent from a search query.

    Returns (cleaned_query, from_date, to_date).
    If no date is found, returns (original_query, None, None).
    """
    # Try month + year: "January 2026"
    m = _MONTH_YEAR.search(query)
    if m:
        month_num = _MONTHS[m.group(1).lower()]
        year = int(m.group(2))
        from_d, to_d = _month_range(year, month_num)
        cleaned = query[: m.start()] + query[m.end() :]
        return cleaned.strip(), from_d, to_d

    # Try ISO month: "2026-01"
    m = _ISO_MONTH.search(query)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            from_d, to_d = _month_range(year, month)
            cleaned = query[: m.start()] + query[m.end() :]
            return cleaned.strip(), from_d, to_d

    # Yesterday
    m = _YESTERDAY.search(query)
    if m:
        d = _days_ago(1)
        cleaned = query[: m.start()] + query[m.end() :]
        return cleaned.strip(), d, d

    # Today
    m = _TODAY.search(query)
    if m:
        d = date.today().isoformat()
        cleaned = query[: m.start()] + query[m.end() :]
        return cleaned.strip(), d, d

    # Last week
    m = _LAST_WEEK.search(query)
    if m:
        cleaned = query[: m.start()] + query[m.end() :]
        return cleaned.strip(), _days_ago(7), None

    # Last N days/weeks/months
    m = _LAST_N.search(query)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        cleaned = query[: m.start()] + query[m.end() :]
        if unit == "day":
            return cleaned.strip(), _days_ago(n), None
        if unit == "week":
            return cleaned.strip(), _days_ago(n * 7), None
        if unit == "month":
            return cleaned.strip(), _months_ago(n), None

    # Past N ...
    m = _PAST_N.search(query)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        cleaned = query[: m.start()] + query[m.end() :]
        if unit == "day":
            return cleaned.strip(), _days_ago(n), None
        if unit == "week":
            return cleaned.strip(), _days_ago(n * 7), None
        if unit == "month":
            return cleaned.strip(), _months_ago(n), None

    # This year
    m = _THIS_YEAR.search(query)
    if m:
        cleaned = query[: m.start()] + query[m.end() :]
        return cleaned.strip(), f"{date.today().year}-01-01", None

    # This month
    m = _THIS_MONTH.search(query)
    if m:
        cleaned = query[: m.start()] + query[m.end() :]
        today = date.today()
        from_d = f"{today.year}-{today.month:02d}-01"
        return cleaned.strip(), from_d, None

    return query, None, None


# ---------------------------------------------------------------------------
# Compact relative-date resolver for CLI --since flags
# ---------------------------------------------------------------------------

# Matches shorthand like "7d", "2w", "3m" (days, weeks, months)
_SHORTHAND_RE = re.compile(r"^(\d+)\s*([dwm])$", re.IGNORECASE)


def resolve_since(value: str | None) -> str | None:
    """Resolve a --since value to an ISO date string (YYYY-MM-DD).

    Accepts:
    - ISO dates: "2026-03-01" → returned as-is
    - Shorthand: "7d" (days), "2w" (weeks), "3m" (months)

    Returns None if value is None.
    Raises click.BadParameter (if click is available) or ValueError for
    unrecognised formats.
    """
    if value is None:
        return None

    # Already ISO date?
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value

    m = _SHORTHAND_RE.match(value)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "d":
            return _days_ago(n)
        if unit == "w":
            return _days_ago(n * 7)
        if unit == "m":
            return _months_ago(n)

    # Not recognised — raise a helpful error
    msg = f"Unrecognised --since format: {value!r}. Use YYYY-MM-DD or shorthand (7d, 2w, 3m)."
    try:
        import click

        raise click.BadParameter(msg)
    except ImportError:
        raise ValueError(msg) from None
