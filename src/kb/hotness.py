"""Hotness/recency scoring for frequently-accessed documents and entities (#67).

Frequently-viewed documents and entities surface higher in search rankings via
a hotness boost composed with the existing relevance score. The formula is:

    hotness = sigmoid(log1p(access_count)) * exp_decay(age_days, half_life)

- Frequency component: ``sigmoid(log1p(n))`` rises steeply for the first few
  accesses and saturates around 0.95+ at 20+ accesses (diminishing returns).
- Recency component: standard exponential decay with a configurable half-life
  (default 7 days, matching the issue #67 design).

A document accessed once today: ~0.60. Ten times today: ~0.92. Ten times a
week ago: ~0.46. Never accessed: 0.0 (the frequency component is 0.5 for
``access_count=0`` but the decay term is 0 since ``last_accessed_at`` is None).
"""

from __future__ import annotations

import math
from datetime import date

# Defaults — match the design in issue #67. Tunable via search() kwargs later.
DEFAULT_HALF_LIFE_DAYS = 7.0
DEFAULT_HOTNESS_WEIGHT = 0.2  # 20% hotness, 80% relevance in the blended score


def compute_hotness_score(
    access_count: int,
    last_accessed_at: str | None,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Compute hotness in ``[0, 1]`` from access frequency + recency.

    Args:
        access_count: Number of times the resource has been viewed/looked up.
        last_accessed_at: ISO-8601 timestamp (``YYYY-MM-DDTHH:MM:SS[+TZ]``)
            of the most recent access, or ``None`` if never accessed.
        half_life_days: Days after which the recency component halves.

    Returns:
        Hotness score in ``[0, 1]``. ``0.0`` when never accessed.
    """
    if access_count <= 0 or last_accessed_at is None:
        return 0.0

    # Frequency: sigmoid(log1p(n)) — rises quickly, saturates near 1.0
    frequency = 1.0 / (1.0 + math.exp(-math.log1p(access_count)))

    # Recency: exponential decay since last access
    try:
        accessed = _parse_iso_to_date(last_accessed_at)
    except ValueError:
        return 0.0

    days_ago = max(0, (date.today() - accessed).days)
    decay = math.exp(-math.log(2) * days_ago / half_life_days)

    return frequency * decay


def _parse_iso_to_date(value: str) -> date:
    """Parse a date or ISO-8601 timestamp string into a ``date`` (day precision).

    Hotness operates at day precision, so the time/timezone portion is
    discarded — taking only the leading ``YYYY-MM-DD`` slice. This avoids
    Python 3.10's stricter ``datetime.fromisoformat`` which rejects the
    ``+0000`` (no-colon) TZ format that ``strftime('%z')`` emits on 3.10
    but accepts on 3.11+.
    """
    return date.fromisoformat(value[:10])
