"""Tests for kb.hotness — frequency x recency score (issue #67)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from kb.hotness import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_HOTNESS_WEIGHT,
    compute_hotness_score,
)


class TestComputeHotnessScore:
    def test_zero_access_is_zero(self):
        """Never-accessed resource gets exactly 0.0 (sentinel for inactive)."""
        assert compute_hotness_score(0, None) == 0.0

    def test_zero_access_with_timestamp_still_zero(self):
        """Even with a timestamp, zero count → zero (count is the primary signal)."""
        today = date.today().isoformat()
        assert compute_hotness_score(0, today) == 0.0

    def test_count_only_no_timestamp_returns_zero(self):
        """Positive count but no timestamp → 0 (cannot decay without an anchor)."""
        assert compute_hotness_score(5, None) == 0.0

    def test_single_access_today_is_moderate(self):
        """One access today: frequency ~0.60 x decay 1.0 ≈ 0.60."""
        today = date.today().isoformat()
        score = compute_hotness_score(1, today)
        assert 0.55 < score < 0.70

    def test_many_accesses_today_saturates_high(self):
        """Twenty accesses today: frequency saturates near 0.95+, decay 1.0."""
        today = date.today().isoformat()
        score = compute_hotness_score(20, today)
        assert score > 0.90

    def test_half_life_halves_score(self):
        """At exactly the half-life, recency decay is 0.5 → hotness halves."""
        days_ago = int(DEFAULT_HALF_LIFE_DAYS)
        ts = (date.today() - timedelta(days=days_ago)).isoformat()
        today_score = compute_hotness_score(10, date.today().isoformat())
        past_score = compute_hotness_score(10, ts)
        assert past_score == pytest.approx(today_score * 0.5, rel=0.05)

    def test_old_access_decays_to_near_zero(self):
        """After 4 half-lives, the recency component is ~0.06 of peak."""
        days_ago = int(DEFAULT_HALF_LIFE_DAYS * 4)
        ts = (date.today() - timedelta(days=days_ago)).isoformat()
        assert compute_hotness_score(10, ts) < 0.10

    def test_returns_in_unit_interval(self):
        """Output is always in [0, 1] for any sensible input."""
        today = date.today().isoformat()
        for n in (0, 1, 5, 20, 100, 10_000):
            score = compute_hotness_score(n, today)
            assert 0.0 <= score <= 1.0

    def test_iso_timestamp_with_tz_parses(self):
        """ISO timestamps with timezone suffix (T HH:MM:SS+00:00) are accepted."""
        score = compute_hotness_score(5, "2026-05-24T12:00:00+00:00")
        # Same-day score; should be > 0
        assert score > 0.0

    def test_iso_timestamp_with_z_parses(self):
        """ISO timestamps with 'Z' suffix (UTC shorthand) are accepted."""
        score = compute_hotness_score(5, "2026-05-24T12:00:00Z")
        assert score > 0.0

    def test_iso_timestamp_no_colon_tz_parses(self):
        """ISO timestamps with no-colon TZ (`+0000`) parse cleanly on Python 3.10+.

        Regression: Python 3.10's ``datetime.fromisoformat`` rejected the
        no-colon TZ format that ``strftime('%z')`` produces. We now slice to
        the date portion only, so the TZ format is irrelevant.
        """
        score = compute_hotness_score(5, "2026-05-24T12:00:00+0000")
        assert score > 0.0

    def test_malformed_timestamp_returns_zero(self):
        """An unparseable timestamp degrades gracefully to 0.0."""
        assert compute_hotness_score(5, "not-a-date") == 0.0

    def test_custom_half_life_changes_decay(self):
        """A shorter half-life makes scores decay faster."""
        ts = (date.today() - timedelta(days=7)).isoformat()
        slow = compute_hotness_score(10, ts, half_life_days=30.0)
        fast = compute_hotness_score(10, ts, half_life_days=3.0)
        assert fast < slow


class TestDefaults:
    def test_half_life_default(self):
        assert DEFAULT_HALF_LIFE_DAYS == 7.0

    def test_hotness_weight_default(self):
        """20% hotness, 80% relevance (per issue #67 design)."""
        assert DEFAULT_HOTNESS_WEIGHT == 0.2
