"""Tests for date extraction from search queries."""

from __future__ import annotations

from datetime import date, timedelta


class TestMonthYearExtraction:
    def test_january_2026(self):
        from kb.dateparse import extract_date_filters

        query, from_d, to_d = extract_date_filters("meetings January 2026")
        assert query.strip() == "meetings"
        assert from_d == "2026-01-01"
        assert to_d == "2026-01-31"

    def test_jan_2026(self):
        from kb.dateparse import extract_date_filters

        query, from_d, to_d = extract_date_filters("notes Jan 2026")
        assert query.strip() == "notes"
        assert from_d == "2026-01-01"
        assert to_d == "2026-01-31"

    def test_iso_month(self):
        from kb.dateparse import extract_date_filters

        query, from_d, to_d = extract_date_filters("meetings 2026-01")
        assert query.strip() == "meetings"
        assert from_d == "2026-01-01"
        assert to_d == "2026-01-31"

    def test_february_end_of_month(self):
        from kb.dateparse import extract_date_filters

        _query, from_d, to_d = extract_date_filters("February 2026")
        assert from_d == "2026-02-01"
        assert to_d == "2026-02-28"


class TestRelativeDates:
    def test_last_week(self):
        from kb.dateparse import extract_date_filters

        query, from_d, _to_d = extract_date_filters("meetings last week")
        assert query.strip() == "meetings"
        assert from_d is not None
        expected = date.today().toordinal() - 7
        assert date.fromisoformat(from_d).toordinal() == expected

    def test_past_3_months(self):
        from kb.dateparse import extract_date_filters

        query, from_d, to_d = extract_date_filters("updates past 3 months")
        assert query.strip() == "updates"
        assert from_d is not None
        assert to_d is None

    def test_this_year(self):
        from kb.dateparse import extract_date_filters

        query, from_d, _to_d = extract_date_filters("projects this year")
        assert query.strip() == "projects"
        assert from_d == f"{date.today().year}-01-01"

    def test_yesterday(self):
        from kb.dateparse import extract_date_filters

        query, from_d, to_d = extract_date_filters("meetings yesterday")
        assert query.strip() == "meetings"
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        assert from_d == yesterday
        assert to_d == yesterday

    def test_last_30_days(self):
        from kb.dateparse import extract_date_filters

        query, from_d, _to_d = extract_date_filters("search last 30 days")
        assert query.strip() == "search"
        assert from_d is not None


class TestNoDateInQuery:
    def test_no_date_returns_none(self):
        from kb.dateparse import extract_date_filters

        query, from_d, to_d = extract_date_filters("Helix refactor status")
        assert query == "Helix refactor status"
        assert from_d is None
        assert to_d is None

    def test_bare_number_not_a_date(self):
        from kb.dateparse import extract_date_filters

        _query, from_d, _to_d = extract_date_filters("top 5 issues")
        assert from_d is None


class TestDateparseEdgeCases:
    def test_today(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, to_d = extract_date_filters("meetings today")
        assert from_d is not None
        assert from_d == to_d
        assert from_d == date.today().isoformat()

    def test_this_month(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, to_d = extract_date_filters("meetings this month")
        assert from_d is not None
        today = date.today()
        assert from_d == f"{today.year}-{today.month:02d}-01"
        assert to_d is None

    def test_past_3_days(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, to_d = extract_date_filters("meetings past 3 days")
        assert from_d is not None
        assert from_d == (date.today() - timedelta(days=3)).isoformat()
        assert to_d is None

    def test_past_2_weeks(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, _to_d = extract_date_filters("notes past 2 weeks")
        assert from_d is not None
        assert from_d == (date.today() - timedelta(days=14)).isoformat()

    def test_past_6_months(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, to_d = extract_date_filters("events past 6 months")
        assert from_d is not None
        assert to_d is None

    def test_last_3_days(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, _to_d = extract_date_filters("search last 3 days")
        assert from_d is not None
        assert from_d == (date.today() - timedelta(days=3)).isoformat()

    def test_last_2_weeks(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, _to_d = extract_date_filters("search last 2 weeks")
        assert from_d is not None
        assert from_d == (date.today() - timedelta(days=14)).isoformat()

    def test_last_3_months(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, _to_d = extract_date_filters("search last 3 months")
        assert from_d is not None

    def test_iso_month_invalid_13(self):
        from kb.dateparse import extract_date_filters

        _q, from_d, _to_d = extract_date_filters("meetings 2026-13")
        # Month 13 is invalid -- should not match as an ISO month
        assert from_d is None

    def test_months_ago_wraps_year(self):
        """Subtracting months that crosses a year boundary."""
        from kb.dateparse import _months_ago

        # This just verifies _months_ago doesn't crash
        result = _months_ago(15)  # 15 months ago crosses at least one year boundary
        date.fromisoformat(result)  # Should be a valid date

    def test_months_ago_clamps_day(self):
        """If current day > days in target month, it should clamp."""
        from kb.dateparse import _months_ago

        # Just verify it returns a valid date regardless
        result = _months_ago(1)
        date.fromisoformat(result)


class TestResolveSince:
    def test_none_returns_none(self):
        from kb.dateparse import resolve_since

        assert resolve_since(None) is None

    def test_iso_date_passthrough(self):
        from kb.dateparse import resolve_since

        assert resolve_since("2026-03-01") == "2026-03-01"

    def test_7d_resolves_to_iso(self):
        from kb.dateparse import resolve_since

        result = resolve_since("7d")
        assert result == (date.today() - timedelta(days=7)).isoformat()

    def test_2w_resolves_to_iso(self):
        from kb.dateparse import resolve_since

        result = resolve_since("2w")
        assert result == (date.today() - timedelta(days=14)).isoformat()

    def test_3m_resolves_to_iso(self):
        from kb.dateparse import resolve_since

        result = resolve_since("3m")
        assert result is not None
        date.fromisoformat(result)  # valid ISO date

    def test_uppercase_7D(self):
        from kb.dateparse import resolve_since

        result = resolve_since("7D")
        assert result == (date.today() - timedelta(days=7)).isoformat()

    def test_invalid_raises(self):
        import pytest

        from kb.dateparse import resolve_since

        with pytest.raises(Exception, match="Unrecognised --since format"):
            resolve_since("banana")
