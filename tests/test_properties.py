"""Property-based tests using Hypothesis for pure functions."""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st


class TestBM25NormalizationProperties:
    @given(scores=st.lists(st.floats(min_value=-100, max_value=0), min_size=1, max_size=50))
    def test_scores_always_in_unit_range(self, scores):
        from kb.search import _normalize_bm25_scores

        results = [{"raw_bm25": s} for s in scores]
        _normalize_bm25_scores(results)
        for r in results:
            assert 0.0 <= r["bm25_score"] <= 1.0

    @given(scores=st.lists(st.floats(min_value=-100, max_value=0), min_size=2, max_size=50))
    def test_best_score_is_one(self, scores):
        from kb.search import _normalize_bm25_scores

        assume(min(scores) != max(scores))
        results = [{"raw_bm25": s} for s in scores]
        _normalize_bm25_scores(results)
        best_idx = scores.index(min(scores))
        assert results[best_idx]["bm25_score"] == 1.0


class TestRecencyWeightProperties:
    @given(days=st.integers(min_value=0, max_value=3650))
    def test_recency_always_in_zero_one(self, days):
        from datetime import date, timedelta

        from kb.search import _recency_weight

        d = (date.today() - timedelta(days=days)).isoformat()
        w = _recency_weight(d)
        assert 0.0 < w <= 1.0


class TestRRFProperties:
    @given(rank=st.integers(min_value=0, max_value=1000))
    def test_rrf_always_positive(self, rank):
        from kb.search import _rrf_score

        assert _rrf_score(rank) > 0

    @given(
        r1=st.integers(min_value=0, max_value=999),
        r2=st.integers(min_value=0, max_value=999),
    )
    def test_rrf_monotonically_decreasing(self, r1, r2):
        from kb.search import _rrf_score

        assume(r1 < r2)
        assert _rrf_score(r1) > _rrf_score(r2)


class TestDateFilterProperties:
    @given(
        month=st.sampled_from(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        ),
        year=st.integers(min_value=2000, max_value=2030),
    )
    def test_month_year_always_extracts(self, month, year):
        from kb.dateparse import extract_date_filters

        query = f"meetings {month} {year}"
        _cleaned, from_d, to_d = extract_date_filters(query)
        assert from_d is not None
        assert to_d is not None
        assert from_d < to_d
        assert from_d.startswith(str(year))

    @given(n=st.integers(min_value=1, max_value=36))
    def test_last_n_months_valid_date(self, n):
        from datetime import date as date_cls

        from kb.dateparse import extract_date_filters

        query = f"meetings last {n} months"
        _cleaned, from_d, _to_d = extract_date_filters(query)
        assert from_d is not None
        date_cls.fromisoformat(from_d)  # Should not raise

    @given(n=st.integers(min_value=1, max_value=365))
    def test_past_n_days_valid(self, n):
        from datetime import date as date_cls

        from kb.dateparse import extract_date_filters

        query = f"events past {n} days"
        _cleaned, from_d, _to_d = extract_date_filters(query)
        assert from_d is not None
        date_cls.fromisoformat(from_d)
