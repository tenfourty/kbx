"""Tests for kb.scoring — score normalisation + composition (issue #95)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from kb.scoring import (
    ENTITY_BOOST,
    apply_entity_boost,
    compose_scores,
    normalise_vector_distance,
    normalize_bm25_scores,
    recency_weight,
    rrf_score,
)


class TestNormalizeBm25Scores:
    def test_most_relevant_gets_1(self):
        """Most-negative raw BM25 (most relevant) normalises to 1.0."""
        results = [{"raw_bm25": -10.0}, {"raw_bm25": -5.0}, {"raw_bm25": -1.0}]
        normalize_bm25_scores(results)
        assert results[0]["bm25_score"] == 1.0

    def test_least_relevant_gets_0(self):
        """Least-negative raw BM25 normalises to 0.0."""
        results = [{"raw_bm25": -10.0}, {"raw_bm25": -5.0}, {"raw_bm25": -1.0}]
        normalize_bm25_scores(results)
        assert results[2]["bm25_score"] == 0.0

    def test_single_result_gets_1(self):
        """A single result is normalised to 1.0 (no spread to divide by)."""
        results = [{"raw_bm25": -3.0}]
        normalize_bm25_scores(results)
        assert results[0]["bm25_score"] == 1.0

    def test_all_equal_get_1(self):
        """When all raw scores are equal, all results get 1.0 (defensive)."""
        results = [{"raw_bm25": -5.0}, {"raw_bm25": -5.0}, {"raw_bm25": -5.0}]
        normalize_bm25_scores(results)
        for r in results:
            assert r["bm25_score"] == 1.0

    def test_empty_does_not_raise(self):
        normalize_bm25_scores([])

    def test_output_range(self):
        """Every normalised score lands in [0, 1]."""
        results = [{"raw_bm25": v} for v in [-100, -10, -1, -0.01]]
        normalize_bm25_scores(results)
        for r in results:
            assert 0.0 <= r["bm25_score"] <= 1.0


class TestRrfScore:
    def test_rank_1(self):
        assert rrf_score(1, k=60) == pytest.approx(1.0 / 61)

    def test_decreasing(self):
        assert rrf_score(1) > rrf_score(2) > rrf_score(10)

    def test_default_k(self):
        """Default k=60 matches the original RRF paper."""
        assert rrf_score(1) == pytest.approx(1.0 / 61)


class TestRecencyWeight:
    def test_today_is_near_one(self):
        today = date.today().isoformat()
        assert recency_weight(today) > 0.99

    def test_half_life_decay(self):
        """At exactly the half-life, weight is ~0.5."""
        d = (date.today() - timedelta(days=90)).isoformat()
        assert 0.45 < recency_weight(d, half_life_days=90) < 0.55

    def test_old_date_decays(self):
        """A year-old document decays well below 0.1."""
        assert recency_weight("2025-01-01") < 0.1

    def test_none_returns_neutral_half(self):
        assert recency_weight(None) == 0.5

    def test_unparseable_returns_neutral_half(self):
        assert recency_weight("not-a-date") == 0.5

    def test_future_date_is_full(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        assert recency_weight(future) == 1.0


class TestApplyEntityBoost:
    def test_boost_when_match(self):
        assert apply_entity_boost(1.0, has_entity_match=True) == ENTITY_BOOST

    def test_no_boost_when_no_match(self):
        assert apply_entity_boost(1.0, has_entity_match=False) == 1.0

    def test_boost_constant(self):
        """ENTITY_BOOST is a 15% multiplier."""
        assert ENTITY_BOOST == 1.15


class TestNormaliseVectorDistance:
    def test_zero_distance_is_one(self):
        """Distance 0 (identical vectors) → similarity 1.0."""
        assert normalise_vector_distance(0.0) == 1.0

    def test_distance_one_is_zero(self):
        """Distance 1.0 (orthogonal for L2-normalised cosine) → similarity 0.0."""
        assert normalise_vector_distance(1.0) == 0.0

    def test_distance_half_is_half(self):
        """Distance 0.5 → similarity 0.5 (linear mapping)."""
        assert normalise_vector_distance(0.5) == 0.5

    def test_clamps_negative(self):
        """Floating-point drift below 0 is clamped to 1.0 (identical)."""
        assert normalise_vector_distance(-0.001) == 1.0

    def test_clamps_above_one(self):
        """Distance > 1 (extreme drift) clamps to 0.0."""
        assert normalise_vector_distance(2.0) == 0.0

    def test_output_in_unit_interval(self):
        """Every output is in [0, 1] for inputs across the natural range."""
        for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
            v = normalise_vector_distance(d)
            assert 0.0 <= v <= 1.0


class TestComposeScores:
    def test_empty_weights_returns_zero(self):
        assert compose_scores({"fts": 0.5}, {}) == 0.0

    def test_weighted_sum(self):
        """Equal-weight composition averages the components."""
        result = compose_scores({"a": 0.4, "b": 0.8}, {"a": 0.5, "b": 0.5})
        assert result == pytest.approx(0.6)

    def test_unequal_weights(self):
        """Weights bias the result toward the higher-weighted component."""
        result = compose_scores(
            {"fts": 0.0, "vector": 1.0},
            {"fts": 0.3, "vector": 0.7},
        )
        assert result == pytest.approx(0.7)

    def test_weights_normalised_to_sum_one(self):
        """Weights that don't sum to 1.0 are renormalised — same effective composition."""
        # Weights (0.2, 0.2) renormalise to (0.5, 0.5), same as equal weighting.
        result = compose_scores({"a": 0.4, "b": 0.8}, {"a": 0.2, "b": 0.2})
        assert result == pytest.approx(0.6)

    def test_missing_component_treated_as_zero(self):
        """Weight keys without a component contribute nothing to the sum."""
        result = compose_scores(
            {"fts": 1.0},  # vector missing
            {"fts": 0.5, "vector": 0.5},
        )
        # renormalised weights are (0.5, 0.5); missing vector = 0 → 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert result == pytest.approx(0.5)

    def test_convex_combination_property(self):
        """Output is in [min(components), max(components)] when components ⊆ [0, 1]."""
        components = {"a": 0.2, "b": 0.6, "c": 0.9}
        weights = {"a": 0.3, "b": 0.3, "c": 0.4}
        result = compose_scores(components, weights)
        assert min(components.values()) <= result <= max(components.values())

    def test_all_zero_weights_returns_zero(self):
        """Defensive: zero/negative-total weights short-circuit to 0."""
        assert compose_scores({"a": 1.0}, {"a": 0.0}) == 0.0

    def test_output_in_unit_interval_for_unit_inputs(self):
        """When every component is in [0, 1], the weighted sum is in [0, 1]."""
        for components in [
            {"a": 0.0, "b": 1.0},
            {"a": 0.5, "b": 0.5},
            {"a": 1.0, "b": 1.0},
            {"a": 0.123, "b": 0.876},
        ]:
            result = compose_scores(components, {"a": 0.5, "b": 0.5})
            assert 0.0 <= result <= 1.0
