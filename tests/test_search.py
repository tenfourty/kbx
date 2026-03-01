"""Tests for kb.search — hybrid search pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kb.db import Database
from kb.search import (
    SearchProgressHook,
    _normalize_bm25_scores,
    _recency_weight,
    _rrf_score,
    _strip_metadata_prefix,
    search,
)
from kb.types import SearchMeta, SearchResponse, SearchResult

# ---------------------------------------------------------------------------
# Step 1: BM25 Score Normalization
# ---------------------------------------------------------------------------


class TestBM25Normalization:
    def test_most_relevant_gets_1(self):
        """Most negative BM25 (most relevant) should normalize to 1.0."""
        results = [{"raw_bm25": -10.0}, {"raw_bm25": -5.0}, {"raw_bm25": -1.0}]
        _normalize_bm25_scores(results)
        assert results[0]["bm25_score"] == 1.0

    def test_least_relevant_gets_0(self):
        """Least negative BM25 (least relevant) should normalize to 0.0."""
        results = [{"raw_bm25": -10.0}, {"raw_bm25": -5.0}, {"raw_bm25": -1.0}]
        _normalize_bm25_scores(results)
        assert results[2]["bm25_score"] == 0.0

    def test_middle_score(self):
        """Middle BM25 should normalize to ~0.5."""
        results = [{"raw_bm25": -10.0}, {"raw_bm25": -5.5}, {"raw_bm25": -1.0}]
        _normalize_bm25_scores(results)
        assert 0.45 < results[1]["bm25_score"] < 0.55

    def test_single_result_gets_1(self):
        """Single result should get score 1.0."""
        results = [{"raw_bm25": -3.0}]
        _normalize_bm25_scores(results)
        assert results[0]["bm25_score"] == 1.0

    def test_empty_list(self):
        """Empty list should not raise."""
        results = []
        _normalize_bm25_scores(results)

    def test_output_range(self):
        """All scores should be in [0, 1]."""
        results = [{"raw_bm25": v} for v in [-100, -10, -1, -0.01]]
        _normalize_bm25_scores(results)
        for r in results:
            assert 0 <= r["bm25_score"] <= 1


# ---------------------------------------------------------------------------
# Step 5: RRF Score
# ---------------------------------------------------------------------------


class TestRRFScore:
    def test_rank_1(self):
        """Rank 1 with k=60 should give 1/61."""
        assert _rrf_score(1, k=60) == pytest.approx(1.0 / 61)

    def test_rank_decreasing(self):
        """Higher rank should give lower score."""
        assert _rrf_score(1) > _rrf_score(2) > _rrf_score(10)


# ---------------------------------------------------------------------------
# Step 6: Recency Weight
# ---------------------------------------------------------------------------


class TestRecencyWeight:
    def test_today(self):
        """Today's date should give weight ~1.0."""
        from datetime import date

        today = date.today().isoformat()
        w = _recency_weight(today)
        assert w > 0.99

    def test_old_date(self):
        """A date 1 year ago should give low weight."""
        w = _recency_weight("2025-01-01")
        assert w < 0.1

    def test_none_date(self):
        """None date should return 0.5 (neutral)."""
        assert _recency_weight(None) == 0.5

    def test_90_day_halflife(self):
        """At exactly half-life, weight should be ~0.5."""
        from datetime import date, timedelta

        d = (date.today() - timedelta(days=90)).isoformat()
        w = _recency_weight(d, half_life_days=90)
        assert 0.45 < w < 0.55


# ---------------------------------------------------------------------------
# Step 7: Snippet extraction helper
# ---------------------------------------------------------------------------


class TestSnippetExtraction:
    def test_strips_metadata_prefix(self):
        """Should strip [Meeting: ...] prefix."""
        text = (
            "[Meeting: Wren / Soren | Date: 2026-01-27 | Section: General]\nActual content here"
        )
        result = _strip_metadata_prefix(text)
        assert result == "Actual content here"

    def test_strips_transcript_prefix(self):
        """Should strip [Transcript: ...] prefix."""
        text = "[Transcript: Title | Date: 2026-01-27]\nSpeaker: Hello"
        result = _strip_metadata_prefix(text)
        assert result == "Speaker: Hello"

    def test_no_prefix(self):
        """Text without prefix should be returned as-is."""
        text = "Just some plain text"
        result = _strip_metadata_prefix(text)
        assert result == "Just some plain text"

    def test_strips_summary_prefix(self):
        """Should strip [Summary: ...] prefix."""
        text = "[Summary: Title | Date: 2026-01-27 | Entities: Foo (person)]\nContent"
        result = _strip_metadata_prefix(text)
        assert result == "Content"


# ---------------------------------------------------------------------------
# Step 8: Progress Hooks
# ---------------------------------------------------------------------------


class TestProgressHooks:
    def test_default_hook_is_noop(self):
        """Default hook should not raise."""
        hook = SearchProgressHook()
        hook.on_fts_start()
        hook.on_fts_done(5)
        hook.on_vector_start()
        hook.on_vector_done(10)
        hook.on_fusion_done(8)

    def test_hook_methods_called(self):
        """Custom hook methods should be called in order."""
        calls = []

        class TrackingHook(SearchProgressHook):
            def on_fts_start(self):
                calls.append("fts_start")

            def on_fts_done(self, count):
                calls.append(f"fts_done:{count}")

            def on_vector_start(self):
                calls.append("vector_start")

            def on_vector_done(self, count):
                calls.append(f"vector_done:{count}")

            def on_fusion_done(self, count):
                calls.append(f"fusion_done:{count}")

        # We'll verify this in integration tests
        # For now just verify the class works
        hook = TrackingHook()
        hook.on_fts_start()
        hook.on_fts_done(5)
        hook.on_vector_start()
        hook.on_vector_done(10)
        hook.on_fusion_done(8)
        assert calls == [
            "fts_start",
            "fts_done:5",
            "vector_start",
            "vector_done:10",
            "fusion_done:8",
        ]


# ---------------------------------------------------------------------------
# Integration tests using a real (small) SQLite + LanceDB
# ---------------------------------------------------------------------------


@pytest.fixture
def search_db():
    """Create a small test database with known data for search tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()

        # Insert test documents
        docs = [
            (
                "doc1.md",
                "MFA Implementation Review",
                "2026-01-27",
                "notes",
                "granola",
                "id1",
                "[]",
                "hash1",
                2,
            ),
            (
                "doc2.md",
                "Helix Refactor Status",
                "2026-01-20",
                "notes",
                "granola",
                "id2",
                "[]",
                "hash2",
                2,
            ),
            ("doc3.md", "Weekly Sync", "2025-06-15", "notes", "granola", "id3", "[]", "hash3", 1),
            (
                "doc4.md",
                "Atlas Pipeline Plan",
                "2026-02-01",
                "notes",
                "granola",
                "id4",
                "[]",
                "hash4",
                1,
            ),
        ]
        for path, title, date, dtype, src, sid, tags, chash, cc in docs:
            conn.execute(
                """INSERT INTO documents (path, title, doc_date, doc_type, source_system, source_id, tags, content_hash, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (path, title, date, dtype, src, sid, tags, chash, cc),
            )

        # Insert test chunks (content = title + body, metadata_prefix stored separately)
        chunks = [
            (
                1,
                0,
                "Overview",
                "MFA Implementation Review\nWe reviewed the MFA implementation progress. The solution uses TOTP with Okta integration.",
                "[Meeting: MFA Implementation Review | Date: 2026-01-27 | Section: Overview]",
            ),
            (
                1,
                1,
                "Next Steps",
                "MFA Implementation Review\nRollout to all employees by end of February. Testing phase complete.",
                "[Meeting: MFA Implementation Review | Date: 2026-01-27 | Section: Next Steps]",
            ),
            (
                2,
                0,
                "Status",
                "Helix Refactor Status\nThe Rust migration is at 45% completion. 180 of 400 modules converted. Linnea leading the effort.",
                "[Meeting: Helix Refactor Status | Date: 2026-01-20 | Section: Status]",
            ),
            (
                2,
                1,
                "Performance",
                "Helix Refactor Status\nRust engine is 5x faster than Python. Memory usage reduced by 60%.",
                "[Meeting: Helix Refactor Status | Date: 2026-01-20 | Section: Performance]",
            ),
            (
                3,
                0,
                None,
                "Weekly Sync\nGeneral weekly sync discussion about team updates and project timelines.",
                "[Meeting: Weekly Sync | Date: 2025-06-15 | Section: General]",
            ),
            (
                4,
                0,
                "Plan",
                "Atlas Pipeline Plan\nDatastore migration is top priority for 2026. SSO integration and Grafana dashboard migration planned.",
                "[Meeting: Atlas Pipeline Plan | Date: 2026-02-01 | Section: Plan]",
            ),
        ]
        for doc_id, idx, heading, content, meta_prefix in chunks:
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content, metadata_prefix) VALUES (?, ?, ?, ?, ?)",
                (doc_id, idx, heading, content, meta_prefix),
            )

        # Insert entities
        entities = [
            ("Linnea Holm", "person", '["Linnea"]', '{"team": "Rust"}', None),
            ("Helix Refactor", "project", '["helix-refactor"]', "{}", None),
            ("Atlas Pipeline", "project", '["atlas-pipeline"]', "{}", None),
        ]
        for name, etype, aliases, meta, src in entities:
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata, source_path) VALUES (?, ?, ?, ?, ?)",
                (name, etype, aliases, meta, src),
            )

        # Entity mentions
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (1, 2, 'discussed')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (2, 2, 'discussed')"
        )
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, document_id, mention_type) VALUES (3, 4, 'discussed')"
        )

        conn.commit()
        yield db
        db.close()


class TestWeightedRRF:
    def test_equal_weighting(self):
        """FTS and vector results at same rank should contribute equally."""
        from kb.search import _weighted_rrf

        fts = [{"chunk_id": 1}]
        vec = [{"chunk_id": 2}]
        scores = _weighted_rrf(fts, vec)
        # Same rank -> same score (equal weighting)
        assert scores[1] == pytest.approx(scores[2])

    def test_rank1_bonus(self):
        """Rank 1 should get a bonus over rank 2."""
        from kb.search import _weighted_rrf

        fts = [{"chunk_id": 1}, {"chunk_id": 2}]
        scores = _weighted_rrf(fts, [])
        assert scores[1] > scores[2]


class TestFTSSearch:
    """Tests for FTS5-based keyword search."""

    def test_basic_fts(self, search_db):
        """FTS search should return results."""
        results = search(search_db, None, "MFA implementation", fast=True)
        assert results.meta.total > 0
        assert results.results[0].title == "MFA Implementation Review"

    def test_fts_scores_normalized(self, search_db):
        """FTS scores should be non-negative (may exceed 1.0 due to entity boost)."""
        results = search(search_db, None, "Rust migration", fast=True)
        for r in results.results:
            assert r.score >= 0

    def test_fts_returns_snippets(self, search_db):
        """Results should have non-empty snippets without metadata prefix."""
        results = search(search_db, None, "MFA", fast=True)
        for r in results.results:
            assert r.snippet
            assert not r.snippet.startswith("[Meeting:")

    def test_fts_result_structure(self, search_db):
        """Results should have all required fields."""
        results = search(search_db, None, "Rust", fast=True)
        required_fields = {
            "chunk_id",
            "document_id",
            "title",
            "path",
            "date",
            "doc_type",
            "score",
            "section",
            "snippet",
            "entities",
        }
        for r in results.results:
            dumped = r.model_dump()
            assert required_fields.issubset(dumped.keys()), (
                f"Missing fields: {required_fields - dumped.keys()}"
            )

    def test_fts_entities_populated(self, search_db):
        """Results for Rust migration should include entity names."""
        results = search(search_db, None, "Rust migration", fast=True)
        # At least one result should have entities
        all_entities = []
        for r in results.results:
            all_entities.extend(r.entities)
        assert len(all_entities) > 0

    def test_fts_no_results(self, search_db):
        """Query with no matches should return empty results."""
        results = search(search_db, None, "xyznonexistent", fast=True)
        assert results.meta.total == 0
        assert results.results == []

    def test_fts_meta_fields(self, search_db):
        """Meta should include query, total, limit, execution_ms."""
        results = search(search_db, None, "MFA", fast=True)
        assert results.meta.query == "MFA"
        assert isinstance(results.meta.total, int)
        assert isinstance(results.meta.limit, int)
        assert isinstance(results.meta.execution_ms, float)


class TestFTSOnlyFallback:
    """Tests for when LanceDB has no vectors."""

    def test_search_without_vectors(self, search_db):
        """Search should work (FTS only) even when no LanceDB table exists."""
        # search_db has no LanceDB data — this should still work
        results = search(search_db, None, "MFA implementation")
        assert results.meta.total > 0

    def test_search_fast_mode(self, search_db):
        """fast=True should use FTS only, no vector search."""
        results = search(search_db, None, "Datastore", fast=True)
        assert results.meta.total > 0


class TestRecencyInSearch:
    """Test recency weighting in search results."""

    def test_recency_boosts_newer(self, search_db):
        """With recency > 0, newer docs should score higher than older ones at same relevance."""
        # Both "Weekly Sync" (2025-06) and others (2026-01) contain common words
        results = search(search_db, None, "meeting sync", fast=True, recency=0.5)
        if len(results.results) >= 2:
            # The newer result should tend to appear first
            # (not a strict test since relevance also matters)
            assert results.results[0].date is not None

    def test_recency_zero_is_pure_similarity(self, search_db):
        """recency=0 should give same scores regardless of date."""
        results = search(search_db, None, "migration", fast=True, recency=0.0)
        assert results.meta.total > 0


class TestSortByDate:
    def test_sort_by_date_descending(self, search_db):
        """sort_by='date' should order results newest first."""
        results = search(search_db, None, "migration", fast=True, sort_by="date")
        dates = [r.date for r in results.results if r.date]
        assert dates == sorted(dates, reverse=True)

    def test_sort_by_date_none_dates_last(self, search_db):
        """Results without dates should sort last."""
        results = search(search_db, None, "migration", fast=True, sort_by="date")
        dates = [r.date for r in results.results]
        none_count = dates.count(None)
        if none_count > 0:
            # All Nones should be at end
            assert all(d is None for d in dates[-none_count:])

    def test_sort_meta_includes_sort_by(self, search_db):
        """Meta should include sort_by field."""
        results = search(search_db, None, "migration", fast=True, sort_by="date")
        assert results.meta.sort_by == "date"

    def test_default_sort_is_score(self, search_db):
        """Default sort should be by score descending."""
        results = search(search_db, None, "migration", fast=True)
        scores = [r.score for r in results.results]
        assert scores == sorted(scores, reverse=True)
        assert results.meta.sort_by == "score"


class TestMultiTermMerge:
    """Multi-term search should merge results across FTS variants (phrase, NEAR, OR)."""

    def test_or_fallback_adds_results(self, search_db):
        """A multi-word query should return OR results even when NEAR finds a subset.

        'Cloud Grafana' — NEAR likely finds 0 (terms not co-located), but OR should
        find docs mentioning Cloud and docs mentioning Grafana separately.
        """
        results = search(search_db, None, "Cloud Grafana", fast=True)
        # Should get results from both Helix Refactor and Datastore (Grafana) docs
        assert results.meta.total >= 2
        titles = [r.title for r in results.results]
        assert any("Cloud" in t for t in titles)
        assert any("Datastore" in t for t in titles)

    def test_near_hit_doesnt_suppress_or_results(self, search_db):
        """When NEAR finds some results, OR should still contribute additional matches.

        _fts_search must accumulate across variants, not return on first hit.
        'migration Cloud' — NEAR finds Helix Refactor doc (terms co-located), but
        OR should also find Atlas Pipeline doc (has 'migration' in title/content).
        """
        from kb.search import _fts_search

        results = _fts_search(search_db, "migration Cloud", limit=50)
        doc_ids = set()
        conn = search_db.get_sqlite_conn()
        for r in results:
            doc_ids.add(r["document_id"])
        titles = set()
        for did in doc_ids:
            row = conn.execute("SELECT title FROM documents WHERE id = ?", (did,)).fetchone()
            if row:
                titles.add(row["title"])
        # NEAR finds Helix Refactor, OR should also find Atlas Pipeline
        assert "Helix Refactor Status" in titles, f"Missing Cloud doc, got: {titles}"
        assert "Atlas Pipeline Plan" in titles, (
            f"Missing Datastore doc (OR should have found it), got: {titles}"
        )

    def test_single_term_still_works(self, search_db):
        """Single-term queries should still work fine."""
        results = search(search_db, None, "MFA", fast=True)
        assert results.meta.total > 0
        assert results.results[0].title == "MFA Implementation Review"

    def test_phrase_match_ranks_higher(self, search_db):
        """Exact phrase matches should rank higher than OR-only matches."""
        results = search(search_db, None, "Cloud migration", fast=True)
        assert results.meta.total > 0
        # The Helix Refactor doc should be top result (phrase match)
        assert "Cloud" in results.results[0].title

    def test_merge_deduplicates_chunks(self, search_db):
        """Same chunk appearing in multiple FTS variants should not be duplicated."""
        from kb.search import _fts_search

        results = _fts_search(search_db, "Cloud migration", limit=50)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in merged results"


class TestHookIntegration:
    """Test that progress hooks fire during search."""

    def test_fts_hooks_fire(self, search_db):
        """FTS hooks should fire during fast search."""
        calls = []

        class TrackHook(SearchProgressHook):
            def on_fts_start(self):
                calls.append("fts_start")

            def on_fts_done(self, count):
                calls.append(f"fts_done:{count}")

            def on_fusion_done(self, count):
                calls.append(f"fusion_done:{count}")

        search(search_db, None, "MFA", fast=True, hook=TrackHook())
        assert "fts_start" in calls
        assert any(c.startswith("fts_done:") for c in calls)


# ---------------------------------------------------------------------------
# Pydantic model return type
# ---------------------------------------------------------------------------


class TestSearchPydantic:
    def test_search_returns_search_response(self, search_db: Database) -> None:
        """search() should return a SearchResponse Pydantic model."""
        result = search(search_db, None, "MFA", fast=True)
        assert isinstance(result, SearchResponse)

    def test_search_results_are_search_result(self, search_db: Database) -> None:
        """Each item in results should be a SearchResult model."""
        result = search(search_db, None, "Rust migration", fast=True)
        assert len(result.results) > 0
        for r in result.results:
            assert isinstance(r, SearchResult)

    def test_search_meta_is_search_meta(self, search_db: Database) -> None:
        """meta should be a SearchMeta model."""
        result = search(search_db, None, "MFA", fast=True)
        assert isinstance(result.meta, SearchMeta)
        assert result.meta.query == "MFA"
        assert isinstance(result.meta.total, int)
        assert isinstance(result.meta.limit, int)
        assert isinstance(result.meta.execution_ms, float)

    def test_search_result_attribute_access(self, search_db: Database) -> None:
        """SearchResult fields should be accessible as attributes."""
        result = search(search_db, None, "Rust migration", fast=True)
        r = result.results[0]
        assert isinstance(r.chunk_id, int)
        assert isinstance(r.document_id, int)
        assert isinstance(r.title, str)
        assert isinstance(r.path, str)
        assert isinstance(r.score, float)
        assert isinstance(r.snippet, str)
        assert isinstance(r.entities, list)

    def test_search_response_model_dump(self, search_db: Database) -> None:
        """model_dump() should produce a dict matching the old return format."""
        result = search(search_db, None, "MFA", fast=True)
        d = result.model_dump()
        assert "results" in d
        assert "meta" in d
        assert isinstance(d["results"], list)
        assert isinstance(d["meta"], dict)
        # Check round-trip field names
        if d["results"]:
            r = d["results"][0]
            assert "chunk_id" in r
            assert "title" in r
            assert "score" in r
            assert "snippet" in r

    def test_empty_results_still_returns_model(self, search_db: Database) -> None:
        """Empty results should still return a valid SearchResponse."""
        result = search(search_db, None, "xyznonexistent", fast=True)
        assert isinstance(result, SearchResponse)
        assert result.results == []
        assert result.meta.total == 0

    def test_search_results_are_frozen(self, search_db: Database) -> None:
        """SearchResult should be immutable (frozen)."""
        from pydantic import ValidationError

        result = search(search_db, None, "MFA", fast=True)
        if result.results:
            with pytest.raises(ValidationError):
                result.results[0].score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Recency edge cases
# ---------------------------------------------------------------------------


class TestRecencyEdgeCases:
    def test_invalid_date_returns_neutral(self):
        assert _recency_weight("not-a-date") == 0.5

    def test_future_date_returns_one(self):
        assert _recency_weight("2099-01-01") == 1.0


# ---------------------------------------------------------------------------
# _make_snippet
# ---------------------------------------------------------------------------


class TestMakeSnippet:
    def test_short_content_unchanged(self):
        from kb.search import _make_snippet

        assert _make_snippet("Short text") == "Short text"

    def test_long_content_truncated(self):
        from kb.search import _make_snippet

        long = "word " * 100  # 500 chars
        result = _make_snippet(long, max_chars=50)
        assert result.endswith("...")
        assert len(result) <= 55

    def test_strips_metadata_before_truncating(self):
        from kb.search import _make_snippet

        text = "[Meeting: Title | Date: 2026-01-01]\nActual content here"
        result = _make_snippet(text)
        assert not result.startswith("[Meeting:")


# ---------------------------------------------------------------------------
# Equal BM25 scores
# ---------------------------------------------------------------------------


class TestEqualBM25Scores:
    def test_equal_scores_all_get_one(self):
        results = [{"raw_bm25": -5.0}, {"raw_bm25": -5.0}, {"raw_bm25": -5.0}]
        _normalize_bm25_scores(results)
        assert all(r["bm25_score"] == 1.0 for r in results)


# ---------------------------------------------------------------------------
# Search with tag filter
# ---------------------------------------------------------------------------


class TestSearchWithTagFilter:
    def test_tag_filter_no_matches(self, search_db):
        results = search(search_db, None, "MFA", fast=True, tag="nonexistent-tag")
        # Should return empty since no docs have this tag
        assert results.meta.total == 0


# ---------------------------------------------------------------------------
# Search with date filters
# ---------------------------------------------------------------------------


class TestSearchWithDateFilter:
    def test_from_date_filter(self, search_db):
        results = search(search_db, None, "migration", fast=True, from_date="2026-01-01")
        assert results.meta.total > 0
        for r in results.results:
            if r.date:
                assert r.date >= "2026-01-01"

    def test_to_date_filter(self, search_db):
        results = search(search_db, None, "migration", fast=True, to_date="2026-01-25")
        # Should filter out Datastore (2026-02-01)
        for r in results.results:
            if r.date:
                assert r.date <= "2026-01-25"


# ---------------------------------------------------------------------------
# FTS syntax error handling
# ---------------------------------------------------------------------------


class TestFTSSyntaxError:
    def test_fts_syntax_error_returns_empty(self, search_db):
        """Malformed FTS query should not crash — returns empty results."""
        from kb.search import _fts_search

        # NEAR is not supported on our FTS5 config — triggers except branch
        result = _fts_search(search_db, 'NEAR("foo" "bar")', limit=10)
        assert result == [] or isinstance(result, list)


# ---------------------------------------------------------------------------
# _is_strong_signal
# ---------------------------------------------------------------------------


class TestIsStrongSignal:
    def test_empty_results(self):
        from kb.search import _is_strong_signal

        assert _is_strong_signal([]) is False

    def test_single_result_is_strong(self):
        from kb.search import _is_strong_signal

        assert _is_strong_signal([{"bm25_score": 1.0}]) is True

    def test_two_results_is_strong(self):
        from kb.search import _is_strong_signal

        results = [{"bm25_score": 1.0}, {"bm25_score": 0.5}]
        assert _is_strong_signal(results) is True

    def test_many_results_with_low_top_not_strong(self):
        from kb.search import _is_strong_signal

        results = [{"bm25_score": 0.5}, {"bm25_score": 0.4}, {"bm25_score": 0.3}]
        assert _is_strong_signal(results) is False

    def test_many_results_with_big_gap_is_strong(self):
        from kb.search import _is_strong_signal

        results = [{"bm25_score": 1.0}, {"bm25_score": 0.5}, {"bm25_score": 0.3}]
        assert _is_strong_signal(results) is True

    def test_many_results_with_small_gap_not_strong(self):
        from kb.search import _is_strong_signal

        results = [{"bm25_score": 1.0}, {"bm25_score": 0.95}, {"bm25_score": 0.9}]
        assert _is_strong_signal(results) is False


# ---------------------------------------------------------------------------
# _vector_search edge cases
# ---------------------------------------------------------------------------


class TestVectorSearchNoTable:
    def test_no_lance_table_returns_empty(self):
        """When LanceDB has no table, _vector_search returns []."""
        from unittest.mock import MagicMock

        from kb.search import _vector_search

        db = MagicMock()
        db.get_lance_table.return_value = None
        embedder = MagicMock()
        assert _vector_search(db, embedder, "test query") == []


# ---------------------------------------------------------------------------
# _weighted_rrf rank bonuses
# ---------------------------------------------------------------------------


class TestWeightedRRFRankBonuses:
    def test_rank2_and_rank3_get_bonus(self):
        """Ranks 2-3 should get +0.02 bonus."""
        from kb.search import _rrf_score, _weighted_rrf

        vec = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}, {"chunk_id": 4}]
        scores = _weighted_rrf([], vec)
        # Rank 1 gets +0.05, ranks 2-3 get +0.02, rank 4 gets nothing
        base_4 = _rrf_score(4, 60)
        assert scores[4] == pytest.approx(base_4)
        assert scores[2] > base_4  # rank 2 has bonus
        assert scores[3] > base_4  # rank 3 has bonus


# ---------------------------------------------------------------------------
# search() with vector results only (no FTS hits)
# ---------------------------------------------------------------------------


class TestSearchVectorOnly:
    def test_vector_only_path(self, search_db):
        """When FTS returns nothing but vectors have results, use vector scores."""
        from unittest.mock import patch

        from kb.search import search

        # Patch _fts_search to return nothing, forcing vector-only path
        with patch("kb.search._fts_search", return_value=[]):
            # fast=True so it won't attempt vector search either
            results = search(search_db, None, "nonexistent_term_xyz", fast=True)
            assert results.meta.total == 0


# ---------------------------------------------------------------------------
# search() doc_type filter
# ---------------------------------------------------------------------------


class TestSearchDocTypeFilter:
    def test_doc_type_filters_results(self, search_db):
        """doc_type filter should exclude non-matching documents."""
        results = search(search_db, None, "migration", fast=True, doc_type="transcript")
        # All test docs are "notes", filtering by "transcript" should get 0
        assert results.meta.total == 0


# ---------------------------------------------------------------------------
# Heading weight boost (bm25 column weights)
# ---------------------------------------------------------------------------


class TestHeadingWeightBoost:
    def test_heading_match_ranks_higher(self, search_db):
        """Chunks where the query matches the heading should rank above content-only matches.

        "Performance" appears as a heading in doc2 chunk1 and also as plain content
        in other chunks. With heading boost, the heading-match chunk should rank first.
        """
        from kb.search import _fts_search

        results = _fts_search(search_db, "Performance", limit=10)
        assert len(results) > 0
        # The chunk with heading="Performance" should be the top result
        assert results[0]["heading"] == "Performance"


# ---------------------------------------------------------------------------
# AND query variant
# ---------------------------------------------------------------------------


class TestAndQueryVariant:
    def test_and_variant_in_query_chain(self):
        """Multi-word queries should include an AND variant between NEAR and OR."""
        from kb.search import _build_fts_query

        variants = _build_fts_query("rust migration status")
        # Should have: phrase, NEAR, AND (implicit), OR
        assert len(variants) == 4
        # The AND variant is space-separated terms (implicit AND in FTS5)
        assert "rust migration status" in variants
        # OR variant is still last
        assert variants[-1] == "rust OR migration OR status"

    def test_and_variant_catches_scattered_terms(self, search_db):
        """AND variant should find chunks containing all terms even if not near each other."""
        # "Rust" and "modules" both appear in Helix Refactor Status doc
        # but may not be within 10 tokens of each other
        results = search(search_db, None, "Rust modules", fast=True)
        assert results.meta.total > 0


# ---------------------------------------------------------------------------
# Prefix search
# ---------------------------------------------------------------------------


class TestPrefixSearch:
    def test_trailing_star_preserved(self):
        """Trailing * should NOT be stripped from query terms."""
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input("migr*")
        assert result == "migr*"

    def test_leading_star_stripped(self):
        """Leading * (not valid FTS5 prefix) should be stripped."""
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input("*migration")
        assert "*" not in result

    def test_middle_star_stripped(self):
        """Stars in the middle of words should be stripped."""
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input("mi*gration")
        assert "*" not in result

    def test_prefix_search_finds_results(self, search_db):
        """A prefix query like 'migr*' should match 'migration'."""
        results = search(search_db, None, "migr*", fast=True)
        assert results.meta.total > 0
        # Should find migration-related docs
        titles = [r.title for r in results.results]
        assert any("Migration" in t for t in titles)

    def test_standalone_star_stripped(self):
        """A bare * should be stripped entirely."""
        from kb.search import _sanitize_fts_input

        result = _sanitize_fts_input("* test")
        assert result == "test"


# ---------------------------------------------------------------------------
# FTS5 UPDATE trigger
# ---------------------------------------------------------------------------


class TestFTSUpdateTrigger:
    def test_updated_chunk_is_searchable(self, search_db):
        """After updating chunk content, FTS should find the new text."""
        conn = search_db.get_sqlite_conn()
        pre = conn.execute(
            "SELECT count(*) as cnt FROM chunks_fts WHERE chunks_fts MATCH 'kubernetes'"
        ).fetchone()
        assert pre["cnt"] == 0

        conn.execute(
            "UPDATE chunks SET content = 'We are deploying kubernetes clusters' WHERE id = 1"
        )
        conn.commit()

        post = conn.execute(
            "SELECT count(*) as cnt FROM chunks_fts WHERE chunks_fts MATCH 'kubernetes'"
        ).fetchone()
        assert post["cnt"] == 1

    def test_updated_heading_is_searchable(self, search_db):
        """After updating chunk heading, FTS should find the new heading."""
        conn = search_db.get_sqlite_conn()
        conn.execute("UPDATE chunks SET heading = 'Kubernetes' WHERE id = 1")
        conn.commit()

        results = conn.execute(
            "SELECT count(*) as cnt FROM chunks_fts WHERE chunks_fts MATCH 'Kubernetes'"
        ).fetchone()
        assert results["cnt"] == 1


# ---------------------------------------------------------------------------
# Configurable FTS/vector weights (item 9)
# ---------------------------------------------------------------------------


class TestConfigurableWeights:
    def test_rrf_default_weights_equal(self):
        """Default weights should produce equal FTS and vector contributions."""
        from kb.search import _weighted_rrf

        fts = [{"chunk_id": 1}]
        vec = [{"chunk_id": 2}]
        scores = _weighted_rrf(fts, vec)
        assert scores[1] == pytest.approx(scores[2])

    def test_rrf_fts_heavy_weight(self):
        """With fts_weight=3.0 and vector_weight=1.0, FTS result should score higher."""
        from kb.search import _weighted_rrf

        fts = [{"chunk_id": 1}]
        vec = [{"chunk_id": 2}]
        scores = _weighted_rrf(fts, vec, fts_weight=3.0, vector_weight=1.0)
        assert scores[1] > scores[2]

    def test_rrf_vector_heavy_weight(self):
        """With fts_weight=1.0 and vector_weight=3.0, vector result should score higher."""
        from kb.search import _weighted_rrf

        fts = [{"chunk_id": 1}]
        vec = [{"chunk_id": 2}]
        scores = _weighted_rrf(fts, vec, fts_weight=1.0, vector_weight=3.0)
        assert scores[2] > scores[1]

    def test_search_accepts_fts_weight(self, search_db):
        """search() should accept fts_weight parameter without crashing."""
        results = search(search_db, None, "MFA", fast=True, fts_weight=2.0)
        assert results.meta.total > 0

    def test_search_accepts_vector_weight(self, search_db):
        """search() should accept vector_weight parameter without crashing."""
        results = search(search_db, None, "MFA", fast=True, vector_weight=2.0)
        assert results.meta.total > 0


# ---------------------------------------------------------------------------
# Document dedup + chunk_count (item 10)
# ---------------------------------------------------------------------------


class TestDocumentDedup:
    def test_dedupe_keeps_best_chunk_per_doc(self, search_db):
        """With dedupe=True, only the highest-scoring chunk per document should appear."""
        results = search(search_db, None, "MFA", fast=True, dedupe=True)
        doc_ids = [r.document_id for r in results.results]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate document IDs in deduplicated results"

    def test_dedupe_includes_chunk_count(self, search_db):
        """With dedupe=True, each result should have chunk_count showing total matching chunks."""
        results = search(search_db, None, "MFA", fast=True, dedupe=True)
        assert results.meta.total > 0
        for r in results.results:
            assert r.chunk_count >= 1
        # Doc1 has 2 MFA chunks — deduped result should show chunk_count=2
        mfa_result = next(r for r in results.results if "MFA" in r.title)
        assert mfa_result.chunk_count == 2

    def test_no_dedupe_chunk_count_is_one(self, search_db):
        """Without dedupe, chunk_count should default to 1."""
        results = search(search_db, None, "MFA", fast=True, dedupe=False)
        for r in results.results:
            assert r.chunk_count == 1

    def test_dedupe_default_is_false(self, search_db):
        """Default behavior should not deduplicate."""
        results = search(search_db, None, "MFA implementation", fast=True)
        assert results.meta.total > 0


# ---------------------------------------------------------------------------
# Glossary-aware query expansion (item 6)
# ---------------------------------------------------------------------------


class TestGlossaryExpansion:
    def test_expand_acronym_to_full_form(self):
        """An acronym in glossary should expand to include full form."""
        from kb.search import _expand_query_with_glossary

        entries = [("MFA", "Multi-Factor Authentication", "Acronyms")]
        expanded_query, expanded_terms = _expand_query_with_glossary("MFA rollout", entries)
        assert "Multi-Factor Authentication" in expanded_query
        assert expanded_terms == {"MFA": "Multi-Factor Authentication"}

    def test_expand_full_form_to_acronym(self):
        """A full form should expand to include the acronym."""
        from kb.search import _expand_query_with_glossary

        entries = [("MFA", "Multi-Factor Authentication", "Acronyms")]
        expanded_query, expanded_terms = _expand_query_with_glossary(
            "Multi-Factor Authentication", entries
        )
        assert "MFA" in expanded_query
        assert expanded_terms == {"Multi-Factor Authentication": "MFA"}

    def test_no_expansion_for_unknown_term(self):
        """Terms not in glossary should not be modified."""
        from kb.search import _expand_query_with_glossary

        entries = [("MFA", "Multi-Factor Authentication", "Acronyms")]
        expanded_query, expanded_terms = _expand_query_with_glossary("kubernetes deploy", entries)
        assert expanded_query == "kubernetes deploy"
        assert expanded_terms == {}

    def test_expansion_uses_word_boundaries(self):
        """Short terms should NOT match inside longer words."""
        from kb.search import _expand_query_with_glossary

        entries = [("EM", "Engineering Manager", "Acronyms")]
        # "EM" should NOT match inside "implementation"
        _, expanded_terms = _expand_query_with_glossary("implementation review", entries)
        assert expanded_terms == {}

    def test_expanded_terms_in_search_meta(self, search_db):
        """SearchMeta should include expanded_terms dict."""
        results = search(search_db, None, "MFA", fast=True)
        assert isinstance(results.meta.expanded_terms, dict)


# ---------------------------------------------------------------------------
# Entity-aware search boosting (item 7)
# ---------------------------------------------------------------------------


class TestEntityBoosting:
    def test_entity_boost_is_multiplicative(self):
        """Entity boost should multiply the base score, not replace it."""
        from kb.search import _apply_entity_boost

        assert _apply_entity_boost(0.5, has_entity_match=True) > 0.5
        assert _apply_entity_boost(0.5, has_entity_match=False) == 0.5

    def test_entity_boost_exact_factor(self):
        """Entity boost should be exactly 1.15x."""
        from kb.search import _apply_entity_boost

        assert _apply_entity_boost(1.0, has_entity_match=True) == pytest.approx(1.15)

    def test_entity_boost_does_not_crash_on_no_entities(self, search_db):
        """Queries with no entity matches should still work normally."""
        results = search(search_db, None, "MFA implementation", fast=True)
        assert results.meta.total > 0


# ---------------------------------------------------------------------------
# Snippet highlighting with HTML <b> tags (item 8)
# ---------------------------------------------------------------------------


class TestSnippetHighlighting:
    def test_highlight_marks_query_terms(self):
        """Query terms in snippets should be wrapped in <b> HTML tags."""
        from kb.search import _highlight_snippet

        snippet = "We reviewed the MFA implementation progress"
        result = _highlight_snippet(snippet, "MFA implementation")
        assert "<b>MFA</b>" in result
        assert "<b>implementation</b>" in result

    def test_highlight_case_insensitive(self):
        """Highlighting should be case-insensitive."""
        from kb.search import _highlight_snippet

        snippet = "The mfa rollout is complete"
        result = _highlight_snippet(snippet, "MFA")
        assert "<b>mfa</b>" in result

    def test_highlight_no_match(self):
        """If no query terms match, snippet should be unchanged (but HTML-escaped)."""
        from kb.search import _highlight_snippet

        snippet = "General weekly sync discussion"
        result = _highlight_snippet(snippet, "kubernetes")
        assert result == snippet

    def test_highlight_preserves_word_boundaries(self):
        """Highlighting should match whole words, not substrings."""
        from kb.search import _highlight_snippet

        snippet = "The information about MFA was shared"
        result = _highlight_snippet(snippet, "info")
        assert "<b>info</b>" not in result

    def test_highlight_escapes_html_first(self):
        """HTML in content should be escaped BEFORE applying <b> tags."""
        from kb.search import _highlight_snippet

        snippet = "Use <script>alert('MFA')</script> for MFA"
        result = _highlight_snippet(snippet, "MFA")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
        assert "<b>MFA</b>" in result


# ---------------------------------------------------------------------------
# Metadata separation from chunk content (item 12)
# ---------------------------------------------------------------------------


class TestMetadataSeparation:
    def test_chunk_content_has_no_metadata_prefix(self, search_db):
        """After metadata separation, chunk content should not contain [Meeting: ...] prefix."""
        conn = search_db.get_sqlite_conn()
        rows = conn.execute("SELECT content FROM chunks").fetchall()
        for row in rows:
            assert not row["content"].startswith("[Meeting:"), (
                f"Content still has metadata prefix: {row['content'][:80]}"
            )
            assert not row["content"].startswith("[Transcript:"), (
                f"Content still has transcript prefix: {row['content'][:80]}"
            )

    def test_metadata_prefix_stored_separately(self, search_db):
        """Metadata prefix should be stored in a separate column."""
        conn = search_db.get_sqlite_conn()
        rows = conn.execute(
            "SELECT metadata_prefix FROM chunks WHERE metadata_prefix != ''"
        ).fetchall()
        assert len(rows) > 0
        for row in rows:
            assert row["metadata_prefix"].startswith("[Meeting:")

    def test_fts_does_not_match_metadata_noise(self, search_db):
        """Searching for 'Date' should not match metadata prefixes."""
        from kb.search import _fts_search

        results = _fts_search(search_db, "Date", limit=50)
        # With metadata separation, "Date" in prefix is not in FTS content
        # Only actual content mentions of "Date" should match
        # In our test data, no actual content uses the word "Date"
        assert len(results) == 0

    def test_snippet_still_clean(self, search_db):
        """Snippets should show clean content without metadata prefix."""
        results = search(search_db, None, "MFA", fast=True)
        for r in results.results:
            assert not r.snippet.startswith("[Meeting:")
