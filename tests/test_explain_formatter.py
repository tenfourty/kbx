"""Tests for the human-readable ``--explain`` formatter (issue #68 Phase 2)."""

from __future__ import annotations

from kb.types import (
    SearchExplain,
    SearchMeta,
    SearchResponse,
    SearchResult,
    TermHit,
    TermMatch,
    VectorNearMiss,
    ZeroResultDiagnostics,
)


def _make_response(
    *,
    results: list[SearchResult] | None = None,
    meta_overrides: dict | None = None,
) -> SearchResponse:
    meta = SearchMeta(
        query="deployment timeline",
        total=len(results) if results else 0,
        limit=10,
        sort_by="score",
        execution_ms=12.3,
        search_mode="hybrid",
        fts_hits=5,
        vector_hits=4,
        both_hits=3,
        **(meta_overrides or {}),
    )
    return SearchResponse(results=results or [], meta=meta)


def _make_result(
    *,
    title: str = "Meeting Notes",
    path: str = "memory/meetings/2026/02/foo.md",
    score: float = 0.847,
    explain: SearchExplain | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=1,
        document_id=10,
        title=title,
        path=path,
        date="2026-02-10",
        doc_type="meeting",
        score=score,
        section="Overview",
        snippet="discussed the refactor timeline...",
        explain=explain,
    )


class TestExplainFormatter:
    """``format_explain_text(response)`` renders SearchResponse as terminal text."""

    def test_returns_string(self):
        from kb.explain import format_explain_text

        out = format_explain_text(_make_response())
        assert isinstance(out, str)

    def test_empty_response_mentions_no_results(self):
        from kb.explain import format_explain_text

        out = format_explain_text(_make_response())
        assert "No results" in out or "0 results" in out

    def test_renders_meta_header_with_query(self):
        from kb.explain import format_explain_text

        out = format_explain_text(_make_response())
        assert "deployment timeline" in out

    def test_renders_result_title_path_score(self):
        from kb.explain import format_explain_text

        explain = SearchExplain(
            fts_score=0.82,
            vector_score=0.79,
            fused_score=0.81,
            recency_weight=0.5,
            entity_boost_applied=False,
            final_score=0.847,
            source="both",
            fts_weight=1.0,
            vector_weight=1.0,
            recency=0.15,
        )
        result = _make_result(explain=explain)
        out = format_explain_text(_make_response(results=[result]))
        # Headline lines
        assert "Meeting Notes" in out
        assert "memory/meetings/2026/02/foo.md" in out
        # Final score appears as a numeric formatted string
        assert "0.847" in out or "0.85" in out

    def test_renders_matched_terms(self):
        """Per-result explain surfaces which query terms matched and where (#3)."""
        from kb.explain import format_explain_text

        explain = SearchExplain(
            fts_score=0.82,
            vector_score=None,
            fused_score=0.82,
            recency_weight=None,
            entity_boost_applied=False,
            final_score=0.82,
            source="fts_only",
            fts_weight=1.0,
            vector_weight=1.0,
            recency=0.0,
            matched_terms=[
                TermMatch(term="deployment", locations=["title", "body"], body_count=3),
                TermMatch(term="timeline", locations=["body"], body_count=2),
            ],
        )
        result = _make_result(score=0.82, explain=explain)
        out = format_explain_text(_make_response(results=[result]))
        assert "deployment" in out
        assert "timeline" in out
        # body occurrence count is surfaced
        assert "body:3" in out

    def test_omits_terms_line_when_no_matches(self):
        """No 'Terms:' line when matched_terms is empty (back-compat)."""
        from kb.explain import format_explain_text

        explain = SearchExplain(
            fts_score=0.82,
            vector_score=None,
            fused_score=0.82,
            recency_weight=None,
            entity_boost_applied=False,
            final_score=0.82,
            source="fts_only",
            fts_weight=1.0,
            vector_weight=1.0,
            recency=0.0,
        )
        result = _make_result(score=0.82, explain=explain)
        out = format_explain_text(_make_response(results=[result]))
        assert "Terms:" not in out

    def test_renders_component_scores(self):
        from kb.explain import format_explain_text

        explain = SearchExplain(
            fts_score=0.82,
            vector_score=0.79,
            fused_score=0.81,
            recency_weight=0.5,
            entity_boost_applied=True,
            pre_hotness_score=0.81,
            hotness_score=0.42,
            access_count=5,
            final_score=0.74,
            source="both",
            fts_weight=1.0,
            vector_weight=1.0,
            recency=0.15,
        )
        result = _make_result(score=0.74, explain=explain)
        out = format_explain_text(_make_response(results=[result]))
        # All score components should appear
        for label in ["FTS", "Vector", "Fused", "Recency", "Hotness"]:
            assert label in out, f"missing {label} in:\n{out}"
        # Numeric components — exact-or-rounded
        assert "0.82" in out
        assert "0.79" in out
        # Boost flag visible
        assert "Entity boost" in out
        # Source pipeline noted
        assert "both" in out

    def test_handles_missing_components(self):
        from kb.explain import format_explain_text

        # Vector-only result: FTS score is None
        explain = SearchExplain(
            fts_score=None,
            vector_score=0.65,
            fused_score=0.65,
            recency_weight=None,
            entity_boost_applied=False,
            final_score=0.65,
            source="vector_only",
            fts_weight=1.0,
            vector_weight=1.0,
            recency=0.0,
        )
        result = _make_result(score=0.65, explain=explain)
        out = format_explain_text(_make_response(results=[result]))
        # Should not crash, should render — None values shown as "—" or "n/a"
        assert "vector_only" in out
        assert "0.65" in out

    def test_meta_footer_includes_search_mode(self):
        from kb.explain import format_explain_text

        result = _make_result(
            explain=SearchExplain(
                fts_score=0.5,
                vector_score=0.5,
                fused_score=0.5,
                recency_weight=0.5,
                entity_boost_applied=False,
                final_score=0.5,
                source="both",
                fts_weight=1.0,
                vector_weight=1.0,
                recency=0.15,
            )
        )
        out = format_explain_text(_make_response(results=[result]))
        # mode + pipeline counts should show somewhere
        assert "hybrid" in out
        assert "5" in out and "4" in out  # fts_hits, vector_hits


class TestSearchExplainCLIRendering:
    """E2E: ``kbx search --explain`` (no --json) emits the formatted text."""

    def test_cli_explain_no_json_emits_formatted_text(self, runner, tmp_path, monkeypatch):
        from kb import search as search_module
        from tests.conftest import invoke_cli

        # Build a fake SearchResponse so the CLI doesn't need a real DB.
        fake_explain = SearchExplain(
            fts_score=0.82,
            vector_score=0.79,
            fused_score=0.81,
            recency_weight=0.5,
            entity_boost_applied=False,
            final_score=0.847,
            source="both",
            fts_weight=1.0,
            vector_weight=1.0,
            recency=0.15,
        )
        fake_result = SearchResult(
            chunk_id=1,
            document_id=10,
            title="Deployment Timeline Sync",
            path="memory/meetings/2026/02/sync.md",
            date="2026-02-10",
            doc_type="meeting",
            score=0.847,
            section=None,
            snippet="...",
            explain=fake_explain,
        )
        fake_response = SearchResponse(
            results=[fake_result],
            meta=SearchMeta(
                query="deployment timeline",
                total=1,
                limit=10,
                sort_by="score",
                execution_ms=10.0,
                search_mode="hybrid",
                fts_hits=1,
                vector_hits=1,
                both_hits=1,
            ),
        )

        def fake_do_search(*args, **kwargs):
            return fake_response

        monkeypatch.setattr(search_module, "search", fake_do_search)

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = invoke_cli(
            runner,
            ["search", "deployment timeline", "--fast", "--explain"],
            data_dir,
        )
        assert result.exit_code == 0, result.output
        # Must include the readable label, NOT raw JSON braces
        assert "Deployment Timeline Sync" in result.output
        assert "FTS" in result.output
        assert "Vector" in result.output
        # Should NOT look like JSON
        assert not result.output.lstrip().startswith("{")


class TestZeroResultDiagnosticsRendering:
    """The formatter renders zero-result diagnostics in place of the generic line (#3)."""

    def _empty_with_diag(self, diag: ZeroResultDiagnostics) -> SearchResponse:
        meta = SearchMeta(
            query="foobar baz",
            total=0,
            limit=10,
            sort_by="score",
            execution_ms=4.0,
            search_mode="fast",
            fts_hits=0,
            vector_hits=0,
            both_hits=0,
            zero_result_diagnostics=diag,
        )
        return SearchResponse(results=[], meta=meta)

    def test_renders_per_term_counts(self):
        from kb.explain import format_explain_text

        diag = ZeroResultDiagnostics(
            term_hits=[
                TermHit(term="foobar", doc_count=0),
                TermHit(term="baz", doc_count=3),
            ],
            vector_near_misses=[],
            suggestions=["Try broadening the query."],
        )
        out = format_explain_text(self._empty_with_diag(diag))
        assert "foobar" in out
        assert "baz" in out
        assert "0" in out and "3" in out  # the per-term counts surface
        assert "Try broadening the query." in out

    def test_renders_vector_near_misses(self):
        from kb.explain import format_explain_text

        diag = ZeroResultDiagnostics(
            term_hits=[TermHit(term="foobar", doc_count=0)],
            vector_near_misses=[
                VectorNearMiss(
                    title="Deploy Notes", path="memory/notes/deploy.md", similarity=0.31
                ),
            ],
            suggestions=["Rephrase the query."],
        )
        out = format_explain_text(self._empty_with_diag(diag))
        assert "Deploy Notes" in out
        assert "memory/notes/deploy.md" in out
        assert "0.31" in out

    def test_renders_suggestions_bullets(self):
        from kb.explain import format_explain_text

        diag = ZeroResultDiagnostics(
            term_hits=[TermHit(term="foobar", doc_count=0)],
            vector_near_misses=[],
            suggestions=["Drop the unmatched term.", "Try semantic search."],
        )
        out = format_explain_text(self._empty_with_diag(diag))
        assert "Drop the unmatched term." in out
        assert "Try semantic search." in out

    def test_empty_results_without_diag_keeps_generic_line(self):
        """Back-compat: empty response with no diagnostics still shows the generic line."""
        from kb.explain import format_explain_text

        meta = SearchMeta(query="x", total=0, limit=10, sort_by="score", execution_ms=1.0)
        out = format_explain_text(SearchResponse(results=[], meta=meta))
        assert "No results" in out
