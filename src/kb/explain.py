"""Human-readable formatter for ``--explain`` search output (issue #68 Phase 2).

Phase 1 of #68 already captures per-result component scores on ``SearchResult.explain``
and pipeline counts on ``SearchMeta``. This module renders that captured data as a
terminal-friendly text table — no extra search work, pure formatting.

The default ``kb_output`` table renderer is great for "show me the top results" but
hides scoring internals behind the snippet column. With ``--explain`` users want to
see *why* each result ranked where it did, so we switch to a custom layout when both
``--explain`` is passed and the format is ``table``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb.types import (
        SearchExplain,
        SearchResponse,
        SearchResult,
        WhyNotDiagnostics,
        ZeroResultDiagnostics,
    )


_RULE = "━" * 60


def _fmt_score(value: float | None) -> str:
    """Format a numeric component score; use a dash for missing data."""
    if value is None:
        return "  —  "
    return f"{value:.3f}"


def _fmt_result(idx: int, result: SearchResult) -> list[str]:
    lines: list[str] = []
    lines.append(_RULE)
    lines.append(f"#{idx}  {result.title}")
    lines.append(f"     {result.path}")
    date_part = f" · {result.date}" if result.date else ""
    section_part = f" · {result.section}" if result.section else ""
    lines.append(f"     final score: {result.score:.3f}{date_part}{section_part}")

    explain = result.explain
    if explain is None:
        lines.append("     (no explain data — was --explain passed?)")
        return lines

    lines.append("")
    lines.append("     Scoring breakdown:")
    lines.append(_explain_component_line(explain))
    lines.append(_explain_blend_line(explain))
    lines.append(_explain_signal_line(explain))
    if explain.matched_terms:
        lines.append(_explain_terms_line(explain))
    return lines


def _explain_component_line(explain: SearchExplain) -> str:
    """First row: raw FTS / Vector / Fused scores with weights."""
    return (
        "       FTS:     "
        f"{_fmt_score(explain.fts_score)}  (w={explain.fts_weight:.2f})    "
        "Vector:  "
        f"{_fmt_score(explain.vector_score)}  (w={explain.vector_weight:.2f})    "
        "Fused:   "
        f"{_fmt_score(explain.fused_score)}"
    )


def _explain_blend_line(explain: SearchExplain) -> str:
    """Second row: recency + hotness contributions."""
    parts = [
        f"       Recency: {_fmt_score(explain.recency_weight)}  (w={explain.recency:.2f})",
        f"Hotness: {_fmt_score(explain.hotness_score)}  (access_count={explain.access_count})",
    ]
    if explain.parent_entity_score is not None:
        parts.append(f"Parent entity: {_fmt_score(explain.parent_entity_score)}")
    return "    ".join(parts)


def _explain_signal_line(explain: SearchExplain) -> str:
    """Third row: pipeline source + entity boost flag."""
    boost = "applied" if explain.entity_boost_applied else "—"
    return (
        f"       Source: {explain.source}    "
        f"Entity boost: {boost}    "
        f"Final: {explain.final_score:.3f}"
    )


def _explain_terms_line(explain: SearchExplain) -> str:
    """Per-term match detail: which query terms matched and where (#3)."""
    parts: list[str] = []
    for tm in explain.matched_terms:
        locs: list[str] = []
        for loc in tm.locations:
            if loc == "body" and tm.body_count:
                locs.append(f"body:{tm.body_count}")
            else:
                locs.append(loc)
        parts.append(f"{tm.term} ({', '.join(locs)})")
    return "       Terms:   " + " · ".join(parts)


def _fmt_meta(response: SearchResponse) -> list[str]:
    meta = response.meta
    lines: list[str] = []
    lines.append(f'Query: "{meta.query}"')
    bits: list[str] = []
    if meta.search_mode:
        bits.append(f"mode={meta.search_mode}")
    if meta.fts_hits is not None:
        bits.append(f"fts_hits={meta.fts_hits}")
    if meta.vector_hits is not None:
        bits.append(f"vector_hits={meta.vector_hits}")
    if meta.both_hits is not None:
        bits.append(f"both_hits={meta.both_hits}")
    bits.append(f"returned={len(response.results)}")
    bits.append(f"limit={meta.limit}")
    bits.append(f"sort={meta.sort_by}")
    bits.append(f"{meta.execution_ms:.1f}ms")
    lines.append("  " + " · ".join(bits))

    if meta.path_filter:
        lines.append(
            f"  path filter: {meta.path_filter}"
            + (
                f"  ({meta.path_filter_doc_count} docs)"
                if meta.path_filter_doc_count is not None
                else ""
            )
        )
    if meta.hierarchy_active:
        alpha = meta.hierarchy_alpha
        alpha_part = f" (alpha={alpha:.2f})" if alpha is not None else ""
        lines.append(f"  hierarchy: active{alpha_part}")
    if meta.fts_variants_tried:
        lines.append("  fts variants tried: " + " | ".join(meta.fts_variants_tried[:4]))
    if meta.expanded_terms:
        expansions = ", ".join(f"{k}→{v}" for k, v in meta.expanded_terms.items())
        lines.append(f"  glossary expansions: {expansions}")
    if meta.timings:
        t = meta.timings
        vec = f"{t.vector_ms:.1f}ms" if t.vector_ms is not None else "—"
        lines.append(f"  timings: fts {t.fts_ms:.1f}ms · vector {vec} · merge {t.merge_ms:.1f}ms")
    return lines


def _fmt_why_not(wn: WhyNotDiagnostics) -> list[str]:
    """Render why-not diagnostics for a targeted document (#3 why-not mode)."""
    lines = ["", f'Why-not "{wn.path}": {wn.status}']
    if wn.rank is not None:
        rank_part = f"  rank #{wn.rank}"
        if wn.cutoff is not None:
            rank_part += f" (cutoff: top {wn.cutoff})"
        lines.append(rank_part)
    lines.append(f"  {wn.detail}")
    return lines


def _fmt_zero_result_diagnostics(diag: ZeroResultDiagnostics) -> list[str]:
    """Render zero-result diagnostics: per-term FTS coverage, near-misses, suggestions (#3)."""
    lines: list[str] = ["No results — diagnostics:"]

    if diag.term_hits:
        lines.append("  FTS term coverage:")
        for th in diag.term_hits:
            mark = "✗" if th.doc_count == 0 else "✓"
            noun = "document" if th.doc_count == 1 else "documents"
            lines.append(f"    {mark} {th.term}: {th.doc_count} {noun}")

    if diag.vector_near_misses:
        lines.append("  Closest semantic matches (below the relevance bar):")
        for nm in diag.vector_near_misses:
            lines.append(f"    {nm.similarity:.3f}  {nm.title}")
            lines.append(f"           {nm.path}")

    if diag.suggestions:
        lines.append("  Suggestions:")
        for suggestion in diag.suggestions:
            lines.append(f"    • {suggestion}")

    return lines


def format_explain_text(response: SearchResponse) -> str:
    """Render a ``SearchResponse`` (with explain data attached) as readable text.

    Layout:
        Query: "..."
          mode=... fts_hits=... vector_hits=... both_hits=... returned=N limit=M sort=...
          (optional: path filter, hierarchy, fts variants)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #1  Title
            path
            final score: ...
            Scoring breakdown:
              FTS / Vector / Fused
              Recency / Hotness / Parent entity (if hierarchy)
              Source / Entity boost / Final
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ...

    Falls back gracefully when individual fields are ``None`` (vector-only or
    FTS-only hits, no recency on undated docs, etc.).
    """
    lines: list[str] = []
    lines.extend(_fmt_meta(response))
    if response.meta.why_not is not None:
        lines.extend(_fmt_why_not(response.meta.why_not))

    if not response.results:
        lines.append("")
        diag = response.meta.zero_result_diagnostics
        if diag is None:
            lines.append("No results — try lowering --threshold or broadening the query.")
        else:
            lines.extend(_fmt_zero_result_diagnostics(diag))
        return "\n".join(lines)

    for idx, result in enumerate(response.results, start=1):
        lines.extend(_fmt_result(idx, result))
    lines.append(_RULE)
    return "\n".join(lines)
