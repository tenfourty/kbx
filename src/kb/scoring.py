"""Score normalisation + composition for blended retrieval (issue #95 Phase 1).

Centralises the scoring helpers that were previously scattered across
``search.py`` so future ranking components (issue #67 hotness, issue #69
parent-entity score, issue #66 abstract similarity) plug into one consistent
composition surface instead of multiplying ad-hoc into the existing fused
score.

Phase 1 scope (this commit): extract the existing helpers + add
``compose_scores()``. Behaviour of ``kb.search.search()`` is preserved
byte-for-byte — this commit is a refactor with one new pure function bolted on.

Phase 2 (under separate issues / commits) will flatten the BM25/RRF/recency
chain into a registry of `Normaliser` objects so every component carries its
own `[0, 1]` contract; weights move into ``kbx.toml``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

# Multiplicative boost applied when a result's linked entities match a query
# word. Lives here because it composes with the other score-shaping pieces.
ENTITY_BOOST = 1.15


def normalize_bm25_scores(results: list[dict[str, Any]]) -> None:
    """Min-max normalise raw FTS5 BM25 scores in-place to ``[0, 1]``.

    FTS5's ``bm25()`` returns negative scores where lower (more negative) is
    more relevant. After normalisation, 1.0 is the best match in the batch and
    0.0 is the worst. A single-result batch is normalised to 1.0.

    The result dicts gain a ``"bm25_score"`` key.
    """
    if not results:
        return

    raw_scores = [r["raw_bm25"] for r in results]
    min_raw = min(raw_scores)  # most relevant (most negative)
    max_raw = max(raw_scores)  # least relevant (closest to 0)

    if min_raw == max_raw:
        for r in results:
            r["bm25_score"] = 1.0
        return

    spread = max_raw - min_raw
    for r in results:
        # Invert: most negative (min_raw) → 1.0, least negative (max_raw) → 0.0
        r["bm25_score"] = (max_raw - r["raw_bm25"]) / spread


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given 1-based rank position.

    Returns ``1 / (k + rank)``. The constant ``k=60`` matches the kbx fusion
    default and is the value used in the original RRF paper.
    """
    return 1.0 / (k + rank)


def recency_weight(doc_date: str | None, half_life_days: int = 90) -> float:
    """Exponential-decay weight on document age. Always in ``[0, 1]``.

    Today → ~1.0. After ``half_life_days``, ~0.5. After 4 half-lives, ~0.06.

    ``None`` or unparseable dates return ``0.5`` (neutral — the document
    contributes equally to ranking instead of being penalised for unknown age).
    Future dates are treated as "today" (weight 1.0).
    """
    if doc_date is None:
        return 0.5

    try:
        d = date.fromisoformat(doc_date)
    except (ValueError, TypeError):
        return 0.5

    days_ago = (date.today() - d).days
    if days_ago < 0:
        return 1.0  # future date treated as very recent

    return math.exp(-math.log(2) * days_ago / half_life_days)


def apply_entity_boost(score: float, has_entity_match: bool) -> float:
    """Multiplicatively boost a result that has an entity-name word matching the query."""
    if has_entity_match:
        return score * ENTITY_BOOST
    return score


def normalise_vector_distance(distance: float) -> float:
    """Map a LanceDB cosine distance into a ``[0, 1]`` similarity.

    For L2-normalised embeddings (which is what kbx uses — Qwen3 outputs are
    L2-normalised), LanceDB's cosine distance is in ``[0, 1]``: 0 = identical,
    1 = orthogonal. We invert to similarity and clamp defensively against
    floating-point drift outside that range.

    This matches the in-line formula used in ``_vector_search`` (``1.0 -
    distance``) — extracted here so #67 hotness and future ranking components
    can compose against an explicit ``[0, 1]`` contract.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def compose_scores(
    components: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Weighted sum of normalised retrieval components.

    Args:
        components: Mapping of component name → normalised score in ``[0, 1]``.
        weights: Mapping of component name → weight. Weights covering keys not
            present in ``components`` are ignored (the missing component is
            treated as score 0). Weights are normalised to sum to 1.0 if they
            don't already — callers that want strict validation should check
            in advance.

    Returns:
        Weighted sum in ``[0, 1]`` assuming all component scores are in
        ``[0, 1]``. The convex-combination property guarantees the output
        cannot exceed ``max(components)`` or fall below ``min(components)``
        when weights sum to 1.0.

    Example:
        >>> compose_scores(
        ...     {"fts": 0.8, "vector": 0.6, "hotness": 0.3},
        ...     {"fts": 0.4, "vector": 0.4, "hotness": 0.2},
        ... )
        0.62
    """
    if not weights:
        return 0.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    # Renormalise weights so they sum to 1.0. Cheap and forgiving.
    normalised = {k: v / total_weight for k, v in weights.items()}
    return sum(components.get(k, 0.0) * w for k, w in normalised.items())
