"""Tests for kb.entity_embeddings — Prereq A for #69 (issue #96)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from kb.db import Database
from kb.entity_embeddings import (
    MAX_FACTS_IN_PROFILE,
    MAX_PROFILE_CHARS,
    MIN_PROFILE_CHARS,
    build_entity_profile_text,
    compute_profile_hash,
    embed_entities,
    search_entities,
)


class TestBuildEntityProfileText:
    def test_name_only(self):
        text = build_entity_profile_text("Wren")
        assert text == "Wren"

    def test_name_aliases_metadata(self):
        text = build_entity_profile_text(
            "Wren Kasper",
            aliases=["Wren", "AR"],
            metadata={"role": "SRE Lead", "team": "Platform"},
        )
        assert "Wren Kasper" in text
        assert "Wren" in text
        assert "AR" in text
        assert "SRE Lead" in text
        assert "Platform" in text
        assert " | " in text

    def test_includes_top_facts(self):
        text = build_entity_profile_text(
            "Wren",
            facts=["leads the migration project", "owns the auth pipeline"],
        )
        assert "leads the migration project" in text
        assert "owns the auth pipeline" in text

    def test_caps_facts_at_max(self):
        many = [f"fact {i}" for i in range(20)]
        text = build_entity_profile_text("Wren", facts=many)
        # Only MAX_FACTS_IN_PROFILE should land in the output
        assert "fact 0" in text
        assert f"fact {MAX_FACTS_IN_PROFILE - 1}" in text
        assert f"fact {MAX_FACTS_IN_PROFILE}" not in text

    def test_skips_empty_fields(self):
        text = build_entity_profile_text(
            "Wren", aliases=[], metadata={}, facts=[""]
        )
        assert text == "Wren"

    def test_skips_falsy_metadata_values(self):
        """Empty `role` / `team` strings don't pollute the profile."""
        text = build_entity_profile_text(
            "Wren", metadata={"role": "", "team": None}
        )
        assert text == "Wren"

    def test_caps_total_length(self):
        long_fact = "x" * 5000
        text = build_entity_profile_text("Wren", facts=[long_fact])
        assert len(text) <= MAX_PROFILE_CHARS


class TestComputeProfileHash:
    def test_deterministic(self):
        assert compute_profile_hash("hello") == compute_profile_hash("hello")

    def test_differs_for_different_inputs(self):
        assert compute_profile_hash("a") != compute_profile_hash("b")

    def test_returns_short_hex(self):
        h = compute_profile_hash("anything")
        assert len(h) == 16
        int(h, 16)  # must parse as hex


@pytest.fixture
def db_with_entities():
    """Fresh DB with three entities + assorted facts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()
        conn.execute(
            """INSERT INTO entities (name, entity_type, aliases, metadata)
               VALUES (?, ?, ?, ?)""",
            ("Wren Kasper", "person", '["Wren"]', '{"role": "SRE Lead", "team": "Platform"}'),
        )
        conn.execute(
            """INSERT INTO entities (name, entity_type, aliases, metadata)
               VALUES (?, ?, ?, ?)""",
            ("Helix Refactor", "project", "[]", '{"status": "In Progress"}'),
        )
        conn.execute(
            """INSERT INTO entities (name, entity_type, aliases, metadata)
               VALUES (?, ?, ?, ?)""",
            # Thin entity — name + nothing else; should be skipped (below MIN_PROFILE_CHARS)
            ("Bo", "person", "[]", "{}"),
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (1, ?, ?)",
            ("Wren leads the platform infrastructure migration project.", "2026-05-01"),
        )
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (2, ?, ?)",
            ("Cloud migration includes auth, storage, and observability rebuilds.", "2026-05-10"),
        )
        conn.commit()
        yield db


def _make_mock_embedder(dim: int = 1024) -> MagicMock:
    """Mock embedder that returns deterministic vectors based on input length."""
    embedder = MagicMock()

    def fake_embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
        # Each row is a deterministic unit-ish vector — length-based seed.
        arr = np.array(
            [[float((len(t) + j) % 7) / 10.0 for j in range(dim)] for t in texts],
            dtype=np.float32,
        )
        # Normalise so cosine distance is well-defined
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def fake_embed_query(q: str) -> np.ndarray:
        return fake_embed([q])

    embedder.embed.side_effect = fake_embed
    embedder.embed_query.side_effect = fake_embed_query
    return embedder


class TestEmbedEntities:
    def test_embeds_only_substantial_entities(self, db_with_entities):
        embedder = _make_mock_embedder()
        n = embed_entities(db_with_entities, embedder, full=True)
        # Wren + Helix Refactor are substantial; Bo (name "Bo") is below MIN_PROFILE_CHARS.
        assert n == 2

    def test_table_populated(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)
        table = db_with_entities.get_lance_entity_table()
        assert table is not None
        rows = table.to_arrow().to_pylist()
        assert len(rows) == 2
        for r in rows:
            assert r["entity_id"] in {1, 2}
            assert r["entity_type"] in {"person", "project"}
            assert r["profile_text"]
            assert len(r["profile_hash"]) == 16

    def test_incremental_skips_unchanged(self, db_with_entities):
        embedder = _make_mock_embedder()
        first = embed_entities(db_with_entities, embedder, full=True)
        assert first == 2

        # Second pass with no changes — nothing to do.
        second = embed_entities(db_with_entities, embedder, full=False)
        assert second == 0

    def test_incremental_picks_up_fact_change(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)

        # Add a new fact to Wren — profile changes, hash changes.
        conn = db_with_entities.get_sqlite_conn()
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date) VALUES (1, ?, ?)",
            ("Wren now leads the global migration program.", "2026-05-20"),
        )
        conn.commit()

        n = embed_entities(db_with_entities, embedder, full=False)
        assert n == 1  # only Wren re-embedded

    def test_full_reembeds_all(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)
        n = embed_entities(db_with_entities, embedder, full=True)
        assert n == 2


class TestSearchEntities:
    def test_returns_empty_when_no_table(self, db_with_entities):
        """No embeddings yet → search returns []."""
        embedder = _make_mock_embedder()
        assert search_entities(db_with_entities, embedder, "anything") == []

    def test_returns_results_after_embedding(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)
        results = search_entities(db_with_entities, embedder, "infrastructure")
        assert len(results) >= 1
        for r in results:
            assert {"entity_id", "name", "entity_type", "score", "profile_text"}.issubset(r)
            assert 0.0 <= r["score"] <= 1.0

    def test_entity_type_filter(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)
        results = search_entities(
            db_with_entities, embedder, "anything", entity_type="project"
        )
        for r in results:
            assert r["entity_type"] == "project"

    def test_threshold_filters_low_scores(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)
        # Threshold above 1.0 should filter everything out.
        assert (
            search_entities(db_with_entities, embedder, "anything", threshold=1.01) == []
        )

    def test_limit_cap(self, db_with_entities):
        embedder = _make_mock_embedder()
        embed_entities(db_with_entities, embedder, full=True)
        results = search_entities(db_with_entities, embedder, "anything", limit=1)
        assert len(results) <= 1


class TestConstants:
    def test_min_profile_chars_positive(self):
        assert MIN_PROFILE_CHARS > 0

    def test_max_profile_chars_reasonable(self):
        assert 100 < MAX_PROFILE_CHARS <= 8192

    def test_max_facts_in_profile_positive(self):
        assert MAX_FACTS_IN_PROFILE > 0
