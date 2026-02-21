"""Smoke tests for kb foundation: embeddings, database, output formatting."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_embedder_shape():
    """Embedding returns (n, 1024) L2-normalized float32 array."""
    from kb.embeddings import EMBEDDING_DIM, Embedder

    e = Embedder()
    result = e.embed(["hello world"])
    assert result.shape == (1, EMBEDDING_DIM), f"Expected (1, {EMBEDDING_DIM}), got {result.shape}"
    assert result.dtype == np.float32

    # Check L2 normalization
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 0.01, f"Expected approximately unit norm, got {norm}"


@pytest.mark.slow
def test_embedder_batch():
    """Batch embedding works correctly."""
    from kb.embeddings import EMBEDDING_DIM, Embedder

    e = Embedder()
    result = e.embed(["hello", "world", "test"])
    assert result.shape == (3, EMBEDDING_DIM)


@pytest.mark.slow
def test_embedder_query():
    """Query embedding uses instruction-aware prompt."""
    from kb.embeddings import EMBEDDING_DIM, Embedder

    e = Embedder()
    result = e.embed_query("what is MFA?")
    assert result.shape == (1, EMBEDDING_DIM)
    assert result.dtype == np.float32


@pytest.mark.slow
def test_embedder_instruction_aware():
    """Different instructions produce different embeddings."""
    from kb.embeddings import INDEX_INSTRUCTION, QUERY_INSTRUCTION, Embedder

    e = Embedder()
    text = "MFA implementation progress"
    idx_emb = e.embed([text], instruction=INDEX_INSTRUCTION)[0]
    query_emb = e.embed([text], instruction=QUERY_INSTRUCTION)[0]

    # They should be similar but not identical (different instructions)
    similarity = np.dot(idx_emb, query_emb)
    assert similarity > 0.5, (
        f"Same text with different instructions should still be similar, got {similarity}"
    )
    assert similarity < 1.0 - 1e-5, "Different instructions should produce different embeddings"


def test_sqlite_schema():
    """SQLite schema creates all expected tables."""
    from kb.db import Database

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row[0] for row in tables}

        assert "documents" in table_names
        assert "chunks" in table_names
        assert "entities" in table_names
        assert "entity_mentions" in table_names
        assert "chunks_fts" in table_names

        db.close()


def test_sqlite_wal_mode():
    """SQLite should be in WAL mode for concurrent reads during writes."""
    from kb.db import Database

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        conn = db.get_sqlite_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal", f"Expected WAL mode, got {mode}"
        db.close()


@pytest.mark.slow
def test_lancedb_roundtrip():
    """Store and retrieve a vector from LanceDB."""
    from kb.db import EMBEDDING_DIM, Database
    from kb.embeddings import Embedder

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir))
        embedder = Embedder()

        # Embed a test string
        vec = embedder.embed(["infrastructure performance monitoring"])[0]
        assert len(vec) == EMBEDDING_DIM

        # Store in LanceDB
        data = [
            {
                "chunk_id": 1,
                "embedding": vec.tolist(),
                "doc_type": "notes",
                "doc_date": "2026-01-27",
                "tags": '["test"]',
                "document_id": 1,
                "entity_ids": "[]",
            }
        ]
        table = db.ensure_lance_table(data)

        # Query — search for the same vector
        query_vec = embedder.embed_query("infrastructure performance")
        results = table.search(query_vec[0].tolist()).limit(1).to_list()

        assert len(results) == 1
        assert results[0]["chunk_id"] == 1
        assert results[0]["doc_type"] == "notes"

        db.close()


def test_output_json():
    """JSON output format works."""
    from kb.output import render

    data = {"results": [{"title": "Test", "score": 0.95}], "meta": {"total": 1}}
    out = render(data, fmt="json")
    parsed = json.loads(out)
    assert parsed["results"][0]["title"] == "Test"


def test_output_jsonl():
    """JSONL output format works."""
    from kb.output import render

    data = {"results": [{"title": "A"}, {"title": "B"}]}
    out = render(data, fmt="jsonl")
    lines = out.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["title"] == "A"
    assert json.loads(lines[1])["title"] == "B"


def test_output_table():
    """Table output format produces non-empty string."""
    from kb.output import render

    data = {"results": [{"title": "Test", "score": 0.95, "date": "2026-01-27"}]}
    out = render(data, fmt="table", columns=["title", "score", "date"])
    assert "Test" in out
    assert "0.95" in out


def test_output_fields():
    """Field selection filters output."""
    from kb.output import select_fields

    data = {"results": [{"title": "A", "score": 0.9, "path": "/foo"}]}
    filtered = select_fields(data, ["title", "score"])
    assert "path" not in filtered["results"][0]
    assert filtered["results"][0]["title"] == "A"


def test_output_csv():
    """CSV output format works."""
    from kb.output import render

    data = {"results": [{"title": "A", "score": 0.9}, {"title": "B", "score": 0.8}]}
    out = render(data, fmt="csv", columns=["title", "score"])
    assert "title,score" in out
    assert "A,0.9" in out
