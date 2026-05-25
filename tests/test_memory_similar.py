"""Tests for ``kb.memory_similar`` — semantic similarity lookup (issue #71)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kb.api import KnowledgeBase
from kb.db import Database


class StubEmbedder:
    """Deterministic, dependency-free embedder for similarity tests.

    Maps known texts to caller-specified unit vectors; unknown texts get a
    reproducible pseudo-random unit vector (orthogonal in expectation).
    """

    def __init__(self, mapping: dict[str, np.ndarray] | None = None) -> None:
        self.mapping = mapping or {}

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _vec_for(self, text: str) -> np.ndarray:
        if text in self.mapping:
            return self._unit(self.mapping[text].astype(np.float32))
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(1024).astype(np.float32)
        return self._unit(v)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec_for(text).reshape(1, -1)

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
        instruction: str | None = None,
    ) -> np.ndarray:
        return np.vstack([self._vec_for(t) for t in texts])

    def release_gpu_memory(self) -> None:
        pass


def _make_kb(tmp_path: Path, embedder: StubEmbedder) -> KnowledgeBase:
    project_root = tmp_path / "project"
    (project_root / "memory" / "people").mkdir(parents=True)
    (project_root / "memory" / "glossary.md").write_text("# Glossary\n")
    data_dir = tmp_path / "data"
    kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
    kb._embedder = embedder  # type: ignore[assignment]
    kb._embedder_failed = False
    return kb


def _seed_entity_with_facts(kb: KnowledgeBase, name: str, facts: list[str]) -> None:
    conn = kb._db.get_sqlite_conn()
    source_path = f"memory/people/{name.lower().replace(' ', '-')}.md"
    (kb._project_root / source_path).write_text(f"# {name}\n\n## Recent Facts\n")
    conn.execute(
        "INSERT INTO entities (name, entity_type, aliases, metadata, source_path)"
        " VALUES (?, ?, ?, ?, ?)",
        (name, "person", "[]", "{}", source_path),
    )
    eid = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()["id"]
    for seq, text in enumerate(facts, start=1):
        conn.execute(
            "INSERT INTO facts (entity_id, fact_text, fact_date, seq, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (eid, text, "2026-03-01", seq),
        )
    conn.commit()


class TestMemorySimilarAPI:
    """Direct tests of ``KnowledgeBase.memory_similar``."""

    def test_returns_response_shape(self, tmp_path):
        kb = _make_kb(tmp_path, StubEmbedder())
        try:
            resp = kb.memory_similar("anything", entity=None, path=None)
            # No data → empty matches but the wrapper still hydrates.
            assert resp.query == "anything"
            assert resp.threshold == 0.85
            assert resp.matches == []
            assert resp.has_similar is False
            assert resp.best_score == 0.0
            assert resp.scope == {"entity": None, "path": None}
        finally:
            kb.close()

    def test_entity_scope_finds_similar_fact(self, tmp_path):
        target = np.zeros(1024, dtype=np.float32)
        target[0] = 1.0  # candidate and one fact share this vector → cosine 1.0
        embedder = StubEmbedder(
            mapping={
                "prefers async communication": target,
                "favours async work-style": target,
            }
        )
        kb = _make_kb(tmp_path, embedder)
        try:
            _seed_entity_with_facts(
                kb,
                "Linnea Roux",
                ["favours async work-style", "based in Paris"],
            )
            resp = kb.memory_similar(
                "prefers async communication",
                entity="Linnea Roux",
                threshold=0.80,
            )
            assert resp.has_similar is True
            assert resp.best_score == pytest.approx(1.0, abs=1e-5)
            # The mapped fact must be the top match.
            assert resp.matches[0].text == "favours async work-style"
            assert resp.matches[0].match_type == "fact"
            assert resp.matches[0].entity_name == "Linnea Roux"
            assert resp.matches[0].fact_seq == 1
        finally:
            kb.close()

    def test_entity_scope_filters_below_threshold(self, tmp_path):
        embedder = StubEmbedder()  # all random → cosine ~ 0
        kb = _make_kb(tmp_path, embedder)
        try:
            _seed_entity_with_facts(kb, "Linnea Roux", ["unrelated fact A"])
            resp = kb.memory_similar("xyz", entity="Linnea Roux", threshold=0.85)
            assert resp.matches == []
            assert resp.has_similar is False
        finally:
            kb.close()

    def test_entity_scope_respects_limit(self, tmp_path):
        # Map candidate + first three facts to identical vector so all clear threshold.
        v = np.zeros(1024, dtype=np.float32)
        v[5] = 1.0
        mapping = {
            "candidate text": v,
            "fact one": v,
            "fact two": v,
            "fact three": v,
        }
        kb = _make_kb(tmp_path, StubEmbedder(mapping=mapping))
        try:
            _seed_entity_with_facts(kb, "Linnea Roux", ["fact one", "fact two", "fact three"])
            resp = kb.memory_similar(
                "candidate text", entity="Linnea Roux", threshold=0.5, limit=2
            )
            assert len(resp.matches) == 2
        finally:
            kb.close()

    def test_entity_scope_unknown_entity_raises(self, tmp_path):
        kb = _make_kb(tmp_path, StubEmbedder())
        try:
            with pytest.raises(ValueError, match="Entity not found"):
                kb.memory_similar("anything", entity="Nobody Here")
        finally:
            kb.close()

    def test_entity_scope_no_facts_returns_empty(self, tmp_path):
        kb = _make_kb(tmp_path, StubEmbedder())
        try:
            conn = kb._db.get_sqlite_conn()
            conn.execute(
                "INSERT INTO entities (name, entity_type, aliases, metadata)"
                " VALUES ('Ghost', 'person', '[]', '{}')"
            )
            conn.commit()
            resp = kb.memory_similar("anything", entity="Ghost")
            assert resp.matches == []
            assert resp.has_similar is False
        finally:
            kb.close()

    def test_no_embedder_raises(self, tmp_path):
        kb = _make_kb(tmp_path, StubEmbedder())
        kb._embedder = None
        kb._embedder_failed = True
        try:
            with pytest.raises(RuntimeError, match=r"[Ee]mbedder"):
                kb.memory_similar("anything", entity=None)
        finally:
            kb.close()

    def test_scope_validation_entity_and_path_exclusive(self, tmp_path):
        kb = _make_kb(tmp_path, StubEmbedder())
        try:
            with pytest.raises(ValueError, match=r"mutually exclusive|either"):
                kb.memory_similar("x", entity="A", path="memory/foo.md")
        finally:
            kb.close()


class TestMemorySimilarPathScope:
    """Path scope uses the LanceDB chunks table — populate it directly."""

    def _seed_doc_with_chunks(
        self,
        kb: KnowledgeBase,
        path: str,
        chunks: list[tuple[str, np.ndarray]],
    ) -> int:
        conn = kb._db.get_sqlite_conn()
        conn.execute(
            "INSERT INTO documents (path, title, doc_date, doc_type, source_system,"
            " content_hash, chunk_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                path.rsplit("/", 1)[-1],
                "2026-03-01",
                "memory_note",
                "memory",
                "h" + path,
                len(chunks),
            ),
        )
        doc_id = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()["id"]
        records = []
        for idx, (content, vec) in enumerate(chunks):
            cur = conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, heading, content)"
                " VALUES (?, ?, ?, ?)",
                (doc_id, idx, None, content),
            )
            chunk_id = cur.lastrowid
            records.append(
                {
                    "chunk_id": int(chunk_id),
                    "embedding": vec.astype(np.float32).tolist(),
                    "doc_type": "memory_note",
                    "doc_date": "2026-03-01",
                    "tags": "",
                    "document_id": int(doc_id),
                    "entity_ids": "",
                }
            )
        conn.commit()
        kb._db.ensure_lance_table(data=records)
        return int(doc_id)

    def test_path_scope_finds_similar_chunk(self, tmp_path):
        target = np.zeros(1024, dtype=np.float32)
        target[7] = 1.0
        other = np.zeros(1024, dtype=np.float32)
        other[200] = 1.0
        embedder = StubEmbedder(
            mapping={
                "candidate query": target,
            }
        )
        kb = _make_kb(tmp_path, embedder)
        try:
            self._seed_doc_with_chunks(
                kb,
                "memory/projects/foo.md",
                [("close chunk", target), ("far chunk", other)],
            )
            resp = kb.memory_similar(
                "candidate query",
                path="memory/projects/foo.md",
                threshold=0.80,
            )
            assert resp.has_similar is True
            assert resp.matches[0].text == "close chunk"
            assert resp.matches[0].match_type == "chunk"
            assert resp.matches[0].source_path == "memory/projects/foo.md"
        finally:
            kb.close()

    def test_path_scope_unknown_path_raises(self, tmp_path):
        kb = _make_kb(tmp_path, StubEmbedder())
        try:
            with pytest.raises(ValueError, match="not found"):
                kb.memory_similar("x", path="memory/does/not/exist.md")
        finally:
            kb.close()


class TestMemorySimilarCLI:
    """E2E CLI tests via Click test runner."""

    def test_cli_requires_text(self, runner, tmp_db):
        from tests.conftest import invoke_cli

        _, db_path = tmp_db
        result = invoke_cli(runner, ["memory", "similar"], db_path)
        # Click error → non-zero exit
        assert result.exit_code != 0

    def test_cli_entity_scope_outputs_json(self, runner, tmp_path, monkeypatch):
        """End-to-end: CLI command emits structured JSON via stubbed embedder."""
        from kb import api as api_module
        from tests.conftest import invoke_cli

        target = np.zeros(1024, dtype=np.float32)
        target[3] = 1.0
        embedder = StubEmbedder(mapping={"candidate text": target, "matching stored fact": target})

        project_root = tmp_path / "project"
        (project_root / "memory" / "people").mkdir(parents=True)
        (project_root / "memory" / "glossary.md").write_text("# Glossary\n")
        data_dir = tmp_path / "data"

        monkeypatch.setattr(api_module.KnowledgeBase, "_get_embedder", lambda self: embedder)
        monkeypatch.setenv("KB_PROJECT_ROOT", str(project_root))

        kb = KnowledgeBase(project_root=project_root, data_dir=data_dir)
        try:
            _seed_entity_with_facts(kb, "Wren", ["matching stored fact", "noise"])
        finally:
            kb.close()

        result = invoke_cli(
            runner,
            [
                "memory",
                "similar",
                "candidate text",
                "--entity",
                "Wren",
                "--threshold",
                "0.5",
                "--format",
                "json",
            ],
            data_dir,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["has_similar"] is True
        assert payload["matches"][0]["text"] == "matching stored fact"


class TestMcpMemorySimilar:
    """MCP handler tests."""

    def test_handler_entity_scope_returns_json(self, tmp_path, monkeypatch):
        from kb import api as api_module
        from kb.mcp_server import handle_kb_memory_similar

        target = np.zeros(1024, dtype=np.float32)
        target[3] = 1.0
        embedder = StubEmbedder(mapping={"q text": target, "stored similar fact": target})
        monkeypatch.setattr(api_module.KnowledgeBase, "_get_embedder", lambda self: embedder)

        project_root = tmp_path / "project"
        (project_root / "memory" / "people").mkdir(parents=True)
        (project_root / "memory" / "glossary.md").write_text("# Glossary\n")
        with tempfile.TemporaryDirectory() as data_dir:
            db = Database(Path(data_dir))
            kb = KnowledgeBase._from_existing(db=db, project_root=project_root)
            _seed_entity_with_facts(kb, "Soren", ["stored similar fact"])
            kb.close()

            result = handle_kb_memory_similar(
                db,
                project_root,
                text="q text",
                entity="Soren",
                path=None,
                threshold=0.5,
                limit=5,
            )
            db.close()

        payload = json.loads(result)
        assert payload["has_similar"] is True
        assert payload["matches"][0]["text"] == "stored similar fact"

    def test_handler_missing_text_returns_error(self, tmp_path):
        from kb.mcp_server import handle_kb_memory_similar

        with tempfile.TemporaryDirectory() as data_dir:
            db = Database(Path(data_dir))
            result = handle_kb_memory_similar(
                db,
                tmp_path,
                text="",
                entity=None,
                path=None,
                threshold=0.85,
                limit=5,
            )
            db.close()
        payload = json.loads(result)
        assert "error" in payload
