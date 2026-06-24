"""Tests for embedding backend selection and output correctness."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestBackendSelection:
    def test_cpu_forces_pytorch(self):
        """device='cpu' always selects PyTorch backend."""
        from kb.embeddings import Embedder

        embedder = Embedder(device="cpu")
        assert embedder.backend_name == "_PyTorchBackend"

    def test_auto_selects_backend(self):
        """Default device selects a valid backend."""
        from kb.embeddings import Embedder

        embedder = Embedder()
        assert embedder.backend_name in ("_MLXBackend", "_PyTorchBackend")

    def test_mlx_preferred_on_apple_silicon(self):
        """On Apple Silicon with mlx installed, MLX is selected."""
        from kb.embeddings import Embedder, _is_apple_silicon, _mlx_available

        if not _is_apple_silicon() or not _mlx_available():
            pytest.skip("Not on Apple Silicon with MLX")

        embedder = Embedder()
        assert embedder.backend_name == "_MLXBackend"

    def test_fallback_without_mlx(self):
        """Without mlx installed, falls back to PyTorch."""
        from kb.embeddings import Embedder

        with (
            patch("kb.embeddings._mlx_available", return_value=False),
            patch("kb.embeddings._is_apple_silicon", return_value=True),
        ):
            embedder = Embedder()
            assert embedder.backend_name == "_PyTorchBackend"


@pytest.mark.slow
class TestEmbeddingOutput:
    """Tests that embedding outputs are correct regardless of backend.

    Requires the Qwen3-Embedding-0.6B model in the local HF cache.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_model(self):
        from conftest import _model_cached

        if not _model_cached():
            pytest.skip("Qwen3 model not in local cache")

    def test_embed_shape_and_dtype(self):
        """Embeddings have correct shape and dtype."""
        from kb.embeddings import EMBEDDING_DIM, Embedder

        embedder = Embedder()
        result = embedder.embed(["Hello world", "Test text"])
        assert result.shape == (2, EMBEDDING_DIM)
        assert result.dtype.name == "float32"

    def test_embed_single_text(self):
        """Single text produces shape (1, dim)."""
        from kb.embeddings import EMBEDDING_DIM, Embedder

        embedder = Embedder()
        result = embedder.embed(["Single text"])
        assert result.shape == (1, EMBEDDING_DIM)

    def test_query_embedding_shape(self):
        """Query embeddings have correct shape."""
        from kb.embeddings import EMBEDDING_DIM, Embedder

        embedder = Embedder()
        result = embedder.embed_query("test query")
        assert result.shape == (1, EMBEDDING_DIM)
        assert result.dtype.name == "float32"

    def test_embeddings_normalized(self):
        """Embeddings are approximately L2-normalized."""
        import numpy as np

        from kb.embeddings import Embedder

        embedder = Embedder()
        result = embedder.embed(["Hello world"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.02)

    def test_query_embeddings_normalized(self):
        """Query embeddings are approximately L2-normalized."""
        import numpy as np

        from kb.embeddings import Embedder

        embedder = Embedder()
        result = embedder.embed_query("test query")
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.02)


@pytest.mark.slow
class TestBackendParity:
    """MLX and PyTorch must produce interchangeable vectors, so a backend
    switch (Python 3.13<->3.14, mlx present<->absent) never silently corrupts
    an existing index. Documents and queries are checked separately.
    """

    @pytest.fixture(autouse=True)
    def _requires_mlx_and_model(self):
        import importlib.util

        from conftest import _model_cached

        from kb.embeddings import _is_apple_silicon

        if not _is_apple_silicon() or importlib.util.find_spec("mlx") is None:
            pytest.skip("Backend parity needs Apple Silicon with mlx")
        if not _model_cached():
            pytest.skip("Qwen3 model not cached locally")

    def test_documents_and_queries_match_across_backends(self):
        import numpy as np

        from kb.config import get_data_dir
        from kb.embeddings import _MLXBackend, _PyTorchBackend

        cache = get_data_dir() / "model"
        docs = [
            "MFA rollout status for the secrets manager squad",
            "Quarterly goal validation for the data platform team",
        ]
        query = "multi-factor authentication progress"

        mlx_doc = _MLXBackend(cache).embed(docs)
        pt_doc = _PyTorchBackend(cache, "cpu").embed(docs)
        mlx_q = _MLXBackend(cache).embed_query(query)
        pt_q = _PyTorchBackend(cache, "cpu").embed_query(query)

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        for i in range(len(docs)):
            sim = cos(mlx_doc[i], pt_doc[i])
            assert sim > 0.99, f"document {i} diverges across backends: cosine={sim:.4f}"
        sim_q = cos(mlx_q[0], pt_q[0])
        assert sim_q > 0.99, f"query diverges across backends: cosine={sim_q:.4f}"


class TestPlatformHelpers:
    def test_is_apple_silicon_returns_bool(self):
        from kb.embeddings import _is_apple_silicon

        assert isinstance(_is_apple_silicon(), bool)

    def test_mlx_available_returns_bool(self):
        from kb.embeddings import _mlx_available

        assert isinstance(_mlx_available(), bool)

    def test_mlx_not_available_without_package(self):
        """_mlx_available returns False when mlx.core can't import."""
        from kb.embeddings import _is_apple_silicon

        with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
            # Even on Apple Silicon, should be False without the package
            if _is_apple_silicon():
                # Need to reimport to pick up mocked module
                import importlib

                import kb.embeddings

                importlib.reload(kb.embeddings)
                assert not kb.embeddings._mlx_available()
                importlib.reload(kb.embeddings)  # restore


class TestMLXVersionGuard:
    """The pre-import version guard should track the *next* untested CPython.

    MLX 0.31.2+ ships cp314 wheels and runs correctly on Python 3.14, so 3.14
    must NOT be blocked. The guard exists only to pre-empt a C-level segfault on
    a CPython release MLX has not built wheels for yet — currently 3.15+.
    """

    @pytest.fixture
    def _fake_mlx_core(self):
        """Inject a fake mlx.core so `import mlx.core` succeeds everywhere."""
        from types import ModuleType

        fake_core = ModuleType("mlx.core")
        fake_mlx = ModuleType("mlx")
        fake_mlx.core = fake_core  # type: ignore[attr-defined]
        originals = {k: sys.modules.get(k) for k in ("mlx", "mlx.core")}
        sys.modules["mlx"] = fake_mlx
        sys.modules["mlx.core"] = fake_core
        yield
        for k, v in originals.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_allows_python_314(self, _fake_mlx_core):
        """Python 3.14 is supported by MLX and must be allowed."""
        from kb.embeddings import _mlx_available

        with (
            patch("kb.embeddings._is_apple_silicon", return_value=True),
            patch.object(sys, "version_info", (3, 14, 5)),
        ):
            assert _mlx_available() is True

    def test_blocks_python_315(self, _fake_mlx_core):
        """Python 3.15 is not yet verified for MLX and must be blocked."""
        from kb.embeddings import _mlx_available

        with (
            patch("kb.embeddings._is_apple_silicon", return_value=True),
            patch.object(sys, "version_info", (3, 15, 0)),
        ):
            assert _mlx_available() is False


class TestInstructionFormat:
    """Qwen3-Embedding asymmetry: queries get an Instruct/Query template,
    documents are embedded as raw text with no instruction (per the model card).
    Both backends must agree so their vectors are interchangeable.
    """

    def test_format_query_matches_qwen3_template(self):
        from kb.embeddings import QUERY_INSTRUCTION, _format_query

        assert _format_query("hello world") == f"Instruct: {QUERY_INSTRUCTION}\nQuery: hello world"

    def test_pytorch_documents_embed_without_instruction(self, tmp_path):
        """Documents go through model.encode with an empty prompt (raw text)."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=tmp_path, device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        backend._model = mock_model
        backend._is_mps = False

        @contextmanager
        def mock_inference_mode():
            yield

        with patch("torch.inference_mode", mock_inference_mode):
            backend.embed(["a document section"])

        assert mock_model.encode.call_args.kwargs["prompt"] == ""


# ---------------------------------------------------------------------------
# GPU memory release tests (mocked)
# ---------------------------------------------------------------------------


class TestGPUMemoryRelease:
    def test_release_noop_on_cpu(self):
        """release_gpu_memory does nothing when not on MPS."""
        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=Path("/tmp/test-cache"), device="cpu")
        backend._is_mps = False
        # Should not raise, and returns immediately
        backend.release_gpu_memory()

    def test_release_calls_mps_sync_and_cache(self):
        """release_gpu_memory calls torch.mps.synchronize and empty_cache."""
        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=Path("/tmp/test-cache"), device=None)
        backend._is_mps = True
        with (
            patch("torch.mps.synchronize") as mock_sync,
            patch("torch.mps.empty_cache") as mock_cache,
        ):
            backend.release_gpu_memory()
            mock_sync.assert_called_once()
            mock_cache.assert_called_once()

    def test_release_swallows_exceptions(self):
        """release_gpu_memory catches exceptions silently."""
        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=Path("/tmp/test-cache"), device=None)
        backend._is_mps = True
        with patch("torch.mps.synchronize", side_effect=RuntimeError("GPU error")):
            # Should not raise
            backend.release_gpu_memory()


# ---------------------------------------------------------------------------
# MPS batch splitting tests (mocked)
# ---------------------------------------------------------------------------


class TestMPSBatchSplitting:
    def test_mps_splits_large_batches(self):
        """On MPS, large batches are split with cache cleanup between."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=Path("/tmp/test-cache"), device=None)
        mock_model = MagicMock()
        backend._model = mock_model
        backend._is_mps = True

        # 5 texts with batch_size=2 -> 3 encode calls
        mock_model.encode.side_effect = [
            np.random.randn(2, 1024).astype(np.float32),
            np.random.randn(2, 1024).astype(np.float32),
            np.random.randn(1, 1024).astype(np.float32),
        ]

        @contextmanager
        def mock_inference_mode():
            yield

        with (
            patch("torch.inference_mode", mock_inference_mode),
            patch("torch.mps.synchronize"),
            patch("torch.mps.empty_cache"),
        ):
            result = backend.embed(["a", "b", "c", "d", "e"], batch_size=2)

        assert result.shape == (5, 1024)
        assert mock_model.encode.call_count == 3

    def test_cpu_no_splitting(self):
        """On CPU, all texts go in single encode call."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=Path("/tmp/test-cache"), device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 1024).astype(np.float32)
        backend._model = mock_model
        backend._is_mps = False

        @contextmanager
        def mock_inference_mode():
            yield

        with patch("torch.inference_mode", mock_inference_mode):
            result = backend.embed(["a", "b", "c", "d", "e"], batch_size=2)

        assert result.shape == (5, 1024)
        # CPU path: single encode() call regardless of batch_size
        assert mock_model.encode.call_count == 1

    def test_mps_small_batch_no_splitting(self):
        """On MPS, batch smaller than batch_size uses single encode call."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=Path("/tmp/test-cache"), device=None)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 1024).astype(np.float32)
        backend._model = mock_model
        backend._is_mps = True

        @contextmanager
        def mock_inference_mode():
            yield

        with patch("torch.inference_mode", mock_inference_mode):
            result = backend.embed(["a", "b"], batch_size=10)

        assert result.shape == (2, 1024)
        # Small batch on MPS: single encode call (len <= batch_size)
        assert mock_model.encode.call_count == 1


# ---------------------------------------------------------------------------
# MLX backend tests (mocked)
# ---------------------------------------------------------------------------


class TestPyTorchLoadMocked:
    """Tests for _PyTorchBackend._load() with mocked SentenceTransformer."""

    def test_load_creates_model(self, tmp_path):
        """_load() creates SentenceTransformer with correct args."""
        from unittest.mock import MagicMock

        from kb.embeddings import MODEL_NAME, _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=tmp_path, device="cpu")
        mock_st_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_st_cls.return_value = mock_model

        with (
            patch("kb.embeddings.SentenceTransformer", mock_st_cls, create=True),
            patch.dict(sys.modules, {}),
        ):
            # Direct approach: mock the import
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "sentence_transformers":
                    mod = MagicMock()
                    mod.SentenceTransformer = mock_st_cls
                    return mod
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                model = backend._load()

        assert model is mock_model
        mock_st_cls.assert_called_once()
        call_kwargs = mock_st_cls.call_args
        assert call_kwargs[0][0] == MODEL_NAME
        assert call_kwargs[1]["device"] == "cpu"

    def test_load_detects_mps_device(self, tmp_path):
        """_load() sets _is_mps=True when model.device is 'mps'."""
        from unittest.mock import MagicMock

        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=tmp_path, device=None)
        mock_model = MagicMock()
        mock_model.device = "mps"
        mock_st_cls = MagicMock(return_value=mock_model)

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                mod = MagicMock()
                mod.SentenceTransformer = mock_st_cls
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            backend._load()

        assert backend._is_mps is True

    def test_embed_query_with_mocked_model(self, tmp_path):
        """embed_query() encodes the Instruct/Query template as raw text (empty prompt)."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _format_query, _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=tmp_path, device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        backend._model = mock_model
        backend._is_mps = False

        @contextmanager
        def mock_inference_mode():
            yield

        with patch("torch.inference_mode", mock_inference_mode):
            result = backend.embed_query("test query")

        assert result.shape == (1, 1024)
        mock_model.encode.assert_called_once()
        call = mock_model.encode.call_args
        assert call.args[0] == [_format_query("test query")]
        assert call.kwargs["prompt"] == ""

    def test_default_batch_size_cpu(self, tmp_path):
        """Default batch_size is 16 on CPU."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _PyTorchBackend

        backend = _PyTorchBackend(cache_dir=tmp_path, device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        backend._model = mock_model
        backend._is_mps = False

        @contextmanager
        def mock_inference_mode():
            yield

        with patch("torch.inference_mode", mock_inference_mode):
            backend.embed(["test"])  # batch_size=None -> should default

        call_kwargs = mock_model.encode.call_args
        assert call_kwargs[1]["batch_size"] == 16


class TestEmbedderDelegation:
    """Tests that Embedder correctly delegates to its backend."""

    def test_embed_delegates_to_backend(self):
        """Embedder.embed() calls backend.embed()."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import Embedder

        embedder = Embedder(device="cpu")
        mock_backend = MagicMock()
        mock_backend.embed.return_value = np.random.randn(1, 1024).astype(np.float32)
        embedder._backend = mock_backend

        result = embedder.embed(["test"])
        assert result.shape == (1, 1024)
        mock_backend.embed.assert_called_once()

    def test_embed_query_delegates_to_backend(self):
        """Embedder.embed_query() calls backend.embed_query()."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import Embedder

        embedder = Embedder(device="cpu")
        mock_backend = MagicMock()
        mock_backend.embed_query.return_value = np.random.randn(1, 1024).astype(np.float32)
        embedder._backend = mock_backend

        result = embedder.embed_query("test query")
        assert result.shape == (1, 1024)
        mock_backend.embed_query.assert_called_once()

    def test_release_gpu_delegates_to_backend(self):
        """Embedder.release_gpu_memory() calls backend.release_gpu_memory()."""
        from unittest.mock import MagicMock

        from kb.embeddings import Embedder

        embedder = Embedder(device="cpu")
        mock_backend = MagicMock()
        embedder._backend = mock_backend

        embedder.release_gpu_memory()
        mock_backend.release_gpu_memory.assert_called_once()


class TestMLXBackendMocked:
    @pytest.fixture(autouse=True)
    def _fake_mlx(self):
        """Inject fake mlx modules so patch targets resolve on non-Apple platforms."""
        import sys
        from types import ModuleType

        fake_metal = ModuleType("mlx.core.metal")
        fake_metal.clear_cache = lambda: None  # type: ignore[attr-defined]
        fake_core = ModuleType("mlx.core")
        fake_core.metal = fake_metal  # type: ignore[attr-defined]
        fake_core.array = lambda x: x  # type: ignore[attr-defined]
        fake_core.eval = lambda *a: None  # type: ignore[attr-defined]
        fake_core.clear_cache = lambda: None  # type: ignore[attr-defined]
        fake_core.float32 = "float32"  # type: ignore[attr-defined]
        fake_mlx = ModuleType("mlx")
        fake_mlx.core = fake_core  # type: ignore[attr-defined]
        originals = {k: sys.modules.get(k) for k in ("mlx", "mlx.core", "mlx.core.metal")}
        sys.modules["mlx"] = fake_mlx
        sys.modules["mlx.core"] = fake_core
        sys.modules["mlx.core.metal"] = fake_metal
        yield
        for k, v in originals.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_mlx_selected_when_available(self):
        """MLX backend selected when _mlx_available() returns True."""
        from kb.embeddings import Embedder

        with (
            patch("kb.embeddings._mlx_available", return_value=True),
            patch("kb.embeddings._is_apple_silicon", return_value=True),
        ):
            embedder = Embedder()
            assert embedder.backend_name == "_MLXBackend"

    def test_mlx_release_gpu_memory(self):
        """MLX release_gpu_memory calls mx.clear_cache."""
        from kb.embeddings import _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)
        with patch("mlx.core.clear_cache") as mock_clear:
            backend.release_gpu_memory()
            mock_clear.assert_called_once()

    def test_mlx_release_swallows_exception(self):
        """MLX release_gpu_memory catches exceptions."""
        from kb.embeddings import _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)
        with patch("mlx.core.clear_cache", side_effect=RuntimeError("Metal error")):
            # Should not raise
            backend.release_gpu_memory()

    def test_mlx_load(self, tmp_path):
        """_MLXBackend._load() calls kb.mlx_backend.load_mlx_model."""
        from unittest.mock import MagicMock

        from kb.embeddings import _MLXBackend

        backend = _MLXBackend(cache_dir=tmp_path)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "kb.mlx_backend.load_mlx_model",
            return_value=(mock_model, mock_tokenizer),
        ):
            model, tokenizer = backend._load()

        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_mlx_tokenize_and_run(self, tmp_path):
        """_tokenize_and_run() tokenizes, runs model, returns numpy float32."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer

        # Mock tokenizer output
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }

        # Mock model output with text_embeds
        mock_output = MagicMock()
        fake_embeds = MagicMock()
        fake_float32 = MagicMock()
        fake_float32.__array__ = MagicMock(return_value=np.random.randn(1, 1024).astype(np.float32))
        fake_embeds.astype.return_value = fake_float32
        mock_output.text_embeds = fake_embeds
        mock_model.return_value = mock_output

        with (
            patch("mlx.core.array", side_effect=lambda x: x),
            patch("mlx.core.eval"),
            patch("numpy.array", return_value=np.random.randn(1, 1024).astype(np.float32)),
        ):
            result = backend._tokenize_and_run(["test text"])

        assert result.shape == (1, 1024)

    def test_mlx_embed_batched(self, tmp_path):
        """_MLXBackend.embed() handles batching and cache clearing."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()

        # Mock _tokenize_and_run to return fake embeddings
        fake_emb = np.random.randn(2, 1024).astype(np.float32)
        with (
            patch.object(backend, "_tokenize_and_run", return_value=fake_emb),
            patch.object(backend, "release_gpu_memory"),
        ):
            result = backend.embed(["a", "b"], batch_size=64)

        assert result.shape == (2, 1024)

    def test_mlx_embed_large_batch_splits(self, tmp_path):
        """Large batches are split with cache clearing between."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()

        batch1 = np.random.randn(2, 1024).astype(np.float32)
        batch2 = np.random.randn(1, 1024).astype(np.float32)

        with (
            patch.object(backend, "_tokenize_and_run", side_effect=[batch1, batch2]),
            patch.object(backend, "release_gpu_memory") as mock_release,
        ):
            result = backend.embed(["a", "b", "c"], batch_size=2)

        assert result.shape == (3, 1024)
        mock_release.assert_called()

    def test_mlx_embed_query(self, tmp_path):
        """_MLXBackend.embed_query() tokenizes the Instruct/Query template."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _format_query, _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()

        fake_emb = np.random.randn(1, 1024).astype(np.float32)
        with patch.object(backend, "_tokenize_and_run", return_value=fake_emb) as mock_run:
            result = backend.embed_query("test query")

        assert result.shape == (1, 1024)
        mock_run.assert_called_once_with([_format_query("test query")])

    def test_mlx_documents_tokenize_raw(self):
        """Documents are tokenized as raw text — no Instruct/Query wrapping."""
        from unittest.mock import MagicMock

        import numpy as np

        from kb.embeddings import _MLXBackend

        backend = _MLXBackend.__new__(_MLXBackend)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }
        mock_output = MagicMock()
        fake_embeds = MagicMock()
        fake_embeds.astype.return_value = MagicMock()
        mock_output.text_embeds = fake_embeds
        mock_model.return_value = mock_output

        with (
            patch("mlx.core.array", side_effect=lambda x: x),
            patch("mlx.core.eval"),
            patch("numpy.array", return_value=np.random.randn(1, 1024).astype(np.float32)),
        ):
            backend.embed(["a raw document section"], batch_size=64)

        assert mock_tokenizer.call_args.args[0] == ["a raw document section"]
