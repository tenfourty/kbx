"""Embedding model wrapper — Qwen3-Embedding-0.6B via pluggable backends.

Backend selection:
  - Apple Silicon + mlx installed -> _MLXBackend  (custom loader in kb.mlx_backend)
  - Otherwise (or device="cpu")   -> _PyTorchBackend
"""

from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt
    from sentence_transformers import SentenceTransformer

# Silence HuggingFace telemetry/auth warnings (model is cached locally)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# MPS (Apple Silicon GPU): disable hard memory cap so PyTorch spills to
# system RAM rather than raising OOM.  The real memory management happens in
# embed() which clears the MPS cache between forward passes to work around
# the PyTorch MPS allocator leak (pytorch#154329).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
# Safety cap: truncate at ~2000 tokens to prevent MPS OOM on long texts
MAX_TEXT_CHARS = 8_000

# Qwen3-Embedding is asymmetric: a query is wrapped in an instruction template,
# a document is embedded as RAW text (per the model card: "no need to add
# instruction for retrieval documents").  A per-task query instruction adds
# +1-5% retrieval.  Both backends MUST format identically so their vectors are
# interchangeable — see _format_query and the parity test.
QUERY_INSTRUCTION = "Find meeting notes, transcripts, and documents relevant to this query"


def _format_query(query: str) -> str:
    """Wrap a search query in Qwen3's ``Instruct: {task}\\nQuery: {q}`` template.

    Documents are never wrapped (they are embedded raw), so a query and the
    document it should match land in the embedding space the model was trained
    on.  Used identically by both backends.
    """
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon (arm64 macOS)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    """Return True if Apple Silicon AND mlx is importable.

    MLX's C extension can fatal-abort (segfault) on a CPython release it has
    not shipped wheels for yet, so we guard with a pre-import version check —
    a catchable ImportError the try/except below would handle, but a segfault
    it would not.

    MLX 0.31.2+ ships cp314 wheels and runs correctly on Python 3.14 (verified
    GPU compute + Qwen3 embedding parity), so the guard tracks the next
    *untested* major, 3.15.  When mlx is merely absent or fails a catchable
    import, the try/except falls back to PyTorch.
    """
    if not _is_apple_silicon():
        return False
    # Pre-import canary: block only CPython releases MLX has no wheels for yet.
    if sys.version_info >= (3, 15):
        return False
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _EmbedderBackend(Protocol):
    """Protocol that all embedding backends must satisfy."""

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> npt.NDArray[np.float32]: ...

    def embed_query(self, query: str) -> npt.NDArray[np.float32]: ...

    def release_gpu_memory(self) -> None: ...


# ---------------------------------------------------------------------------
# PyTorch backend (sentence-transformers)
# ---------------------------------------------------------------------------


class _PyTorchBackend:
    """Embedding backend using sentence-transformers + PyTorch."""

    def __init__(self, cache_dir: Path, device: str | None) -> None:
        self._model: SentenceTransformer | None = None
        self._cache_dir = cache_dir
        self._device = device
        self._is_mps = False

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "trust_remote_code": False,
                "cache_folder": str(self._cache_dir),
            }
            if self._device:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(MODEL_NAME, **kwargs)
            self._is_mps = str(self._model.device) == "mps"
        return self._model

    def release_gpu_memory(self) -> None:
        """Synchronize MPS and release cached GPU memory."""
        if not self._is_mps:
            return
        try:
            import torch

            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a list of documents for indexing.

        Returns L2-normalized float32 array of shape (n, 1024).
        Documents are embedded as raw text (empty prompt) per the Qwen3 model
        card; only queries carry an instruction. batch_size defaults to 32 on
        MPS, 16 on CPU.

        On MPS, processes batch_size texts per encode() call with cache
        cleanup between calls to work around the PyTorch MPS memory leak
        (pytorch#154329). Without this, the MPS allocator accumulates
        cached GPU memory across forward passes within a single encode()
        call until the system runs out of RAM and locks up.
        """
        import numpy as np

        model = self._load()
        if batch_size is None:
            batch_size = 32 if self._is_mps else 16
        truncated = [t[:MAX_TEXT_CHARS] for t in texts]

        import torch

        with torch.inference_mode():
            if not self._is_mps or len(truncated) <= batch_size:
                # CPU or small batch: single encode() call is fine
                embeddings = model.encode(
                    truncated,
                    prompt="",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                return np.asarray(embeddings, dtype=np.float32)

            # MPS: chunk texts into batch_size groups, encode each separately,
            # and clear GPU cache between calls to prevent memory accumulation.
            chunks = []
            for i in range(0, len(truncated), batch_size):
                chunk = truncated[i : i + batch_size]
                embs = model.encode(
                    chunk,
                    prompt="",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                chunks.append(np.asarray(embs, dtype=np.float32))
                # Release MPS cached memory between forward passes
                torch.mps.synchronize()
                torch.mps.empty_cache()

            return np.concatenate(chunks, axis=0)

    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        """Embed a search query.

        Returns L2-normalized float32 array of shape (1, 1024).
        The query is wrapped in Qwen3's instruction template and encoded as
        raw text (empty prompt) so it matches the raw-text document space.
        """
        import numpy as np
        import torch

        model = self._load()
        with torch.inference_mode():
            embeddings = model.encode(
                [_format_query(query)],
                prompt="",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon — unified memory, no MPS leak)
# ---------------------------------------------------------------------------


class _MLXBackend:
    """MLX backend for Apple Silicon — unified memory, no MPS leak.

    Uses a custom model loader (kb.mlx_backend) that depends only on
    MIT/Apache-2.0 libraries (mlx, transformers, safetensors).  Tokenizes
    with the HuggingFace tokenizer directly and calls the MLX model.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None  # raw HF tokenizer

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from kb.mlx_backend import load_mlx_model

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            model, tokenizer = load_mlx_model(MODEL_NAME, cache_dir=self._cache_dir)
            self._model = model
            self._tokenizer = tokenizer
        return self._model, self._tokenizer

    def _tokenize_and_run(self, texts: list[str], max_length: int = 512) -> npt.NDArray[np.float32]:
        """Tokenize texts as-is, run through the MLX model, return numpy float32.

        Callers pre-format their input: documents are passed raw, queries are
        wrapped with ``_format_query`` before reaching here.
        """
        import mlx.core as mx
        import numpy as np

        model, tokenizer = self._load()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Cast to float32 in MLX before converting to numpy — the model
        # may use bfloat16 internally, and numpy's buffer protocol does
        # not support bfloat16 directly.
        embeds = output.text_embeds.astype(mx.float32)
        # mx.eval forces lazy MLX computation to complete (not Python eval)
        mx.eval(embeds)
        # copy=True so numpy owns the memory and MLX arrays can be freed
        result = np.array(embeds, copy=True)
        # Explicitly delete MLX intermediates so their memory can be reclaimed
        del embeds, output, input_ids, attention_mask
        return result

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed documents using MLX (raw text). Returns float32 (n, 1024)."""
        import numpy as np

        truncated = [t[:MAX_TEXT_CHARS] for t in texts]

        if batch_size is None:
            batch_size = 64

        all_embeddings = []
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i : i + batch_size]
            all_embeddings.append(self._tokenize_and_run(batch))
            # Clear Metal cache between batches to prevent memory accumulation
            if len(truncated) > batch_size:
                self.release_gpu_memory()

        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.concatenate(all_embeddings, axis=0)

    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        """Embed a search query wrapped in Qwen3's instruction template."""
        return self._tokenize_and_run([_format_query(query)])

    def release_gpu_memory(self) -> None:
        """Clear MLX Metal cache to release unified memory."""
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


class Embedder:
    """Lazy-loading embedding model with automatic backend selection.

    Selects the best available backend:
      - device="cpu"                    -> PyTorch
      - Apple Silicon + mlx installed   -> MLX  (custom loader in kb.mlx_backend)
      - else                            -> PyTorch

    Public API is unchanged regardless of which backend is active.
    """

    def __init__(self, cache_dir: Path | None = None, device: str | None = None) -> None:
        if cache_dir is None:
            from kb.config import get_data_dir

            cache_dir = get_data_dir() / "model"
        self._cache_dir = cache_dir
        self._device = device  # None = auto, "cpu" = force CPU
        self._backend: _EmbedderBackend | None = None

    def _get_backend(self) -> _EmbedderBackend:
        """Lazy-initialise and return the active backend."""
        if self._backend is None:
            backend: _EmbedderBackend
            if self._device == "cpu":
                backend = _PyTorchBackend(self._cache_dir, self._device)
            elif _mlx_available():
                backend = _MLXBackend(self._cache_dir)
            else:
                backend = _PyTorchBackend(self._cache_dir, self._device)
            self._backend = backend
        return self._backend

    @property
    def backend_name(self) -> str:
        """Return the class name of the active backend."""
        return type(self._get_backend()).__name__

    # -- delegated methods --------------------------------------------------

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a list of documents for indexing (raw text, no instruction).

        Returns L2-normalized float32 array of shape (n, 1024).
        """
        return self._get_backend().embed(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        """Embed a search query.

        Returns L2-normalized float32 array of shape (1, 1024).
        """
        return self._get_backend().embed_query(query)

    def release_gpu_memory(self) -> None:
        """Release GPU memory held by the active backend."""
        return self._get_backend().release_gpu_memory()
