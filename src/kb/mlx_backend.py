"""Custom MLX model loader for Qwen3-Embedding — replaces GPL mlx-embeddings.

Only depends on MIT/Apache-2.0 libraries:
  - mlx (MIT) — Apple's ML framework
  - huggingface_hub (Apache-2.0) — model download
  - transformers (Apache-2.0) — tokenizer
  - safetensors (Apache-2.0) — weight format

Architecture: Qwen3 decoder-only transformer with last-token pooling and L2 norm.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pure-numpy helpers (testable without MLX)
# ---------------------------------------------------------------------------


def last_token_pool_np(
    hidden_states: Any,
    attention_mask: Any,
) -> Any:
    """Extract the embedding at the last non-pad position for each sequence.

    Args:
        hidden_states: shape (batch, seq_len, dim)
        attention_mask: shape (batch, seq_len), 1 = valid token, 0 = pad

    Returns:
        Pooled embeddings, shape (batch, dim)
    """
    import numpy as np

    # sequence_lengths[i] = index of last valid token in sequence i
    sequence_lengths = np.sum(attention_mask, axis=1).astype(np.intp) - 1
    batch_indices = np.arange(hidden_states.shape[0])
    return hidden_states[batch_indices, sequence_lengths]


def l2_normalize_np(embeddings: Any) -> Any:
    """L2-normalize each row. Zero vectors remain zero (no div-by-zero).

    Args:
        embeddings: shape (batch, dim)

    Returns:
        Unit-norm embeddings, shape (batch, dim)
    """
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    # Avoid division by zero: replace 0 norms with 1 (result will be zero vector)
    safe_norms = np.maximum(norms, 1e-9)
    return embeddings / safe_norms


# ---------------------------------------------------------------------------
# Model output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelOutput:
    """Output from Qwen3ForEmbedding — matches mlx-embeddings BaseModelOutput API."""

    text_embeds: Any = None  # pooled + L2-normalized, shape (batch, dim)
    last_hidden_state: Any = None  # full sequence, shape (batch, seq_len, dim)


# ---------------------------------------------------------------------------
# Qwen3 config
# ---------------------------------------------------------------------------


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 embedding model."""

    model_type: str = "qwen3"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    vocab_size: int = 151669
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Qwen3Config:
        """Build config from HuggingFace config.json dict, ignoring unknown keys."""
        import inspect

        valid_keys = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# MLX nn.Module classes — Qwen3 architecture
# ---------------------------------------------------------------------------


def _build_qwen3_model(config: Qwen3Config) -> Any:  # pragma: no cover
    """Build the Qwen3 embedding model using MLX nn.Module classes.

    Returns a Qwen3ForEmbedding instance (nn.Module) ready for weight loading.
    The model's __call__ returns a ModelOutput with .text_embeds.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""

        def __init__(self, dims: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = mx.ones((dims,))
            self.eps = eps

        def __call__(self, x: mx.array) -> mx.array:
            return mx.fast.rms_norm(x, self.weight, self.eps)

    class Qwen3MLP(nn.Module):
        """SwiGLU feed-forward: SiLU(gate(x)) * up(x), then down."""

        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class Qwen3Attention(nn.Module):
        """Grouped-query attention with QK-norm and RoPE."""

        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            self.num_heads = cfg.num_attention_heads
            self.num_kv_heads = cfg.num_key_value_heads  # guaranteed int by __post_init__
            self.head_dim = cfg.head_dim  # guaranteed int by __post_init__
            self.num_kv_groups = self.num_heads // self.num_kv_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)

            self.q_proj = nn.Linear(
                cfg.hidden_size, self.num_heads * self.head_dim, bias=cfg.attention_bias
            )
            self.k_proj = nn.Linear(
                cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias
            )
            self.v_proj = nn.Linear(
                cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias
            )

            # QK-norm (Qwen3 applies RMSNorm to Q and K before attention)
            self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

            # Rotary position embeddings
            self.rotary_emb = nn.RoPE(
                self.head_dim,
                traditional=False,
                base=cfg.rope_theta,
            )

        def __call__(
            self,
            hidden_states: mx.array,
            attention_mask: mx.array | None = None,
        ) -> mx.array:
            bsz, seq_len, _ = hidden_states.shape

            # Project to Q, K, V
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # Reshape: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
            q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            # QK-norm
            q = self.q_norm(q)
            k = self.k_norm(k)

            # RoPE
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

            # Expand KV for grouped-query attention
            if self.num_kv_groups > 1:
                k = mx.repeat(k, self.num_kv_groups, axis=1)
                v = mx.repeat(v, self.num_kv_groups, axis=1)

            # Scaled dot-product attention
            attn_output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=attention_mask
            )

            # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden_size)
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
                bsz, seq_len, self.num_heads * self.head_dim
            )
            return self.o_proj(attn_output)

    class Qwen3DecoderLayer(nn.Module):
        """Transformer decoder layer: pre-norm attention + pre-norm MLP with residuals."""

        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            self.self_attn = Qwen3Attention(cfg)
            self.mlp = Qwen3MLP(cfg)
            self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        def __call__(
            self,
            hidden_states: mx.array,
            attention_mask: mx.array | None = None,
        ) -> mx.array:
            # Self-attention with residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
            hidden_states = residual + hidden_states

            # MLP with residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states

    class Qwen3EmbeddingModel(nn.Module):
        """Core Qwen3 transformer: embed_tokens + N decoder layers + final norm."""

        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
            self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        def _create_causal_mask(self, seq_len: int, dtype: mx.Dtype) -> mx.array:
            """Causal (lower-triangular) mask: 0 for valid, -inf for masked."""
            mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
            mask = mx.where(mask, 0.0, -mx.inf).astype(dtype)
            return mx.expand_dims(mask, axis=(0, 1))  # (1, 1, seq, seq)

        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
        ) -> mx.array:
            _batch_size, seq_len = input_ids.shape
            hidden = self.embed_tokens(input_ids)

            # Build combined causal + padding mask
            if attention_mask is None:
                combined_mask = self._create_causal_mask(seq_len, hidden.dtype)
            elif attention_mask.ndim == 2:
                # padding_mask: (batch, 1, 1, seq) — 0 for valid, -inf for pad
                pad_mask = attention_mask[:, None, None, :]
                pad_mask = mx.where(pad_mask == 0, -mx.inf, 0.0).astype(hidden.dtype)
                causal_mask = self._create_causal_mask(seq_len, hidden.dtype)
                combined_mask = causal_mask + pad_mask  # broadcast
            else:
                combined_mask = attention_mask

            for layer in self.layers:
                hidden = layer(hidden, attention_mask=combined_mask)

            return self.norm(hidden)

    class Qwen3ForEmbedding(nn.Module):
        """Top-level embedding model: transformer + last-token pool + L2 norm.

        __call__ returns a ModelOutput with .text_embeds (matches mlx-embeddings API).
        """

        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            self.config = cfg
            self.model = Qwen3EmbeddingModel(cfg)

        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
        ) -> ModelOutput:
            batch_size, seq_len = input_ids.shape

            if attention_mask is None:
                attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

            last_hidden = self.model(input_ids, attention_mask=attention_mask)

            # Last-token pooling
            pooled = _last_token_pool_mx(last_hidden, attention_mask)

            # L2 normalize
            text_embeds = _l2_normalize_mx(pooled)

            return ModelOutput(text_embeds=text_embeds, last_hidden_state=last_hidden)

        def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
            """Map weight names from HF checkpoint format to our model structure.

            Handles:
              - Dropping lm_head weights (not used for embeddings)
              - Remapping 'transformer.' prefix to 'model.' if present
            """
            sanitized: dict[str, Any] = {}
            for key, value in weights.items():
                if "lm_head.weight" in key:
                    continue

                new_key = key
                if key.startswith("transformer."):
                    new_key = key.replace("transformer.", "model.", 1)
                elif not key.startswith("model.") and "." in key:
                    new_key = f"model.{key}"

                sanitized[new_key] = value
            return sanitized

    return Qwen3ForEmbedding(config)


# ---------------------------------------------------------------------------
# MLX helpers (used inside the model)
# ---------------------------------------------------------------------------


def _last_token_pool_mx(last_hidden_states: Any, attention_mask: Any) -> Any:  # pragma: no cover
    """Last-token pooling in MLX — picks the embedding at the last non-pad position."""
    import mlx.core as mx

    # Check for left-padding (all sequences end with valid tokens)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(axis=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[mx.arange(batch_size), sequence_lengths]


def _l2_normalize_mx(embeddings: Any) -> Any:  # pragma: no cover
    """L2-normalize embeddings in MLX."""
    import mlx.core as mx

    norms = mx.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    return embeddings / mx.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Public API: load_mlx_model()
# ---------------------------------------------------------------------------


def load_mlx_model(  # pragma: no cover
    model_name: str,
    cache_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Download and load a Qwen3 embedding model using MLX.

    This replaces the GPL ``mlx_embeddings.utils.load()`` function. It:
      1. Downloads the model via huggingface_hub.snapshot_download()
      2. Reads config.json from the model directory
      3. Loads all .safetensors weight files via mx.load()
      4. Builds the Qwen3 model architecture
      5. Loads weights, sets model to inference mode
      6. Loads the tokenizer via AutoTokenizer

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen3-Embedding-0.6B")
        cache_dir: Optional cache directory for model download

    Returns:
        (model, tokenizer) tuple where:
          - model: Qwen3ForEmbedding nn.Module (callable, returns ModelOutput with .text_embeds)
          - tokenizer: HuggingFace PreTrainedTokenizer (raw, not wrapped)
    """
    import glob as glob_mod

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # 1. Download model (or use cached version)
    dl_kwargs: dict[str, Any] = {
        "repo_id": model_name,
        "allow_patterns": [
            "*.json",
            "*.safetensors",
            "*.tiktoken",
            "*.txt",
            "*.model",
        ],
    }
    if cache_dir is not None:
        dl_kwargs["cache_dir"] = str(cache_dir)

    model_path = Path(snapshot_download(**dl_kwargs))  # nosec B615

    # 2. Read config
    with open(model_path / "config.json") as f:
        config_dict = json.load(f)

    config = Qwen3Config.from_dict(config_dict)

    # 3. Load safetensor weights
    weight_files = sorted(glob_mod.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob_mod.glob(str(model_path / "weight*.safetensors")))
    if not weight_files:
        msg = f"No safetensors found in {model_path}"
        raise FileNotFoundError(msg)

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # 4. Build model
    model = _build_qwen3_model(config)

    # 5. Sanitize and load weights
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    # mx.eval forces lazy MLX computation — materialises all parameter tensors.
    # NOTE: mx.eval is MLX's graph evaluator, NOT Python's built-in eval().
    mx.eval(model.parameters())
    model.eval()  # set to inference mode (disables dropout)

    # 6. Load tokenizer (raw HF tokenizer, no wrapper)
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # nosec B615

    return model, tokenizer
