"""Tests for custom MLX model loader (replaces GPL mlx-embeddings)."""

import numpy as np

from kb.mlx_backend import ModelOutput, Qwen3Config, l2_normalize_np, last_token_pool_np


def test_last_token_pool_right_padded():
    """last_token_pool extracts the embedding at the last non-pad position."""
    # shape: (2, 4, 3) — 2 sequences, 4 tokens, dim 3
    hidden = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],  # pad at pos 3
            [[10, 11, 12], [13, 14, 15], [0, 0, 0], [0, 0, 0]],  # pad at pos 2,3
        ],
        dtype=np.float32,
    )
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int32)

    result = last_token_pool_np(hidden, mask)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[0], [7, 8, 9])
    np.testing.assert_array_equal(result[1], [13, 14, 15])


def test_last_token_pool_no_padding():
    """last_token_pool works when there is no padding (all tokens valid)."""
    hidden = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
    mask = np.array([[1, 1, 1]], dtype=np.int32)

    result = last_token_pool_np(hidden, mask)
    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result[0], [5, 6])


def test_l2_normalize():
    """l2_normalize returns unit vectors."""
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = l2_normalize_np(vecs)
    np.testing.assert_allclose(normed[0], [0.6, 0.8], atol=1e-6)
    # Zero vector stays zero (no div-by-zero)
    np.testing.assert_allclose(np.linalg.norm(normed[1]), 0.0, atol=1e-9)


def test_l2_normalize_already_unit():
    """l2_normalize is idempotent on unit vectors."""
    vecs = np.array([[0.6, 0.8]], dtype=np.float32)
    normed = l2_normalize_np(vecs)
    np.testing.assert_allclose(np.linalg.norm(normed[0]), 1.0, atol=1e-6)


class TestQwen3Config:
    """Tests for Qwen3Config dataclass."""

    def test_defaults(self):
        """Default config matches Qwen3-Embedding-0.6B architecture."""
        cfg = Qwen3Config()
        assert cfg.hidden_size == 1024
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.vocab_size == 151669

    def test_post_init_fills_kv_heads(self):
        """num_key_value_heads defaults to num_attention_heads."""
        cfg = Qwen3Config(num_attention_heads=16)
        assert cfg.num_key_value_heads == 16

    def test_post_init_fills_head_dim(self):
        """head_dim defaults to hidden_size // num_attention_heads."""
        cfg = Qwen3Config(hidden_size=1024, num_attention_heads=16)
        assert cfg.head_dim == 64

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict filters out keys not in the dataclass."""
        d = {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "unknown_key": "should_be_ignored",
            "another_unknown": 42,
        }
        cfg = Qwen3Config.from_dict(d)
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 12
        assert not hasattr(cfg, "unknown_key")

    def test_from_dict_partial(self):
        """from_dict with partial keys uses defaults for the rest."""
        cfg = Qwen3Config.from_dict({"hidden_size": 768})
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 28  # default


def test_mlx_backend_does_not_import_mlx_embeddings():
    """Verify no GPL dependency is imported."""
    import ast
    from pathlib import Path

    source = Path("src/kb/embeddings.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "mlx_embeddings" not in node.module, (
                    f"Found GPL import: {node.module} at line {node.lineno}"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mlx_embeddings" not in alias.name


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_default_none(self):
        """ModelOutput fields default to None."""
        out = ModelOutput()
        assert out.text_embeds is None
        assert out.last_hidden_state is None

    def test_with_values(self):
        """ModelOutput stores provided arrays."""
        embeds = np.array([[1.0, 2.0]])
        hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        out = ModelOutput(text_embeds=embeds, last_hidden_state=hidden)
        np.testing.assert_array_equal(out.text_embeds, embeds)
        np.testing.assert_array_equal(out.last_hidden_state, hidden)
