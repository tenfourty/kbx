# MLX Plugin

MLX acceleration provides faster embedding generation on Apple Silicon Macs using Apple's MLX framework. It replaces the default PyTorch/sentence-transformers backend with a custom model loader that runs on unified memory without the MPS allocator leak.

## Install

```bash
pip install "kbx[mlx]"
```

Requires Apple Silicon (arm64 macOS). On other platforms, the MLX extra installs but the backend is not activated.

## How It Works

**Model:** Qwen3-Embedding-0.6B (1024-dim, instruction-aware)

**Architecture:** The MLX backend (`src/kb/mlx_backend.py`) implements the full Qwen3 decoder-only transformer in pure MLX `nn.Module` classes:

- Token embeddings + 28 decoder layers + RMSNorm
- Grouped-query attention with QK-norm and RoPE
- SwiGLU feed-forward (SiLU gate)
- Last-token pooling + L2 normalization

**Weight loading:** Downloads via `huggingface_hub.snapshot_download()`, loads `.safetensors` files directly with `mx.load()`. The `sanitize()` method remaps HuggingFace weight names and drops unused `lm_head` weights.

**Dependencies:** Only MIT/Apache-2.0 libraries -- `mlx`, `transformers` (tokenizer), `huggingface_hub`, `safetensors`. No GPL dependencies.

## Backend Selection

The `Embedder` class in `embeddings.py` auto-selects the backend:

| Condition | Backend |
|-----------|---------|
| `device="cpu"` explicitly set | PyTorch |
| Apple Silicon + `mlx` importable | MLX |
| Everything else | PyTorch |

No configuration needed. Check the active backend:

```python
from kb.embeddings import Embedder
e = Embedder()
print(e.backend_name)  # "_MLXBackend" or "_PyTorchBackend"
```

## Batch Processing

| Backend | Default batch size | Memory management |
|---------|-------------------|-------------------|
| MLX | 64 | `mx.metal.clear_cache()` between batches |
| PyTorch (MPS) | 32 | `torch.mps.empty_cache()` between batches |
| PyTorch (CPU) | 16 | None needed |

Texts are truncated to 8,000 characters before embedding.

## Model Cache

First run downloads ~1.1 GB of model weights. Default location: `~/.cache/huggingface/hub/` or `~/.config/kbx/model/`. The cache is portable across runs.

## Instruction-Aware Embeddings

Qwen3 supports task-specific instructions for improved retrieval:

- **Indexing:** "Represent this document section for retrieval"
- **Queries:** "Find meeting notes, transcripts, and documents relevant to this query"
