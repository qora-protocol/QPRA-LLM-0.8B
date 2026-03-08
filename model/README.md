---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language:
  - en
  - zh
  - multilingual
library_name: rust
tags:
  - text-generation
  - image-text-to-text
  - video-text-to-text
  - multimodal
  - vision
  - rust
  - pure-rust
  - no-python
  - quantized
  - deltanet
  - hybrid-attention
  - mobile
pipeline_tag: image-text-to-text
model-index:
  - name: QORA-0.8B
    results: []
---

# QORA-0.8B

Pure Rust multimodal inference engine based on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). No Python, no CUDA, no external ML frameworks. Single executable + model weights = portable AI that runs on any machine.

**Designed for mobile and edge devices** — only 600 MB model file, loads in under 1 second, and runs at ~4 tok/s on a standard CPU. **Smart system awareness** — automatically detects your hardware (RAM, CPU threads) on Windows, Linux, and macOS, and adjusts generation parameters so the model runs well even on constrained systems.

## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The base model [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) is released by the Qwen team under Apache 2.0.

## What It Does

QORA-0.8B is a 0.8-billion parameter language model with built-in vision. It can:

- **Text generation** — answer questions, write code, summarize text
- **Image understanding** — describe photos, answer questions about images
- **Video understanding** — analyze frame sequences, describe motion and temporal changes
- **Thinking mode** — chain-of-thought reasoning with configurable budget

## Architecture

QORA-0.8B uses a hybrid architecture combining two attention mechanisms:

| Component | Details |
|-----------|---------|
| **Parameters** | 0.8B total |
| **Hidden dim** | 1024 |
| **Layers** | 24 (18 DeltaNet + 6 Full Attention) |
| **Layer pattern** | 3x DeltaNet + 1x Full Attention, repeated 6 times |
| **Vocabulary** | 248,320 tokens |
| **Context** | 262K tokens natively |

### DeltaNet Layers (18 of 24)
- Gated linear attention with delta rule state updates
- 16 QK heads + 16 V heads, head_dim=128
- Causal Conv1d (kernel=4) + SiLU activation
- O(1) memory per token (recurrent state, no KV cache needed)

### Full Attention Layers (6 of 24)
- Grouped Query Attention (8Q / 2KV heads), head_dim=256
- QK-norm + partial RoPE (64/256 dims rotated), theta=10M
- Output gating (sigmoid gate on attention output)
- Standard KV cache

### Vision Encoder
- 12-layer ViT, hidden=768, 12 heads
- Conv3d patch embedding [768, 3, 2, 16, 16] (temporal_patch_size=2)
- Learned positional embedding with bilinear interpolation from 48x48 grid
- 2D spatial RoPE (dim=32, theta=10000)
- 2x2 spatial merger: LayerNorm → concat → MLP(3072 → 1024)
- **Images**: single frame duplicated along temporal axis
- **Video**: actual Conv3d over consecutive frame pairs (N frames → N/2 temporal patches)

## Smart System Awareness

QORA-0.8B detects your system at startup and automatically adjusts generation limits:

```
QORA-0.8B - Pure Rust Multimodal Inference Engine
System: 16384 MB RAM (9856 MB free), 12 threads
```

| Available RAM | Think Budget | Max Tokens | Behavior |
|---------------|-------------|------------|----------|
| < 4 GB | 128 (cap 256) | 256 (cap 512) | Minimal generation, warning displayed |
| 4-8 GB | 256 (cap 1024) | 512 (cap 1024) | Constrained, warning displayed |
| 8-12 GB | 1024 (cap 2048) | 1024 (cap 2048) | Normal operation |
| >= 12 GB | 2048 (cap 8192) | 2048 (cap 8192) | Full capability |

**Hard caps apply even to explicit user values** — if you pass `--max-tokens 5000` on a system with 6 GB free RAM, it gets clamped to 1024 automatically. This prevents the model from running for too long on weak systems.

Supports **Windows** (wmic), **Linux** (/proc/meminfo), and **macOS** (sysctl/vm_stat).

## Weight Format

| Format | Size | Quality | Speed |
|--------|------|---------|-------|
| **Q4** (default) | ~600 MB | Good | ~3.9 tok/s |

Q4 uses 4-bit symmetric quantization with group_size=32 and LUT-optimized dequantization. Multi-threaded GEMV/GEMM via rayon for large matrices.

The model is small enough that Q4 is the only format needed — it loads in under 1 second and uses minimal RAM.

## Quick Start

1. Download `qor08b.exe`, `model.qor08b`, and `tokenizer.json` into the same folder
2. Run:

```bash
# Text generation
qor08b --prompt "Explain quantum computing" --max-tokens 500

# Image understanding
qor08b --prompt "What's in this image?" --image photo.jpg

# Video understanding (directory of frame images)
qor08b --prompt "What happens in this video?" --video frames_dir/

# Thinking mode (default, extended reasoning)
qor08b --prompt "What is the capital of France?" --think-budget 512

# No-think mode (faster, direct answers)
qor08b --prompt "What is 2+2?" --no-think

# Greedy decoding (deterministic output)
qor08b --prompt "Hello" --greedy
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Input prompt (default: "Hello, how are you?") |
| `--image PATH` | Path to an image file (PNG/JPG) |
| `--video PATH` | Path to directory of frame images (PNG/JPG, sorted by name) |
| `--max-tokens N` | Max tokens to generate (default: 1024) |
| `--think-budget N` | Max thinking tokens before forcing answer (default: 1024) |
| `--no-think` | Disable thinking mode (direct answers) |
| `--show-think` | Display thinking tokens on stderr |
| `--greedy` | Greedy decoding (temperature=0, not recommended with thinking mode) |

### Sampling Defaults

| Parameter | Think mode | No-think mode |
|-----------|-----------|---------------|
| temperature | 1.0 | 0.7 |
| top_k | 20 | 20 |
| top_p | 0.95 | 0.95 |
| presence_penalty | 1.5 | 1.5 |

### Video Input

Video is provided as a directory of frame images (not a video file). Extract frames however you like:

```bash
# Example: extract 4 frames from a video with ffmpeg
ffmpeg -i video.mp4 -vf "select=not(mod(n\,30))" -frames:v 4 frames/frame_%02d.png

# Then run
qor08b --prompt "Describe what happens" --video frames/
```

Frames are loaded in alphabetical order, resized to uniform dimensions (max 768px, divisible by 32), and processed as temporal pairs via Conv3d. Odd frame counts are padded by duplicating the last frame.

## Built With

- **Language**: Pure Rust (2024 edition)
- **Dependencies**: `half` (f16), `rayon` (parallelism), `image` (image loading), `tokenizers` (HuggingFace tokenizer), `memmap2` (mmap for converter), `serde_json` (config parsing)
- **No ML framework** for inference — all matrix ops are hand-written Rust
- **Burn framework** used only as a build dependency (for binary format types)

## File Structure

```
src/
  main.rs      — CLI entry point, argument parsing
  config.rs    — Model architecture configuration
  gemv.rs      — GEMV/GEMM kernels (F16 + Q4), hybrid forward pass, prefill
  generate.rs  — Text generation loop (text, image, video modes)
  tokenizer.rs — Tokenizer wrapper and chat templates
  vision.rs    — Vision encoder (ViT + merger), image/video loading
  save.rs      — Binary model format (.qor08b) save/load
  convert.rs   — One-time safetensors → .qor08b converter
  system.rs    — System awareness (RAM detection, smart limits)
  lib.rs       — Module exports
```

## Model Binary Format (.qor08b)

Custom binary format for fast loading:

```
Header:  "QR08" magic + version(u32) + format(u8: 0=F16, 1=Q4)
Config:  Architecture params (vocab, hidden, layers, heads, etc.)
Layers:  24 layers, each with type byte + layer-specific weights
Global:  Embedding + final norm + precomputed RoPE tables
Vision:  Conv3d patch embed + pos_embed + 12 ViT blocks + merger MLP
```

Loading is ~500ms for the Q4 model (~600 MB) via buffered sequential reads.

## Performance

Tested on i5-11500 (6C/12T), 16GB RAM:

| Task | Speed |
|------|-------|
| Text decode | ~3.9 tok/s |
| Text prefill | ~13 tok/s (batched DeltaNet) |
| Model load | ~500ms (Q4, 600 MB) |
| RAM usage | ~791 MB |

CPU-only by design — the 0.8B model is small enough that CPU inference is fast and efficient, making it ideal for mobile and edge deployment without GPU dependencies. Batched DeltaNet prefill processes all GEMM projections in parallel across tokens, with only the lightweight conv1d and recurrent state update running sequentially.

## Comparison with QORA-4B

| | QORA-0.8B | QORA-4B |
|---|-----------|---------|
| Parameters | 0.8B | 4B |
| Model size (Q4) | 600 MB | 2.9 GB |
| Load time | ~500ms | ~30s |
| Decode speed | ~3.9 tok/s | ~3.3 tok/s (GPU), ~1.3 tok/s (CPU) |
| RAM usage | ~791 MB | ~3.5 GB |
| GPU support | No (not needed) | Yes (Vulkan) |
| Vision | 12L ViT (768) | 24L ViT (1024) |
| Best for | Mobile, edge, quick tasks | Desktop, complex reasoning |
