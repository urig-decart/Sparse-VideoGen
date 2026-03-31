# SVG Benchmark: Dense vs Sparse Attention Comparison

Compare dense, SVG (striped), and SAP (k-means) attention strategies on Wan2.1 text-to-video generation. Measures wall-clock time and visual quality (PSNR/SSIM against dense baseline).

## Prerequisites

- NVIDIA GPU (tested on B200, works on H100/A100)
- Working SVG installation (see [SETUP_B200.md](SETUP_B200.md) for B200-specific setup)
- Python environment with all SVG dependencies installed

```bash
# Verify your setup
export PYTHONPATH="svg/kernels/build:$PYTHONPATH"
python -c "
import sys; sys.path.insert(0, 'svg/kernels/build')
import torch, _kernels, flashinfer, cuvs, flash_attn
print(f'torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}')
print('All imports OK')
"
```

## Quick start

### Single prompt test

```bash
export PYTHONPATH="svg/kernels/build:$PYTHONPATH"

# Dense baseline
python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A serene mountain lake at sunset with reflections on the water" \
    --height 480 --width 832 --num_frames 81 --seed 42 \
    --num_inference_steps 50 --pattern dense \
    --output_file results/test_dense.mp4

# SVG (striped sparse attention)
python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A serene mountain lake at sunset with reflections on the water" \
    --height 480 --width 832 --num_frames 81 --seed 42 \
    --num_inference_steps 50 --pattern SVG \
    --sparsity 0.3 --num_sampled_rows 64 \
    --first_times_fp 0.2 --first_layers_fp 0.03 \
    --output_file results/test_svg.mp4

# SAP (k-means sparse attention)
python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A serene mountain lake at sunset with reflections on the water" \
    --height 480 --width 832 --num_frames 81 --seed 42 \
    --num_inference_steps 50 --pattern SAP \
    --num_q_centroids 300 --num_k_centroids 1000 \
    --top_p_kmeans 0.9 --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 --kmeans_iter_step 2 \
    --first_times_fp 0.2 --first_layers_fp 0.03 \
    --output_file results/test_sap.mp4
```

### Full benchmark (3 prompts, 720p, all patterns)

```bash
bash compare_720p.sh
```

Results are written to `results/comparison-1.3b-720p/results.txt`.

## Scripts

| Script | What it does |
|--------|-------------|
| `compare_720p.sh` | Dense vs SVG vs SAP, 1.3B model, 720p, 3 prompts |
| `compare_dense_vs_sap.sh` | Dense vs SAP, 1.3B model, 480p, 5 prompts |
| `scripts/wan/run_t2v_720p_sap_1.3b_all.sh` | SAP on all 20 prompts, 1.3B, 720p |
| `scripts/wan/run_t2v_720p_sap_all.sh` | SAP on all 20 prompts, 14B, 720p |
| `scripts/wan/run_t2v_720p_dense_all.sh` | Dense on all 20 prompts, 14B, 720p |

## Prompts

All scripts use `configs/prompts.txt` (20 prompts from entropy-sparse-attn):

```
A serene mountain lake at sunset with reflections on the water
A cat playing with a ball of yarn on a wooden floor
A breakdancer spinning on their head in a studio
...
```

## Parameters

### Attention patterns

| Pattern | Flag | Description |
|---------|------|-------------|
| `dense` | `--pattern dense` | Standard full attention (baseline) |
| `SVG` | `--pattern SVG` | Striped sparse attention. Lightweight, lower overhead. |
| `SAP` | `--pattern SAP` | K-means block sparse attention. Higher quality, more overhead from clustering. |

### SVG parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sparsity` | 0.3 | Fraction of attention kept (0.0-1.0) |
| `--num_sampled_rows` | 64 | Rows sampled for mask estimation |
| `--first_times_fp` | 0.2 | Fraction of initial timesteps using dense attention |
| `--first_layers_fp` | 0.03 | Fraction of first layers using dense attention |

### SAP parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_q_centroids` | 300 | Query centroids for k-means |
| `--num_k_centroids` | 1000 | Key centroids for k-means |
| `--top_p_kmeans` | 0.9 | Top-p threshold for block selection |
| `--min_kc_ratio` | 0.10 | Minimum fraction of key blocks to keep |
| `--kmeans_iter_init` | 50 | K-means iterations for first sparse step |
| `--kmeans_iter_step` | 2 | K-means iterations for subsequent steps |
| `--first_times_fp` | 0.2 | Fraction of initial timesteps using dense attention |
| `--first_layers_fp` | 0.03 | Fraction of first layers using dense attention |

### Resolution presets

| Resolution | Height | Width | Tokens | flow_shift |
|------------|--------|-------|--------|------------|
| 480p | 480 | 832 | ~24K | 3.0 |
| 720p | 720 | 1280 | ~74K | 5.0 |

### Models

| Model | HuggingFace ID | Heads | Notes |
|-------|---------------|-------|-------|
| Wan2.1-1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 12 | Fast, good for testing |
| Wan2.1-14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 40 | Better quality, sparse shows more speedup |

## Output format

### Directory structure

```
results/comparison-1.3b-720p/
  dense/p000_s42.mp4      # Dense baseline video
  svg/p000_s42.mp4        # SVG sparse video
  sap/p000_s42.mp4        # SAP sparse video
  sap/p000_s42.jsonl      # SAP timing logs per layer
  results.txt             # Timing comparison table
  quality.json            # PSNR/SSIM metrics
```

### Naming convention

Videos: `p{NNN}_s{SEED}.mp4` where NNN is the zero-padded prompt index.

Matches the entropy-sparse-attn naming convention for cross-project comparison.

## Quality metrics

PSNR and SSIM are computed frame-by-frame against the dense baseline:

- **PSNR**: Peak signal-to-noise ratio (dB). Higher = more similar. >30 dB is very close.
- **SSIM**: Structural similarity. Higher = more similar. >0.95 is visually near-identical.

Requires `imageio[pyav]`:

```bash
pip install 'imageio[pyav]'
```

## Reference results (B200, Wan2.1-1.3B, 720p, 81 frames)

```
Prompt  Dense      SVG         SAP         SVG spdup   SAP spdup
------------------------------------------------------------------------
p000    241.4s     291.3s      263.4s      0.82x       0.91x
p001    175.5s     130.1s      153.1s      1.34x       1.14x
p002    144.4s     127.4s      148.8s      1.13x       0.97x
------------------------------------------------------------------------
AVG     187.1s     183.0s      188.5s      1.02x       0.99x

Prompt  SVG PSNR   SVG SSIM    SAP PSNR    SAP SSIM
----------------------------------------------------
p000    28.1 dB    0.9873      30.5 dB     0.9928
p001    26.1 dB    0.9793      28.4 dB     0.9878
p002    24.1 dB    0.9294      26.9 dB     0.9626
----------------------------------------------------
AVG     26.1 dB    0.9653      28.6 dB     0.9811
```

**Notes**:
- p000 is slower due to Triton JIT compilation warmup on first run
- SVG (striped) has less overhead than SAP (k-means), slightly faster
- SAP produces higher quality output (PSNR 28.6 vs 26.1, SSIM 0.98 vs 0.97)
- Sparse methods benefit more from the 14B model (40 heads) where attention is a larger fraction of compute
- All runs use seed=42 for reproducibility

## B200 code changes from upstream

The upstream SVG repo targets H100 (sm_90). Three code changes were needed to run on B200,
plus a diffusers upgrade. Full setup details in [SETUP_B200.md](SETUP_B200.md).

### 1. CUDA architecture target

**File**: `svg/kernels/CMakeLists.txt` (line 8)

```diff
-set(CMAKE_CUDA_ARCHITECTURES 90a)
+set(CMAKE_CUDA_ARCHITECTURES 100a)
```

H100 sm_90 → B200 sm_100.

### 2. Diffusers upgrade: 0.34.0 → 0.37.0

```bash
pip install diffusers==0.37.0
```

Required because `transformers>=5.0` removed `FLAX_WEIGHTS_NAME` which diffusers 0.34.0 imports.
This changes the rotary embedding format, requiring patch #3 below.

### 3. RoPE format compatibility

**File**: `svg/models/wan/custom_models.py` (~line 147)

diffusers 0.37+ changed `self.rope()` from returning a complex tensor to a `(cos, sin)` tuple
shaped `(1, seq_len, 1, head_dim)` with interleaved values `[c0,c0,c1,c1,...]`.
The custom CUDA kernel expects `(seq_len, half_head_dim)`.

```diff
 if ENABLE_FAST_KERNEL:
-    rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
-    rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
+    if isinstance(rotary_emb, tuple):
+        rot_real = rotary_emb[0].squeeze(0).squeeze(1)[:, ::2].contiguous().to(torch.float32)
+        rot_imag = rotary_emb[1].squeeze(0).squeeze(1)[:, ::2].contiguous().to(torch.float32)
+    else:
+        rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
+        rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
     rotary_emb = (rot_real, rot_imag)
```

Backward-compatible with older diffusers via the `isinstance` check.

### 4. LayerNorm row masking

**File**: `svg/kernels/triton/layernorm.py`

Same bug that was fixed for RMSNorm in commit `28a1a9b`. When `BLOCK_M > 1` (N ≤ 512),
the last thread block reads past the end of the tensor → CUDA illegal memory access.

Both `_layer_norm_param_fwd_fused` and `_layer_norm_noparam_fwd_fused` need:

```diff
+    M: tl.constexpr,  # number of rows in X
     N: tl.constexpr,
     ...
     rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
     cols = tl.arange(0, N2)
-    mask = cols < N
+    row_mask = rows < M
+    col_mask = cols < N
+    mask = row_mask[:, None] & col_mask[None, :]
     ...
-    tl.store(Mean + rows, _mean)
-    tl.store(Rstd + rows, _rstd)
+    tl.store(Mean + rows, _mean, mask=row_mask)
+    tl.store(Rstd + rows, _rstd, mask=row_mask)
```

And pass `M` from the Python callers (`triton_layernorm_param_forward`, `triton_layernorm_noparam_forward`).

### Summary

| File | Change | Why |
|------|--------|-----|
| `svg/kernels/CMakeLists.txt` | `90a` → `100a` | Target B200 sm_100 |
| `svg/models/wan/custom_models.py` | Handle tuple RoPE + de-interleave `[:, ::2]` | diffusers 0.37+ changed format |
| `svg/kernels/triton/layernorm.py` | Add `row_mask = rows < M` to both kernels | Prevent illegal memory access |
| pip: `diffusers` | `0.34.0` → `0.37.0` | transformers 5.x compatibility |

The FlashInfer patch (`assets/patches/modifications.patch`) is already in the repo — just needs to be applied to the submodule.

## Adapting for your repository

1. Copy these files to your repo:
   - `configs/prompts.txt`
   - `compare_720p.sh`
   - `compare_dense_vs_sap.sh`

2. Set environment:
   ```bash
   export PYTHONPATH="svg/kernels/build:$PYTHONPATH"
   ```

3. Edit the scripts to change:
   - `MODEL` — switch between 1.3B and 14B
   - `NUM_PROMPTS` — how many prompts to run
   - `HEIGHT`/`WIDTH` — resolution
   - SAP/SVG parameters

4. Run:
   ```bash
   bash compare_720p.sh
   ```
