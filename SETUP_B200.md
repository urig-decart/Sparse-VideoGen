# Sparse-VideoGen: B200 Setup Guide

Reproducible setup for running SVG on NVIDIA B200 (compute capability 10.0).

## Environment

- **GPU**: NVIDIA B200 (sm_100)
- **CUDA Toolkit**: 13.0 (system-installed at `/usr/local/cuda`)
- **Driver**: 580.105.08
- **OS**: Linux (Ubuntu 22.04 base)
- **GCC**: 11.4.0

## Step-by-step setup

### 1. Clone repo (skip LFS demo files)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/svg-project/Sparse-VideoGen.git
cd Sparse-VideoGen
```

### 2. Create conda environment

```bash
conda create -n SVG python==3.11.15 -y
conda activate SVG
```

### 3. Install base dependencies

```bash
pip install uv
uv pip install -e .
```

This installs torch 2.11.0+cu130 which has native B200 (sm_100) support.

### 4. Install flash-attn

```bash
pip install flash-attn --no-build-isolation
```

> **NOTE**: This compiles CUDA kernels for sm_80/90/100/120. Takes 30-60+ minutes.

### 5. Install cmake and init submodules

```bash
conda install -c conda-forge cmake -y
pip install wheel-stub
git submodule update --init --recursive
```

### 6. B200 modification: Update CUDA architecture

Edit `svg/kernels/CMakeLists.txt`, line 8:

```diff
-set(CMAKE_CUDA_ARCHITECTURES 90a)
+set(CMAKE_CUDA_ARCHITECTURES 100a)
```

This changes the target from H100 (sm_90a) to B200 (sm_100a).

### 7. Apply FlashInfer patch (block sparse attention)

```bash
cd svg/kernels/3rdparty/flashinfer
cp ../../../../assets/patches/modifications.patch ./
git apply modifications.patch
pip install --no-build-isolation --editable .
cd ../../../..
```

### 8. Build custom kernels

```bash
cd svg/kernels
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
mkdir -p build && cd build
cmake \
  -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  ..
make -j$(nproc)
cd ../../..
```

> **NOTE**: You must pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` explicitly
> if nvcc is not on your PATH. The `-DUSE_SYSTEM_NVTX` flag is not needed.

### 9. Install cuVS

```bash
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com
```

Then restore torch's CUDA bindings (cuVS downgrades them):

```bash
pip install 'cuda-bindings>=13.0.3,<14' 'cuda-toolkit==13.0.2'
```

> pip will warn about dependency conflicts between cu12 (cuVS) and cu13 (torch).
> This is a metadata-only conflict; both work at runtime with CUDA 13.0.

### 10. Remove deprecated pynvml (optional, silences warning)

```bash
pip uninstall pynvml -y
```

### 11. Upgrade diffusers for transformers 5.x compatibility

```bash
pip install diffusers==0.37.0
```

> SVG pins `diffusers==0.34.0` but that version can't import with `transformers>=5.0`
> (`FLAX_WEIGHTS_NAME` was removed). 0.37.0 works but changes the rotary embedding
> format, which requires the code patches below.

### 12. Patch SVG code for diffusers 0.37+ and B200

Three files need patching:

**a) `svg/models/wan/custom_models.py`** — RoPE format change

diffusers 0.37+ returns `(cos, sin)` tuple instead of complex tensor from `self.rope()`.
The cos/sin tensors are shaped `(1, seq_len, 1, head_dim)` with `repeat_interleave_real=True`
(each value doubled: `[c0,c0,c1,c1,...]`). The custom CUDA kernel expects `(seq_len, half_head_dim)`.

In the `WanTransformer3DModel_Sparse.forward` method (~line 147), replace:

```python
# OLD:
rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
rotary_emb = (rot_real, rot_imag)
```

with:

```python
# NEW:
if isinstance(rotary_emb, tuple):
    rot_real = rotary_emb[0].squeeze(0).squeeze(1)[:, ::2].contiguous().to(torch.float32)
    rot_imag = rotary_emb[1].squeeze(0).squeeze(1)[:, ::2].contiguous().to(torch.float32)
else:
    rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
    rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
rotary_emb = (rot_real, rot_imag)
```

**b) `svg/kernels/triton/layernorm.py`** — Row masking to prevent illegal memory access

Same bug as the RMSNorm fix in commit `28a1a9b`. Both `_layer_norm_param_fwd_fused` and
`_layer_norm_noparam_fwd_fused` kernels need:

1. Add `M: tl.constexpr` parameter (number of rows)
2. Add `row_mask = rows < M` guard
3. Use combined `mask = row_mask[:, None] & col_mask[None, :]` for loads/stores
4. Use `row_mask` for Mean/Rstd stores
5. Pass `M` from the Python caller

## Changes from upstream

| File | Change | Why |
|------|--------|-----|
| `svg/kernels/CMakeLists.txt` | `CMAKE_CUDA_ARCHITECTURES` 90a → 100a | Target B200 sm_100 instead of H100 sm_90 |
| `svg/models/wan/custom_models.py` | Handle tuple rotary_emb + de-interleave | diffusers 0.37+ changed RoPE return format |
| `svg/kernels/triton/layernorm.py` | Add row masking to both LayerNorm kernels | Prevent CUDA illegal memory access on B200 |

The FlashInfer patch (`assets/patches/modifications.patch`) is already part of the repo — it just needs to be applied to the submodule.

## Final package versions

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| FlashInfer | 0.2.10 (patched, editable) |
| flash-attn | 2.8.3 (compiled from source) |
| diffusers | 0.37.0 |
| transformers | 5.3.0 |
| triton | 3.6.0 |
| cuVS | 26.2.0 |
| numpy | 2.4.3 |

## Verification

```bash
cd /path/to/Sparse-VideoGen
python -c "
import sys; sys.path.insert(0, 'svg/kernels/build')
import torch; print(f'torch {torch.__version__} GPU={torch.cuda.get_device_name(0)}')
import _kernels; print('Custom kernels: OK')
import flashinfer; print(f'FlashInfer {flashinfer.__version__}: OK')
import cuvs; print('cuVS: OK')
import flash_attn; print('flash-attn: OK')
print('All good!')
"
```

## Running inference

```bash
# Set PYTHONPATH for custom kernels
export PYTHONPATH="svg/kernels/build:$PYTHONPATH"

# Single prompt (1.3B model, fast)
python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A serene mountain lake at sunset" \
    --height 720 --width 1280 --seed 42 \
    --num_inference_steps 50 --pattern SAP \
    --num_q_centroids 300 --num_k_centroids 1000 \
    --top_p_kmeans 0.9 --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 --kmeans_iter_step 2 \
    --first_times_fp 0.2 --first_layers_fp 0.03 \
    --output_file "results/test.mp4"

# All 20 prompts (1.3B, SAP, seed=42, matches entropy-sparse-attn format)
bash scripts/wan/run_t2v_720p_sap_1.3b_all.sh

# All 20 prompts (14B, SAP)
bash scripts/wan/run_t2v_720p_sap_all.sh

# All 20 prompts (14B, dense baseline)
bash scripts/wan/run_t2v_720p_dense_all.sh
```
