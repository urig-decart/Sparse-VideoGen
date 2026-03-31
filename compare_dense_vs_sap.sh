#!/bin/bash
# Compare dense vs SAP sparse attention: Wan2.1-1.3B, 480p, 81 frames
# Runs each prompt twice (dense + SAP) and records wall-clock time
set -e

export PYTHONPATH="svg/kernels/build:$PYTHONPATH"

SEED=42
MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
STEPS=50
NUM_PROMPTS=5

DENSE_DIR="results/comparison-1.3b-480p/dense"
SAP_DIR="results/comparison-1.3b-480p/sap"
RESULTS_FILE="results/comparison-1.3b-480p/results.txt"

mkdir -p "$DENSE_DIR" "$SAP_DIR"

PROMPTS_FILE="configs/prompts.txt"

echo "===== Dense vs SAP Comparison: Wan2.1-1.3B, 480p, 81f =====" | tee "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
printf "%-6s %-12s %-12s %-10s %s\n" "Prompt" "Dense(s)" "SAP(s)" "Speedup" "Prompt text" | tee -a "$RESULTS_FILE"
printf "%s\n" "----------------------------------------------------------------------" | tee -a "$RESULTS_FILE"

PROMPT_IDX=0
TOTAL_DENSE=0
TOTAL_SAP=0

while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    [ $PROMPT_IDX -ge $NUM_PROMPTS ] && break

    PADDED=$(printf "%03d" $PROMPT_IDX)

    # Dense run
    START=$(date +%s%N)
    python wan_t2v_inference.py \
        --model_id "$MODEL" \
        --prompt "$prompt" \
        --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
        --seed $SEED --num_inference_steps $STEPS \
        --pattern dense \
        --output_file "${DENSE_DIR}/p${PADDED}_s${SEED}.mp4" \
        --skip_existing 2>&1 | grep -E "^\s*\d+%|Warmup|Prompt:" || true
    END=$(date +%s%N)
    DENSE_MS=$(( (END - START) / 1000000 ))
    DENSE_S=$(echo "scale=1; $DENSE_MS / 1000" | bc)

    # SAP run
    START=$(date +%s%N)
    python wan_t2v_inference.py \
        --model_id "$MODEL" \
        --prompt "$prompt" \
        --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
        --seed $SEED --num_inference_steps $STEPS \
        --pattern SAP \
        --num_q_centroids 300 --num_k_centroids 1000 \
        --top_p_kmeans 0.9 --min_kc_ratio 0.10 \
        --kmeans_iter_init 50 --kmeans_iter_step 2 \
        --first_times_fp 0.2 --first_layers_fp 0.03 \
        --output_file "${SAP_DIR}/p${PADDED}_s${SEED}.mp4" \
        --logging_file "${SAP_DIR}/p${PADDED}_s${SEED}.jsonl" \
        --skip_existing 2>&1 | grep -E "^\s*\d+%|Warmup|Prompt:|Centroid" || true
    END=$(date +%s%N)
    SAP_MS=$(( (END - START) / 1000000 ))
    SAP_S=$(echo "scale=1; $SAP_MS / 1000" | bc)

    SPEEDUP=$(echo "scale=2; $DENSE_MS / $SAP_MS" | bc 2>/dev/null || echo "N/A")

    TOTAL_DENSE=$((TOTAL_DENSE + DENSE_MS))
    TOTAL_SAP=$((TOTAL_SAP + SAP_MS))

    SHORT_PROMPT="${prompt:0:50}"
    printf "%-6s %-12s %-12s %-10s %s\n" "p${PADDED}" "${DENSE_S}s" "${SAP_S}s" "${SPEEDUP}x" "${SHORT_PROMPT}" | tee -a "$RESULTS_FILE"

    PROMPT_IDX=$((PROMPT_IDX + 1))
done < "$PROMPTS_FILE"

AVG_DENSE=$(echo "scale=1; $TOTAL_DENSE / $PROMPT_IDX / 1000" | bc)
AVG_SAP=$(echo "scale=1; $TOTAL_SAP / $PROMPT_IDX / 1000" | bc)
AVG_SPEEDUP=$(echo "scale=2; $TOTAL_DENSE / $TOTAL_SAP" | bc 2>/dev/null || echo "N/A")

echo "" | tee -a "$RESULTS_FILE"
printf "%s\n" "----------------------------------------------------------------------" | tee -a "$RESULTS_FILE"
printf "%-6s %-12s %-12s %-10s\n" "AVG" "${AVG_DENSE}s" "${AVG_SAP}s" "${AVG_SPEEDUP}x" | tee -a "$RESULTS_FILE"

echo "" | tee -a "$RESULTS_FILE"
echo "Now computing quality metrics (PSNR/SSIM)..." | tee -a "$RESULTS_FILE"

# Quality comparison
python3 -c "
import torch, numpy as np, os, json
from PIL import Image
import imageio.v3 as iio

dense_dir = '${DENSE_DIR}'
sap_dir = '${SAP_DIR}'
n = ${PROMPT_IDX}

results = []
for i in range(n):
    padded = f'{i:03d}'
    d_path = f'{dense_dir}/p{padded}_s${SEED}.mp4'
    s_path = f'{sap_dir}/p{padded}_s${SEED}.mp4'

    d_frames = iio.imread(d_path, plugin='pyav')  # (T, H, W, C) uint8
    s_frames = iio.imread(s_path, plugin='pyav')

    d = torch.from_numpy(d_frames).float()
    s = torch.from_numpy(s_frames).float()

    # Truncate to min length
    T = min(d.shape[0], s.shape[0])
    d, s = d[:T], s[:T]

    mse = ((d - s) ** 2).mean().item()
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float('inf')

    # SSIM per frame
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    ssims = []
    for t in range(T):
        a, b = d[t], s[t]
        mu_a, mu_b = a.mean(), b.mean()
        sa = ((a-mu_a)**2).mean()
        sb = ((b-mu_b)**2).mean()
        sab = ((a-mu_a)*(b-mu_b)).mean()
        ssim = ((2*mu_a*mu_b+C1)*(2*sab+C2)) / ((mu_a**2+mu_b**2+C1)*(sa+sb+C2))
        ssims.append(ssim.item())
    avg_ssim = np.mean(ssims)

    results.append({'prompt': i, 'psnr': round(psnr, 2), 'ssim': round(avg_ssim, 4)})
    print(f'  p{padded}: PSNR={psnr:.1f}dB  SSIM={avg_ssim:.4f}')

avg_psnr = np.mean([r['psnr'] for r in results])
avg_ssim = np.mean([r['ssim'] for r in results])
print(f'  AVG:  PSNR={avg_psnr:.1f}dB  SSIM={avg_ssim:.4f}')

with open('results/comparison-1.3b-480p/quality.json', 'w') as f:
    json.dump({'results': results, 'avg_psnr': round(avg_psnr,2), 'avg_ssim': round(avg_ssim,4)}, f, indent=2)
" 2>&1 | tee -a "$RESULTS_FILE"

echo "" | tee -a "$RESULTS_FILE"
echo "Done. Full results in results/comparison-1.3b-480p/" | tee -a "$RESULTS_FILE"
