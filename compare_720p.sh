#!/bin/bash
# Compare dense vs SVG vs SAP: Wan2.1-1.3B, 720p, 81 frames
set -e

export PYTHONPATH="svg/kernels/build:$PYTHONPATH"

SEED=42
MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=720
WIDTH=1280
NUM_FRAMES=81
STEPS=50
NUM_PROMPTS=3
OUTDIR="results/comparison-1.3b-720p"

PROMPTS_FILE="configs/prompts.txt"

echo "===== Dense vs SVG vs SAP — Wan2.1-1.3B, 720p, 81f, B200 =====" | tee "$OUTDIR/results.txt"
echo "" | tee -a "$OUTDIR/results.txt"
printf "%-6s %-12s %-12s %-12s %-10s %-10s\n" "Prompt" "Dense(s)" "SVG(s)" "SAP(s)" "SVG spdup" "SAP spdup" | tee -a "$OUTDIR/results.txt"
printf "%s\n" "------------------------------------------------------------------------" | tee -a "$OUTDIR/results.txt"

PROMPT_IDX=0
T_DENSE=0; T_SVG=0; T_SAP=0

while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    [ $PROMPT_IDX -ge $NUM_PROMPTS ] && break
    PADDED=$(printf "%03d" $PROMPT_IDX)

    # --- Dense ---
    START=$(date +%s%N)
    python wan_t2v_inference.py \
        --model_id "$MODEL" --prompt "$prompt" \
        --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
        --seed $SEED --num_inference_steps $STEPS \
        --pattern dense \
        --output_file "$OUTDIR/dense/p${PADDED}_s${SEED}.mp4" \
        --skip_existing 2>&1 | grep -E "100%" || true
    END=$(date +%s%N)
    D_MS=$(( (END - START) / 1000000 ))
    D_S=$(echo "scale=1; $D_MS / 1000" | bc)

    # --- SVG (striped) ---
    START=$(date +%s%N)
    python wan_t2v_inference.py \
        --model_id "$MODEL" --prompt "$prompt" \
        --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
        --seed $SEED --num_inference_steps $STEPS \
        --pattern SVG \
        --sparsity 0.3 \
        --num_sampled_rows 64 \
        --first_times_fp 0.2 --first_layers_fp 0.03 \
        --output_file "$OUTDIR/svg/p${PADDED}_s${SEED}.mp4" \
        --skip_existing 2>&1 | grep -E "100%" || true
    END=$(date +%s%N)
    SVG_MS=$(( (END - START) / 1000000 ))
    SVG_S=$(echo "scale=1; $SVG_MS / 1000" | bc)

    # --- SAP (k-means) ---
    START=$(date +%s%N)
    python wan_t2v_inference.py \
        --model_id "$MODEL" --prompt "$prompt" \
        --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
        --seed $SEED --num_inference_steps $STEPS \
        --pattern SAP \
        --num_q_centroids 300 --num_k_centroids 1000 \
        --top_p_kmeans 0.9 --min_kc_ratio 0.10 \
        --kmeans_iter_init 50 --kmeans_iter_step 2 \
        --first_times_fp 0.2 --first_layers_fp 0.03 \
        --output_file "$OUTDIR/sap/p${PADDED}_s${SEED}.mp4" \
        --logging_file "$OUTDIR/sap/p${PADDED}_s${SEED}.jsonl" \
        --skip_existing 2>&1 | grep -E "100%" || true
    END=$(date +%s%N)
    SAP_MS=$(( (END - START) / 1000000 ))
    SAP_S=$(echo "scale=1; $SAP_MS / 1000" | bc)

    SVG_SPD=$(echo "scale=2; $D_MS / $SVG_MS" | bc 2>/dev/null || echo "N/A")
    SAP_SPD=$(echo "scale=2; $D_MS / $SAP_MS" | bc 2>/dev/null || echo "N/A")

    T_DENSE=$((T_DENSE + D_MS))
    T_SVG=$((T_SVG + SVG_MS))
    T_SAP=$((T_SAP + SAP_MS))

    printf "%-6s %-12s %-12s %-12s %-10s %-10s\n" \
        "p${PADDED}" "${D_S}s" "${SVG_S}s" "${SAP_S}s" "${SVG_SPD}x" "${SAP_SPD}x" | tee -a "$OUTDIR/results.txt"

    PROMPT_IDX=$((PROMPT_IDX + 1))
done < "$PROMPTS_FILE"

A_D=$(echo "scale=1; $T_DENSE / $PROMPT_IDX / 1000" | bc)
A_SVG=$(echo "scale=1; $T_SVG / $PROMPT_IDX / 1000" | bc)
A_SAP=$(echo "scale=1; $T_SAP / $PROMPT_IDX / 1000" | bc)
A_SVG_SPD=$(echo "scale=2; $T_DENSE / $T_SVG" | bc 2>/dev/null || echo "N/A")
A_SAP_SPD=$(echo "scale=2; $T_DENSE / $T_SAP" | bc 2>/dev/null || echo "N/A")

printf "%s\n" "------------------------------------------------------------------------" | tee -a "$OUTDIR/results.txt"
printf "%-6s %-12s %-12s %-12s %-10s %-10s\n" \
    "AVG" "${A_D}s" "${A_SVG}s" "${A_SAP}s" "${A_SVG_SPD}x" "${A_SAP_SPD}x" | tee -a "$OUTDIR/results.txt"

echo "" | tee -a "$OUTDIR/results.txt"
echo "Computing quality metrics..." | tee -a "$OUTDIR/results.txt"

python3 -c "
import torch, numpy as np, json
import imageio.v3 as iio

outdir = '$OUTDIR'
n, seed = $PROMPT_IDX, 42

def quality(ref_path, test_path):
    r = torch.from_numpy(iio.imread(ref_path, plugin='pyav')).float()
    t = torch.from_numpy(iio.imread(test_path, plugin='pyav')).float()
    T = min(r.shape[0], t.shape[0])
    r, t = r[:T], t[:T]
    mse = ((r - t)**2).mean().item()
    psnr = 10*np.log10(255.**2/mse) if mse > 0 else float('inf')
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    ssims = []
    for f in range(T):
        a, b = r[f], t[f]
        ma, mb = a.mean(), b.mean()
        sa, sb = ((a-ma)**2).mean(), ((b-mb)**2).mean()
        sab = ((a-ma)*(b-mb)).mean()
        ssims.append(((2*ma*mb+C1)*(2*sab+C2))/((ma**2+mb**2+C1)*(sa+sb+C2)).item())
    return psnr, np.mean(ssims)

print(f\"{'Prompt':<8} {'SVG PSNR':>10} {'SVG SSIM':>10} {'SAP PSNR':>10} {'SAP SSIM':>10}\")
print('-'*52)
all_results = []
for i in range(n):
    p = f'{i:03d}'
    dp = f'{outdir}/dense/p{p}_s{seed}.mp4'
    svg_psnr, svg_ssim = quality(dp, f'{outdir}/svg/p{p}_s{seed}.mp4')
    sap_psnr, sap_ssim = quality(dp, f'{outdir}/sap/p{p}_s{seed}.mp4')
    print(f'p{p:<5} {svg_psnr:>10.1f} {svg_ssim:>10.4f} {sap_psnr:>10.1f} {sap_ssim:>10.4f}')
    all_results.append({'prompt':i, 'svg_psnr':round(svg_psnr,2), 'svg_ssim':round(svg_ssim,4),
                        'sap_psnr':round(sap_psnr,2), 'sap_ssim':round(sap_ssim,4)})

print('-'*52)
print(f\"{'AVG':<8} {np.mean([r['svg_psnr'] for r in all_results]):>10.1f} {np.mean([r['svg_ssim'] for r in all_results]):>10.4f} {np.mean([r['sap_psnr'] for r in all_results]):>10.1f} {np.mean([r['sap_ssim'] for r in all_results]):>10.4f}\")

with open(f'{outdir}/quality.json','w') as f:
    json.dump(all_results, f, indent=2)
" 2>&1 | tee -a "$OUTDIR/results.txt"

echo "" | tee -a "$OUTDIR/results.txt"
echo "Done. Results in $OUTDIR/" | tee -a "$OUTDIR/results.txt"
