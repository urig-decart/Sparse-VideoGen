#!/bin/bash
# Run SVG SAP text-to-video on all prompts from configs/prompts.txt
# Matches entropy-sparse-attn conventions: seed=42, per-prompt outputs, 720p
set -e

SEED=42
RESOLUTION="720p"
HEIGHT=720
WIDTH=1280
INFER_STEPS=50
PATTERN="SAP"

# SAP parameters
QC=300
KC=1000
TOP_P=0.9
MIN_KC_RATIO=0.10
KMEANS_INIT=50
KMEANS_STEP=2

# Warmup
FIRST_TIMES_FP=0.2
FIRST_LAYERS_FP=0.03

# Output
OUTPUT_DIR="results/wan-14b-${RESOLUTION}-sap"
mkdir -p "${OUTPUT_DIR}"

PROMPTS_FILE="configs/prompts.txt"
PROMPT_IDX=0

while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue

    PADDED=$(printf "%03d" $PROMPT_IDX)
    OUT_VIDEO="${OUTPUT_DIR}/p${PADDED}_s${SEED}.mp4"
    LOG_FILE="${OUTPUT_DIR}/p${PADDED}_s${SEED}.jsonl"

    echo "=========================================="
    echo "Prompt ${PROMPT_IDX}: ${prompt}"
    echo "Output: ${OUT_VIDEO}"
    echo "=========================================="

    python wan_t2v_inference.py \
        --model_id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
        --prompt "${prompt}" \
        --height $HEIGHT \
        --width $WIDTH \
        --seed $SEED \
        --num_inference_steps $INFER_STEPS \
        --pattern $PATTERN \
        --num_q_centroids $QC \
        --num_k_centroids $KC \
        --top_p_kmeans $TOP_P \
        --min_kc_ratio $MIN_KC_RATIO \
        --kmeans_iter_init $KMEANS_INIT \
        --kmeans_iter_step $KMEANS_STEP \
        --first_times_fp $FIRST_TIMES_FP \
        --first_layers_fp $FIRST_LAYERS_FP \
        --output_file "${OUT_VIDEO}" \
        --logging_file "${LOG_FILE}" \
        --skip_existing

    PROMPT_IDX=$((PROMPT_IDX + 1))
done < "$PROMPTS_FILE"

echo "Done. Results in ${OUTPUT_DIR}/"
