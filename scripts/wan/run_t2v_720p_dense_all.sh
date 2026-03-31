#!/bin/bash
# Run dense (baseline) text-to-video on all prompts from configs/prompts.txt
# Matches entropy-sparse-attn conventions: seed=42, per-prompt outputs, 720p
set -e

SEED=42
HEIGHT=720
WIDTH=1280
INFER_STEPS=50
PATTERN="dense"

OUTPUT_DIR="results/wan-14b-720p-dense"
mkdir -p "${OUTPUT_DIR}"

PROMPTS_FILE="configs/prompts.txt"
PROMPT_IDX=0

while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue

    PADDED=$(printf "%03d" $PROMPT_IDX)
    OUT_VIDEO="${OUTPUT_DIR}/p${PADDED}_s${SEED}.mp4"

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
        --output_file "${OUT_VIDEO}" \
        --skip_existing

    PROMPT_IDX=$((PROMPT_IDX + 1))
done < "$PROMPTS_FILE"

echo "Done. Results in ${OUTPUT_DIR}/"
