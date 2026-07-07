#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-google/t5gemma-2-1b-1b}"
DATASET="${DATASET:-ultrachat}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dflash_t5gemma2_1b_1b}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
# Use 5 verifier layers, evenly spaced across the decoder, matching the DFlash
# paper's setup (5 target hidden states between the early and late layers).
TARGET_LAYER_IDS=(2 7 13 18 21)

if [[ ! -f "$OUTPUT_DIR/dataset_info.json" ]]; then
    python scripts/prepare_t5gemma_data.py \
        --model "$MODEL" \
        --data "$DATASET" \
        --output "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --encoder-seq-length 2048 \
        --decoder-seq-length 1024
else
    echo "Prepared dataset already exists at $OUTPUT_DIR; reusing it."
fi

python scripts/extract_t5gemma_hidden_states.py \
    --model "$MODEL" \
    --preprocessed-data "$OUTPUT_DIR" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}"

TORCHDYNAMO_DISABLE=1 python scripts/train_t5gemma.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size 32000 \
    --speculator-type dflash \
    --draft-arch llama \
    --draft-hidden-act silu \
    --draft-attn-impl eager \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 5 \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --total-seq-len 2048 \
    --epochs 6 \
    --lr 6e-4 \
    --loss-fn kl_div \
    --on-missing raise \
    --num-workers 2 \
    --prefetch-factor 2
