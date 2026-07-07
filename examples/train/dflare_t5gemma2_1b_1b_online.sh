#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-google/t5gemma-2-1b-1b}"
DATASET="${DATASET:-ultrachat}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dflare_t5gemma2_online_test}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="${T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE:-4}"
# DFlare benefits from a wider set of verifier states because each draft layer
# learns its own fusion over these layers. Keep these explicit for reproducible
# training and vLLM hidden-state extraction.
TARGET_LAYER_IDS=(2 5 8 11 14 17 20 22)

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

PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCHDYNAMO_DISABLE=1 \
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="$T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE" \
python scripts/train_t5gemma_online.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size 32000 \
    --speculator-type dflare \
    --draft-arch llama \
    --draft-hidden-act silu \
    --draft-attn-impl eager \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 6 \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --total-seq-len 2048 \
    --epochs 6 \
    --lr 6e-4 \
    --loss-fn kl_div \
    --on-missing raise \
    --num-workers 0 \
    --prefetch-factor 1
