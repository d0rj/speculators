#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-google/t5gemma-2-1b-1b}"
DATASET="${DATASET:-ultrachat}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dspark_t5gemma2_online_test}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="${T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE:-4}"
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

PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCHDYNAMO_DISABLE=1 \
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="$T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE" \
python scripts/train_t5gemma_online.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size 32000 \
    --speculator-type dspark \
    --draft-arch llama \
    --draft-hidden-act silu \
    --draft-attn-impl eager \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 5 \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    --total-seq-len 2048 \
    --epochs 6 \
    --lr 6e-4 \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --on-missing raise \
    --num-workers 0 \
    --prefetch-factor 1
