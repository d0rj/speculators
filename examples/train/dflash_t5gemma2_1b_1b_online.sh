#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-google/t5gemma-2-1b-1b}"
DATASET="${DATASET:-ultrachat}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dflash_t5gemma2_online_test}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
# Number of samples processed by the frozen verifier in one online extraction
# microbatch. Increase to 8/16 for higher GPU utilization if VRAM allows;
# reduce to 1/2 if full-length samples OOM.
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="${T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE:-4}"
# Use 5 verifier layers, evenly spaced across the decoder, matching the DFlash
# paper's setup (5 target hidden states between the early and late layers).
TARGET_LAYER_IDS=(2 7 13 18 21)

# The Arrow dataset is small: it stores only encoder/decoder token IDs and masks.
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

# The verifier is kept frozen on the same GPU as the drafter. num_workers must
# remain zero: worker processes cannot share this in-process CUDA model.
PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCHDYNAMO_DISABLE=1 \
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="$T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE" \
python scripts/train_t5gemma_online.py \
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
    --num-workers 0 \
    --prefetch-factor 1
