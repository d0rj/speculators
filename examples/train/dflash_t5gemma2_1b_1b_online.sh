#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-google/t5gemma-2-1b-1b}"
MIXTURE="${MIXTURE:-t5gemma2_200k}"
DATASET="${DATASET:-}"
DATA_DIR="${DATA_DIR:-$HOME/dflash-output/t5gemma2_mixture_200k}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dflash_t5gemma2_mixture_200k}"
MAX_SAMPLES="${MAX_SAMPLES:-200000}"
SEED="${SEED:-42}"
# Number of samples processed by the frozen verifier in one online extraction
# microbatch. Increase to 8/16 for higher GPU utilization if VRAM allows;
# reduce to 1/2 if full-length samples OOM.
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="${T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE:-32}"
# Use 5 verifier layers, evenly spaced across the decoder, matching the DFlash
# paper's setup (5 target hidden states between the early and late layers).
TARGET_LAYER_IDS=(2 7 13 18 21)

# The Arrow dataset is shared by DFlash/DFlare/DSpark runs. It stores only
# encoder/decoder token IDs and masks; hidden states are extracted online.
if [[ ! -f "$DATA_DIR/dataset_info.json" ]]; then
    PREPARE_ARGS=(
        scripts/prepare_t5gemma_data.py
        --model "$MODEL"
        --output "$DATA_DIR"
        --max-samples "$MAX_SAMPLES"
        --seed "$SEED"
        --encoder-seq-length 2048
        --decoder-seq-length 1024
    )
    if [[ -n "$MIXTURE" ]]; then
        PREPARE_ARGS+=(--mixture "$MIXTURE")
    else
        PREPARE_ARGS+=(--data "$DATASET")
    fi
    python "${PREPARE_ARGS[@]}"
else
    echo "Prepared dataset already exists at $DATA_DIR; reusing it."
fi

# The verifier is kept frozen on the same GPU as the drafter. num_workers must
# remain zero: worker processes cannot share this in-process CUDA model.
PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCHDYNAMO_DISABLE=1 \
T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE="$T5GEMMA_ONLINE_EXTRACTOR_BATCH_SIZE" \
python scripts/train_t5gemma_online.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
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
