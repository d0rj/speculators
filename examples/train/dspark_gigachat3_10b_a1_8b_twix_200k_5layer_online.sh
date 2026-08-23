#!/bin/bash
# Expand the trained three-layer DSpark to five layers and fine-tune on T-Wix.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
TRAIN_PYTHON="${TRAIN_PYTHON:-$(command -v python)}"

export DATASET="${DATASET:-t-wix}"
export MAX_SAMPLES="${MAX_SAMPLES:-200000}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export SEED="${SEED:-42}"
export DATA_DIR="${DATA_DIR:-$HOME/dflash-output/gigachat3_twix_200k_8k_data}"

# By default, consume the best completed three-layer checkpoint. The expansion
# preserves all trained tensors and adds two exact identity residual layers.
SOURCE_DRAFT="${SOURCE_DRAFT:-$HOME/dflash-output/dspark_gigachat3_twix_200k_8k_3layer_lk/checkpoints/checkpoint_best}"
EXPANDED_DRAFT="${EXPANDED_DRAFT:-$HOME/dflash-output/dspark_gigachat3_twix_200k_8k_5layer_init}"
if [[ ! -f "$SOURCE_DRAFT/model.safetensors" ]]; then
    echo "Three-layer source checkpoint is missing: $SOURCE_DRAFT" >&2
    exit 1
fi
if [[ ! -f "$EXPANDED_DRAFT/model.safetensors" ]]; then
    "$TRAIN_PYTHON" "$REPO_DIR/scripts/expand_dspark_layers.py" \
        --source "$SOURCE_DRAFT" \
        --output "$EXPANDED_DRAFT" \
        --num-layers 5
fi
export FROM_PRETRAINED="$EXPANDED_DRAFT"

export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dspark_gigachat3_twix_200k_8k_5layer_lk_sdpa}"
export LOG_DIR="${LOG_DIR:-$HOME/dflash-output/tensorboard/gigachat3_twix_200k_8k_5layer_lk_sdpa}"
export RUN_NAME="${RUN_NAME:-dspark_gigachat3_twix_200k_8k_5layer_lk_sdpa}"

export NUM_LAYERS=5
export MAX_ANCHORS="${MAX_ANCHORS:-128}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
export MARKOV_RANK="${MARKOV_RANK:-256}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export LOSS_FN="${LOSS_FN:-lk_hybrid}"
export CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-0.25}"

# Fine-tune instead of repeating cold-start training. The old three layers and
# all heads begin from the trained checkpoint; only the two added residual
# layers begin as identity functions.
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-1e-4}"
export SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-0.25}"

# GigaChat3 uses CPU offload on a 16-GiB card. Multiple simultaneous verifier
# requests contend for CPU memory bandwidth and PCIe transfers and are much
# slower than a single prefetched request. One worker overlaps extraction with
# draft training while keeping the host responsive.
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-1}"
export TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-1}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8193}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.53}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-NONE}"

exec bash "$SCRIPT_DIR/dspark_gigachat3_10b_a1_8b_online.sh"
