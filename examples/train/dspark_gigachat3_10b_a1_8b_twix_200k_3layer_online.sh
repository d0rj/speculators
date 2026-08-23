#!/bin/bash
# Three-layer DSpark training for GigaChat3 on the deterministic T-Wix 200K set.
# This is a fresh run: the previous one-layer checkpoint is intentionally not
# loaded because adding layers changes the draft architecture.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-t-wix}"
export MAX_SAMPLES="${MAX_SAMPLES:-200000}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export SEED="${SEED:-42}"

# Reuse the already prepared deterministic 8K dataset. If it is absent, the
# memory-bounded preparation path rebuilds it with the same revision and seed.
export DATA_DIR="${DATA_DIR:-$HOME/dflash-output/gigachat3_twix_200k_8k_data}"
export PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-2}"
export PREPROCESSING_CANDIDATE_MULTIPLIER="${PREPROCESSING_CANDIDATE_MULTIPLIER:-1.1}"

export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dspark_gigachat3_twix_200k_8k_3layer_lk}"
export LOG_DIR="${LOG_DIR:-$HOME/dflash-output/tensorboard/gigachat3_twix_200k_8k_3layer_lk}"
export RUN_NAME="${RUN_NAME:-dspark_gigachat3_twix_200k_8k_3layer_lk}"

# Higher-capacity draft and denser anchor supervision. Across two epochs this
# samples up to roughly half of the available assistant-token positions.
export NUM_LAYERS="${NUM_LAYERS:-3}"
export MAX_ANCHORS="${MAX_ANCHORS:-128}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
export MARKOV_RANK="${MARKOV_RANK:-256}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"

# LK hybrid starts KL-heavy when overlap is low and gradually shifts toward TV,
# avoiding the weak cold-start gradient of the old 0.1 CE + 0.9 TV objective.
export LOSS_FN="${LOSS_FN:-lk_hybrid}"
export CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-0.25}"
export EPOCHS="${EPOCHS:-2}"
export LR="${LR:-2e-4}"
export SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-0.5}"

# Leave additional headroom for three draft layers and 128 anchor blocks. The
# verifier serves a single 8K sequence, so it does not need a large KV reserve.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"

exec bash "$SCRIPT_DIR/dspark_gigachat3_10b_a1_8b_online.sh"
