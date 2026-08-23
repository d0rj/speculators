#!/bin/bash
# Production-oriented one-GPU online DSpark training on t-tech/T-Wix.
# The implementation is shared with the validated GigaChat3 smoke pipeline;
# every setting below can still be overridden as an environment variable.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-t-wix}"
export MAX_SAMPLES="${MAX_SAMPLES:-200000}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export EPOCHS="${EPOCHS:-1}"
export SEED="${SEED:-42}"
export PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-2}"
export PREPROCESSING_CANDIDATE_MULTIPLIER="${PREPROCESSING_CANDIDATE_MULTIPLIER:-1.1}"

export DATA_DIR="${DATA_DIR:-$HOME/dflash-output/gigachat3_twix_200k_8k_data}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dspark_gigachat3_twix_200k_8k}"
export LOG_DIR="${LOG_DIR:-$HOME/dflash-output/tensorboard/gigachat3_twix_200k_8k}"
export RUN_NAME="${RUN_NAME:-dspark_gigachat3_twix_200k_8k}"

# Keep the architecture identical to the validated smoke checkpoint so that
# quality and speed changes are attributable to the larger T-Wix training set.
export NUM_LAYERS="${NUM_LAYERS:-1}"
export MAX_ANCHORS="${MAX_ANCHORS:-64}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
export MARKOV_RANK="${MARKOV_RANK:-256}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"

exec bash "$SCRIPT_DIR/dspark_gigachat3_10b_a1_8b_online.sh"
