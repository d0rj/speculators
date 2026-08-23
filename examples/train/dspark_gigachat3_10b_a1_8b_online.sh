#!/bin/bash
# One-GPU online DSpark smoke training for ai-sage/GigaChat3-10B-A1.8B.
# All settings can be overridden as environment variables.

set -euo pipefail

MODEL="${MODEL:-ai-sage/GigaChat3-10B-A1.8B}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$(command -v python)}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/pc/.venvs/vllm_t5_wheel/bin/python}"
DATASET="${DATASET:-ultrachat}"
DATA_DIR="${DATA_DIR:-$HOME/dflash-output/gigachat3_dspark_smoke_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dflash-output/dspark_gigachat3_smoke}"
LOG_DIR="${LOG_DIR:-$HOME/dflash-output/tensorboard/gigachat3_dspark}"
RUN_NAME="${RUN_NAME:-dspark_gigachat3_smoke}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-$OUTPUT_DIR/online_hidden_states}"

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((SEQ_LENGTH + 1))}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-3e-4}"
LOSS_FN="${LOSS_FN:-}"
if [[ -z "$LOSS_FN" ]]; then
    LOSS_FN='{"ce": 0.1, "tv": 0.9}'
fi
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1.0}"
FROM_PRETRAINED="${FROM_PRETRAINED:-}"
SEED="${SEED:-42}"
NUM_LAYERS="${NUM_LAYERS:-1}"
MAX_ANCHORS="${MAX_ANCHORS:-64}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
MARKOV_RANK="${MARKOV_RANK:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
PREPROCESSING_CANDIDATE_MULTIPLIER="${PREPROCESSING_CANDIDATE_MULTIPLIER:-3.0}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-1}"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-sdpa}"
TARGET_LAYER_IDS=(2 7 13 19 24)

VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-$VLLM_GPU}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_PLUGINS="${VLLM_PLUGINS:-gigachat3_vllm_plugin}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.62}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$VLLM_MAX_MODEL_LEN}"
VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-triton}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-NONE}"
VLLM_OFFLOAD_BACKEND="${VLLM_OFFLOAD_BACKEND:-auto}"
OFFLOAD_GROUP_SIZE="${OFFLOAD_GROUP_SIZE:-2}"
OFFLOAD_NUM_IN_GROUP="${OFFLOAD_NUM_IN_GROUP:-1}"
OFFLOAD_PREFETCH_STEP="${OFFLOAD_PREFETCH_STEP:-1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "VLLM_PYTHON is not executable: $VLLM_PYTHON" >&2
    exit 1
fi
VLLM_BIN_DIR="$(dirname "$VLLM_PYTHON")"
if [[ ! -x "$VLLM_BIN_DIR/ninja" ]]; then
    echo "The vLLM environment is missing the ninja executable." >&2
    echo "Install it with:" >&2
    echo "  uv pip install --python $VLLM_PYTHON ninja" >&2
    exit 1
fi
if ! "$VLLM_PYTHON" -c "import gigachat3_vllm_plugin, vllm" 2>/dev/null; then
    echo "The vLLM environment is missing gigachat3-vllm-plugin." >&2
    echo "Install it with:" >&2
    echo "  uv pip install --python $VLLM_PYTHON -e ../gigachat3-vllm-plugin --no-deps" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$HIDDEN_STATES_PATH"

if [[ ! -f "$DATA_DIR/dataset_info.json" ]]; then
    echo "=== Preparing $MAX_SAMPLES training samples ==="
    "$TRAIN_PYTHON" scripts/prepare_data.py \
        --model "$MODEL" \
        --trust-remote-code \
        --data "$DATASET" \
        --output "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --seq-length "$SEQ_LENGTH" \
        --seed "$SEED" \
        --num-preprocessing-workers "$PREPROCESSING_WORKERS" \
        --preprocessing-candidate-multiplier "$PREPROCESSING_CANDIDATE_MULTIPLIER" \
        --overwrite \
        --minimum-valid-tokens 16
else
    echo "Prepared dataset already exists at $DATA_DIR; reusing it."
fi

echo "=== Launching the hidden-state verifier on GPU $VLLM_GPU ==="
VLLM_EXECUTION_ARGS=()
case "$VLLM_ENFORCE_EAGER" in
    1|true|TRUE|yes|YES)
        VLLM_EXECUTION_ARGS+=(--enforce-eager)
        ;;
    0|false|FALSE|no|NO)
        # Layerwise prefetch offload copies ordinary CPU tensors to CUDA. Such
        # copies can be compiled by Inductor but cannot be captured in a CUDA
        # Graph unless all offloaded weight storage is pinned. Keep compilation
        # enabled while disabling only graph capture by default.
        VLLM_EXECUTION_ARGS+=(
            --compilation-config
            "{\"cudagraph_mode\":\"$VLLM_CUDAGRAPH_MODE\"}"
        )
        ;;
    *)
        echo "VLLM_ENFORCE_EAGER must be 0/1 or false/true, got: $VLLM_ENFORCE_EAGER" >&2
        exit 1
        ;;
esac

PATH="$VLLM_BIN_DIR:$PATH" \
VLLM_PLUGINS="$VLLM_PLUGINS" \
CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
"$VLLM_PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --include-last-layer \
    -- \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --dtype auto \
    --linear-backend "$VLLM_LINEAR_BACKEND" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --offload-backend "$VLLM_OFFLOAD_BACKEND" \
    --offload-group-size "$OFFLOAD_GROUP_SIZE" \
    --offload-num-in-group "$OFFLOAD_NUM_IN_GROUP" \
    --offload-prefetch-step "$OFFLOAD_PREFETCH_STEP" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    "${VLLM_EXECUTION_ARGS[@]}" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for vLLM at http://127.0.0.1:${VLLM_PORT}/health ..."
deadline=$((SECONDS + SERVER_START_TIMEOUT))
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        wait "$VLLM_PID" || true
        echo "vLLM exited before becoming healthy." >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM after ${SERVER_START_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 2
done

DRAFT_INIT_ARGS=(
    --draft-arch qwen3
    --draft-hidden-act silu
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
    --num-layers "$NUM_LAYERS"
)
if [[ -n "$FROM_PRETRAINED" ]]; then
    DRAFT_INIT_ARGS=(
        --from-pretrained "$FROM_PRETRAINED"
        --draft-attn-impl "$DRAFT_ATTN_IMPL"
    )
fi

echo "=== Training DSpark on GPU $TRAIN_GPU ==="
PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCHDYNAMO_DISABLE=1 \
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
"$TRAIN_PYTHON" scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --trust-remote-code \
    --data-path "$DATA_DIR" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type dspark \
    "${DRAFT_INIT_ARGS[@]}" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler-type "$SCHEDULER_TYPE" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --seed "$SEED" \
    --logger tensorboard \
    --log-dir "$LOG_DIR" \
    --run-name "$RUN_NAME" \
    --on-missing generate \
    --on-generate delete \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --prefetch-factor "$TRAIN_PREFETCH_FACTOR"

echo "Training finished. Checkpoints: $OUTPUT_DIR/checkpoints"
