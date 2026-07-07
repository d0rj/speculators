#!/usr/bin/env bash
# Sequential one-GPU baseline versus DFlash benchmark for T5Gemma 2.

set -euo pipefail

export VLLM_PLUGINS="${VLLM_PLUGINS:-t5gemma2_vllm_plugin}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="$ROOT/scripts/evaluate"
VERIFIER="${VERIFIER:-google/t5gemma-2-1b-1b}"
DRAFT="${DRAFT:-$ROOT/dflash-output/t5gemma-5k/checkpoints/checkpoint_best}"
PORT="${PORT:-8000}"
SUBSETS="${SUBSETS:-summarization,translation,qa}"
MAX_REQUESTS="${MAX_REQUESTS:-40}"
OUTPUT="${OUTPUT:-$ROOT/benchmark-results/t5gemma-5k}"
SERVER_URL="http://127.0.0.1:$PORT"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT

wait_for_server() {
    local pid="$1"
    for _ in $(seq 1 300); do
        if curl -sf "$SERVER_URL/health" >/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "vLLM exited before becoming healthy" >&2
            return 1
        fi
        sleep 2
    done
    echo "Timed out waiting for vLLM" >&2
    return 1
}

run_eval() {
    local output_dir="$1"
    shift
    python "$EVAL_DIR/evaluate.py" \
        --target "$SERVER_URL/v1" \
        --subsets "$SUBSETS" \
        --max-requests "$MAX_REQUESTS" \
        --output-dir "$output_dir" \
        "$@" sweep
}

mkdir -p "$OUTPUT/logs"

echo "=== Baseline: $VERIFIER ==="
vllm serve "$VERIFIER" \
    --host 127.0.0.1 --port "$PORT" \
    --served-model-name t5gemma-2-1b-1b \
    --trust-remote-code --no-enable-chunked-prefill \
    >"$OUTPUT/logs/baseline.log" 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_PID"
run_eval "$OUTPUT/baseline" --allow-no-spec
cleanup

echo "=== DFlash: $DRAFT ==="
python "$ROOT/scripts/serve_t5gemma_dflash.py" \
    --checkpoint "$DRAFT" --port "$PORT" \
    >"$OUTPUT/logs/dflash.log" 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_PID"
run_eval "$OUTPUT/dflash"
cleanup

python "$EVAL_DIR/summarize_t5gemma_benchmark.py" \
    "$OUTPUT/baseline/perf_results.csv" \
    "$OUTPUT/dflash/perf_results.csv" \
    --output "$OUTPUT/comparison.csv"

echo "Acceptance metrics: $OUTPUT/dflash/acceptance.csv"
