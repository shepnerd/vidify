#!/usr/bin/env bash
# Serve Qwen3.5 on Ascend/NPU with vLLM's Ascend backend.

set -euo pipefail

MODEL="${1:-${VIDIFY_VLLM_MODEL:-Qwen/Qwen3.5-9B}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$(pwd)/cache}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh || true
fi

mkdir -p "$ALLOWED_LOCAL_MEDIA_PATH"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
    --reasoning-parser qwen3 \
    --enforce-eager
