#!/usr/bin/env bash
# Run a Vidify analysis against an Ascend/NPU-backed OpenAI-compatible endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen3.5-9b}"
VIDEO_PATH="${1:-}"
MODE="${MODE:-detailed}"
export LLM_BASE_URL="$API_BASE"
export LLM_MODEL="$MODEL"

if [[ -z "$VIDEO_PATH" ]]; then
    echo "Usage: API_BASE=http://host:8000/v1 bash scripts/run_analysis_ascend.sh VIDEO_PATH" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
python -m agent.main --config "${CONFIG:-config.yaml}" analyze local "$VIDEO_PATH" \
    --mode "$MODE" \
    --cache-root ./cache \
    --max-frames "${MAX_FRAMES:-128}"
