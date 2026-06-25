#!/usr/bin/env bash
# Convenience wrapper for starting an Ascend/NPU-backed vLLM server and Vidify.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_PATH="${1:-}"
SERVER_ONLY="${SERVER_ONLY:-0}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
export API_BASE

if [[ "${1:-}" == "--server-only" ]]; then
    SERVER_ONLY=1
    VIDEO_PATH=""
fi

cd "$PROJECT_ROOT"

bash scripts/serving_qwen3_5_ascend.sh &
SERVER_PID=$!

echo "Started vLLM process ${SERVER_PID}; waiting for ${API_BASE} ..."
python - <<'PY'
import os
import time
import requests

base = os.environ.get("API_BASE", "http://localhost:8000/v1").rstrip("/")
for _ in range(120):
    try:
        r = requests.get(f"{base}/models", timeout=5)
        if r.status_code == 200:
            print("vLLM is ready.")
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(5)
raise SystemExit("Timed out waiting for vLLM.")
PY

if [[ "$SERVER_ONLY" == "1" ]]; then
    wait "$SERVER_PID"
elif [[ -n "$VIDEO_PATH" ]]; then
    python -m agent.main chat local "$VIDEO_PATH" --chat-api-base "$API_BASE"
else
    uvicorn server.app:app --host 0.0.0.0 --port "${VIDIFY_PORT:-9000}"
fi
