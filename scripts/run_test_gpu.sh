#!/usr/bin/env bash
# Run the Vidify validation suite against a GPU-backed vLLM endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_BASE="${API_BASE:-http://localhost:8000/v1}"
VIDEO_PATH="${VIDEO_PATH:-}"
TESTS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_test_gpu.sh [--api-base URL] --video PATH [--tests "test_a test_b"]

This script does not start a managed job. Start vLLM separately with
your preferred GPU scheduler, or run it locally, then pass --api-base.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-base) API_BASE="$2"; shift 2 ;;
        --video|--video-path) VIDEO_PATH="$2"; shift 2 ;;
        --tests) TESTS=(--tests $2); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$VIDEO_PATH" ]]; then
    echo "ERROR: --video PATH is required." >&2
    usage
    exit 2
fi

cd "$PROJECT_ROOT"
python scripts/test_all.py \
    --video-path "$VIDEO_PATH" \
    --api-base "$API_BASE" \
    "${TESTS[@]}"
