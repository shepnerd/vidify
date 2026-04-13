#!/usr/bin/env bash
# run_analysis_ascend.sh — Run full detailed analysis on Ascend 910C.
#
# Starts vLLM with Qwen3.5-9B, then runs Vidify detailed analysis
# (ASR, emotion analysis, translation, OCR, object detection, captioning).
#
# Usage (inside D-cluster pod):
#   bash scripts/run_analysis_ascend.sh                                    # auto-find video
#   bash scripts/run_analysis_ascend.sh /workspace/vidify/cache/downloads/SSya123u9Yk.mp4
#   bash scripts/run_analysis_ascend.sh --server-only                      # only start vLLM

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}[analysis]${NC} $*"; }
ok()    { echo -e "${GREEN}[analysis]${NC} $*"; }
err()   { echo -e "${RED}[analysis]${NC} $*" >&2; }

# ── Parse args ─────────────────────────────────────────────────────────────
VIDEO_PATH=""
SERVER_ONLY=false
VLLM_PORT=8000
MODE="detailed"
ASCEND_CONFIG="/tmp/vidify_ascend_config.yaml"

trap 'rm -f "$ASCEND_CONFIG"' EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-only) SERVER_ONLY=true; shift;;
        --port)        VLLM_PORT="$2"; shift 2;;
        --mode)        MODE="$2"; shift 2;;
        -h|--help)     sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"; exit 0;;
        *)
            if [[ -z "$VIDEO_PATH" ]]; then VIDEO_PATH="$1"; shift
            else err "Unknown argument: $1"; exit 1; fi;;
    esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Vidify Analysis — Qwen3.5 on Ascend 910C              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Ascend environment ─────────────────────────────────────────────
info "Setting up Ascend environment..."
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
set -u
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

NPU_COUNT=$(python3 -c "import torch, torch_npu; print(torch.npu.device_count())" 2>/dev/null || echo "0")
if [[ "$NPU_COUNT" -eq 0 ]]; then
    err "No Ascend NPUs detected!"; exit 1
fi
ok "Detected ${BOLD}${NPU_COUNT} NPUs${NC}"

# ── Step 2: Find video ────────────────────────────────────────────────────
if [[ -z "$VIDEO_PATH" ]]; then
    # Auto-detect cached video
    for candidate in \
        cache/downloads/SSya123u9Yk.mp4 \
        cache/downloads/*.mp4 \
        /data/videos/*.mp4; do
        # Expand glob
        for f in $candidate; do
            if [[ -f "$f" ]]; then
                VIDEO_PATH="$f"
                break 2
            fi
        done
    done
fi

if [[ -z "$VIDEO_PATH" || ! -f "$VIDEO_PATH" ]]; then
    err "No video found! Provide a path: bash $0 <video.mp4>"
    exit 1
fi
ok "Video: ${BOLD}${VIDEO_PATH}${NC} ($(du -sh "$VIDEO_PATH" | cut -f1))"

# ── Step 3: Find Qwen3.5 model ────────────────────────────────────────────
MODEL_PATH=""
for candidate in \
    /data/models/Qwen3.5-9B \
    /data/sfteval/models/Qwen3.5-9B; do
    if [[ -d "$candidate" ]]; then
        MODEL_PATH="$candidate"; break
    fi
done

if [[ -z "$MODEL_PATH" ]]; then
    info "Downloading Qwen3.5-9B via hf-mirror..."
    MODEL_PATH="/data/models/Qwen3.5-9B"
    HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B', local_dir='$MODEL_PATH')
"
fi
ok "Model: ${BOLD}${MODEL_PATH}${NC}"

# ── Step 4: Verify local models (transcript-first workflow) ───────────────
info "Checking local skill models..."
for model_dir in models/wav2vec2-base-superb-er models/opus-mt-en-zh; do
    if [[ -d "$model_dir" ]]; then
        ok "  $(basename "$model_dir") ✓"
    else
        err "  $(basename "$model_dir") MISSING — related skill may degrade"
    fi
done

WHISPER_MODEL=""
if [[ -d models/whisper-small ]]; then
    WHISPER_MODEL="small"
    ok "  whisper-small ✓"
else
    info "  whisper-small not found — subtitles/meta stay primary, offline ASR fallback disabled"
fi

# ── Step 5: Start vLLM ────────────────────────────────────────────────────
API_BASE="http://localhost:${VLLM_PORT}/v1"

if curl -sf "${API_BASE}/models" >/dev/null 2>&1; then
    SERVED_MODEL=$(curl -sf "${API_BASE}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "vLLM already running: ${BOLD}${SERVED_MODEL}${NC}"
else
    info "Starting vLLM (TP=4, enforce-eager, max-model-len=16384)..."
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 4 \
        --max-model-len 16384 \
        --enforce-eager \
        --trust-remote-code \
        --reasoning-parser qwen3 \
        --allowed-local-media-path "$(pwd)/cache" \
        > /tmp/vllm.log 2>&1 &
    VLLM_PID=$!

    info "Waiting for vLLM (PID=${VLLM_PID})..."
    WAITED=0
    while ! curl -sf "${API_BASE}/models" >/dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM died! Last logs:"; tail -30 /tmp/vllm.log; exit 1
        fi
        sleep 10; WAITED=$((WAITED + 10))
        if (( WAITED >= 1200 )); then
            err "vLLM not ready after 20min"; tail -50 /tmp/vllm.log; exit 1
        fi
        if (( WAITED % 60 == 0 )); then info "  Still waiting... (${WAITED}s)"; fi
    done

    SERVED_MODEL=$(curl -sf "${API_BASE}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "vLLM ready after ${WAITED}s — ${BOLD}${SERVED_MODEL}${NC}"
fi

cat > "${ASCEND_CONFIG}" <<YAML
llm_base_url: "${API_BASE}"
llm_model: "${SERVED_MODEL}"
whisper_model: "${WHISPER_MODEL}"
YAML

info "Transcript-first mode enabled: subtitles/local ASR first, visual fallback only when needed"
info "Using served model id: ${SERVED_MODEL}"
if [[ -n "${WHISPER_MODEL}" ]]; then
    info "Using offline Whisper model: ${WHISPER_MODEL}"
fi

if $SERVER_ONLY; then
    ok "Server-only mode. vLLM running on port ${VLLM_PORT}."
    echo "  tail -f /tmp/vllm.log"
    exit 0
fi

# ── Step 6: Run analysis ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
info "Running ${MODE} analysis on: $(basename "$VIDEO_PATH")"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

START_TIME=$(date +%s)

python3 agent/main.py --config "${ASCEND_CONFIG}" analyze local "$VIDEO_PATH" \
    --mode "$MODE" \
    --cache-root ./cache \
    2>&1 | tee /tmp/analysis.log

EXIT_CODE=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    ok "Analysis completed in ${MINUTES}m${SECONDS_REM}s"
    echo ""
    # Show output location
    info "Results in ./cache/ — look for analysis JSON files"
    ls -la cache/*.json 2>/dev/null || ls -la cache/videos/*/analysis*.json 2>/dev/null || true
else
    err "Analysis failed (exit code ${EXIT_CODE}) after ${MINUTES}m${SECONDS_REM}s"
    err "Check /tmp/analysis.log for details"
fi

echo ""
ok "Pod still alive. You can:"
echo "  - Inspect results: cat cache/videos/*/analysis*.json | python3 -m json.tool | head -100"
echo "  - Start chat:     python3 agent/main.py chat local $VIDEO_PATH --cache-root ./cache"
echo "  - Re-run:         bash scripts/run_analysis_ascend.sh $VIDEO_PATH --mode highlights"
