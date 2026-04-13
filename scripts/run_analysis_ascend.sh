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
elif [[ -d models/faster-whisper-small ]]; then
    WHISPER_MODEL="small"
    ok "  faster-whisper-small ✓"
else
    info "  whisper-small / faster-whisper-small not found — subtitles/meta stay primary, offline ASR fallback disabled"
fi

# ── Step 5: Start vLLM ────────────────────────────────────────────────────
TP_SIZE_PER_INSTANCE="${TP_SIZE_PER_INSTANCE:-4}"
INSTANCE_COUNT="${INSTANCE_COUNT:-}"
if [[ -z "${INSTANCE_COUNT}" ]]; then
    if (( NPU_COUNT >= 16 )) && (( TP_SIZE_PER_INSTANCE == 4 )); then
        INSTANCE_COUNT=$(( NPU_COUNT / TP_SIZE_PER_INSTANCE ))
    else
        INSTANCE_COUNT=1
    fi
fi
if (( INSTANCE_COUNT < 1 )); then
    INSTANCE_COUNT=1
fi
MAX_INSTANCES=$(( NPU_COUNT / TP_SIZE_PER_INSTANCE ))
if (( MAX_INSTANCES < 1 )); then
    MAX_INSTANCES=1
fi
if (( INSTANCE_COUNT > MAX_INSTANCES )); then
    info "Reducing INSTANCE_COUNT=${INSTANCE_COUNT} to ${MAX_INSTANCES} based on NPU count"
    INSTANCE_COUNT="${MAX_INSTANCES}"
fi

declare -a API_BASES=()

collect_running_bases() {
    local bases=()
    local i port
    for (( i=0; i<INSTANCE_COUNT; i++ )); do
        port=$(( VLLM_PORT + i ))
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            bases+=("http://localhost:${port}/v1")
        fi
    done
    printf '%s\n' "${bases[@]}"
}

start_vllm_instance() {
    local idx="$1"
    local port="$2"
    local start_dev=$(( idx * TP_SIZE_PER_INSTANCE ))
    local end_dev=$(( start_dev + TP_SIZE_PER_INSTANCE - 1 ))
    local devices=""
    local d
    for (( d=start_dev; d<=end_dev; d++ )); do
        if [[ -n "${devices}" ]]; then devices+=",${d}"; else devices="${d}"; fi
    done

    info "Starting vLLM instance ${idx} on port ${port} (devices=${devices}, TP=${TP_SIZE_PER_INSTANCE})..."
    env ASCEND_RT_VISIBLE_DEVICES="${devices}" ASCEND_VISIBLE_DEVICES="${devices}" \
        vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$port" \
            --tensor-parallel-size "${TP_SIZE_PER_INSTANCE}" \
            --max-model-len 16384 \
            --enforce-eager \
            --trust-remote-code \
            --reasoning-parser qwen3 \
            --allowed-local-media-path "$(pwd)/cache" \
            > "/tmp/vllm_${port}.log" 2>&1 &
}

while IFS= read -r base; do
    [[ -n "${base}" ]] && API_BASES+=("${base}")
done < <(collect_running_bases)

if (( ${#API_BASES[@]} == INSTANCE_COUNT )); then
    SERVED_MODEL=$(curl -sf "${API_BASES[0]}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "Reusing ${#API_BASES[@]} running vLLM endpoint(s): ${BOLD}${SERVED_MODEL}${NC}"
else
    if (( ${#API_BASES[@]} > 0 )); then
        info "Found ${#API_BASES[@]} existing endpoint(s); starting the remaining $(( INSTANCE_COUNT - ${#API_BASES[@]} )) instance(s)..."
    else
        info "Starting ${INSTANCE_COUNT} vLLM instance(s) for pooled scene-parallel processing..."
    fi

    for (( i=0; i<INSTANCE_COUNT; i++ )); do
        port=$(( VLLM_PORT + i ))
        if ! curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            start_vllm_instance "$i" "$port"
        fi
    done

    WAITED=0
    while true; do
        API_BASES=()
        while IFS= read -r base; do
            [[ -n "${base}" ]] && API_BASES+=("${base}")
        done < <(collect_running_bases)

        if (( ${#API_BASES[@]} == INSTANCE_COUNT )); then
            break
        fi

        sleep 10
        WAITED=$(( WAITED + 10 ))
        if (( WAITED >= 1200 )); then
            err "vLLM pool not ready after 20min"
            ls /tmp/vllm_*.log 2>/dev/null | xargs -r tail -20
            exit 1
        fi
        if (( WAITED % 60 == 0 )); then
            info "  Still waiting for vLLM pool... (${WAITED}s)"
        fi
    done

    SERVED_MODEL=$(curl -sf "${API_BASES[0]}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "vLLM pool ready after ${WAITED}s — ${BOLD}${SERVED_MODEL}${NC} across ${#API_BASES[@]} endpoint(s)"
fi

API_BASE="${API_BASES[0]}"
API_BASES_CSV="$(IFS=,; echo "${API_BASES[*]}")"

export VIDIFY_PARALLEL_SEGMENTS=1
export VIDIFY_SEGMENTOR="${VIDIFY_SEGMENTOR:-scene}"
export VIDIFY_SEGMENT_DURATION="${VIDIFY_SEGMENT_DURATION:-180}"
export VIDIFY_SCENE_THRESHOLD="${VIDIFY_SCENE_THRESHOLD:-0.25}"
export VIDIFY_PARALLEL_WORKERS="${VIDIFY_PARALLEL_WORKERS:-$(( INSTANCE_COUNT * 2 ))}"
export VIDIFY_MIN_VIDEO_DURATION="${VIDIFY_MIN_VIDEO_DURATION:-120}"
export VIDIFY_MIN_SEGMENT_DURATION="${VIDIFY_MIN_SEGMENT_DURATION:-20}"
export VIDIFY_PARALLEL_ASR="${VIDIFY_PARALLEL_ASR:-1}"
export VIDIFY_ASR_WORKERS="${VIDIFY_ASR_WORKERS:-4}"
export VIDIFY_ASR_SEGMENT_DURATION="${VIDIFY_ASR_SEGMENT_DURATION:-240}"
export VIDIFY_ASR_MIN_AUDIO_DURATION="${VIDIFY_ASR_MIN_AUDIO_DURATION:-300}"
export VIDIFY_ASR_MIN_SEGMENT_DURATION="${VIDIFY_ASR_MIN_SEGMENT_DURATION:-30}"
export VIDIFY_ASR_DEVICES="${VIDIFY_ASR_DEVICES:-cpu,cpu,cpu,cpu}"

cat > "${ASCEND_CONFIG}" <<YAML
llm_base_url: "${API_BASES_CSV}"
llm_model: "${SERVED_MODEL}"
whisper_model: "${WHISPER_MODEL}"
YAML

info "Transcript-first mode enabled: subtitles/local ASR first, visual fallback only when needed"
info "Using served model id: ${SERVED_MODEL}"
info "Scene-parallel endpoints: ${API_BASES_CSV}"
info "Parallel ASR workers: ${VIDIFY_ASR_WORKERS} on devices=${VIDIFY_ASR_DEVICES}"
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
