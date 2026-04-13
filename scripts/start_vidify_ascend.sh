#!/usr/bin/env bash
# start_vidify_ascend.sh — One-command start for Vidify + Qwen3.5 on Ascend 910C.
#
# Starts vLLM with Qwen3.5-9B in background, waits for readiness, then drops
# into interactive chat mode. Run inside a D-cluster pod.
#
# Usage:
#   bash scripts/start_vidify_ascend.sh                              # interactive chat
#   bash scripts/start_vidify_ascend.sh /data/videos/myvideo.mp4     # with a specific video
#   bash scripts/start_vidify_ascend.sh --model /data/models/Qwen3.5-9B  # custom model path
#   bash scripts/start_vidify_ascend.sh --server-only                # only start vLLM, no chat
#
# What it does:
#   1. Checks Ascend NPU availability
#   2. Downloads Qwen3.5-9B model if not present (via hf-mirror)
#   3. Installs Python deps (if needed)
#   4. Starts vLLM server in background (TP=4, --enforce-eager)
#   5. Waits for vLLM to be ready
#   6. Launches interactive video chat

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[vidify]${NC} $*"; }
ok()    { echo -e "${GREEN}[vidify]${NC} $*"; }
err()   { echo -e "${RED}[vidify]${NC} $*" >&2; }

# ── Parse args ──────────────────────────────────────────────────────────────
MODEL_PATH=""
VIDEO_PATH=""
SERVER_ONLY=false
VLLM_PORT=8000
CACHE_ROOT="./cache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL_PATH="$2"; shift 2;;
        --port)        VLLM_PORT="$2"; shift 2;;
        --cache-root)  CACHE_ROOT="$2"; shift 2;;
        --server-only) SERVER_ONLY=true; shift;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"
            exit 0;;
        *)
            if [[ -z "$VIDEO_PATH" ]]; then
                VIDEO_PATH="$1"; shift
            else
                err "Unknown argument: $1"; exit 1
            fi;;
    esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Vidify + Qwen3.5 on Ascend 910C              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Ascend environment ──────────────────────────────────────────────
info "Setting up Ascend environment..."
# Temporarily disable nounset — Ascend set_env.sh references ZSH_VERSION which is unbound
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
set -u
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

NPU_COUNT=$(python3 -c "import torch, torch_npu; print(torch.npu.device_count())" 2>/dev/null || echo "0")
if [[ "$NPU_COUNT" -eq 0 ]]; then
    err "No Ascend NPUs detected! Make sure you're in a D-cluster pod with NPUs."
    exit 1
fi
ok "Detected ${BOLD}${NPU_COUNT} NPUs${NC}"

# ── Step 2: Find or download model ─────────────────────────────────────────
if [[ -z "$MODEL_PATH" ]]; then
    for candidate in \
        /data/models/Qwen3.5-9B \
        /data/sfteval/models/Qwen3.5-9B \
        "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"; do
        if [[ -d "$candidate" ]]; then
            if [[ "$candidate" == *snapshots ]]; then
                MODEL_PATH=$(ls -1d "$candidate"/*/ 2>/dev/null | head -1 | sed 's:/$::')
            else
                MODEL_PATH="$candidate"
            fi
            break
        fi
    done
fi

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    info "Qwen3.5-9B not found locally. Downloading via hf-mirror..."
    MODEL_PATH="/data/models/Qwen3.5-9B"
    HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B', local_dir='$MODEL_PATH')
"
fi

ok "Model: ${BOLD}${MODEL_PATH}${NC}"

# ── Step 3: Install Python dependencies ─────────────────────────────────────
info "Checking Python dependencies..."
python3 -c "import click, openai, rich, yaml, faiss" 2>/dev/null || {
    info "Installing missing Python packages..."
    # IMPORTANT: Do NOT use requirements.txt — it will upgrade torch/vllm/numpy and break NPU.
    # Only install what's missing in the image. Use --no-deps to prevent pulling in torch/vllm.
    pip install -i https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/ \
        --trusted-host pkg.pjlab.org.cn --no-cache-dir --no-deps \
        rich click openai pydantic PyYAML requests faiss-cpu \
        beautifulsoup4 soupsieve yt-dlp jinja2 aiofiles ffmpeg-python \
        uvicorn fastapi 2>/dev/null || true
    # These need deps but are safe (they won't pull torch)
    pip install -i https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/ \
        --trusted-host pkg.pjlab.org.cn --no-cache-dir \
        faiss-cpu 2>/dev/null || true
}

# Ensure ffmpeg/ffprobe are available (needed for video processing)
which ffprobe >/dev/null 2>&1 || {
    info "Installing ffmpeg..."
    yum install -y ffmpeg 2>/dev/null || dnf install -y ffmpeg 2>/dev/null || \
        apt-get update && apt-get install -y ffmpeg 2>/dev/null || true
}

# Ensure transformers is new enough for Qwen3.5
python3 -c "
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert v >= (4, 51), f'transformers {transformers.__version__} too old, need >= 4.51'
" 2>/dev/null || {
    info "Upgrading transformers for Qwen3.5 support..."
    pip install -i https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/ \
        --trusted-host pkg.pjlab.org.cn --no-cache-dir \
        "transformers>=4.51" 2>/dev/null || true
}

WHISPER_MODEL=""
if [[ -d models/whisper-small ]]; then
    WHISPER_MODEL="small"
    ok "Offline ASR model available: ${BOLD}whisper-small${NC}"
elif [[ -d models/faster-whisper-small ]]; then
    WHISPER_MODEL="small"
    ok "Offline ASR model available: ${BOLD}faster-whisper-small${NC}"
else
    info "Offline ASR model not found; chat will stay transcript/meta-first and skip Whisper fallback"
fi

# ── Step 4: Start vLLM server ───────────────────────────────────────────────
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
if (( MAX_INSTANCES < 1 )); then MAX_INSTANCES=1; fi
if (( INSTANCE_COUNT > MAX_INSTANCES )); then INSTANCE_COUNT="${MAX_INSTANCES}"; fi

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
    mkdir -p /tmp
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
        WAITED=$((WAITED + 10))
        if (( WAITED >= 1200 )); then
            err "vLLM pool not ready after 20 minutes"
            ls /tmp/vllm_*.log 2>/dev/null | xargs -r tail -20
            exit 1
        fi
        if (( WAITED % 60 == 0 )); then
            info "  Still waiting... (${WAITED}s elapsed)"
        fi
    done

    SERVED_MODEL=$(curl -sf "${API_BASES[0]}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "vLLM pool ready after ${WAITED}s — serving ${BOLD}${SERVED_MODEL}${NC} across ${#API_BASES[@]} endpoint(s)"
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

# ── Step 5: Write config ────────────────────────────────────────────────────
cat > config.yaml <<YAML
llm_base_url: "${API_BASES_CSV}"
llm_model: "${SERVED_MODEL}"
whisper_model: "${WHISPER_MODEL}"
YAML

info "Using served model id: ${SERVED_MODEL}"
info "Scene-parallel endpoints: ${API_BASES_CSV}"
info "Parallel ASR workers: ${VIDIFY_ASR_WORKERS} on devices=${VIDIFY_ASR_DEVICES}"
if [[ -n "${WHISPER_MODEL}" ]]; then
    info "Configured offline Whisper fallback: ${WHISPER_MODEL}"
fi

if $SERVER_ONLY; then
    ok "Server-only mode. vLLM is running on port ${VLLM_PORT}."
    echo ""
    echo "  API base: ${API_BASE}"
    echo "  Model:    ${SERVED_MODEL}"
    echo "  Logs:     tail -f /tmp/vllm.log"
    echo ""
    echo "To start chat manually:"
    echo "  python agent/main.py chat local <video.mp4> --cache-root ${CACHE_ROOT}"
    echo ""
    exit 0
fi

# ── Step 6: Interactive chat ────────────────────────────────────────────────
echo ""
ok "Ready for interactive chat!"
echo ""

if [[ -n "$VIDEO_PATH" && -f "$VIDEO_PATH" ]]; then
    info "Starting chat with video: ${VIDEO_PATH}"
    echo ""
    exec python3 agent/main.py chat local "$VIDEO_PATH" \
        --cache-root "$CACHE_ROOT" \
        --chat-api-base "$API_BASE"
else
    echo -e "${BOLD}Usage:${NC}"
    echo "  Enter a video path to start chatting about it."
    echo ""
    echo "  Examples:"
    echo "    python agent/main.py chat local /data/videos/myvideo.mp4 --cache-root $CACHE_ROOT"
    echo "    python agent/main.py chat local cache/downloads/video.mp4"
    echo ""
    echo "  Or run analysis first:"
    echo "    python agent/main.py analyze local /data/videos/myvideo.mp4 --mode detailed"
    echo ""

    # Prompt for video path
    echo -ne "${CYAN}Enter video path (or 'skip' to get a shell): ${NC}"
    read -r user_video
    if [[ -n "$user_video" && "$user_video" != "skip" ]]; then
        exec python3 agent/main.py chat local "$user_video" \
            --cache-root "$CACHE_ROOT" \
            --chat-api-base "$API_BASE"
    else
        echo ""
        ok "vLLM running in background. Start chat anytime with:"
        echo "  python agent/main.py chat local <video.mp4> --cache-root $CACHE_ROOT"
        echo ""
    fi
fi
