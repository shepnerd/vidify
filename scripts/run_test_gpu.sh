#!/usr/bin/env bash
# run_test_gpu.sh — One-command script to launch vLLM on GPU cluster and run test suite.
#
# Handles the full lifecycle:
#   1. Load .env (cluster config, CUDA paths, mounts)
#   2. Discover existing vLLM endpoint or launch a new one via rl.sh
#   3. Wait for vLLM to be ready (flashinfer JIT compilation may take several minutes)
#   4. Run test_all.py against the video
#
# Usage:
#   bash scripts/run_test_gpu.sh                                          # defaults: Qwen3.5, 4 GPUs, media/taste_in_china_s1e1.mp4
#   bash scripts/run_test_gpu.sh --video media/my_video.mp4               # custom video
#   bash scripts/run_test_gpu.sh --gpu 2 --video media/my_video.mp4       # 2 GPUs
#   bash scripts/run_test_gpu.sh --api-base http://10.0.0.1:8000/v1       # skip launch, use existing endpoint
#   bash scripts/run_test_gpu.sh --tests "frame_caption video_qa"         # run specific tests only
#   bash scripts/run_test_gpu.sh --model qwen3vl                          # use Qwen3-VL instead of Qwen3.5

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU=4
VIDEO="${PROJECT_ROOT}/media/taste_in_china_s1e1.mp4"
API_BASE=""
TESTS=""
MODEL="qwen3.5"     # "qwen3.5" or "qwen3vl"
TIMEOUT=900          # max seconds to wait for vLLM

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2"; shift 2;;
        --video)     VIDEO="$2"; shift 2;;
        --api-base)  API_BASE="$2"; shift 2;;
        --tests)     TESTS="$2"; shift 2;;
        --model)     MODEL="$2"; shift 2;;
        --timeout)   TIMEOUT="$2"; shift 2;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    echo "[env] Loading ${PROJECT_ROOT}/.env"
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
else
    echo "[WARN] No .env found at ${PROJECT_ROOT}/.env" >&2
fi

echo "============================================================"
echo " VidCopilot GPU Test Runner"
echo "============================================================"
echo "  Model:       ${MODEL}"
echo "  GPU:         ${GPU}"
echo "  Video:       ${VIDEO}"
echo "  CUDA_HOME:   ${CUDA_HOME:-<not set>}"
echo "  RL_MOUNT:    ${RL_MOUNT:-<not set>}"
echo "  RL_GROUP:    ${RL_CHARGED_GROUP:-<not set>}"
echo "============================================================"

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ ! -f "${VIDEO}" ]]; then
    echo "ERROR: Video not found: ${VIDEO}" >&2
    exit 1
fi

# ── Step 1: Discover or launch vLLM ─────────────────────────────────────────
if [[ -n "${API_BASE}" ]]; then
    echo "[vllm] Using provided endpoint: ${API_BASE}"
    # Quick health check
    if ! curl -sf "${API_BASE}/models" >/dev/null 2>&1; then
        echo "ERROR: vLLM not reachable at ${API_BASE}" >&2
        exit 1
    fi
    echo "[vllm] Endpoint is healthy"
else
    # Check for a cached serving IP from a previous launch
    SERVING_IP_FILE="${PROJECT_ROOT}/cache/.serving/serving_ip.txt"
    if [[ -f "${SERVING_IP_FILE}" ]]; then
        CACHED_IP=$(cat "${SERVING_IP_FILE}")
        CANDIDATE="http://${CACHED_IP}:8000/v1"
        echo "[vllm] Found cached IP: ${CACHED_IP}, probing ${CANDIDATE} ..."
        if curl -sf "${CANDIDATE}/models" >/dev/null 2>&1; then
            echo "[vllm] Existing vLLM is alive!"
            API_BASE="${CANDIDATE}"
        else
            echo "[vllm] Cached endpoint is stale, will launch new instance"
        fi
    fi

    if [[ -z "${API_BASE}" ]]; then
        echo "[vllm] Launching vLLM on ${GPU} GPUs via rl.sh ..."

        # Build the inner command
        SERVING_DIR="${PROJECT_ROOT}/cache/.serving"
        LOG_FILE="${SERVING_DIR}/vllm.log"
        mkdir -p "${SERVING_DIR}"
        rm -f "${SERVING_IP_FILE}" "${LOG_FILE}"

        # Resolve model path
        if [[ "${MODEL}" == "qwen3.5" ]]; then
            QWEN35_CACHE="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"
            if [[ -d "${QWEN35_CACHE}" ]]; then
                MODEL_PATH=$(ls -1d "${QWEN35_CACHE}"/*/ 2>/dev/null | head -1 | sed 's:/$::')
            else
                MODEL_PATH="Qwen/Qwen3.5-9B"
            fi
            EXTRA_VLLM_ARGS="--reasoning-parser qwen3 --max-model-len 65536"
        else
            QWEN3VL_CACHE="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
            if [[ -d "${QWEN3VL_CACHE}" ]]; then
                MODEL_PATH=$(ls -1d "${QWEN3VL_CACHE}"/*/ 2>/dev/null | head -1 | sed 's:/$::')
            else
                MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
            fi
            EXTRA_VLLM_ARGS="--max-model-len 32768"
        fi

        # Build env setup for CUDA (needed for flashinfer JIT on Qwen3.5)
        ENV_SETUP=""
        if [[ -n "${CUDA_HOME:-}" ]]; then
            ENV_SETUP="export CUDA_HOME=${CUDA_HOME}; export PATH=\$CUDA_HOME/bin:\$PATH; "
        fi

        INNER_CMD="${ENV_SETUP}"'IP=$(hostname -I | awk '"'"'{print $1}'"'"'); '
        INNER_CMD+="echo \"\$IP\" > ${SERVING_IP_FILE}; "
        INNER_CMD+="echo \"[serving] Node IP: \$IP\" | tee ${LOG_FILE}; "
        INNER_CMD+="exec vllm serve ${MODEL_PATH} "
        INNER_CMD+="--host 0.0.0.0 --port 8000 "
        INNER_CMD+="--tensor-parallel-size ${GPU} "
        INNER_CMD+="--allowed-local-media-path ${PROJECT_ROOT}/cache "
        INNER_CMD+="${EXTRA_VLLM_ARGS} "
        INNER_CMD+="2>&1 | tee -a ${LOG_FILE}"

        # Launch via rl.sh in background
        bash "${SCRIPT_DIR}/rl.sh" -gpu "${GPU}" -- bash -c "${INNER_CMD}" &
        RL_PID=$!
        echo "[vllm] rlaunch launched (pid=${RL_PID})"

        # Wait for IP file
        echo "[vllm] Waiting for GPU node to report its IP ..."
        ELAPSED=0
        while [[ ! -f "${SERVING_IP_FILE}" ]] || [[ ! -s "${SERVING_IP_FILE}" ]]; do
            if ! kill -0 "${RL_PID}" 2>/dev/null; then
                echo "ERROR: rlaunch process died. Check logs at ${LOG_FILE}" >&2
                exit 1
            fi
            if (( ELAPSED >= TIMEOUT )); then
                echo "ERROR: Timed out (${TIMEOUT}s) waiting for GPU node IP" >&2
                kill "${RL_PID}" 2>/dev/null || true
                exit 1
            fi
            sleep 10
            ELAPSED=$((ELAPSED + 10))
            echo "  ... ${ELAPSED}s elapsed"
        done
        NODE_IP=$(cat "${SERVING_IP_FILE}")
        API_BASE="http://${NODE_IP}:8000/v1"
        echo "[vllm] GPU node IP: ${NODE_IP}"

        # Wait for vLLM /v1/models (flashinfer JIT may take 5-10 min on first run)
        echo "[vllm] Waiting for vLLM to be ready at ${API_BASE} ..."
        while ! curl -sf "${API_BASE}/models" >/dev/null 2>&1; do
            if ! kill -0 "${RL_PID}" 2>/dev/null; then
                echo "ERROR: rlaunch process died. Check logs:" >&2
                tail -20 "${LOG_FILE}" 2>/dev/null
                exit 1
            fi
            if (( ELAPSED >= TIMEOUT )); then
                echo "ERROR: Timed out (${TIMEOUT}s) waiting for vLLM" >&2
                kill "${RL_PID}" 2>/dev/null || true
                exit 1
            fi
            sleep 15
            ELAPSED=$((ELAPSED + 15))
            echo "  ... ${ELAPSED}s elapsed (flashinfer JIT compilation may take a while on first run)"
        done
        echo "[vllm] vLLM is ready!"
    fi
fi

# ── Step 2: Run test suite ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Running test_all.py"
echo "============================================================"

TEST_CMD=(python "${SCRIPT_DIR}/test_all.py"
    --video-path "${VIDEO}"
    --api-base "${API_BASE}"
    --gpu "${GPU}"
)
if [[ -n "${TESTS}" ]]; then
    TEST_CMD+=(--tests ${TESTS})
fi

"${TEST_CMD[@]}"
