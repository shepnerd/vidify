#!/usr/bin/env bash
# run_test_ascend.sh — Run Vidify test suite on D-cluster Ascend 910C nodes.
#
# This script handles two modes:
#   1. Connect to an already-running vLLM endpoint (--api-base)
#   2. Launch a job on D-cluster, start vLLM inside, run tests, then clean up
#
# Usage:
#   bash scripts/run_test_ascend.sh --api-base http://10.x.x.x:8000/v1           # existing endpoint
#   bash scripts/run_test_ascend.sh --video /data/videos/test.mp4                  # launch new job
#   bash scripts/run_test_ascend.sh --npus 4 --tests "frame_caption video_qa"      # custom config

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
NPUS=2
VIDEO="${PROJECT_ROOT}/media/taste_in_china_s1e1.mp4"
API_BASE=""
TESTS=""
MODEL="qwen3.5"
TIMEOUT=600
JOB_NAME=""
INFRA_DIR="${PROJECT_ROOT}/../workbench/infra/d-cluster"

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --npus)      NPUS="$2"; shift 2;;
        --video)     VIDEO="$2"; shift 2;;
        --api-base)  API_BASE="$2"; shift 2;;
        --tests)     TESTS="$2"; shift 2;;
        --model)     MODEL="$2"; shift 2;;
        --timeout)   TIMEOUT="$2"; shift 2;;
        --infra-dir) INFRA_DIR="$2"; shift 2;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

echo "============================================================"
echo " Vidify Ascend 910C Test Runner"
echo "============================================================"
echo "  Model:       ${MODEL}"
echo "  NPUs:        ${NPUS}"
echo "  Video:       ${VIDEO}"
echo "  Infra dir:   ${INFRA_DIR}"
echo "============================================================"

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ ! -f "${VIDEO}" ]]; then
    echo "WARNING: Video not found locally: ${VIDEO}" >&2
    echo "  (it may exist on the cluster shared filesystem)" >&2
fi

# ── Step 1: Discover or launch vLLM ─────────────────────────────────────────
if [[ -n "${API_BASE}" ]]; then
    echo "[vllm] Using provided endpoint: ${API_BASE}"
    if curl -sf "${API_BASE}/models" >/dev/null 2>&1; then
        echo "[vllm] Endpoint is healthy"
    else
        echo "ERROR: vLLM not reachable at ${API_BASE}" >&2
        exit 1
    fi
else
    # Check for cached serving IP
    SERVING_DIR="${PROJECT_ROOT}/cache/.serving"
    SERVING_IP_FILE="${SERVING_DIR}/serving_ip_ascend.txt"
    if [[ -f "${SERVING_IP_FILE}" ]]; then
        CACHED_IP=$(cat "${SERVING_IP_FILE}")
        CANDIDATE="http://${CACHED_IP}:8000/v1"
        echo "[vllm] Found cached IP: ${CACHED_IP}, probing ..."
        if curl -sf "${CANDIDATE}/models" >/dev/null 2>&1; then
            echo "[vllm] Existing vLLM is alive!"
            API_BASE="${CANDIDATE}"
        else
            echo "[vllm] Cached endpoint is stale"
        fi
    fi

    if [[ -z "${API_BASE}" ]]; then
        # Verify infra scripts exist
        if [[ ! -f "${INFRA_DIR}/job-run.sh" ]]; then
            echo "ERROR: infra scripts not found at ${INFRA_DIR}" >&2
            echo "  Set --infra-dir to the d-cluster infra directory" >&2
            exit 1
        fi

        JOB_NAME="vidify-test-$(date +%H%M%S)"
        echo "[cluster] Launching job '${JOB_NAME}' with ${NPUS} NPUs ..."

        # Launch job using the vidify template
        bash "${INFRA_DIR}/job-run.sh" "${JOB_NAME}" \
            -f "${PROJECT_ROOT}/infra/d-cluster/job-vidify.yaml"

        # Wait for pod to be running
        echo "[cluster] Waiting for pod to be ready ..."
        ELAPSED=0
        POD_NAME="${JOB_NAME}-master-0"
        while true; do
            STATUS=$(kubectl get pod "${POD_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
            if [[ "$STATUS" == "Running" ]]; then
                echo "[cluster] Pod is running"
                break
            fi
            if (( ELAPSED >= TIMEOUT )); then
                echo "ERROR: Timed out (${TIMEOUT}s) waiting for pod" >&2
                exit 1
            fi
            sleep 10
            ELAPSED=$((ELAPSED + 10))
            echo "  ... ${ELAPSED}s elapsed (status: ${STATUS})"
        done

        # Get pod IP
        POD_IP=$(kubectl get pod "${POD_NAME}" -o jsonpath='{.status.podIP}')
        echo "[cluster] Pod IP: ${POD_IP}"
        mkdir -p "${SERVING_DIR}"
        echo "${POD_IP}" > "${SERVING_IP_FILE}"

        # Start vLLM inside the pod
        echo "[vllm] Starting vLLM on NPU inside pod ..."
        kubectl exec "${POD_NAME}" -- bash -c \
            "cd /workspace/vidify && nohup bash scripts/serving_qwen3_5_ascend.sh > /tmp/vllm.log 2>&1 &"

        # Wait for vLLM to be ready
        API_BASE="http://${POD_IP}:8000/v1"
        echo "[vllm] Waiting for vLLM at ${API_BASE} ..."
        while ! curl -sf "${API_BASE}/models" >/dev/null 2>&1; do
            if (( ELAPSED >= TIMEOUT )); then
                echo "ERROR: Timed out (${TIMEOUT}s) waiting for vLLM" >&2
                echo "Last logs:" >&2
                kubectl exec "${POD_NAME}" -- tail -20 /tmp/vllm.log 2>/dev/null || true
                exit 1
            fi
            sleep 15
            ELAPSED=$((ELAPSED + 15))
            echo "  ... ${ELAPSED}s elapsed"
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
)
if [[ -n "${TESTS}" ]]; then
    TEST_CMD+=(--tests ${TESTS})
fi

"${TEST_CMD[@]}"
TEST_EXIT=$?

# ── Step 3: Cleanup ─────────────────────────────────────────────────────────
if [[ -n "${JOB_NAME}" ]]; then
    echo ""
    echo "[cleanup] Deleting job '${JOB_NAME}' ..."
    bash "${INFRA_DIR}/job-delete.sh" "${JOB_NAME}" -y 2>/dev/null || true
    rm -f "${SERVING_IP_FILE}"
fi

exit ${TEST_EXIT}
