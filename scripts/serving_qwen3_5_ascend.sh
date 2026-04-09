#!/usr/bin/env bash
# Serve Qwen3.5-9B on Ascend 910C NPU via vLLM + vllm_ascend.
#
# Prerequisites:
#   - Ascend 910C node with CANN toolkit installed
#   - vLLM >= 0.13.0 with vllm_ascend plugin (pre-installed in testenv:v0.1)
#   - torch_npu matching the CANN/PyTorch versions
#
# Usage (inside a D-cluster pod):
#   bash scripts/serving_qwen3_5_ascend.sh                          # auto-detect model
#   bash scripts/serving_qwen3_5_ascend.sh /data/models/Qwen3.5-9B  # explicit path
#   TP_SIZE=4 bash scripts/serving_qwen3_5_ascend.sh                # 4 NPUs

set -euo pipefail

# ── Ascend environment ──────────────────────────────────────────────────────
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || true
fi
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

# ── Model path ──────────────────────────────────────────────────────────────
MODEL="${1:-}"

if [[ -z "$MODEL" ]]; then
    # Try shared filesystem first
    for candidate in \
        /data/models/Qwen3.5-9B \
        /data/sfteval/models/Qwen3.5-9B \
        "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"; do
        if [[ -d "$candidate" ]]; then
            if [[ "$candidate" == *snapshots ]]; then
                MODEL=$(ls -1d "$candidate"/*/ 2>/dev/null | head -1 | sed 's:/$::')
            else
                MODEL="$candidate"
            fi
            break
        fi
    done

    if [[ -z "$MODEL" ]]; then
        MODEL="Qwen/Qwen3.5-9B"
        echo "No local model found, will download from HuggingFace: $MODEL"
    fi
fi

# ── Parameters ──────────────────────────────────────────────────────────────
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
TP_SIZE="${TP_SIZE:-2}"  # 910C cluster minimum is 2 NPUs

# ── Verify NPU availability ────────────────────────────────────────────────
NPU_COUNT=$(python3 -c "import torch, torch_npu; print(torch.npu.device_count())" 2>/dev/null || echo "0")
if [[ "$NPU_COUNT" -eq 0 ]]; then
    echo "ERROR: No Ascend NPUs detected. Ensure torch_npu is installed and NPU drivers are loaded." >&2
    exit 1
fi
echo "Detected ${NPU_COUNT} Ascend NPU(s)"

if [[ "$TP_SIZE" -gt "$NPU_COUNT" ]]; then
    echo "WARNING: TP_SIZE=${TP_SIZE} > available NPUs=${NPU_COUNT}, reducing to ${NPU_COUNT}" >&2
    TP_SIZE="$NPU_COUNT"
fi

# ── Launch ──────────────────────────────────────────────────────────────────
echo "Starting vLLM on Ascend NPU ..."
echo "  Model:       $MODEL"
echo "  Context len: $MAX_MODEL_LEN"
echo "  TP size:     $TP_SIZE"
echo "  NPUs:        $NPU_COUNT"
echo "  Port:        8000"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --allowed-local-media-path "$(pwd)/cache"
