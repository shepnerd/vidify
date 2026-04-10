#!/usr/bin/env bash
# Serve Qwen3.5-9B on Ascend 910C NPU via vLLM 0.18 + vllm_ascend.
#
# Qwen3.5-9B uses a hybrid GDN+Attention architecture:
#   - 3/4 layers: Gated DeltaNet (linear attention, uses fused_recurrent_gated_delta_rule)
#   - 1/4 layers: Standard attention (head_dim=256, requires --enforce-eager on NPU)
#   - num_attention_heads=16, num_kv_heads=4 → valid TP: 1, 2, 4
#
# Image: registry2.d.pjlab.org.cn/ccr-hw/910c:vllm-ascend-0.18.0rc1-a3-0409
#
# Prerequisites:
#   - Ascend 910C node with CANN toolkit installed
#   - vLLM >= 0.18.0 with vllm_ascend plugin
#   - torch_npu matching the CANN/PyTorch versions
#   - transformers >= 4.51 (for Qwen3.5 model type support)
#
# Usage (inside a D-cluster pod):
#   bash scripts/serving_qwen3_5_ascend.sh                          # auto-detect model
#   bash scripts/serving_qwen3_5_ascend.sh /data/models/Qwen3.5-9B  # explicit path
#   TP_SIZE=4 bash scripts/serving_qwen3_5_ascend.sh                # 4 NPUs

set -euo pipefail

# ── Ascend environment ──────────────────────────────────────────────────────
# Temporarily disable nounset — Ascend set_env.sh references ZSH_VERSION which is unbound
set +u
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || true
fi
set -u
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
# Qwen3.5-9B: num_attention_heads=16, num_kv_heads=4 → TP must divide both → max TP=4
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TP_SIZE="${TP_SIZE:-4}"

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

# Validate TP divides num_attention_heads (16) and num_kv_heads (4)
if (( 16 % TP_SIZE != 0 )) || (( 4 % TP_SIZE != 0 )); then
    echo "ERROR: TP_SIZE=${TP_SIZE} must divide both num_attention_heads=16 and num_kv_heads=4. Use 1, 2, or 4." >&2
    exit 1
fi

# ── Launch ──────────────────────────────────────────────────────────────────
echo "Starting vLLM on Ascend NPU ..."
echo "  Model:       $MODEL"
echo "  Context len: $MAX_MODEL_LEN"
echo "  TP size:     $TP_SIZE"
echo "  NPUs:        $NPU_COUNT"
echo "  Port:        8000"
echo "  Flags:       --enforce-eager (NPU graph capture unsupported for head_dim=256)"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --allowed-local-media-path "$(pwd)/cache"
