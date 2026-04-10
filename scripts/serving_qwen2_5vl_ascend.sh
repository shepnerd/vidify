#!/usr/bin/env bash
# Serve Qwen2.5-VL-7B-Instruct on Ascend 910C NPU via vLLM + vllm_ascend.
#
# Why not Qwen3.5-9B?
#   Qwen3.5's head_dim=256 is unsupported by the NPU fused attention kernel
#   (npu_fused_infer_attention_score only supports 64/128/192).
#   Qwen2.5-VL-7B-Instruct has head_dim=128 and works correctly.
#
# Prerequisites:
#   - Ascend 910C node with CANN toolkit installed
#   - vLLM >= 0.18.0 with vllm_ascend plugin (A3-specific build)
#   - torch_npu matching the CANN/PyTorch versions
#   - Image: registry2.d.pjlab.org.cn/ccr-hw/910c:vllm-ascend-0.18.0rc1-a3-0409
#
# Usage (inside a D-cluster pod):
#   bash scripts/serving_qwen2_5vl_ascend.sh                                    # auto-detect model
#   bash scripts/serving_qwen2_5vl_ascend.sh /data/models/Qwen2.5-VL-7B-Instruct  # explicit path
#   TP_SIZE=2 bash scripts/serving_qwen2_5vl_ascend.sh                          # 2 NPUs

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
    for candidate in \
        /data/models/Qwen2.5-VL-7B-Instruct \
        /data/sfteval/models/Qwen2.5-VL-7B-Instruct \
        "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"; do
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
        echo "No local model found. Download via hf-mirror:"
        echo "  HF_ENDPOINT=https://hf-mirror.com python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/data/models/Qwen2.5-VL-7B-Instruct')\""
        echo ""
        echo "Or set HF_ENDPOINT and let vLLM download:"
        export HF_ENDPOINT=https://hf-mirror.com
        MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
        echo "Attempting download from hf-mirror: $MODEL"
    fi
fi

# ── Parameters ──────────────────────────────────────────────────────────────
# Qwen2.5-VL-7B has 28 attention heads → TP must divide 28 (1,2,4,7,14)
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

# Validate TP divides num_attention_heads (28 for Qwen2.5-VL-7B)
if (( 28 % TP_SIZE != 0 )); then
    echo "ERROR: TP_SIZE=${TP_SIZE} does not divide num_attention_heads=28. Use 1, 2, 4, 7, or 14." >&2
    exit 1
fi

# ── Launch ──────────────────────────────────────────────────────────────────
echo "Starting vLLM on Ascend NPU ..."
echo "  Model:       $MODEL"
echo "  Context len: $MAX_MODEL_LEN"
echo "  TP size:     $TP_SIZE"
echo "  NPUs:        $NPU_COUNT"
echo "  Port:        8000"
echo "  Flags:       --enforce-eager (NPU graph capture causes OOM)"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --allowed-local-media-path "$(pwd)/cache"
