#!/usr/bin/env bash
# Serve Qwen3.5-9B for VidCopilot.
#
# Requires vLLM nightly (>= 0.17) for Qwen3.5 support:
#   uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
#
# Qwen3.5 is a unified vision-language model with:
#   - Gated DeltaNet hybrid architecture (linear + full attention)
#   - 262K native context length (configurable, min recommended 128K)
#   - Thinking mode by default (<think>...</think>)
#   - Video frame sampling via mm_processor_kwargs
#
# Usage:
#   bash scripts/serving_qwen3_5.sh                   # Auto-detect local model
#   bash scripts/serving_qwen3_5.sh /path/to/model    # Explicit model path
#   bash scripts/serving_qwen3_5.sh Qwen/Qwen3.5-9B   # HuggingFace model ID

set -euo pipefail

MODEL="${1:-}"

# Auto-detect local cached model if no argument given
if [ -z "$MODEL" ]; then
    LOCAL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"
    if [ -d "$LOCAL_CACHE/snapshots" ]; then
        # Use the latest snapshot
        MODEL=$(ls -1d "$LOCAL_CACHE/snapshots"/*/ 2>/dev/null | head -1 | sed 's:/$::')
        echo "Using local model: $MODEL"
    else
        MODEL="Qwen/Qwen3.5-9B"
        echo "No local model found, downloading from HuggingFace: $MODEL"
    fi
fi

# Context length: 262144 is native, but requires significant VRAM.
# Reduce to 65536 or 32768 if you hit OOM.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# Tensor parallel: set based on available GPUs
TP_SIZE="${TP_SIZE:-1}"

echo "Starting vLLM server for Qwen3.5..."
echo "  Model:       $MODEL"
echo "  Context len:  $MAX_MODEL_LEN"
echo "  TP size:      $TP_SIZE"
echo "  Port:         8000"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --allowed-local-media-path "$(pwd)/cache"
