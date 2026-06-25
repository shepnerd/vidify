#!/usr/bin/env bash
# Serve Qwen3-MLA (Multi-head Latent Attention variant of Qwen3-VL) for Vidify.
#
# This model uses custom modeling code (trust_remote_code) that replaces GQA with
# MLA for more efficient KV-cache usage during inference.
#
# NOTE: vLLM cannot serve this model because its built-in Qwen3VL implementation
# expects standard GQA attention weights (q_norm/k_norm), but MLA uses different
# projections (kv_a_proj_with_mqa, kv_b_proj). We use a lightweight transformers-
# based OpenAI-compatible server instead.
#
# Usage:
#   bash scripts/serving_qwen3_mla.sh /path/to/model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${1:-}"

if [ -z "$MODEL" ]; then
    MODEL="${VIDIFY_MLA_MODEL:-models/qwen3-mla}"
    if [ ! -d "$MODEL" ]; then
        echo "ERROR: qwen3-mla checkpoint not found: $MODEL" >&2
        echo "Set VIDIFY_MLA_MODEL or pass the model path as the first argument." >&2
        exit 1
    fi
fi

# Port 8001 to coexist with qwen3.5 on 8000
PORT="${PORT:-8001}"

# Tensor parallel: set based on available GPUs
TP_SIZE="${TP_SIZE:-1}"

echo "Starting Qwen3-MLA server (transformers backend)..."
echo "  Model:       $MODEL"
echo "  TP size:      $TP_SIZE"
echo "  Port:         $PORT"

exec python "${SCRIPT_DIR}/serving_qwen3_mla_transformers.py" \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tp "$TP_SIZE"
