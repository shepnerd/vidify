#!/bin/bash
# Vidify Setup Script

set -euo pipefail

echo "Setting up Vidify..."

# ── System dependencies ──────────────────────────────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt &>/dev/null; then
        sudo apt update
        sudo apt install -y ffmpeg python3 python3-pip
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg python3
elif [[ "$OSTYPE" == "msys" ]]; then
    choco install ffmpeg python
fi

# ── yt-dlp ───────────────────────────────────────────────────────────────────
if ! command -v yt-dlp &>/dev/null; then
    echo "Installing yt-dlp..."
    pip install yt-dlp
fi

# ── Python dependencies ──────────────────────────────────────────────────────
pip install -r requirements.txt

# ── Cluster environment ──────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        echo ""
        echo "No .env file found. Copying from .env.example ..."
        cp .env.example .env
        echo "Please edit .env with your cluster settings:"
        echo "  - RL_CHARGED_GROUP  (your GPU quota group)"
        echo "  - RL_MOUNT          (GPFS mount points)"
        echo "  - CUDA_HOME         (shared CUDA toolkit path)"
    fi
else
    echo ".env already exists, skipping."
fi

# ── Verify key tools ────────────────────────────────────────────────────────
echo ""
echo "Checking tools..."
for cmd in python3 ffmpeg; do
    if command -v "$cmd" &>/dev/null; then
        echo "  ✓ $cmd"
    else
        echo "  ✗ $cmd (not found)"
    fi
done

# Check optional tools
for cmd in yt-dlp rlaunch; do
    if command -v "$cmd" &>/dev/null; then
        echo "  ✓ $cmd"
    else
        echo "  - $cmd (not found, optional)"
    fi
done

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit .env with your cluster settings (if on GPU cluster)"
echo "  2. Start model serving:  bash scripts/serving_qwen3_5.sh"
echo "  3. Run analysis:         python agent/main.py analyze local video.mp4 --mode detailed"
echo "  4. Run tests:            bash scripts/run_test_gpu.sh"
