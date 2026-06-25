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

# ── Runtime environment ──────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        echo ""
        echo "No .env file found. Copying from .env.example ..."
        cp .env.example .env
        echo "Please edit .env with your local endpoint and model settings."
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
for cmd in yt-dlp vllm; do
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
echo "  1. Edit .env with your local endpoint/model settings"
echo "  2. Start model serving:  bash scripts/serving_qwen3_5.sh"
echo "  3. Run analysis:         python agent/main.py analyze local video.mp4 --mode detailed"
echo "  4. Run tests:            bash scripts/run_test_gpu.sh --video media/my_video.mp4"
