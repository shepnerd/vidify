#!/bin/bash
# Run all Vidify skill tests on a local video.
# Usage:
#   bash scripts/test_all.sh                              # default: taste_in_china video, 4 GPUs
#   bash scripts/test_all.sh --api-base http://HOST:8000  # use existing serving
#   bash scripts/test_all.sh --gpu 2                      # override GPU count

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO="${SCRIPT_DIR}/../media/taste_in_china_s1e1.mp4"

python "${SCRIPT_DIR}/test_all.py" --video-path "$VIDEO" "$@"
