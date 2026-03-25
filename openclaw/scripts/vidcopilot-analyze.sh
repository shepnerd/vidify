#!/usr/bin/env bash
# vidcopilot-analyze.sh — Wrapper for VidCopilot video analysis
# Usage: vidcopilot-analyze.sh <source_type> <uri> [mode]
#   source_type: youtube | url | local
#   uri:         YouTube URL, HTTP URL, or local file path
#   mode:        quick | detailed (default: detailed)

set -euo pipefail

SOURCE_TYPE="${1:?Usage: vidcopilot-analyze.sh <source_type> <uri> [mode]}"
URI="${2:?Usage: vidcopilot-analyze.sh <source_type> <uri> [mode]}"
MODE="${3:-detailed}"

if ! command -v vidcopilot &>/dev/null; then
  echo "Error: vidcopilot not found. Install with: pip install vidcopilot" >&2
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "Error: ffmpeg not found. Install ffmpeg first." >&2
  exit 1
fi

exec vidcopilot analyze "$SOURCE_TYPE" "$URI" --mode "$MODE"
