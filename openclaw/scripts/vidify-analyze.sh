#!/usr/bin/env bash
# vidify-analyze.sh — Wrapper for Vidify video analysis
# Usage: vidify-analyze.sh <source_type> <uri> [mode]
#   source_type: youtube | url | local
#   uri:         YouTube URL, HTTP URL, or local file path
#   mode:        quick | detailed (default: detailed)

set -euo pipefail

SOURCE_TYPE="${1:?Usage: vidify-analyze.sh <source_type> <uri> [mode]}"
URI="${2:?Usage: vidify-analyze.sh <source_type> <uri> [mode]}"
MODE="${3:-detailed}"

if ! command -v vidify &>/dev/null; then
  echo "Error: vidify not found. Install with: pip install vidify" >&2
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "Error: ffmpeg not found. Install ffmpeg first." >&2
  exit 1
fi

exec vidify analyze "$SOURCE_TYPE" "$URI" --mode "$MODE"
