#!/usr/bin/env bash
# vidify-ask.sh — Ask a question about a video using Vidify
# Usage: vidify-ask.sh <source_type> <uri> <question>
#   source_type: youtube | url | local
#   uri:         YouTube URL, HTTP URL, or local file path
#   question:    The question to ask about the video

set -euo pipefail

SOURCE_TYPE="${1:?Usage: vidify-ask.sh <source_type> <uri> <question>}"
URI="${2:?Usage: vidify-ask.sh <source_type> <uri> <question>}"
QUESTION="${3:?Usage: vidify-ask.sh <source_type> <uri> <question>}"

if ! command -v vidify &>/dev/null; then
  echo "Error: vidify not found. Install with: pip install vidify" >&2
  exit 1
fi

exec vidify analyze "$SOURCE_TYPE" "$URI" --mode ask --question "$QUESTION"
