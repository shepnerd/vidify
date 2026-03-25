#!/usr/bin/env bash
# vidcopilot-ask.sh — Ask a question about a video using VidCopilot
# Usage: vidcopilot-ask.sh <source_type> <uri> <question>
#   source_type: youtube | url | local
#   uri:         YouTube URL, HTTP URL, or local file path
#   question:    The question to ask about the video

set -euo pipefail

SOURCE_TYPE="${1:?Usage: vidcopilot-ask.sh <source_type> <uri> <question>}"
URI="${2:?Usage: vidcopilot-ask.sh <source_type> <uri> <question>}"
QUESTION="${3:?Usage: vidcopilot-ask.sh <source_type> <uri> <question>}"

if ! command -v vidcopilot &>/dev/null; then
  echo "Error: vidcopilot not found. Install with: pip install vidcopilot" >&2
  exit 1
fi

exec vidcopilot analyze "$SOURCE_TYPE" "$URI" --mode ask --question "$QUESTION"
