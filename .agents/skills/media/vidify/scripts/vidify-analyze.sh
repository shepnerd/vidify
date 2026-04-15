#!/usr/bin/env bash
set -euo pipefail

SOURCE_TYPE="${1:?Usage: vidify-analyze.sh <source_type> <uri> [mode]}"
URI="${2:?Usage: vidify-analyze.sh <source_type> <uri> [mode]}"
MODE="${3:-detailed}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found on PATH." >&2
  exit 1
fi

if command -v vidify >/dev/null 2>&1; then
  exec vidify analyze "$SOURCE_TYPE" "$URI" --mode "$MODE"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m agent.main analyze "$SOURCE_TYPE" "$URI" --mode "$MODE"

