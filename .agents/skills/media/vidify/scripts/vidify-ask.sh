#!/usr/bin/env bash
set -euo pipefail

SOURCE_TYPE="${1:?Usage: vidify-ask.sh <source_type> <uri> <question>}"
URI="${2:?Usage: vidify-ask.sh <source_type> <uri> <question>}"
QUESTION="${3:?Usage: vidify-ask.sh <source_type> <uri> <question>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v vidify >/dev/null 2>&1; then
  exec vidify analyze "$SOURCE_TYPE" "$URI" --mode ask --question "$QUESTION"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m agent.main analyze "$SOURCE_TYPE" "$URI" --mode ask --question "$QUESTION"

