#!/usr/bin/env bash
# chat.sh - minimal client for vLLM OpenAI-compatible chat/completions
# Usage:
#   ./chat.sh -i "介绍一下你自己"
#   ./chat.sh -i "hello" -u http://127.0.0.1:8000 -m /path/to/model

set -euo pipefail

URL="http://localhost:8000"
MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
INPUT=""

usage() {
  cat <<EOF
Usage: $0 -i "your prompt" [-u base_url] [-m model_path]

  -i  Prompt text (required)
  -u  Base URL, default: $URL
  -m  Model identifier/path, default: $MODEL

Example:
  $0 -i "介绍一下你自己"
EOF
}

while getopts ":i:u:m:h" opt; do
  case "$opt" in
    i) INPUT="$OPTARG" ;;
    u) URL="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${INPUT}" ]]; then
  echo "Error: -i is required." >&2
  usage
  exit 2
fi

# Escape backslashes and double-quotes for JSON string
json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

PROMPT_ESCAPED="$(json_escape "$INPUT")"

read -r -d '' PAYLOAD <<EOF || true
{
  "model": "$(json_escape "$MODEL")",
  "messages": [{"role":"user","content":"$PROMPT_ESCAPED"}]
}
EOF

curl -sS "${URL%/}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
| python -c 'import sys, json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])'