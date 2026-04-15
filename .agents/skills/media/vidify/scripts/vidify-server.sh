#!/usr/bin/env bash
set -euo pipefail

PIDFILE="${TMPDIR:-/tmp}/vidify-hermes-server.pid"
PORT="${VIDIFY_PORT:-9000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${1:?Usage: vidify-server.sh start|stop|status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Vidify server already running (PID $(cat "$PIDFILE")) on port $PORT"
      exit 0
    fi
    cd "$REPO_ROOT"
    echo "Starting Vidify server on port $PORT..."
    nohup "$PYTHON_BIN" -m uvicorn server.app:app --host 0.0.0.0 --port "$PORT" \
      > /tmp/vidify-hermes-server.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "Server started (PID $!)"
    ;;
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill "$(cat "$PIDFILE")"
      rm -f "$PIDFILE"
      echo "Server stopped."
    else
      echo "No running server found."
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Running (PID $(cat "$PIDFILE")) on port $PORT"
    else
      echo "Not running."
      rm -f "$PIDFILE" 2>/dev/null
    fi
    ;;
  *)
    echo "Usage: vidify-server.sh start|stop|status" >&2
    exit 1
    ;;
esac

