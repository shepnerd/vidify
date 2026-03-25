#!/usr/bin/env bash
# vidcopilot-server.sh — Start/stop the VidCopilot REST API server
# Usage: vidcopilot-server.sh start|stop|status
#   start:  Launch the API server on port 9000 (background)
#   stop:   Stop the running server
#   status: Check if the server is running

set -euo pipefail

PIDFILE="${TMPDIR:-/tmp}/vidcopilot-server.pid"
PORT="${VIDCOPILOT_PORT:-9000}"

case "${1:?Usage: vidcopilot-server.sh start|stop|status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "VidCopilot server already running (PID $(cat "$PIDFILE")) on port $PORT"
      exit 0
    fi
    echo "Starting VidCopilot server on port $PORT..."
    nohup uvicorn server.app:app --host 0.0.0.0 --port "$PORT" > /tmp/vidcopilot-server.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "Server started (PID $!) — logs at /tmp/vidcopilot-server.log"
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
    echo "Usage: vidcopilot-server.sh start|stop|status" >&2
    exit 1
    ;;
esac
