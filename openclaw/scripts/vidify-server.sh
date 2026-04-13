#!/usr/bin/env bash
# vidify-server.sh — Start/stop the Vidify REST API server
# Usage: vidify-server.sh start|stop|status
#   start:  Launch the API server on port 9000 (background)
#   stop:   Stop the running server
#   status: Check if the server is running

set -euo pipefail

PIDFILE="${TMPDIR:-/tmp}/vidify-server.pid"
PORT="${VIDIFY_PORT:-9000}"

case "${1:?Usage: vidify-server.sh start|stop|status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Vidify server already running (PID $(cat "$PIDFILE")) on port $PORT"
      exit 0
    fi
    echo "Starting Vidify server on port $PORT..."
    nohup uvicorn server.app:app --host 0.0.0.0 --port "$PORT" > /tmp/vidify-server.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "Server started (PID $!) — logs at /tmp/vidify-server.log"
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
