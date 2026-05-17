#!/bin/bash
# Stop a vLLM server started by serve_llama.sh or serve_pangu.sh.
#
# Usage:
#   bash stop_server.sh [MODEL] [PORT]
#
# Arguments:
#   MODEL   llama | pangu  (default: llama)
#   PORT    Server port used when starting (default: 8000 for llama, 8001 for pangu)
#
# Examples:
#   bash stop_server.sh llama 8000
#   bash stop_server.sh pangu 8001

MODEL="${1:-llama}"
PORT="${2}"

if [ -z "$PORT" ]; then
    if [ "$MODEL" = "pangu" ]; then
        PORT=8001
    else
        PORT=8000
    fi
fi

PID_FILE="vllm_${MODEL}_${PORT}.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found: ${PID_FILE}"
    echo "Server may not be running or was started with a different port."
    exit 1
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping vLLM (${MODEL}) server PID ${PID} on port ${PORT}..."
    kill "$PID"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process did not stop gracefully, sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Server stopped."
else
    echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
