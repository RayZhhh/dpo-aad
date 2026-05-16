#!/bin/bash

# Usage: bash start_vllm_server.sh [port]

export CUDA_VISIBLE_DEVICES=0

PORT=${1:-8000}
MODEL="meta-llama/Llama-3.1-8B-Instruct"
LOG_FILE="vllm_server_${PORT}.log"
PID_FILE="vllm_server_${PORT}.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "vLLM server already running on port $PORT (PID $(cat "$PID_FILE"))"
    exit 1
fi

echo "Starting vLLM server on port $PORT, logs: $LOG_FILE"

nohup vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --enforce-eager \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID $! saved to $PID_FILE"

echo -n "Waiting for server to be ready"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "Server ready at http://localhost:${PORT}"
        exit 0
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "Server did not become ready within 5 minutes. Check $LOG_FILE"
exit 1
