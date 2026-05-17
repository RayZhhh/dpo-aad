#!/bin/bash
# Start vLLM server for a trained Llama model (with LoRA adapter or merged weights).
#
# Usage:
#   bash serve_llama.sh [PORT] [LORA_PATH] [BASE_MODEL]
#
# Arguments:
#   PORT        Server port (default: 8000)
#   LORA_PATH   Path to the DPO-trained output directory.
#               - If it contains adapter_config.json  →  LoRA mode (base model stays separate)
#               - If it contains config.json only     →  merged model (served directly)
#   BASE_MODEL  Base model path/name (only needed in LoRA mode, default below)
#
# Examples:
#   bash serve_llama.sh 8000 /data/llama_dpo_adapter
#   bash serve_llama.sh 8000 /data/llama_merged_model

export CUDA_VISIBLE_DEVICES=0

PORT="${1:-8000}"
LORA_PATH="${2:-/data/llama_dpo_adapter}"
BASE_MODEL="${3:-meta-llama/Llama-3.1-8B-Instruct}"

ADAPTER_NAME="llama-dpo"
LOG_FILE="vllm_llama_${PORT}.log"
PID_FILE="vllm_llama_${PORT}.pid"

# Prevent double-start
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "vLLM (Llama) already running on port $PORT (PID $(cat "$PID_FILE"))"
    exit 1
fi

# Detect whether the output dir is a LoRA adapter or a merged model
if [ -f "${LORA_PATH}/adapter_config.json" ]; then
    # --- LoRA adapter mode ---
    echo "Detected LoRA adapter at ${LORA_PATH}"
    echo "Base model : ${BASE_MODEL}"
    echo "Adapter    : ${ADAPTER_NAME} -> ${LORA_PATH}"
    echo "Starting vLLM on port ${PORT}, logs: ${LOG_FILE}"

    nohup vllm serve "${BASE_MODEL}" \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 16384 \
        --dtype bfloat16 \
        --enable-lora \
        --lora-modules "${ADAPTER_NAME}=${LORA_PATH}" \
        --max-lora-rank 64 \
        > "${LOG_FILE}" 2>&1 &

elif [ -f "${LORA_PATH}/config.json" ]; then
    # --- Merged model mode ---
    echo "Detected merged model at ${LORA_PATH}"
    echo "Starting vLLM on port ${PORT}, logs: ${LOG_FILE}"
    ADAPTER_NAME="${LORA_PATH}"   # the served model ID is the path itself

    nohup vllm serve "${LORA_PATH}" \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 16384 \
        --dtype bfloat16 \
        > "${LOG_FILE}" 2>&1 &

else
    echo "Error: ${LORA_PATH} contains neither adapter_config.json nor config.json"
    echo "Make sure LORA_PATH points to the DPO training output directory."
    exit 1
fi

echo $! > "${PID_FILE}"
echo "PID $! saved to ${PID_FILE}"
echo ""
echo "Model name to use in search configs: ${ADAPTER_NAME}"
echo ""

echo -n "Waiting for server to be ready"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "Server ready at http://localhost:${PORT}"
        echo "OpenAI-compatible endpoint: http://localhost:${PORT}/v1"
        exit 0
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "Server did not become ready within 5 minutes. Check ${LOG_FILE}"
exit 1
