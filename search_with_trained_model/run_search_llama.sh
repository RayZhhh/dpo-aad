#!/bin/bash
# Run FunSearch with the DPO-trained Llama model.
#
# This script:
#   1. Optionally starts the vLLM server if not already running.
#   2. Launches multiple independent FunSearch runs in parallel (background).
#
# Usage:
#   bash run_search_llama.sh [LORA_PATH] [BASE_MODEL] [PORT] [NUM_RUNS]
#
# Arguments:
#   LORA_PATH   Path to DPO-trained adapter or merged model (default: /data/llama_dpo_adapter)
#   BASE_MODEL  Base model path/name (only needed in LoRA mode)
#   PORT        vLLM server port (default: 8000)
#   NUM_RUNS    Number of parallel FunSearch runs (default: 3)
#
# Prerequisites:
#   - vLLM server must be running, or set AUTO_START_SERVER=1 to start it here.
#   - Run from the dpo-aad/ root so that algodisco/task_examples/ paths resolve correctly.
#
# Example:
#   cd /path/to/dpo-aad
#   bash search_with_trained_model/run_search_llama.sh \
#       /data/llama_dpo_adapter meta-llama/Llama-3.1-8B-Instruct 8000 3

LORA_PATH="${1:-/data/llama_dpo_adapter}"
BASE_MODEL="${2:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${3:-8000}"
NUM_RUNS="${4:-3}"

# Set to 1 to automatically start the vLLM server if it is not already running
AUTO_START_SERVER="${AUTO_START_SERVER:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

export CUDA_VISIBLE_DEVICES=0

# ── Optionally start the vLLM server ──────────────────────────────────────────
if [ "$AUTO_START_SERVER" = "1" ]; then
    PID_FILE="${SCRIPT_DIR}/vllm_llama_${PORT}.pid"
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "vLLM (Llama) server already running on port ${PORT}."
    else
        echo "Starting vLLM (Llama) server..."
        bash "${SCRIPT_DIR}/serve_llama.sh" "${PORT}" "${LORA_PATH}" "${BASE_MODEL}"
        if [ $? -ne 0 ]; then
            echo "Failed to start vLLM server. Aborting."
            exit 1
        fi
    fi
else
    # Verify the server is reachable before launching search runs
    if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Error: vLLM server is not running on port ${PORT}."
        echo "Start it first:  bash ${SCRIPT_DIR}/serve_llama.sh ${PORT} ${LORA_PATH} ${BASE_MODEL}"
        echo "Or set AUTO_START_SERVER=1 to let this script start it automatically."
        exit 1
    fi
fi

# ── Launch FunSearch runs ──────────────────────────────────────────────────────
mkdir -p results/logs

echo "Launching ${NUM_RUNS} FunSearch run(s) with trained Llama (port ${PORT})..."
echo ""

for i in $(seq 1 "${NUM_RUNS}"); do
    # Each run gets its own config with an updated experiment name and log dir.
    RUN_CONFIG="${SCRIPT_DIR}/configs/funsearch_llama_trained_run${i}.yaml"

    # Generate a per-run config by patching the logdir and experiment_name fields.
    sed \
        -e "s|funsearch_llama_trained_run1|funsearch_llama_trained_run${i}|g" \
        -e "s|funsearch_llama_dpo_run1|funsearch_llama_dpo_run${i}|g" \
        "${CONFIG_DIR}/funsearch_llama_trained.yaml" > "${RUN_CONFIG}"

    LOG_OUT="results/logs/funsearch_llama_trained_run${i}.out"

    nohup python -m algodisco.methods.funsearch.main_funsearch \
        --config "${RUN_CONFIG}" \
        > "${LOG_OUT}" 2>&1 &

    echo "Run ${i}: PID $!  log: ${LOG_OUT}"
done

echo ""
echo "All ${NUM_RUNS} run(s) launched in background."
echo "Monitor progress:  tail -f results/logs/funsearch_llama_trained_run1.out"
