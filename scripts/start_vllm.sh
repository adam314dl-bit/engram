#!/bin/bash
# Engram vLLM server for Kimi K2 Thinking
# CRITICAL: --reasoning-parser separates thinking from content

MODEL="${VLLM_MODEL:-moonshotai/Kimi-K2-Instruct-Thinking}"
PORT="${VLLM_PORT:-8888}"
TP_SIZE="${VLLM_TP_SIZE:-8}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-120000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"

echo "Starting vLLM server for $MODEL"
echo "Port: $PORT, TP: $TP_SIZE, Max length: $MAX_MODEL_LEN"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --enable-auto-tool-choice \
    --max-num-batched-tokens 32768 \
    "$@"
