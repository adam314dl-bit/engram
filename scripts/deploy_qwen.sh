#!/bin/bash
# Deploy Qwen3-32B on vLLM stable
# Port: 8001

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER_NAME="qwen3-32b"
PORT=8001
GPUS="${1:-0,1,2,3}"  # Default: all 4 GPUs, or pass as argument

# Count GPUs
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')

echo -e "${GREEN}=== Deploying Qwen3-32B ===${NC}"
echo "  GPUs: $GPUS (tensor-parallel: $GPU_COUNT)"
echo "  Port: $PORT"
echo ""

# Pull stable image
echo -e "${YELLOW}Pulling vllm/vllm-openai:v0.8.5...${NC}"
docker pull vllm/vllm-openai:v0.8.5

# Stop existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Deploy
echo -e "${YELLOW}Starting container...${NC}"
docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus "\"device=$GPUS\"" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p ${PORT}:8000 \
    --ipc=host \
    --restart unless-stopped \
    vllm/vllm-openai:v0.8.5 \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size "$GPU_COUNT" \
    --reasoning-parser qwen3 \
    --enable-reasoning \
    --served-model-name qwen3-32b \
    --max-model-len 32768 \
    --trust-remote-code \
    --dtype bfloat16

echo ""
echo -e "${GREEN}=== Deployed ===${NC}"
echo ""
echo "# Check logs:"
echo "docker logs -f $CONTAINER_NAME"
echo ""
echo "# Test (Russian):"
echo 'curl http://localhost:8001/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '\''{"model": "qwen3-32b", "messages": [{"role": "user", "content": "Привет! Как дела?"}], "max_tokens": 500}'\'''
echo ""
echo "# Engram .env:"
echo "LLM_BASE_URL=http://localhost:8001/v1"
echo "LLM_MODEL=qwen3-32b"
