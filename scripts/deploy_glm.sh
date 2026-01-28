#!/bin/bash
# Deploy GLM-4.7-Flash on vLLM nightly
# Port: 8000

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER_NAME="glm-4.7-flash"
PORT=8000
GPUS="${1:-0,1,2,3}"  # Default: all 4 GPUs, or pass as argument

# Count GPUs
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')

echo -e "${GREEN}=== Deploying GLM-4.7-Flash ===${NC}"
echo "  GPUs: $GPUS (tensor-parallel: $GPU_COUNT)"
echo "  Port: $PORT"
echo ""

# Pull nightly image
echo -e "${YELLOW}Pulling vllm/vllm-openai:nightly...${NC}"
docker pull vllm/vllm-openai:nightly

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
    vllm/vllm-openai:nightly \
    --model zai-org/GLM-4.7-Flash \
    --tensor-parallel-size "$GPU_COUNT" \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash \
    --max-model-len 32768 \
    --trust-remote-code \
    --dtype bfloat16

echo ""
echo -e "${GREEN}=== Deployed ===${NC}"
echo ""
echo "# Check logs:"
echo "docker logs -f $CONTAINER_NAME"
echo ""
echo "# Test:"
echo 'curl http://localhost:8000/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '\''{"model": "glm-4.7-flash", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'\'''
echo ""
echo "# Engram .env:"
echo "LLM_BASE_URL=http://localhost:8000/v1"
echo "LLM_MODEL=glm-4.7-flash"
