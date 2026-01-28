#!/bin/bash
# Verify vLLM deployments are working
# Run this after deploy_vllm.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== vLLM Verification ===${NC}"
echo ""

# Check GLM
echo -e "${YELLOW}Checking GLM-4.7-Flash (port 8000)...${NC}"
if curl -s --connect-timeout 5 http://localhost:8000/v1/models > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:8000/v1/models | jq -r '.data[].id' 2>/dev/null || echo "unknown")
    echo -e "  Status: ${GREEN}Running${NC}"
    echo "  Models: $MODELS"

    # Test chat completion
    echo "  Testing chat..."
    RESPONSE=$(curl -s http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "glm-4.7-flash", "messages": [{"role": "user", "content": "Say hello in one word"}], "max_tokens": 10}' \
        | jq -r '.choices[0].message.content' 2>/dev/null || echo "error")
    if [ "$RESPONSE" != "error" ] && [ "$RESPONSE" != "null" ]; then
        echo -e "  Chat: ${GREEN}OK${NC} (response: $RESPONSE)"
    else
        echo -e "  Chat: ${RED}FAILED${NC}"
    fi
else
    echo -e "  Status: ${RED}Not responding${NC}"
    echo "  Run: docker logs glm-4.7-flash"
fi
echo ""

# Check Qwen
echo -e "${YELLOW}Checking Qwen3-32B (port 8001)...${NC}"
if curl -s --connect-timeout 5 http://localhost:8001/v1/models > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:8001/v1/models | jq -r '.data[].id' 2>/dev/null || echo "unknown")
    echo -e "  Status: ${GREEN}Running${NC}"
    echo "  Models: $MODELS"

    # Test chat completion (Russian)
    echo "  Testing chat (Russian)..."
    RESPONSE=$(curl -s http://localhost:8001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "qwen3-32b", "messages": [{"role": "user", "content": "Скажи привет одним словом"}], "max_tokens": 10}' \
        | jq -r '.choices[0].message.content' 2>/dev/null || echo "error")
    if [ "$RESPONSE" != "error" ] && [ "$RESPONSE" != "null" ]; then
        echo -e "  Chat: ${GREEN}OK${NC} (response: $RESPONSE)"
    else
        echo -e "  Chat: ${RED}FAILED${NC}"
    fi
else
    echo -e "  Status: ${RED}Not responding${NC}"
    echo "  Run: docker logs qwen3-32b"
fi
echo ""

# Summary
echo -e "${GREEN}=== Summary ===${NC}"
docker ps --filter "name=glm-4.7-flash" --filter "name=qwen3-32b" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
