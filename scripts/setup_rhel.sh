#!/bin/bash
#
# Minimal Engram Setup for RHEL 9+
#
# Usage:
#   chmod +x scripts/setup_rhel.sh
#   ./scripts/setup_rhel.sh
#

set -e

echo "=== Engram Minimal Setup ==="

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo dnf install -y dnf-plugins-core
    sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
    sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo systemctl enable --now docker
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes."
fi

# Create data directory
mkdir -p ./data/neo4j

# Start Neo4j container
echo "Starting Neo4j..."
docker run -d \
    --name neo4j \
    --restart unless-stopped \
    -p 7474:7474 \
    -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/engram2024 \
    -v $(pwd)/data/neo4j:/data \
    neo4j:5

# Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install Python 3.11 and sync dependencies
echo "Setting up Python environment..."
~/.local/bin/uv sync --python 3.11

echo ""
echo "=== Done ==="
echo ""
echo "Neo4j: http://localhost:7474 (neo4j/engram2024)"
echo "Data: ./data/neo4j"
echo ""
echo "Next steps:"
echo "  1. cp .env.production .env"
echo "  2. Edit .env:"
echo "     - LLM_BASE_URL=http://your-llm-host:8888/v1"
echo "     - LLM_MODEL=kimi-k2"
echo "     - LLM_API_KEY=your-key"
echo "  3. source .venv/bin/activate"
echo "  4. python -m engram.api.main"
echo ""
echo "Note: Embedding model (ai-sage/Giga-Embeddings-instruct) will be"
echo "downloaded automatically on first run from HuggingFace."
