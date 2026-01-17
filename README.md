# Engram

Cognitive-inspired knowledge system with dual memory architecture.

> **engram** (noun): a hypothetical permanent change in the brain accounting for the existence of memory; a memory trace.

## Overview

Engram transforms documentation into an intelligent assistant that reasons like a brain — connecting concepts, learning from experience, and consolidating knowledge over time.

Unlike traditional RAG that retrieves document chunks, Engram uses a brain-inspired architecture:

- **Concept Network**: Atomic ideas connected by typed, weighted edges
- **Semantic Memory**: Facts and procedures linked to concepts
- **Episodic Memory**: Past reasoning traces with outcomes

## Features

- **Spreading Activation**: Brain-like associative retrieval through concept networks
- **Hybrid Search**: Combines graph traversal, vector similarity, and BM25
- **Learning from Feedback**: Positive feedback strengthens memories, negative triggers re-reasoning
- **Memory Consolidation**: Successful episodes crystallize into semantic memories
- **OpenAI-Compatible API**: Works with Open WebUI and other clients

## Tech Stack

| Component | Choice |
|-----------|--------|
| Python | 3.11+ |
| Package Manager | uv |
| Graph DB | Neo4j 5 (Docker) |
| API | FastAPI |
| Embeddings | sentence-transformers (local HuggingFace) |
| LLM | OpenAI-compatible endpoint (remote) |

## Quick Start

### Prerequisites

- Docker
- uv (Python package manager)

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd engram

# Start Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/engram_password neo4j:5

# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your LLM endpoint

# Run tests
uv run pytest

# Start API
uv run python -m engram.api.main
```

### Production Setup (RHEL 9+)

```bash
# Run setup script
chmod +x scripts/setup_rhel.sh
./scripts/setup_rhel.sh

# Configure environment
cp .env.production .env
# Edit .env with your settings

# Start API
source .venv/bin/activate
python -m engram.api.main
```

## Configuration

### Environment Variables

```bash
# LLM (remote OpenAI-compatible endpoint)
LLM_BASE_URL=http://your-host:8888/v1
LLM_MODEL=kimi-k2-thinking
LLM_API_KEY=your-key

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=engram2024

# Embeddings (local HuggingFace model)
# Dev: all-MiniLM-L6-v2 (384 dims)
# Prod: ai-sage/Giga-Embeddings-instruct (1024 dims)
EMBEDDING_MODEL=ai-sage/Giga-Embeddings-instruct
EMBEDDING_DIMENSIONS=1024
```

## API Endpoints

### Chat (OpenAI-Compatible)

```bash
POST /v1/chat/completions
```

```json
{
  "messages": [{"role": "user", "content": "What is Docker?"}],
  "model": "engram"
}
```

### Feedback

```bash
POST /v1/feedback
```

```json
{
  "episode_id": "...",
  "feedback": "positive"  // or "negative", "correction"
}
```

### Admin

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /admin/stats` | System statistics |
| `GET /admin/concepts` | List concepts |
| `GET /admin/memories` | List memories |
| `GET /admin/episodes` | List episodes |
| `POST /admin/calibrate` | Run memory calibration |
| `GET /admin/graph` | Interactive memory graph visualization |

## Memory Graph Visualization

Engram includes an interactive 2D graph visualization of the memory network, accessible at `/admin/graph`.

**Features:**
- **Node Types**: Concepts (blue), Semantic Memories (purple), Episodic Memories (orange)
- **Node Size**: Based on connection count — more connected nodes appear larger
- **Search**: Find and focus on specific nodes by name or content
- **Type Filtering**: Click legend buttons to highlight all nodes of a specific type
- **Node Selection**: Click any node to see details and highlight its connections
- **Progressive Labels**: Labels appear as you zoom in, larger nodes show labels first

**Controls:**
- **Click node**: Select and show info panel with full content
- **Click legend button**: Filter by type (multi-select supported)
- **Search box**: Type 2+ characters to search, click result to focus
- **Escape**: Clear all selections
- **Mouse wheel**: Zoom in/out
- **Drag**: Pan the view

## Interactive Chat CLI

```bash
uv run python scripts/chat.py
```

Commands:
- `/stats` - Show system statistics
- `/concepts [n]` - List top concepts
- `/memories [n]` - List memories
- `/episodes [n]` - List episodes
- `/feedback +` - Positive feedback
- `/feedback -` - Negative feedback
- `/help` - Show help
- `/quit` - Exit

## Document Ingestion

```bash
# Generate mock documents (for testing)
uv run python scripts/generate_mock_docs.py

# Ingest documents
uv run python scripts/run_ingestion.py
```

## Project Structure

```
engram/
├── src/engram/
│   ├── models/          # Concept, SemanticMemory, EpisodicMemory
│   ├── ingestion/       # Document parsing, concept/memory extraction
│   ├── storage/         # Neo4j client and schema
│   ├── retrieval/       # Embeddings, spreading activation, hybrid search
│   ├── reasoning/       # Response synthesis, episode management
│   ├── learning/        # Feedback, consolidation, memory strength
│   └── api/             # FastAPI routes
├── tests/
│   ├── unit/            # 169 unit tests
│   ├── integration/
│   └── fixtures/mock_docs/  # 32 sample documents
├── scripts/
│   ├── setup_rhel.sh    # Production setup
│   ├── chat.py          # Interactive CLI
│   ├── generate_mock_docs.py
│   └── run_ingestion.py
└── CLAUDE.md            # AI assistant instructions
```

## Development

```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=engram

# Type checking
uv run mypy src/engram

# Linting
uv run ruff check src/engram
```

## License

MIT
