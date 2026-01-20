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
# Prod: ai-sage/Giga-Embeddings-instruct (2048 dims)
EMBEDDING_MODEL=ai-sage/Giga-Embeddings-instruct
EMBEDDING_DIMENSIONS=2048
```

## Running the Server

```bash
# Start the API server (default: http://localhost:8000)
uv run python -m engram.api.main

# Or with custom host/port
uv run python -m engram.api.main --host 0.0.0.0 --port 8080
```

The server exposes an OpenAI-compatible API at `/v1/chat/completions`.

## Open WebUI Integration

Engram works as a custom OpenAI-compatible endpoint in [Open WebUI](https://github.com/open-webui/open-webui).

**Setup:**

1. Start the Engram server:
   ```bash
   uv run python -m engram.api.main
   ```

2. In Open WebUI, go to **Admin Panel** → **Settings** → **Connections**

3. Under **OpenAI API**, click **+** to add a new connection:
   - **URL**: `http://localhost:8000/v1` (or your server address)
   - **API Key**: `engram` (any value works, not validated)

4. Click **Save**

5. In the chat interface, select **engram** from the model dropdown

Now you can chat with your knowledge base through Open WebUI.

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
| `POST /admin/reset?confirm=true` | Reset database (delete all data) |
| `GET /admin/graph` | Interactive memory graph visualization |

#### Reset Database

To clear all data and re-ingest (useful when changing embedding models):

```bash
# Reset database (WARNING: deletes ALL data)
curl -X POST "http://localhost:8000/admin/reset?confirm=true"

# Re-run ingestion
uv run python scripts/run_ingestion.py
```

## Memory Graph Visualization

Engram includes an interactive WebGL graph visualization of the memory network, accessible at `/admin/graph`. Optimized for large graphs (30k+ nodes) with pre-computed layouts.

**Architecture:**
- **WebGL Rendering**: Hardware-accelerated graphics for smooth performance
- **Pre-computed Layout**: Server-side layout using igraph/cuGraph, stored in Neo4j
- **Viewport Culling**: Only loads nodes visible in current view
- **Louvain Clustering**: Community detection for visual grouping

**Node Types:**
- **Concepts** (teal `#5eead4`): Atomic ideas from ingested documents
- **Semantic Memories** (purple `#a78bfa`): Facts and procedures linked to concepts
- **Episodic Memories** (pink `#f472b6`): Past reasoning traces with outcomes

**Features:**
- **Integrated Chat**: Chat panel with memory activation visualization
- **Debug Mode**: See retrieved memories with scores, sources, and force include/exclude
- **Live Activation**: See which nodes are used when answering questions (cyan highlight)
- **Cluster Coloring**: Toggle to color nodes by community
- **Search**: Find and focus on specific nodes by name or content
- **Type Filtering**: Click legend items to filter by node type
- **Neighbor Highlighting**: Click node to see its connections highlighted
- **Connected Nodes Panel**: View and navigate to connected nodes

**Controls:**
- **💬 Button**: Toggle chat panel
- **🔍 Debug**: Toggle debug panel showing retrieval details
- **Click node**: Select and show info panel with connections
- **Click legend item**: Filter by type (click again to clear)
- **"Clusters" button**: Toggle cluster-based coloring
- **"Show Activation" button**: Re-highlight last chat response nodes
- **Search box**: Type 2+ characters to search, click result to focus
- **Escape**: Clear all selections, filters, and highlights
- **Mouse wheel**: Zoom in/out
- **Drag**: Pan the view

**Debug Panel:**
- Shows retrieved memories with scores (0-1) and sources (V=Vector, B=BM25, G=Graph, F=Forced)
- Shows activated concepts with activation levels
- **+ button**: Force include node in next query
- **− button**: Force exclude node from next query
- Click node name to highlight in graph

**Chat Integration:**
- Ask questions directly in the graph interface
- Activated concepts and memories highlighted in cyan
- Click "Activated: X concepts, Y memories" to re-highlight
- View pans to center of activated nodes

**Setup for Large Graphs:**
```bash
# Compute layout (required after ingestion)
uv run python scripts/compute_layout.py

# For GPU acceleration (100-1000x faster):
uv sync --extra gpu
```

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

## Roadmap

### Completed

- [x] **Pre-computed layout** — Server-side layout using igraph (CPU) or cuGraph (GPU)
- [x] **Viewport culling** — Client requests nodes by viewport bounds
- [x] **WebGL rendering** — Hardware-accelerated for 30k+ nodes
- [x] **Louvain clustering** — Community detection stored in Neo4j
- [x] **Integrated chat** — Chat with memory activation visualization
- [x] **Debug mode** — View retrieval scores, sources, force include/exclude nodes
- [x] **Level-based lazy loading** — Hierarchical drill-down (L0→L1→L2→nodes)

### Planned

- [ ] **Spatial indexing** — Neo4j point indexes for faster viewport queries
- [ ] **Feedback in chat** — 👍/👎 buttons to strengthen/weaken memories
- [ ] **Path finder** — Find shortest path between two nodes
- [ ] **Export subgraph** — Export selected nodes to markdown

## License

MIT
