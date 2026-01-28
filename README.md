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

**Core Retrieval:**
- **BM25+Graph Mode**: Default mode - no embedding model required, uses BM25 + graph traversal
- **Hybrid Mode**: Optional - adds BGE-M3 vector similarity search (set `RETRIEVAL_MODE=hybrid`)
- **Spreading Activation**: Brain-like associative retrieval through concept networks
- **Path-Based Retrieval**: Finds memories bridging multiple query concepts (shared memories, bridge concepts)
- **4-Way RRF Fusion**: Weighted fusion with Vector (0.25), BM25 (0.25), Path (0.30), Graph (0.20)
- **BGE-reranker-v2-m3**: Multilingual cross-encoder for improved retrieval precision (replaces Jina)
- **FAISS Vector Index**: Fast similarity search with flat (exact) or IVF (approximate) indexes
- **MMR Diversity**: Maximal Marginal Relevance prevents redundant results (hybrid mode only)
- **Dynamic top_k**: Query complexity classification adjusts retrieval depth
- **Retrieval Debugging**: Trace memories through pipeline stages with `debug_retrieval.py`
- **Evaluation Framework**: Recall@K, MRR, NDCG metrics with golden query testing

**Dual-Content Memory:**
- **Separate Search vs Display**: `search_content` for vector/BM25 search, `content` for LLM generation
- **Search-Optimized**: Summary + keywords optimized for retrieval
- **Near-Verbatim Facts**: Actual data preserved for accurate LLM responses
- **Memory Embeddings**: BGE-M3 embeds memories (not raw chunks) for semantic search

**NLP & Processing:**
- **Russian NLP**: PyMorphy3 lemmatization and stopword removal for Russian content
- **Transliteration**: Handles mixed Cyrillic/Latin content with query expansion
- **Person Extraction**: Natasha NER with PyMorphy3 validation extracts people, roles, and team affiliations
- **Table Extraction**: Multi-vector strategy with searchable summaries and raw tables
- **List Extraction**: Structure-aware extraction of definitions, procedures, bullets
- **Quality Filtering**: Chunk scoring and source weighting for cleaner retrieval

**Memory & Learning:**
- **ACT-R Memory Model**: Cognitive-inspired forgetting with base-level activation
- **Contradiction Detection**: LLM-based detection and auto-resolution
- **Learning from Feedback**: Positive feedback strengthens memories, negative triggers re-reasoning
- **Memory Consolidation**: Successful episodes crystallize into semantic memories
- **Table Enrichment**: LLM-based table enrichment with multi-vector retrieval
- **Fast Ingestion**: Batch Neo4j writes, embeddings generated separately via `build_vector_index.py`

**Integration:**
- **Source Attribution**: Shows document sources (title + URL) in responses
- **Confluence Integration**: Extracts metadata from Confluence exports, strips headers
- **Time-Aware Responses**: Current date context enables relevant temporal reasoning
- **OpenAI-Compatible API**: Works with Open WebUI and other clients

## Tech Stack

| Component | Choice |
|-----------|--------|
| Python | 3.11+ |
| Package Manager | uv |
| Graph DB | Neo4j 5 (Docker) |
| API | FastAPI |
| Embeddings | BGE-M3 via FlagEmbedding (1024-dim, multilingual) |
| Vector Index | FAISS (flat or IVF) |
| Reranker | BGE-reranker-v2-m3 (default) or Jina Reranker v3 |
| Russian NLP | PyMorphy3, Natasha NER |
| Transliteration | cyrtranslit |
| LLM | OpenAI-compatible endpoint (remote) |

## Getting Started

Complete step-by-step guide to get Engram running with your own documents.

### Prerequisites

| Requirement | Purpose | Install |
|-------------|---------|---------|
| Docker | Run Neo4j database | [docker.com](https://docker.com) |
| Python 3.11+ | Runtime | `brew install python@3.11` or system package |
| uv | Package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Ollama (optional) | Local LLM for extraction | [ollama.com](https://ollama.com) |

### Step 1: Clone and Install

```bash
# Clone repository
git clone <repo-url>
cd engram

# Install Python dependencies
uv sync

# Copy environment template
cp .env.example .env
```

### Step 2: Configure Environment

Edit `.env` with your settings:

```bash
# Required: LLM endpoint for extraction and synthesis
LLM_BASE_URL=http://localhost:11434/v1   # Ollama local
LLM_MODEL=qwen3:8b                        # Recommended model

# Neo4j (defaults work with Docker setup below)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=engram_password

# Retrieval mode (start with bm25_graph, no GPU needed)
RETRIEVAL_MODE=bm25_graph
```

### Step 3: Start Neo4j Database

```bash
# Start Neo4j with persistent storage
docker run -d --name engram-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v ~/engram-data/neo4j/data:/data \
  -v ~/engram-data/neo4j/logs:/logs \
  -e NEO4J_AUTH=neo4j/engram_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:5.15-community

# Wait for Neo4j to start (~30 seconds)
sleep 30

# Verify Neo4j is running
curl -s http://localhost:7474 | head -1
# Should show: {"bolt":...}
```

### Step 4: Start LLM (Ollama)

```bash
# Start Ollama server
ollama serve &

# Pull recommended model (one-time)
ollama pull qwen3:8b

# Verify Ollama is running
curl -s http://localhost:11434/api/tags | head -1
```

### Step 5: Ingest Your Documents

Put your `.md` or `.txt` files in a directory, then run ingestion:

```bash
# Ingest documents (replace with your path)
uv run python scripts/run_ingestion.py /path/to/your/docs

# Or use --clear to reset and re-ingest
uv run python scripts/run_ingestion.py --clear /path/to/your/docs
```

**What happens during ingestion:**
- Documents are parsed and normalized
- LLM extracts concepts (entities, tools, actions)
- LLM extracts memories (facts, procedures)
- Tables are enriched with summaries
- Everything is stored in Neo4j graph

### Step 6: Optimize Graph Quality (Recommended)

After ingestion, improve the knowledge graph:

```bash
# Run deduplication + enrichment
uv run python scripts/improve_graph_quality.py all

# Check graph statistics
uv run python scripts/improve_graph_quality.py stats
```

**What this does:**
- Merges duplicate concepts (e.g., "Docker" and "докер")
- Adds world knowledge (definitions, relationships)
- Improves retrieval accuracy

### Step 7: Build Vector Index (Optional, for Hybrid Mode)

If you want vector search (better semantic matching):

```bash
# Build FAISS index with BGE-M3 embeddings
uv run python scripts/build_vector_index.py

# Enable hybrid mode in .env
# RETRIEVAL_MODE=hybrid
```

### Step 8: Compute Graph Layout (For Visualization)

Required if you want to use the `/constellation` graph visualization:

```bash
uv run python scripts/compute_layout.py
```

### Step 9: Start the Server

```bash
uv run python -m engram.api.main
```

Server starts at `http://localhost:8000`.

### Step 10: Test Your Setup

**Option A: Quick test with curl**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What can you tell me about Docker?"}]}'
```

**Option B: Interactive chat CLI**

```bash
uv run python scripts/chat.py
```

Commands:
- Type your question and press Enter
- `/stats` - Show system statistics
- `/concepts` - List extracted concepts
- `/memories` - List stored memories
- `/quit` - Exit

**Option C: Web visualization**

Open `http://localhost:8000/constellation` in your browser to see the knowledge graph and chat with activation visualization.

**Option D: Open WebUI**

See [Open WebUI Integration](#open-webui-integration) section below.

### Quick Reference: Full Setup Commands

Copy-paste this entire block to set up from scratch:

```bash
# 1. Install dependencies
uv sync

# 2. Start Neo4j
docker run -d --name engram-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v ~/engram-data/neo4j/data:/data \
  -e NEO4J_AUTH=neo4j/engram_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:5.15-community
sleep 30

# 3. Start Ollama
ollama serve &
ollama pull qwen3:8b

# 4. Ingest your documents
uv run python scripts/run_ingestion.py /path/to/your/docs

# 5. Optimize graph quality
uv run python scripts/improve_graph_quality.py all

# 6. (Optional) Build vector index for hybrid mode
uv run python scripts/build_vector_index.py

# 7. Compute layout for visualization
uv run python scripts/compute_layout.py

# 8. Start server
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

# Retrieval Mode (v4.7)
RETRIEVAL_MODE=bm25_graph # Default: BM25 + Graph only (no embeddings)
# RETRIEVAL_MODE=hybrid   # Enable vector search (requires embedding model)

# Retrieval
RETRIEVAL_TOP_K=100       # Final memories sent to LLM
RETRIEVAL_BM25_K=200      # BM25 candidates before fusion
RETRIEVAL_VECTOR_K=200    # Vector candidates (only in hybrid mode)
RETRIEVAL_GRAPH_K=200     # Graph spreading candidates before fusion

# Reranker (v5: BGE default)
RERANKER_ENABLED=true     # Enable cross-encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3  # Default: BGE (or jinaai/jina-reranker-v3)
RERANKER_CANDIDATES=64    # Candidates per batch
RERANKER_USE_FP16=true    # Use FP16 for faster inference

# v5 BGE-M3 Embeddings
BGE_MODEL_NAME=BAAI/bge-m3
BGE_USE_FP16=true
BGE_BATCH_SIZE=32

# v5 Vector Index (FAISS)
VECTOR_INDEX_PATH=./data/vector_index
VECTOR_INDEX_TYPE=flat    # "flat" (exact) or "ivf" (approximate)
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

**Request:**
```json
{
  "messages": [{"role": "user", "content": "What is Docker?"}],
  "model": "engram"
}
```

**With debug info:**
```json
{
  "messages": [{"role": "user", "content": "What is Docker?"}],
  "model": "engram",
  "debug": true
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
| `GET /constellation` | Interactive memory graph visualization |

#### Reset Database

To clear all data and re-ingest (useful when changing embedding models):

```bash
# Reset database (WARNING: deletes ALL data)
curl -X POST "http://localhost:8000/admin/reset?confirm=true"

# Re-run ingestion
uv run python scripts/run_ingestion.py
```

## Constellation (Memory Graph Visualization)

Engram includes an interactive WebGL graph visualization of the memory network, accessible at `/constellation`. Optimized for large graphs (30k+ nodes) with pre-computed layouts.

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
- Shows retrieved memories with scores (0-1) and sources:
  - `B` = BM25, `V` = Vector (hybrid mode only), `G` = Graph, `F` = Forced
  - `BE` = BM25 Expanded, `S` = Semantic rewrite, `H` = HyDE
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

See [Step 5 in Getting Started](#step-5-ingest-your-documents) for basic ingestion.

**Additional commands:**

```bash
# Generate mock documents (for testing)
uv run python scripts/generate_mock_docs.py

# Create concept index (if not auto-created)
uv run python scripts/create_concept_index.py
```

**Supported formats:** `.md`, `.txt`, `.markdown`

**Confluence Export Support:**
- Extracts title from `Заголовок страницы:` field
- Extracts URL from `URL страницы:` field
- Strips metadata header before sending content to LLM
- Sources displayed in chat responses with clickable links

## Graph Quality Optimization (v4.4)

Improves knowledge graph quality through concept deduplication and semantic enrichment.

### Server Commands

```bash
# 1. Show graph quality statistics
uv run python scripts/improve_graph_quality.py stats

# 2. Run concept deduplication (finds and merges duplicate concepts)
uv run python scripts/improve_graph_quality.py dedup

# 3. Run semantic enrichment (adds world knowledge edges)
uv run python scripts/improve_graph_quality.py enrich

# 4. Enrichment with options
uv run python scripts/improve_graph_quality.py enrich --limit 100      # Test with 100 concepts
uv run python scripts/improve_graph_quality.py enrich --min-degree 2   # Only concepts with 2+ edges

# 5. Run both deduplication and enrichment
uv run python scripts/improve_graph_quality.py all

# 6. Preview changes without applying (dry run)
uv run python scripts/improve_graph_quality.py dedup --dry-run

# 7. Interactive review of medium-confidence duplicates
uv run python scripts/review_duplicates.py
```

### Features

**Deduplication:**
- Cross-lingual duplicate detection using LaBSE embeddings
- Phonetic matching via transliteration (Jaro-Winkler)
- Auto-merge high-confidence duplicates (≥0.95)
- POSSIBLE_DUPLICATE edges for medium confidence (≥0.80)

**Enrichment:**
- LLM-generated concept definitions
- World knowledge relations (is_a, contains, uses, needs)
- Edge classification (is_semantic, is_universal)
- 1.5x activation boost for semantic edges

### Configuration

```bash
# Add to .env
SEMANTIC_EDGE_BOOST=1.5           # Boost for semantic edges
DEDUP_AUTO_MERGE_THRESHOLD=0.95   # Auto-merge threshold
DEDUP_REVIEW_THRESHOLD=0.80       # Review threshold
DEDUP_POSSIBLE_THRESHOLD=0.60     # Tracking threshold

# Enrichment LLM (separate fast model for graph enrichment + retrieval concept extraction)
ENRICHMENT_LLM_ENABLED=true
ENRICHMENT_LLM_BASE_URL=http://localhost:8889/v1
ENRICHMENT_LLM_MODEL=Qwen/Qwen3-4B
ENRICHMENT_LLM_TIMEOUT=30.0
ENRICHMENT_LLM_MAX_CONCURRENT=128  # Increase with more GPUs
```

### vLLM Setup for Enrichment

For fast enrichment of large graphs (10k+ concepts), use a dedicated vLLM server:

```bash
# Single GPU
docker run -d --name engram-enrichment --runtime nvidia --gpus '"device=1"' \
  -v /data/cache/huggingface:/root/.cache/huggingface -p 8889:8000 --ipc=host \
  --restart unless-stopped vllm/vllm-openai:latest --model Qwen/Qwen3-4B \
  --gpu-memory-utilization 0.3 --max-model-len 8192 --dtype bfloat16

# Multi-GPU with tensor parallelism (2x throughput)
docker run -d --name engram-enrichment --runtime nvidia --gpus '"device=0,1"' \
  -v /data/cache/huggingface:/root/.cache/huggingface -p 8889:8000 --ipc=host \
  --restart unless-stopped vllm/vllm-openai:latest --model Qwen/Qwen3-4B \
  --gpu-memory-utilization 0.3 --max-model-len 8192 --dtype bfloat16 \
  --tensor-parallel-size 2
```

Recommended concurrency by GPU count:
| GPUs | `--tensor-parallel-size` | `ENRICHMENT_LLM_MAX_CONCURRENT` |
|------|--------------------------|--------------------------------|
| 1    | 1 (default)              | 64                             |
| 2    | 2                        | 128                            |
| 4    | 4                        | 256                            |

## v5 Migration (Upgrading from v4.x)

> **New users:** Skip this section. Follow [Getting Started](#getting-started) instead.

To upgrade an existing v4.x installation to v5 with BGE-M3 vector retrieval:

```bash
# 1. Install new dependencies
uv sync

# 2. Run migration (re-embeds all memories with BGE-M3, builds FAISS index)
uv run python scripts/migrate_to_v5.py

# 3. Optional: Set hybrid mode in .env
# RETRIEVAL_MODE=hybrid

# 4. Start API server
uv run python -m engram.api.main
```

**Migration options:**
```bash
# Custom batch size (for memory-constrained systems)
uv run python scripts/migrate_to_v5.py --batch-size 50

# Custom index path
uv run python scripts/migrate_to_v5.py --index-path ./data/my_index

# Skip if index already exists
uv run python scripts/migrate_to_v5.py --skip-existing
```

**Retrieval evaluation:**
```bash
# Run evaluation against golden queries
uv run python -m engram.evaluation.runner tests/golden_queries.json

# With options
uv run python -m engram.evaluation.runner tests/golden_queries.json -n 10 -v -o results.json
```

## Debug Retrieval CLI (v4.5)

Trace chunks through every pipeline stage to understand where relevant results appear and disappear.

### Basic Usage

```bash
# Debug a query (includes answer generation)
uv run python scripts/debug_retrieval.py "какие типы задач в jira"

# Skip answer generation (faster, retrieval only)
uv run python scripts/debug_retrieval.py "какие типы задач в jira" --no-answer

# Search for chunks containing specific text
uv run python scripts/debug_retrieval.py "типы задач в jira" --search-text "Epic"

# Track a specific chunk through the pipeline
uv run python scripts/debug_retrieval.py "jira" --find-chunk "mem-abc123"

# Verbose output with full context
uv run python scripts/debug_retrieval.py "jira" -v --show-context

# Save trace to file for later analysis
uv run python scripts/debug_retrieval.py "jira" --save-trace
```

### Options

| Option | Description |
|--------|-------------|
| `--search-text TEXT` | Find chunks containing this text at each stage |
| `--find-chunk ID` | Track a specific memory ID through the pipeline |
| `--top-k N` | Number of results to return (default: 20) |
| `--no-answer` | Skip answer generation (retrieval only) |
| `--show-context` | Show full content context for top results |
| `--save-trace` | Save trace to JSON file |
| `-v, --verbose` | Verbose output with full journey details |

### Example Output

```
Query: docker container
------------------------------------------------------------

Running traced retrieval...

============================================================
Retrieval Trace: abc123-def456
Query: docker container
Timestamp: 2025-01-27T12:00:00
Total duration: 245.3ms
Extracted concepts: docker, container

Pipeline Steps:
  query_embedding: 0 -> 0 (0.0ms)
  concept_extraction: 0 -> 2 (45.2ms)
  concept_matching: 2 -> 2 (12.1ms)
  spreading_activation: 2 -> 15 (28.4ms)
  graph_retrieval: 15 -> 42 (35.6ms)
  path_retrieval: 2 -> 8 (22.3ms)
  bm25_search: 0 -> 50 (18.7ms)
  hybrid_fusion: 85 -> 20 (82.9ms)
    dropped: 65

Sources:
  BM25: 50
  Graph (spreading): 42
  Path: 8

Final: 20 chunks included
============================================================
```

### Source Codes

| Code | Source |
|------|--------|
| `B` | BM25 full-text search |
| `G` | Graph (spreading activation) |
| `P` | Path-based retrieval |
| `V` | Vector search (hybrid mode only) |
| `BE` | BM25 Expanded |
| `S` | Semantic rewrite |
| `H` | HyDE |

### Path Retrieval Details

When using `-v --show-context`, path retrieval details are shown:

```
--- Path Retrieval Details ---
  Paths found: 3
  Bridge concepts: 5
  Shared memories: 1
  Path memories: 4

  Paths between concepts:
    docker... -> container... (len=1)
    docker... -> kubernetes... (len=2)

  Top bridge concepts:
    kubernetes (connects 2 query concepts)
    virtualization (connects 2 query concepts)
```

## Project Structure

```
engram/
├── src/engram/
│   ├── models/          # Concept, SemanticMemory, EpisodicMemory
│   ├── ingestion/       # Document parsing, concept/memory extraction
│   ├── storage/         # Neo4j client and schema
│   ├── retrieval/       # Embeddings, spreading activation, hybrid search
│   ├── reasoning/       # Synthesis, episode management
│   ├── learning/        # Feedback, consolidation, memory strength
│   └── api/             # FastAPI routes
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/mock_docs/
├── scripts/
│   ├── setup_rhel.sh    # Production setup
│   ├── chat.py          # Interactive CLI
│   ├── run_ingestion.py
│   ├── compute_layout.py       # Pre-compute graph layout
│   ├── create_concept_index.py # Concept fulltext index
│   ├── debug_retrieval.py      # Debug retrieval pipeline (v4.5)
│   ├── migrate_to_v5.py        # v5 migration (re-embed + build FAISS)
│   └── build_vector_index.py   # Standalone FAISS index builder
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

**Constellation:**
- [x] Pre-computed layout (igraph/cuGraph)
- [x] Viewport culling
- [x] WebGL rendering (30k+ nodes)
- [x] Louvain clustering
- [x] Integrated chat with activation visualization
- [x] Debug mode (retrieval scores, sources, force include/exclude)

**v3 Retrieval & Memory:**
- [x] RRF fusion with configurable k parameter
- [x] BGE cross-encoder reranking
- [x] Russian NLP (PyMorphy3 lemmatization, stopwords)
- [x] ACT-R base-level activation for memory decay
- [x] Memory status lifecycle (active → deprioritized → archived)
- [x] LLM-based contradiction detection (Russian prompts)
- [x] Calibration CLI (`scripts/calibrate.py`)

**v3.1 Sources & Ingestion:**
- [x] Source attribution in responses (title + URL)
- [x] Confluence metadata extraction (title, URL)
- [x] Metadata stripping before LLM processing
- [x] Ingestion `--clear` flag for database reset
- [x] Thread-safe embedding service
- [x] Configurable LLM timeout

**v3.2 Table & List Extraction:**
- [x] Symbol normalizer (Unicode NFC, whitespace, bullets)
- [x] Table cell normalization (+/- → да/нет, checkmarks → да)
- [x] Markdown table detection and parsing
- [x] Comparison table detection (feature matrices)
- [x] LLM-based table enrichment (descriptions, key facts)
- [x] Automatic memory creation from tables
- [x] Tables processed separately, removed from text before LLM

**v3.3 Retrieval Quality:**
- [x] Quality filtering with chunk scoring and source weights
- [x] MMR (Maximal Marginal Relevance) for result diversity
- [x] Dynamic top_k based on query complexity classification
- [x] Reranker integration in hybrid search pipeline
- [x] Structure-aware list extraction (definition, procedure, bullet)
- [x] Multi-vector table retrieval (summary for search, raw for generation)
- [x] Russian/Latin transliteration with query expansion
- [x] Person/role extraction using Natasha NER

**v3.4 Response & Extraction Improvements:**
- [x] Current date context in system prompt for time-aware answers
- [x] Lower synthesis temperature (0.4) for more factual RAG responses
- [x] Table enricher merges surrounding context into unified summary
- [x] PyMorphy3 validation for person extraction (reduces false positives)
- [x] Reranker preloading at startup (faster first query)
- [x] Reranker single GPU mode (avoids conflicts with vLLM on multi-GPU setups)

**v3.5 Ingestion Improvements:**
- [x] Batch Neo4j writes (UNWIND queries instead of individual MERGEs)
- [x] High concurrency config (32 docs parallel)
- [x] Separate extractors with table enrichment (2 + N LLM calls per doc)
- [x] Embeddings removed from ingestion (v5 uses separate `build_vector_index.py`)

**Dual-Content Memory:**
- [x] Separate `search_content` (summary + keywords) from `content` (actual facts)
- [x] `search_content` used for BM25 search
- [x] `content` contains near-verbatim facts sent to LLM
- [x] Parser backward-compatible with v3/v2 formats
- [x] Neo4j fulltext index includes both fields

**Weighted Retrieval:**
- [x] Configurable RRF weights (v5: BM25=0.25, Vector=0.25, Graph=0.20, Path=0.30)
- [x] BM25 searches `content` field (original facts)
- [x] Vector searches `search_content` embedding (summary + keywords)
- [x] `weighted_rrf()` function for source-prioritized fusion

**BM25+Graph Mode & Jina Reranker:**
- [x] Configurable `retrieval_mode`: `bm25_graph` (default) or `hybrid`
- [x] In `bm25_graph` mode: no embedding model needed, faster startup and ingestion
- [x] Jina Reranker v3 (better multilingual support)
- [x] Skip vector search and embedding preload in `bm25_graph` mode
- [x] BM25 concept search fallback when vector search disabled
- [x] Batch reranking with 64 candidates per pass

**v4.4 Graph Quality Optimization:**
- [x] Concept deduplication with LaBSE cross-lingual embeddings
- [x] Phonetic matching via transliteration (Jaro-Winkler)
- [x] Auto-merge high-confidence duplicates (≥0.95)
- [x] POSSIBLE_DUPLICATE edges for medium confidence (≥0.80)
- [x] Interactive duplicate review CLI
- [x] LLM-based semantic enrichment (definitions, relations)
- [x] Edge classification (is_semantic, is_universal, source_type)
- [x] Semantic edge boost in spreading activation (1.5x)
- [x] ConceptResolver for duplicate prevention during ingestion
- [x] Graph quality metrics and reporting

**v4.5 Path-Based Retrieval & Observability:**
- [x] Path-based retrieval for memories bridging multiple query concepts
- [x] Shared memories (linked to 2+ query concepts)
- [x] Bridge concepts (connecting multiple query concepts)
- [x] Path memories (from intermediate concepts on paths)
- [x] Retrieval observability with ChunkTrace, StepTrace, RetrievalTrace
- [x] TracedRetriever wrapper for pipeline tracing
- [x] Debug CLI (`scripts/debug_retrieval.py`) with chunk journey tracking
- [x] Trace export to JSON files

**v5 Vector Retrieval & BGE Integration:**
- [x] BGE-M3 embedding service (1024-dim, multilingual)
- [x] FAISS vector index (flat and IVF index types)
- [x] VectorRetriever for FAISS-based search
- [x] BGE-reranker-v2-m3 (replaces Jina as default)
- [x] 4-way RRF fusion: Vector (0.25), BM25 (0.25), Path (0.30), Graph (0.20)
- [x] Memory embeddings with BGE-M3 (`search_content` field)
- [x] VectorStepMetrics for observability
- [x] Migration script (`scripts/migrate_to_v5.py`)
- [x] Retrieval evaluation metrics (Recall@K, MRR, NDCG)
- [x] EvaluationRunner for golden query testing

### Planned

- [ ] Spatial indexing (Neo4j point indexes)
- [ ] Feedback buttons in chat
- [ ] Test query collection (save/replay, pass/fail tracking)

## License

MIT
