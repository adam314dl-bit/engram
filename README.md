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
- **Hybrid Mode**: Optional - adds vector similarity search (set `RETRIEVAL_MODE=hybrid`)
- **Spreading Activation**: Brain-like associative retrieval through concept networks
- **Weighted RRF**: Prioritized fusion with BM25 (0.45) > Vector (0.35) > Graph (0.20)
- **Jina Reranker v3**: Multilingual cross-encoder for improved retrieval precision
- **MMR Diversity**: Maximal Marginal Relevance prevents redundant results (hybrid mode only)
- **Dynamic top_k**: Query complexity classification adjusts retrieval depth

**Dual-Content Memory:**
- **Separate Search vs Display**: `search_content` for search, `content` for LLM generation
- **Search-Optimized**: Summary + keywords for BM25 search
- **Near-Verbatim Facts**: Actual data preserved for accurate LLM responses
- **Backward Compatible**: Works with existing memories (falls back to content)

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
- **Fast Ingestion**: Batch Neo4j writes and batch embeddings

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
| Embeddings | sentence-transformers (optional in bm25_graph mode) |
| Reranker | Jina Reranker v3 (multilingual cross-encoder) |
| Russian NLP | PyMorphy3, Natasha NER |
| Transliteration | cyrtranslit |
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

# Retrieval Mode (v4.7)
RETRIEVAL_MODE=bm25_graph # Default: BM25 + Graph only (no embeddings)
# RETRIEVAL_MODE=hybrid   # Enable vector search (requires embedding model)

# Retrieval
RETRIEVAL_TOP_K=100       # Final memories sent to LLM
RETRIEVAL_BM25_K=200      # BM25 candidates before fusion
RETRIEVAL_VECTOR_K=200    # Vector candidates (only in hybrid mode)

# Reranker
RERANKER_ENABLED=true     # Enable Jina cross-encoder
RERANKER_MODEL=jinaai/jina-reranker-v3
RERANKER_CANDIDATES=64    # Candidates per batch
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

```bash
# Generate mock documents (for testing)
uv run python scripts/generate_mock_docs.py

# Ingest documents
uv run python scripts/run_ingestion.py

# Ingest from custom directory
uv run python scripts/run_ingestion.py /path/to/docs

# Reset database and re-ingest (clears all data first)
uv run python scripts/run_ingestion.py --clear /path/to/docs

# Create concept index (required for BM25+Graph mode)
uv run python scripts/create_concept_index.py
```

**Confluence Export Support:**
- Extracts title from `Заголовок страницы:` field
- Extracts URL from `URL страницы:` field
- Strips metadata header before sending content to LLM
- Sources displayed in chat responses with clickable links

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
│   ├── compute_layout.py      # Pre-compute graph layout
│   └── create_concept_index.py # Concept fulltext index
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
- [x] Combined embedding calls (single batch for concepts + memories)
- [x] High concurrency config (32 docs parallel)
- [x] Separate extractors with table enrichment (2 + N LLM calls per doc)
- [x] Skip embeddings in `bm25_graph` mode for faster ingestion

**Dual-Content Memory:**
- [x] Separate `search_content` (summary + keywords) from `content` (actual facts)
- [x] `search_content` used for BM25 search
- [x] `content` contains near-verbatim facts sent to LLM
- [x] Parser backward-compatible with v3/v2 formats
- [x] Neo4j fulltext index includes both fields

**Weighted Retrieval:**
- [x] Configurable RRF weights: BM25 (0.45) > Vector (0.35) > Graph (0.20)
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

### Planned

- [ ] Spatial indexing (Neo4j point indexes)
- [ ] Feedback buttons in chat (👍/👎)
- [ ] Permanent graph fixes (add edges, boost weights, aliases)
- [ ] Test query collection (save/replay, pass/fail tracking)

## License

MIT
