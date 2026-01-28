# Engram Architecture

A comprehensive guide for backend engineers new to RAG systems.

## Table of Contents

1. [What is Engram?](#what-is-engram)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Data Model](#data-model)
5. [Retrieval Pipeline](#retrieval-pipeline)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Getting Started](#getting-started)
9. [Common Development Tasks](#common-development-tasks)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)

---

## What is Engram?

> **engram** (noun): a hypothetical permanent change in the brain accounting for the existence of memory; a memory trace.

Engram is a **cognitive-inspired knowledge system** that transforms documentation into an intelligent assistant. It reasons like a brain — connecting concepts, learning from experience, and consolidating knowledge over time.

Unlike traditional RAG that retrieves document chunks, Engram uses a **brain-inspired dual memory architecture**:

- **Concept Network**: Atomic ideas connected by typed, weighted edges
- **Semantic Memory**: Facts and procedures linked to concepts
- **Episodic Memory**: Past reasoning traces with outcomes

This architecture enables Engram to answer questions about Russian technical documentation with high accuracy and source attribution.

### Problem It Solves

Traditional search (keyword or vector-only) struggles with:
- **Multi-hop questions**: "What tools can I use to solve X?" requires connecting concepts
- **Russian morphology**: "контейнеры" and "контейнер" should match
- **Context loss**: Pure vector search returns isolated chunks without relationships

### Key Insight

**Graph + Hybrid Retrieval > Pure Vector Search**

By combining multiple retrieval methods (BM25, vector, graph traversal, path-based), Engram finds relevant information that any single method would miss.

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION (offline)                         │
│                                                                     │
│   Documents (.md, .txt)                                             │
│        │                                                            │
│        ▼                                                            │
│   LLM Extraction ─────► Concepts (entities, tools, actions)        │
│        │                     │                                      │
│        ▼                     ▼                                      │
│   Memories (facts, procedures) ◄────── RELATES_TO ──────►          │
│        │                                                            │
│        ▼                                                            │
│   Neo4j Graph + FAISS Vector Index                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL (runtime)                          │
│                                                                     │
│   User Query: "Как освободить место на диске?"                      │
│        │                                                            │
│        ▼                                                            │
│   Concept Extraction ───► ["диск", "место", "освободить"]          │
│        │                                                            │
│        ├──────────────┬──────────────┬──────────────┐              │
│        ▼              ▼              ▼              ▼              │
│      BM25          Vector        Graph          Path               │
│    (keywords)    (embeddings)  (spreading)   (bridging)            │
│        │              │              │              │              │
│        └──────────────┴──────────────┴──────────────┘              │
│                           │                                         │
│                           ▼                                         │
│                    RRF Fusion (merge rankings)                      │
│                           │                                         │
│                           ▼                                         │
│                    Reranker (cross-encoder)                         │
│                           │                                         │
│                           ▼                                         │
│                    Top-K Memories ───► LLM ───► Response           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

If you're new to RAG systems, here's what you need to know.

### Embeddings

**Text → Vector** (list of numbers representing meaning)

```
"Docker container" → [0.12, -0.34, 0.56, ..., 0.78]  (1024 dimensions)
```

Similar texts have similar vectors. We use **BGE-M3** (multilingual, 1024 dimensions).

### Vector Search

Find similar content by comparing vector distances:

```
Query: "container platform"  →  embedding  →  [0.11, -0.33, 0.55, ...]
                                                    │
                                                    ▼
                                            FAISS index search
                                                    │
                                                    ▼
                                            Top-K similar memories
```

**FAISS** is a fast vector index library from Meta.

### BM25

Traditional keyword search (like Elasticsearch's default algorithm):

- Matches exact terms
- Considers term frequency and document length
- Works well for specific terminology

We enhance BM25 with **Russian lemmatization** (PyMorphy3):
```
"контейнеры" → "контейнер" (normalized form)
```

### Knowledge Graph

Concepts connected by relationships:

```
     ┌─────────┐                    ┌─────────┐
     │ Docker  │───RELATES_TO──────►│ container│
     └─────────┘                    └─────────┘
          │                              │
          │                              │
    LINKED_TO                      LINKED_TO
          │                              │
          ▼                              ▼
    ┌─────────────────────────────────────────┐
    │ Memory: "Docker uses containers to      │
    │ isolate applications"                   │
    └─────────────────────────────────────────┘
```

### RAG (Retrieval-Augmented Generation)

Pattern for grounding LLM responses in facts:

1. **Retrieve** relevant documents/memories
2. **Augment** the prompt with retrieved context
3. **Generate** response using LLM

Engram implements RAG with a sophisticated multi-method retrieval pipeline.

---

## Architecture Overview

### System Components

```
┌──────────────────────────────────────────────────────────────────┐
│                           API Layer                               │
│                         (FastAPI)                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ /v1/chat     │  │ /constellation│  │ /admin/*            │   │
│  │ completions  │  │ (graph UI)    │  │ (stats, reset)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Retrieval Pipeline                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Concept  │ │  BM25   │ │ Vector  │ │ Graph   │ │  Path   │   │
│  │Extract  │ │ Search  │ │ Search  │ │Spreading│ │Retrieval│   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                               │                                   │
│                               ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              RRF Fusion + Reranker                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Response Generation                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           ResponseSynthesizer (LLM call)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Storage Layer                              │
│  ┌─────────────────┐              ┌─────────────────────────┐   │
│  │     Neo4j       │              │   FAISS Vector Index    │   │
│  │  (graph store)  │              │   (similarity search)   │   │
│  └─────────────────┘              └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Query arrives** at `/v1/chat/completions`
2. **Concept extraction**: LLM identifies key concepts from query
3. **Parallel retrieval**: 4 methods search simultaneously
4. **Fusion**: RRF merges ranked results from all methods
5. **Reranking**: Cross-encoder re-scores top candidates
6. **Synthesis**: LLM generates answer from top-K memories
7. **Response**: Answer with source attribution returned

---

## Data Model

### Node Types

#### Concept

Atomic ideas/entities in the knowledge graph.

```python
Concept:
    id: str              # "concept-docker-abc123"
    name: str            # "docker" (normalized, lowercase)
    type: str            # "tool" | "resource" | "action" | "state" | "config" | "error" | "general"
    description: str     # LLM-generated definition (optional)
    aliases: list[str]   # ["докер", "Docker"] - alternative names
    embedding: list[float]       # BGE-M3 embedding (1024-dim)
    labse_embedding: list[float] # LaBSE embedding (for deduplication)
```

#### Memory (SemanticMemory)

Units of knowledge: facts, procedures, relationships.

```python
SemanticMemory:
    id: str              # "mem-abc123"
    content: str         # "Docker использует контейнеры для изоляции приложений"
    search_content: str  # Optimized text for vector search (summary + keywords)
    memory_type: str     # "fact" | "procedure" | "relationship"
    concept_ids: list[str]    # Links to concepts
    source_url: str      # Original document URL
    source_title: str    # Document title
    confidence: float    # 0.0-1.0
    status: str          # "active" | "deprioritized" | "archived"
```

### Relationships

```
Concept ──RELATES_TO──► Concept
    │
    └── weight: float (0.0-1.0)
    └── is_semantic: bool (from world knowledge vs document)
    └── is_universal: bool (general truth vs context-specific)

Concept ──LINKED_TO──► Memory
    │
    └── relevance: float

Memory ──DERIVED_FROM──► Document
```

### Graph Structure (ASCII)

```
                    ┌─────────────┐
                    │   Docker    │ (Concept)
                    │  type=tool  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        RELATES_TO    RELATES_TO   RELATES_TO
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │container│  │  image  │  │  prune  │
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
         LINKED_TO    LINKED_TO    LINKED_TO
             │            │            │
             ▼            ▼            ▼
        ┌─────────────────────────────────────┐
        │ Memory: "To free disk space in     │
        │ Docker, run: docker system prune"  │
        │ source: confluence.example.com/... │
        └─────────────────────────────────────┘
```

---

## Retrieval Pipeline

Each component explained with its purpose, why it matters, and where to find the code.

### 1. Concept Extraction

**What it does**: Extracts key concepts from the user's query.

**Why it matters**: Seeds the graph traversal and helps focus retrieval.

**Example**:
```
Query: "Как освободить место на диске Docker?"
Concepts: ["docker", "диск", "место", "освободить"]
```

**Key file**: `src/engram/ingestion/concept_extractor.py`

### 2. BM25 Search

**What it does**: Keyword-based search with Russian language support.

**Why it matters**: Catches exact terminology that embeddings might miss.

**How it works**:
1. Query is lemmatized: "контейнеры" → "контейнер"
2. Stopwords removed: "как", "на", "в"
3. BM25 scores computed against `content` field
4. Top-K candidates returned

**Key file**: `src/engram/retrieval/hybrid_search.py`

### 3. Vector Search

**What it does**: Semantic similarity search using embeddings.

**Why it matters**: Finds conceptually related content even without exact keyword matches.

**How it works**:
1. Query embedded with BGE-M3 → 1024-dim vector
2. FAISS index searched for nearest neighbors
3. Results scored by cosine similarity

**Key files**:
- `src/engram/embeddings/bge_service.py` - Embedding generation
- `src/engram/embeddings/vector_index.py` - FAISS index
- `src/engram/retrieval/vector_retriever.py` - Search logic

### 4. Graph Spreading Activation

**What it does**: Traverses the concept graph from seed concepts, activating related concepts.

**Why it matters**: Discovers related information through graph structure (not just text similarity).

**How it works**:
```
1. Seed concepts activated (activation = 1.0)
2. For each hop (max 3):
   - Get neighbors of active concepts
   - Transfer activation: neighbor += current × weight × decay
   - Semantic edges get 1.5x boost
   - Keep top-k per hop (lateral inhibition)
3. Collect memories linked to activated concepts
```

**Key file**: `src/engram/retrieval/spreading_activation.py`

### 5. Path-Based Retrieval

**What it does**: Finds memories that bridge multiple query concepts.

**Why it matters**: Handles multi-hop questions where the answer connects several concepts.

**Example**:
```
Query: "How do containers and images relate?"
→ Finds memories linked to BOTH "container" AND "image"
→ Finds memories on paths between these concepts
```

**Key file**: `src/engram/retrieval/path_retrieval.py`

### 6. RRF Fusion

**What it does**: Merges ranked results from all retrieval methods.

**Why it matters**: Each method has strengths; fusion combines them optimally.

**Algorithm** (Reciprocal Rank Fusion):
```
score(doc) = Σ (weight_i / (k + rank_i))

where:
  k = 60 (smoothing parameter)
  weight_i = method weight (BM25=0.25, Vector=0.25, Graph=0.20, Path=0.30)
  rank_i = document rank from method i
```

**Key file**: `src/engram/retrieval/fusion.py`

### 7. Reranker

**What it does**: Cross-encoder that re-scores candidates with full query-document attention.

**Why it matters**: More accurate than bi-encoder similarity, catches nuances.

**Model**: BGE-reranker-v2-m3 (multilingual, good Russian support)

**Key file**: `src/engram/retrieval/reranker.py`

### 8. Response Synthesis

**What it does**: LLM generates answer from retrieved memories.

**Key file**: `src/engram/reasoning/synthesizer.py`

---

## Tech Stack

| Component | Technology | Purpose | File |
|-----------|------------|---------|------|
| **Database** | Neo4j 5.15 | Graph storage | `storage/neo4j_client.py` |
| **Embeddings** | BGE-M3 | 1024-dim multilingual | `embeddings/bge_service.py` |
| **Vector Index** | FAISS | Fast similarity search | `embeddings/vector_index.py` |
| **Reranker** | BGE-reranker-v2-m3 | Cross-encoder scoring | `retrieval/reranker.py` |
| **Russian NLP** | PyMorphy3 | Lemmatization | `preprocessing/russian.py` |
| **NER** | Natasha | Named entity recognition | `ingestion/person_extractor.py` |
| **Transliteration** | cyrtranslit | Cyrillic↔Latin | `preprocessing/transliteration.py` |
| **API** | FastAPI | HTTP endpoints | `api/main.py` |
| **LLM** | OpenAI-compatible | Extraction & synthesis | `ingestion/llm_client.py` |

---

## Project Structure

```
src/engram/
├── api/                    # HTTP endpoints
│   ├── main.py             # FastAPI app, startup/shutdown
│   ├── routes.py           # /v1/chat/completions, /admin/*
│   └── graph.py            # /constellation visualization
│
├── ingestion/              # Document processing (offline)
│   ├── pipeline.py         # Main ingestion orchestration
│   ├── concept_extractor.py
│   ├── memory_extractor.py
│   ├── relationship_extractor.py
│   ├── list_extractor.py   # Structure-aware list parsing
│   ├── person_extractor.py # NER-based person extraction
│   ├── llm_client.py       # LLM API wrapper
│   └── prompts.py          # Russian extraction prompts
│
├── retrieval/              # Query processing (runtime)
│   ├── pipeline.py         # Orchestrates full retrieval
│   ├── hybrid_search.py    # BM25 + vector + graph
│   ├── vector_retriever.py # FAISS-based search
│   ├── spreading_activation.py
│   ├── path_retrieval.py   # Multi-concept bridging
│   ├── fusion.py           # RRF merge
│   ├── reranker.py         # Cross-encoder
│   ├── quality_filter.py   # Chunk scoring
│   └── observability.py    # Trace models
│
├── embeddings/             # Vector operations
│   ├── bge_service.py      # BGE-M3 embedding
│   └── vector_index.py     # FAISS index
│
├── storage/                # Database layer
│   ├── neo4j_client.py     # All Neo4j operations
│   └── schema.py           # Cypher for indexes/constraints
│
├── preprocessing/          # Text processing
│   ├── russian.py          # PyMorphy3 lemmatization
│   ├── normalizer.py       # Unicode normalization
│   ├── table_parser.py     # Markdown tables
│   ├── table_enricher.py   # LLM table enhancement
│   └── transliteration.py  # Cyrillic↔Latin
│
├── graph/                  # Graph quality (v4.4)
│   ├── deduplication.py    # LaBSE cross-lingual dedup
│   ├── enrichment.py       # LLM world knowledge
│   ├── resolver.py         # Alias→canonical mapping
│   └── metrics.py          # Quality reporting
│
├── reasoning/              # Response generation
│   ├── synthesizer.py      # LLM response synthesis
│   ├── pipeline.py         # Full reasoning flow
│   └── episode_manager.py  # Episodic memory
│
├── models/                 # Data classes
│   ├── concept.py
│   ├── semantic_memory.py
│   ├── episodic_memory.py
│   └── document.py
│
├── evaluation/             # Testing & metrics
│   ├── evaluator.py        # Test set evaluation
│   ├── metrics.py          # Recall@K, MRR, NDCG
│   └── runner.py           # Golden query testing
│
└── config.py               # Settings from environment
```

---

## Getting Started

### Prerequisites

- **Docker**: For Neo4j
- **Python 3.11+**: Runtime
- **uv**: Package manager (not pip)
- **Ollama** (optional): Local LLM for extraction

### Quick Setup

```bash
# 1. Clone and install dependencies
git clone <repo>
cd engram
uv sync

# 2. Start Neo4j with persistent storage
docker run -d --name engram-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v ~/engram-data/neo4j/data:/data \
  -v ~/engram-data/neo4j/logs:/logs \
  -e NEO4J_AUTH=neo4j/engram_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:5.15-community

# Wait for Neo4j (~30 seconds)
sleep 30

# 3. (Optional) Start Ollama for local LLM
ollama serve &
ollama pull qwen3:8b

# 4. Start API server
uv run python -m engram.api.main
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check Neo4j
curl http://localhost:7474

# Test a query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Common Development Tasks

### Run Tests

```bash
uv run pytest                    # All tests
uv run pytest --cov=engram       # With coverage
uv run pytest tests/test_fusion.py -v  # Specific file
```

### Ingest Documents

```bash
# From default directory
uv run python scripts/run_ingestion.py

# From custom directory
uv run python scripts/run_ingestion.py /path/to/docs

# Clear and re-ingest
uv run python scripts/run_ingestion.py --clear /path/to/docs
```

### Build Vector Index

After ingestion, build the FAISS index for vector search:

```bash
uv run python scripts/build_vector_index.py
```

### Debug Retrieval

Trace a query through the pipeline:

```bash
# Basic debug
uv run python scripts/debug_retrieval.py "какие типы задач в jira"

# Skip answer generation
uv run python scripts/debug_retrieval.py "jira" --no-answer

# Search for specific text in results
uv run python scripts/debug_retrieval.py "jira" --search-text "Epic"

# Verbose output
uv run python scripts/debug_retrieval.py "jira" -v --show-context
```

### Compute Graph Layout

Required for the `/constellation` visualization:

```bash
uv run python scripts/compute_layout.py
```

### Type Checking & Linting

```bash
uv run mypy src/engram           # Type checking
uv run ruff check src/engram     # Linting
```

---

## Configuration Reference

Key environment variables (add to `.env`):

### Retrieval Mode

```bash
RETRIEVAL_MODE=bm25_graph    # BM25 + Graph only (no embeddings, faster startup)
# RETRIEVAL_MODE=hybrid      # Full 4-way retrieval (requires vector index)
```

### RRF Fusion Weights

```bash
RRF_K=60                     # Smoothing parameter
RRF_BM25_WEIGHT=0.25
RRF_VECTOR_WEIGHT=0.25
RRF_GRAPH_WEIGHT=0.20
RRF_PATH_WEIGHT=0.30
```

### Retrieval Limits

```bash
RETRIEVAL_TOP_K=100          # Final memories sent to LLM
RETRIEVAL_BM25_K=200         # BM25 candidates
RETRIEVAL_VECTOR_K=200       # Vector candidates
RETRIEVAL_GRAPH_K=200        # Graph candidates
```

### Reranker

```bash
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_CANDIDATES=64
RERANKER_DEVICE=cuda:0       # or "cpu"
```

### Russian NLP

```bash
BM25_LEMMATIZE=true
BM25_REMOVE_STOPWORDS=true
```

### LLM

```bash
LLM_BASE_URL=http://localhost:11434/v1   # Ollama
LLM_MODEL=qwen3:8b
LLM_MAX_CONCURRENT=32
LLM_TIMEOUT=300.0
```

### Neo4j

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=engram_password
```

---

## Troubleshooting

### Neo4j Connection Failed

```
Error: Unable to connect to Neo4j at bolt://localhost:7687
```

**Solution**:
```bash
# Check if container is running
docker ps | grep engram-neo4j

# Start if stopped
docker start engram-neo4j

# Check logs
docker logs engram-neo4j
```

### FAISS Index Not Found

```
Error: Vector index not found at ./data/vector_index
```

**Solution**:
```bash
# Build the index
uv run python scripts/build_vector_index.py
```

### Out of Memory (Embeddings)

```
Error: CUDA out of memory
```

**Solution**: Reduce batch size in `.env`:
```bash
EMBEDDING_BATCH_SIZE=32    # Lower from default 128
BGE_BATCH_SIZE=16          # Lower from default 32
```

Or use CPU:
```bash
RERANKER_DEVICE=cpu
```

### Slow Ingestion

**Solution**: Increase parallelism (if you have resources):
```bash
INGESTION_MAX_CONCURRENT=64
LLM_MAX_CONCURRENT=64
```

### Russian Text Not Matching

**Symptoms**: Queries in Russian don't find expected results.

**Check**:
1. Lemmatization enabled: `BM25_LEMMATIZE=true`
2. PyMorphy3 installed: `uv run python -c "import pymorphy3"`
3. Test lemmatization:
   ```bash
   uv run python scripts/calibrate.py test-russian
   ```

### Graph Visualization Empty

**Solution**: Compute layout first:
```bash
uv run python scripts/compute_layout.py
```

### LLM Timeout

```
Error: Request timeout after 300s
```

**Solution**:
```bash
LLM_TIMEOUT=600.0    # Increase timeout
```

Or use a faster model:
```bash
LLM_MODEL=qwen3:4b   # Smaller, faster
```

---

## Next Steps

- Read `README.md` for detailed CLI commands and configuration options
- Explore `/constellation` in browser to visualize the knowledge graph
- Run `scripts/debug_retrieval.py` to understand retrieval behavior
- Check `scripts/calibrate.py` for testing individual components
