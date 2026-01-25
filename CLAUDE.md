# Claude Code Instructions

## Package Manager: uv

This project uses **uv** (not pip). uv is a fast Python package manager.

Commands:
- `uv sync` - Install dependencies from pyproject.toml/uv.lock
- `uv sync --python 3.11` - Same, but use specific Python version
- `uv run <cmd>` - Run command in project environment (e.g., `uv run pytest`)
- `uv add <pkg>` - Add dependency
- `uv remove <pkg>` - Remove dependency
- `uv lock` - Update lock file

Never use:
- `pip install`
- `uv pip install`

## Quick Start (Full Setup)

Run these commands to start everything from scratch:

```bash
# 1. Start Neo4j with persistent storage
docker run -d --name engram-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v ~/engram-data/neo4j/data:/data \
  -v ~/engram-data/neo4j/logs:/logs \
  -e NEO4J_AUTH=neo4j/engram_password \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:5.15-community

# Wait for Neo4j to start (~30 seconds)
sleep 30

# 2. Start Ollama (if not running)
ollama serve &

# 3. Start API server
uv run python -m engram.api.main
```

If Neo4j data already exists (persistent volume), skip ingestion and go directly to step 3.

## First-Time Setup (Ingestion)

Only needed once or when adding new documents:

```bash
# 1. Ingest documents (uses qwen3:8b via Ollama)
uv run python scripts/run_ingestion.py

# 2. Compute graph layout (required for visualization)
uv run python scripts/compute_layout.py
```

## Neo4j Setup (Persistent Storage)

**Start Neo4j (first time):**
```bash
docker run -d --name engram-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v ~/engram-data/neo4j/data:/data \
  -v ~/engram-data/neo4j/logs:/logs \
  -e NEO4J_AUTH=neo4j/engram_password \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:5.15-community
```

**Subsequent starts:**
```bash
docker start engram-neo4j
```

**Data location:** `~/engram-data/neo4j/` (persists across container restarts)

## Ollama Setup

**Recommended model:** `qwen3:8b` (good balance of speed and quality)

```bash
# Install model (one-time)
ollama pull qwen3:8b

# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

**Alternative models:**
- `qwen3:4b` - Faster, less accurate
- `qwen3:32b` - Slower, more accurate (needs more RAM)
- `qwen2.5:3b` - Very fast, basic quality

## Enrichment LLM Setup (v4.3.1)

Optional fast model for query enrichment. Falls back to main LLM if unavailable.

**Local Development (Ollama):**
```bash
ollama pull qwen3:4b
# Uses same Ollama server, just different model name
```

**Production (vLLM Docker on GPU 1):**
```bash
docker run -d --name engram-enrichment \
  --runtime nvidia --gpus '"device=1"' \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -p 8889:8000 \
  --ipc=host \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B \
  --gpu-memory-utilization 0.3 \
  --max-model-len 8192 \
  --dtype bfloat16
```

**Test it:**
```bash
curl http://localhost:8889/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "test"}]}'
```

**Configure in `.env`:**
```bash
ENRICHMENT_LLM_ENABLED=true
ENRICHMENT_LLM_BASE_URL=http://localhost:8889/v1  # vLLM Docker
ENRICHMENT_LLM_MODEL=Qwen/Qwen3-4B
```

## Project Setup

```bash
uv sync              # Install deps
uv run pytest        # Run tests
uv run python -m engram.api.main  # Run API
```

## Architecture

- **LLM**: Remote OpenAI-compatible endpoint (configured via LLM_BASE_URL)
- **Embeddings**: Local HuggingFace model via sentence-transformers (2048 dims with Giga-Embeddings)
- **Reranker**: FlagEmbedding BGE-reranker-v2-m3 (cross-encoder)
- **Russian NLP**: PyMorphy3 for lemmatization and stopwords, Natasha for NER
- **Transliteration**: cyrtranslit for Russian/Latin conversion
- **Database**: Neo4j (Docker)
- **HTTP Client**: Uses `requests` library (not httpx)

## Key Files

- `src/engram/api/main.py` - FastAPI application entry point
- `src/engram/api/routes.py` - API routes (chat, feedback, admin)
- `src/engram/api/graph.py` - Memory graph visualization (serves `/constellation`)
- `src/engram/storage/neo4j_client.py` - Neo4j database client
- `src/engram/retrieval/` - Embeddings, spreading activation, hybrid search
- `src/engram/reasoning/` - Response synthesis, episode management
- `src/engram/ingestion/` - Document parsing, concept/memory extraction (Russian prompts)

**v3 Modules:**
- `src/engram/preprocessing/russian.py` - PyMorphy3 lemmatization, Russian stopwords
- `src/engram/retrieval/fusion.py` - RRF (Reciprocal Rank Fusion)
- `src/engram/retrieval/reranker.py` - BGE cross-encoder reranking
- `src/engram/learning/forgetting.py` - ACT-R base-level activation, memory decay
- `src/engram/learning/contradiction.py` - LLM-based contradiction detection

**v3.2 Modules (Table Extraction):**
- `src/engram/preprocessing/normalizer.py` - Unicode, whitespace, bullet, table cell normalization
- `src/engram/preprocessing/table_parser.py` - Markdown table detection, parsing, extraction
- `src/engram/preprocessing/table_enricher.py` - LLM-based table enrichment, multi-vector memory creation

**v3.3 Modules (Retrieval Quality):**
- `src/engram/retrieval/quality_filter.py` - Chunk quality scoring, source weights
- `src/engram/ingestion/list_extractor.py` - Structure-aware list extraction (definition, procedure, bullet)
- `src/engram/preprocessing/transliteration.py` - Russian/Latin transliteration, query expansion
- `src/engram/ingestion/person_extractor.py` - Person/role extraction using Natasha NER with PyMorphy3 validation

**v3.4 Modules (Recent Improvements):**
- `src/engram/reasoning/synthesizer.py` - Response synthesis with date in system prompt, temperature 0.4
- `src/engram/preprocessing/table_enricher.py` - Table enrichment with context merging

**v3.5 Modules (Ingestion Improvements):**
- `src/engram/ingestion/pipeline.py` - Separate extractors + table enrichment + batch writes
- `src/engram/storage/neo4j_client.py` - Batch save methods for concepts, memories, relations
- `src/engram/preprocessing/table_enricher.py` - LLM table enrichment with context merging

**v4 Modules (Agentic RAG):**
- `src/engram/reasoning/intent_classifier.py` - Retrieval decision (RETRIEVE/NO_RETRIEVE/CLARIFY)
- `src/engram/retrieval/crag.py` - CRAG document grading and query rewrite
- `src/engram/reasoning/self_rag.py` - Self-RAG validation loop (max 3 iterations)
- `src/engram/reasoning/hallucination_detector.py` - NLI-based claim verification
- `src/engram/reasoning/citations.py` - Inline citations with verification
- `src/engram/reasoning/confidence.py` - Confidence calibration and abstention
- `src/engram/reasoning/ircot.py` - IRCoT multi-hop reasoning (max 7 steps)
- `src/engram/evaluation/ragas_eval.py` - RAGAS quality metrics
- `src/engram/reasoning/research_agent.py` - Async research mode with checkpointing
- `src/engram/reasoning/agentic_pipeline.py` - Main agentic pipeline orchestrator

**v4.2 Modules (Test Set Evaluation):**
- `src/engram/evaluation/evaluator.py` - Test set evaluation for incomplete/lazy reference answers

**v4.3 Modules (Query Understanding & Enrichment):**
- `src/engram/indexing/kb_summary.py` - KB summary generation and Neo4j storage
- `src/engram/query/enrichment.py` - Query enrichment pipeline with multi-query generation
- `scripts/generate_kb_summary.py` - Post-ingestion script for KB summary generation

**v4.3.1 Modules (Enrichment LLM):**
- `src/engram/ingestion/llm_client.py` - Named LLM client registry (main, enrichment)

**v4.4 Modules (LLM-Enhanced KB Summary):**
- `src/engram/indexing/kb_summary.py` - LLMKBSummaryEnhancer class for domain description, capabilities, limitations, sample questions

## Constellation (Memory Graph Visualization)

The graph visualization at `/constellation` displays the memory network with WebGL rendering, optimized for 30k+ nodes.

**Architecture:**
- **WebGL Rendering**: Custom shaders for nodes and edges
- **Pre-computed Layout**: Uses igraph (CPU) or cuGraph (GPU), stored in Neo4j
- **Viewport Culling**: Only loads visible nodes via bounding box queries
- **Cluster Metadata**: Pre-computed cluster centers for zoomed-out view

**Node Types:**
- **Concepts** (teal `#5eead4`): Atomic ideas from ingested documents
- **Semantic Memories** (purple `#a78bfa`): Facts and procedures linked to concepts
- **Episodic Memories** (pink `#f472b6`): Past reasoning traces with outcomes
- **Activated** (golden glow): Nodes used in chat response

**Features:**
- Integrated chat panel (💬 button)
- Live activation highlighting (cyan)
- Cluster coloring toggle
- Type filtering via clickable legend
- Global search
- Neighbor highlighting on node selection
- Connected nodes panel
- **Debug mode (🔍 button)**: Shows retrieved nodes with scores and sources

**Debug Panel (🔍 Debug button in chat):**
- Shows all retrieved memories with scores (0-1) and sources (V=Vector, B=BM25, G=Graph, F=Forced)
- Shows activated concepts with activation levels
- Visual score bars for quick comparison
- `+` button: Force include node in next query
- `-` button: Force exclude node from next query
- Click node name to highlight in graph

**Key Files:**
- `src/engram/api/graph.py` - WebGL visualization, chat panel, all endpoints
- `scripts/compute_layout.py` - Pre-compute layout with igraph/cuGraph

**Constellation Endpoints:**
- `GET /constellation` - Main visualization page
- `GET /constellation/bounds` - Get graph bounding box
- `GET /constellation/data` - Get graph data (nodes, edges, clusters)
- `GET /constellation/search` - Search nodes
- `GET /constellation/neighbors` - Get connected nodes
- `GET /constellation/stats` - Node counts and clusters
- `GET /constellation/cluster-meta` - Cluster metadata

## Ingestion

Extraction prompts are in Russian for better quality with Russian content.

```bash
# Ingest from default directory
uv run python scripts/run_ingestion.py

# Ingest from custom directory (recursive)
uv run python scripts/run_ingestion.py /path/to/docs

# Reset database and re-ingest (clears all data first)
uv run python scripts/run_ingestion.py --clear /path/to/docs
```

Supported formats: `.md`, `.txt`, `.markdown`

**Confluence Export Support:**
- Extracts title from `Заголовок страницы:` field
- Extracts URL from `URL страницы:` field
- Strips metadata header (Навигация, ID страницы, Автор, dates) before LLM processing
- Sources displayed in chat responses with clickable links

**Table/List Handling:**
- Tables extracted and enriched separately (N LLM calls for N tables)
- Table enricher creates multi-vector memories (summary + raw)
- Lists extracted via regex, chunked into memories

**Ingestion LLM Calls (per document):**
- 1 call for concepts
- 1 call for memories
- N calls for N tables (table enrichment)
- = 2 + N total LLM calls

## Admin Endpoints

- `GET /health` - Health check
- `GET /constellation` - Interactive graph visualization
- `POST /admin/reset?confirm=true` - Reset database (drops vector indexes too)
- `GET /admin/stats` - System statistics

## Chat API Parameters

The `/v1/chat/completions` endpoint supports these parameters:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "agentic": false,
  "debug": true,
  "force_include_nodes": ["node-id-1", "node-id-2"],
  "force_exclude_nodes": ["node-id-3"]
}
```

**Parameters:**
- `agentic` (bool, default: false): Use v4 agentic pipeline with intent classification, CRAG, Self-RAG, hallucination detection, citations, and confidence calibration
- `debug` (bool, default: false): Include debug info in response
- `force_include_nodes` (list): Force specific nodes to be included in retrieval
- `force_exclude_nodes` (list): Exclude specific nodes from retrieval

**Debug response** (when `debug: true`):
- `retrieved_memories`: List of memories with `score`, `sources`, `included`
- `activated_concepts`: List of concepts with `activation`, `hop`, `included`
- `query_concepts`: Concepts extracted from the query
- `thresholds`: Activation and score thresholds used

**Agentic debug response** (when `agentic: true` and `debug: true`):
- All standard debug fields plus:
- `intent`: Classification result (decision, complexity, confidence)
- `crag`: Document grading (quality, relevant_count, corrective_action)
- `self_rag`: Validation result (support_level, iterations)
- `hallucination`: Claim verification (faithfulness_score, verified_claims)
- `confidence`: Calibration result (level, action, combined_score)

## Testing

```bash
uv run pytest                    # Run all tests
uv run pytest --cov=engram       # With coverage
uv run mypy src/engram           # Type checking
uv run ruff check src/engram     # Linting
```

## Test Set Evaluation (v4.2)

Evaluates Engram against human reference answers. Designed for incomplete/lazy annotations where:
- Human answer may be just a URL
- Human answer may be a short snippet
- Full answer is not required for evaluation

**Key insight:** Checks if human answer is CONTAINED in Engram answer (not equality).

```bash
# Basic usage (uses LLM from .env as judge)
uv run python -m engram.evaluation.evaluator test_data.csv

# Full options
uv run python -m engram.evaluation.evaluator test_data.csv \
  --engram-url http://localhost:7777/v1 \
  --engram-model engram \
  --judge-url http://localhost:11434/v1 \
  --judge-model qwen3:8b \
  --workers 8 \
  --output results.csv
```

**Default settings:**
- `--engram-url`: `http://localhost:7777/v1`
- `--judge-url`: Uses `LLM_BASE_URL` from `.env`
- `--judge-model`: Uses `LLM_MODEL` from `.env`

**Input CSV format:**
```csv
question;answer;url
Кто лид фронтенда?;Или;https://confluence.example.com/team
Как задеплоить?;;https://confluence.example.com/deploy
```

**Supported formats:**
- Delimiters: `;` or `,` (auto-detected)
- Encoding: UTF-8 with or without BOM
- Columns: `question,answer,url` or `Вопрос,Правильный ответ,Ссылка на правильный ответ`

**Output:**
- `{input}_results.csv` - Full results per question
- `{input}_results.summary.json` - Aggregate metrics

**Metrics:**
- `source_match`: Does Engram use same source as human URL?
- `key_info_match`: Does Engram CONTAIN key info from human answer?
- `relevance`: Does Engram answer the question?
- `no_contradiction`: No conflicts with human answer?
- `overall`: Weighted combination by answer type

**Answer types and weights:**
- `url_only`: source_match weighted highest (0.50)
- `short`: key_info_match weighted highest (0.40)
- `full`: balanced key_info_match + relevance
- `empty`: relevance weighted highest (0.70)

**Startup checks:** Before evaluation begins, the tool verifies:
- Engram API is available (health endpoint check)
- Judge LLM responds correctly (sends test prompt, checks for None content)

If either check fails, evaluation stops with a clear error and troubleshooting tips.

## Layout Computation

After ingestion, run layout computation to position nodes and compute hierarchical clusters:

```bash
# Standard (uses igraph if available, falls back to NetworkX)
uv run python scripts/compute_layout.py

# With GPU acceleration (requires NVIDIA GPU)
uv sync --extra gpu
uv run python scripts/compute_layout.py
```

**What it computes:**
1. **2D positions**: Force-directed layout for node placement
2. **5-level hierarchy**: Recursive subdivision clustering (L0-L4)
3. **Cluster merging**: Ensures ~sqrt(n)/3 L0 super-clusters

**Adaptive clustering (works on any graph size):**
- 100 nodes → ~3 L0 clusters
- 1,000 nodes → ~10 L0 clusters
- 10,000 nodes → ~33 L0 clusters
- 31,000 nodes → ~58 L0 clusters

**Layout backends (auto-detected):**
- **cuGraph (GPU)**: 100-1000x faster, requires `uv sync --extra gpu`
- **igraph (CPU)**: 5-20x faster than NetworkX
- **NetworkX (CPU)**: Fallback, slowest

## Calibration CLI

Test and tune v3 parameters:

```bash
uv run python scripts/calibrate.py config        # Show current configuration
uv run python scripts/calibrate.py test-russian  # Test Russian NLP preprocessing
uv run python scripts/calibrate.py test-reranker # Test BGE reranker
uv run python scripts/calibrate.py test-actr     # Test ACT-R activation calculations
uv run python scripts/calibrate.py benchmark     # Run retrieval benchmark
```

## v3 Configuration

New environment variables (add to `.env`):

```bash
# RRF Fusion
RRF_K=60                          # Higher = more uniform ranking

# Retrieval
RETRIEVAL_TOP_K=100               # Final number of memories sent to LLM
RETRIEVAL_BM25_K=200              # BM25 candidates before fusion
RETRIEVAL_VECTOR_K=200            # Vector search candidates before fusion

# Reranker
RERANKER_ENABLED=true             # Enable BGE cross-encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_CANDIDATES=150           # Candidates to rerank
RERANKER_DEVICE=cuda:0            # Single GPU device (avoids multi-GPU conflicts with vLLM)

# BM25
BM25_LEMMATIZE=true               # Russian lemmatization for BM25 search
BM25_REMOVE_STOPWORDS=true        # Remove Russian stopwords

# Embeddings
EMBEDDING_BATCH_SIZE=128          # Batch size for embedding (higher for powerful GPUs)
EMBEDDING_MULTI_GPU=false         # Multi-GPU support (disable if using vLLM on same GPUs)

# Parallelism (v3.6 combined extraction)
# v3.6: 1 LLM call per document (unified extraction)
# Formula: INGESTION_MAX_CONCURRENT = LLM_MAX_CONCURRENT
INGESTION_MAX_CONCURRENT=32       # Parallel document processing
LLM_MAX_CONCURRENT=32             # Parallel LLM calls (1 per doc)
LLM_TIMEOUT=300.0                 # LLM request timeout in seconds

# ACT-R Forgetting
ACTR_DECAY_D=0.5                  # Decay rate (higher = faster forgetting)
ACTR_THRESHOLD_TAU=-2.0           # Retrieval threshold

# Memory Status Thresholds
FORGETTING_DEPRIORITIZE_THRESHOLD=-1.0
FORGETTING_ARCHIVE_THRESHOLD=-2.5

# Contradiction Detection
CONTRADICTION_AUTO_RESOLVE_GAP=0.3

# v3.3 Quality Filtering
CHUNK_QUALITY_THRESHOLD=0.4       # Minimum quality score for chunks (0-1)
MIN_CHUNK_WORDS=20                # Minimum word count for quality chunks

# v3.3 MMR (Maximal Marginal Relevance)
MMR_ENABLED=true                  # Enable MMR for result diversity
MMR_LAMBDA=0.5                    # Balance: 1.0=relevance, 0.0=diversity
MMR_FETCH_K=200                   # Candidates to consider for MMR

# v3.3 Dynamic top_k
DYNAMIC_TOPK_ENABLED=false        # Adjust k based on query complexity
TOPK_SIMPLE=4                     # k for simple factoid queries
TOPK_MODERATE=6                   # k for moderate queries
TOPK_COMPLEX=8                    # k for complex analytical queries

# v4 Agentic RAG (all enabled by default when agentic=true)
INTENT_CLASSIFICATION_ENABLED=true
INTENT_USE_LLM_FALLBACK=true      # Use LLM for ambiguous intent classification

CRAG_ENABLED=true
CRAG_MIN_RELEVANT_RATIO=0.3       # Min ratio of relevant docs before query rewrite
CRAG_REWRITE_ON_FAILURE=true      # Rewrite query when all docs irrelevant

SELF_RAG_ENABLED=true
SELF_RAG_MAX_ITERATIONS=3         # Max regeneration attempts

NLI_ENABLED=true
NLI_USE_MODEL=false               # false=LLM fallback, true=mDeBERTa model
NLI_MODEL=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
NLI_ENTAILMENT_THRESHOLD=0.6      # Threshold for "supported" classification

CITATIONS_ENABLED=true
CITATIONS_VERIFY_NLI=true         # Verify citations with NLI

CONFIDENCE_HIGH_THRESHOLD=0.8
CONFIDENCE_MEDIUM_THRESHOLD=0.5
CONFIDENCE_LOW_THRESHOLD=0.3
CONFIDENCE_ABSTAIN_ON_VERY_LOW=true

IRCOT_ENABLED=true
IRCOT_MAX_STEPS=7                 # Max reasoning iterations
IRCOT_MAX_PARAGRAPHS=15           # Max paragraphs to accumulate

RAGAS_ENABLED=true
RAGAS_ASYNC_EVALUATION=true       # Run evaluation asynchronously

RESEARCH_MODE_ENABLED=true
RESEARCH_CHECKPOINT_DIR=/tmp/engram_research

# v4.3 Query Enrichment (works in BOTH standard and agentic modes)
QUERY_ENRICHMENT_ENABLED=false     # Master switch for v4.3 (opt-in)
QUERY_ENRICHMENT_USE_HYDE=false    # HyDE for complex queries (adds ~150ms)
QUERY_ENRICHMENT_MAX_VARIANTS=4    # Max query variants to generate

# v4.3 Standard Pipeline Enhancements (active when QUERY_ENRICHMENT_ENABLED=true)
STANDARD_INTENT_ENABLED=true       # Intent classification in standard mode
STANDARD_CRAG_ENABLED=true         # Light CRAG (grade top N docs)
STANDARD_CRAG_TOP_K=10             # Docs to grade (limits LLM calls)
STANDARD_CONFIDENCE_ENABLED=true   # Confidence calibration (no extra LLM)

# v4.3.1 Enrichment LLM (fast model for query enrichment)
ENRICHMENT_LLM_ENABLED=true        # Use dedicated LLM (falls back to main if unavailable)
ENRICHMENT_LLM_BASE_URL=http://localhost:11434/v1  # Ollama default
ENRICHMENT_LLM_MODEL=qwen3:4b      # Fast model for enrichment
ENRICHMENT_LLM_TIMEOUT=30.0        # Shorter timeout for fast model
ENRICHMENT_LLM_MAX_CONCURRENT=24   # High concurrency for small model

# v4.4 KB Summary LLM Enhancement
KB_SUMMARY_USE_LLM=false           # Opt-in LLM enhancement for KB summary
KB_SUMMARY_SAMPLE_SIZE=40          # Memories to sample for LLM analysis
KB_SUMMARY_MAX_QUESTIONS=7         # Max sample questions to generate
```

## Roadmap

### Completed

**Constellation:**
- [x] Pre-computed layout (igraph/cuGraph)
- [x] Viewport culling
- [x] WebGL rendering
- [x] Integrated chat with activation visualization
- [x] Query debug panel (show retrieved nodes, scores, sources)
- [x] Force include/exclude nodes for testing retrieval
- [x] Cluster metadata for zoomed-out view

**v3 Retrieval & Memory:**
- [x] RRF fusion with configurable k parameter
- [x] BGE cross-encoder reranking
- [x] Russian NLP (PyMorphy3 lemmatization, stopwords)
- [x] ACT-R base-level activation for memory decay
- [x] Memory status lifecycle (active → deprioritized → archived)
- [x] LLM-based contradiction detection (Russian prompts)
- [x] Calibration CLI

**v3.1 Sources & Ingestion:**
- [x] Source attribution in responses (title + URL)
- [x] Confluence metadata extraction and stripping
- [x] Ingestion `--clear` flag for database reset
- [x] Thread-safe embedding service
- [x] Configurable batch size and timeout

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

**v4 Agentic RAG:**
- [x] Intent classification (RETRIEVE/NO_RETRIEVE/CLARIFY) with pattern + LLM fallback
- [x] CRAG document grading (CORRECT/INCORRECT/AMBIGUOUS) with query rewrite
- [x] Self-RAG validation loop (max 3 iterations, abstain on failure)
- [x] NLI hallucination detection (LLM fallback or mDeBERTa model)
- [x] Inline citations `[1]`, `[2]` with optional NLI verification
- [x] Confidence calibration (HIGH/MEDIUM/LOW/VERY_LOW → respond/caveat/clarify/abstain)
- [x] IRCoT multi-hop reasoning (max 7 steps, convergence detection)
- [x] RAGAS evaluation metrics (faithfulness, relevancy, precision, recall)
- [x] Async research mode with subtasks and checkpointing
- [x] AgenticPipeline integrating all components
- [x] API `agentic` parameter (opt-in, default: false)

**v4.2 Test Set Evaluation:**
- [x] Test set evaluation for incomplete/lazy human reference answers
- [x] Answer type classification (url_only, short, full, empty)
- [x] Source URL matching with fuzzy path/pageId matching
- [x] LLM judge for containment-based evaluation (not equality)
- [x] Weighted scoring by answer type
- [x] CLI with parallel evaluation
- [x] Startup availability checks for Engram API and Judge LLM
- [x] CLI with parallel evaluation

**v4.3 Query Understanding & Enrichment:**
- [x] KB summary generation for domain awareness (run after ingestion)
- [x] Query understanding (type, complexity, entity detection)
- [x] BM25 expansion (synonyms, lemmas, domain terms, transliteration)
- [x] Semantic query rewrite (intent clarification)
- [x] Optional HyDE (Hypothetical Document Embedding) for complex queries
- [x] Multi-query retrieval (original + expanded + rewritten variants)
- [x] RRF fusion of all query variants
- [x] Agentic-lite for standard pipeline (intent, light CRAG, light confidence)
- [x] Works in BOTH standard and agentic modes via `QUERY_ENRICHMENT_ENABLED=true`

**v4.3.1 Enrichment LLM (Fast Model for Query Enrichment):**
- [x] Named LLM client registry (main, enrichment)
- [x] Dedicated fast LLM for query enrichment (Qwen3-4B)
- [x] Startup health check with automatic fallback to main LLM
- [x] Docker deployment support (vLLM on GPU 1)
- [x] Works with Ollama for local development
- [x] Thinking model support (uses reasoning field when content is empty)

**v4.4 LLM-Enhanced KB Summary:**
- [x] LLMKBSummaryEnhancer class using enrichment LLM (qwen3:4b)
- [x] Domain description generation ("This KB is about...")
- [x] Capabilities extraction (what questions can be answered)
- [x] Limitations detection (what's NOT covered)
- [x] Sample questions generation (HyDE-style examples)
- [x] Diverse memory sampling (popular, important, diverse types, random)
- [x] CLI flags: `--llm` / `--no-llm` for generate_kb_summary.py
- [x] Graceful fallback to base summary on LLM failure

### Planned
- [ ] Permanent graph fixes (add edges, boost weights, aliases)
- [ ] Test query collection (save/replay queries, pass/fail tracking)
- [ ] Spatial indexing (Neo4j point indexes)
- [ ] Feedback buttons in chat

## Version History

### Branch Timeline

| Branch | Python Files | Key Changes |
|--------|--------------|-------------|
| **v0.1** | 63 | Base version: ingestion, retrieval, reasoning, basic learning, constellation UI |
| **v0.2** | 64 | Torch dependency fixes |
| **v0.3** | 72 | UI loading animation, preprocessing modules added |
| **v0.4** | 85 | v3.0-v3.4: Russian NLP, reranker, quality filter, table/list extraction, person extraction |
| **main** | 103 | v3.5 + v4 + v4.2 + v4.3 + v4.4: Batch writes, hybrid search, agentic RAG, test set evaluation, query enrichment, LLM-enhanced KB summary |

### Module Additions by Version

**v0.1 → v0.3 (Base → UI improvements):**
- `preprocessing/normalizer.py` - Unicode normalization
- `preprocessing/table_parser.py` - Markdown table parsing

**v0.3 → v0.4 (Major feature additions):**
- `preprocessing/russian.py` - PyMorphy3 lemmatization
- `preprocessing/transliteration.py` - Russian/Latin query expansion
- `preprocessing/table_enricher.py` - LLM table enrichment
- `retrieval/reranker.py` - BGE cross-encoder
- `retrieval/fusion.py` - RRF fusion
- `retrieval/quality_filter.py` - Chunk scoring
- `ingestion/list_extractor.py` - Structure-aware lists
- `ingestion/person_extractor.py` - Natasha NER
- `learning/forgetting.py` - ACT-R memory decay
- `learning/contradiction.py` - LLM contradiction detection

**v0.4 → main (Current):**
- Batch Neo4j writes (UNWIND queries)
- Hybrid search (vector + BM25 + graph + RRF fusion)
- v4 agentic modules:
- `reasoning/intent_classifier.py` - Retrieval decision (RETRIEVE/NO_RETRIEVE/CLARIFY)
- `retrieval/crag.py` - Document grading + query rewrite
- `reasoning/self_rag.py` - Validation loop (max 3 iterations)
- `reasoning/hallucination_detector.py` - NLI claim verification
- `reasoning/citations.py` - Inline `[N]` citations
- `reasoning/confidence.py` - Calibration + abstention
- `reasoning/ircot.py` - Multi-hop reasoning (max 7 steps)
- `evaluation/ragas_eval.py` - RAGAS quality metrics
- `reasoning/research_agent.py` - Async research with checkpointing
- `reasoning/agentic_pipeline.py` - Main agentic orchestrator

**main (v4.2 additions):**
- `evaluation/evaluator.py` - Test set evaluation for incomplete reference answers

**main (v4.3 additions):**
- `indexing/kb_summary.py` - KB summary generation and storage
- `query/__init__.py` - Query module exports
- `query/enrichment.py` - Query enrichment pipeline (understanding, BM25 expansion, semantic rewrite, HyDE)
- `scripts/generate_kb_summary.py` - Post-ingestion KB summary generation

**main (v4.3.1 additions):**
- `ingestion/llm_client.py` - Named LLM client registry (main, enrichment) with automatic fallback and thinking model support

**main (v4.4 additions):**
- `indexing/kb_summary.py` - LLMKBSummaryEnhancer class, new KBSummary fields (domain_description, capabilities, limitations, sample_questions)

### Codebase Size

- **Core functionality:** ~4,000 LOC
- **v4 agentic modules:** ~3,000 LOC
- **v4.2 evaluation:** ~950 LOC
- **v4.3 query enrichment:** ~800 LOC
- **Optional features:** ~5,500 LOC
- **Total:** ~20,000 LOC (103 Python files)

### Key Checkpoints

- **v0.4**: Stable version with table enrichment (2 + N LLM calls per doc)
- **main**: Current with hybrid search + v4 agentic RAG (opt-in via `agentic: true`) + v4.2 test set evaluation + v4.3 query enrichment (opt-in via `QUERY_ENRICHMENT_ENABLED=true`) + v4.3.1 enrichment LLM (opt-in via `ENRICHMENT_LLM_ENABLED=true`) + v4.4 LLM-enhanced KB summary (opt-in via `KB_SUMMARY_USE_LLM=true`)
