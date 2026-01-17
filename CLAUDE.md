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

## Project Setup

```bash
uv sync              # Install deps
uv run pytest        # Run tests
uv run python -m engram.api.main  # Run API
```

## Architecture

- **LLM**: Remote OpenAI-compatible endpoint (configured via LLM_BASE_URL)
- **Embeddings**: Local HuggingFace model via sentence-transformers (2048 dims with Giga-Embeddings)
- **Database**: Neo4j (Docker)
- **HTTP Client**: Uses `requests` library (not httpx)

## Key Files

- `src/engram/api/main.py` - FastAPI application entry point
- `src/engram/api/routes.py` - API routes (chat, feedback, admin)
- `src/engram/api/graph.py` - Memory graph visualization (serves `/admin/graph`)
- `src/engram/storage/neo4j_client.py` - Neo4j database client
- `src/engram/retrieval/` - Embeddings, spreading activation, hybrid search
- `src/engram/reasoning/` - Response synthesis, episode management
- `src/engram/ingestion/` - Document parsing, concept/memory extraction (Russian prompts)

## Memory Graph Visualization

The graph visualization at `/admin/graph` displays the memory network with a cyberpunk dark theme.

**Node Types (soft colors):**
- **Concepts** (teal `#5eead4`): Atomic ideas from ingested documents
- **Semantic Memories** (purple `#a78bfa`): Facts and procedures linked to concepts
- **Episodic Memories** (pink `#f472b6`): Past reasoning traces with outcomes

**Features:**
- Cluster detection (Label Propagation algorithm)
- Path finder between any two nodes
- Importance filter slider
- Type filtering via legend buttons
- Search by name/content
- Node info panel with connections

**Implementation:** `src/engram/api/graph.py` - uses force-graph library (2D canvas)

## Ingestion

Extraction prompts are in Russian for better quality with Russian content.

```bash
# Ingest from default directory
uv run python scripts/run_ingestion.py

# Ingest from custom directory (recursive)
uv run python scripts/run_ingestion.py /path/to/docs
```

Supported formats: `.md`, `.txt`, `.markdown`

## Admin Endpoints

- `GET /health` - Health check
- `GET /admin/graph` - Interactive graph visualization
- `POST /admin/reset?confirm=true` - Reset database (drops vector indexes too)
- `GET /admin/stats` - System statistics

## Testing

```bash
uv run pytest                    # Run all tests
uv run pytest --cov=engram       # With coverage
uv run mypy src/engram           # Type checking
uv run ruff check src/engram     # Linting
```
