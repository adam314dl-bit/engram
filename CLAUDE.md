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
- **Embeddings**: Local HuggingFace model via sentence-transformers
- **Database**: Neo4j (Docker)

## Key Files

- `src/engram/api/main.py` - FastAPI application entry point
- `src/engram/api/graph.py` - Memory graph visualization (serves `/admin/graph`)
- `src/engram/storage/neo4j_client.py` - Neo4j database client
- `src/engram/retrieval/` - Embeddings, spreading activation, hybrid search
- `src/engram/reasoning/` - Response synthesis, episode management

## Memory Graph Visualization

The graph visualization at `/admin/graph` displays the memory network:
- **Concepts** (blue): Atomic ideas from ingested documents
- **Semantic Memories** (purple): Facts and procedures linked to concepts
- **Episodic Memories** (orange): Past reasoning traces with outcomes

Implementation: `src/engram/api/graph.py` - uses force-graph library (2D canvas)
