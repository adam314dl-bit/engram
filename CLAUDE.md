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
