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

## Tech Stack

- Python 3.11+
- Neo4j 5.15+ (graph database)
- FastAPI (API)
- sentence-transformers (embeddings)
- Ollama/Qwen3 (LLM)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- uv (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd engram
```

2. Start services:
```bash
docker-compose up -d
```

3. Install dependencies:
```bash
uv sync
```

4. Copy environment file:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run tests:
```bash
uv run pytest
```

### Using Ollama for LLM

If using Ollama on Mac (not in Docker):

```bash
# Install Ollama (if not already)
brew install ollama

# Pull the model
ollama pull qwen3:8b

# Start Ollama server
ollama serve
```

The default configuration expects Ollama at `http://localhost:11434`.

## Project Structure

```
engram/
├── src/engram/
│   ├── models/          # Data models (Concept, Memory, etc.)
│   ├── ingestion/       # Document parsing and extraction
│   ├── storage/         # Neo4j client and schema
│   ├── retrieval/       # Embeddings and search
│   ├── reasoning/       # Response synthesis
│   ├── learning/        # Feedback and consolidation
│   └── api/            # FastAPI endpoints
├── tests/
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── fixtures/       # Test data
└── scripts/            # Utility scripts
```

## Development

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=engram

# Specific test file
uv run pytest tests/unit/test_models.py
```

### Type Checking

```bash
uv run mypy src/engram
```

### Linting

```bash
uv run ruff check src/engram
```

### Seeding Test Data

```bash
# Ensure Neo4j is running
uv run python scripts/seed_test_data.py
```

## Implementation Phases

- [x] **Phase 1**: Foundation (models, parsing, basic storage)
- [ ] **Phase 2**: Spreading Activation
- [ ] **Phase 3**: Hybrid Retrieval
- [ ] **Phase 4**: Reasoning & Synthesis
- [ ] **Phase 5**: Learning System
- [ ] **Phase 6**: API & Integration

## License

MIT
