"""FastAPI application for Engram API.

Provides OpenAI-compatible endpoints for chat completions,
plus feedback and admin endpoints for learning.
"""

# IMPORTANT: Set HuggingFace offline mode BEFORE any imports
# This prevents HTTP requests during Jina reranker calls
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engram.api.graph import router as graph_router
from engram.api.routes import router
from engram.config import settings
from engram.retrieval.embeddings import preload_embedding_model
from engram.retrieval.reranker import preload_reranker
from engram.storage.neo4j_client import Neo4jClient

# v5: BGE-M3 and FAISS imports
from engram.embeddings.bge_service import preload_bge_model
from engram.embeddings.vector_index import load_or_create_index
from engram.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

# Global database connection
db: Neo4jClient | None = None


def get_db() -> Neo4jClient:
    """Get database connection."""
    if db is None:
        raise RuntimeError("Database not initialized")
    return db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown."""
    global db

    # Startup
    logger.info("Starting Engram API...")
    logger.info(f"Retrieval mode: {settings.retrieval_mode}")

    # Initialize database
    db = Neo4jClient()
    await db.connect()
    logger.info("Connected to Neo4j")

    # Preload embedding model only in hybrid mode (skip in bm25_graph mode)
    if settings.retrieval_mode != "bm25_graph":
        logger.info("Preloading embedding model...")
        preload_embedding_model()
        logger.info("Embedding model ready")

        # v5: Preload BGE-M3 for vector retrieval
        logger.info("Preloading BGE-M3 embedding model...")
        preload_bge_model()
        logger.info("BGE-M3 model ready")

        # v5: Load FAISS vector index and create VectorRetriever
        logger.info("Loading FAISS vector index...")
        index = load_or_create_index()
        if index.count > 0:
            logger.info(f"FAISS index loaded with {index.count} vectors")
            # Create VectorRetriever with the loaded index
            app.state.vector_retriever = VectorRetriever(
                db=db,
                vector_index=index,
            )
            logger.info("VectorRetriever initialized")
        else:
            logger.info("FAISS index empty (run migration to populate)")
            app.state.vector_retriever = None
    else:
        logger.info("BM25+Graph mode: skipping embedding model preload")
        app.state.vector_retriever = None

    # Preload reranker model (if enabled) to avoid delay on first query
    preload_reranker()

    # Store db in app state for routes
    app.state.db = db

    yield

    # Shutdown
    logger.info("Shutting down Engram API...")
    if db:
        await db.close()
        logger.info("Disconnected from Neo4j")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Engram",
        description="Cognitive-inspired knowledge system with dual memory architecture",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware for Open WebUI compatibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)
    app.include_router(graph_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "engram.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
