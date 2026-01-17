"""FastAPI application for Engram API.

Provides OpenAI-compatible endpoints for chat completions,
plus feedback and admin endpoints for learning.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engram.api.routes import router
from engram.config import settings
from engram.storage.neo4j_client import Neo4jClient

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

    # Initialize database
    db = Neo4jClient()
    await db.connect()
    logger.info("Connected to Neo4j")

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
