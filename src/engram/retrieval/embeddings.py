"""Embedding service using sentence-transformers (local HuggingFace models)."""

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from engram.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using local models."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    def _get_model(self) -> SentenceTransformer:
        """Get or load the embedding model."""
        if self._model is None:
            import torch
            logger.info(f"Loading embedding model: {self.model_name}")

            # Determine target device
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model with low_cpu_mem_usage=False to avoid meta tensor issues
            # Some models (e.g., Giga-Embeddings-instruct) fail with meta tensors on GPU
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=target_device,
                model_kwargs={"low_cpu_mem_usage": False}
            )

            logger.info(f"Embedding model loaded on {target_device}. Dimensions: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._get_model().get_sentence_embedding_dimension()

    def embed_sync(self, text: str) -> list[float]:
        """Generate embedding for a single text (synchronous)."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch_sync(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (synchronous)."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(list(texts), convert_to_numpy=True)
        return embeddings.tolist()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_sync, text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (async)."""
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch_sync, list(texts))


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_batch(
    query: list[float],
    vectors: list[list[float]],
) -> list[float]:
    """Compute cosine similarity between query and multiple vectors."""
    if not vectors:
        return []
    q = np.array(query)
    v = np.array(vectors)
    # Normalize
    q_norm = q / np.linalg.norm(q)
    v_norms = v / np.linalg.norm(v, axis=1, keepdims=True)
    # Dot products
    similarities = np.dot(v_norms, q_norm)
    return similarities.tolist()


# Global service instance
_embedding_service: EmbeddingService | None = None


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service."""
    return EmbeddingService()


async def embed(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    service = get_embedding_service()
    return await service.embed(text)


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    service = get_embedding_service()
    return await service.embed_batch(texts)
