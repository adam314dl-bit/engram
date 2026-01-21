"""Embedding service using sentence-transformers (local HuggingFace models)."""

import asyncio
import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

from engram.config import settings

logger = logging.getLogger(__name__)

# Shared thread pool for async operations (sized for high-core servers)
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Get shared thread pool executor."""
    global _executor
    with _executor_lock:
        if _executor is None:
            # Use more workers for high-core servers
            max_workers = min(32, (settings.ingestion_max_concurrent * 2))
            _executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Created thread pool with {max_workers} workers")
        return _executor


class EmbeddingService:
    """Service for generating text embeddings using local models.

    Thread-safe singleton that loads the model once and reuses it.
    """

    _instance: "EmbeddingService | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls, model_name: str | None = None) -> "EmbeddingService":
        """Ensure only one instance exists (singleton pattern)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        # Skip re-initialization if already done
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._pool = None  # Multi-process pool for multi-GPU
        self._initialized = True

    def load_model(self) -> None:
        """Explicitly load the embedding model. Thread-safe."""
        with self._model_lock:
            if self._model is not None:
                return  # Already loaded

            import torch
            logger.info(f"Loading embedding model: {self.model_name}")

            # Check GPU availability
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0

            # Determine target device
            target_device = "cuda" if cuda_available else "cpu"

            # Load model with low_cpu_mem_usage=False to avoid meta tensor issues
            # Some models (e.g., Giga-Embeddings-instruct) fail with meta tensors on GPU
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=target_device,
                model_kwargs={"low_cpu_mem_usage": False}
            )

            # Setup multi-GPU pool if enabled and multiple GPUs available
            if settings.embedding_multi_gpu and gpu_count > 1:
                target_gpus = settings.embedding_gpu_count if settings.embedding_gpu_count > 0 else gpu_count
                target_gpus = min(target_gpus, gpu_count)
                target_devices = [f"cuda:{i}" for i in range(target_gpus)]
                logger.info(f"Starting multi-GPU pool with {target_gpus} GPUs: {target_devices}")
                self._pool = self._model.start_multi_process_pool(target_devices)
                logger.info(f"Multi-GPU pool started successfully")
            else:
                logger.info(f"Single GPU mode (multi_gpu={settings.embedding_multi_gpu}, gpus={gpu_count})")

            logger.info(
                f"Embedding model loaded on {target_device}. "
                f"Dimensions: {self._model.get_sentence_embedding_dimension()}, "
                f"Batch size: {settings.embedding_batch_size}"
            )

    def _get_model(self) -> SentenceTransformer:
        """Get the embedding model, loading it if necessary."""
        if self._model is None:
            self.load_model()
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
        """Generate embeddings for multiple texts (synchronous).

        Uses multi-GPU pool if available, otherwise single GPU with batching.
        """
        if not texts:
            return []
        model = self._get_model()
        texts_list = list(texts)

        # Replace empty strings with placeholder (some models fail on empty input)
        texts_cleaned = [t if t.strip() else " " for t in texts_list]

        # Use multi-GPU pool if available
        if self._pool is not None:
            embeddings = model.encode_multi_process(
                texts_cleaned,
                self._pool,
                batch_size=settings.embedding_batch_size,
            )
        else:
            # Single GPU with configured batch size
            embeddings = model.encode(
                texts_cleaned,
                convert_to_numpy=True,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=len(texts_cleaned) > 100,  # Show progress for large batches
            )

        result = embeddings.tolist()

        # Sanity check: ensure output count matches input count
        if len(result) != len(texts_list):
            logger.error(
                f"Embedding count mismatch: got {len(result)} embeddings for {len(texts_list)} texts"
            )
            # Pad or truncate to match (fallback, shouldn't happen)
            if len(result) < len(texts_list):
                # Pad with zero vectors
                dim = len(result[0]) if result else self.dimensions
                while len(result) < len(texts_list):
                    result.append([0.0] * dim)
            else:
                result = result[:len(texts_list)]

        return result

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_executor(), self.embed_sync, text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (async)."""
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_executor(), self.embed_batch_sync, list(texts))


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


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service (singleton)."""
    return EmbeddingService()


def shutdown_embedding_service() -> None:
    """Shutdown the embedding service and release GPU resources."""
    service = get_embedding_service()
    if service._pool is not None:
        logger.info("Stopping multi-GPU pool...")
        service._model.stop_multi_process_pool(service._pool)
        service._pool = None
        logger.info("Multi-GPU pool stopped")


def preload_embedding_model() -> None:
    """Preload the embedding model at startup. Call this during app initialization."""
    service = get_embedding_service()
    service.load_model()


async def embed(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    service = get_embedding_service()
    return await service.embed(text)


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    service = get_embedding_service()
    return await service.embed_batch(texts)
