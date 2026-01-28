"""BGE-M3 embedding service using FlagEmbedding.

v5: Provides 1024-dimensional dense embeddings for vector retrieval.
BGE-M3 is multilingual and supports Russian well without query prefixes.
"""

import asyncio
import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from engram.config import settings

logger = logging.getLogger(__name__)

# Shared thread pool for async operations
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Get shared thread pool executor."""
    global _executor
    with _executor_lock:
        if _executor is None:
            max_workers = min(32, (settings.ingestion_max_concurrent * 2))
            _executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Created BGE thread pool with {max_workers} workers")
        return _executor


class BGEEmbeddingService:
    """
    BGE-M3 embedding service for vector retrieval.

    Features:
    - 1024-dimensional dense embeddings
    - Multilingual support (Russian, English, etc.)
    - No query prefixes needed (symmetric model)
    - FP16 support for faster inference
    - Batch processing with configurable size
    """

    _instance: "BGEEmbeddingService | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "BGEEmbeddingService":
        """Ensure only one instance exists (singleton pattern)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        # Skip re-initialization if already done
        if getattr(self, "_initialized", False):
            return

        self._model = None
        self._model_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._initialized = True

    def load_model(self) -> None:
        """Load BGE-M3 model. Thread-safe."""
        with self._model_lock:
            if self._model is not None:
                return

            import torch
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading BGE-M3 model: {settings.bge_model_name}")

            # Determine device
            device = settings.bge_device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"

            # Load model with FP16 if enabled (use devices=[device] to prevent multi-GPU pooling)
            self._model = BGEM3FlagModel(
                settings.bge_model_name,
                use_fp16=settings.bge_use_fp16 and device != "cpu",
                devices=[device],  # Single device list prevents multi-process pool
            )

            logger.info(f"BGE-M3 model loaded on {device} (fp16={settings.bge_use_fp16})")

    def _get_model(self):
        """Get the model, loading if necessary."""
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions (BGE-M3 uses 1024)."""
        return 1024

    def embed_sync(self, text: str) -> list[float]:
        """Generate embedding for a single text (synchronous)."""
        return self.embed_batch_sync([text])[0]

    def embed_batch_sync(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (synchronous).

        Args:
            texts: Texts to embed
            show_progress: Show progress bar (for large batches)

        Returns:
            List of 1024-dimensional embedding vectors
        """
        if not texts:
            return []

        model = self._get_model()
        texts_list = list(texts)

        # Replace empty strings with placeholder
        texts_cleaned = [t if t.strip() else " " for t in texts_list]

        # Serialize encode calls
        with self._encode_lock:
            # BGE-M3 encode returns dict with 'dense_vecs'
            result = model.encode(
                texts_cleaned,
                batch_size=settings.bge_batch_size,
                max_length=settings.bge_max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

            # Extract dense vectors
            dense_vecs = result["dense_vecs"]

            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            dense_vecs = dense_vecs / (norms + 1e-10)

            embeddings = dense_vecs.tolist()

        # Sanity check
        if len(embeddings) != len(texts_list):
            logger.error(
                f"BGE embedding count mismatch: got {len(embeddings)} for {len(texts_list)} texts"
            )
            # Pad with zeros if needed
            dim = len(embeddings[0]) if embeddings else self.dimensions
            while len(embeddings) < len(texts_list):
                embeddings.append([0.0] * dim)
            embeddings = embeddings[:len(texts_list)]

        return embeddings

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_executor(), self.embed_sync, text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (async)."""
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            get_executor(), self.embed_batch_sync, list(texts)
        )


@lru_cache(maxsize=1)
def get_bge_embedding_service() -> BGEEmbeddingService:
    """Get the global BGE embedding service (singleton)."""
    return BGEEmbeddingService()


def preload_bge_model() -> None:
    """Preload the BGE-M3 model at startup."""
    service = get_bge_embedding_service()
    service.load_model()


async def embed_bge(text: str) -> list[float]:
    """Convenience function to embed a single text with BGE-M3."""
    service = get_bge_embedding_service()
    return await service.embed(text)


async def embed_batch_bge(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts with BGE-M3."""
    service = get_bge_embedding_service()
    return await service.embed_batch(texts)
