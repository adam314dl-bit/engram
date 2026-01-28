"""Embeddings module for vector representations."""

from engram.embeddings.bge_service import BGEEmbeddingService, get_bge_embedding_service
from engram.embeddings.vector_index import VectorIndex

__all__ = ["BGEEmbeddingService", "get_bge_embedding_service", "VectorIndex"]
