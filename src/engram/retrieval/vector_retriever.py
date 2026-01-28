"""Vector retriever using BGE-M3 embeddings and FAISS index.

v5: Provides dense vector retrieval as a signal for hybrid search.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from engram.config import settings
from engram.embeddings.bge_service import BGEEmbeddingService, get_bge_embedding_service
from engram.embeddings.vector_index import VectorIndex, get_vector_index, load_or_create_index
from engram.models import SemanticMemory
from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    """Result from vector retrieval."""

    chunk_id: str
    score: float
    text: str = ""
    metadata: dict | None = None


class VectorRetriever:
    """
    Vector retriever using BGE-M3 embeddings and FAISS index.

    Features:
    - Embeds queries using BGE-M3 (1024-dim)
    - Searches FAISS index for similar vectors
    - Fetches full chunk content from Neo4j
    - Integrates with hybrid search pipeline
    """

    def __init__(
        self,
        db: Neo4jClient,
        embedding_service: BGEEmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        """
        Initialize vector retriever.

        Args:
            db: Neo4j client for fetching chunk content
            embedding_service: BGE-M3 embedding service
            vector_index: FAISS vector index
            index_path: Path to load index from (if not provided via vector_index)
        """
        self.db = db
        self.embeddings = embedding_service or get_bge_embedding_service()
        self._index = vector_index
        self._index_path = index_path

    def _get_index(self) -> VectorIndex:
        """Get vector index, loading if necessary."""
        if self._index is None:
            self._index = load_or_create_index(self._index_path)
        return self._index

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[VectorResult]:
        """
        Retrieve relevant chunks using vector similarity.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of VectorResult sorted by similarity score desc
        """
        top_k = top_k or settings.retrieval_vector_k

        # Embed query
        query_embedding = await self.embeddings.embed(query)

        # Search index
        index = self._get_index()
        search_results = index.search(query_embedding, top_k=top_k)

        if not search_results:
            return []

        # Fetch chunk content from Neo4j
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        memories = await self.db.get_memories_by_ids(chunk_ids)

        # Build result map
        memory_map = {m.id: m for m in memories}

        # Build results with content
        results = []
        for chunk_id, score in search_results:
            memory = memory_map.get(chunk_id)
            if memory:
                results.append(VectorResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=memory.content or "",
                    metadata=memory.metadata,
                ))
            else:
                # Memory not found in Neo4j (index out of sync)
                logger.warning(f"Memory {chunk_id} not found in Neo4j (index out of sync)")
                results.append(VectorResult(
                    chunk_id=chunk_id,
                    score=score,
                    text="",
                    metadata=None,
                ))

        return results

    async def retrieve_with_embedding(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[VectorResult]:
        """
        Retrieve using pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return

        Returns:
            List of VectorResult sorted by similarity score desc
        """
        top_k = top_k or settings.retrieval_vector_k

        # Search index
        index = self._get_index()
        search_results = index.search(query_embedding, top_k=top_k)

        if not search_results:
            return []

        # Fetch chunk content from Neo4j
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        memories = await self.db.get_memories_by_ids(chunk_ids)

        # Build result map
        memory_map = {m.id: m for m in memories}

        # Build results
        results = []
        for chunk_id, score in search_results:
            memory = memory_map.get(chunk_id)
            if memory:
                results.append(VectorResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=memory.content or "",
                    metadata=memory.metadata,
                ))

        return results

    async def retrieve_memories(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[SemanticMemory, float]]:
        """
        Retrieve memories using vector similarity.

        Compatible interface with existing Neo4j vector search.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (SemanticMemory, score) tuples
        """
        top_k = top_k or settings.retrieval_vector_k

        # Embed query
        query_embedding = await self.embeddings.embed(query)

        return await self.retrieve_memories_with_embedding(query_embedding, top_k)

    async def retrieve_memories_with_embedding(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[tuple[SemanticMemory, float]]:
        """
        Retrieve memories using pre-computed embedding.

        Compatible interface with existing Neo4j vector search.

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return

        Returns:
            List of (SemanticMemory, score) tuples
        """
        top_k = top_k or settings.retrieval_vector_k

        # Search index
        index = self._get_index()
        search_results = index.search(query_embedding, top_k=top_k)

        if not search_results:
            return []

        # Fetch memories from Neo4j
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        memories = await self.db.get_memories_by_ids(chunk_ids)

        # Build result map preserving order
        memory_map = {m.id: m for m in memories}
        results = []
        for chunk_id, score in search_results:
            memory = memory_map.get(chunk_id)
            if memory:
                results.append((memory, score))

        return results

    def add_to_index(
        self,
        memories: list[SemanticMemory],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add memories to the vector index.

        Args:
            memories: Memories to add
            embeddings: Corresponding BGE-M3 embeddings
        """
        if len(memories) != len(embeddings):
            raise ValueError("Memories and embeddings must have same length")

        index = self._get_index()
        chunk_ids = [m.id for m in memories]
        index.add(embeddings, chunk_ids)

    def save_index(self, path: str | Path | None = None) -> Path:
        """Save the vector index to disk."""
        index = self._get_index()
        return index.save(path)

    def load_index(self, path: str | Path | None = None) -> None:
        """Load the vector index from disk."""
        index = self._get_index()
        index.load(path)

    @property
    def index_count(self) -> int:
        """Number of vectors in the index."""
        return self._get_index().count


# Global retriever instance
_global_retriever: VectorRetriever | None = None


def get_vector_retriever(db: Neo4jClient) -> VectorRetriever:
    """Get the global vector retriever (creates if needed)."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = VectorRetriever(db=db)
    return _global_retriever


def initialize_vector_retriever(
    db: Neo4jClient,
    index_path: str | Path | None = None,
) -> VectorRetriever:
    """Initialize and load the global vector retriever."""
    global _global_retriever
    _global_retriever = VectorRetriever(db=db, index_path=index_path)

    # Load index if it exists
    index = _global_retriever._get_index()
    if index.count > 0:
        logger.info(f"Vector retriever initialized with {index.count} vectors")
    else:
        logger.info("Vector retriever initialized (empty index)")

    return _global_retriever
