"""FAISS vector index for fast similarity search.

v5: Provides efficient vector search for BGE-M3 embeddings.
Supports flat (exact) and IVF (approximate) index types.
"""

import logging
import threading
from pathlib import Path

import numpy as np

from engram.config import settings

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    FAISS-based vector index for fast similarity search.

    Features:
    - Inner product on normalized vectors (cosine similarity)
    - Flat (exact) and IVF (approximate) index types
    - Persistence to disk (save/load)
    - Thread-safe operations
    - Maps FAISS indices to chunk IDs
    """

    def __init__(
        self,
        dimensions: int = 1024,
        index_type: str | None = None,
        nlist: int = 100,
    ) -> None:
        """
        Initialize vector index.

        Args:
            dimensions: Vector dimensions (default 1024 for BGE-M3)
            index_type: "flat" (exact) or "ivf" (approximate)
            nlist: Number of clusters for IVF index
        """
        self.dimensions = dimensions
        self.index_type = index_type or settings.vector_index_type
        self.nlist = nlist

        self._index = None
        self._id_map: list[str] = []  # Maps FAISS index to chunk ID
        self._lock = threading.Lock()
        self._trained = False

    def _create_index(self):
        """Create FAISS index based on type."""
        import faiss

        if self.index_type == "flat":
            # Exact search using inner product (cosine on normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimensions)
            self._trained = True
        elif self.index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(self.dimensions)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimensions, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._trained = False
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        logger.info(f"Created FAISS {self.index_type} index (dim={self.dimensions})")

    def _ensure_index(self):
        """Ensure index is created."""
        if self._index is None:
            self._create_index()

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        with self._lock:
            if self._index is None:
                return 0
            return self._index.ntotal

    def add(
        self,
        embeddings: list[list[float]] | np.ndarray,
        chunk_ids: list[str],
    ) -> None:
        """
        Add vectors to the index.

        Args:
            embeddings: Vector embeddings (normalized)
            chunk_ids: Corresponding chunk IDs
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Embeddings and chunk_ids must have same length")

        if len(embeddings) == 0:
            return

        with self._lock:
            self._ensure_index()

            # Convert to numpy array
            if isinstance(embeddings, list):
                vectors = np.array(embeddings, dtype=np.float32)
            else:
                vectors = embeddings.astype(np.float32)

            # Ensure vectors are normalized (for cosine similarity via inner product)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-10)

            # Train IVF index if needed
            if self.index_type == "ivf" and not self._trained:
                if len(vectors) < self.nlist:
                    logger.warning(
                        f"Not enough vectors ({len(vectors)}) to train IVF index "
                        f"(need {self.nlist}). Using all for training."
                    )
                    train_vectors = vectors
                else:
                    train_vectors = vectors[:min(len(vectors), self.nlist * 50)]
                self._index.train(train_vectors)
                self._trained = True
                logger.info(f"Trained IVF index with {len(train_vectors)} vectors")

            # Add to index
            self._index.add(vectors)
            self._id_map.extend(chunk_ids)

            logger.debug(f"Added {len(chunk_ids)} vectors to index (total: {self._index.ntotal})")

    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 10,
        nprobe: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector (normalized)
            top_k: Number of results to return
            nprobe: Number of clusters to search (for IVF)

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by score desc
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []

            # Convert to numpy
            if isinstance(query_embedding, list):
                query = np.array([query_embedding], dtype=np.float32)
            else:
                query = query_embedding.reshape(1, -1).astype(np.float32)

            # Normalize query
            query = query / (np.linalg.norm(query) + 1e-10)

            # Set nprobe for IVF
            if self.index_type == "ivf":
                self._index.nprobe = nprobe

            # Search
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query, k)

            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._id_map):
                    chunk_id = self._id_map[idx]
                    results.append((chunk_id, float(score)))

            return results

    def search_batch(
        self,
        query_embeddings: list[list[float]] | np.ndarray,
        top_k: int = 10,
        nprobe: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """
        Batch search for similar vectors.

        Args:
            query_embeddings: Query vectors (normalized)
            top_k: Number of results per query
            nprobe: Number of clusters to search (for IVF)

        Returns:
            List of result lists, each containing (chunk_id, score) tuples
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return [[] for _ in query_embeddings]

            # Convert to numpy
            if isinstance(query_embeddings, list):
                queries = np.array(query_embeddings, dtype=np.float32)
            else:
                queries = query_embeddings.astype(np.float32)

            # Normalize queries
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / (norms + 1e-10)

            # Set nprobe for IVF
            if self.index_type == "ivf":
                self._index.nprobe = nprobe

            # Search
            k = min(top_k, self._index.ntotal)
            all_scores, all_indices = self._index.search(queries, k)

            # Build results
            all_results = []
            for scores, indices in zip(all_scores, all_indices):
                results = []
                for score, idx in zip(scores, indices):
                    if idx >= 0 and idx < len(self._id_map):
                        chunk_id = self._id_map[idx]
                        results.append((chunk_id, float(score)))
                all_results.append(results)

            return all_results

    def remove(self, chunk_ids: list[str]) -> int:
        """
        Remove vectors by chunk ID.

        Note: FAISS doesn't support efficient deletion, so we rebuild the index
        without the removed vectors. This is expensive for large indexes.

        Args:
            chunk_ids: Chunk IDs to remove

        Returns:
            Number of vectors removed
        """
        if not chunk_ids:
            return 0

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return 0

            remove_set = set(chunk_ids)
            keep_indices = [
                i for i, cid in enumerate(self._id_map)
                if cid not in remove_set
            ]

            if len(keep_indices) == len(self._id_map):
                return 0  # Nothing to remove

            removed_count = len(self._id_map) - len(keep_indices)

            # Reconstruct vectors for kept indices
            import faiss
            vectors = np.zeros((len(keep_indices), self.dimensions), dtype=np.float32)
            for new_idx, old_idx in enumerate(keep_indices):
                vectors[new_idx] = self._index.reconstruct(old_idx)

            # Update ID map
            new_id_map = [self._id_map[i] for i in keep_indices]

            # Recreate index
            self._create_index()
            self._id_map = []

            # Re-add vectors
            if len(vectors) > 0:
                self.add(vectors, new_id_map)

            logger.info(f"Removed {removed_count} vectors from index")
            return removed_count

    def clear(self) -> None:
        """Clear all vectors from the index."""
        with self._lock:
            self._create_index()
            self._id_map = []
            logger.info("Cleared vector index")

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save index to disk.

        Args:
            path: Directory to save to (default from settings)

        Returns:
            Path to saved index directory
        """
        import faiss

        path = Path(path) if path else Path(settings.vector_index_path)
        path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._index is None:
                raise ValueError("No index to save")

            # Save FAISS index
            index_path = path / "index.faiss"
            faiss.write_index(self._index, str(index_path))

            # Save ID map
            id_map_path = path / "id_map.txt"
            with open(id_map_path, "w") as f:
                for chunk_id in self._id_map:
                    f.write(f"{chunk_id}\n")

            # Save metadata
            meta_path = path / "metadata.txt"
            total = self._index.ntotal
            with open(meta_path, "w") as f:
                f.write(f"dimensions={self.dimensions}\n")
                f.write(f"index_type={self.index_type}\n")
                f.write(f"count={total}\n")
                f.write(f"trained={self._trained}\n")

            logger.info(f"Saved vector index to {path} ({total} vectors)")
            return path

    def load(self, path: str | Path | None = None) -> None:
        """
        Load index from disk.

        Args:
            path: Directory to load from (default from settings)
        """
        import faiss

        path = Path(path) if path else Path(settings.vector_index_path)

        with self._lock:
            # Load FAISS index
            index_path = path / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")

            self._index = faiss.read_index(str(index_path))

            # Load ID map
            id_map_path = path / "id_map.txt"
            self._id_map = []
            if id_map_path.exists():
                with open(id_map_path) as f:
                    self._id_map = [line.strip() for line in f if line.strip()]

            # Load metadata
            meta_path = path / "metadata.txt"
            if meta_path.exists():
                with open(meta_path) as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            if key == "trained":
                                self._trained = value.lower() == "true"

            logger.info(f"Loaded vector index from {path} ({self._index.ntotal} vectors)")

    def exists(self, path: str | Path | None = None) -> bool:
        """Check if a saved index exists at the given path."""
        path = Path(path) if path else Path(settings.vector_index_path)
        return (path / "index.faiss").exists()


# Global index instance
_global_index: VectorIndex | None = None
_global_index_lock = threading.Lock()


def get_vector_index() -> VectorIndex:
    """Get the global vector index (singleton)."""
    global _global_index
    with _global_index_lock:
        if _global_index is None:
            _global_index = VectorIndex()
        return _global_index


def load_or_create_index(path: str | Path | None = None) -> VectorIndex:
    """Load existing index or create new one."""
    index = get_vector_index()

    if path:
        load_path = Path(path)
    else:
        load_path = Path(settings.vector_index_path)

    if index.exists(load_path):
        index.load(load_path)
    else:
        logger.info(f"No existing index at {load_path}, starting fresh")

    return index
