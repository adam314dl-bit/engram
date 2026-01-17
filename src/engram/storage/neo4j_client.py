"""Neo4j client for graph database operations."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from engram.config import settings
from engram.models import (
    Concept,
    ConceptRelation,
    Document,
    EpisodicMemory,
    SemanticMemory,
)
from engram.storage.schema import get_all_schema_queries

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async Neo4j client for Engram graph operations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connectivity
            try:
                await self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except ServiceUnavailable as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    async def close(self) -> None:
        """Close the database connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if self._driver is None:
            await self.connect()
        assert self._driver is not None
        async with self._driver.session(database=self.database) as session:
            yield session

    async def setup_schema(self) -> None:
        """Create all constraints and indexes."""
        queries = get_all_schema_queries()
        async with self.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                    logger.debug(f"Executed schema query: {query[:50]}...")
                except Exception as e:
                    # Some indexes might already exist, that's ok
                    logger.warning(f"Schema query warning: {e}")
        logger.info("Schema setup completed")

    # ==========================================================================
    # Concept operations
    # ==========================================================================

    async def save_concept(self, concept: Concept) -> None:
        """Save or update a concept node."""
        query = """
        MERGE (c:Concept {id: $id})
        SET c += $props
        """
        props = concept.to_dict()
        props.pop("id")  # Don't duplicate id in props
        async with self.session() as session:
            await session.run(query, id=concept.id, props=props)

    async def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept by ID."""
        query = "MATCH (c:Concept {id: $id}) RETURN c"
        async with self.session() as session:
            result = await session.run(query, id=concept_id)
            record = await result.single()
            if record:
                return Concept.from_dict(dict(record["c"]))
            return None

    async def get_concept_by_name(self, name: str) -> Concept | None:
        """Get a concept by name (case-insensitive)."""
        query = "MATCH (c:Concept) WHERE toLower(c.name) = toLower($name) RETURN c"
        async with self.session() as session:
            result = await session.run(query, name=name)
            record = await result.single()
            if record:
                return Concept.from_dict(dict(record["c"]))
            return None

    async def get_concept_neighbors(
        self, concept_id: str, limit: int = 50
    ) -> list[tuple[Concept, ConceptRelation]]:
        """Get neighboring concepts with their relationship info."""
        query = """
        MATCH (c:Concept {id: $id})-[r:RELATED_TO]->(neighbor:Concept)
        RETURN neighbor, r
        LIMIT $limit
        """
        neighbors: list[tuple[Concept, ConceptRelation]] = []
        async with self.session() as session:
            result = await session.run(query, id=concept_id, limit=limit)
            async for record in result:
                concept = Concept.from_dict(dict(record["neighbor"]))
                rel_data = dict(record["r"])
                relation = ConceptRelation(
                    source_id=concept_id,
                    target_id=concept.id,
                    relation_type=rel_data.get("type", "related_to"),
                    weight=rel_data.get("weight", 0.5),
                    edge_embedding=rel_data.get("edge_embedding"),
                )
                neighbors.append((concept, relation))
        return neighbors

    async def save_concept_relation(self, relation: ConceptRelation) -> None:
        """Save or update a concept relationship."""
        query = """
        MATCH (source:Concept {id: $source_id})
        MATCH (target:Concept {id: $target_id})
        MERGE (source)-[r:RELATED_TO]->(target)
        SET r.type = $relation_type,
            r.weight = $weight,
            r.edge_embedding = $edge_embedding,
            r.co_occurrence_count = $co_occurrence_count,
            r.last_used = $last_used
        """
        async with self.session() as session:
            await session.run(query, **relation.to_dict())

    async def update_concept_activation(self, concept_id: str) -> None:
        """Increment activation count and update timestamp."""
        query = """
        MATCH (c:Concept {id: $id})
        SET c.activation_count = c.activation_count + 1,
            c.last_activated = datetime()
        """
        async with self.session() as session:
            await session.run(query, id=concept_id)

    # ==========================================================================
    # Semantic memory operations
    # ==========================================================================

    async def save_semantic_memory(self, memory: SemanticMemory) -> None:
        """Save or update a semantic memory."""
        query = """
        MERGE (s:SemanticMemory {id: $id})
        SET s += $props
        """
        props = memory.to_dict()
        props.pop("id")
        async with self.session() as session:
            await session.run(query, id=memory.id, props=props)

    async def get_semantic_memory(self, memory_id: str) -> SemanticMemory | None:
        """Get a semantic memory by ID."""
        query = "MATCH (s:SemanticMemory {id: $id}) RETURN s"
        async with self.session() as session:
            result = await session.run(query, id=memory_id)
            record = await result.single()
            if record:
                return SemanticMemory.from_dict(dict(record["s"]))
            return None

    async def link_memory_to_concept(self, memory_id: str, concept_id: str) -> None:
        """Create ABOUT relationship between memory and concept."""
        query = """
        MATCH (s:SemanticMemory {id: $memory_id})
        MATCH (c:Concept {id: $concept_id})
        MERGE (s)-[:ABOUT]->(c)
        """
        async with self.session() as session:
            await session.run(query, memory_id=memory_id, concept_id=concept_id)

    async def link_memory_to_document(self, memory_id: str, doc_id: str) -> None:
        """Create EXTRACTED_FROM relationship between memory and document."""
        query = """
        MATCH (s:SemanticMemory {id: $memory_id})
        MATCH (d:Document {id: $doc_id})
        MERGE (s)-[:EXTRACTED_FROM]->(d)
        """
        async with self.session() as session:
            await session.run(query, memory_id=memory_id, doc_id=doc_id)

    async def get_memories_for_concepts(
        self,
        concept_ids: list[str],
        min_activation: float = 0.0,
        exclude_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[SemanticMemory]:
        """Get memories connected to given concepts."""
        exclude_ids = exclude_ids or []
        query = """
        MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
        WHERE c.id IN $concept_ids
          AND s.status = 'active'
          AND NOT s.id IN $exclude_ids
        WITH s, count(c) as concept_match_count
        ORDER BY concept_match_count DESC, s.importance DESC
        LIMIT $limit
        RETURN s
        """
        memories: list[SemanticMemory] = []
        async with self.session() as session:
            result = await session.run(
                query,
                concept_ids=concept_ids,
                exclude_ids=exclude_ids,
                limit=limit,
            )
            async for record in result:
                memories.append(SemanticMemory.from_dict(dict(record["s"])))
        return memories

    async def update_memory_access(self, memory_id: str) -> None:
        """Update access count and timestamp for a memory."""
        query = """
        MATCH (s:SemanticMemory {id: $id})
        SET s.access_count = s.access_count + 1,
            s.last_accessed = datetime()
        """
        async with self.session() as session:
            await session.run(query, id=memory_id)

    async def update_memory_strength(self, memory_id: str, new_strength: float) -> None:
        """Update memory strength (SM-2)."""
        query = """
        MATCH (s:SemanticMemory {id: $id})
        SET s.strength = $strength
        """
        async with self.session() as session:
            await session.run(query, id=memory_id, strength=new_strength)

    # ==========================================================================
    # Episodic memory operations
    # ==========================================================================

    async def save_episodic_memory(self, episode: EpisodicMemory) -> None:
        """Save or update an episodic memory."""
        query = """
        MERGE (e:EpisodicMemory {id: $id})
        SET e += $props
        """
        props = episode.to_dict()
        props.pop("id")
        async with self.session() as session:
            await session.run(query, id=episode.id, props=props)

    async def get_episodic_memory(self, episode_id: str) -> EpisodicMemory | None:
        """Get an episodic memory by ID."""
        query = "MATCH (e:EpisodicMemory {id: $id}) RETURN e"
        async with self.session() as session:
            result = await session.run(query, id=episode_id)
            record = await result.single()
            if record:
                return EpisodicMemory.from_dict(dict(record["e"]))
            return None

    async def link_episode_to_concept(self, episode_id: str, concept_id: str) -> None:
        """Create ACTIVATED relationship."""
        query = """
        MATCH (e:EpisodicMemory {id: $episode_id})
        MATCH (c:Concept {id: $concept_id})
        MERGE (e)-[:ACTIVATED]->(c)
        """
        async with self.session() as session:
            await session.run(query, episode_id=episode_id, concept_id=concept_id)

    async def link_episode_to_memory(self, episode_id: str, memory_id: str) -> None:
        """Create USED relationship."""
        query = """
        MATCH (e:EpisodicMemory {id: $episode_id})
        MATCH (s:SemanticMemory {id: $memory_id})
        MERGE (e)-[:USED]->(s)
        """
        async with self.session() as session:
            await session.run(query, episode_id=episode_id, memory_id=memory_id)

    async def get_recent_episodes(
        self, hours: int = 24, limit: int = 100
    ) -> list[EpisodicMemory]:
        """Get recent episodic memories."""
        query = """
        MATCH (e:EpisodicMemory)
        WITH e, datetime(e.created_at) as created
        WHERE created > datetime() - duration({hours: $hours})
        RETURN e
        ORDER BY created DESC
        LIMIT $limit
        """
        episodes: list[EpisodicMemory] = []
        async with self.session() as session:
            result = await session.run(query, hours=hours, limit=limit)
            async for record in result:
                episodes.append(EpisodicMemory.from_dict(dict(record["e"])))
        return episodes

    # ==========================================================================
    # Document operations
    # ==========================================================================

    async def save_document(self, document: Document) -> None:
        """Save or update a document."""
        query = """
        MERGE (d:Document {id: $id})
        SET d += $props
        """
        props = document.to_dict()
        props.pop("id")
        async with self.session() as session:
            await session.run(query, id=document.id, props=props)

    async def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        query = "MATCH (d:Document {id: $id}) RETURN d"
        async with self.session() as session:
            result = await session.run(query, id=doc_id)
            record = await result.single()
            if record:
                return Document.from_dict(dict(record["d"]))
            return None

    async def get_pending_documents(self, limit: int = 100) -> list[Document]:
        """Get documents pending processing."""
        query = """
        MATCH (d:Document {status: 'pending'})
        ORDER BY d.created_at ASC
        LIMIT $limit
        RETURN d
        """
        docs: list[Document] = []
        async with self.session() as session:
            result = await session.run(query, limit=limit)
            async for record in result:
                docs.append(Document.from_dict(dict(record["d"])))
        return docs

    # ==========================================================================
    # Vector search operations
    # ==========================================================================

    async def vector_search_concepts(
        self, embedding: list[float], k: int = 10
    ) -> list[tuple[Concept, float]]:
        """Search concepts by vector similarity."""
        query = """
        CALL db.index.vector.queryNodes('concept_embeddings', $k, $embedding)
        YIELD node, score
        RETURN node, score
        """
        results: list[tuple[Concept, float]] = []
        async with self.session() as session:
            result = await session.run(query, k=k, embedding=embedding)
            async for record in result:
                concept = Concept.from_dict(dict(record["node"]))
                results.append((concept, record["score"]))
        return results

    async def vector_search_memories(
        self, embedding: list[float], k: int = 20
    ) -> list[tuple[SemanticMemory, float]]:
        """Search semantic memories by vector similarity."""
        query = """
        CALL db.index.vector.queryNodes('semantic_embeddings', $k, $embedding)
        YIELD node, score
        WHERE node.status = 'active'
        RETURN node, score
        """
        results: list[tuple[SemanticMemory, float]] = []
        async with self.session() as session:
            result = await session.run(query, k=k, embedding=embedding)
            async for record in result:
                memory = SemanticMemory.from_dict(dict(record["node"]))
                results.append((memory, record["score"]))
        return results

    async def vector_search_episodes(
        self, embedding: list[float], k: int = 10
    ) -> list[tuple[EpisodicMemory, float]]:
        """Search episodic memories by vector similarity."""
        query = """
        CALL db.index.vector.queryNodes('episodic_embeddings', $k, $embedding)
        YIELD node, score
        RETURN node, score
        """
        results: list[tuple[EpisodicMemory, float]] = []
        async with self.session() as session:
            result = await session.run(query, k=k, embedding=embedding)
            async for record in result:
                episode = EpisodicMemory.from_dict(dict(record["node"]))
                results.append((episode, record["score"]))
        return results

    async def fulltext_search_memories(
        self, query_text: str, k: int = 20
    ) -> list[tuple[SemanticMemory, float]]:
        """Search semantic memories using full-text (BM25)."""
        query = """
        CALL db.index.fulltext.queryNodes('semantic_content', $query_text)
        YIELD node, score
        WHERE node.status = 'active'
        RETURN node, score
        LIMIT $k
        """
        results: list[tuple[SemanticMemory, float]] = []
        async with self.session() as session:
            result = await session.run(query, query_text=query_text, k=k)
            async for record in result:
                memory = SemanticMemory.from_dict(dict(record["node"]))
                results.append((memory, record["score"]))
        return results

    # ==========================================================================
    # Utility operations
    # ==========================================================================

    async def execute_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a raw Cypher query."""
        results: list[dict[str, Any]] = []
        async with self.session() as session:
            result = await session.run(query, **params)
            async for record in result:
                results.append(dict(record))
        return results

    async def clear_all(self) -> None:
        """Delete all nodes and relationships. Use with caution!"""
        query = "MATCH (n) DETACH DELETE n"
        async with self.session() as session:
            await session.run(query)
        logger.warning("All data cleared from Neo4j")


# Global client instance
_client: Neo4jClient | None = None


async def get_client() -> Neo4jClient:
    """Get or create the global Neo4j client."""
    global _client
    if _client is None:
        _client = Neo4jClient()
        await _client.connect()
    return _client


async def close_client() -> None:
    """Close the global Neo4j client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
