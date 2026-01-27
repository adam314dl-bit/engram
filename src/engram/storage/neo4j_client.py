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


def escape_lucene_query(query: str) -> str:
    """Escape special characters for Lucene fulltext search.

    Lucene special characters: + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /
    """
    special_chars = r'+-&|!(){}[]^"~*?:\/'
    escaped = []
    for char in query:
        if char in special_chars:
            escaped.append(f"\\{char}")
        else:
            escaped.append(char)
    return "".join(escaped)


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

    async def save_concepts_batch(self, concepts: list[Concept]) -> None:
        """Save multiple concepts in a single batch operation."""
        if not concepts:
            return

        query = """
        UNWIND $items AS item
        MERGE (c:Concept {id: item.id})
        SET c += item.props
        """
        items = []
        for concept in concepts:
            props = concept.to_dict()
            concept_id = props.pop("id")
            items.append({"id": concept_id, "props": props})

        async with self.session() as session:
            await session.run(query, items=items)
        logger.debug(f"Batch saved {len(concepts)} concepts")

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
        WHERE neighbor.status IS NULL OR neighbor.status = 'active'
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
                    # v4.4: Semantic edge properties
                    is_semantic=rel_data.get("is_semantic", False),
                    is_universal=rel_data.get("is_universal", False),
                    source_type=rel_data.get("source_type", "document"),
                    provenance_doc_id=rel_data.get("provenance_doc_id"),
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

    async def save_relations_batch(self, relations: list[ConceptRelation]) -> None:
        """Save multiple concept relations in a single batch operation."""
        if not relations:
            return

        query = """
        UNWIND $items AS item
        MATCH (source:Concept {id: item.source_id})
        MATCH (target:Concept {id: item.target_id})
        MERGE (source)-[r:RELATED_TO]->(target)
        SET r.type = item.relation_type,
            r.weight = item.weight,
            r.edge_embedding = item.edge_embedding,
            r.co_occurrence_count = item.co_occurrence_count,
            r.last_used = item.last_used
        """
        items = [relation.to_dict() for relation in relations]

        async with self.session() as session:
            await session.run(query, items=items)
        logger.debug(f"Batch saved {len(relations)} relations")

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
    # v4.4: Concept deduplication operations
    # ==========================================================================

    async def get_concept_by_alias(self, alias: str) -> Concept | None:
        """Get a concept by alias (case-insensitive).

        Searches in the aliases list field.
        """
        query = """
        MATCH (c:Concept)
        WHERE any(a IN c.aliases WHERE toLower(a) = toLower($alias))
          AND (c.status IS NULL OR c.status = 'active')
        RETURN c
        """
        async with self.session() as session:
            result = await session.run(query, alias=alias)
            record = await result.single()
            if record:
                return Concept.from_dict(dict(record["c"]))
            return None

    async def merge_concepts(
        self,
        canonical_id: str,
        merge_ids: list[str],
        aliases: list[str],
    ) -> dict[str, int]:
        """Merge duplicate concepts into a canonical concept.

        Redirects all edges and memory links from merge_ids to canonical_id,
        then marks merge concepts as 'merged'.

        Args:
            canonical_id: ID of the canonical concept to keep
            merge_ids: IDs of concepts to merge
            aliases: Names of merged concepts to add as aliases

        Returns:
            Dict with counts of redirected edges and memories
        """
        # Add aliases to canonical concept
        alias_query = """
        MATCH (c:Concept {id: $canonical_id})
        SET c.aliases = coalesce(c.aliases, []) + $aliases,
            c.updated_at = datetime()
        """
        async with self.session() as session:
            await session.run(query=alias_query, canonical_id=canonical_id, aliases=aliases)

        # Redirect outgoing edges
        redirect_out_query = """
        MATCH (m:Concept)-[r:RELATED_TO]->(target:Concept)
        WHERE m.id IN $merge_ids
        WITH m, r, target
        MATCH (canonical:Concept {id: $canonical_id})
        MERGE (canonical)-[new:RELATED_TO]->(target)
        ON CREATE SET new = properties(r)
        ON MATCH SET new.weight = CASE WHEN new.weight < r.weight THEN r.weight ELSE new.weight END
        DELETE r
        RETURN count(r) as count
        """
        async with self.session() as session:
            result = await session.run(
                redirect_out_query, merge_ids=merge_ids, canonical_id=canonical_id
            )
            record = await result.single()
            edges_out = record["count"] if record else 0

        # Redirect incoming edges
        redirect_in_query = """
        MATCH (source:Concept)-[r:RELATED_TO]->(m:Concept)
        WHERE m.id IN $merge_ids
        WITH source, r, m
        MATCH (canonical:Concept {id: $canonical_id})
        WHERE source.id <> $canonical_id
        MERGE (source)-[new:RELATED_TO]->(canonical)
        ON CREATE SET new = properties(r)
        ON MATCH SET new.weight = CASE WHEN new.weight < r.weight THEN r.weight ELSE new.weight END
        DELETE r
        RETURN count(r) as count
        """
        async with self.session() as session:
            result = await session.run(
                redirect_in_query, merge_ids=merge_ids, canonical_id=canonical_id
            )
            record = await result.single()
            edges_in = record["count"] if record else 0

        # Redirect memory links
        redirect_memory_query = """
        MATCH (s:SemanticMemory)-[r:ABOUT]->(m:Concept)
        WHERE m.id IN $merge_ids
        WITH s, r, m
        MATCH (canonical:Concept {id: $canonical_id})
        MERGE (s)-[:ABOUT]->(canonical)
        DELETE r
        RETURN count(r) as count
        """
        async with self.session() as session:
            result = await session.run(
                redirect_memory_query, merge_ids=merge_ids, canonical_id=canonical_id
            )
            record = await result.single()
            memories = record["count"] if record else 0

        # Mark merged concepts
        mark_merged_query = """
        MATCH (c:Concept)
        WHERE c.id IN $merge_ids
        SET c.status = 'merged',
            c.canonical_id = $canonical_id,
            c.is_canonical = false,
            c.updated_at = datetime()
        """
        async with self.session() as session:
            await session.run(mark_merged_query, merge_ids=merge_ids, canonical_id=canonical_id)

        logger.info(
            f"Merged {len(merge_ids)} concepts into {canonical_id}: "
            f"{edges_out + edges_in} edges, {memories} memories redirected"
        )

        return {
            "edges_redirected": edges_out + edges_in,
            "memories_redirected": memories,
        }

    # ==========================================================================
    # v4.5: Path-based retrieval operations
    # ==========================================================================

    async def find_shortest_paths(
        self,
        concept_ids: list[str],
        max_length: int = 3,
    ) -> list[dict]:
        """Find shortest paths between all pairs of query concepts.

        Args:
            concept_ids: List of concept IDs to find paths between
            max_length: Maximum path length (number of hops)

        Returns:
            List of dicts with keys: src_id, tgt_id, path_ids, length
        """
        if len(concept_ids) < 2:
            return []

        # Note: Cypher shortestPath requires literal path length, so we use APOC
        # or iterate with BFS. Using variable-length pattern with limit.
        query = """
        UNWIND $concept_ids AS src_id
        UNWIND $concept_ids AS tgt_id
        WITH src_id, tgt_id WHERE src_id < tgt_id
        MATCH (s:Concept {id: src_id}), (t:Concept {id: tgt_id})
        MATCH path = shortestPath((s)-[:RELATED_TO*1..3]-(t))
        WHERE length(path) <= $max_length
        RETURN src_id, tgt_id,
               [n IN nodes(path) | n.id] AS path_ids,
               length(path) AS len
        ORDER BY len ASC
        """
        results: list[dict] = []
        async with self.session() as session:
            result = await session.run(
                query,
                concept_ids=concept_ids,
                max_length=max_length,
            )
            async for record in result:
                results.append({
                    "src_id": record["src_id"],
                    "tgt_id": record["tgt_id"],
                    "path_ids": list(record["path_ids"]),
                    "length": record["len"],
                })
        return results

    async def find_bridge_concepts(
        self,
        concept_ids: list[str],
        max_hops: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """Find concepts connected to 2+ query concepts (bridge nodes).

        Args:
            concept_ids: List of query concept IDs
            max_hops: Maximum hops from query concepts
            limit: Maximum number of bridge concepts to return

        Returns:
            List of dicts with keys: id, name, connected, count
        """
        if len(concept_ids) < 2:
            return []

        query = """
        UNWIND $concept_ids AS qid
        MATCH (q:Concept {id: qid})-[:RELATED_TO*1..2]-(bridge:Concept)
        WHERE NOT bridge.id IN $concept_ids
          AND (bridge.status IS NULL OR bridge.status = 'active')
        WITH bridge, collect(DISTINCT qid) AS connected
        WHERE size(connected) >= 2
        RETURN bridge.id AS id, bridge.name AS name,
               connected, size(connected) AS count
        ORDER BY count DESC
        LIMIT $limit
        """
        results: list[dict] = []
        async with self.session() as session:
            result = await session.run(
                query,
                concept_ids=concept_ids,
                limit=limit,
            )
            async for record in result:
                results.append({
                    "id": record["id"],
                    "name": record["name"],
                    "connected": list(record["connected"]),
                    "count": record["count"],
                })
        return results

    async def find_shared_memories(
        self,
        concept_ids: list[str],
        min_links: int = 2,
        limit: int = 50,
    ) -> list[tuple[SemanticMemory, list[str], int]]:
        """Find memories linked to multiple query concepts.

        Args:
            concept_ids: List of query concept IDs
            min_links: Minimum number of concept links required
            limit: Maximum number of memories to return

        Returns:
            List of tuples: (memory, linked_concept_ids, link_count)
        """
        if len(concept_ids) < 2:
            return []

        query = """
        MATCH (m:SemanticMemory)-[:ABOUT]->(c:Concept)
        WHERE c.id IN $concept_ids
          AND m.status IN ['active', 'deprioritized']
        WITH m, collect(DISTINCT c.id) AS linked
        WHERE size(linked) >= $min_links
        RETURN m, linked, size(linked) AS link_count
        ORDER BY link_count DESC, m.importance DESC
        LIMIT $limit
        """
        results: list[tuple[SemanticMemory, list[str], int]] = []
        async with self.session() as session:
            result = await session.run(
                query,
                concept_ids=concept_ids,
                min_links=min_links,
                limit=limit,
            )
            async for record in result:
                memory = SemanticMemory.from_dict(dict(record["m"]))
                linked = list(record["linked"])
                link_count = record["link_count"]
                results.append((memory, linked, link_count))
        return results

    async def get_memories_for_path_concepts(
        self,
        path_concept_ids: list[str],
        exclude_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[SemanticMemory]:
        """Get memories from intermediate path concepts.

        Args:
            path_concept_ids: Concept IDs from paths (intermediate nodes)
            exclude_ids: Memory IDs to exclude
            limit: Maximum number of memories to return

        Returns:
            List of SemanticMemory objects
        """
        if not path_concept_ids:
            return []

        exclude_ids = exclude_ids or []

        query = """
        MATCH (m:SemanticMemory)-[:ABOUT]->(c:Concept)
        WHERE c.id IN $path_ids
          AND NOT m.id IN $exclude_ids
          AND m.status IN ['active', 'deprioritized']
        RETURN DISTINCT m
        ORDER BY m.importance DESC
        LIMIT $limit
        """
        memories: list[SemanticMemory] = []
        async with self.session() as session:
            result = await session.run(
                query,
                path_ids=path_concept_ids,
                exclude_ids=exclude_ids,
                limit=limit,
            )
            async for record in result:
                memories.append(SemanticMemory.from_dict(dict(record["m"])))
        return memories

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

    async def save_memories_batch(self, memories: list[SemanticMemory]) -> None:
        """Save multiple semantic memories in a single batch operation."""
        if not memories:
            return

        query = """
        UNWIND $items AS item
        MERGE (s:SemanticMemory {id: item.id})
        SET s += item.props
        """
        items = []
        for memory in memories:
            props = memory.to_dict()
            memory_id = props.pop("id")
            items.append({"id": memory_id, "props": props})

        async with self.session() as session:
            await session.run(query, items=items)
        logger.debug(f"Batch saved {len(memories)} memories")

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

    async def link_memories_to_concepts_batch(
        self, links: list[tuple[str, str]]
    ) -> None:
        """Batch create ABOUT relationships between memories and concepts.

        Args:
            links: List of (memory_id, concept_id) tuples
        """
        if not links:
            return

        query = """
        UNWIND $links AS link
        MATCH (s:SemanticMemory {id: link.memory_id})
        MATCH (c:Concept {id: link.concept_id})
        MERGE (s)-[:ABOUT]->(c)
        """
        items = [{"memory_id": m, "concept_id": c} for m, c in links]

        async with self.session() as session:
            await session.run(query, links=items)
        logger.debug(f"Batch linked {len(links)} memory-concept pairs")

    async def link_memory_to_document(self, memory_id: str, doc_id: str) -> None:
        """Create EXTRACTED_FROM relationship between memory and document."""
        query = """
        MATCH (s:SemanticMemory {id: $memory_id})
        MATCH (d:Document {id: $doc_id})
        MERGE (s)-[:EXTRACTED_FROM]->(d)
        """
        async with self.session() as session:
            await session.run(query, memory_id=memory_id, doc_id=doc_id)

    async def link_memories_to_documents_batch(
        self, links: list[tuple[str, str]]
    ) -> None:
        """Batch create EXTRACTED_FROM relationships between memories and documents.

        Args:
            links: List of (memory_id, doc_id) tuples
        """
        if not links:
            return

        query = """
        UNWIND $links AS link
        MATCH (s:SemanticMemory {id: link.memory_id})
        MATCH (d:Document {id: link.doc_id})
        MERGE (s)-[:EXTRACTED_FROM]->(d)
        """
        items = [{"memory_id": m, "doc_id": d} for m, d in links]

        async with self.session() as session:
            await session.run(query, links=items)
        logger.debug(f"Batch linked {len(links)} memory-document pairs")

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
          AND s.status IN ['active', 'deprioritized']
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

    async def get_source_documents_for_memories(
        self,
        memory_ids: list[str],
    ) -> list[Document]:
        """
        Get source documents for a list of memory IDs.

        Follows: Memory -[:EXTRACTED_FROM]-> Document
        Returns deduplicated documents.
        """
        if not memory_ids:
            return []

        query = """
        MATCH (s:SemanticMemory)-[:EXTRACTED_FROM]->(d:Document)
        WHERE s.id IN $memory_ids
        RETURN DISTINCT d
        """

        documents: list[Document] = []
        async with self.session() as session:
            result = await session.run(query, memory_ids=memory_ids)
            async for record in result:
                documents.append(Document.from_dict(dict(record["d"])))

        return documents

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

    async def fulltext_search_concepts(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Concept, float]]:
        """Search concepts by full-text (BM25) on name field.

        Used in bm25_graph mode when vector search is disabled.
        Uses concept_content fulltext index.
        """
        # Escape Lucene special characters
        escaped_query = escape_lucene_query(query_text)
        query = """
        CALL db.index.fulltext.queryNodes('concept_content', $query_text)
        YIELD node, score
        RETURN node, score
        LIMIT $k
        """
        results: list[tuple[Concept, float]] = []
        async with self.session() as session:
            result = await session.run(query, query_text=escaped_query, k=k)
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
        WHERE node.status IN ['active', 'deprioritized']
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
        """Search semantic memories using full-text (BM25) on content field.

        v4.6: BM25 searches content field while vector search uses search_content embedding.
        Uses semantic_content index which covers the content field.
        """
        # Escape Lucene special characters to prevent query parsing errors
        escaped_query = escape_lucene_query(query_text)
        query = """
        CALL db.index.fulltext.queryNodes('semantic_content', $query_text)
        YIELD node, score
        WHERE node.status IN ['active', 'deprioritized']
        RETURN node, score
        LIMIT $k
        """
        results: list[tuple[SemanticMemory, float]] = []
        async with self.session() as session:
            result = await session.run(query, query_text=escaped_query, k=k)
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
