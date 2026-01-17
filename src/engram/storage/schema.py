"""Neo4j schema setup - constraints, indexes, and graph structure."""

from engram.config import settings

# Schema setup queries
SCHEMA_QUERIES = [
    # Uniqueness constraints
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT semantic_id IF NOT EXISTS FOR (s:SemanticMemory) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT episodic_id IF NOT EXISTS FOR (e:EpisodicMemory) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    # Text indexes for search
    "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
    "CREATE INDEX semantic_status IF NOT EXISTS FOR (s:SemanticMemory) ON (s.status)",
    "CREATE INDEX episodic_domain IF NOT EXISTS FOR (e:EpisodicMemory) ON (e.domain)",
    "CREATE INDEX doc_status IF NOT EXISTS FOR (d:Document) ON (d.status)",
]

# Vector index queries (separate due to different syntax)
VECTOR_INDEX_QUERIES = [
    f"""
    CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
    FOR (c:Concept) ON (c.embedding)
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {settings.embedding_dimensions},
        `vector.similarity_function`: 'cosine'
    }}}}
    """,
    f"""
    CREATE VECTOR INDEX semantic_embeddings IF NOT EXISTS
    FOR (s:SemanticMemory) ON (s.embedding)
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {settings.embedding_dimensions},
        `vector.similarity_function`: 'cosine'
    }}}}
    """,
    f"""
    CREATE VECTOR INDEX episodic_embeddings IF NOT EXISTS
    FOR (e:EpisodicMemory) ON (e.embedding)
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {settings.embedding_dimensions},
        `vector.similarity_function`: 'cosine'
    }}}}
    """,
]

# Full-text index for BM25 search
FULLTEXT_INDEX_QUERY = """
CREATE FULLTEXT INDEX semantic_content IF NOT EXISTS
FOR (s:SemanticMemory) ON EACH [s.content]
"""

# Relationship types used in the graph:
# (c1:Concept)-[:RELATED_TO {weight: 0.8, type: "uses"}]->(c2:Concept)
# (c:Concept)-[:IS_A]->(parent:Concept)
# (s:SemanticMemory)-[:ABOUT]->(c:Concept)
# (s:SemanticMemory)-[:EXTRACTED_FROM]->(d:Document)
# (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
# (e:EpisodicMemory)-[:USED]->(s:SemanticMemory)
# (e:EpisodicMemory)-[:CRYSTALLIZED_TO]->(s:SemanticMemory)


def get_all_schema_queries() -> list[str]:
    """Get all schema setup queries."""
    queries = SCHEMA_QUERIES.copy()
    queries.extend(VECTOR_INDEX_QUERIES)
    queries.append(FULLTEXT_INDEX_QUERY)
    return queries


# Cleanup query for testing
DROP_ALL_QUERY = """
MATCH (n) DETACH DELETE n
"""

DROP_INDEXES_QUERY = """
CALL apoc.schema.assert({}, {})
"""
