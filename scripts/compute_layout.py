"""Compute graph layout positions and store in Neo4j.

This script:
1. Fetches all nodes and relationships from Neo4j
2. Computes 2D layout using NetworkX spring layout
3. Stores x/y coordinates back to each node in Neo4j

Run after ingestion or when graph structure changes significantly.
"""

import asyncio
import os
import sys
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Add src to path
sys.path.insert(0, str(project_root / "src"))

from engram.storage.neo4j_client import Neo4jClient


async def compute_and_store_layout():
    """Compute layout and store positions in Neo4j."""

    # Connect to Neo4j
    db = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "engram_password"),
    )
    await db.connect()

    print("Fetching nodes from Neo4j...")

    # Fetch all nodes
    concepts = await db.execute_query(
        "MATCH (c:Concept) RETURN c.id as id"
    )
    semantic = await db.execute_query(
        "MATCH (s:SemanticMemory) RETURN s.id as id"
    )
    episodic = await db.execute_query(
        "MATCH (e:EpisodicMemory) RETURN e.id as id"
    )

    all_nodes = [n["id"] for n in concepts + semantic + episodic]
    print(f"Found {len(all_nodes)} nodes")

    # Fetch all relationships
    print("Fetching relationships...")

    concept_rels = await db.execute_query(
        "MATCH (c1:Concept)-[:RELATED_TO]->(c2:Concept) RETURN c1.id as source, c2.id as target"
    )
    memory_rels = await db.execute_query(
        "MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept) RETURN s.id as source, c.id as target"
    )
    episode_rels = await db.execute_query(
        "MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept) RETURN e.id as source, c.id as target"
    )

    all_rels = concept_rels + memory_rels + episode_rels
    print(f"Found {len(all_rels)} relationships")

    # Build NetworkX graph
    print("Building NetworkX graph...")
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    for rel in all_rels:
        if rel["source"] in G.nodes and rel["target"] in G.nodes:
            G.add_edge(rel["source"], rel["target"])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute layout
    print("Computing spring layout (this may take a minute for large graphs)...")

    # Use spring layout with parameters tuned for large graphs
    # k controls optimal distance between nodes (higher = more spread)
    # iterations controls quality vs speed
    # scale controls overall coordinate range
    positions = nx.spring_layout(
        G,
        k=5.0 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 0 else 1.0,
        iterations=100,
        seed=42,  # Reproducible layout
        scale=3000,  # Larger scale for more spread
    )

    print("Layout computed. Storing positions in Neo4j...")

    # Store positions back to Neo4j in batches
    batch_size = 500
    node_list = list(positions.items())

    for i in range(0, len(node_list), batch_size):
        batch = node_list[i:i + batch_size]

        # Build batch update query
        for node_id, (x, y) in batch:
            await db.execute_query(
                """
                MATCH (n) WHERE n.id = $id
                SET n.layout_x = $x, n.layout_y = $y
                """,
                id=node_id, x=float(x), y=float(y)
            )

        progress = min(i + batch_size, len(node_list))
        print(f"  Stored {progress}/{len(node_list)} positions...")

    # Compute bounding box for viewport queries
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        print(f"Bounding box: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")

    await db.close()
    print("Done! Layout positions stored in Neo4j as layout_x, layout_y properties.")


if __name__ == "__main__":
    asyncio.run(compute_and_store_layout())
