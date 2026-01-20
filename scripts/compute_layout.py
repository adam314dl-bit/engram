"""Compute graph layout positions and clusters, store in Neo4j.

This script:
1. Fetches all nodes and relationships from Neo4j
2. Computes 2D layout using GPU (cuGraph) or CPU (graph-tool/NetworkX)
3. Detects communities using Louvain algorithm
4. Stores x/y coordinates and cluster IDs back to each node in Neo4j

Run after ingestion or when graph structure changes significantly.

For GPU acceleration (100-1000x faster), install RAPIDS cuGraph:
    uv add cugraph-cu12 cudf-cu12 --index-url https://pypi.nvidia.com

For multi-core CPU (5-20x faster), install igraph:
    uv add igraph
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Add src to path
sys.path.insert(0, str(project_root / "src"))

from engram.storage.neo4j_client import Neo4jClient


def compute_layout_cugraph(nodes, edges, scale=3000):
    """GPU-accelerated layout using NVIDIA cuGraph (100-1000x faster)."""
    import cudf
    import cugraph

    print("Using cuGraph (GPU accelerated)...")

    # Create edge dataframe
    if edges:
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        edge_df = cudf.DataFrame({"src": src, "dst": dst})
    else:
        edge_df = cudf.DataFrame({"src": [], "dst": []})

    # Create graph
    G = cugraph.Graph()
    if len(edge_df) > 0:
        G.from_cudf_edgelist(edge_df, source="src", destination="dst")

    # ForceAtlas2 layout - GPU accelerated
    positions_df = cugraph.force_atlas2(
        G,
        max_iter=500,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=True,
        barnes_hut_theta=1.2,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
    )

    # Convert to dict and scale
    positions = {}
    for row in positions_df.to_pandas().itertuples():
        positions[row.vertex] = (row.x * scale, row.y * scale)

    # Add isolated nodes at random positions
    import random
    random.seed(42)
    for node in nodes:
        if node not in positions:
            positions[node] = (random.uniform(-scale, scale), random.uniform(-scale, scale))

    return positions


def compute_layout_igraph(nodes, edges, scale=3000):
    """Fast CPU layout using igraph (5-20x faster than NetworkX)."""
    import igraph as ig

    print("Using igraph (optimized C backend)...")

    # Create node index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    # Create edge list with indices
    edge_indices = []
    for src, dst in edges:
        if src in node_to_idx and dst in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[dst]))

    # Create graph
    G = ig.Graph(n=len(nodes), edges=edge_indices, directed=False)

    print(f"Graph: {G.vcount()} vertices, {G.ecount()} edges")

    # Use DrL layout for large graphs (faster) or Fruchterman-Reingold for smaller
    if len(nodes) > 5000:
        print("Using DrL layout (optimized for large graphs)...")
        layout = G.layout_drl()
    else:
        print("Using Fruchterman-Reingold layout...")
        layout = G.layout_fruchterman_reingold(niter=100)

    # Convert to dict and scale
    positions = {}
    coords = layout.coords
    for i, (x, y) in enumerate(coords):
        positions[idx_to_node[i]] = (x * scale, y * scale)

    return positions


def compute_layout_networkx(nodes, edges, scale=3000):
    """Single-core CPU layout using NetworkX (baseline)."""
    import networkx as nx

    print("Using NetworkX (single-core CPU)...")

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for src, dst in edges:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    positions = nx.spring_layout(
        G,
        k=5.0 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 0 else 1.0,
        iterations=100,
        seed=42,
        scale=scale,
    )

    return {k: (float(v[0]), float(v[1])) for k, v in positions.items()}


def compute_layout(nodes, edges, scale=3000):
    """Compute layout using best available backend."""
    # Try GPU first (cuGraph)
    try:
        return compute_layout_cugraph(nodes, edges, scale)
    except ImportError:
        print("cuGraph not available, trying igraph...")
    except Exception as e:
        print(f"cuGraph failed: {e}, trying igraph...")

    # Try fast CPU (igraph)
    try:
        return compute_layout_igraph(nodes, edges, scale)
    except ImportError:
        print("igraph not available, falling back to NetworkX...")
    except Exception as e:
        print(f"igraph failed: {e}, falling back to NetworkX...")

    # Fallback to single-core (NetworkX)
    return compute_layout_networkx(nodes, edges, scale)


def compute_communities(nodes, edges):
    """Detect communities using Louvain algorithm."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for src, dst in edges:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)

    try:
        communities = nx.community.louvain_communities(G, seed=42)
        node_clusters = {}
        for cluster_id, community in enumerate(communities):
            for node_id in community:
                node_clusters[node_id] = cluster_id
        print(f"Found {len(communities)} communities")
        return node_clusters
    except Exception as e:
        print(f"Community detection failed: {e}, using single cluster")
        return {node_id: 0 for node_id in nodes}


def compute_cluster_metadata(positions, node_clusters, edges):
    """Compute cluster centers and inter-cluster edge counts."""
    from collections import defaultdict

    # Compute cluster centers
    cluster_positions = defaultdict(list)
    for node_id, (x, y) in positions.items():
        cluster_id = node_clusters.get(node_id, 0)
        cluster_positions[cluster_id].append((x, y))

    cluster_centers = {}
    for cluster_id, pos_list in cluster_positions.items():
        xs = [p[0] for p in pos_list]
        ys = [p[1] for p in pos_list]
        cluster_centers[cluster_id] = {
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "node_count": len(pos_list),
        }

    print(f"Computed centers for {len(cluster_centers)} clusters")

    # Compute inter-cluster edges
    cluster_edges = defaultdict(int)
    for src, dst in edges:
        src_cluster = node_clusters.get(src)
        dst_cluster = node_clusters.get(dst)
        if src_cluster is not None and dst_cluster is not None and src_cluster != dst_cluster:
            # Use sorted tuple to avoid duplicates (a->b same as b->a)
            edge_key = tuple(sorted([src_cluster, dst_cluster]))
            cluster_edges[edge_key] += 1

    print(f"Found {len(cluster_edges)} inter-cluster edge groups")

    return cluster_centers, dict(cluster_edges)


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
    concepts = await db.execute_query("MATCH (c:Concept) RETURN c.id as id")
    semantic = await db.execute_query("MATCH (s:SemanticMemory) RETURN s.id as id")
    episodic = await db.execute_query("MATCH (e:EpisodicMemory) RETURN e.id as id")

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

    all_edges = [(r["source"], r["target"]) for r in concept_rels + memory_rels + episode_rels]
    print(f"Found {len(all_edges)} relationships")

    # Compute layout (tries GPU -> multi-core -> single-core)
    print("Computing layout...")
    positions = compute_layout(all_nodes, all_edges, scale=3000)
    print("Layout computed.")

    # Detect communities
    print("Detecting communities...")
    node_clusters = compute_communities(all_nodes, all_edges)

    # Compute cluster metadata (centers and inter-cluster edges)
    print("Computing cluster metadata...")
    cluster_centers, cluster_edges = compute_cluster_metadata(positions, node_clusters, all_edges)

    print("Storing positions and clusters in Neo4j...")

    # Build node type mapping
    concept_ids = {n["id"] for n in concepts}
    semantic_ids = {n["id"] for n in semantic}
    episodic_ids = {n["id"] for n in episodic}

    # Group nodes by type for per-label indexed writes (much faster)
    by_type = {"Concept": [], "SemanticMemory": [], "EpisodicMemory": []}
    for node_id, (x, y) in positions.items():
        cluster_id = node_clusters.get(node_id, 0)
        data = {"id": node_id, "x": float(x), "y": float(y), "cluster": cluster_id}
        if node_id in concept_ids:
            by_type["Concept"].append(data)
        elif node_id in semantic_ids:
            by_type["SemanticMemory"].append(data)
        elif node_id in episodic_ids:
            by_type["EpisodicMemory"].append(data)

    # Write each type separately using UNWIND batching (much faster)
    batch_size = 2000
    total = len(positions)
    written = 0

    for label, nodes_data in by_type.items():
        if not nodes_data:
            continue

        for i in range(0, len(nodes_data), batch_size):
            batch = nodes_data[i : i + batch_size]

            await db.execute_query(
                f"""
                UNWIND $batch AS row
                MATCH (n:{label}) WHERE n.id = row.id
                SET n.layout_x = row.x, n.layout_y = row.y, n.cluster = row.cluster
                """,
                batch=batch,
            )

            written += len(batch)
            print(f"  Stored {written}/{total} positions...")

    # Store cluster metadata
    print("Storing cluster metadata...")

    # Clear old cluster metadata
    await db.execute_query("MATCH (c:ClusterMeta) DETACH DELETE c")

    # Create ClusterMeta nodes using UNWIND
    cluster_data = [
        {"cluster_id": cid, "x": c["x"], "y": c["y"], "node_count": c["node_count"]}
        for cid, c in cluster_centers.items()
    ]
    await db.execute_query(
        """
        UNWIND $batch AS row
        CREATE (c:ClusterMeta {
            cluster_id: row.cluster_id,
            center_x: row.x,
            center_y: row.y,
            node_count: row.node_count
        })
        """,
        batch=cluster_data,
    )
    print(f"  Created {len(cluster_centers)} ClusterMeta nodes")

    # Create inter-cluster edges using UNWIND
    edge_data = [
        {"src": src, "dst": dst, "count": count}
        for (src, dst), count in cluster_edges.items()
    ]
    if edge_data:
        await db.execute_query(
            """
            UNWIND $batch AS row
            MATCH (c1:ClusterMeta {cluster_id: row.src}), (c2:ClusterMeta {cluster_id: row.dst})
            CREATE (c1)-[:CLUSTER_EDGE {count: row.count}]->(c2)
            """,
            batch=edge_data,
        )
    print(f"  Created {len(cluster_edges)} inter-cluster edges")

    # Compute bounding box
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
