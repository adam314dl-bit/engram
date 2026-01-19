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


def compute_layout_igraph(nodes, edges, scale=10000):
    """Fast CPU layout using igraph (5-20x faster than NetworkX).

    Parameters tuned to match NetworkX spring_layout output.
    """
    import igraph as ig
    import math

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
        # Match NetworkX spring_layout parameters: k = 5.0 / sqrt(n)
        # igraph doesn't support seed like NetworkX, just use niter
        layout = G.layout_fruchterman_reingold(niter=100)

    # Convert to dict and NORMALIZE to match NetworkX scale (-scale to +scale)
    # igraph produces unbounded coordinates, we need to rescale
    coords = layout.coords
    if coords:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        # Center and scale to match NetworkX output range
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        max_extent = max(max(abs(x - cx) for x in xs), max(abs(y - cy) for y in ys))

        if max_extent > 0:
            scale_factor = scale / max_extent
        else:
            scale_factor = 1.0

        print(f"Scaling: center=({cx:.1f}, {cy:.1f}), max_extent={max_extent:.1f}, scale_factor={scale_factor:.4f}")

    positions = {}
    coords = layout.coords
    for i, (x, y) in enumerate(coords):
        # Apply centering and scaling to match NetworkX output
        nx = (x - cx) * scale_factor
        ny = (y - cy) * scale_factor
        positions[idx_to_node[i]] = (nx, ny)

    # Add isolated nodes at random positions
    import random
    random.seed(42)
    for node in nodes:
        if node not in positions:
            positions[node] = (random.uniform(-scale, scale), random.uniform(-scale, scale))

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


def separate_clusters(positions, node_clusters, separation_factor=2.0):
    """Push clusters apart while keeping nodes within clusters close.

    Args:
        positions: dict of node_id -> (x, y)
        node_clusters: dict of node_id -> cluster_id
        separation_factor: how much to push clusters apart (2.0 = double the distance)

    Returns:
        Updated positions dict
    """
    import math

    print(f"Separating clusters with factor {separation_factor}...")

    # Compute cluster centers
    cluster_sums = {}
    cluster_counts = {}

    for node_id, (x, y) in positions.items():
        cluster_id = node_clusters.get(node_id, 0)
        if cluster_id not in cluster_sums:
            cluster_sums[cluster_id] = [0.0, 0.0]
            cluster_counts[cluster_id] = 0
        cluster_sums[cluster_id][0] += x
        cluster_sums[cluster_id][1] += y
        cluster_counts[cluster_id] += 1

    cluster_centers = {}
    for cluster_id in cluster_sums:
        count = cluster_counts[cluster_id]
        cluster_centers[cluster_id] = (
            cluster_sums[cluster_id][0] / count,
            cluster_sums[cluster_id][1] / count
        )

    # Compute global center
    all_x = [c[0] for c in cluster_centers.values()]
    all_y = [c[1] for c in cluster_centers.values()]
    global_cx = sum(all_x) / len(all_x) if all_x else 0
    global_cy = sum(all_y) / len(all_y) if all_y else 0

    print(f"Found {len(cluster_centers)} clusters, global center: ({global_cx:.1f}, {global_cy:.1f})")

    # Compute new cluster centers (pushed away from global center)
    new_cluster_centers = {}
    for cluster_id, (cx, cy) in cluster_centers.items():
        # Vector from global center to cluster center
        dx = cx - global_cx
        dy = cy - global_cy

        # Push cluster center away by separation_factor
        new_cx = global_cx + dx * separation_factor
        new_cy = global_cy + dy * separation_factor
        new_cluster_centers[cluster_id] = (new_cx, new_cy)

    # Move each node by the same offset as its cluster
    new_positions = {}
    for node_id, (x, y) in positions.items():
        cluster_id = node_clusters.get(node_id, 0)
        old_center = cluster_centers[cluster_id]
        new_center = new_cluster_centers[cluster_id]

        # Offset = new_center - old_center
        offset_x = new_center[0] - old_center[0]
        offset_y = new_center[1] - old_center[1]

        new_positions[node_id] = (x + offset_x, y + offset_y)

    return new_positions


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
    positions = compute_layout(all_nodes, all_edges, scale=10000)
    print("Layout computed.")

    # Detect communities
    print("Detecting communities...")
    node_clusters = compute_communities(all_nodes, all_edges)

    # Separate clusters - push clusters apart while keeping nodes within clusters close
    print("Separating clusters...")
    positions = separate_clusters(positions, node_clusters, separation_factor=2.5)

    print("Storing positions and clusters in Neo4j...")

    # Store positions and clusters back to Neo4j in batches
    batch_size = 500
    node_list = list(positions.items())

    for i in range(0, len(node_list), batch_size):
        batch = node_list[i : i + batch_size]

        for node_id, (x, y) in batch:
            cluster_id = node_clusters.get(node_id, 0)
            await db.execute_query(
                """
                MATCH (n) WHERE n.id = $id
                SET n.layout_x = $x, n.layout_y = $y, n.cluster = $cluster
                """,
                id=node_id,
                x=float(x),
                y=float(y),
                cluster=cluster_id,
            )

        progress = min(i + batch_size, len(node_list))
        print(f"  Stored {progress}/{len(node_list)} positions...")

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
