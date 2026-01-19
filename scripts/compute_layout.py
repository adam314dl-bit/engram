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


def compute_local_layout_task(task):
    """Wrapper for parallel processing of local layout."""
    cluster_id, node_list, node_edges, radius, center = task
    local_positions = compute_local_layout(node_list, node_edges, radius)
    # Apply center offset
    return {
        node_id: (center[0] + lx, center[1] + ly)
        for node_id, (lx, ly) in local_positions.items()
    }


def find_tangent_positions(c1, c2, r):
    """Find positions where a circle of radius r is tangent to circles c1 and c2."""
    import math

    x1, y1, r1 = c1
    x2, y2, r2 = c2

    # Distance between centers
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if d == 0:
        return []

    # Distances from c1 and c2 to new circle center
    d1 = r1 + r
    d2 = r2 + r

    # Check if solution exists
    if d > d1 + d2 or d < abs(d1 - d2):
        return []

    # Find intersection points of two circles centered at c1, c2 with radii d1, d2
    a = (d1**2 - d2**2 + d**2) / (2 * d)

    h_sq = d1**2 - a**2
    if h_sq < 0:
        return []
    h = math.sqrt(h_sq)

    # Point on line between centers
    px = x1 + a * (x2 - x1) / d
    py = y1 + a * (y2 - y1) / d

    # Perpendicular offset
    dx = h * (y2 - y1) / d
    dy = h * (x1 - x2) / d

    return [(px + dx, py + dy), (px - dx, py - dy)]


def compute_local_layout(node_list, node_edges, radius):
    """Compute force-directed layout + collision resolution for nodes within a cluster.

    Two phases:
    1. Force-directed to preserve relationships (connected nodes near each other)
    2. Collision resolution to guarantee minimum spacing between all nodes
    """
    import math
    import random

    random.seed(42)

    # Calculate minimum node spacing based on cluster size
    # Ensure nodes don't overlap - minimum distance between node centers
    min_spacing = max(30, radius / math.sqrt(len(node_list)) * 0.8)

    # Initialize random positions
    positions = {}
    for node_id in node_list:
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, radius * 0.5)
        positions[node_id] = [r * math.cos(angle), r * math.sin(angle)]

    node_set = set(node_list)

    # Phase 1: Force-directed iterations (preserve relationships)
    iterations = 50
    k = radius / math.sqrt(len(node_list))  # Optimal distance

    for _ in range(iterations):
        # Calculate repulsive forces
        displacements = {node_id: [0.0, 0.0] for node_id in node_list}

        for i, n1 in enumerate(node_list):
            for n2 in node_list[i+1:]:
                dx = positions[n1][0] - positions[n2][0]
                dy = positions[n1][1] - positions[n2][1]
                dist = max(0.1, math.sqrt(dx*dx + dy*dy))

                # Repulsive force
                force = k * k / dist
                fx = dx / dist * force
                fy = dy / dist * force

                displacements[n1][0] += fx
                displacements[n1][1] += fy
                displacements[n2][0] -= fx
                displacements[n2][1] -= fy

        # Calculate attractive forces (edges)
        for n1 in node_list:
            if n1 in node_edges:
                for n2 in node_edges[n1]:
                    if n2 in node_set and n2 != n1:
                        dx = positions[n1][0] - positions[n2][0]
                        dy = positions[n1][1] - positions[n2][1]
                        dist = max(0.1, math.sqrt(dx*dx + dy*dy))

                        # Attractive force
                        force = dist * dist / k
                        fx = dx / dist * force
                        fy = dy / dist * force

                        displacements[n1][0] -= fx * 0.5
                        displacements[n1][1] -= fy * 0.5

        # Apply displacements with cooling
        cooling = 1.0 - _ / iterations
        max_disp = radius * 0.1 * cooling

        for node_id in node_list:
            dx, dy = displacements[node_id]
            dist = max(0.1, math.sqrt(dx*dx + dy*dy))

            # Limit displacement
            dx = dx / dist * min(dist, max_disp)
            dy = dy / dist * min(dist, max_disp)

            positions[node_id][0] += dx
            positions[node_id][1] += dy

            # Keep within radius
            px, py = positions[node_id]
            d = math.sqrt(px*px + py*py)
            if d > radius * 0.9:
                positions[node_id][0] = px / d * radius * 0.9
                positions[node_id][1] = py / d * radius * 0.9

    # Phase 2: Collision resolution - push overlapping nodes apart
    collision_iterations = 100
    for iteration in range(collision_iterations):
        max_overlap = 0

        for i, n1 in enumerate(node_list):
            for n2 in node_list[i+1:]:
                dx = positions[n2][0] - positions[n1][0]
                dy = positions[n2][1] - positions[n1][1]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < min_spacing:
                    overlap = min_spacing - dist
                    max_overlap = max(max_overlap, overlap)

                    if dist > 0.01:
                        # Push apart along the line between them
                        push = (overlap / 2 + 0.5) / dist
                        positions[n1][0] -= dx * push
                        positions[n1][1] -= dy * push
                        positions[n2][0] += dx * push
                        positions[n2][1] += dy * push
                    else:
                        # Same position - push in random direction
                        angle = random.uniform(0, 2 * math.pi)
                        push = min_spacing / 2
                        positions[n1][0] -= push * math.cos(angle)
                        positions[n1][1] -= push * math.sin(angle)
                        positions[n2][0] += push * math.cos(angle)
                        positions[n2][1] += push * math.sin(angle)

        # Keep within expanded radius (allow some overflow to fit all nodes)
        max_radius = radius * 1.2
        for node_id in node_list:
            px, py = positions[node_id]
            d = math.sqrt(px*px + py*py)
            if d > max_radius:
                positions[node_id][0] = px / d * max_radius
                positions[node_id][1] = py / d * max_radius

        # Converged if no significant overlaps
        if max_overlap < 1:
            break

    return {k: tuple(v) for k, v in positions.items()}


def compute_communities(nodes, edges):
    """Detect communities using balanced partitioning (legacy, for backward compatibility)."""
    hierarchy = compute_hierarchical_communities(nodes, edges, levels=5)
    # Return level 2 (finest level) for backward compatibility
    return {node_id: h['level2'] for node_id, h in hierarchy.items()}


def compute_hierarchical_communities(nodes, edges, levels=5, splits_per_level=10):
    """Balanced hierarchical clustering using recursive spatial partitioning.

    Strategy:
    1. Compute quick initial positions using force-directed layout
    2. Recursively split into ~10 equal parts by angle from centroid
    3. Each level divides evenly, guaranteeing predictable cluster sizes

    For 30k nodes with 10 splits per level:
    - L0: 10 clusters (~3000 each)
    - L1: 100 clusters (~300 each)
    - L2: 1000 clusters (~30 each)
    - L3: 10000 clusters (~3 each)
    - L4: individual nodes

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples
        levels: number of hierarchy levels (default 5)
        splits_per_level: target splits at each level (default 10)

    Returns:
        dict of node_id -> {'level0': int, 'level1': int, ...}
    """
    import math

    print(f"Computing {levels}-level balanced hierarchy ({splits_per_level} splits per level)...")

    n = len(nodes)
    print(f"Total nodes: {n}")

    if n == 0:
        return {}

    # Step 1: Compute quick initial positions for spatial clustering
    print("  Computing initial positions...")
    positions = _quick_layout(nodes, edges)

    # Initialize hierarchy
    hierarchy = {node_id: {f'level{i}': 0 for i in range(levels)} for node_id in nodes}

    # Step 2: Recursive balanced splitting
    global_counters = {i: 0 for i in range(levels)}

    def split_balanced(node_ids, level):
        """Recursively split nodes into balanced clusters."""
        if level >= levels:
            return

        # Too few nodes to split further
        if len(node_ids) <= 1:
            cluster_id = global_counters[level]
            global_counters[level] += 1
            for nid in node_ids:
                hierarchy[nid][f'level{level}'] = cluster_id
            # Fill remaining levels
            for remaining in range(level + 1, levels):
                remaining_id = global_counters[remaining]
                global_counters[remaining] += 1
                for nid in node_ids:
                    hierarchy[nid][f'level{remaining}'] = remaining_id
            return

        # Determine how many splits (fewer for small groups)
        num_splits = min(splits_per_level, max(2, len(node_ids) // 3))

        # Split by angle from centroid (keeps spatially close nodes together)
        node_list = list(node_ids)

        # Compute centroid
        cx = sum(positions[nid][0] for nid in node_list) / len(node_list)
        cy = sum(positions[nid][1] for nid in node_list) / len(node_list)

        # Sort by angle from centroid
        def angle_from_center(nid):
            x, y = positions[nid]
            return math.atan2(y - cy, x - cx)

        node_list.sort(key=angle_from_center)

        # Split into equal chunks
        chunk_size = max(1, len(node_list) // num_splits)
        chunks = []
        for i in range(num_splits):
            start = i * chunk_size
            if i == num_splits - 1:
                # Last chunk gets remainder
                chunks.append(node_list[start:])
            else:
                chunks.append(node_list[start:start + chunk_size])

        # Remove empty chunks
        chunks = [c for c in chunks if c]

        # Assign cluster IDs and recurse
        for chunk in chunks:
            cluster_id = global_counters[level]
            global_counters[level] += 1

            for nid in chunk:
                hierarchy[nid][f'level{level}'] = cluster_id

            # Recurse
            split_balanced(chunk, level + 1)

    # Start recursive splitting
    split_balanced(nodes, 0)

    # Print summary
    for level in range(levels):
        cluster_ids = set(hierarchy[nid][f'level{level}'] for nid in nodes)
        sizes = {}
        for nid in nodes:
            cid = hierarchy[nid][f'level{level}']
            sizes[cid] = sizes.get(cid, 0) + 1
        if sizes:
            min_size = min(sizes.values())
            max_size = max(sizes.values())
            avg_size = sum(sizes.values()) / len(sizes)
            print(f"  Level {level}: {len(cluster_ids)} clusters (size: {min_size}-{max_size}, avg: {avg_size:.1f})")

    return hierarchy


def _quick_layout(nodes, edges, iterations=50):
    """Quick force-directed layout for initial positions."""
    import math
    import random

    random.seed(42)

    # Initialize random positions
    positions = {}
    for nid in nodes:
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, 1000)
        positions[nid] = [r * math.cos(angle), r * math.sin(angle)]

    if not edges:
        return {k: tuple(v) for k, v in positions.items()}

    # Build neighbor map
    neighbors = {nid: [] for nid in nodes}
    node_set = set(nodes)
    for s, t in edges:
        if s in node_set and t in node_set:
            neighbors[s].append(t)
            neighbors[t].append(s)

    # Simple force-directed (Fruchterman-Reingold style)
    k = 1000 / math.sqrt(len(nodes))  # Optimal distance
    temp = 100  # Temperature for simulated annealing

    for iteration in range(iterations):
        # Compute repulsive forces (simplified: random sampling for large graphs)
        displacement = {nid: [0.0, 0.0] for nid in nodes}

        # Sample-based repulsion for large graphs
        sample_size = min(100, len(nodes))
        sample = random.sample(nodes, sample_size) if len(nodes) > sample_size else nodes

        for nid in nodes:
            for other in sample:
                if nid == other:
                    continue
                dx = positions[nid][0] - positions[other][0]
                dy = positions[nid][1] - positions[other][1]
                dist = max(0.1, math.sqrt(dx*dx + dy*dy))
                force = k * k / dist
                displacement[nid][0] += dx / dist * force
                displacement[nid][1] += dy / dist * force

        # Attractive forces (edges)
        for nid in nodes:
            for neighbor in neighbors[nid]:
                dx = positions[nid][0] - positions[neighbor][0]
                dy = positions[nid][1] - positions[neighbor][1]
                dist = max(0.1, math.sqrt(dx*dx + dy*dy))
                force = dist * dist / k
                displacement[nid][0] -= dx / dist * force
                displacement[nid][1] -= dy / dist * force

        # Apply displacement with temperature
        for nid in nodes:
            dx, dy = displacement[nid]
            dist = max(0.1, math.sqrt(dx*dx + dy*dy))
            move = min(dist, temp)
            positions[nid][0] += dx / dist * move
            positions[nid][1] += dy / dist * move

        # Cool down
        temp *= 0.95

    return {k: tuple(v) for k, v in positions.items()}


def compute_hierarchical_layout(nodes, edges, hierarchy, base_radius=300):
    """Compute hierarchical layout with circle packing at each level.

    Level 0: Super-clusters packed in main space
    Level 1: Sub-clusters packed within each super-cluster
    Level 2: Nodes laid out within each sub-cluster

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples
        hierarchy: dict of node_id -> {'level0': int, 'level1': int, 'level2': int}
        base_radius: base radius multiplier

    Returns:
        dict of node_id -> (x, y)
    """
    import math

    print("Computing hierarchical layout...")

    # Group nodes by hierarchy levels
    level0_groups = {}  # level0_id -> [node_ids]
    level1_groups = {}  # (level0_id, level1_id) -> [node_ids]
    level2_groups = {}  # (level0_id, level1_id, level2_id) -> [node_ids]

    for node_id in nodes:
        h = hierarchy[node_id]
        l0, l1, l2 = h['level0'], h['level1'], h['level2']

        if l0 not in level0_groups:
            level0_groups[l0] = []
        level0_groups[l0].append(node_id)

        key1 = (l0, l1)
        if key1 not in level1_groups:
            level1_groups[key1] = []
        level1_groups[key1].append(node_id)

        key2 = (l0, l1, l2)
        if key2 not in level2_groups:
            level2_groups[key2] = []
        level2_groups[key2].append(node_id)

    print(f"  Level 0: {len(level0_groups)} super-clusters")
    print(f"  Level 1: {len(level1_groups)} sub-clusters")
    print(f"  Level 2: {len(level2_groups)} fine clusters")

    # Build edge lookup
    node_edges = {}
    for src, dst in edges:
        if src not in node_edges:
            node_edges[src] = []
        if dst not in node_edges:
            node_edges[dst] = []
        node_edges[src].append(dst)
        node_edges[dst].append(src)

    # Step 1: Compute radii for each level
    # Level 0 radius based on total nodes in super-cluster
    level0_radii = {}
    for l0, node_list in level0_groups.items():
        level0_radii[l0] = base_radius * math.sqrt(len(node_list)) * 2

    # Level 1 radius based on nodes in sub-cluster
    level1_radii = {}
    for key1, node_list in level1_groups.items():
        level1_radii[key1] = base_radius * math.sqrt(len(node_list)) * 0.8

    # Level 2 radius based on nodes in fine cluster
    level2_radii = {}
    for key2, node_list in level2_groups.items():
        level2_radii[key2] = base_radius * math.sqrt(len(node_list)) * 0.4

    # Step 2: Pack level 0 (super-clusters)
    print("  Packing level 0 super-clusters...")
    level0_centers = pack_circles(
        list(level0_groups.keys()),
        {l0: level0_radii[l0] for l0 in level0_groups.keys()},
        padding=1.3
    )

    # Step 3: Pack level 1 within each level 0
    print("  Packing level 1 sub-clusters...")
    level1_centers = {}
    for l0 in level0_groups.keys():
        # Get all level1 clusters within this level0
        l1_in_l0 = [key1 for key1 in level1_groups.keys() if key1[0] == l0]
        if not l1_in_l0:
            continue

        # Pack them within the level0 circle
        l1_ids = [key1[1] for key1 in l1_in_l0]
        l1_radii = {key1[1]: level1_radii[key1] for key1 in l1_in_l0}

        local_centers = pack_circles(l1_ids, l1_radii, padding=1.2)

        # Scale to fit within level0 radius and offset by level0 center
        l0_center = level0_centers[l0]
        l0_radius = level0_radii[l0]

        # Find extent of local layout
        if local_centers:
            max_extent = max(
                math.sqrt(c[0]**2 + c[1]**2) + l1_radii.get(l1_id, 0)
                for l1_id, c in local_centers.items()
            )
            scale = (l0_radius * 0.85) / max_extent if max_extent > 0 else 1

            for key1 in l1_in_l0:
                l1_id = key1[1]
                lc = local_centers.get(l1_id, (0, 0))
                level1_centers[key1] = (
                    l0_center[0] + lc[0] * scale,
                    l0_center[1] + lc[1] * scale
                )

    # Step 4: Pack level 2 within each level 1
    print("  Packing level 2 fine clusters...")
    level2_centers = {}
    for key1 in level1_groups.keys():
        # Get all level2 clusters within this level1
        l2_in_l1 = [key2 for key2 in level2_groups.keys() if key2[:2] == key1]
        if not l2_in_l1:
            continue

        l2_ids = [key2[2] for key2 in l2_in_l1]
        l2_radii = {key2[2]: level2_radii[key2] for key2 in l2_in_l1}

        local_centers = pack_circles(l2_ids, l2_radii, padding=1.2)

        # Scale and offset by level1 center
        l1_center = level1_centers.get(key1, (0, 0))
        l1_radius = level1_radii[key1]

        if local_centers:
            max_extent = max(
                math.sqrt(c[0]**2 + c[1]**2) + l2_radii.get(l2_id, 0)
                for l2_id, c in local_centers.items()
            )
            scale = (l1_radius * 0.85) / max_extent if max_extent > 0 else 1

            for key2 in l2_in_l1:
                l2_id = key2[2]
                lc = local_centers.get(l2_id, (0, 0))
                level2_centers[key2] = (
                    l1_center[0] + lc[0] * scale,
                    l1_center[1] + lc[1] * scale
                )

    # Step 5: Layout nodes within each level 2 cluster
    print("  Computing node layouts within fine clusters...")
    positions = {}
    complex_tasks = []

    for key2, node_list in level2_groups.items():
        center = level2_centers.get(key2, (0, 0))
        radius = level2_radii.get(key2, base_radius)

        if len(node_list) == 1:
            positions[node_list[0]] = center
        elif len(node_list) <= 10:
            # Small cluster: arrange in circle
            for i, node_id in enumerate(node_list):
                angle = 2 * math.pi * i / len(node_list)
                r = radius * 0.6
                positions[node_id] = (
                    center[0] + r * math.cos(angle),
                    center[1] + r * math.sin(angle)
                )
        else:
            # Queue for parallel processing
            cluster_node_set = set(node_list)
            local_edges = {}
            for node_id in node_list:
                if node_id in node_edges:
                    local_edges[node_id] = [n for n in node_edges[node_id] if n in cluster_node_set]
            complex_tasks.append((key2, node_list, local_edges, radius * 0.8, center))

    # Process complex clusters in parallel
    if complex_tasks:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        num_workers = min(multiprocessing.cpu_count(), len(complex_tasks))
        print(f"    Processing {len(complex_tasks)} complex clusters with {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(compute_local_layout_task, task): task[0]
                for task in complex_tasks
            }

            completed = 0
            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    result = future.result()
                    positions.update(result)
                    completed += 1
                    if completed % 50 == 0:
                        print(f"      Completed {completed}/{len(complex_tasks)} clusters...")
                except Exception as e:
                    print(f"      Error in cluster {cluster_id}: {e}")

    print("  Hierarchical layout complete.")
    return positions


def pack_circles(ids, radii, padding=1.3):
    """Simple circle packing for a list of circles.

    Args:
        ids: list of circle identifiers
        radii: dict of id -> radius
        padding: space between circles

    Returns:
        dict of id -> (x, y) center position
    """
    import math

    if not ids:
        return {}

    # Sort by radius descending
    sorted_ids = sorted(ids, key=lambda x: -radii.get(x, 0))
    centers = {}
    placed = []  # [(x, y, radius), ...]

    for i, cid in enumerate(sorted_ids):
        r = radii.get(cid, 100) * padding

        if i == 0:
            centers[cid] = (0.0, 0.0)
            placed.append((0.0, 0.0, r))
        elif i == 1:
            x = placed[0][2] + r
            centers[cid] = (x, 0.0)
            placed.append((x, 0.0, r))
        else:
            # Find best position tangent to two existing circles
            best_pos = None
            best_dist = float('inf')

            for j in range(min(len(placed), 20)):  # Limit search for speed
                for k in range(j + 1, min(len(placed), 20)):
                    c1 = placed[j]
                    c2 = placed[k]
                    positions = find_tangent_positions(c1, c2, r)

                    for pos in positions:
                        valid = True
                        for existing in placed:
                            dist = math.sqrt((pos[0] - existing[0])**2 + (pos[1] - existing[1])**2)
                            if dist < r + existing[2] - 1:
                                valid = False
                                break

                        if valid:
                            dist_to_origin = math.sqrt(pos[0]**2 + pos[1]**2)
                            if dist_to_origin < best_dist:
                                best_dist = dist_to_origin
                                best_pos = pos

            if best_pos is None:
                # Fallback: spiral placement
                angle = i * 0.5
                dist = sum(c[2] for c in placed) / len(placed) * math.sqrt(i)
                best_pos = (dist * math.cos(angle), dist * math.sin(angle))

            centers[cid] = best_pos
            placed.append((best_pos[0], best_pos[1], r))

    return centers


async def compute_and_store_layout():
    """Compute layout and store positions in Neo4j."""
    import math

    # Connect to Neo4j
    db = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "engram_password"),
    )
    await db.connect()

    print("Fetching nodes from Neo4j...")

    # Fetch all nodes with their types (needed for indexed writes)
    concepts = await db.execute_query("MATCH (c:Concept) RETURN c.id as id")
    semantic = await db.execute_query("MATCH (s:SemanticMemory) RETURN s.id as id")
    episodic = await db.execute_query("MATCH (e:EpisodicMemory) RETURN e.id as id")

    # Track node types for indexed writes
    node_types = {}
    for n in concepts:
        node_types[n["id"]] = "Concept"
    for n in semantic:
        node_types[n["id"]] = "SemanticMemory"
    for n in episodic:
        node_types[n["id"]] = "EpisodicMemory"

    all_nodes = list(node_types.keys())
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

    # Detect hierarchical communities using Leiden
    print("Detecting hierarchical communities...")
    hierarchy = compute_hierarchical_communities(all_nodes, all_edges, levels=5)

    # Extract level2 clusters for layout (finest level)
    node_clusters = {node_id: h['level2'] for node_id, h in hierarchy.items()}

    # Compute hierarchical layout
    print("Computing hierarchical layout...")
    positions = compute_hierarchical_layout(
        all_nodes, all_edges, hierarchy,
        base_radius=max(300, 100 * math.sqrt(len(all_nodes) / 1000))
    )

    # Pre-compute connection counts for faster graph loading
    print("Computing connection counts...")
    conn_counts = {node_id: 0 for node_id in all_nodes}
    for src, dst in all_edges:
        if src in conn_counts:
            conn_counts[src] += 1
        if dst in conn_counts:
            conn_counts[dst] += 1

    print("Storing positions, hierarchy, and connection counts in Neo4j...")

    # Group nodes by type for indexed writes (uses per-label indexes)
    by_type = {"Concept": [], "SemanticMemory": [], "EpisodicMemory": []}
    for node_id, (x, y) in positions.items():
        h = hierarchy.get(node_id, {'level0': 0, 'level1': 0, 'level2': 0, 'level3': 0, 'level4': 0})
        conn = conn_counts.get(node_id, 0)
        node_type = node_types.get(node_id, "Concept")
        by_type[node_type].append({
            "id": node_id,
            "x": float(x),
            "y": float(y),
            "cluster": h['level4'],
            "level0": h['level0'],
            "level1": h['level1'],
            "level2": h['level2'],
            "level3": h['level3'],
            "level4": h['level4'],
            "conn": conn,
        })

    # Write each type separately to use per-label indexes (much faster)
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
                SET n.layout_x = row.x, n.layout_y = row.y,
                    n.cluster = row.cluster,
                    n.level0 = row.level0,
                    n.level1 = row.level1,
                    n.level2 = row.level2,
                    n.level3 = row.level3,
                    n.level4 = row.level4,
                    n.conn = row.conn
                """,
                batch=batch,
            )

            written += len(batch)
            print(f"  Stored {written}/{total} positions...")

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
