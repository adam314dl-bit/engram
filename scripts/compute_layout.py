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


def spiral_galaxy_layout(positions, node_clusters, arms=4, spread=10000, rotation=2.0):
    """Transform layout into a spiral galaxy shape like Milky Way.

    Args:
        positions: dict of node_id -> (x, y)
        node_clusters: dict of node_id -> cluster_id
        arms: number of spiral arms (default 4)
        spread: how far spiral extends (default 10000)
        rotation: how many times spiral rotates (default 2.0 = 2 full rotations)

    Returns:
        Updated positions dict
    """
    import math
    import random

    print(f"Creating spiral galaxy layout with {arms} arms...")

    # Compute cluster centers and sizes
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

    # Sort clusters by size (larger clusters closer to center)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])

    # Assign each cluster to a position along a spiral arm
    new_cluster_centers = {}
    random.seed(42)

    for i, (cluster_id, count) in enumerate(sorted_clusters):
        # Which arm this cluster belongs to
        arm = i % arms
        # Position along the arm (0 = center, 1 = outer)
        t = i / len(sorted_clusters)

        # Spiral equation: r = a * theta
        # theta increases with t, r increases with t
        theta = arm * (2 * math.pi / arms) + t * rotation * 2 * math.pi
        r = t * spread

        # Add some randomness to make it look natural
        theta += random.uniform(-0.3, 0.3)
        r += random.uniform(-spread * 0.1, spread * 0.1)

        new_x = r * math.cos(theta)
        new_y = r * math.sin(theta)

        new_cluster_centers[cluster_id] = (new_x, new_y)

    # Move each node by the same offset as its cluster
    new_positions = {}
    for node_id, (x, y) in positions.items():
        cluster_id = node_clusters.get(node_id, 0)
        old_center = cluster_centers[cluster_id]
        new_center = new_cluster_centers[cluster_id]

        # Keep relative position within cluster but scale down
        rel_x = (x - old_center[0]) * 0.3  # Tighter clusters
        rel_y = (y - old_center[1]) * 0.3

        new_positions[node_id] = (new_center[0] + rel_x, new_center[1] + rel_y)

    print(f"Spiral galaxy layout complete")
    return new_positions


def gpu_cluster_layout(nodes, edges, node_clusters, base_radius=500, padding=1.4):
    """GPU-accelerated cluster layout using cuGraph ForceAtlas2.

    1. Creates super-graph with clusters as nodes
    2. Uses cuGraph ForceAtlas2 for cluster positions (GPU)
    3. Scales positions to avoid overlap based on cluster radii
    4. Computes local layouts within clusters (parallel CPU)

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples
        node_clusters: dict of node_id -> cluster_id
        base_radius: base radius multiplier
        padding: space between clusters

    Returns:
        dict of node_id -> (x, y)
    """
    import cudf
    import cugraph
    import math

    print("Computing GPU-accelerated cluster layout...")

    # Group nodes by cluster
    cluster_nodes = {}
    for node_id in nodes:
        cluster_id = node_clusters.get(node_id, 0)
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node_id)

    # Compute radius for each cluster
    cluster_radii = {}
    for cluster_id, node_list in cluster_nodes.items():
        cluster_radii[cluster_id] = base_radius * math.sqrt(len(node_list))

    print(f"Clusters: {len(cluster_nodes)}, radius range: {min(cluster_radii.values()):.0f} - {max(cluster_radii.values()):.0f}")

    # Build inter-cluster edge list with weights
    inter_cluster_edges = {}
    for src, dst in edges:
        c1 = node_clusters.get(src, 0)
        c2 = node_clusters.get(dst, 0)
        if c1 != c2:
            key = (min(c1, c2), max(c1, c2))
            inter_cluster_edges[key] = inter_cluster_edges.get(key, 0) + 1

    # Create cluster ID mapping (cuGraph needs integer indices)
    cluster_ids = list(cluster_nodes.keys())
    cluster_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    idx_to_cluster = {i: cid for i, cid in enumerate(cluster_ids)}

    print(f"Inter-cluster edges: {len(inter_cluster_edges)}")

    # Create edge dataframe for cuGraph
    if inter_cluster_edges:
        src_list = [cluster_to_idx[e[0]] for e in inter_cluster_edges.keys()]
        dst_list = [cluster_to_idx[e[1]] for e in inter_cluster_edges.keys()]
        weight_list = list(inter_cluster_edges.values())
        edge_df = cudf.DataFrame({"src": src_list, "dst": dst_list, "weight": weight_list})
    else:
        edge_df = cudf.DataFrame({"src": [], "dst": [], "weight": []})

    # Create cuGraph graph
    G = cugraph.Graph()
    if len(edge_df) > 0:
        G.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight")

    print("Running ForceAtlas2 on GPU...")

    # Run ForceAtlas2 for cluster layout
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

    # Extract positions
    cluster_positions = {}
    for row in positions_df.to_pandas().itertuples():
        cluster_id = idx_to_cluster[row.vertex]
        cluster_positions[cluster_id] = (row.x, row.y)

    # Add isolated clusters (not in any edge)
    import random
    random.seed(42)
    for cluster_id in cluster_ids:
        if cluster_id not in cluster_positions:
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(1000, 5000)
            cluster_positions[cluster_id] = (r * math.cos(angle), r * math.sin(angle))

    print("Scaling cluster positions to avoid overlap...")

    # Scale positions based on cluster radii to avoid overlap
    cluster_centers = separate_cluster_positions(
        cluster_positions, cluster_radii, padding=padding
    )

    print("Computing local layouts (parallel CPU)...")

    # Build edge lookup for local layouts
    node_edges = {}
    for src, dst in edges:
        if src not in node_edges:
            node_edges[src] = []
        if dst not in node_edges:
            node_edges[dst] = []
        node_edges[src].append(dst)
        node_edges[dst].append(src)

    # Compute local layouts (reuse parallel code from circle_packing_layout)
    positions = compute_local_layouts_parallel(
        cluster_nodes, cluster_centers, cluster_radii, node_edges
    )

    print("GPU cluster layout complete.")
    return positions


def separate_cluster_positions(cluster_positions, cluster_radii, padding=1.4, iterations=100):
    """Iteratively separate clusters to avoid overlap based on their radii."""
    import math

    # Convert to mutable positions
    positions = {cid: list(pos) for cid, pos in cluster_positions.items()}
    cluster_ids = list(positions.keys())

    for iteration in range(iterations):
        max_overlap = 0

        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                p1 = positions[c1]
                p2 = positions[c2]
                r1 = cluster_radii[c1] * padding
                r2 = cluster_radii[c2] * padding

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = r1 + r2

                if dist < min_dist and dist > 0:
                    overlap = min_dist - dist
                    max_overlap = max(max_overlap, overlap)

                    # Push apart
                    push = (overlap / 2 + 1) / dist
                    positions[c1][0] -= dx * push
                    positions[c1][1] -= dy * push
                    positions[c2][0] += dx * push
                    positions[c2][1] += dy * push
                elif dist == 0:
                    # Same position, push randomly
                    import random
                    angle = random.uniform(0, 2 * math.pi)
                    push = min_dist / 2
                    positions[c1][0] -= push * math.cos(angle)
                    positions[c1][1] -= push * math.sin(angle)
                    positions[c2][0] += push * math.cos(angle)
                    positions[c2][1] += push * math.sin(angle)

        if max_overlap < 1:
            print(f"  Separation converged after {iteration + 1} iterations")
            break

    return {cid: tuple(pos) for cid, pos in positions.items()}


def compute_local_layouts_parallel(cluster_nodes, cluster_centers, cluster_radii, node_edges):
    """Compute local layouts for all clusters in parallel."""
    import math

    positions = {}
    complex_tasks = []

    for cluster_id, node_list in cluster_nodes.items():
        center = cluster_centers[cluster_id]
        radius = cluster_radii[cluster_id]

        if len(node_list) == 1:
            positions[node_list[0]] = center
        elif len(node_list) <= 10:
            for i, node_id in enumerate(node_list):
                angle = 2 * math.pi * i / len(node_list)
                r = radius * 0.6
                positions[node_id] = (
                    center[0] + r * math.cos(angle),
                    center[1] + r * math.sin(angle)
                )
        else:
            cluster_node_set = set(node_list)
            local_edges = {}
            for node_id in node_list:
                if node_id in node_edges:
                    local_edges[node_id] = [n for n in node_edges[node_id] if n in cluster_node_set]
            complex_tasks.append((cluster_id, node_list, local_edges, radius * 0.8, center))

    if complex_tasks:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        num_workers = min(multiprocessing.cpu_count(), len(complex_tasks))
        print(f"  Processing {len(complex_tasks)} complex clusters with {num_workers} workers...")

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
                    if completed % 100 == 0:
                        print(f"    Completed {completed}/{len(complex_tasks)} clusters...")
                except Exception as e:
                    print(f"    Error in cluster {cluster_id}: {e}")

    return positions


def circle_packing_layout(nodes, edges, node_clusters, base_radius=500, padding=1.3):
    """Circle packing layout: each cluster gets a circle proportional to its size.

    Falls back from GPU to CPU automatically.

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples
        node_clusters: dict of node_id -> cluster_id
        base_radius: base radius multiplier
        padding: space between circles (1.0 = touching, 1.3 = 30% gap)

    Returns:
        dict of node_id -> (x, y)
    """
    # Try GPU first
    try:
        return gpu_cluster_layout(nodes, edges, node_clusters, base_radius, padding)
    except ImportError:
        print("cuGraph not available, using CPU circle packing...")
    except Exception as e:
        print(f"GPU layout failed: {e}, falling back to CPU...")

    import math
    import random

    print("Computing CPU circle packing layout...")

    # Group nodes by cluster
    cluster_nodes = {}
    for node_id in nodes:
        cluster_id = node_clusters.get(node_id, 0)
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node_id)

    # Compute radius for each cluster: r = base * sqrt(node_count)
    cluster_radii = {}
    for cluster_id, node_list in cluster_nodes.items():
        cluster_radii[cluster_id] = base_radius * math.sqrt(len(node_list))

    print(f"Clusters: {len(cluster_nodes)}, radius range: {min(cluster_radii.values()):.0f} - {max(cluster_radii.values()):.0f}")

    # Sort clusters by radius descending (pack largest first)
    sorted_clusters = sorted(cluster_radii.items(), key=lambda x: -x[1])

    # Circle packing using front-chain algorithm
    # Place circles one by one, each tangent to previous circles
    cluster_centers = {}
    placed_circles = []  # [(x, y, radius), ...]

    for i, (cluster_id, radius) in enumerate(sorted_clusters):
        if i == 0:
            # First circle at origin
            cluster_centers[cluster_id] = (0.0, 0.0)
            placed_circles.append((0.0, 0.0, radius * padding))
        elif i == 1:
            # Second circle to the right of first
            x = placed_circles[0][2] + radius * padding
            cluster_centers[cluster_id] = (x, 0.0)
            placed_circles.append((x, 0.0, radius * padding))
        else:
            # Find best position tangent to two existing circles
            best_pos = None
            best_dist = float('inf')

            # Try all pairs of placed circles
            for j in range(len(placed_circles)):
                for k in range(j + 1, len(placed_circles)):
                    c1 = placed_circles[j]
                    c2 = placed_circles[k]

                    # Find positions where new circle is tangent to both
                    positions = find_tangent_positions(c1, c2, radius * padding)

                    for pos in positions:
                        # Check if position overlaps with any existing circle
                        valid = True
                        for existing in placed_circles:
                            dist = math.sqrt((pos[0] - existing[0])**2 + (pos[1] - existing[1])**2)
                            if dist < radius * padding + existing[2] - 1:  # Small tolerance
                                valid = False
                                break

                        if valid:
                            # Prefer positions closer to origin (tighter packing)
                            dist_to_origin = math.sqrt(pos[0]**2 + pos[1]**2)
                            if dist_to_origin < best_dist:
                                best_dist = dist_to_origin
                                best_pos = pos

            if best_pos is None:
                # Fallback: place on expanding spiral
                angle = i * 0.5
                r = sum(c[2] for c in placed_circles) / len(placed_circles) * math.sqrt(i)
                best_pos = (r * math.cos(angle), r * math.sin(angle))

            cluster_centers[cluster_id] = best_pos
            placed_circles.append((best_pos[0], best_pos[1], radius * padding))

        if (i + 1) % 100 == 0:
            print(f"  Placed {i + 1}/{len(sorted_clusters)} clusters...")

    print(f"Circle packing complete. Computing local layouts...")

    # Build edge lookup for local layouts
    node_edges = {}
    for src, dst in edges:
        if src not in node_edges:
            node_edges[src] = []
        if dst not in node_edges:
            node_edges[dst] = []
        node_edges[src].append(dst)
        node_edges[dst].append(src)

    # Prepare tasks for parallel processing
    # Separate into simple (small clusters) and complex (need force-directed)
    positions = {}
    complex_tasks = []

    for cluster_id, node_list in cluster_nodes.items():
        center = cluster_centers[cluster_id]
        radius = cluster_radii[cluster_id]

        if len(node_list) == 1:
            # Single node: place at center
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
            # Extract edges relevant to this cluster
            cluster_node_set = set(node_list)
            local_edges = {}
            for node_id in node_list:
                if node_id in node_edges:
                    local_edges[node_id] = [n for n in node_edges[node_id] if n in cluster_node_set]
            complex_tasks.append((cluster_id, node_list, local_edges, radius * 0.8, center))

    # Process complex clusters in parallel
    if complex_tasks:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        num_workers = min(multiprocessing.cpu_count(), len(complex_tasks))
        print(f"  Processing {len(complex_tasks)} complex clusters with {num_workers} workers...")

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
                    if completed % 100 == 0:
                        print(f"    Completed {completed}/{len(complex_tasks)} clusters...")
                except Exception as e:
                    print(f"    Error in cluster {cluster_id}: {e}")

    print("Circle packing layout complete.")
    return positions


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
    """Detect communities using Louvain algorithm (legacy, for backward compatibility)."""
    hierarchy = compute_hierarchical_communities(nodes, edges, levels=3)
    # Return level 2 (finest level) for backward compatibility
    return {node_id: h['level2'] for node_id, h in hierarchy.items()}


def compute_hierarchical_communities(nodes, edges, levels=3):
    """Detect hierarchical communities using Leiden algorithm.

    Creates a multi-level hierarchy:
    - Level 0: Super-clusters (coarsest, ~10-30 groups)
    - Level 1: Sub-clusters (medium, ~100-500 groups)
    - Level 2: Fine clusters (finest, current granularity)

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples
        levels: number of hierarchy levels (default 3)

    Returns:
        dict of node_id -> {'level0': int, 'level1': int, 'level2': int}
    """
    import igraph as ig

    print(f"Computing {levels}-level hierarchical communities using Leiden...")

    # Create igraph graph
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    edge_indices = []
    for src, dst in edges:
        if src in node_to_idx and dst in node_to_idx:
            edge_indices.append((node_to_idx[src], node_to_idx[dst]))

    G = ig.Graph(n=len(nodes), edges=edge_indices, directed=False)
    print(f"Graph: {G.vcount()} vertices, {G.ecount()} edges")

    # Initialize hierarchy storage
    hierarchy = {node_id: {} for node_id in nodes}

    # Detect communities at each level with different resolutions
    # Higher resolution = more communities (finer granularity)
    resolutions = [0.3, 0.7, 1.5]  # Level 0 (coarse) to Level 2 (fine)

    for level in range(levels):
        resolution = resolutions[level] if level < len(resolutions) else 1.0

        try:
            # Use Leiden algorithm
            partition = G.community_leiden(
                objective_function='modularity',
                resolution=resolution,
                n_iterations=10
            )

            # Assign cluster IDs to nodes
            membership = partition.membership
            num_communities = len(set(membership))

            for idx, cluster_id in enumerate(membership):
                node_id = idx_to_node[idx]
                hierarchy[node_id][f'level{level}'] = cluster_id

            print(f"  Level {level} (resolution={resolution}): {num_communities} communities")

        except Exception as e:
            print(f"  Level {level} Leiden failed: {e}, using single community")
            for node_id in nodes:
                hierarchy[node_id][f'level{level}'] = 0

    # Ensure hierarchical consistency: nodes in same fine cluster should be in same coarse cluster
    hierarchy = make_hierarchy_consistent(hierarchy, levels)

    return hierarchy


def make_hierarchy_consistent(hierarchy, levels):
    """Ensure hierarchical consistency - children always nest within parents.

    If two nodes are in the same level2 cluster, they should be in the same level1 and level0 clusters.
    """
    # Build mapping from fine to coarse clusters
    # For each level2 cluster, find the most common level1 cluster, etc.

    for fine_level in range(levels - 1, 0, -1):
        coarse_level = fine_level - 1
        fine_key = f'level{fine_level}'
        coarse_key = f'level{coarse_level}'

        # Group nodes by fine cluster
        fine_clusters = {}
        for node_id, h in hierarchy.items():
            fine_id = h[fine_key]
            if fine_id not in fine_clusters:
                fine_clusters[fine_id] = []
            fine_clusters[fine_id].append(node_id)

        # For each fine cluster, assign the most common coarse cluster
        for fine_id, node_ids in fine_clusters.items():
            # Count coarse cluster occurrences
            coarse_counts = {}
            for node_id in node_ids:
                coarse_id = hierarchy[node_id][coarse_key]
                coarse_counts[coarse_id] = coarse_counts.get(coarse_id, 0) + 1

            # Find most common
            most_common_coarse = max(coarse_counts, key=coarse_counts.get)

            # Assign to all nodes in this fine cluster
            for node_id in node_ids:
                hierarchy[node_id][coarse_key] = most_common_coarse

    # Renumber clusters to be contiguous at each level
    for level in range(levels):
        level_key = f'level{level}'
        unique_clusters = sorted(set(h[level_key] for h in hierarchy.values()))
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        for node_id in hierarchy:
            hierarchy[node_id][level_key] = cluster_map[hierarchy[node_id][level_key]]

    # Print final stats
    for level in range(levels):
        level_key = f'level{level}'
        num_clusters = len(set(h[level_key] for h in hierarchy.values()))
        print(f"  Final level {level}: {num_clusters} clusters")

    return hierarchy


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

    # Detect hierarchical communities using Leiden
    print("Detecting hierarchical communities...")
    hierarchy = compute_hierarchical_communities(all_nodes, all_edges, levels=3)

    # Extract level2 clusters for layout (finest level)
    node_clusters = {node_id: h['level2'] for node_id, h in hierarchy.items()}

    # Compute hierarchical layout
    print("Computing hierarchical layout...")
    positions = compute_hierarchical_layout(
        all_nodes, all_edges, hierarchy,
        base_radius=max(300, 100 * math.sqrt(len(all_nodes) / 1000))
    )

    print("Storing positions and hierarchy in Neo4j...")

    # Store positions and all hierarchy levels back to Neo4j in batches
    batch_size = 500
    node_list = list(positions.items())

    for i in range(0, len(node_list), batch_size):
        batch = node_list[i : i + batch_size]

        for node_id, (x, y) in batch:
            h = hierarchy.get(node_id, {'level0': 0, 'level1': 0, 'level2': 0})
            await db.execute_query(
                """
                MATCH (n) WHERE n.id = $id
                SET n.layout_x = $x, n.layout_y = $y,
                    n.cluster = $cluster,
                    n.level0 = $level0,
                    n.level1 = $level1,
                    n.level2 = $level2
                """,
                id=node_id,
                x=float(x),
                y=float(y),
                cluster=h['level2'],  # Keep 'cluster' as finest level for backward compat
                level0=h['level0'],
                level1=h['level1'],
                level2=h['level2'],
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
