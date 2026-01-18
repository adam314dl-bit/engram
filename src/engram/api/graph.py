"""3D Graph visualization endpoint."""

import logging
import math
import random
from collections import defaultdict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

router = APIRouter()


def compute_static_layout(nodes: list, links: list) -> None:
    """Compute static x, y positions for nodes using cluster-based layout.

    This runs server-side so the client doesn't need force simulation.
    Modifies nodes in-place to add x, y coordinates.
    """
    if not nodes:
        return

    # Build adjacency list
    adj: dict[str, list[str]] = defaultdict(list)
    for link in links:
        s, t = link["source"], link["target"]
        adj[s].append(t)
        adj[t].append(s)

    # Label Propagation clustering
    labels = {n["id"]: i for i, n in enumerate(nodes)}

    for _ in range(15):  # Max iterations
        changed = False
        shuffled = nodes.copy()
        random.shuffle(shuffled)

        for node in shuffled:
            nid = node["id"]
            neighbors = adj[nid]
            if not neighbors:
                continue

            # Count neighbor labels
            label_count: dict[int, int] = defaultdict(int)
            for neighbor_id in neighbors:
                if neighbor_id in labels:
                    label_count[labels[neighbor_id]] += 1

            if label_count:
                # Find most common label
                max_label = max(label_count.items(), key=lambda x: x[1])[0]
                if labels[nid] != max_label:
                    labels[nid] = max_label
                    changed = True

        if not changed:
            break

    # Count cluster sizes
    cluster_sizes: dict[int, int] = defaultdict(int)
    for label in labels.values():
        cluster_sizes[label] += 1

    # Merge small clusters (< 3 nodes) into largest neighbor's cluster
    for node in nodes:
        nid = node["id"]
        if cluster_sizes[labels[nid]] < 3:
            neighbors = adj[nid]
            if neighbors:
                best_neighbor = max(
                    (n for n in neighbors if n in labels),
                    key=lambda n: cluster_sizes.get(labels.get(n, 0), 0),
                    default=None,
                )
                if best_neighbor:
                    labels[nid] = labels[best_neighbor]

    # Recount after merging
    cluster_sizes = defaultdict(int)
    for label in labels.values():
        cluster_sizes[label] += 1

    # Map labels to cluster indices (0, 1, 2, ...)
    unique_labels = sorted(set(labels.values()))
    label_to_cluster = {lbl: i for i, lbl in enumerate(unique_labels)}

    # Assign cluster index to nodes
    for node in nodes:
        node["cluster"] = label_to_cluster[labels[node["id"]]]

    # Sort clusters by size (largest first)
    num_clusters = len(unique_labels)
    cluster_order = sorted(
        range(num_clusters),
        key=lambda c: sum(1 for n in nodes if n["cluster"] == c),
        reverse=True,
    )

    # Compute grid layout for clusters
    cols = max(1, math.ceil(math.sqrt(num_clusters)))
    rows = max(1, math.ceil(num_clusters / cols))
    base_spacing = math.sqrt(len(nodes)) * 25  # Spacing between clusters

    cluster_centers: dict[int, tuple[float, float, int]] = {}
    for order_idx, cluster_idx in enumerate(cluster_order):
        size = sum(1 for n in nodes if n["cluster"] == cluster_idx)
        col = order_idx % cols
        row = order_idx // cols

        # Add some jitter to avoid perfect grid
        jitter_x = (random.random() - 0.5) * base_spacing * 0.1
        jitter_y = (random.random() - 0.5) * base_spacing * 0.1

        cx = (col - cols / 2 + 0.5) * base_spacing + jitter_x
        cy = (row - rows / 2 + 0.5) * base_spacing + jitter_y

        cluster_centers[cluster_idx] = (cx, cy, size)

    # Position nodes within their clusters
    # High-connection nodes ("hubs") go near center, others orbit around
    cluster_nodes: dict[int, list] = defaultdict(list)
    for node in nodes:
        cluster_nodes[node["cluster"]].append(node)

    for cluster_idx, cnodes in cluster_nodes.items():
        cx, cy, size = cluster_centers[cluster_idx]

        # Sort by connection count (most connected first)
        cnodes.sort(key=lambda n: n.get("conn", 0), reverse=True)

        # Position using spiral pattern from center outward
        for i, node in enumerate(cnodes):
            if i == 0:
                # Hub node at center
                node["x"] = cx
                node["y"] = cy
            else:
                # Spiral outward - angle increases, radius increases
                angle = i * 2.4  # Golden angle approximation
                # Radius based on position in spiral + some randomness
                base_radius = 15 + math.sqrt(i) * 12
                radius = base_radius + (random.random() - 0.5) * 10

                node["x"] = cx + math.cos(angle) * radius
                node["y"] = cy + math.sin(angle) * radius

# SVG favicon matching the graph theme
FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
<circle cx='50' cy='50' r='40' fill='#0a0a12' stroke='#5eead4' stroke-width='6'/>
<circle cx='50' cy='50' r='15' fill='#5eead4'/>
<circle cx='30' cy='35' r='8' fill='#a78bfa'/>
<circle cx='70' cy='35' r='8' fill='#f472b6'/>
<line x1='50' y1='50' x2='30' y2='35' stroke='#5eead4' stroke-width='2'/>
<line x1='50' y1='50' x2='70' y2='35' stroke='#5eead4' stroke-width='2'/>
</svg>"""


@router.get("/favicon.ico")
async def favicon() -> Response:
    """Return SVG favicon."""
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


def get_db(request: Request) -> Neo4jClient:
    """Get database from app state."""
    return request.app.state.db


@router.get("/admin/graph/data")
async def get_graph_data(request: Request) -> dict:
    """Get graph data for visualization.

    Returns top N most connected nodes with pre-computed positions.
    Static layout - no force simulation needed on client.
    """
    db = get_db(request)

    # Higher limit now that we use static layout (no force simulation)
    MAX_NODES = 3000

    # Get all nodes with their connection counts in one query
    # This lets us pick the most connected nodes across all types
    all_nodes_query = await db.execute_query(
        """
        // Get concepts with connection count
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) as conn
        RETURN c.id as id, c.name as name, 'concept' as type,
               c.type as subtype, coalesce(c.activation_count, 0) + 1 as weight,
               null as fullContent, conn
        ORDER BY conn DESC

        UNION ALL

        // Get semantic memories with connection count
        MATCH (s:SemanticMemory)
        OPTIONAL MATCH (s)-[r]-()
        WITH s, count(r) as conn
        RETURN s.id as id,
               CASE WHEN size(s.content) > 50
                    THEN substring(s.content, 0, 50) + '...'
                    ELSE s.content END as name,
               'semantic' as type,
               coalesce(s.memory_type, 'fact') as subtype,
               coalesce(s.importance, 5) / 2.0 as weight,
               s.content as fullContent, conn
        ORDER BY conn DESC

        UNION ALL

        // Get episodic memories with connection count
        MATCH (e:EpisodicMemory)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as conn
        RETURN e.id as id,
               CASE WHEN size(e.query) > 40
                    THEN substring(e.query, 0, 40) + '...'
                    ELSE e.query END as name,
               'episodic' as type,
               coalesce(e.behavior_name, 'unknown') as subtype,
               coalesce(e.importance, 5) / 2.0 as weight,
               e.query as fullContent, conn
        ORDER BY conn DESC
        """
    )

    # Sort all nodes by connection count and take top N
    all_nodes = sorted(all_nodes_query, key=lambda x: x["conn"] or 0, reverse=True)
    top_nodes = all_nodes[:MAX_NODES]

    # Build node list and ID set for filtering links
    nodes = []
    node_ids = set()
    for n in top_nodes:
        node_ids.add(n["id"])
        node_data = {
            "id": n["id"],
            "name": n["name"] or "unnamed",
            "type": n["type"],
            "subtype": n["subtype"] or "unknown",
            "weight": n["weight"] or 1,
            "conn": n["conn"] or 0,
        }
        if n["fullContent"]:
            node_data["fullContent"] = n["fullContent"]
        nodes.append(node_data)

    # Get links only between nodes we're keeping
    links = []

    # Concept-to-concept relationships
    concept_rels = await db.execute_query(
        """
        MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
        RETURN c1.id as source, c2.id as target,
               r.type as relType, coalesce(r.weight, 0.5) as weight
        """
    )
    for r in concept_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({
                "source": r["source"],
                "target": r["target"],
                "type": "concept_rel",
                "relType": r["relType"] or "related_to",
                "weight": r["weight"],
            })

    # Memory-to-concept relationships
    memory_rels = await db.execute_query(
        """
        MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
        RETURN s.id as source, c.id as target
        """
    )
    for r in memory_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({
                "source": r["source"],
                "target": r["target"],
                "type": "memory_concept",
                "relType": "about",
                "weight": 0.3,
            })

    # Episode-to-concept relationships
    episode_rels = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        RETURN e.id as source, c.id as target
        """
    )
    for r in episode_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({
                "source": r["source"],
                "target": r["target"],
                "type": "episode_concept",
                "relType": "activated",
                "weight": 0.2,
            })

    # Compute static layout (adds x, y, cluster to each node)
    compute_static_layout(nodes, links)

    return {"nodes": nodes, "links": links}


GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Knowledge Graph</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='40' fill='%230a0a12' stroke='%235eead4' stroke-width='6'/><circle cx='50' cy='50' r='15' fill='%235eead4'/><circle cx='30' cy='35' r='8' fill='%23a78bfa'/><circle cx='70' cy='35' r='8' fill='%23f472b6'/><line x1='50' y1='50' x2='30' y2='35' stroke='%235eead4' stroke-width='2'/><line x1='50' y1='50' x2='70' y2='35' stroke='%235eead4' stroke-width='2'/></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; }
        body { background: #0a0a12; font-family: -apple-system, sans-serif; }
        #graph { width: 100vw; height: 100vh; }
        #info { position: absolute; top: 16px; left: 16px; color: #8b949e; z-index: 10; pointer-events: none; }
        #info h1 { font-size: 11px; color: #5eead480; font-weight: 400; text-transform: uppercase; letter-spacing: 3px; margin-top: 6px; }
        #info .brand {
            font-family: 'Orbitron', sans-serif;
            font-size: 32px;
            font-weight: 700;
            color: #5eead4;
            text-shadow: 0 0 20px #5eead440;
            letter-spacing: 4px;
        }
        #legend {
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: rgba(10,10,18,0.95);
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #1a1a2e;
            font-size: 11px;
            color: #8b949e;
            z-index: 100;
            box-shadow: 0 0 15px rgba(94,234,212,0.08);
        }
        .legend-btn {
            display: flex;
            align-items: center;
            width: 100%;
            padding: 8px 12px;
            margin: 4px 0;
            background: #0f0f1a;
            border: 1px solid #1a1a2e;
            border-radius: 6px;
            color: #e0e0ff;
            cursor: pointer;
            font-size: 12px;
            font-family: inherit;
            transition: all 0.2s;
        }
        .legend-btn:hover { background: #1a1a2e; border-color: #5eead440; box-shadow: 0 0 8px rgba(94,234,212,0.15); }
        .legend-btn.active { background: #5eead415; border-color: #5eead4; box-shadow: 0 0 10px rgba(94,234,212,0.2); }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 8px currentColor; }
        .legend-count { margin-left: auto; padding-left: 16px; color: #5eead4; font-weight: 500; }
        #search-box {
            position: absolute;
            top: 16px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
        }
        #search-input {
            width: 300px;
            padding: 10px 16px;
            background: rgba(10,10,18,0.95);
            border: 1px solid #1a1a2e;
            border-radius: 6px;
            color: #e0e0ff;
            font-size: 14px;
            font-family: inherit;
            box-shadow: 0 0 20px rgba(0,240,255,0.05);
        }
        #search-input:focus { outline: none; border-color: #5eead4; box-shadow: 0 0 10px rgba(94,234,212,0.2); }
        #search-input::placeholder { color: #4a4a6a; }
        #search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 300px;
            overflow-y: auto;
            background: rgba(10,10,18,0.98);
            border: 1px solid #1a1a2e;
            border-top: none;
            border-radius: 0 0 6px 6px;
            display: none;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        #search-results.visible { display: block; }
        .search-result {
            padding: 10px 16px;
            cursor: pointer;
            border-bottom: 1px solid #1a1a2e;
            display: flex;
            align-items: center;
        }
        .search-result:hover { background: #1a1a2e; }
        .search-result:last-child { border-bottom: none; }
        .search-result-name { flex: 1; color: #e0e0ff; }
        .search-result-type { font-size: 10px; padding: 2px 6px; border-radius: 4px; text-transform: uppercase; }
        #stats {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(10,10,18,0.95);
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #1a1a2e;
            font-size: 11px;
            color: #8b8ba0;
            z-index: 10;
            box-shadow: 0 0 15px rgba(94,234,212,0.08);
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #5eead4;
            z-index: 10;
            pointer-events: none;
            text-shadow: 0 0 8px #5eead450;
        }
        #node-info {
            position: absolute;
            top: 100px;
            right: 16px;
            width: 280px;
            max-height: 400px;
            overflow-y: auto;
            background: rgba(10,10,18,0.95);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #1a1a2e;
            font-size: 12px;
            color: #e0e0ff;
            display: none;
            z-index: 10;
            box-shadow: 0 0 20px rgba(94,234,212,0.08);
        }
        #node-info.visible { display: block; }
        #node-info h3 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #fff;
            word-break: break-word;
            text-shadow: 0 0 10px rgba(255,255,255,0.3);
        }
        #node-info .type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
            margin-bottom: 12px;
            box-shadow: 0 0 10px currentColor;
        }
        #node-info .info-row {
            margin: 8px 0;
            padding: 8px 0;
            border-top: 1px solid #1a1a2e;
        }
        #node-info .info-label {
            color: #6b6b8a;
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 4px;
            letter-spacing: 1px;
        }
        #node-info .info-value {
            color: #e0e0ff;
            word-break: break-word;
        }
        #node-info .close-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: none;
            border: none;
            color: #6b6b8a;
            cursor: pointer;
            font-size: 16px;
        }
        #node-info .close-btn:hover { color: #5eead4; text-shadow: 0 0 8px #5eead450; }
        #edge-tooltip {
            position: absolute;
            padding: 6px 12px;
            background: rgba(10,10,18,0.95);
            border: 1px solid #1a1a2e;
            border-radius: 6px;
            color: #e0e0ff;
            font-size: 11px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            white-space: nowrap;
            box-shadow: 0 0 10px rgba(94,234,212,0.15);
        }
        #edge-tooltip.visible { display: block; }
        #edge-tooltip .rel-type {
            color: #5eead4;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-shadow: none;
        }
        #edge-tooltip .rel-weight {
            color: #8b8ba0;
            margin-left: 8px;
        }
        /* Controls panel - Cyberpunk style */
        #controls {
            position: absolute;
            bottom: 16px;
            right: 16px;
            width: 200px;
            background: linear-gradient(135deg, rgba(10,10,18,0.95) 0%, rgba(20,15,35,0.95) 100%);
            border: 1px solid #5eead430;
            border-radius: 8px;
            padding: 16px;
            z-index: 100;
            box-shadow: 0 0 20px rgba(94,234,212,0.15), inset 0 0 30px rgba(94,234,212,0.03);
            backdrop-filter: blur(10px);
        }
        #controls::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, #5eead4, transparent);
        }
        .control-label {
            font-size: 10px;
            color: #5eead4;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
            text-shadow: 0 0 10px rgba(94,234,212,0.5);
            font-weight: 600;
        }
        .control-group {
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(94,234,212,0.1);
        }
        .control-group:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }
        .cyber-slider {
            width: 100%;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: linear-gradient(90deg, #1a1a2e 0%, #2a2a4e 100%);
            border-radius: 2px;
            outline: none;
            margin: 8px 0;
        }
        .cyber-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            background: #5eead4;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 12px #5eead4;
            border: none;
            margin-top: -4px;
        }
        .cyber-slider::-moz-range-thumb {
            width: 12px;
            height: 12px;
            background: #5eead4;
            border-radius: 50%;
            cursor: pointer;
            border: none;
            box-shadow: 0 0 12px #5eead4;
        }
        .cyber-slider::-webkit-slider-runnable-track {
            height: 4px;
            border-radius: 2px;
            background: linear-gradient(90deg, #5eead430 0%, #1a1a2e 100%);
        }
        .control-value {
            font-size: 12px;
            color: #5eead4;
            text-align: right;
            font-family: 'Orbitron', monospace;
            text-shadow: 0 0 10px rgba(94,234,212,0.6);
            letter-spacing: 1px;
        }
        .slider-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-row .cyber-slider {
            flex: 1;
        }
        /* Path finder - Cyberpunk style */
        #pathfinder {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .path-node {
            display: flex;
            align-items: center;
            padding: 8px 10px;
            background: linear-gradient(135deg, #0a0a15 0%, #12121f 100%);
            border: 1px solid #5eead420;
            border-radius: 6px;
            font-size: 11px;
            color: #e0e0ff;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .path-node::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(94,234,212,0.1), transparent);
            transition: left 0.5s;
        }
        .path-node:hover::before {
            left: 100%;
        }
        .path-node:hover {
            border-color: #5eead460;
            box-shadow: 0 0 15px rgba(94,234,212,0.1);
        }
        .path-node.selected {
            border-color: #5eead4;
            box-shadow: 0 0 15px rgba(94,234,212,0.3), inset 0 0 20px rgba(94,234,212,0.05);
            background: linear-gradient(135deg, #0f1520 0%, #151a2a 100%);
        }
        .path-node .label {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .path-node .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            box-shadow: 0 0 8px currentColor;
        }
        .cyber-btn {
            padding: 8px 14px;
            background: linear-gradient(135deg, #0a0a15 0%, #15152a 100%);
            border: 1px solid #5eead430;
            border-radius: 6px;
            color: #5eead4;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-shadow: 0 0 8px rgba(94,234,212,0.5);
        }
        .cyber-btn:hover {
            background: linear-gradient(135deg, #10152a 0%, #1a2040 100%);
            border-color: #5eead4;
            box-shadow: 0 0 20px rgba(94,234,212,0.3);
        }
        .cyber-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            text-shadow: none;
        }
        .cyber-btn-ghost {
            background: transparent;
            border-color: transparent;
            color: #6b6b8a;
            text-shadow: none;
        }
        .cyber-btn-ghost:hover {
            color: #f472b6;
            background: transparent;
            border-color: transparent;
            box-shadow: none;
            text-shadow: 0 0 8px rgba(244,114,182,0.5);
        }
        #clear-path-btn {
            padding: 8px 10px;
            background: transparent;
            border: none;
            color: #6b6b8a;
            font-size: 10px;
            cursor: pointer;
        }
        #clear-path-btn:hover {
            color: #f87171;
        }
        #path-result {
            font-size: 10px;
            color: #6b6b8a;
            margin-top: 4px;
            min-height: 14px;
        }
        #path-result.found {
            color: #5eead4;
        }
        #path-result.not-found {
            color: #f87171;
        }
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="info"><div class="brand">Engram</div><h1>Memory Graph</h1></div>
    <div id="search-box">
        <input type="text" id="search-input" placeholder="Search nodes...">
        <div id="search-results"></div>
    </div>
    <div id="node-info">
        <button class="close-btn" onclick="closeNodeInfo()">&times;</button>
        <div id="node-info-content"></div>
    </div>
    <div id="legend">
        <button class="legend-btn" data-type="concept" onclick="toggleType('concept')"><span class="legend-dot" style="background:#5eead4;color:#5eead4"></span>Concept<span class="legend-count" id="c-count">0</span></button>
        <button class="legend-btn" data-type="semantic" onclick="toggleType('semantic')"><span class="legend-dot" style="background:#a78bfa;color:#a78bfa"></span>Semantic<span class="legend-count" id="s-count">0</span></button>
        <button class="legend-btn" data-type="episodic" onclick="toggleType('episodic')"><span class="legend-dot" style="background:#f472b6;color:#f472b6"></span>Episodic<span class="legend-count" id="e-count">0</span></button>
        <div style="border-top:1px solid #1a1a2e;margin:8px 0;"></div>
        <button class="legend-btn" id="cluster-btn" onclick="toggleClusterMode()"><span class="legend-dot" style="background:linear-gradient(135deg,#5eead4,#a78bfa,#f472b6);color:#5eead4"></span>Clusters<span class="legend-count" id="cluster-count">-</span></button>
    </div>
    <div id="stats">Loading...</div>
    <div id="loading">Loading graph...</div>
    <div id="edge-tooltip"><span class="rel-type"></span><span class="rel-weight"></span></div>

    <!-- Controls Panel -->
    <div id="controls">
        <div class="control-group">
            <div class="control-label">Importance</div>
            <div class="slider-row">
                <input type="range" class="cyber-slider" id="importance-slider" min="0" max="10" step="0.5" value="0">
                <div class="control-value" id="importance-value">ALL</div>
            </div>
        </div>
        <div class="control-group">
            <div class="control-label">Cluster Labels</div>
            <div class="slider-row">
                <input type="range" class="cyber-slider" id="cluster-size-slider" min="2" max="50" step="1" value="10">
                <div class="control-value" id="cluster-size-value">≥10</div>
            </div>
        </div>
        <div class="control-group">
            <div class="control-label">Path Finder</div>
            <div id="pathfinder">
                <div class="path-node" id="path-start" onclick="setPathNode('start')">
                    <span class="dot" style="background:#5eead4"></span>
                    <span class="label">Set origin</span>
                </div>
                <div class="path-node" id="path-end" onclick="setPathNode('end')">
                    <span class="dot" style="background:#f472b6"></span>
                    <span class="label">Set target</span>
                </div>
                <div style="display:flex;gap:6px;margin-top:4px;">
                    <button class="cyber-btn" id="find-path-btn" onclick="findPath()" disabled>Trace</button>
                    <button class="cyber-btn cyber-btn-ghost" id="clear-path-btn" onclick="clearPath()">Reset</button>
                </div>
                <div id="path-result"></div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/force-graph"></script>
    <script>
        const typeColors = { concept: '#5eead4', semantic: '#a78bfa', episodic: '#f472b6' };
        const typeLabels = { concept: 'Concept', semantic: 'Semantic Memory', episodic: 'Episodic Memory' };

        // Cluster color palette - soft, distinct colors
        const clusterPalette = [
            '#5eead4', // teal
            '#a78bfa', // purple
            '#f472b6', // pink
            '#fbbf24', // amber
            '#34d399', // emerald
            '#60a5fa', // blue
            '#fb7185', // rose
            '#a3e635', // lime
            '#22d3ee', // cyan
            '#c084fc', // violet
            '#f97316', // orange
            '#2dd4bf', // teal-400
        ];

        let clusterColors = {};  // nodeId -> color
        let nodeCluster = {};    // nodeId -> cluster index
        let clusterCenters = {}; // cluster index -> {x, y, size, name}
        let clusterNames = {};   // cluster index -> name (most important node)
        let colors = typeColors;  // default to type colors
        let useClusterColors = false;

        let selected = null;
        let neighbors = new Set();
        let allNodes = {};
        let selectedTypes = new Set();
        let typeNeighbors = new Set();
        let hoveredNode = null;
        const edgeTooltip = document.getElementById('edge-tooltip');

        // New feature state
        let importanceThreshold = 0;
        let minClusterLabelSize = 10;  // Only show labels for top ~3 clusters initially
        let pathStartNode = null;
        let pathEndNode = null;
        let pathNodes = new Set();
        let pathLinks = new Set();
        let settingPathNode = null;  // 'start' or 'end'

        // LOD (Level of Detail) state
        let currentZoom = 1;
        let lodThresholds = {
            0.2: 200,   // Very zoomed out: top 200 nodes
            0.4: 500,   // Zoomed out: top 500 nodes
            0.6: 800,   // Medium: top 800 nodes
            1.0: 1500,  // Normal: top 1500 nodes
            1.5: 3000,  // Zoomed in: all nodes
            999: 99999  // Very zoomed in: show all
        };
        let nodeRanks = {};  // nodeId -> rank (lower = more important)

        // Get LOD limit based on zoom level
        function getLodLimit() {
            for (const [threshold, limit] of Object.entries(lodThresholds).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
                if (currentZoom < parseFloat(threshold)) {
                    return limit;
                }
            }
            return 99999;
        }

        // Check if node is visible at current LOD
        function isNodeVisibleAtLod(node) {
            const rank = nodeRanks[node.id];
            if (rank === undefined) return true;
            return rank < getLodLimit();
        }

        // Importance slider
        const importanceSlider = document.getElementById('importance-slider');
        const importanceValue = document.getElementById('importance-value');

        importanceSlider.addEventListener('input', (e) => {
            importanceThreshold = parseFloat(e.target.value);
            if (importanceThreshold === 0) {
                importanceValue.textContent = 'ALL';
            } else {
                importanceValue.textContent = `≥${importanceThreshold}`;
            }
            refresh();
        });

        // Cluster size slider
        const clusterSizeSlider = document.getElementById('cluster-size-slider');
        const clusterSizeValue = document.getElementById('cluster-size-value');

        clusterSizeSlider.addEventListener('input', (e) => {
            minClusterLabelSize = parseInt(e.target.value);
            clusterSizeValue.textContent = `≥${minClusterLabelSize}`;
            refresh();
        });

        // Path finder functions
        function setPathNode(type) {
            settingPathNode = type;
            document.getElementById('path-start').classList.toggle('selected', type === 'start');
            document.getElementById('path-end').classList.toggle('selected', type === 'end');
            // Visual hint
            document.body.style.cursor = 'crosshair';
        }

        function updatePathUI() {
            const startEl = document.getElementById('path-start');
            const endEl = document.getElementById('path-end');
            const btn = document.getElementById('find-path-btn');

            if (pathStartNode) {
                startEl.querySelector('.label').textContent = pathStartNode.name.slice(0, 20) + (pathStartNode.name.length > 20 ? '..' : '');
                startEl.querySelector('.dot').style.background = getNodeColor(pathStartNode);
            } else {
                startEl.querySelector('.label').textContent = 'Click to set start';
                startEl.querySelector('.dot').style.background = '#5eead4';
            }

            if (pathEndNode) {
                endEl.querySelector('.label').textContent = pathEndNode.name.slice(0, 20) + (pathEndNode.name.length > 20 ? '..' : '');
                endEl.querySelector('.dot').style.background = getNodeColor(pathEndNode);
            } else {
                endEl.querySelector('.label').textContent = 'Click to set end';
                endEl.querySelector('.dot').style.background = '#5eead4';
            }

            btn.disabled = !(pathStartNode && pathEndNode);
        }

        function findPath() {
            if (!pathStartNode || !pathEndNode) return;

            // BFS to find shortest path
            const links = Graph.graphData().links;
            const adjacency = {};

            // Build adjacency list
            links.forEach(l => {
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                if (!adjacency[sourceId]) adjacency[sourceId] = [];
                if (!adjacency[targetId]) adjacency[targetId] = [];
                adjacency[sourceId].push({ node: targetId, link: l });
                adjacency[targetId].push({ node: sourceId, link: l });
            });

            // BFS
            const queue = [{ node: pathStartNode.id, path: [pathStartNode.id], links: [] }];
            const visited = new Set([pathStartNode.id]);

            while (queue.length > 0) {
                const { node, path, links: pathLinksList } = queue.shift();

                if (node === pathEndNode.id) {
                    // Found path!
                    pathNodes = new Set(path);
                    pathLinks = new Set(pathLinksList);
                    document.getElementById('path-result').textContent = `Path found: ${path.length} nodes`;
                    document.getElementById('path-result').className = 'found';
                    refresh();
                    return;
                }

                const neighbors = adjacency[node] || [];
                for (const { node: neighborId, link } of neighbors) {
                    if (!visited.has(neighborId)) {
                        visited.add(neighborId);
                        queue.push({
                            node: neighborId,
                            path: [...path, neighborId],
                            links: [...pathLinksList, link]
                        });
                    }
                }
            }

            // No path found
            pathNodes.clear();
            pathLinks.clear();
            document.getElementById('path-result').textContent = 'No path found';
            document.getElementById('path-result').className = 'not-found';
            refresh();
        }

        function clearPath() {
            pathStartNode = null;
            pathEndNode = null;
            pathNodes.clear();
            pathLinks.clear();
            settingPathNode = null;
            document.getElementById('path-start').classList.remove('selected');
            document.getElementById('path-end').classList.remove('selected');
            document.getElementById('path-result').textContent = '';
            document.getElementById('path-result').className = '';
            document.body.style.cursor = 'default';
            updatePathUI();
            refresh();
        }

        // NOTE: Cluster detection is now done server-side for performance
        // The server computes clusters and positions, client just renders

        function toggleClusterMode() {
            useClusterColors = !useClusterColors;
            document.getElementById('cluster-btn').classList.toggle('active', useClusterColors);
            refresh();
        }

        function getNodeColor(node) {
            if (useClusterColors && clusterColors[node.id]) {
                return clusterColors[node.id];
            }
            return typeColors[node.type] || '#5eead4';
        }

        function toggleType(type) {
            if (selectedTypes.has(type)) {
                selectedTypes.delete(type);
            } else {
                selectedTypes.add(type);
            }
            // Update legend UI
            document.querySelectorAll('.legend-btn').forEach(el => {
                el.classList.toggle('active', selectedTypes.has(el.dataset.type));
            });
            // Calculate type neighbors (nodes of selected types + their inter-connections)
            updateTypeNeighbors();
            // Clear node selection when using type filter
            if (selectedTypes.size > 0) {
                selected = null;
                neighbors.clear();
                closeNodeInfo();
            }
            refresh();
        }

        // Search functionality
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (query.length < 2) {
                searchResults.classList.remove('visible');
                return;
            }
            const matches = Object.values(allNodes)
                .filter(n => n.name.toLowerCase().includes(query) || (n.fullContent && n.fullContent.toLowerCase().includes(query)))
                .slice(0, 10);
            if (matches.length === 0) {
                searchResults.innerHTML = '<div style="padding:12px;color:#6b6b8a;">No results found</div>';
            } else {
                searchResults.innerHTML = matches.map(n => `
                    <div class="search-result" data-id="${n.id}">
                        <span class="search-result-name">${n.name.length > 40 ? n.name.slice(0,40) + '...' : n.name}</span>
                        <span class="search-result-type" style="background:${getNodeColor(n)}20;color:${getNodeColor(n)}">${n.type}</span>
                    </div>
                `).join('');
            }
            searchResults.classList.add('visible');
        });

        searchResults.addEventListener('click', (e) => {
            const result = e.target.closest('.search-result');
            if (result) {
                const nodeId = result.dataset.id;
                const node = allNodes[nodeId];
                if (node) {
                    selectNode(node);
                    searchInput.value = '';
                    searchResults.classList.remove('visible');
                }
            }
        });

        searchInput.addEventListener('blur', () => {
            setTimeout(() => searchResults.classList.remove('visible'), 200);
        });

        function selectNode(node) {
            // Clear type filters
            selectedTypes.clear();
            typeNeighbors.clear();
            document.querySelectorAll('.legend-btn').forEach(el => el.classList.remove('active'));

            selected = node;
            neighbors = new Set([node.id]);
            Graph.graphData().links.forEach(l => {
                if (l.source.id === node.id) neighbors.add(l.target.id);
                if (l.target.id === node.id) neighbors.add(l.source.id);
            });
            showNodeInfo(node);
            // Center on node
            Graph.centerAt(node.x, node.y, 500);
            Graph.zoom(2, 500);
            refresh();
        }

        function updateTypeNeighbors() {
            typeNeighbors.clear();
            if (selectedTypes.size === 0) return;

            // Add all nodes of selected types
            Object.values(allNodes).forEach(n => {
                if (selectedTypes.has(n.type)) {
                    typeNeighbors.add(n.id);
                }
            });
            // Add nodes connected to selected type nodes (only if they're also selected types)
            Graph.graphData().links.forEach(l => {
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                const sourceNode = allNodes[sourceId];
                const targetNode = allNodes[targetId];
                if (sourceNode && targetNode) {
                    if (selectedTypes.has(sourceNode.type) && selectedTypes.has(targetNode.type)) {
                        typeNeighbors.add(sourceId);
                        typeNeighbors.add(targetId);
                    }
                }
            });
        }

        function showNodeInfo(node) {
            const panel = document.getElementById('node-info');
            const content = document.getElementById('node-info-content');
            const color = getNodeColor(node);
            const typeLabel = typeLabels[node.type] || node.type;

            let html = `
                <span class="type-badge" style="background:${color}20;color:${color}">${typeLabel}</span>
                <h3>${node.name}</h3>
                <div class="info-row">
                    <div class="info-label">Connections</div>
                    <div class="info-value">${node.conn || 0} linked nodes</div>
                </div>
            `;

            if (node.fullContent && node.fullContent !== node.name) {
                html += `
                    <div class="info-row">
                        <div class="info-label">Content</div>
                        <div class="info-value" style="white-space:pre-wrap;max-height:150px;overflow-y:auto;">${node.fullContent}</div>
                    </div>
                `;
            }

            if (node.subtype) {
                html += `
                    <div class="info-row">
                        <div class="info-label">Subtype</div>
                        <div class="info-value">${node.subtype}</div>
                    </div>
                `;
            }

            // Show connected nodes
            const connected = [];
            Graph.graphData().links.forEach(l => {
                if (l.source.id === node.id && allNodes[l.target.id]) {
                    connected.push(allNodes[l.target.id]);
                } else if (l.target.id === node.id && allNodes[l.source.id]) {
                    connected.push(allNodes[l.source.id]);
                }
            });

            if (connected.length > 0) {
                html += `
                    <div class="info-row">
                        <div class="info-label">Connected To</div>
                        <div class="info-value">
                            ${connected.slice(0, 10).map(n =>
                                `<div style="margin:4px 0;padding:4px 8px;background:#0f0f1a;border:1px solid #1a1a2e;border-radius:4px;font-size:11px;">
                                    <span style="color:${getNodeColor(n)}">\u25CF</span> ${n.name.slice(0,30)}${n.name.length > 30 ? '..' : ''}
                                </div>`
                            ).join('')}
                            ${connected.length > 10 ? `<div style="color:#6b6b8a;font-size:11px;margin-top:4px;">...and ${connected.length - 10} more</div>` : ''}
                        </div>
                    </div>
                `;
            }

            content.innerHTML = html;
            panel.classList.add('visible');
        }

        function closeNodeInfo() {
            document.getElementById('node-info').classList.remove('visible');
        }

        const Graph = ForceGraph()
            (document.getElementById('graph'))
            .backgroundColor('#0a0a12')
            .nodeCanvasObject((node, ctx, globalScale) => {
                const size = Math.sqrt(node.conn || 1) * 4 + 4;
                const color = getNodeColor(node);

                // LOD filter - hide nodes below current zoom level threshold
                const visibleAtLod = isNodeVisibleAtLod(node);
                if (!visibleAtLod) {
                    // Don't render nodes outside LOD
                    return;
                }

                // Importance filter
                const meetsImportance = (node.weight || 0) >= importanceThreshold;
                if (!meetsImportance) {
                    // Draw very faint node
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, size * 0.5, 0, 2 * Math.PI);
                    ctx.fillStyle = '#0f0f1a';
                    ctx.fill();
                    return;
                }

                // Check if active based on node selection OR type filter
                const isNodeActive = !selected || neighbors.has(node.id);
                const isTypeActive = selectedTypes.size === 0 || typeNeighbors.has(node.id);
                const isOnPath = pathNodes.has(node.id);
                const isPathEndpoint = node === pathStartNode || node === pathEndNode;
                const isActive = (isNodeActive && isTypeActive) || isOnPath;
                const isHovered = hoveredNode === node;
                const isSelected = selected === node;
                const conn = node.conn || 0;
                const isHub = conn > 8;  // High-connection node = sun
                const isMajorHub = conn > 15;
                const isSuperHub = conn > 25;

                // Sun corona effect for hubs - pulsing synced with particle flow
                if (isHub && isActive) {
                    // Slower pulse synced with particle arrival (~3 seconds per cycle)
                    const time = Date.now() * 0.001;
                    const pulseSpeed = isSuperHub ? 0.8 : isMajorHub ? 0.6 : 0.5;
                    const pulseAmount = isSuperHub ? 0.15 : isMajorHub ? 0.12 : 0.08;
                    const pulse = 1 + Math.sin(time * pulseSpeed * Math.PI) * pulseAmount;

                    // Secondary faster micro-pulse (like particles arriving)
                    const microPulse = 1 + Math.sin(time * 3 + conn * 0.5) * 0.03;

                    const coronaSize = size * (isSuperHub ? 3.5 : isMajorHub ? 2.8 : 2) * pulse * microPulse;

                    // Outer glow with breathing effect
                    const glowIntensity = 0.4 + Math.sin(time * pulseSpeed * Math.PI) * 0.2;
                    const gradient = ctx.createRadialGradient(node.x, node.y, size * 0.3, node.x, node.y, coronaSize);
                    gradient.addColorStop(0, color + Math.floor(glowIntensity * 255).toString(16).padStart(2, '0'));
                    gradient.addColorStop(0.4, color + '25');
                    gradient.addColorStop(0.7, color + '10');
                    gradient.addColorStop(1, 'transparent');
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, coronaSize, 0, 2 * Math.PI);
                    ctx.fillStyle = gradient;
                    ctx.fill();

                    // Extra ring for super hubs - pulses when "receiving" particles
                    if (isSuperHub) {
                        const ringPulse = 1 + Math.sin(time * 1.5) * 0.15;
                        ctx.beginPath();
                        ctx.arc(node.x, node.y, size * 2.2 * ringPulse, 0, 2 * Math.PI);
                        ctx.strokeStyle = color + '50';
                        ctx.lineWidth = 2 + Math.sin(time * 1.5) * 1;
                        ctx.stroke();
                    }
                }

                // Outer glow for hover/selection/path
                if (isOnPath) {
                    ctx.shadowColor = '#5eead4';
                    ctx.shadowBlur = 25;
                } else if (isHovered || isSelected) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = isSelected ? 25 : 18;
                } else if (isSuperHub && isActive) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 30;
                } else if (isMajorHub && isActive) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 22;
                } else if (isHub && isActive) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 14;
                } else if (isActive) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 5;
                }

                // Draw node (hubs are larger based on connection count)
                const nodeSize = isOnPath ? size * 1.2 : isSuperHub ? size * 1.3 : isMajorHub ? size * 1.2 : isHub ? size * 1.1 : size;
                ctx.beginPath();
                ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
                ctx.fillStyle = isActive ? color : '#1a1a2e';
                ctx.fill();
                ctx.shadowBlur = 0;

                // Inner bright core for hubs
                if (isHub && isActive) {
                    const coreSize = isSuperHub ? 0.6 : isMajorHub ? 0.5 : 0.4;
                    const coreAlpha = isSuperHub ? 'cc' : isMajorHub ? 'aa' : '80';
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, nodeSize * coreSize, 0, 2 * Math.PI);
                    ctx.fillStyle = '#ffffff' + coreAlpha;
                    ctx.fill();
                }

                // Path endpoint markers
                if (isPathEndpoint) {
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI);
                    ctx.strokeStyle = '#5eead4';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // White ring on hover/selection
                if (isHovered || isSelected) {
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, size + 2, 0, 2 * Math.PI);
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = isSelected ? 2 : 1.5;
                    ctx.stroke();
                }

                // Draw label - only for important nodes, scales with zoom
                // Stricter thresholds to reduce clutter
                const showLabel = isOnPath ||
                                  (selected && neighbors.has(node.id)) ||
                                  (selectedTypes.size > 0 && typeNeighbors.has(node.id)) ||
                                  (globalScale > 0.5 && size > 14) ||
                                  (globalScale > 1.0 && size > 10) ||
                                  (globalScale > 1.5 && size > 7) ||
                                  (globalScale > 2.5);
                if (showLabel && isActive) {
                    const label = node.name.length > 20 ? node.name.slice(0,20) + '..' : node.name;
                    ctx.font = `${11/globalScale}px -apple-system, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.fillStyle = '#e0e0ff';
                    ctx.fillText(label, node.x, node.y + size + 12/globalScale);
                }
            })
            .nodePointerAreaPaint((node, color, ctx) => {
                const size = Math.sqrt(node.conn || 1) * 4 + 8;
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
            })
            .linkColor(l => {
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                const sourceNode = allNodes[sourceId];
                const targetNode = allNodes[targetId];

                // LOD filter - hide links if either node is not visible
                if (sourceNode && !isNodeVisibleAtLod(sourceNode)) return 'transparent';
                if (targetNode && !isNodeVisibleAtLod(targetNode)) return 'transparent';

                // Importance filter - hide links if either node doesn't meet threshold
                const sourceWeight = sourceNode?.weight || 0;
                const targetWeight = targetNode?.weight || 0;
                if (importanceThreshold > 0) {
                    if (sourceWeight < importanceThreshold && targetWeight < importanceThreshold) {
                        return 'transparent';
                    }
                }

                // Path mode
                if (pathLinks.has(l)) {
                    return '#5eead4';
                }
                // Type filter mode
                if (selectedTypes.size > 0) {
                    if (sourceNode && targetNode && selectedTypes.has(sourceNode.type) && selectedTypes.has(targetNode.type)) {
                        return '#5eead4';
                    }
                    return '#0a0a12';
                }
                // Node selection mode
                if (!selected) return '#3a3a5e';  // More visible default
                return (sourceId === selected.id || targetId === selected.id) ? '#5eead4' : '#1a1a2e';
            })
            .linkWidth(l => {
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                const sourceNode = allNodes[sourceId];
                const targetNode = allNodes[targetId];
                const baseWidth = 1 + (l.weight || 0.5) * 2.5;  // Thicker base

                // LOD filter - hide links if either node is not visible
                if (sourceNode && !isNodeVisibleAtLod(sourceNode)) return 0;
                if (targetNode && !isNodeVisibleAtLod(targetNode)) return 0;

                // Path mode
                if (pathLinks.has(l)) {
                    return 5;
                }
                // Type filter mode
                if (selectedTypes.size > 0) {
                    if (sourceNode && targetNode && selectedTypes.has(sourceNode.type) && selectedTypes.has(targetNode.type)) {
                        return baseWidth * 1.5;
                    }
                    return 0.2;
                }
                // Node selection mode
                if (!selected) return baseWidth;
                return (sourceId === selected.id || targetId === selected.id) ? baseWidth * 2 : 0.2;
            })
            .linkCurvature(0.15)  // Slightly curved edges
            .linkDirectionalParticles(l => {
                // Show particles on path links
                if (pathLinks.has(l)) return 4;

                const sourceNode = allNodes[l.source.id || l.source];
                const targetNode = allNodes[l.target.id || l.target];

                // Hide particles if nodes are filtered by importance
                if (importanceThreshold > 0) {
                    const sourceWeight = sourceNode?.weight || 0;
                    const targetWeight = targetNode?.weight || 0;
                    if (sourceWeight < importanceThreshold && targetWeight < importanceThreshold) {
                        return 0;
                    }
                }

                // Show particles flowing to hub nodes (suns)
                const sourceConn = sourceNode?.conn || 0;
                const targetConn = targetNode?.conn || 0;
                const maxConn = Math.max(sourceConn, targetConn);

                // More particles for connections to bigger hubs
                if (maxConn > 20) return 3;
                if (maxConn > 12) return 2;
                if (maxConn > 6) return 1;

                // Show particles on selected links
                if (selected) {
                    const sourceId = l.source.id || l.source;
                    const targetId = l.target.id || l.target;
                    if (sourceId === selected.id || targetId === selected.id) return 2;
                }
                return 0;
            })
            .linkDirectionalParticleWidth(l => {
                if (pathLinks.has(l)) return 4;
                const sourceNode = allNodes[l.source.id || l.source];
                const targetNode = allNodes[l.target.id || l.target];
                const maxConn = Math.max(sourceNode?.conn || 0, targetNode?.conn || 0);
                return maxConn > 15 ? 3 : maxConn > 8 ? 2.5 : 2;
            })
            .linkDirectionalParticleSpeed(l => {
                if (pathLinks.has(l)) return 0.006;
                const sourceNode = allNodes[l.source.id || l.source];
                const targetNode = allNodes[l.target.id || l.target];
                const maxConn = Math.max(sourceNode?.conn || 0, targetNode?.conn || 0);
                // Slower, more graceful particle movement
                return maxConn > 15 ? 0.003 : maxConn > 8 ? 0.0025 : 0.002;
            })
            .linkDirectionalParticleColor(l => {
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                const sourceNode = allNodes[sourceId];
                const targetNode = allNodes[targetId];

                // Use cluster colors when in cluster mode
                if (useClusterColors) {
                    // Color based on which end is the hub (use that node's cluster color)
                    if ((sourceNode?.conn || 0) > (targetNode?.conn || 0)) {
                        return clusterColors[sourceId] || '#5eead4';
                    }
                    return clusterColors[targetId] || '#5eead4';
                }

                // Default: color based on node type
                if ((sourceNode?.conn || 0) > (targetNode?.conn || 0)) {
                    return typeColors[sourceNode?.type] || '#5eead4';
                }
                return typeColors[targetNode?.type] || '#5eead4';
            })
            .onNodeHover(node => {
                hoveredNode = node;
                document.body.style.cursor = node ? 'pointer' : 'default';
                refresh();
            })
            .onLinkHover((link, prevLink) => {
                if (link) {
                    edgeTooltip.querySelector('.rel-type').textContent = link.relType || 'related';
                    edgeTooltip.querySelector('.rel-weight').textContent = `(${((link.weight || 0.5) * 100).toFixed(0)}%)`;
                    edgeTooltip.classList.add('visible');
                } else {
                    edgeTooltip.classList.remove('visible');
                }
            })
            .onNodeClick(node => {
                // Handle path finder mode
                if (settingPathNode) {
                    if (settingPathNode === 'start') {
                        pathStartNode = node;
                    } else {
                        pathEndNode = node;
                    }
                    settingPathNode = null;
                    document.getElementById('path-start').classList.remove('selected');
                    document.getElementById('path-end').classList.remove('selected');
                    document.body.style.cursor = 'default';
                    updatePathUI();
                    refresh();
                    return;
                }

                // Clear type filter when clicking a node
                selectedTypes.clear();
                typeNeighbors.clear();
                document.querySelectorAll('.legend-btn').forEach(el => el.classList.remove('active'));

                if (selected === node) {
                    selected = null;
                    neighbors.clear();
                    closeNodeInfo();
                } else {
                    selected = node;
                    neighbors = new Set([node.id]);
                    Graph.graphData().links.forEach(l => {
                        if (l.source.id === node.id) neighbors.add(l.target.id);
                        if (l.target.id === node.id) neighbors.add(l.source.id);
                    });
                    showNodeInfo(node);
                }
                refresh();
            })
            .onBackgroundClick(() => {
                selected = null;
                neighbors.clear();
                selectedTypes.clear();
                typeNeighbors.clear();
                document.querySelectorAll('.legend-btn').forEach(el => el.classList.remove('active'));
                closeNodeInfo();
                refresh();
            })
            .onZoom(({k}) => {
                currentZoom = k;
                refresh();
            })
            .nodeLabel(null)  // Disable hover tooltip
            .onRenderFramePost((ctx, globalScale) => {
                // Draw cluster labels when zoomed out
                if (currentZoom > 1.2) return;  // Show labels when not too zoomed in

                ctx.save();
                Object.entries(clusterCenters).forEach(([clusterId, center]) => {
                    if (!center.name) return;

                    // Calculate actual center from nodes in this cluster
                    const clusterNodes = Graph.graphData().nodes.filter(n => n.cluster == clusterId);
                    if (clusterNodes.length < minClusterLabelSize) return;  // Skip small clusters (user controlled)

                    let cx = 0, cy = 0;
                    clusterNodes.forEach(n => { cx += n.x; cy += n.y; });
                    cx /= clusterNodes.length;
                    cy /= clusterNodes.length;

                    // Draw cluster label with glow
                    const color = clusterPalette[clusterId % clusterPalette.length];
                    const fontSize = Math.max(16, 24 / currentZoom);

                    // Glow effect
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 20;

                    // Background pill
                    ctx.font = `bold ${fontSize}px -apple-system, sans-serif`;
                    const textWidth = ctx.measureText(center.name).width;
                    const padding = fontSize * 0.4;

                    ctx.fillStyle = 'rgba(10, 10, 18, 0.85)';
                    ctx.beginPath();
                    ctx.roundRect(
                        cx - textWidth/2 - padding,
                        cy - fontSize/2 - padding * 0.6,
                        textWidth + padding * 2,
                        fontSize + padding * 1.2,
                        fontSize * 0.3
                    );
                    ctx.fill();

                    // Border
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Text
                    ctx.shadowBlur = 0;
                    ctx.fillStyle = color;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(center.name, cx, cy);
                });
                ctx.restore();
            });

        // Position edge tooltip at mouse
        document.getElementById('graph').addEventListener('mousemove', e => {
            edgeTooltip.style.left = (e.clientX + 15) + 'px';
            edgeTooltip.style.top = (e.clientY + 15) + 'px';
        });

        // Force refresh function
        function refresh() {
            Graph.nodeColor(Graph.nodeColor());
            Graph.linkColor(Graph.linkColor());
            Graph.linkWidth(Graph.linkWidth());
        }

        // STATIC LAYOUT: Disable all forces - positions come from server
        // This dramatically improves performance for large graphs
        Graph.cooldownTicks(0);  // No simulation iterations
        Graph.d3Force('charge', null);
        Graph.d3Force('link', null);
        Graph.d3Force('center', null);

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') {
                selected = null;
                neighbors.clear();
                selectedTypes.clear();
                typeNeighbors.clear();
                document.querySelectorAll('.legend-btn').forEach(el => el.classList.remove('active'));
                closeNodeInfo();
                searchInput.value = '';
                searchResults.classList.remove('visible');
                // Clear path finder
                clearPath();
                refresh();
            }
        });

        fetch('/admin/graph/data')
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';

                // Nodes come with x, y, cluster from server (static layout)
                const c = data.nodes.filter(n => n.type === 'concept').length;
                const s = data.nodes.filter(n => n.type === 'semantic').length;
                const e = data.nodes.filter(n => n.type === 'episodic').length;
                document.getElementById('c-count').textContent = c;
                document.getElementById('s-count').textContent = s;
                document.getElementById('e-count').textContent = e;
                document.getElementById('stats').innerHTML =
                    '<div>Nodes: <strong style="color:#5eead4;text-shadow:0 0 5px #5eead4">' + data.nodes.length + '</strong></div>' +
                    '<div>Links: <strong style="color:#5eead4;text-shadow:0 0 5px #5eead4">' + data.links.length + '</strong></div>';

                // Compute node ranks by importance (weight + connections)
                // Lower rank = more important = shown earlier when zoomed out
                const sortedNodes = [...data.nodes].sort((a, b) => {
                    const scoreA = (a.weight || 0) + (a.conn || 0) * 0.5;
                    const scoreB = (b.weight || 0) + (b.conn || 0) * 0.5;
                    return scoreB - scoreA;  // Higher score = lower rank
                });
                sortedNodes.forEach((n, i) => nodeRanks[n.id] = i);

                // Store nodes for lookup
                data.nodes.forEach(n => allNodes[n.id] = n);

                // Use cluster info from server to set up client-side cluster colors
                // Server already computed clusters, just need to assign colors
                const clusterSet = new Set(data.nodes.map(n => n.cluster));
                const numClusters = clusterSet.size;
                document.getElementById('cluster-count').textContent = numClusters;

                // Assign cluster colors and compute cluster centers from actual positions
                data.nodes.forEach(n => {
                    clusterColors[n.id] = clusterPalette[n.cluster % clusterPalette.length];
                    nodeCluster[n.id] = n.cluster;
                });

                // Compute cluster centers from actual node positions
                const clusterNodes = {};
                data.nodes.forEach(n => {
                    if (!clusterNodes[n.cluster]) clusterNodes[n.cluster] = [];
                    clusterNodes[n.cluster].push(n);
                });

                // Find best representative name for each cluster
                Object.entries(clusterNodes).forEach(([clusterId, nodes]) => {
                    // Calculate center
                    let cx = 0, cy = 0;
                    nodes.forEach(n => { cx += n.x; cy += n.y; });
                    cx /= nodes.length;
                    cy /= nodes.length;

                    // Find best node for cluster name (prefer high-connection concepts)
                    let bestNode = nodes[0];
                    let bestScore = 0;
                    nodes.forEach(n => {
                        const score = (n.weight || 0) + (n.conn || 0) * 0.5 + (n.type === 'concept' ? 10 : 0);
                        if (score > bestScore) {
                            bestScore = score;
                            bestNode = n;
                        }
                    });

                    let name = bestNode.name || '';
                    if (name.length > 25) name = name.slice(0, 22) + '...';

                    clusterCenters[clusterId] = {
                        x: cx,
                        y: cy,
                        size: nodes.length,
                        name: name
                    };
                    clusterNames[clusterId] = name;
                });

                // Fix node positions (prevent any movement)
                data.nodes.forEach(n => {
                    n.fx = n.x;
                    n.fy = n.y;
                });

                Graph.graphData(data);

                // Zoom to dense area immediately (no simulation to wait for)
                setTimeout(() => {
                    // Find top connected nodes (the "suns")
                    const sortedByConn = [...data.nodes].sort((a, b) => (b.conn || 0) - (a.conn || 0));
                    const topNodes = sortedByConn.slice(0, 10);  // Top 10 hubs

                    if (topNodes.length > 0) {
                        // Calculate centroid of top hubs
                        let cx = 0, cy = 0, totalWeight = 0;
                        topNodes.forEach(n => {
                            const weight = n.conn || 1;
                            cx += (n.x || 0) * weight;
                            cy += (n.y || 0) * weight;
                            totalWeight += weight;
                        });
                        cx /= totalWeight;
                        cy /= totalWeight;

                        // Zoom to the dense area
                        Graph.centerAt(cx, cy, 800);
                        Graph.zoom(0.55, 800);
                    }
                }, 100);  // Short delay for render
            });
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    """Serve the 3D graph visualization page."""
    return GRAPH_HTML
