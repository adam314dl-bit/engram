"""3D Graph visualization endpoint."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db(request: Request) -> Neo4jClient:
    """Get database from app state."""
    return request.app.state.db


@router.get("/admin/graph/data")
async def get_graph_data(request: Request) -> dict:
    """Get graph data for visualization."""
    db = get_db(request)

    nodes = []
    links = []

    # Get concepts (ordered by activation for importance-based LOD)
    concepts = await db.execute_query(
        """
        MATCH (c:Concept)
        RETURN c.id as id, c.name as name, c.type as type,
               coalesce(c.activation_count, 0) as weight
        ORDER BY coalesce(c.activation_count, 0) DESC
        LIMIT 2000
        """
    )
    for c in concepts:
        nodes.append({
            "id": c["id"],
            "name": c["name"],
            "type": "concept",
            "subtype": c["type"] or "unknown",
            "weight": c["weight"] + 1,
        })

    # Get semantic memories (ordered by importance for LOD)
    memories = await db.execute_query(
        """
        MATCH (s:SemanticMemory)
        RETURN s.id as id, s.content as content, s.memory_type as type,
               s.importance as importance
        ORDER BY coalesce(s.importance, 0) DESC
        LIMIT 800
        """
    )
    for m in memories:
        content = m["content"] or ""
        short_content = content[:50] + "..." if len(content) > 50 else content
        nodes.append({
            "id": m["id"],
            "name": short_content,
            "fullContent": content,
            "type": "semantic",
            "subtype": m["type"] or "fact",
            "weight": (m["importance"] or 5) / 2,
        })

    # Get episodic memories (ordered by importance for LOD)
    episodes = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)
        RETURN e.id as id, e.query as query, e.behavior_name as behavior,
               e.importance as importance
        ORDER BY coalesce(e.importance, 0) DESC
        LIMIT 500
        """
    )
    for e in episodes:
        query = e["query"] or ""
        short_query = query[:40] + "..." if len(query) > 40 else query
        nodes.append({
            "id": e["id"],
            "name": short_query,
            "fullContent": query,
            "type": "episodic",
            "subtype": e["behavior"] or "unknown",
            "weight": (e["importance"] or 5) / 2,
        })

    # Get concept-to-concept relationships
    concept_rels = await db.execute_query(
        """
        MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
        RETURN c1.id as source, c2.id as target,
               r.type as relType, coalesce(r.weight, 0.5) as weight
        ORDER BY coalesce(r.weight, 0.5) DESC
        LIMIT 3000
        """
    )
    for r in concept_rels:
        links.append({
            "source": r["source"],
            "target": r["target"],
            "type": "concept_rel",
            "relType": r["relType"] or "related_to",
            "weight": r["weight"],
        })

    # Get memory-to-concept relationships
    memory_rels = await db.execute_query(
        """
        MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
        RETURN s.id as source, c.id as target
        LIMIT 2000
        """
    )
    for r in memory_rels:
        links.append({
            "source": r["source"],
            "target": r["target"],
            "type": "memory_concept",
            "relType": "about",
            "weight": 0.3,
        })

    # Get episode-to-concept relationships
    episode_rels = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        RETURN e.id as source, c.id as target
        LIMIT 1500
        """
    )
    for r in episode_rels:
        links.append({
            "source": r["source"],
            "target": r["target"],
            "type": "episode_concept",
            "relType": "activated",
            "weight": 0.2,
        })

    return {"nodes": nodes, "links": links}


GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Knowledge Graph</title>
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
        /* Controls panel */
        #controls {
            position: absolute;
            bottom: 16px;
            right: 16px;
            width: 180px;
            background: rgba(10,10,18,0.95);
            border: 1px solid #1a1a2e;
            border-radius: 6px;
            padding: 12px;
            z-index: 100;
            box-shadow: 0 0 15px rgba(94,234,212,0.08);
        }
        .control-label {
            font-size: 9px;
            color: #6b6b8a;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .control-group {
            margin-bottom: 12px;
        }
        .control-group:last-child {
            margin-bottom: 0;
        }
        #importance-slider {
            width: 100%;
            height: 4px;
            -webkit-appearance: none;
            background: #1a1a2e;
            border-radius: 2px;
            outline: none;
        }
        #importance-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: #5eead4;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px #5eead4;
        }
        #importance-slider::-moz-range-thumb {
            width: 14px;
            height: 14px;
            background: #5eead4;
            border-radius: 50%;
            cursor: pointer;
            border: none;
            box-shadow: 0 0 10px #5eead4;
        }
        #importance-value {
            font-size: 11px;
            color: #5eead4;
            text-align: right;
            margin-top: 4px;
        }
        /* Path finder */
        #pathfinder {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .path-node {
            display: flex;
            align-items: center;
            padding: 6px 8px;
            background: #0f0f1a;
            border: 1px solid #1a1a2e;
            border-radius: 4px;
            font-size: 10px;
            color: #e0e0ff;
            cursor: pointer;
            transition: all 0.2s;
        }
        .path-node:hover {
            border-color: #5eead440;
        }
        .path-node.selected {
            border-color: #5eead4;
            box-shadow: 0 0 8px rgba(94,234,212,0.15);
        }
        .path-node .label {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .path-node .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            margin-right: 6px;
        }
        #find-path-btn {
            padding: 6px 12px;
            background: #0f0f1a;
            border: 1px solid #1a1a2e;
            border-radius: 4px;
            color: #e0e0ff;
            font-size: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        #find-path-btn:hover {
            background: #1a1a2e;
            border-color: #5eead4;
            box-shadow: 0 0 10px rgba(0,240,255,0.3);
        }
        #find-path-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        #clear-path-btn {
            padding: 6px 8px;
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
            <div class="control-label">Importance Filter</div>
            <input type="range" id="importance-slider" min="0" max="10" step="0.5" value="0">
            <div id="importance-value">Show all</div>
        </div>
        <div class="control-group">
            <div class="control-label">Path Finder</div>
            <div id="pathfinder">
                <div class="path-node" id="path-start" onclick="setPathNode('start')">
                    <span class="dot" style="background:#5eead4"></span>
                    <span class="label">Click to set start</span>
                </div>
                <div class="path-node" id="path-end" onclick="setPathNode('end')">
                    <span class="dot" style="background:#5eead4"></span>
                    <span class="label">Click to set end</span>
                </div>
                <div style="display:flex;gap:4px;">
                    <button id="find-path-btn" onclick="findPath()" disabled>Find Path</button>
                    <button id="clear-path-btn" onclick="clearPath()">Clear</button>
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
        let pathStartNode = null;
        let pathEndNode = null;
        let pathNodes = new Set();
        let pathLinks = new Set();
        let settingPathNode = null;  // 'start' or 'end'

        // LOD (Level of Detail) state
        let currentZoom = 1;
        let lodThresholds = {
            0.3: 100,   // Very zoomed out: top 100 nodes
            0.5: 300,   // Zoomed out: top 300 nodes
            0.8: 600,   // Medium: top 600 nodes
            1.2: 1200,  // Slightly zoomed: top 1200 nodes
            2.0: 2500,  // Zoomed in: top 2500 nodes
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
                importanceValue.textContent = 'Show all';
            } else {
                importanceValue.textContent = `≥ ${importanceThreshold}`;
            }
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

        // Cluster detection using Label Propagation algorithm
        function detectClusters(nodes, links) {
            // Build adjacency list
            const adj = {};
            nodes.forEach(n => adj[n.id] = []);
            links.forEach(l => {
                const s = l.source.id || l.source;
                const t = l.target.id || l.target;
                if (adj[s]) adj[s].push(t);
                if (adj[t]) adj[t].push(s);
            });

            // Initialize each node with its own label
            const labels = {};
            nodes.forEach((n, i) => labels[n.id] = i);

            // Iterate until convergence (max 20 iterations)
            for (let iter = 0; iter < 20; iter++) {
                let changed = false;
                // Shuffle nodes for randomness
                const shuffled = [...nodes].sort(() => Math.random() - 0.5);

                for (const node of shuffled) {
                    const neighbors = adj[node.id];
                    if (neighbors.length === 0) continue;

                    // Count neighbor labels
                    const labelCount = {};
                    neighbors.forEach(nId => {
                        const lbl = labels[nId];
                        labelCount[lbl] = (labelCount[lbl] || 0) + 1;
                    });

                    // Find most common label
                    let maxCount = 0;
                    let maxLabel = labels[node.id];
                    for (const [lbl, count] of Object.entries(labelCount)) {
                        if (count > maxCount) {
                            maxCount = count;
                            maxLabel = parseInt(lbl);
                        }
                    }

                    if (labels[node.id] !== maxLabel) {
                        labels[node.id] = maxLabel;
                        changed = true;
                    }
                }

                if (!changed) break;
            }

            // Map labels to cluster indices
            const uniqueLabels = [...new Set(Object.values(labels))];
            const labelToCluster = {};
            uniqueLabels.forEach((lbl, i) => labelToCluster[lbl] = i);

            // Count nodes per cluster to determine sizes
            const clusterSizes = {};
            nodes.forEach(n => {
                const cluster = labelToCluster[labels[n.id]];
                clusterSizes[cluster] = (clusterSizes[cluster] || 0) + 1;
            });

            // Calculate cluster centers arranged in a spiral/circle pattern
            // Larger clusters get more space, positioned based on size
            const numClusters = uniqueLabels.length;
            const sortedClusters = Object.entries(clusterSizes)
                .sort((a, b) => b[1] - a[1])  // Largest first
                .map(([idx]) => parseInt(idx));

            const baseRadius = Math.sqrt(nodes.length) * 25;  // Base spread radius
            clusterCenters = {};

            sortedClusters.forEach((clusterIdx, i) => {
                // Spiral arrangement - larger clusters near center, smaller ones outside
                const angle = (i / numClusters) * 2 * Math.PI + Math.random() * 0.3;
                const ringIndex = Math.floor(i / 6);  // 6 clusters per ring
                const radius = baseRadius * (0.5 + ringIndex * 0.8);
                clusterCenters[clusterIdx] = {
                    x: Math.cos(angle) * radius,
                    y: Math.sin(angle) * radius,
                    size: clusterSizes[clusterIdx]
                };
            });

            // Assign colors and cluster index to nodes, find cluster representative names
            clusterColors = {};
            nodeCluster = {};
            const clusterBestNode = {};  // cluster -> {node, score}

            nodes.forEach(n => {
                const cluster = labelToCluster[labels[n.id]];
                clusterColors[n.id] = clusterPalette[cluster % clusterPalette.length];
                nodeCluster[n.id] = cluster;
                n.cluster = cluster;  // Store on node for force access

                // Track best node for cluster name (prefer concepts with high weight)
                const score = (n.weight || 0) + (n.conn || 0) * 0.5 + (n.type === 'concept' ? 10 : 0);
                if (!clusterBestNode[cluster] || score > clusterBestNode[cluster].score) {
                    clusterBestNode[cluster] = { node: n, score };
                }
            });

            // Set cluster names from best representative node
            clusterNames = {};
            Object.entries(clusterBestNode).forEach(([cluster, {node}]) => {
                let name = node.name || '';
                // Truncate long names
                if (name.length > 25) name = name.slice(0, 22) + '...';
                clusterNames[cluster] = name;
                clusterCenters[cluster].name = name;
            });

            return uniqueLabels.length;
        }

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

                // Outer glow for hover/selection/path
                if (isOnPath) {
                    ctx.shadowColor = '#5eead4';
                    ctx.shadowBlur = 25;
                } else if (isHovered || isSelected) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = isSelected ? 20 : 15;
                } else if (isActive) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 6;
                }

                // Importance border (thicker for higher weight)
                const borderWidth = Math.min(3, (node.weight || 1) / 2);
                if (isActive && borderWidth > 1) {
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, size + borderWidth, 0, 2 * Math.PI);
                    ctx.fillStyle = isHovered || isSelected ? '#fff' : color;
                    ctx.globalAlpha = 0.3;
                    ctx.fill();
                    ctx.globalAlpha = 1;
                }

                // Draw node
                ctx.beginPath();
                ctx.arc(node.x, node.y, isOnPath ? size * 1.2 : size, 0, 2 * Math.PI);
                ctx.fillStyle = isActive ? color : '#1a1a2e';
                ctx.fill();
                ctx.shadowBlur = 0;

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

                // Draw label - bigger nodes show first, then smaller as you zoom (NOT on hover)
                const showLabel = isOnPath ||
                                  (selected && neighbors.has(node.id)) ||
                                  (selectedTypes.size > 0 && typeNeighbors.has(node.id)) ||
                                  (globalScale > 0.4 && size > 12) ||
                                  (globalScale > 0.7 && size > 8) ||
                                  (globalScale > 1.0 && size > 5) ||
                                  (globalScale > 1.5);
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
            .linkCurvature(0.2)  // Curved edges
            .linkDirectionalParticles(l => {
                // Show particles on path links
                if (pathLinks.has(l)) return 3;
                // Show particles on selected links
                if (!selected) return 0;
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                return (sourceId === selected.id || targetId === selected.id) ? 2 : 0;
            })
            .linkDirectionalParticleWidth(l => pathLinks.has(l) ? 3 : 2)
            .linkDirectionalParticleSpeed(l => pathLinks.has(l) ? 0.008 : 0.005)
            .linkDirectionalParticleColor(() => '#5eead4')
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
            .d3VelocityDecay(0.4)
            .cooldownTime(2500)
            .nodeLabel(null)  // Disable hover tooltip
            .onRenderFramePost((ctx, globalScale) => {
                // Draw cluster labels when zoomed out
                if (currentZoom > 0.6) return;  // Only show when zoomed out

                ctx.save();
                Object.entries(clusterCenters).forEach(([clusterId, center]) => {
                    if (!center.name) return;

                    // Calculate actual center from nodes in this cluster
                    const clusterNodes = Graph.graphData().nodes.filter(n => n.cluster == clusterId);
                    if (clusterNodes.length === 0) return;

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

        // Galaxy layout forces
        // Base repulsion between all nodes
        Graph.d3Force('charge').strength(-150);
        Graph.d3Force('link').distance(50).strength(0.3);

        // Custom cluster force - pulls nodes toward their cluster center
        function clusterForce(alpha) {
            return function(d) {
                if (d.cluster === undefined) return;
                const center = clusterCenters[d.cluster];
                if (!center) return;

                // Pull toward cluster center
                const clusterStrength = 0.15;
                d.vx -= (d.x - center.x) * clusterStrength * alpha;
                d.vy -= (d.y - center.y) * clusterStrength * alpha;
            };
        }

        // Inter-cluster repulsion - push nodes of different clusters apart
        function interClusterForce(alpha) {
            const nodes = Graph.graphData().nodes;
            const repulsionStrength = 800;

            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const a = nodes[i];
                    const b = nodes[j];

                    // Only apply extra repulsion between different clusters
                    if (a.cluster === b.cluster) continue;

                    const dx = a.x - b.x;
                    const dy = a.y - b.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;

                    // Strong repulsion at close range between different clusters
                    if (dist < 300) {
                        const force = (repulsionStrength * alpha) / (dist * dist);
                        const fx = dx / dist * force;
                        const fy = dy / dist * force;

                        a.vx += fx;
                        a.vy += fy;
                        b.vx -= fx;
                        b.vy -= fy;
                    }
                }
            }
        }

        // Apply custom forces after graph data loads
        function applyGalaxyForces() {
            const simulation = Graph.d3Force('link').simulation;
            if (!simulation) {
                // Simulation not ready, try again
                setTimeout(applyGalaxyForces, 100);
                return;
            }

            // Add cluster attraction force
            simulation.force('cluster', (alpha) => {
                Graph.graphData().nodes.forEach(clusterForce(alpha));
            });

            // Add inter-cluster repulsion
            simulation.force('interCluster', interClusterForce);

            // Reheat simulation
            simulation.alpha(1).restart();
        }

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

                // Count connections
                const connCount = {};
                data.links.forEach(l => {
                    connCount[l.source] = (connCount[l.source] || 0) + 1;
                    connCount[l.target] = (connCount[l.target] || 0) + 1;
                });
                data.nodes.forEach(n => n.conn = connCount[n.id] || 0);

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

                // Detect clusters
                const numClusters = detectClusters(data.nodes, data.links);
                document.getElementById('cluster-count').textContent = numClusters;

                // Set initial node positions near their cluster centers
                data.nodes.forEach(n => {
                    const center = clusterCenters[n.cluster];
                    if (center) {
                        // Start near cluster center with some random spread
                        const spread = Math.sqrt(center.size) * 8;
                        n.x = center.x + (Math.random() - 0.5) * spread;
                        n.y = center.y + (Math.random() - 0.5) * spread;
                    }
                });

                Graph.graphData(data);

                // Apply galaxy forces after a short delay
                setTimeout(applyGalaxyForces, 200);
            });
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    """Serve the 3D graph visualization page."""
    return GRAPH_HTML
