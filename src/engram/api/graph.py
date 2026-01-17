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

    # Get concepts
    concepts = await db.execute_query(
        """
        MATCH (c:Concept)
        RETURN c.id as id, c.name as name, c.type as type,
               coalesce(c.activation_count, 0) as weight
        LIMIT 500
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

    # Get semantic memories
    memories = await db.execute_query(
        """
        MATCH (s:SemanticMemory)
        RETURN s.id as id, s.content as content, s.memory_type as type,
               s.importance as importance
        LIMIT 300
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

    # Get episodic memories
    episodes = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)
        RETURN e.id as id, e.query as query, e.behavior_name as behavior,
               e.importance as importance
        LIMIT 200
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
               r.type as type, coalesce(r.weight, 0.5) as weight
        LIMIT 1000
        """
    )
    for r in concept_rels:
        links.append({
            "source": r["source"],
            "target": r["target"],
            "type": "concept_rel",
            "weight": r["weight"],
        })

    # Get memory-to-concept relationships
    memory_rels = await db.execute_query(
        """
        MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
        RETURN s.id as source, c.id as target
        LIMIT 1000
        """
    )
    for r in memory_rels:
        links.append({
            "source": r["source"],
            "target": r["target"],
            "type": "memory_concept",
            "weight": 0.3,
        })

    # Get episode-to-concept relationships
    episode_rels = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        RETURN e.id as source, c.id as target
        LIMIT 500
        """
    )
    for r in episode_rels:
        links.append({
            "source": r["source"],
            "target": r["target"],
            "type": "episode_concept",
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
        body { background: #0d1117; font-family: -apple-system, sans-serif; }
        #graph { width: 100vw; height: 100vh; }
        #info { position: absolute; top: 16px; left: 16px; color: #8b949e; z-index: 10; pointer-events: none; }
        #info h1 { font-size: 11px; color: #6e7681; font-weight: 400; text-transform: uppercase; letter-spacing: 3px; margin-top: 6px; }
        #info .brand {
            font-family: 'Orbitron', sans-serif;
            font-size: 32px;
            font-weight: 700;
            color: #7dd3fc;
            text-shadow:
                0 0 10px #0ea5e9,
                0 0 20px #0ea5e9,
                0 0 40px #0ea5e9,
                0 0 80px #0284c7;
            letter-spacing: 4px;
        }
        #legend {
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: rgba(13,17,23,0.9);
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #30363d;
            font-size: 11px;
            color: #8b949e;
            z-index: 100;
        }
        .legend-btn {
            display: flex;
            align-items: center;
            width: 100%;
            padding: 8px 12px;
            margin: 4px 0;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            cursor: pointer;
            font-size: 12px;
            font-family: inherit;
        }
        .legend-btn:hover { background: #30363d; border-color: #484f58; }
        .legend-btn.active { background: #388bfd20; border-color: #58a6ff; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 6px currentColor; }
        .legend-count { margin-left: auto; padding-left: 16px; color: #58a6ff; font-weight: 500; }
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
            background: rgba(13,17,23,0.95);
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
            font-family: inherit;
        }
        #search-input:focus { outline: none; border-color: #58a6ff; }
        #search-input::placeholder { color: #6e7681; }
        #search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 300px;
            overflow-y: auto;
            background: rgba(13,17,23,0.98);
            border: 1px solid #30363d;
            border-top: none;
            border-radius: 0 0 6px 6px;
            display: none;
        }
        #search-results.visible { display: block; }
        .search-result {
            padding: 10px 16px;
            cursor: pointer;
            border-bottom: 1px solid #21262d;
            display: flex;
            align-items: center;
        }
        .search-result:hover { background: #21262d; }
        .search-result:last-child { border-bottom: none; }
        .search-result-name { flex: 1; color: #c9d1d9; }
        .search-result-type { font-size: 10px; padding: 2px 6px; border-radius: 4px; text-transform: uppercase; }
        #stats {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(13,17,23,0.9);
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #30363d;
            font-size: 11px;
            color: #8b949e;
            z-index: 10;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #58a6ff;
            z-index: 10;
            pointer-events: none;
        }
        #node-info {
            position: absolute;
            top: 100px;
            right: 16px;
            width: 280px;
            max-height: 400px;
            overflow-y: auto;
            background: rgba(13,17,23,0.95);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #30363d;
            font-size: 12px;
            color: #c9d1d9;
            display: none;
            z-index: 10;
        }
        #node-info.visible { display: block; }
        #node-info h3 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #fff;
            word-break: break-word;
        }
        #node-info .type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        #node-info .info-row {
            margin: 8px 0;
            padding: 8px 0;
            border-top: 1px solid #21262d;
        }
        #node-info .info-label {
            color: #8b949e;
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        #node-info .info-value {
            color: #c9d1d9;
            word-break: break-word;
        }
        #node-info .close-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: none;
            border: none;
            color: #8b949e;
            cursor: pointer;
            font-size: 16px;
        }
        #node-info .close-btn:hover { color: #fff; }
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
        <button class="legend-btn" data-type="concept" onclick="toggleType('concept')"><span class="legend-dot" style="background:#58a6ff;color:#58a6ff"></span>Concept<span class="legend-count" id="c-count">0</span></button>
        <button class="legend-btn" data-type="semantic" onclick="toggleType('semantic')"><span class="legend-dot" style="background:#a371f7;color:#a371f7"></span>Semantic<span class="legend-count" id="s-count">0</span></button>
        <button class="legend-btn" data-type="episodic" onclick="toggleType('episodic')"><span class="legend-dot" style="background:#f97316;color:#f97316"></span>Episodic<span class="legend-count" id="e-count">0</span></button>
    </div>
    <div id="stats">Loading...</div>
    <div id="loading">Loading graph...</div>

    <script src="https://unpkg.com/force-graph"></script>
    <script>
        const colors = { concept: '#58a6ff', semantic: '#a371f7', episodic: '#f97316' };
        const typeLabels = { concept: 'Concept', semantic: 'Semantic Memory', episodic: 'Episodic Memory' };
        let selected = null;
        let neighbors = new Set();
        let allNodes = {};
        let selectedTypes = new Set();
        let typeNeighbors = new Set();

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
                searchResults.innerHTML = '<div style="padding:12px;color:#8b949e;">No results found</div>';
            } else {
                searchResults.innerHTML = matches.map(n => `
                    <div class="search-result" data-id="${n.id}">
                        <span class="search-result-name">${n.name.length > 40 ? n.name.slice(0,40) + '...' : n.name}</span>
                        <span class="search-result-type" style="background:${colors[n.type]}20;color:${colors[n.type]}">${n.type}</span>
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
            const color = colors[node.type] || '#58a6ff';
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
                                `<div style="margin:4px 0;padding:4px 8px;background:#21262d;border-radius:4px;font-size:11px;">
                                    <span style="color:${colors[n.type]}">\u25CF</span> ${n.name.slice(0,30)}${n.name.length > 30 ? '..' : ''}
                                </div>`
                            ).join('')}
                            ${connected.length > 10 ? `<div style="color:#8b949e;font-size:11px;margin-top:4px;">...and ${connected.length - 10} more</div>` : ''}
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
            .backgroundColor('#0d1117')
            .nodeCanvasObject((node, ctx, globalScale) => {
                const size = Math.sqrt(node.conn || 1) * 4 + 4;
                const color = colors[node.type] || '#58a6ff';
                // Check if active based on node selection OR type filter
                const isNodeActive = !selected || neighbors.has(node.id);
                const isTypeActive = selectedTypes.size === 0 || typeNeighbors.has(node.id);
                const isActive = isNodeActive && isTypeActive;

                // Glow
                if (isActive) {
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 8;
                }

                // Draw node
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
                ctx.fillStyle = isActive ? color : '#21262d';
                ctx.fill();
                ctx.shadowBlur = 0;

                // Draw label - bigger nodes show first, then smaller as you zoom
                const showLabel = (selected && neighbors.has(node.id)) ||
                                  (selectedTypes.size > 0 && typeNeighbors.has(node.id)) ||
                                  (globalScale > 0.4 && size > 12) ||
                                  (globalScale > 0.7 && size > 8) ||
                                  (globalScale > 1.0 && size > 5) ||
                                  (globalScale > 1.5);
                if (showLabel && isActive) {
                    const label = node.name.length > 20 ? node.name.slice(0,20) + '..' : node.name;
                    ctx.font = `${11/globalScale}px -apple-system, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.fillStyle = '#c9d1d9';
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
                // Type filter mode
                if (selectedTypes.size > 0) {
                    const sourceNode = allNodes[sourceId];
                    const targetNode = allNodes[targetId];
                    if (sourceNode && targetNode && selectedTypes.has(sourceNode.type) && selectedTypes.has(targetNode.type)) {
                        return '#58a6ff';
                    }
                    return '#161b22';
                }
                // Node selection mode
                if (!selected) return '#30363d';
                return (sourceId === selected.id || targetId === selected.id) ? '#58a6ff' : '#161b22';
            })
            .linkWidth(l => {
                const sourceId = l.source.id || l.source;
                const targetId = l.target.id || l.target;
                // Type filter mode
                if (selectedTypes.size > 0) {
                    const sourceNode = allNodes[sourceId];
                    const targetNode = allNodes[targetId];
                    if (sourceNode && targetNode && selectedTypes.has(sourceNode.type) && selectedTypes.has(targetNode.type)) {
                        return 1.5;
                    }
                    return 0.2;
                }
                // Node selection mode
                if (!selected) return 0.5;
                return (sourceId === selected.id || targetId === selected.id) ? 2 : 0.2;
            })
            .onNodeClick(node => {
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
            .onZoom(() => refresh())
            .d3VelocityDecay(0.4)
            .cooldownTime(2500);

        // Force refresh function
        function refresh() {
            Graph.nodeColor(Graph.nodeColor());
            Graph.linkColor(Graph.linkColor());
            Graph.linkWidth(Graph.linkWidth());
        }

        // More spacing
        Graph.d3Force('charge').strength(-300);
        Graph.d3Force('link').distance(80);

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
                    '<div>Nodes: <strong style="color:#58a6ff">' + data.nodes.length + '</strong></div>' +
                    '<div>Links: <strong style="color:#58a6ff">' + data.links.length + '</strong></div>';

                // Limit for performance
                if (data.nodes.length > 400) {
                    const concepts = data.nodes.filter(n => n.type === 'concept').slice(0, 250);
                    const semantic = data.nodes.filter(n => n.type === 'semantic').slice(0, 120);
                    const episodic = data.nodes.filter(n => n.type === 'episodic').slice(0, 30);
                    data.nodes = [...concepts, ...semantic, ...episodic];
                    const ids = new Set(data.nodes.map(n => n.id));
                    data.links = data.links.filter(l => ids.has(l.source) && ids.has(l.target));
                    data.nodes.forEach(n => {
                        n.conn = data.links.filter(l => l.source === n.id || l.target === n.id).length;
                    });
                }

                // Store nodes for lookup
                data.nodes.forEach(n => allNodes[n.id] = n);

                Graph.graphData(data);
            });
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    """Serve the 3D graph visualization page."""
    return GRAPH_HTML
