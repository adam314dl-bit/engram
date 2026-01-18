"""Graph visualization with WebGL (3D force-graph) for better performance."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

router = APIRouter()

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
    """Get 75% most important data (by connection count)."""
    db = get_db(request)

    nodes = []
    links = []

    # Get top 75% concepts by connection count
    concepts = await db.execute_query(
        """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) as conn
        ORDER BY conn DESC
        LIMIT 1500
        RETURN c.id as id, c.name as name, c.type as type, conn
        """
    )
    for c in concepts:
        nodes.append({
            "id": c["id"],
            "name": c["name"],
            "type": "concept",
            "subtype": c["type"] or "unknown",
            "conn": c["conn"] or 0,
        })

    # Get top 75% semantic memories by connection count
    memories = await db.execute_query(
        """
        MATCH (s:SemanticMemory)
        OPTIONAL MATCH (s)-[r]-()
        WITH s, count(r) as conn
        ORDER BY conn DESC
        LIMIT 600
        RETURN s.id as id, s.content as content, s.memory_type as type, conn
        """
    )
    for m in memories:
        content = m["content"] or ""
        short = content[:50] + "..." if len(content) > 50 else content
        nodes.append({
            "id": m["id"],
            "name": short,
            "fullContent": content,
            "type": "semantic",
            "subtype": m["type"] or "fact",
            "conn": m["conn"] or 0,
        })

    # Get top 75% episodic memories by connection count
    episodes = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as conn
        ORDER BY conn DESC
        LIMIT 375
        RETURN e.id as id, e.query as query, e.behavior_name as behavior, conn
        """
    )
    for e in episodes:
        query = e["query"] or ""
        short = query[:40] + "..." if len(query) > 40 else query
        nodes.append({
            "id": e["id"],
            "name": short,
            "fullContent": query,
            "type": "episodic",
            "subtype": e["behavior"] or "unknown",
            "conn": e["conn"] or 0,
        })

    # Get node IDs for filtering links
    node_ids = {n["id"] for n in nodes}

    # Get relationships between the selected nodes (most important links)
    concept_rels = await db.execute_query(
        """
        MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
        RETURN c1.id as source, c2.id as target, r.type as relType, coalesce(r.weight, 0.5) as weight
        ORDER BY r.weight DESC
        LIMIT 3750
        """
    )
    for r in concept_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({
                "source": r["source"],
                "target": r["target"],
                "relType": r["relType"] or "related_to",
            })

    memory_rels = await db.execute_query(
        """
        MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
        RETURN s.id as source, c.id as target
        LIMIT 2250
        """
    )
    for r in memory_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({
                "source": r["source"],
                "target": r["target"],
                "relType": "about",
            })

    episode_rels = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        RETURN e.id as source, c.id as target
        LIMIT 1500
        """
    )
    for r in episode_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({
                "source": r["source"],
                "target": r["target"],
                "relType": "activated",
            })

    return {"nodes": nodes, "links": links}


GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Knowledge Graph</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='40' fill='%230a0a12' stroke='%235eead4' stroke-width='6'/><circle cx='50' cy='50' r='15' fill='%235eead4'/></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; }
        body { background: #0a0a12; font-family: -apple-system, sans-serif; overflow: hidden; }
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
        .legend-btn:hover { background: #1a1a2e; border-color: #5eead440; }
        .legend-btn.active { background: #5eead415; border-color: #5eead4; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
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
        }
        #search-input:focus { outline: none; border-color: #5eead4; }
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
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #5eead4;
            z-index: 10;
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
        }
        #node-info.visible { display: block; }
        #node-info h3 { margin: 0 0 12px 0; font-size: 14px; color: #fff; word-break: break-word; }
        #node-info .type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        #node-info .info-row { margin: 8px 0; padding: 8px 0; border-top: 1px solid #1a1a2e; }
        #node-info .info-label { color: #6b6b8a; font-size: 10px; text-transform: uppercase; margin-bottom: 4px; }
        #node-info .info-value { color: #e0e0ff; word-break: break-word; }
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
        #node-info .close-btn:hover { color: #5eead4; }
        #mode-indicator {
            position: absolute;
            bottom: 16px;
            right: 16px;
            background: rgba(94,234,212,0.1);
            border: 1px solid #5eead4;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 10px;
            color: #5eead4;
            z-index: 10;
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
        <button class="legend-btn" data-type="concept" onclick="toggleType('concept')"><span class="legend-dot" style="background:#5eead4"></span>Concept<span class="legend-count" id="c-count">0</span></button>
        <button class="legend-btn" data-type="semantic" onclick="toggleType('semantic')"><span class="legend-dot" style="background:#a78bfa"></span>Semantic<span class="legend-count" id="s-count">0</span></button>
        <button class="legend-btn" data-type="episodic" onclick="toggleType('episodic')"><span class="legend-dot" style="background:#f472b6"></span>Episodic<span class="legend-count" id="e-count">0</span></button>
        <div style="border-top:1px solid #1a1a2e;margin:8px 0;"></div>
        <button class="legend-btn" id="cluster-btn" onclick="toggleClusterMode()"><span class="legend-dot" style="background:linear-gradient(135deg,#5eead4,#a78bfa,#f472b6)"></span>Clusters<span class="legend-count" id="cluster-count">-</span></button>
    </div>
    <div id="stats">Loading...</div>
    <div id="loading">Loading graph (WebGL)...</div>
    <div id="mode-indicator">WebGL Mode</div>

    <script src="https://unpkg.com/3d-force-graph"></script>
    <script>
        const typeColors = { concept: '#5eead4', semantic: '#a78bfa', episodic: '#f472b6' };
        const typeLabels = { concept: 'Concept', semantic: 'Semantic Memory', episodic: 'Episodic Memory' };

        const clusterPalette = [
            '#5eead4', '#a78bfa', '#f472b6', '#fbbf24', '#34d399',
            '#60a5fa', '#fb7185', '#a3e635', '#22d3ee', '#c084fc',
            '#f97316', '#2dd4bf'
        ];

        let clusterColors = {};
        let nodeCluster = {};
        let useClusterColors = false;

        let selected = null;
        let neighbors = new Set();
        let allNodes = {};
        let selectedTypes = new Set();
        let typeNeighbors = new Set();

        function getNodeColor(node) {
            if (useClusterColors && clusterColors[node.id]) {
                return clusterColors[node.id];
            }
            return typeColors[node.type] || '#5eead4';
        }

        function toggleClusterMode() {
            useClusterColors = !useClusterColors;
            document.getElementById('cluster-btn').classList.toggle('active', useClusterColors);
            Graph.nodeColor(n => getNodeColor(n));
        }

        function detectClusters(nodes, links) {
            const adj = {};
            nodes.forEach(n => adj[n.id] = []);
            links.forEach(l => {
                const s = l.source.id || l.source;
                const t = l.target.id || l.target;
                if (adj[s]) adj[s].push(t);
                if (adj[t]) adj[t].push(s);
            });

            const labels = {};
            nodes.forEach((n, i) => labels[n.id] = i);

            for (let iter = 0; iter < 20; iter++) {
                let changed = false;
                const shuffled = [...nodes].sort(() => Math.random() - 0.5);

                for (const node of shuffled) {
                    const neighbors = adj[node.id];
                    if (neighbors.length === 0) continue;

                    const labelCount = {};
                    neighbors.forEach(nId => {
                        const lbl = labels[nId];
                        labelCount[lbl] = (labelCount[lbl] || 0) + 1;
                    });

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

            const labelSizes = {};
            nodes.forEach(n => {
                const lbl = labels[n.id];
                labelSizes[lbl] = (labelSizes[lbl] || 0) + 1;
            });

            nodes.forEach(n => {
                if (labelSizes[labels[n.id]] < 3) {
                    const neighbors = adj[n.id];
                    if (neighbors.length > 0) {
                        let bestNeighbor = neighbors[0];
                        let bestSize = labelSizes[labels[bestNeighbor]] || 0;
                        for (const nId of neighbors) {
                            const size = labelSizes[labels[nId]] || 0;
                            if (size > bestSize) {
                                bestSize = size;
                                bestNeighbor = nId;
                            }
                        }
                        labels[n.id] = labels[bestNeighbor];
                    }
                }
            });

            const uniqueLabels = [...new Set(Object.values(labels))];
            const labelToCluster = {};
            uniqueLabels.forEach((lbl, i) => labelToCluster[lbl] = i);

            clusterColors = {};
            nodeCluster = {};
            nodes.forEach(n => {
                const cluster = labelToCluster[labels[n.id]];
                clusterColors[n.id] = clusterPalette[cluster % clusterPalette.length];
                nodeCluster[n.id] = cluster;
            });

            return uniqueLabels.length;
        }

        function toggleType(type) {
            if (selectedTypes.has(type)) {
                selectedTypes.delete(type);
            } else {
                selectedTypes.add(type);
            }
            document.querySelectorAll('.legend-btn[data-type]').forEach(el => {
                el.classList.toggle('active', selectedTypes.has(el.dataset.type));
            });
            updateTypeNeighbors();
            if (selectedTypes.size > 0) {
                selected = null;
                neighbors.clear();
                closeNodeInfo();
            }
            updateVisibility();
        }

        function updateTypeNeighbors() {
            typeNeighbors.clear();
            if (selectedTypes.size === 0) return;
            Object.values(allNodes).forEach(n => {
                if (selectedTypes.has(n.type)) typeNeighbors.add(n.id);
            });
        }

        function updateVisibility() {
            Graph
                .nodeColor(n => {
                    if (selectedTypes.size > 0 && !selectedTypes.has(n.type)) return '#1a1a2e';
                    if (selected && !neighbors.has(n.id)) return '#1a1a2e';
                    return getNodeColor(n);
                })
                .linkColor(l => {
                    const sourceId = l.source.id || l.source;
                    const targetId = l.target.id || l.target;
                    if (selectedTypes.size > 0) {
                        const sn = allNodes[sourceId];
                        const tn = allNodes[targetId];
                        if (sn && tn && selectedTypes.has(sn.type) && selectedTypes.has(tn.type)) {
                            return 'rgba(94,234,212,0.6)';
                        }
                        return 'rgba(0,0,0,0)';
                    }
                    if (selected) {
                        if (sourceId === selected.id || targetId === selected.id) {
                            return 'rgba(94,234,212,0.8)';
                        }
                        return 'rgba(0,0,0,0)';
                    }
                    return 'rgba(94,234,212,0.15)';
                });
        }

        // Search
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
                const node = allNodes[result.dataset.id];
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
            selectedTypes.clear();
            typeNeighbors.clear();
            document.querySelectorAll('.legend-btn[data-type]').forEach(el => el.classList.remove('active'));
            selected = node;
            neighbors = new Set([node.id]);
            Graph.graphData().links.forEach(l => {
                const sid = l.source.id || l.source;
                const tid = l.target.id || l.target;
                if (sid === node.id) neighbors.add(tid);
                if (tid === node.id) neighbors.add(sid);
            });
            showNodeInfo(node);
            Graph.cameraPosition({ x: node.x, y: node.y, z: 300 }, node, 1000);
            updateVisibility();
        }

        function showNodeInfo(node) {
            const panel = document.getElementById('node-info');
            const content = document.getElementById('node-info-content');
            const color = getNodeColor(node);

            let html = `
                <span class="type-badge" style="background:${color}20;color:${color}">${typeLabels[node.type] || node.type}</span>
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

            const connected = [];
            Graph.graphData().links.forEach(l => {
                const sid = l.source.id || l.source;
                const tid = l.target.id || l.target;
                if (sid === node.id && allNodes[tid]) connected.push(allNodes[tid]);
                else if (tid === node.id && allNodes[sid]) connected.push(allNodes[sid]);
            });

            if (connected.length > 0) {
                html += `
                    <div class="info-row">
                        <div class="info-label">Connected To</div>
                        <div class="info-value">
                            ${connected.slice(0, 8).map(n =>
                                `<div style="margin:4px 0;padding:4px 8px;background:#0f0f1a;border-radius:4px;font-size:11px;">
                                    <span style="color:${getNodeColor(n)}">●</span> ${n.name.slice(0,30)}${n.name.length > 30 ? '..' : ''}
                                </div>`
                            ).join('')}
                            ${connected.length > 8 ? `<div style="color:#6b6b8a;font-size:11px;margin-top:4px;">...and ${connected.length - 8} more</div>` : ''}
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

        // Initialize 3D Force Graph with WebGL
        const Graph = ForceGraph3D()
            (document.getElementById('graph'))
            .backgroundColor('#0a0a12')
            .nodeColor(n => getNodeColor(n))
            .nodeVal(n => Math.max(1, Math.sqrt(n.conn || 1) * 2))
            .nodeOpacity(0.9)
            .linkColor(() => 'rgba(94,234,212,0.15)')
            .linkOpacity(0.6)
            .linkWidth(0.5)
            .linkDirectionalParticles(2)
            .linkDirectionalParticleWidth(1.5)
            .linkDirectionalParticleSpeed(0.005)
            .linkDirectionalParticleColor(() => '#5eead4')
            .onNodeClick(node => {
                selectedTypes.clear();
                typeNeighbors.clear();
                document.querySelectorAll('.legend-btn[data-type]').forEach(el => el.classList.remove('active'));

                if (selected === node) {
                    selected = null;
                    neighbors.clear();
                    closeNodeInfo();
                    updateVisibility();
                } else {
                    selectNode(node);
                }
            })
            .onBackgroundClick(() => {
                selected = null;
                neighbors.clear();
                selectedTypes.clear();
                typeNeighbors.clear();
                document.querySelectorAll('.legend-btn[data-type]').forEach(el => el.classList.remove('active'));
                closeNodeInfo();
                updateVisibility();
            })
            .cooldownTime(3000)
            .d3VelocityDecay(0.3)
            .nodeLabel(n => n.name);

        // Configure forces
        Graph.d3Force('charge').strength(-50);
        Graph.d3Force('link').distance(40);

        // Set top-down view (like 2D)
        Graph.cameraPosition({ x: 0, y: 0, z: 1000 });

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') {
                selected = null;
                neighbors.clear();
                selectedTypes.clear();
                typeNeighbors.clear();
                document.querySelectorAll('.legend-btn[data-type]').forEach(el => el.classList.remove('active'));
                closeNodeInfo();
                searchInput.value = '';
                searchResults.classList.remove('visible');
                updateVisibility();
            }
        });

        fetch('/admin/graph/data')
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';

                data.nodes.forEach(n => {
                    allNodes[n.id] = n;
                    // Flatten to 2D (z = 0)
                    n.fz = 0;
                });

                const c = data.nodes.filter(n => n.type === 'concept').length;
                const s = data.nodes.filter(n => n.type === 'semantic').length;
                const e = data.nodes.filter(n => n.type === 'episodic').length;
                document.getElementById('c-count').textContent = c;
                document.getElementById('s-count').textContent = s;
                document.getElementById('e-count').textContent = e;
                document.getElementById('stats').innerHTML =
                    '<div>Nodes: <strong style="color:#5eead4">' + data.nodes.length + '</strong></div>' +
                    '<div>Links: <strong style="color:#5eead4">' + data.links.length + '</strong></div>';

                const numClusters = detectClusters(data.nodes, data.links);
                document.getElementById('cluster-count').textContent = numClusters;

                Graph.graphData(data);
            });
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    """Serve the graph visualization page."""
    return GRAPH_HTML
