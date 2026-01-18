"""Graph visualization with pre-computed layout and viewport culling."""

import logging

from fastapi import APIRouter, Request, Query
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


@router.get("/admin/graph/bounds")
async def get_graph_bounds(request: Request) -> dict:
    """Get bounding box of all nodes with layout positions."""
    db = get_db(request)

    result = await db.execute_query(
        """
        MATCH (n)
        WHERE n.layout_x IS NOT NULL AND n.layout_y IS NOT NULL
        RETURN min(n.layout_x) as min_x, max(n.layout_x) as max_x,
               min(n.layout_y) as min_y, max(n.layout_y) as max_y,
               count(n) as total
        """
    )

    if result and result[0]["total"] > 0:
        r = result[0]
        return {
            "min_x": r["min_x"],
            "max_x": r["max_x"],
            "min_y": r["min_y"],
            "max_y": r["max_y"],
            "total": r["total"],
            "has_layout": True
        }
    else:
        return {"has_layout": False, "total": 0}


@router.get("/admin/graph/data")
async def get_graph_data(
    request: Request,
    min_x: float = Query(None),
    max_x: float = Query(None),
    min_y: float = Query(None),
    max_y: float = Query(None),
) -> dict:
    """Get graph data, optionally filtered by viewport bounds."""
    db = get_db(request)

    nodes = []
    links = []

    # Build viewport filter if bounds provided
    viewport_filter = ""
    params = {}
    if all(v is not None for v in [min_x, max_x, min_y, max_y]):
        viewport_filter = """
            AND n.layout_x >= $min_x AND n.layout_x <= $max_x
            AND n.layout_y >= $min_y AND n.layout_y <= $max_y
        """
        params = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

    # Get concepts
    concepts = await db.execute_query(
        f"""
        MATCH (n:Concept)
        WHERE n.layout_x IS NOT NULL {viewport_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as conn
        RETURN n.id as id, n.name as name, n.type as type,
               n.layout_x as x, n.layout_y as y, conn
        """,
        **params
    )
    for c in concepts:
        nodes.append({
            "id": c["id"],
            "name": c["name"],
            "type": "concept",
            "subtype": c["type"] or "unknown",
            "x": c["x"],
            "y": c["y"],
            "conn": c["conn"] or 0,
        })

    # Get semantic memories
    memories = await db.execute_query(
        f"""
        MATCH (n:SemanticMemory)
        WHERE n.layout_x IS NOT NULL {viewport_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as conn
        RETURN n.id as id, n.content as content, n.memory_type as type,
               n.layout_x as x, n.layout_y as y, conn
        """,
        **params
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
            "x": m["x"],
            "y": m["y"],
            "conn": m["conn"] or 0,
        })

    # Get episodic memories
    episodes = await db.execute_query(
        f"""
        MATCH (n:EpisodicMemory)
        WHERE n.layout_x IS NOT NULL {viewport_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as conn
        RETURN n.id as id, n.query as query, n.behavior_name as behavior,
               n.layout_x as x, n.layout_y as y, conn
        """,
        **params
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
            "x": e["x"],
            "y": e["y"],
            "conn": e["conn"] or 0,
        })

    # Get node IDs for filtering links
    node_ids = {n["id"] for n in nodes}

    # Get relationships (only between visible nodes)
    concept_rels = await db.execute_query(
        """
        MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
        RETURN c1.id as source, c2.id as target, r.type as relType
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


@router.get("/admin/graph/stats")
async def get_graph_stats(request: Request) -> dict:
    """Get total node counts for display."""
    db = get_db(request)

    # Use separate counts to avoid issues with empty labels
    concepts_result = await db.execute_query("MATCH (c:Concept) RETURN count(c) as count")
    semantic_result = await db.execute_query("MATCH (s:SemanticMemory) RETURN count(s) as count")
    episodic_result = await db.execute_query("MATCH (e:EpisodicMemory) RETURN count(e) as count")

    concepts = concepts_result[0]["count"] if concepts_result else 0
    semantic = semantic_result[0]["count"] if semantic_result else 0
    episodic = episodic_result[0]["count"] if episodic_result else 0

    return {
        "concepts": concepts,
        "semantic": semantic,
        "episodic": episodic,
        "total": concepts + semantic + episodic
    }


GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Knowledge Graph</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='40' fill='%230a0a12' stroke='%235eead4' stroke-width='6'/><circle cx='50' cy='50' r='15' fill='%235eead4'/></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a12; font-family: -apple-system, sans-serif; overflow: hidden; }
        #canvas { width: 100vw; height: 100vh; display: block; }
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
        .legend-item {
            display: flex;
            align-items: center;
            padding: 6px 0;
        }
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
        #viewport-info {
            position: absolute;
            bottom: 50px;
            right: 16px;
            background: rgba(10,10,18,0.9);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 10px;
            color: #6b6b8a;
            z-index: 10;
        }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <div id="info"><div class="brand">Engram</div><h1>Memory Graph</h1></div>
    <div id="search-box">
        <input type="text" id="search-input" placeholder="Search nodes...">
    </div>
    <div id="node-info">
        <button class="close-btn" onclick="closeNodeInfo()">&times;</button>
        <div id="node-info-content"></div>
    </div>
    <div id="legend">
        <div class="legend-item"><span class="legend-dot" style="background:#5eead4"></span>Concept<span class="legend-count" id="c-count">0</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#a78bfa"></span>Semantic<span class="legend-count" id="s-count">0</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#f472b6"></span>Episodic<span class="legend-count" id="e-count">0</span></div>
    </div>
    <div id="stats">Loading...</div>
    <div id="loading">Loading graph...</div>
    <div id="viewport-info">Visible: <span id="visible-count">0</span></div>
    <div id="mode-indicator">Pre-computed Layout</div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        const typeColors = { concept: '#5eead4', semantic: '#a78bfa', episodic: '#f472b6' };
        const typeLabels = { concept: 'Concept', semantic: 'Semantic Memory', episodic: 'Episodic Memory' };

        let nodes = [];
        let links = [];
        let nodeMap = {};
        let bounds = { min_x: -1000, max_x: 1000, min_y: -1000, max_y: 1000 };

        // View state
        let viewX = 0, viewY = 0;
        let scale = 1;
        let selectedNode = null;

        // Interaction state
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;

        // Viewport culling state
        let loadingViewport = false;
        let lastViewport = null;
        let viewportDebounceTimer = null;

        function resize() {
            canvas.width = window.innerWidth * window.devicePixelRatio;
            canvas.height = window.innerHeight * window.devicePixelRatio;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            render();
        }

        function screenToWorld(sx, sy) {
            return {
                x: (sx - window.innerWidth / 2) / scale + viewX,
                y: (sy - window.innerHeight / 2) / scale + viewY
            };
        }

        function worldToScreen(wx, wy) {
            return {
                x: (wx - viewX) * scale + window.innerWidth / 2,
                y: (wy - viewY) * scale + window.innerHeight / 2
            };
        }

        function getViewportBounds() {
            const margin = 100 / scale; // Extra margin for smoother loading
            const tl = screenToWorld(0, 0);
            const br = screenToWorld(window.innerWidth, window.innerHeight);
            return {
                min_x: tl.x - margin,
                max_x: br.x + margin,
                min_y: tl.y - margin,
                max_y: br.y + margin
            };
        }

        function viewportChanged(v1, v2) {
            if (!v1 || !v2) return true;
            const threshold = 50 / scale;
            return Math.abs(v1.min_x - v2.min_x) > threshold ||
                   Math.abs(v1.max_x - v2.max_x) > threshold ||
                   Math.abs(v1.min_y - v2.min_y) > threshold ||
                   Math.abs(v1.max_y - v2.max_y) > threshold;
        }

        async function loadViewportData() {
            if (loadingViewport) return;

            const vp = getViewportBounds();
            if (!viewportChanged(vp, lastViewport)) return;

            loadingViewport = true;
            lastViewport = vp;

            try {
                const url = `/admin/graph/data?min_x=${vp.min_x}&max_x=${vp.max_x}&min_y=${vp.min_y}&max_y=${vp.max_y}`;
                const response = await fetch(url);
                const data = await response.json();

                nodes = data.nodes;
                links = data.links;
                nodeMap = {};
                nodes.forEach(n => nodeMap[n.id] = n);

                document.getElementById('visible-count').textContent = nodes.length;
                render();
            } catch (e) {
                console.error('Failed to load viewport data:', e);
            } finally {
                loadingViewport = false;
            }
        }

        function scheduleViewportLoad() {
            if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
            viewportDebounceTimer = setTimeout(loadViewportData, 150);
        }

        function render() {
            ctx.save();
            ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);

            // Clear
            ctx.fillStyle = '#0a0a12';
            ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

            // Draw links
            ctx.strokeStyle = 'rgba(94,234,212,0.12)';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            for (const link of links) {
                const source = nodeMap[link.source] || nodeMap[link.source?.id];
                const target = nodeMap[link.target] || nodeMap[link.target?.id];
                if (source && target) {
                    const s = worldToScreen(source.x, source.y);
                    const t = worldToScreen(target.x, target.y);
                    ctx.moveTo(s.x, s.y);
                    ctx.lineTo(t.x, t.y);
                }
            }
            ctx.stroke();

            // Draw nodes - scale with zoom for better visibility
            for (const node of nodes) {
                const pos = worldToScreen(node.x, node.y);
                // Base size from connections, then scale with zoom
                const baseSize = Math.max(3, Math.min(12, Math.sqrt(node.conn || 1) * 2));
                const radius = baseSize * Math.max(0.5, Math.min(3, scale));

                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

                if (selectedNode === node) {
                    ctx.fillStyle = '#ffffff';
                    ctx.shadowColor = typeColors[node.type];
                    ctx.shadowBlur = 20;
                } else {
                    ctx.fillStyle = typeColors[node.type] || '#5eead4';
                    ctx.shadowBlur = 0;
                }
                ctx.fill();
                ctx.shadowBlur = 0;

                // Draw label at higher zoom
                if (scale > 1.2 || (scale > 0.6 && node.conn > 5)) {
                    const fontSize = Math.max(10, Math.min(16, 12 * scale));
                    ctx.font = `${fontSize}px -apple-system, sans-serif`;
                    ctx.fillStyle = 'rgba(255,255,255,0.8)';
                    ctx.textAlign = 'center';
                    const label = node.name.length > 25 ? node.name.slice(0, 25) + '...' : node.name;
                    ctx.fillText(label, pos.x, pos.y + radius + fontSize + 2);
                }
            }

            ctx.restore();
        }

        function findNodeAt(sx, sy) {
            const world = screenToWorld(sx, sy);

            for (const node of nodes) {
                const dx = node.x - world.x;
                const dy = node.y - world.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                // Use same radius calculation as rendering
                const baseSize = Math.max(3, Math.min(12, Math.sqrt(node.conn || 1) * 2));
                const radius = baseSize * Math.max(0.5, Math.min(3, scale));
                // Convert screen radius to world radius, add padding for easier clicking
                const worldRadius = (radius / scale) + 5;
                if (dist < worldRadius) {
                    return node;
                }
            }
            return null;
        }

        function showNodeInfo(node) {
            const panel = document.getElementById('node-info');
            const content = document.getElementById('node-info-content');
            const color = typeColors[node.type];

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

            content.innerHTML = html;
            panel.classList.add('visible');
        }

        function closeNodeInfo() {
            document.getElementById('node-info').classList.remove('visible');
            selectedNode = null;
            render();
        }

        // Mouse events
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const dx = e.clientX - lastMouseX;
                const dy = e.clientY - lastMouseY;
                viewX -= dx / scale;
                viewY -= dy / scale;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
                scheduleViewportLoad();
            }
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!isDragging) return;

            const dx = Math.abs(e.clientX - lastMouseX);
            const dy = Math.abs(e.clientY - lastMouseY);

            // If minimal movement, treat as click
            if (dx < 5 && dy < 5) {
                const node = findNodeAt(e.clientX, e.clientY);
                if (node) {
                    selectedNode = node;
                    showNodeInfo(node);
                } else {
                    closeNodeInfo();
                }
                render();
            }

            isDragging = false;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;

            // Zoom toward mouse position
            const mouseWorld = screenToWorld(e.clientX, e.clientY);
            scale *= zoomFactor;
            scale = Math.max(0.1, Math.min(10, scale));

            const newMouseWorld = screenToWorld(e.clientX, e.clientY);
            viewX += mouseWorld.x - newMouseWorld.x;
            viewY += mouseWorld.y - newMouseWorld.y;

            render();
            scheduleViewportLoad();
        });

        // Search
        const searchInput = document.getElementById('search-input');
        searchInput.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                const query = searchInput.value.trim().toLowerCase();
                if (query.length < 2) return;

                // Search in current nodes first
                let found = nodes.find(n =>
                    n.name.toLowerCase().includes(query) ||
                    (n.fullContent && n.fullContent.toLowerCase().includes(query))
                );

                if (found) {
                    viewX = found.x;
                    viewY = found.y;
                    scale = 2;
                    selectedNode = found;
                    showNodeInfo(found);
                    render();
                    scheduleViewportLoad();
                }
            }
        });

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeNodeInfo();
                searchInput.value = '';
            }
        });

        // Initialize
        async function init() {
            resize();
            window.addEventListener('resize', resize);

            // Load bounds
            const boundsRes = await fetch('/admin/graph/bounds');
            const boundsData = await boundsRes.json();

            if (!boundsData.has_layout) {
                document.getElementById('loading').innerHTML =
                    'No layout computed. Run:<br><code>uv run python scripts/compute_layout.py</code>';
                return;
            }

            bounds = boundsData;

            // Center view on graph
            viewX = (bounds.min_x + bounds.max_x) / 2;
            viewY = (bounds.min_y + bounds.max_y) / 2;

            // Set initial scale to fit graph
            const graphWidth = bounds.max_x - bounds.min_x;
            const graphHeight = bounds.max_y - bounds.min_y;
            scale = Math.min(
                window.innerWidth / graphWidth * 0.8,
                window.innerHeight / graphHeight * 0.8
            );
            scale = Math.max(0.1, Math.min(2, scale));

            // Load stats
            const statsRes = await fetch('/admin/graph/stats');
            const stats = await statsRes.json();
            document.getElementById('c-count').textContent = stats.concepts;
            document.getElementById('s-count').textContent = stats.semantic;
            document.getElementById('e-count').textContent = stats.episodic;
            document.getElementById('stats').innerHTML =
                '<div>Total: <strong style="color:#5eead4">' + stats.total + '</strong></div>';

            // Load initial viewport data
            document.getElementById('loading').style.display = 'none';
            await loadViewportData();
        }

        init();
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    """Serve the graph visualization page."""
    return GRAPH_HTML
