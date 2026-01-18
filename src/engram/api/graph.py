"""Graph visualization with WebGL, clustering, and advanced features."""

import logging
from typing import Optional

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
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


def get_db(request: Request) -> Neo4jClient:
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
            "min_x": r["min_x"], "max_x": r["max_x"],
            "min_y": r["min_y"], "max_y": r["max_y"],
            "total": r["total"], "has_layout": True
        }
    return {"has_layout": False, "total": 0}


@router.get("/admin/graph/search")
async def search_nodes(request: Request, q: str = Query(..., min_length=2)) -> dict:
    """Global search across all nodes, returns matches with positions."""
    db = get_db(request)
    query_lower = q.lower()

    # Search concepts
    concepts = await db.execute_query(
        """
        MATCH (n:Concept)
        WHERE toLower(n.name) CONTAINS $q AND n.layout_x IS NOT NULL
        RETURN n.id as id, n.name as name, 'concept' as type,
               n.layout_x as x, n.layout_y as y, n.cluster as cluster
        LIMIT 50
        """,
        q=query_lower
    )

    # Search semantic memories
    semantic = await db.execute_query(
        """
        MATCH (n:SemanticMemory)
        WHERE toLower(n.content) CONTAINS $q AND n.layout_x IS NOT NULL
        RETURN n.id as id, n.content as name, 'semantic' as type,
               n.layout_x as x, n.layout_y as y, n.cluster as cluster
        LIMIT 50
        """,
        q=query_lower
    )

    # Search episodic memories
    episodic = await db.execute_query(
        """
        MATCH (n:EpisodicMemory)
        WHERE toLower(n.query) CONTAINS $q AND n.layout_x IS NOT NULL
        RETURN n.id as id, n.query as name, 'episodic' as type,
               n.layout_x as x, n.layout_y as y, n.cluster as cluster
        LIMIT 50
        """,
        q=query_lower
    )

    results = []
    for r in concepts + semantic + episodic:
        name = r["name"] or ""
        results.append({
            "id": r["id"],
            "name": name[:60] + "..." if len(name) > 60 else name,
            "type": r["type"],
            "x": r["x"],
            "y": r["y"],
            "cluster": r["cluster"] or 0
        })

    return {"results": results[:100], "total": len(results)}


@router.get("/admin/graph/data")
async def get_graph_data(
    request: Request,
    min_x: Optional[float] = Query(None),
    max_x: Optional[float] = Query(None),
    min_y: Optional[float] = Query(None),
    max_y: Optional[float] = Query(None),
) -> dict:
    """Get graph data, optionally filtered by viewport bounds."""
    db = get_db(request)
    nodes = []
    links = []

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
               n.layout_x as x, n.layout_y as y, n.cluster as cluster, conn
        """,
        **params
    )
    for c in concepts:
        nodes.append({
            "id": c["id"], "name": c["name"], "type": "concept",
            "subtype": c["type"] or "unknown",
            "x": c["x"], "y": c["y"],
            "cluster": c["cluster"] or 0, "conn": c["conn"] or 0,
        })

    # Get semantic memories
    memories = await db.execute_query(
        f"""
        MATCH (n:SemanticMemory)
        WHERE n.layout_x IS NOT NULL {viewport_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as conn
        RETURN n.id as id, n.content as content, n.memory_type as type,
               n.layout_x as x, n.layout_y as y, n.cluster as cluster, conn
        """,
        **params
    )
    for m in memories:
        content = m["content"] or ""
        nodes.append({
            "id": m["id"], "name": content[:50] + "..." if len(content) > 50 else content,
            "fullContent": content, "type": "semantic",
            "subtype": m["type"] or "fact",
            "x": m["x"], "y": m["y"],
            "cluster": m["cluster"] or 0, "conn": m["conn"] or 0,
        })

    # Get episodic memories
    episodes = await db.execute_query(
        f"""
        MATCH (n:EpisodicMemory)
        WHERE n.layout_x IS NOT NULL {viewport_filter}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as conn
        RETURN n.id as id, n.query as query, n.behavior_name as behavior,
               n.layout_x as x, n.layout_y as y, n.cluster as cluster, conn
        """,
        **params
    )
    for e in episodes:
        query = e["query"] or ""
        nodes.append({
            "id": e["id"], "name": query[:40] + "..." if len(query) > 40 else query,
            "fullContent": query, "type": "episodic",
            "subtype": e["behavior"] or "unknown",
            "x": e["x"], "y": e["y"],
            "cluster": e["cluster"] or 0, "conn": e["conn"] or 0,
        })

    node_ids = {n["id"] for n in nodes}

    # Get relationships
    for query, rel_type in [
        ("MATCH (a:Concept)-[r:RELATED_TO]->(b:Concept) RETURN a.id as source, b.id as target", "related"),
        ("MATCH (a:SemanticMemory)-[:ABOUT]->(b:Concept) RETURN a.id as source, b.id as target", "about"),
        ("MATCH (a:EpisodicMemory)-[:ACTIVATED]->(b:Concept) RETURN a.id as source, b.id as target", "activated"),
    ]:
        rels = await db.execute_query(query)
        for r in rels:
            if r["source"] in node_ids and r["target"] in node_ids:
                links.append({"source": r["source"], "target": r["target"], "type": rel_type})

    return {"nodes": nodes, "links": links}


@router.get("/admin/graph/stats")
async def get_graph_stats(request: Request) -> dict:
    """Get total node counts."""
    db = get_db(request)
    concepts = await db.execute_query("MATCH (c:Concept) RETURN count(c) as count")
    semantic = await db.execute_query("MATCH (s:SemanticMemory) RETURN count(s) as count")
    episodic = await db.execute_query("MATCH (e:EpisodicMemory) RETURN count(e) as count")
    clusters = await db.execute_query("MATCH (n) WHERE n.cluster IS NOT NULL RETURN count(DISTINCT n.cluster) as count")

    c = concepts[0]["count"] if concepts else 0
    s = semantic[0]["count"] if semantic else 0
    e = episodic[0]["count"] if episodic else 0
    cl = clusters[0]["count"] if clusters else 0

    return {"concepts": c, "semantic": s, "episodic": e, "total": c + s + e, "clusters": cl}


GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Knowledge Graph</title>
    <link rel="icon" href="/favicon.ico">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a12; font-family: -apple-system, sans-serif; overflow: hidden; }
        #canvas { width: 100vw; height: 100vh; display: block; }

        #info { position: absolute; top: 16px; left: 16px; color: #8b949e; z-index: 10; pointer-events: none; }
        #info h1 { font-size: 11px; color: #5eead480; font-weight: 400; text-transform: uppercase; letter-spacing: 3px; margin-top: 6px; }
        #info .brand { font-family: 'Orbitron', sans-serif; font-size: 28px; font-weight: 700; color: #5eead4; text-shadow: 0 0 20px #5eead440; letter-spacing: 3px; }

        #controls {
            position: absolute; bottom: 16px; left: 16px;
            background: rgba(10,10,18,0.95); padding: 12px 16px;
            border-radius: 8px; border: 1px solid #1a1a2e;
            font-size: 11px; color: #8b949e; z-index: 100;
        }
        .legend-item { display: flex; align-items: center; padding: 5px 0; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
        .legend-count { margin-left: auto; padding-left: 16px; color: #5eead4; font-weight: 500; }
        .control-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #1a1a2e; }
        .toggle-btn {
            padding: 6px 12px; background: #0f0f1a; border: 1px solid #1a1a2e;
            border-radius: 4px; color: #8b949e; cursor: pointer; font-size: 11px;
        }
        .toggle-btn:hover { border-color: #5eead4; color: #5eead4; }
        .toggle-btn.active { background: #5eead420; border-color: #5eead4; color: #5eead4; }

        #search-box { position: absolute; top: 16px; left: 50%; transform: translateX(-50%); z-index: 100; }
        #search-input {
            width: 350px; padding: 10px 16px;
            background: rgba(10,10,18,0.95); border: 1px solid #1a1a2e;
            border-radius: 6px; color: #e0e0ff; font-size: 14px;
        }
        #search-input:focus { outline: none; border-color: #5eead4; }
        #search-input::placeholder { color: #4a4a6a; }
        #search-results {
            position: absolute; top: 100%; left: 0; right: 0;
            max-height: 300px; overflow-y: auto;
            background: rgba(10,10,18,0.98); border: 1px solid #1a1a2e;
            border-top: none; border-radius: 0 0 6px 6px; display: none;
        }
        #search-results.visible { display: block; }
        .search-result { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #1a1a2e; display: flex; align-items: center; }
        .search-result:hover { background: #1a1a2e; }
        .search-result-name { flex: 1; color: #e0e0ff; font-size: 12px; }
        .search-result-type { font-size: 10px; padding: 2px 6px; border-radius: 4px; text-transform: uppercase; }

        #stats { position: absolute; top: 16px; right: 16px; background: rgba(10,10,18,0.95); padding: 12px 16px; border-radius: 6px; border: 1px solid #1a1a2e; font-size: 11px; color: #8b8ba0; z-index: 10; }
        #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #5eead4; z-index: 10; }

        #tooltip {
            position: absolute; pointer-events: none; z-index: 1000;
            background: rgba(10,10,18,0.95); padding: 8px 12px;
            border-radius: 6px; border: 1px solid #5eead4;
            font-size: 12px; color: #e0e0ff; max-width: 300px;
            display: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        #tooltip.visible { display: block; }

        #node-info {
            position: absolute; top: 100px; right: 16px; width: 280px; max-height: 400px; overflow-y: auto;
            background: rgba(10,10,18,0.95); padding: 16px; border-radius: 8px; border: 1px solid #1a1a2e;
            font-size: 12px; color: #e0e0ff; display: none; z-index: 10;
        }
        #node-info.visible { display: block; }
        #node-info h3 { margin: 0 0 12px 0; font-size: 14px; color: #fff; word-break: break-word; }
        #node-info .type-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 500; text-transform: uppercase; margin-bottom: 12px; }
        #node-info .info-row { margin: 8px 0; padding: 8px 0; border-top: 1px solid #1a1a2e; }
        #node-info .info-label { color: #6b6b8a; font-size: 10px; text-transform: uppercase; margin-bottom: 4px; }
        #node-info .info-value { color: #e0e0ff; word-break: break-word; }
        #node-info .close-btn { position: absolute; top: 8px; right: 8px; background: none; border: none; color: #6b6b8a; cursor: pointer; font-size: 16px; }
        #node-info .close-btn:hover { color: #5eead4; }

        #mode-indicator { position: absolute; bottom: 16px; right: 16px; background: rgba(94,234,212,0.1); border: 1px solid #5eead4; padding: 8px 12px; border-radius: 6px; font-size: 10px; color: #5eead4; z-index: 10; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <div id="info"><div class="brand">Engram</div><h1>Memory Graph</h1></div>

    <div id="search-box">
        <input type="text" id="search-input" placeholder="Search all nodes... (Enter to search)">
        <div id="search-results"></div>
    </div>

    <div id="tooltip"></div>

    <div id="node-info">
        <button class="close-btn" onclick="closeNodeInfo()">&times;</button>
        <div id="node-info-content"></div>
    </div>

    <div id="controls">
        <div class="legend-item"><span class="legend-dot" style="background:#5eead4"></span>Concept<span class="legend-count" id="c-count">0</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#a78bfa"></span>Semantic<span class="legend-count" id="s-count">0</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#f472b6"></span>Episodic<span class="legend-count" id="e-count">0</span></div>
        <div class="control-row">
            <button class="toggle-btn" id="cluster-btn" onclick="toggleClusters()">Clusters</button>
            <button class="toggle-btn" id="bundle-btn" onclick="toggleBundling()">Bundle</button>
        </div>
    </div>

    <div id="stats">Loading...</div>
    <div id="loading">Loading graph...</div>
    <div id="mode-indicator">WebGL</div>

    <script>
        const canvas = document.getElementById('canvas');
        const gl = canvas.getContext('webgl', { antialias: true, alpha: false }) || canvas.getContext('experimental-webgl');
        const ctx2d = document.createElement('canvas').getContext('2d'); // For text rendering

        if (!gl) {
            document.getElementById('loading').innerHTML = 'WebGL not supported';
            throw new Error('WebGL not supported');
        }

        const typeColors = { concept: [0.37, 0.92, 0.83], semantic: [0.65, 0.55, 0.98], episodic: [0.96, 0.45, 0.71] };
        const typeColorsHex = { concept: '#5eead4', semantic: '#a78bfa', episodic: '#f472b6' };

        // Cluster color palette (12 distinct colors)
        const clusterPalette = [
            [0.37, 0.92, 0.83], [0.65, 0.55, 0.98], [0.96, 0.45, 0.71], [0.98, 0.75, 0.14],
            [0.20, 0.83, 0.60], [0.38, 0.65, 0.98], [0.98, 0.44, 0.52], [0.64, 0.90, 0.21],
            [0.13, 0.83, 0.93], [0.75, 0.52, 0.99], [0.98, 0.45, 0.09], [0.18, 0.83, 0.75]
        ];

        let nodes = [], links = [], nodeMap = {};
        let bounds = { min_x: -1000, max_x: 1000, min_y: -1000, max_y: 1000 };
        let viewX = 0, viewY = 0, scale = 1;
        let selectedNode = null, hoveredNode = null;
        let highlightedNodes = new Set();
        let showClusters = false, showBundling = false;

        let isDragging = false, lastMouseX = 0, lastMouseY = 0, dragStartX = 0, dragStartY = 0;
        let loadingViewport = false, lastViewport = null, viewportDebounceTimer = null;

        // WebGL setup
        const vertexShaderSrc = `
            attribute vec2 a_position;
            attribute vec3 a_color;
            attribute float a_size;
            uniform vec2 u_resolution;
            uniform vec2 u_view;
            uniform float u_scale;
            varying vec3 v_color;
            void main() {
                vec2 pos = (a_position - u_view) * u_scale;
                vec2 clipSpace = (pos / u_resolution) * 2.0;
                gl_Position = vec4(clipSpace, 0, 1);
                gl_PointSize = a_size * u_scale;
                v_color = a_color;
            }
        `;

        const fragmentShaderSrc = `
            precision mediump float;
            varying vec3 v_color;
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (length(coord) > 0.5) discard;
                float alpha = 1.0 - smoothstep(0.4, 0.5, length(coord));
                gl_FragColor = vec4(v_color, alpha);
            }
        `;

        const lineVertexShaderSrc = `
            attribute vec2 a_position;
            uniform vec2 u_resolution;
            uniform vec2 u_view;
            uniform float u_scale;
            void main() {
                vec2 pos = (a_position - u_view) * u_scale;
                vec2 clipSpace = (pos / u_resolution) * 2.0;
                gl_Position = vec4(clipSpace, 0, 1);
            }
        `;

        const lineFragmentShaderSrc = `
            precision mediump float;
            uniform vec4 u_color;
            void main() {
                gl_FragColor = u_color;
            }
        `;

        function createShader(type, source) {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error(gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }
            return shader;
        }

        function createProgram(vertexSrc, fragmentSrc) {
            const vertexShader = createShader(gl.VERTEX_SHADER, vertexSrc);
            const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentSrc);
            const program = gl.createProgram();
            gl.attachShader(program, vertexShader);
            gl.attachShader(program, fragmentShader);
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                console.error(gl.getProgramInfoLog(program));
                return null;
            }
            return program;
        }

        const nodeProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);
        const lineProgram = createProgram(lineVertexShaderSrc, lineFragmentShaderSrc);

        const nodeBuffer = gl.createBuffer();
        const lineBuffer = gl.createBuffer();

        function resize() {
            canvas.width = window.innerWidth * window.devicePixelRatio;
            canvas.height = window.innerHeight * window.devicePixelRatio;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
            gl.viewport(0, 0, canvas.width, canvas.height);
            render();
        }

        function screenToWorld(sx, sy) {
            return {
                x: (sx - window.innerWidth / 2) / scale + viewX,
                y: -(sy - window.innerHeight / 2) / scale + viewY
            };
        }

        function worldToScreen(wx, wy) {
            return {
                x: (wx - viewX) * scale + window.innerWidth / 2,
                y: -(wy - viewY) * scale + window.innerHeight / 2
            };
        }

        function getViewportBounds() {
            const margin = 200 / scale;
            const tl = screenToWorld(0, 0);
            const br = screenToWorld(window.innerWidth, window.innerHeight);
            return {
                min_x: Math.min(tl.x, br.x) - margin,
                max_x: Math.max(tl.x, br.x) + margin,
                min_y: Math.min(tl.y, br.y) - margin,
                max_y: Math.max(tl.y, br.y) + margin
            };
        }

        function viewportChanged(v1, v2) {
            if (!v1 || !v2) return true;
            const threshold = 100 / scale;
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

                render();
            } catch (e) {
                console.error('Failed to load viewport data:', e);
            } finally {
                loadingViewport = false;
            }
        }

        function scheduleViewportLoad() {
            if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
            viewportDebounceTimer = setTimeout(loadViewportData, 100);
        }

        function getNodeColor(node) {
            if (highlightedNodes.size > 0 && !highlightedNodes.has(node.id)) {
                return [0.15, 0.15, 0.2];
            }
            if (showClusters) {
                return clusterPalette[node.cluster % clusterPalette.length];
            }
            return typeColors[node.type] || typeColors.concept;
        }

        function render() {
            gl.clearColor(0.039, 0.039, 0.071, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;

            // Draw lines
            if (links.length > 0) {
                gl.useProgram(lineProgram);
                gl.uniform2f(gl.getUniformLocation(lineProgram, 'u_resolution'), w / 2, h / 2);
                gl.uniform2f(gl.getUniformLocation(lineProgram, 'u_view'), viewX, viewY);
                gl.uniform1f(gl.getUniformLocation(lineProgram, 'u_scale'), scale);

                const lineOpacity = highlightedNodes.size > 0 ? 0.05 : 0.12;
                gl.uniform4f(gl.getUniformLocation(lineProgram, 'u_color'), 0.37, 0.92, 0.83, lineOpacity);

                const lineData = [];
                for (const link of links) {
                    const s = nodeMap[link.source];
                    const t = nodeMap[link.target];
                    if (s && t) {
                        if (showBundling) {
                            // Curved edges via control point
                            const mx = (s.x + t.x) / 2;
                            const my = (s.y + t.y) / 2;
                            const dx = t.x - s.x;
                            const dy = t.y - s.y;
                            const cx = mx - dy * 0.15;
                            const cy = my + dx * 0.15;
                            // Approximate curve with segments
                            for (let i = 0; i < 8; i++) {
                                const t1 = i / 8, t2 = (i + 1) / 8;
                                const x1 = (1-t1)*(1-t1)*s.x + 2*(1-t1)*t1*cx + t1*t1*t.x;
                                const y1 = (1-t1)*(1-t1)*s.y + 2*(1-t1)*t1*cy + t1*t1*t.y;
                                const x2 = (1-t2)*(1-t2)*s.x + 2*(1-t2)*t2*cx + t2*t2*t.x;
                                const y2 = (1-t2)*(1-t2)*s.y + 2*(1-t2)*t2*cy + t2*t2*t.y;
                                lineData.push(x1, y1, x2, y2);
                            }
                        } else {
                            lineData.push(s.x, s.y, t.x, t.y);
                        }
                    }
                }

                gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineData), gl.DYNAMIC_DRAW);

                const posLoc = gl.getAttribLocation(lineProgram, 'a_position');
                gl.enableVertexAttribArray(posLoc);
                gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.LINES, 0, lineData.length / 2);
            }

            // Draw nodes
            if (nodes.length > 0) {
                gl.useProgram(nodeProgram);
                gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_resolution'), w / 2, h / 2);
                gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_view'), viewX, viewY);
                gl.uniform1f(gl.getUniformLocation(nodeProgram, 'u_scale'), scale);

                const nodeData = [];
                for (const node of nodes) {
                    const color = getNodeColor(node);
                    const baseSize = Math.max(4, Math.min(16, Math.sqrt(node.conn || 1) * 2.5));
                    const size = baseSize * Math.max(0.8, Math.min(2.5, scale * 0.7));

                    // Highlight selected/hovered
                    let finalColor = color;
                    if (node === selectedNode || node === hoveredNode) {
                        finalColor = [1, 1, 1];
                    }

                    nodeData.push(node.x, node.y, finalColor[0], finalColor[1], finalColor[2], size);
                }

                gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nodeData), gl.DYNAMIC_DRAW);

                const stride = 6 * 4;
                const posLoc = gl.getAttribLocation(nodeProgram, 'a_position');
                const colorLoc = gl.getAttribLocation(nodeProgram, 'a_color');
                const sizeLoc = gl.getAttribLocation(nodeProgram, 'a_size');

                gl.enableVertexAttribArray(posLoc);
                gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
                gl.enableVertexAttribArray(colorLoc);
                gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, stride, 8);
                gl.enableVertexAttribArray(sizeLoc);
                gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, stride, 20);

                gl.drawArrays(gl.POINTS, 0, nodes.length);
            }
        }

        function findNodeAt(sx, sy) {
            const world = screenToWorld(sx, sy);
            let closest = null, closestDist = Infinity;

            for (const node of nodes) {
                const dx = node.x - world.x;
                const dy = node.y - world.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const baseSize = Math.max(4, Math.min(16, Math.sqrt(node.conn || 1) * 2.5));
                const threshold = (baseSize * 2) / scale + 10;

                if (dist < threshold && dist < closestDist) {
                    closest = node;
                    closestDist = dist;
                }
            }
            return closest;
        }

        // Tooltip
        const tooltip = document.getElementById('tooltip');
        function showTooltip(node, x, y) {
            tooltip.textContent = node.name;
            tooltip.style.left = (x + 15) + 'px';
            tooltip.style.top = (y + 15) + 'px';
            tooltip.classList.add('visible');
        }
        function hideTooltip() {
            tooltip.classList.remove('visible');
        }

        // Node info panel
        function showNodeInfo(node) {
            const panel = document.getElementById('node-info');
            const content = document.getElementById('node-info-content');
            const color = typeColorsHex[node.type];

            let html = `
                <span class="type-badge" style="background:${color}20;color:${color}">${node.type}</span>
                <h3>${node.name}</h3>
                <div class="info-row">
                    <div class="info-label">Connections</div>
                    <div class="info-value">${node.conn || 0}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">Cluster</div>
                    <div class="info-value">${node.cluster}</div>
                </div>
            `;
            if (node.fullContent && node.fullContent !== node.name) {
                html += `<div class="info-row"><div class="info-label">Content</div><div class="info-value" style="max-height:150px;overflow-y:auto;">${node.fullContent}</div></div>`;
            }
            content.innerHTML = html;
            panel.classList.add('visible');
        }

        function closeNodeInfo() {
            document.getElementById('node-info').classList.remove('visible');
            selectedNode = null;
            render();
        }

        // Toggle functions
        function toggleClusters() {
            showClusters = !showClusters;
            document.getElementById('cluster-btn').classList.toggle('active', showClusters);
            render();
        }

        function toggleBundling() {
            showBundling = !showBundling;
            document.getElementById('bundle-btn').classList.toggle('active', showBundling);
            render();
        }

        // Search
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        let searchTimeout = null;

        searchInput.addEventListener('input', () => {
            if (searchTimeout) clearTimeout(searchTimeout);
            const q = searchInput.value.trim();
            if (q.length < 2) {
                searchResults.classList.remove('visible');
                highlightedNodes.clear();
                render();
                return;
            }
            searchTimeout = setTimeout(() => doSearch(q), 300);
        });

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const q = searchInput.value.trim();
                if (q.length >= 2) doSearch(q);
            }
            if (e.key === 'Escape') {
                searchInput.value = '';
                searchResults.classList.remove('visible');
                highlightedNodes.clear();
                render();
            }
        });

        async function doSearch(q) {
            try {
                const res = await fetch(`/admin/graph/search?q=${encodeURIComponent(q)}`);
                const data = await res.json();

                highlightedNodes.clear();
                data.results.forEach(r => highlightedNodes.add(r.id));

                if (data.results.length === 0) {
                    searchResults.innerHTML = '<div style="padding:12px;color:#6b6b8a;">No results</div>';
                } else {
                    searchResults.innerHTML = data.results.slice(0, 20).map(r => `
                        <div class="search-result" data-x="${r.x}" data-y="${r.y}" data-id="${r.id}">
                            <span class="search-result-name">${r.name}</span>
                            <span class="search-result-type" style="background:${typeColorsHex[r.type]}20;color:${typeColorsHex[r.type]}">${r.type}</span>
                        </div>
                    `).join('');
                }
                searchResults.classList.add('visible');
                render();
            } catch (e) {
                console.error('Search failed:', e);
            }
        }

        searchResults.addEventListener('click', (e) => {
            const result = e.target.closest('.search-result');
            if (result) {
                const x = parseFloat(result.dataset.x);
                const y = parseFloat(result.dataset.y);
                viewX = x;
                viewY = y;
                scale = 2;
                searchResults.classList.remove('visible');
                render();
                scheduleViewportLoad();
            }
        });

        // Mouse events
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragStartX = lastMouseX = e.clientX;
            dragStartY = lastMouseY = e.clientY;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const dx = e.clientX - lastMouseX;
                const dy = e.clientY - lastMouseY;
                viewX -= dx / scale;
                viewY += dy / scale;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
                scheduleViewportLoad();
            } else {
                // Hover tooltip
                const node = findNodeAt(e.clientX, e.clientY);
                if (node !== hoveredNode) {
                    hoveredNode = node;
                    if (node) {
                        showTooltip(node, e.clientX, e.clientY);
                    } else {
                        hideTooltip();
                    }
                    render();
                }
            }
        });

        canvas.addEventListener('mouseup', (e) => {
            const dx = Math.abs(e.clientX - dragStartX);
            const dy = Math.abs(e.clientY - dragStartY);

            if (dx < 5 && dy < 5) {
                const node = findNodeAt(e.clientX, e.clientY);
                if (node) {
                    selectedNode = node;
                    showNodeInfo(node);
                } else {
                    closeNodeInfo();
                    highlightedNodes.clear();
                }
                render();
            }
            isDragging = false;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const mouseWorld = screenToWorld(e.clientX, e.clientY);
            scale *= zoomFactor;
            scale = Math.max(0.05, Math.min(10, scale));
            const newMouseWorld = screenToWorld(e.clientX, e.clientY);
            viewX += mouseWorld.x - newMouseWorld.x;
            viewY += mouseWorld.y - newMouseWorld.y;
            render();
            scheduleViewportLoad();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeNodeInfo();
                highlightedNodes.clear();
                searchInput.value = '';
                searchResults.classList.remove('visible');
                render();
            }
        });

        // Initialize
        async function init() {
            resize();
            window.addEventListener('resize', resize);

            const boundsRes = await fetch('/admin/graph/bounds');
            const boundsData = await boundsRes.json();

            if (!boundsData.has_layout) {
                document.getElementById('loading').innerHTML = 'No layout. Run: uv run python scripts/compute_layout.py';
                return;
            }

            bounds = boundsData;
            viewX = (bounds.min_x + bounds.max_x) / 2;
            viewY = (bounds.min_y + bounds.max_y) / 2;

            const graphWidth = bounds.max_x - bounds.min_x;
            const graphHeight = bounds.max_y - bounds.min_y;
            scale = Math.min(window.innerWidth / graphWidth * 0.8, window.innerHeight / graphHeight * 0.8);
            scale = Math.max(0.05, Math.min(1, scale));

            const statsRes = await fetch('/admin/graph/stats');
            const stats = await statsRes.json();
            document.getElementById('c-count').textContent = stats.concepts;
            document.getElementById('s-count').textContent = stats.semantic;
            document.getElementById('e-count').textContent = stats.episodic;
            document.getElementById('stats').innerHTML = `<div>Total: <strong style="color:#5eead4">${stats.total}</strong></div><div>Clusters: <strong style="color:#a78bfa">${stats.clusters}</strong></div>`;

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
    return GRAPH_HTML
