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
    db = get_db(request)
    query_lower = q.lower()

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


@router.get("/admin/graph/neighbors")
async def get_neighbors(request: Request, node_id: str = Query(...)) -> dict:
    """Get neighbors of a node."""
    db = get_db(request)

    neighbors = await db.execute_query(
        """
        MATCH (n)-[r]-(neighbor)
        WHERE n.id = $node_id AND neighbor.layout_x IS NOT NULL
        RETURN neighbor.id as id,
               COALESCE(neighbor.name, neighbor.content, neighbor.query) as name,
               CASE
                   WHEN 'Concept' IN labels(neighbor) THEN 'concept'
                   WHEN 'SemanticMemory' IN labels(neighbor) THEN 'semantic'
                   ELSE 'episodic'
               END as type,
               neighbor.layout_x as x, neighbor.layout_y as y
        LIMIT 50
        """,
        node_id=node_id
    )

    results = []
    for r in neighbors:
        name = r["name"] or ""
        results.append({
            "id": r["id"],
            "name": name[:40] + "..." if len(name) > 40 else name,
            "type": r["type"],
            "x": r["x"],
            "y": r["y"]
        })

    return {"neighbors": results}


@router.get("/admin/graph/data")
async def get_graph_data(
    request: Request,
    min_x: Optional[float] = Query(None),
    max_x: Optional[float] = Query(None),
    min_y: Optional[float] = Query(None),
    max_y: Optional[float] = Query(None),
    sample: int = Query(1, description="Load every Nth node (1=all, 10=every 10th)"),
) -> dict:
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

    # Sampling filter: use random sampling for performance
    # rand() < 1/sample gives ~1/sample of nodes
    sample_filter = f"AND rand() < {1.0 / sample}" if sample > 1 else ""

    concepts = await db.execute_query(
        f"""
        MATCH (n:Concept)
        WHERE n.layout_x IS NOT NULL {viewport_filter} {sample_filter}
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

    memories = await db.execute_query(
        f"""
        MATCH (n:SemanticMemory)
        WHERE n.layout_x IS NOT NULL {viewport_filter} {sample_filter}
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

    episodes = await db.execute_query(
        f"""
        MATCH (n:EpisodicMemory)
        WHERE n.layout_x IS NOT NULL {viewport_filter} {sample_filter}
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


@router.get("/admin/graph/cluster-meta")
async def get_cluster_metadata(request: Request) -> dict:
    """Get cluster centers and inter-cluster edges for cluster-level rendering."""
    db = get_db(request)

    # Get cluster centers
    centers = await db.execute_query(
        """
        MATCH (c:ClusterMeta)
        RETURN c.cluster_id as id, c.center_x as x, c.center_y as y, c.node_count as nodeCount
        """
    )

    # Get inter-cluster edges
    edges = await db.execute_query(
        """
        MATCH (c1:ClusterMeta)-[r:CLUSTER_EDGE]->(c2:ClusterMeta)
        RETURN c1.cluster_id as source, c2.cluster_id as target, r.count as count
        """
    )

    return {
        "centers": [{"id": c["id"], "x": c["x"], "y": c["y"], "nodeCount": c["nodeCount"]} for c in centers],
        "edges": [{"source": e["source"], "target": e["target"], "count": e["count"]} for e in edges],
    }


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
        #label-canvas { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 5; }

        #info { position: absolute; top: 16px; left: 16px; color: #8b949e; z-index: 10; pointer-events: none; }
        #info h1 { font-size: 11px; color: #5eead480; font-weight: 400; text-transform: uppercase; letter-spacing: 3px; margin-top: 6px; }
        #info .brand { font-family: 'Orbitron', sans-serif; font-size: 28px; font-weight: 700; color: #5eead4; text-shadow: 0 0 20px #5eead440; letter-spacing: 3px; }

        #controls {
            position: absolute; bottom: 16px; left: 16px;
            background: rgba(10,10,18,0.95); padding: 12px 16px;
            border-radius: 8px; border: 1px solid #1a1a2e;
            font-size: 11px; color: #8b949e; z-index: 100;
        }
        .legend-item { display: flex; align-items: center; padding: 5px 0; cursor: pointer; border-radius: 4px; padding: 6px 8px; margin: 2px -8px; }
        .legend-item:hover { background: rgba(255,255,255,0.05); }
        .legend-item.active { background: rgba(94,234,212,0.15); }
        .legend-item.dimmed { opacity: 0.4; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
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
            position: absolute; top: 100px; right: 16px; width: 300px; max-height: 500px; overflow-y: auto;
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
        .neighbor-item { padding: 6px 8px; margin: 4px 0; background: rgba(255,255,255,0.03); border-radius: 4px; cursor: pointer; display: flex; align-items: center; }
        .neighbor-item:hover { background: rgba(94,234,212,0.1); }
        .neighbor-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
        .neighbor-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        #mode-indicator { position: absolute; bottom: 16px; right: 16px; background: rgba(94,234,212,0.1); border: 1px solid #5eead4; padding: 8px 12px; border-radius: 6px; font-size: 10px; color: #5eead4; z-index: 10; }

        /* Chat Panel */
        #chat-toggle {
            position: absolute; top: 50%; right: 0; transform: translateY(-50%);
            background: rgba(10,10,18,0.95); border: 1px solid #5eead4; border-right: none;
            padding: 12px 8px; border-radius: 8px 0 0 8px; cursor: pointer;
            color: #5eead4; font-size: 18px; z-index: 200;
            transition: right 0.3s ease;
        }
        #chat-toggle:hover { background: rgba(94,234,212,0.2); }
        #chat-toggle.open { right: 380px; }

        #chat-panel {
            position: absolute; top: 0; right: -400px; width: 380px; height: 100vh;
            background: rgba(10,10,18,0.98); border-left: 1px solid #1a1a2e;
            display: flex; flex-direction: column; z-index: 150;
            transition: right 0.3s ease;
        }
        #chat-panel.open { right: 0; }

        #chat-header {
            padding: 16px; border-bottom: 1px solid #1a1a2e;
            display: flex; align-items: center; justify-content: space-between;
        }
        #chat-header h2 { font-size: 14px; color: #5eead4; font-weight: 500; margin: 0; }
        #chat-header .chat-actions { display: flex; gap: 8px; }
        .chat-action-btn {
            padding: 4px 10px; background: #0f0f1a; border: 1px solid #1a1a2e;
            border-radius: 4px; color: #8b949e; cursor: pointer; font-size: 10px;
        }
        .chat-action-btn:hover { border-color: #5eead4; color: #5eead4; }
        .chat-action-btn.active { background: #5eead420; border-color: #5eead4; color: #5eead4; }

        #chat-messages {
            flex: 1; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
        }
        .chat-message {
            max-width: 90%; padding: 10px 14px; border-radius: 12px;
            font-size: 13px; line-height: 1.5; word-wrap: break-word;
        }
        .chat-message.user {
            align-self: flex-end; background: #5eead420; color: #e0e0ff;
            border-bottom-right-radius: 4px;
        }
        .chat-message.assistant {
            align-self: flex-start; background: #1a1a2e; color: #e0e0ff;
            border-bottom-left-radius: 4px;
        }
        .chat-message.system {
            align-self: center; background: transparent; color: #6b6b8a;
            font-size: 11px; text-align: center;
        }
        .chat-message .activated-info {
            margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a2a4e;
            font-size: 11px; color: #5eead4; cursor: pointer;
        }
        .chat-message .activated-info:hover { text-decoration: underline; }

        #chat-input-area {
            padding: 12px 16px; border-top: 1px solid #1a1a2e;
            display: flex; gap: 8px;
        }
        #chat-input {
            flex: 1; padding: 10px 14px; background: #0f0f1a;
            border: 1px solid #1a1a2e; border-radius: 8px;
            color: #e0e0ff; font-size: 13px; resize: none;
        }
        #chat-input:focus { outline: none; border-color: #5eead4; }
        #chat-input::placeholder { color: #4a4a6a; }
        #chat-send {
            padding: 10px 16px; background: #5eead4; border: none;
            border-radius: 8px; color: #0a0a12; font-weight: 600;
            cursor: pointer; font-size: 13px;
        }
        #chat-send:hover { background: #4dd4c0; }
        #chat-send:disabled { background: #2a2a4e; color: #6b6b8a; cursor: not-allowed; }

        /* Activation glow animation */
        @keyframes activationPulse {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        .activation-glow { animation: activationPulse 1.5s ease-in-out infinite; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <canvas id="label-canvas"></canvas>
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
        <div class="legend-item" data-type="concept" onclick="toggleTypeFilter('concept')">
            <span class="legend-dot" style="background:#5eead4"></span>Concept<span class="legend-count" id="c-count">0</span>
        </div>
        <div class="legend-item" data-type="semantic" onclick="toggleTypeFilter('semantic')">
            <span class="legend-dot" style="background:#a78bfa"></span>Semantic<span class="legend-count" id="s-count">0</span>
        </div>
        <div class="legend-item" data-type="episodic" onclick="toggleTypeFilter('episodic')">
            <span class="legend-dot" style="background:#f472b6"></span>Episodic<span class="legend-count" id="e-count">0</span>
        </div>
        <div class="control-row">
            <button class="toggle-btn" id="cluster-btn" onclick="toggleClusters()">Clusters</button>
            <button class="toggle-btn" id="bundle-btn" onclick="toggleBundling()">Bundle</button>
            <button class="toggle-btn" id="constellation-btn" onclick="toggleConstellation()">✨ Constellation</button>
        </div>
    </div>

    <div id="stats">Loading...</div>
    <div id="loading">Loading graph...</div>
    <div id="mode-indicator">WebGL</div>

    <button id="chat-toggle" onclick="toggleChat()">💬</button>
    <div id="chat-panel">
        <div id="chat-header">
            <h2>Chat with Memory</h2>
            <div class="chat-actions">
                <button class="chat-action-btn" id="show-activation-btn" onclick="showLastActivation()">Show Activation</button>
                <button class="chat-action-btn" onclick="clearChat()">Clear</button>
            </div>
        </div>
        <div id="chat-messages">
            <div class="chat-message system">Ask questions about your knowledge graph. Activated memories will be highlighted.</div>
        </div>
        <div id="chat-input-area">
            <textarea id="chat-input" placeholder="Ask a question..." rows="1"></textarea>
            <button id="chat-send" onclick="sendChat()">Send</button>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const labelCanvas = document.getElementById('label-canvas');
        const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
        const labelCtx = labelCanvas.getContext('2d');

        if (!gl) {
            document.getElementById('loading').innerHTML = 'WebGL not supported';
            throw new Error('WebGL not supported');
        }

        const typeColors = { concept: [0.37, 0.92, 0.83], semantic: [0.65, 0.55, 0.98], episodic: [0.96, 0.45, 0.71] };
        const typeColorsHex = { concept: '#5eead4', semantic: '#a78bfa', episodic: '#f472b6' };

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
        let neighborNodes = new Set();
        let activatedNodes = new Set(); // Nodes activated by chat
        let showClusters = false, showBundling = false, showConstellation = false;
        let clusterCenters = {}; // { clusterId: { x, y, name, nodeCount } }
        let clusterMeta = { centers: [], edges: [] }; // Precomputed cluster data
        let renderMode = 'culled'; // 'cluster', 'culled', 'badge'
        let clickRevealedNode = null; // Node whose edges are revealed in badge mode
        let typeFilters = new Set(); // empty = show all

        let isDragging = false, lastMouseX = 0, lastMouseY = 0, dragStartX = 0, dragStartY = 0;
        let animationFrameId = null;
        let loadingViewport = false, lastViewport = null, viewportDebounceTimer = null;
        let currentSampleRate = 10; // Start with sampling (every 10th node)

        // WebGL shaders
        const vertexShaderSrc = `
            attribute vec2 a_position;
            attribute vec4 a_color;
            attribute float a_size;
            uniform vec2 u_resolution;
            uniform vec2 u_view;
            uniform float u_scale;
            varying vec4 v_color;
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
            varying vec4 v_color;
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);
                if (dist > 0.5) discard;
                // Glow effect for important nodes
                float glow = v_color.a > 0.95 ? (1.0 - dist * 1.5) * 0.3 : 0.0;
                float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
                gl_FragColor = vec4(v_color.rgb + glow, alpha * v_color.a);
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
                return null;
            }
            return shader;
        }

        function createProgram(vertexSrc, fragmentSrc) {
            const program = gl.createProgram();
            gl.attachShader(program, createShader(gl.VERTEX_SHADER, vertexSrc));
            gl.attachShader(program, createShader(gl.FRAGMENT_SHADER, fragmentSrc));
            gl.linkProgram(program);
            return program;
        }

        const nodeProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);
        const lineProgram = createProgram(lineVertexShaderSrc, lineFragmentShaderSrc);
        const nodeBuffer = gl.createBuffer();
        const lineBuffer = gl.createBuffer();

        function resize() {
            const dpr = window.devicePixelRatio;
            canvas.width = window.innerWidth * dpr;
            canvas.height = window.innerHeight * dpr;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
            labelCanvas.width = window.innerWidth * dpr;
            labelCanvas.height = window.innerHeight * dpr;
            labelCanvas.style.width = window.innerWidth + 'px';
            labelCanvas.style.height = window.innerHeight + 'px';
            gl.viewport(0, 0, canvas.width, canvas.height);
            render();
        }

        function screenToWorld(sx, sy) {
            // Match shader: clipSpace = (pos / u_resolution) * 2.0, with u_resolution = size/2
            // This means effective multiplier is 2 in the transform
            return {
                x: (sx - window.innerWidth / 2) / (scale * 2) + viewX,
                y: -(sy - window.innerHeight / 2) / (scale * 2) + viewY
            };
        }

        function worldToScreen(wx, wy) {
            // Match shader transform: pos * 2 / resolution * 2 = pos * 4 / size
            // Screen = (clip + 1) * size/2, so final = pos * 2 + size/2
            return {
                x: (wx - viewX) * scale * 2 + window.innerWidth / 2,
                y: -(wy - viewY) * scale * 2 + window.innerHeight / 2
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
                   Math.abs(v1.max_x - v2.max_x) > threshold;
        }

        async function loadViewportData() {
            if (loadingViewport) return;
            const vp = getViewportBounds();

            // Determine sample rate based on zoom level
            // At very low zoom (zoomed out), use high sampling
            // At higher zoom (zoomed in), load all nodes
            const newSampleRate = scale < 0.01 ? 10 : (scale < 0.05 ? 5 : 1);

            // Skip if viewport hasn't changed AND sample rate is same
            if (!viewportChanged(vp, lastViewport) && newSampleRate === currentSampleRate) return;

            loadingViewport = true;
            lastViewport = vp;
            currentSampleRate = newSampleRate;

            try {
                const url = `/admin/graph/data?min_x=${vp.min_x}&max_x=${vp.max_x}&min_y=${vp.min_y}&max_y=${vp.max_y}&sample=${currentSampleRate}`;
                const data = await (await fetch(url)).json();

                nodes = data.nodes;
                links = data.links;
                nodeMap = {};
                nodes.forEach(n => nodeMap[n.id] = n);

                console.log(`Loaded ${nodes.length} nodes (sample=${currentSampleRate}, scale=${scale.toFixed(4)})`);
                render();
            } catch (e) {
                console.error('Failed to load:', e);
            } finally {
                loadingViewport = false;
            }
        }

        function scheduleViewportLoad() {
            if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
            viewportDebounceTimer = setTimeout(loadViewportData, 100);
        }

        function isNodeVisible(node) {
            if (typeFilters.size > 0 && !typeFilters.has(node.type)) return false;
            return true;
        }

        function getNodeColor(node) {
            if (!isNodeVisible(node)) return [0.1, 0.1, 0.12, 0.2];

            const isHighlighted = highlightedNodes.size === 0 || highlightedNodes.has(node.id);
            const isNeighbor = neighborNodes.has(node.id);
            const isSelected = node === selectedNode;
            const isActivated = activatedNodes.has(node.id);

            // Activated nodes from chat get bright glow
            if (isActivated) {
                // Pulsing glow effect using time
                const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 300);
                return [1.0, 0.95, 0.3, pulse]; // Golden glow for activated
            }

            if (!isHighlighted && !isNeighbor && !isSelected && highlightedNodes.size > 0) {
                return [0.15, 0.15, 0.2, 0.3];
            }
            if (!isHighlighted && !isNeighbor && !isSelected && neighborNodes.size > 0) {
                return [0.15, 0.15, 0.2, 0.3];
            }

            let color;
            if (showClusters) {
                color = clusterPalette[node.cluster % clusterPalette.length];
            } else {
                color = typeColors[node.type] || typeColors.concept;
            }

            // Add glow (alpha > 0.95) for important nodes
            const importance = node.conn > 10 ? 1.0 : 0.9;
            return [color[0], color[1], color[2], importance];
        }

        function isInViewport(x, y) {
            const pos = worldToScreen(x, y);
            const margin = 50;
            return pos.x >= -margin && pos.x <= window.innerWidth + margin &&
                   pos.y >= -margin && pos.y <= window.innerHeight + margin;
        }

        function determineRenderMode() {
            // Count visible nodes and find max connections
            let visibleCount = 0;
            let maxConn = 0;
            for (const node of nodes) {
                if (isNodeVisible(node) && isInViewport(node.x, node.y)) {
                    visibleCount++;
                    if (node.conn > maxConn) maxConn = node.conn;
                }
            }

            // Adaptive thresholds based on edge complexity
            const edgeComplexity = visibleCount * maxConn;

            if (edgeComplexity > 50000 || maxConn > 500) {
                return 'badge';
            } else if (visibleCount > 300 || edgeComplexity > 10000) {
                return 'cluster';
            } else {
                return 'culled';
            }
        }

        function render() {
            gl.clearColor(0.039, 0.039, 0.071, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;
            const dpr = window.devicePixelRatio;

            // Clear label canvas
            labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);

            // Determine render mode adaptively
            renderMode = determineRenderMode();

            // Draw edges based on render mode
            gl.useProgram(lineProgram);
            gl.uniform2f(gl.getUniformLocation(lineProgram, 'u_resolution'), w / 2, h / 2);
            gl.uniform2f(gl.getUniformLocation(lineProgram, 'u_view'), viewX, viewY);
            gl.uniform1f(gl.getUniformLocation(lineProgram, 'u_scale'), scale);

            const posLoc = gl.getAttribLocation(lineProgram, 'a_position');
            gl.enableVertexAttribArray(posLoc);

            if (renderMode === 'cluster' && clusterMeta.centers.length > 0) {
                // CLUSTER MODE: Draw on 2D canvas for better visibility (constellation style)
                // Skip WebGL edges, we'll draw them on labelCtx after nodes

            } else if (renderMode === 'culled' && links.length > 0) {
                // CULLED MODE: Draw node edges only if BOTH endpoints in viewport
                const hasSelection = selectedNode || neighborNodes.size > 0;
                const normalLineData = [];
                const glowLineData = [];

                for (const link of links) {
                    const s = nodeMap[link.source];
                    const t = nodeMap[link.target];
                    // Viewport culling: only draw if BOTH nodes are visible in viewport
                    if (s && t && isNodeVisible(s) && isNodeVisible(t) &&
                        isInViewport(s.x, s.y) && isInViewport(t.x, t.y)) {

                        const isGlowLink = selectedNode && (
                            (s === selectedNode && neighborNodes.has(t.id)) ||
                            (t === selectedNode && neighborNodes.has(s.id))
                        );

                        const targetArray = isGlowLink ? glowLineData : normalLineData;

                        if (showBundling) {
                            const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2;
                            const dx = t.x - s.x, dy = t.y - s.y;
                            const cx = mx - dy * 0.15, cy = my + dx * 0.15;
                            for (let i = 0; i < 8; i++) {
                                const t1 = i / 8, t2 = (i + 1) / 8;
                                targetArray.push(
                                    (1-t1)*(1-t1)*s.x + 2*(1-t1)*t1*cx + t1*t1*t.x,
                                    (1-t1)*(1-t1)*s.y + 2*(1-t1)*t1*cy + t1*t1*t.y,
                                    (1-t2)*(1-t2)*s.x + 2*(1-t2)*t2*cx + t2*t2*t.x,
                                    (1-t2)*(1-t2)*s.y + 2*(1-t2)*t2*cy + t2*t2*t.y
                                );
                            }
                        } else {
                            targetArray.push(s.x, s.y, t.x, t.y);
                        }
                    }
                }

                if (normalLineData.length > 0) {
                    const normalOpacity = hasSelection ? 0.1 : 0.4;
                    gl.uniform4f(gl.getUniformLocation(lineProgram, 'u_color'), 0.37, 0.92, 0.83, normalOpacity);
                    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
                    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normalLineData), gl.DYNAMIC_DRAW);
                    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
                    gl.drawArrays(gl.LINES, 0, normalLineData.length / 2);
                }

                if (glowLineData.length > 0) {
                    gl.uniform4f(gl.getUniformLocation(lineProgram, 'u_color'), 0.37, 0.92, 0.83, 0.6);
                    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
                    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(glowLineData), gl.DYNAMIC_DRAW);
                    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
                    gl.drawArrays(gl.LINES, 0, glowLineData.length / 2);
                    gl.uniform4f(gl.getUniformLocation(lineProgram, 'u_color'), 0.6, 1.0, 0.95, 0.9);
                    gl.drawArrays(gl.LINES, 0, glowLineData.length / 2);
                }

            } else if (renderMode === 'badge') {
                // BADGE MODE: No regular edges, but draw click-revealed edges
                if (clickRevealedNode && neighborNodes.size > 0) {
                    const revealedLineData = [];
                    for (const neighborId of neighborNodes) {
                        const neighbor = nodeMap[neighborId];
                        if (neighbor && isInViewport(neighbor.x, neighbor.y)) {
                            revealedLineData.push(
                                clickRevealedNode.x, clickRevealedNode.y,
                                neighbor.x, neighbor.y
                            );
                        }
                    }
                    if (revealedLineData.length > 0) {
                        gl.uniform4f(gl.getUniformLocation(lineProgram, 'u_color'), 0.6, 1.0, 0.95, 0.7);
                        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
                        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(revealedLineData), gl.DYNAMIC_DRAW);
                        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
                        gl.drawArrays(gl.LINES, 0, revealedLineData.length / 2);
                    }
                }
            }

            // Draw nodes
            if (nodes.length > 0) {
                gl.useProgram(nodeProgram);
                gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_resolution'), w / 2, h / 2);
                gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_view'), viewX, viewY);
                gl.uniform1f(gl.getUniformLocation(nodeProgram, 'u_scale'), scale);

                const nodeData = [];
                const labelsToRender = [];

                for (const node of nodes) {
                    const color = getNodeColor(node);
                    // Node sizes: base by connections, scale with zoom for visibility
                    const baseSize = Math.max(30, Math.min(120, Math.sqrt(node.conn || 1) * 15));
                    // Scale up when zoomed in (larger dots at higher zoom) - very aggressive
                    const zoomBoost = Math.max(1, Math.min(50, scale * 80));
                    // At low zoom, ensure min 8px on screen
                    const minScreenSize = 8 / scale;
                    const size = Math.max(minScreenSize, baseSize * zoomBoost);

                    let finalColor = color;
                    if (node === selectedNode) {
                        finalColor = [1, 1, 1, 1];
                    } else if (node === hoveredNode) {
                        finalColor = [1, 1, 1, 0.9];
                    } else if (neighborNodes.has(node.id)) {
                        finalColor = [color[0] * 1.2, color[1] * 1.2, color[2] * 1.2, 1];
                    }

                    nodeData.push(node.x, node.y, finalColor[0], finalColor[1], finalColor[2], finalColor[3], size);

                    // Collect labels for high-connection nodes
                    if (isNodeVisible(node) && node.conn > 5 && scale > 0.3) {
                        const pos = worldToScreen(node.x, node.y);
                        labelsToRender.push({ node, pos, size });
                    }
                }

                gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nodeData), gl.DYNAMIC_DRAW);

                const stride = 7 * 4;
                const posLoc = gl.getAttribLocation(nodeProgram, 'a_position');
                const colorLoc = gl.getAttribLocation(nodeProgram, 'a_color');
                const sizeLoc = gl.getAttribLocation(nodeProgram, 'a_size');

                gl.enableVertexAttribArray(posLoc);
                gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
                gl.enableVertexAttribArray(colorLoc);
                gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 8);
                gl.enableVertexAttribArray(sizeLoc);
                gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, stride, 24);

                gl.drawArrays(gl.POINTS, 0, nodes.length);

                // Render labels on 2D canvas
                labelCtx.save();
                labelCtx.scale(dpr, dpr);
                labelCtx.font = '11px -apple-system, sans-serif';
                labelCtx.textAlign = 'center';

                for (const { node, pos, size } of labelsToRender) {
                    if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                    const label = node.name.length > 20 ? node.name.slice(0, 20) + '...' : node.name;
                    const screenSize = size * scale;

                    labelCtx.fillStyle = 'rgba(255,255,255,0.7)';
                    labelCtx.fillText(label, pos.x, pos.y + screenSize / 2 + 14);
                }

                // BADGE MODE: Draw connection count badges
                if (renderMode === 'badge') {
                    labelCtx.font = 'bold 10px -apple-system, sans-serif';
                    labelCtx.textAlign = 'center';
                    labelCtx.textBaseline = 'middle';

                    for (const node of nodes) {
                        if (!isNodeVisible(node) || !isInViewport(node.x, node.y)) continue;
                        if (node.conn < 1) continue;

                        const pos = worldToScreen(node.x, node.y);
                        const baseSize = Math.max(24, Math.min(96, Math.sqrt(node.conn || 1) * 16));
                        const screenSize = Math.max(4, baseSize * scale);

                        // Badge position (top-right of node)
                        const badgeX = pos.x + screenSize / 2 + 8;
                        const badgeY = pos.y - screenSize / 2 - 4;

                        // Badge background
                        const badgeText = node.conn > 999 ? (node.conn / 1000).toFixed(1) + 'k' : String(node.conn);
                        const badgeWidth = labelCtx.measureText(badgeText).width + 8;

                        // Color based on connection count
                        let badgeColor;
                        if (node.conn > 1000) badgeColor = '#f472b6'; // pink for supernodes
                        else if (node.conn > 100) badgeColor = '#fbbf24'; // yellow for hubs
                        else badgeColor = '#5eead4'; // teal for normal

                        labelCtx.fillStyle = badgeColor + '40'; // semi-transparent bg
                        labelCtx.beginPath();
                        labelCtx.roundRect(badgeX - badgeWidth/2, badgeY - 8, badgeWidth, 16, 4);
                        labelCtx.fill();

                        labelCtx.fillStyle = badgeColor;
                        labelCtx.fillText(badgeText, badgeX, badgeY);
                    }
                }

                labelCtx.restore();
            }

            // CLUSTER MODE: Draw cluster centers and edges (constellation style)
            if (renderMode === 'cluster' && clusterMeta.centers.length > 0) {
                labelCtx.save();
                labelCtx.scale(dpr, dpr);

                const centerMap = {};
                for (const c of clusterMeta.centers) {
                    centerMap[c.id] = c;
                }

                // Draw cluster-to-cluster edges (dashed, bright)
                labelCtx.strokeStyle = 'rgba(94, 234, 212, 0.5)';
                labelCtx.lineWidth = 2;
                labelCtx.setLineDash([8, 12]);

                for (const edge of clusterMeta.edges) {
                    const s = centerMap[edge.source];
                    const t = centerMap[edge.target];
                    if (!s || !t) continue;

                    const pos1 = worldToScreen(s.x, s.y);
                    const pos2 = worldToScreen(t.x, t.y);

                    // Skip if both off screen
                    if ((pos1.x < -50 || pos1.x > w + 50) && (pos2.x < -50 || pos2.x > w + 50)) continue;

                    // Line thickness based on edge count
                    labelCtx.lineWidth = Math.min(6, 1 + Math.log(edge.count + 1));

                    labelCtx.beginPath();
                    labelCtx.moveTo(pos1.x, pos1.y);
                    labelCtx.lineTo(pos2.x, pos2.y);
                    labelCtx.stroke();
                }

                labelCtx.setLineDash([]);

                // Draw cluster centers as small dots (constellation style, no glow)
                for (const center of clusterMeta.centers) {
                    const pos = worldToScreen(center.x, center.y);
                    if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                    const color = clusterPalette[center.id % clusterPalette.length];
                    // Small fixed size like constellation
                    const dotSize = Math.min(12, 4 + Math.log(center.nodeCount) * 2);

                    // Simple colored dot
                    labelCtx.fillStyle = `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 1)`;
                    labelCtx.beginPath();
                    labelCtx.arc(pos.x, pos.y, dotSize, 0, Math.PI * 2);
                    labelCtx.fill();

                    // Node count label
                    labelCtx.font = 'bold 11px -apple-system, sans-serif';
                    labelCtx.fillStyle = `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 0.9)`;
                    labelCtx.textAlign = 'center';
                    labelCtx.fillText(`${center.nodeCount}`, pos.x, pos.y + dotSize + 12);
                }

                labelCtx.restore();
            }

            // Draw constellation mode overlay
            if (showConstellation && Object.keys(clusterCenters).length > 0) {
                labelCtx.save();
                labelCtx.scale(dpr, dpr);

                const centersList = Object.entries(clusterCenters);

                // Draw constellation lines between nearby clusters
                labelCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                labelCtx.lineWidth = 1;
                labelCtx.setLineDash([5, 10]);

                for (let i = 0; i < centersList.length; i++) {
                    const [id1, c1] = centersList[i];
                    const pos1 = worldToScreen(c1.x, c1.y);

                    // Connect to 2-3 nearest clusters
                    const distances = [];
                    for (let j = 0; j < centersList.length; j++) {
                        if (i === j) continue;
                        const [id2, c2] = centersList[j];
                        const dx = c2.x - c1.x, dy = c2.y - c1.y;
                        distances.push({ j, dist: Math.sqrt(dx*dx + dy*dy) });
                    }
                    distances.sort((a, b) => a.dist - b.dist);

                    // Draw lines to nearest 2 clusters
                    for (let k = 0; k < Math.min(2, distances.length); k++) {
                        const [id2, c2] = centersList[distances[k].j];
                        const pos2 = worldToScreen(c2.x, c2.y);

                        labelCtx.beginPath();
                        labelCtx.moveTo(pos1.x, pos1.y);
                        labelCtx.lineTo(pos2.x, pos2.y);
                        labelCtx.stroke();
                    }
                }

                labelCtx.setLineDash([]);

                // Draw cluster centers as stars and labels
                for (const [clusterId, center] of centersList) {
                    const pos = worldToScreen(center.x, center.y);
                    if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                    // Draw star shape at cluster center
                    const starSize = Math.min(20, 8 + Math.log(center.nodeCount) * 3);
                    const color = clusterPalette[clusterId % clusterPalette.length];

                    // Glow
                    const gradient = labelCtx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, starSize * 2);
                    gradient.addColorStop(0, `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 0.6)`);
                    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
                    labelCtx.fillStyle = gradient;
                    labelCtx.beginPath();
                    labelCtx.arc(pos.x, pos.y, starSize * 2, 0, Math.PI * 2);
                    labelCtx.fill();

                    // Star center
                    labelCtx.fillStyle = `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 1)`;
                    labelCtx.beginPath();
                    labelCtx.arc(pos.x, pos.y, starSize / 2, 0, Math.PI * 2);
                    labelCtx.fill();

                    // Cluster name
                    labelCtx.font = 'bold 12px -apple-system, sans-serif';
                    labelCtx.fillStyle = `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 0.9)`;
                    labelCtx.textAlign = 'center';
                    labelCtx.fillText(center.name, pos.x, pos.y - starSize - 8);

                    // Node count
                    labelCtx.font = '10px -apple-system, sans-serif';
                    labelCtx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                    labelCtx.fillText(`${center.nodeCount} nodes`, pos.x, pos.y + starSize + 16);
                }

                labelCtx.restore();
            }
        }

        function findNodeAt(sx, sy) {
            const world = screenToWorld(sx, sy);
            let closest = null, closestDist = Infinity;

            for (const node of nodes) {
                if (!isNodeVisible(node)) continue;
                const dx = node.x - world.x;
                const dy = node.y - world.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                // Match bigger node sizes
                const baseSize = Math.max(24, Math.min(96, Math.sqrt(node.conn || 1) * 16));
                const threshold = (baseSize * 2) / scale + 30;

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
        function hideTooltip() { tooltip.classList.remove('visible'); }

        // Node info panel with neighbors
        async function showNodeInfo(node) {
            const panel = document.getElementById('node-info');
            const content = document.getElementById('node-info-content');
            const color = typeColorsHex[node.type];

            // Set click-revealed node for badge mode edge drawing
            clickRevealedNode = node;

            // Fetch neighbors
            let neighborsHtml = '<div style="color:#6b6b8a;padding:8px 0;">Loading...</div>';
            try {
                const res = await fetch(`/admin/graph/neighbors?node_id=${encodeURIComponent(node.id)}`);
                const data = await res.json();

                neighborNodes.clear();
                data.neighbors.forEach(n => neighborNodes.add(n.id));
                render();

                if (data.neighbors.length > 0) {
                    neighborsHtml = data.neighbors.map(n => `
                        <div class="neighbor-item" data-x="${n.x}" data-y="${n.y}" data-id="${n.id}">
                            <span class="neighbor-dot" style="background:${typeColorsHex[n.type]}"></span>
                            <span class="neighbor-name">${n.name}</span>
                        </div>
                    `).join('');
                } else {
                    neighborsHtml = '<div style="color:#6b6b8a;padding:8px 0;">No connections</div>';
                }
            } catch (e) {
                neighborsHtml = '<div style="color:#6b6b8a;padding:8px 0;">Failed to load</div>';
            }

            content.innerHTML = `
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
                ${node.fullContent && node.fullContent !== node.name ? `
                <div class="info-row">
                    <div class="info-label">Content</div>
                    <div class="info-value" style="max-height:100px;overflow-y:auto;">${node.fullContent}</div>
                </div>` : ''}
                <div class="info-row">
                    <div class="info-label">Connected Nodes</div>
                    <div style="max-height:200px;overflow-y:auto;margin-top:8px;">${neighborsHtml}</div>
                </div>
            `;
            panel.classList.add('visible');

            // Add click handlers for neighbors
            content.querySelectorAll('.neighbor-item').forEach(item => {
                item.addEventListener('click', () => {
                    const x = parseFloat(item.dataset.x);
                    const y = parseFloat(item.dataset.y);
                    const id = item.dataset.id;
                    viewX = x;
                    viewY = y;
                    if (nodeMap[id]) {
                        selectedNode = nodeMap[id];
                        showNodeInfo(selectedNode);
                    }
                    render();
                    scheduleViewportLoad();
                });
            });
        }

        function closeNodeInfo() {
            document.getElementById('node-info').classList.remove('visible');
            selectedNode = null;
            clickRevealedNode = null;
            neighborNodes.clear();
            render();
        }

        // Type filter
        function toggleTypeFilter(type) {
            const item = document.querySelector(`.legend-item[data-type="${type}"]`);

            if (typeFilters.has(type)) {
                typeFilters.delete(type);
                item.classList.remove('active');
            } else {
                // If clicking same type that's the only active one, clear all
                if (typeFilters.size === 1 && typeFilters.has(type)) {
                    typeFilters.clear();
                    document.querySelectorAll('.legend-item').forEach(el => el.classList.remove('active', 'dimmed'));
                } else {
                    typeFilters.add(type);
                    item.classList.add('active');
                }
            }

            // Update dimmed state
            document.querySelectorAll('.legend-item').forEach(el => {
                const t = el.dataset.type;
                if (typeFilters.size > 0 && !typeFilters.has(t)) {
                    el.classList.add('dimmed');
                } else {
                    el.classList.remove('dimmed');
                }
            });

            render();
        }

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

        function toggleConstellation() {
            showConstellation = !showConstellation;
            document.getElementById('constellation-btn').classList.toggle('active', showConstellation);
            if (showConstellation) {
                computeClusterCenters();
            }
            render();
        }

        function computeClusterCenters() {
            clusterCenters = {};
            for (const node of nodes) {
                const cid = node.cluster || 0;
                if (!clusterCenters[cid]) {
                    clusterCenters[cid] = { x: 0, y: 0, count: 0, names: {} };
                }
                clusterCenters[cid].x += node.x;
                clusterCenters[cid].y += node.y;
                clusterCenters[cid].count++;
                // Track top concept names for cluster naming
                if (node.type === 'concept' && node.name) {
                    const name = node.name.split(' ')[0]; // First word
                    clusterCenters[cid].names[name] = (clusterCenters[cid].names[name] || 0) + (node.conn || 1);
                }
            }
            // Finalize centers and pick names
            for (const cid of Object.keys(clusterCenters)) {
                const c = clusterCenters[cid];
                c.x /= c.count;
                c.y /= c.count;
                c.nodeCount = c.count;
                // Pick most connected concept name
                const sortedNames = Object.entries(c.names).sort((a, b) => b[1] - a[1]);
                c.name = sortedNames.length > 0 ? sortedNames[0][0] : `Cluster ${cid}`;
                delete c.names;
                delete c.count;
            }
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
                const data = await (await fetch(`/admin/graph/search?q=${encodeURIComponent(q)}`)).json();

                highlightedNodes.clear();
                data.results.forEach(r => highlightedNodes.add(r.id));

                searchResults.innerHTML = data.results.length === 0
                    ? '<div style="padding:12px;color:#6b6b8a;">No results</div>'
                    : data.results.slice(0, 20).map(r => `
                        <div class="search-result" data-x="${r.x}" data-y="${r.y}" data-id="${r.id}">
                            <span class="search-result-name">${r.name}</span>
                            <span class="search-result-type" style="background:${typeColorsHex[r.type]}20;color:${typeColorsHex[r.type]}">${r.type}</span>
                        </div>
                    `).join('');

                searchResults.classList.add('visible');
                render();
            } catch (e) {
                console.error('Search failed:', e);
            }
        }

        searchResults.addEventListener('click', (e) => {
            const result = e.target.closest('.search-result');
            if (result) {
                viewX = parseFloat(result.dataset.x);
                viewY = parseFloat(result.dataset.y);
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
                viewX -= (e.clientX - lastMouseX) / scale;
                viewY += (e.clientY - lastMouseY) / scale;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
                scheduleViewportLoad();
            } else {
                const node = findNodeAt(e.clientX, e.clientY);
                if (node !== hoveredNode) {
                    hoveredNode = node;
                    node ? showTooltip(node, e.clientX, e.clientY) : hideTooltip();
                    render();
                }
            }
        });

        canvas.addEventListener('mouseup', (e) => {
            if (Math.abs(e.clientX - dragStartX) < 5 && Math.abs(e.clientY - dragStartY) < 5) {
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
            scale = Math.max(0.001, Math.min(10, scale * zoomFactor));
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
                stopActivationAnimation();
                typeFilters.clear();
                document.querySelectorAll('.legend-item').forEach(el => el.classList.remove('active', 'dimmed'));
                searchInput.value = '';
                searchResults.classList.remove('visible');
                render();
            }
        });

        // ============ CHAT FUNCTIONALITY ============
        let lastActivation = { concepts: [], memories: [] };
        let chatOpen = false;

        function toggleChat() {
            chatOpen = !chatOpen;
            document.getElementById('chat-panel').classList.toggle('open', chatOpen);
            document.getElementById('chat-toggle').classList.toggle('open', chatOpen);
        }

        function clearChat() {
            document.getElementById('chat-messages').innerHTML =
                '<div class="chat-message system">Ask questions about your knowledge graph. Activated memories will be highlighted.</div>';
            lastActivation = { concepts: [], memories: [] };
            stopActivationAnimation();
            highlightedNodes.clear();
            render();
        }

        function showLastActivation() {
            if (lastActivation.concepts.length === 0 && lastActivation.memories.length === 0) {
                return;
            }
            activatedNodes.clear();
            lastActivation.concepts.forEach(id => activatedNodes.add(id));
            lastActivation.memories.forEach(id => activatedNodes.add(id));
            highlightedNodes.clear();
            activatedNodes.forEach(id => highlightedNodes.add(id));

            // Find center of activated nodes and pan to them
            let sumX = 0, sumY = 0, count = 0;
            for (const node of nodes) {
                if (activatedNodes.has(node.id)) {
                    sumX += node.x;
                    sumY += node.y;
                    count++;
                }
            }
            if (count > 0) {
                viewX = sumX / count;
                viewY = sumY / count;
                scale = Math.min(2, scale * 1.5);
            }
            render();
            scheduleViewportLoad();
        }

        function highlightActivation(concepts, memories) {
            lastActivation = { concepts, memories };
            activatedNodes.clear();
            concepts.forEach(id => activatedNodes.add(id));
            memories.forEach(id => activatedNodes.add(id));
            highlightedNodes.clear();
            activatedNodes.forEach(id => highlightedNodes.add(id));
            startActivationAnimation();
        }

        function startActivationAnimation() {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            function animate() {
                if (activatedNodes.size === 0) {
                    animationFrameId = null;
                    return;
                }
                render();
                animationFrameId = requestAnimationFrame(animate);
            }
            animate();
        }

        function stopActivationAnimation() {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            activatedNodes.clear();
            render();
        }

        async function sendChat() {
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('chat-send');
            const messages = document.getElementById('chat-messages');
            const query = input.value.trim();

            if (!query) return;

            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'chat-message user';
            userMsg.textContent = query;
            messages.appendChild(userMsg);

            // Clear input and disable
            input.value = '';
            sendBtn.disabled = true;
            input.disabled = true;

            // Add loading message
            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'chat-message assistant';
            loadingMsg.innerHTML = '<span class="activation-glow">Thinking...</span>';
            messages.appendChild(loadingMsg);
            messages.scrollTop = messages.scrollHeight;

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'engram',
                        messages: [{ role: 'user', content: query }],
                        top_k_memories: 20,
                        top_k_episodes: 5
                    })
                });

                const data = await response.json();

                // Remove loading message
                messages.removeChild(loadingMsg);

                // Extract response
                const answer = data.choices[0].message.content;
                const concepts = data.concepts_activated || [];
                const memories = data.memories_used || [];
                const memoriesCount = data.memories_count || 0;

                // Add assistant message
                const assistantMsg = document.createElement('div');
                assistantMsg.className = 'chat-message assistant';

                // Format answer (basic markdown-like formatting)
                let formattedAnswer = answer
                    .replace(/\\n/g, '<br>')
                    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.+?)\*/g, '<em>$1</em>');

                assistantMsg.innerHTML = formattedAnswer;

                // Add activation info if there are activated nodes
                if (concepts.length > 0 || memories.length > 0) {
                    const activationInfo = document.createElement('div');
                    activationInfo.className = 'activated-info';
                    activationInfo.textContent = `Activated: ${concepts.length} concepts, ${memoriesCount} memories`;
                    activationInfo.onclick = () => highlightActivation(concepts, memories);
                    assistantMsg.appendChild(activationInfo);

                    // Auto-highlight
                    highlightActivation(concepts, memories);
                }

                messages.appendChild(assistantMsg);
                messages.scrollTop = messages.scrollHeight;

            } catch (error) {
                messages.removeChild(loadingMsg);
                const errorMsg = document.createElement('div');
                errorMsg.className = 'chat-message system';
                errorMsg.textContent = 'Error: ' + error.message;
                messages.appendChild(errorMsg);
            } finally {
                sendBtn.disabled = false;
                input.disabled = false;
                input.focus();
            }
        }

        // Chat input handlers
        document.getElementById('chat-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChat();
            }
        });

        // Auto-resize chat input
        document.getElementById('chat-input').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        async function init() {
            resize();
            window.addEventListener('resize', resize);

            const boundsData = await (await fetch('/admin/graph/bounds')).json();
            if (!boundsData.has_layout) {
                document.getElementById('loading').innerHTML = 'No layout. Run: uv run python scripts/compute_layout.py';
                return;
            }

            bounds = boundsData;
            viewX = (bounds.min_x + bounds.max_x) / 2;
            viewY = (bounds.min_y + bounds.max_y) / 2;
            // Start at max zoom out to show entire graph
            // Fit graph on screen: scale so graph fills ~90% of viewport (fully zoomed out)
            const graphWidth = bounds.max_x - bounds.min_x;
            const graphHeight = bounds.max_y - bounds.min_y;
            scale = Math.min(
                window.innerWidth * 0.9 / (graphWidth * 2),
                window.innerHeight * 0.9 / (graphHeight * 2)
            );
            // Allow very low scale for large graphs (0.001 min)
            scale = Math.max(0.001, Math.min(0.5, scale));  // Cap at 0.5 to start zoomed out
            console.log(`Initial scale: ${scale.toFixed(4)}, graph size: ${graphWidth.toFixed(0)} x ${graphHeight.toFixed(0)}`);

            const stats = await (await fetch('/admin/graph/stats')).json();
            document.getElementById('c-count').textContent = stats.concepts;
            document.getElementById('s-count').textContent = stats.semantic;
            document.getElementById('e-count').textContent = stats.episodic;
            document.getElementById('stats').innerHTML = `<div>Total: <strong style="color:#5eead4">${stats.total}</strong></div><div>Clusters: <strong style="color:#a78bfa">${stats.clusters}</strong></div>`;

            // Load cluster metadata for cluster-level rendering
            try {
                clusterMeta = await (await fetch('/admin/graph/cluster-meta')).json();
                console.log(`Loaded ${clusterMeta.centers.length} cluster centers, ${clusterMeta.edges.length} cluster edges`);
            } catch (e) {
                console.warn('No cluster metadata available, run compute_layout.py');
            }

            document.getElementById('loading').style.display = 'none';
            await loadViewportData();

            // Update mode indicator periodically
            setInterval(() => {
                const modeNames = { cluster: 'Cluster Edges', culled: 'Culled Edges', badge: 'Badge Mode' };
                document.getElementById('mode-indicator').textContent = modeNames[renderMode] || 'WebGL';
            }, 500);
        }

        init();
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    return GRAPH_HTML
