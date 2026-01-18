"""Simple graph visualization endpoint."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

router = APIRouter()

FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
<circle cx='50' cy='50' r='40' fill='#0a0a12' stroke='#5eead4' stroke-width='6'/>
<circle cx='50' cy='50' r='15' fill='#5eead4'/>
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
    """Get graph data for visualization - 10% of total."""
    db = get_db(request)

    nodes = []
    links = []

    # Get concepts (10% - ~200 from typical 2000)
    concepts = await db.execute_query(
        """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) as conn
        RETURN c.id as id, c.name as name, c.type as type, conn
        ORDER BY conn DESC
        LIMIT 200
        """
    )
    for c in concepts:
        nodes.append({
            "id": c["id"],
            "name": c["name"],
            "type": "concept",
            "conn": c["conn"] or 0,
        })

    # Get semantic memories (10% - ~80 from typical 800)
    memories = await db.execute_query(
        """
        MATCH (s:SemanticMemory)
        OPTIONAL MATCH (s)-[r]-()
        WITH s, count(r) as conn
        RETURN s.id as id, s.content as content, conn
        ORDER BY conn DESC
        LIMIT 80
        """
    )
    for m in memories:
        content = m["content"] or ""
        short = content[:40] + "..." if len(content) > 40 else content
        nodes.append({
            "id": m["id"],
            "name": short,
            "type": "semantic",
            "conn": m["conn"] or 0,
        })

    # Get episodic memories (10% - ~50 from typical 500)
    episodes = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as conn
        RETURN e.id as id, e.query as query, conn
        ORDER BY conn DESC
        LIMIT 50
        """
    )
    for e in episodes:
        query = e["query"] or ""
        short = query[:40] + "..." if len(query) > 40 else query
        nodes.append({
            "id": e["id"],
            "name": short,
            "type": "episodic",
            "conn": e["conn"] or 0,
        })

    # Get node IDs for filtering links
    node_ids = {n["id"] for n in nodes}

    # Get concept-to-concept relationships (10% - ~300)
    concept_rels = await db.execute_query(
        """
        MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
        RETURN c1.id as source, c2.id as target
        LIMIT 300
        """
    )
    for r in concept_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({"source": r["source"], "target": r["target"]})

    # Get memory-to-concept relationships (10% - ~200)
    memory_rels = await db.execute_query(
        """
        MATCH (s:SemanticMemory)-[:ABOUT]->(c:Concept)
        RETURN s.id as source, c.id as target
        LIMIT 200
        """
    )
    for r in memory_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({"source": r["source"], "target": r["target"]})

    # Get episode-to-concept relationships (10% - ~150)
    episode_rels = await db.execute_query(
        """
        MATCH (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
        RETURN e.id as source, c.id as target
        LIMIT 150
        """
    )
    for r in episode_rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            links.append({"source": r["source"], "target": r["target"]})

    return {"nodes": nodes, "links": links}


GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Memory Graph</title>
    <style>
        * { margin: 0; padding: 0; }
        body { background: #0a0a12; font-family: -apple-system, sans-serif; }
        #graph { width: 100vw; height: 100vh; }
        #info {
            position: absolute;
            top: 16px;
            left: 16px;
            color: #5eead4;
            font-size: 24px;
            font-weight: bold;
            z-index: 10;
        }
        #stats {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(10,10,18,0.9);
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #1a1a2e;
            font-size: 12px;
            color: #8b949e;
            z-index: 10;
        }
        #legend {
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: rgba(10,10,18,0.9);
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #1a1a2e;
            font-size: 12px;
            color: #e0e0ff;
            z-index: 10;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 6px 0;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #5eead4;
            z-index: 10;
        }
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="info">Engram</div>
    <div id="stats">Loading...</div>
    <div id="legend">
        <div class="legend-item"><span class="legend-dot" style="background:#5eead4"></span>Concept</div>
        <div class="legend-item"><span class="legend-dot" style="background:#a78bfa"></span>Semantic</div>
        <div class="legend-item"><span class="legend-dot" style="background:#f472b6"></span>Episodic</div>
    </div>
    <div id="loading">Loading...</div>

    <script src="https://unpkg.com/force-graph"></script>
    <script>
        const colors = {
            concept: '#5eead4',
            semantic: '#a78bfa',
            episodic: '#f472b6'
        };

        const Graph = ForceGraph()
            (document.getElementById('graph'))
            .backgroundColor('#0a0a12')
            .nodeCanvasObject((node, ctx) => {
                // Size based on connections (min 4, max 20)
                const size = Math.min(20, Math.max(4, Math.sqrt(node.conn || 1) * 3));
                const color = colors[node.type] || '#5eead4';

                // Simple circle
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
            })
            .nodePointerAreaPaint((node, color, ctx) => {
                const size = Math.min(20, Math.max(4, Math.sqrt(node.conn || 1) * 3)) + 4;
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
            })
            .linkColor(() => '#2a2a4e')
            .linkWidth(1)
            .nodeLabel(n => n.name)
            .onNodeClick(node => {
                Graph.centerAt(node.x, node.y, 500);
                Graph.zoom(2, 500);
            })
            .cooldownTime(3000)
            .d3VelocityDecay(0.3);

        // Simple forces - just spread out evenly
        Graph.d3Force('charge').strength(-80);
        Graph.d3Force('link').distance(60);

        fetch('/admin/graph/data')
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';

                const c = data.nodes.filter(n => n.type === 'concept').length;
                const s = data.nodes.filter(n => n.type === 'semantic').length;
                const e = data.nodes.filter(n => n.type === 'episodic').length;

                document.getElementById('stats').innerHTML =
                    'Nodes: <b style="color:#5eead4">' + data.nodes.length + '</b><br>' +
                    'Links: <b style="color:#5eead4">' + data.links.length + '</b><br><br>' +
                    'Concepts: ' + c + '<br>' +
                    'Semantic: ' + s + '<br>' +
                    'Episodic: ' + e;

                Graph.graphData(data);

                // Zoom out after load
                setTimeout(() => Graph.zoom(0.5, 1000), 2000);
            });
    </script>
</body>
</html>
"""


@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    """Serve the graph visualization page."""
    return GRAPH_HTML
