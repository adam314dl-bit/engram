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
    limit: int = Query(5000, ge=100, le=50000),
) -> dict:
    """Get graph data with optimized single query."""
    db = get_db(request)

    viewport_filter = ""
    params: dict = {"limit": limit}
    if all(v is not None for v in [min_x, max_x, min_y, max_y]):
        viewport_filter = """
            AND n.layout_x >= $min_x AND n.layout_x <= $max_x
            AND n.layout_y >= $min_y AND n.layout_y <= $max_y
        """
        params.update({"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y})

    # Single optimized query for all node types (removed expensive connection count)
    all_nodes = await db.execute_query(
        f"""
        MATCH (n)
        WHERE n.layout_x IS NOT NULL
          AND (n:Concept OR n:SemanticMemory OR n:EpisodicMemory)
          {viewport_filter}
        RETURN n.id as id,
               COALESCE(n.name, LEFT(n.content, 50), LEFT(n.query, 40)) as name,
               n.content as fullContent,
               CASE
                   WHEN n:Concept THEN 'concept'
                   WHEN n:SemanticMemory THEN 'semantic'
                   ELSE 'episodic'
               END as type,
               COALESCE(n.type, n.memory_type, n.behavior_name, 'unknown') as subtype,
               n.layout_x as x, n.layout_y as y,
               n.cluster as cluster, n.conn as conn,
               n.level0 as level0, n.level1 as level1, n.level2 as level2,
               n.level3 as level3, n.level4 as level4
        LIMIT $limit
        """,
        **params
    )

    nodes = []
    for n in all_nodes:
        name = n["name"] or ""
        nodes.append({
            "id": n["id"],
            "name": name[:50] + "..." if len(name) > 50 else name,
            "fullContent": n["fullContent"],
            "type": n["type"],
            "subtype": n["subtype"] or "unknown",
            "x": n["x"], "y": n["y"],
            "cluster": n["cluster"] or 0,
            "conn": n["conn"] or 0,
            "level0": n["level0"] or 0, "level1": n["level1"] or 0, "level2": n["level2"] or 0,
            "level3": n["level3"] or 0, "level4": n["level4"] or 0,
        })

    node_ids = {n["id"] for n in nodes}

    # Single query for all relationships
    rels = await db.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE (a:Concept OR a:SemanticMemory OR a:EpisodicMemory)
          AND (b:Concept OR b:SemanticMemory OR b:EpisodicMemory)
        RETURN a.id as source, b.id as target, type(r) as relType
        """
    )

    links = []
    for r in rels:
        if r["source"] in node_ids and r["target"] in node_ids:
            rel_type = r["relType"].lower() if r["relType"] else "related"
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


@router.get("/admin/graph/clusters")
async def get_cluster_info(request: Request) -> dict:
    """Get L0 clusters + inter-L0 edges (initial view)."""
    db = get_db(request)

    # Get L0 clusters with centers
    l0_clusters = await db.execute_query(
        """
        MATCH (n)
        WHERE n.layout_x IS NOT NULL AND n.level0 IS NOT NULL
          AND (n:Concept OR n:SemanticMemory OR n:EpisodicMemory)
        WITH n.level0 as l0,
             avg(n.layout_x) as center_x,
             avg(n.layout_y) as center_y,
             count(n) as node_count,
             collect(n.name)[0] as top_name
        RETURN l0, center_x, center_y, node_count, top_name
        ORDER BY node_count DESC
        """
    )

    clusters = [{
        "id": c["l0"],
        "x": c["center_x"],
        "y": c["center_y"],
        "count": c["node_count"],
        "name": (c["top_name"] or "Cluster")[:30]
    } for c in l0_clusters]

    # Get inter-L0 edges
    edges = await db.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.level0 IS NOT NULL AND b.level0 IS NOT NULL AND a.level0 <> b.level0
          AND (a:Concept OR a:SemanticMemory OR a:EpisodicMemory)
          AND (b:Concept OR b:SemanticMemory OR b:EpisodicMemory)
        WITH a.level0 as from_id, b.level0 as to_id, count(r) as weight
        RETURN from_id, to_id, weight
        ORDER BY weight DESC
        LIMIT 200
        """
    )

    return {
        "clusters": clusters,
        "edges": [{"from": e["from_id"], "to": e["to_id"], "weight": e["weight"]} for e in edges],
        "total_nodes": sum(c["count"] for c in clusters)
    }


@router.get("/admin/graph/clusters/l1/{l0_id}")
async def get_l1_clusters(request: Request, l0_id: int) -> dict:
    """Get L1 clusters within an L0 + inter-L1 edges."""
    db = get_db(request)

    # Get L1 clusters within this L0
    l1_clusters = await db.execute_query(
        """
        MATCH (n)
        WHERE n.layout_x IS NOT NULL AND n.level0 = $l0 AND n.level1 IS NOT NULL
          AND (n:Concept OR n:SemanticMemory OR n:EpisodicMemory)
        WITH n.level1 as l1,
             avg(n.layout_x) as center_x,
             avg(n.layout_y) as center_y,
             count(n) as node_count,
             collect(n.name)[0] as top_name
        RETURN l1, center_x, center_y, node_count, top_name
        ORDER BY node_count DESC
        """,
        l0=l0_id
    )

    clusters = [{
        "id": c["l1"],
        "x": c["center_x"],
        "y": c["center_y"],
        "count": c["node_count"],
        "name": (c["top_name"] or "Group")[:25]
    } for c in l1_clusters]

    # Get inter-L1 edges within this L0
    edges = await db.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.level0 = $l0 AND b.level0 = $l0
          AND a.level1 IS NOT NULL AND b.level1 IS NOT NULL AND a.level1 <> b.level1
          AND (a:Concept OR a:SemanticMemory OR a:EpisodicMemory)
          AND (b:Concept OR b:SemanticMemory OR b:EpisodicMemory)
        WITH a.level1 as from_id, b.level1 as to_id, count(r) as weight
        RETURN from_id, to_id, weight
        ORDER BY weight DESC
        LIMIT 200
        """,
        l0=l0_id
    )

    return {
        "clusters": clusters,
        "edges": [{"from": e["from_id"], "to": e["to_id"], "weight": e["weight"]} for e in edges],
        "parent": {"l0": l0_id}
    }


@router.get("/admin/graph/clusters/l2/{l0_id}/{l1_id}")
async def get_l2_clusters(request: Request, l0_id: int, l1_id: int) -> dict:
    """Get L2 clusters within an L1 + inter-L2 edges."""
    db = get_db(request)

    # Get L2 clusters within this L1
    l2_clusters = await db.execute_query(
        """
        MATCH (n)
        WHERE n.layout_x IS NOT NULL AND n.level0 = $l0 AND n.level1 = $l1 AND n.level2 IS NOT NULL
          AND (n:Concept OR n:SemanticMemory OR n:EpisodicMemory)
        WITH n.level2 as l2,
             avg(n.layout_x) as center_x,
             avg(n.layout_y) as center_y,
             count(n) as node_count,
             collect(n.name)[0] as top_name
        RETURN l2, center_x, center_y, node_count, top_name
        ORDER BY node_count DESC
        """,
        l0=l0_id, l1=l1_id
    )

    clusters = [{
        "id": c["l2"],
        "x": c["center_x"],
        "y": c["center_y"],
        "count": c["node_count"],
        "name": (c["top_name"] or "Set")[:20]
    } for c in l2_clusters]

    # Get inter-L2 edges within this L1
    edges = await db.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.level0 = $l0 AND b.level0 = $l0
          AND a.level1 = $l1 AND b.level1 = $l1
          AND a.level2 IS NOT NULL AND b.level2 IS NOT NULL AND a.level2 <> b.level2
          AND (a:Concept OR a:SemanticMemory OR a:EpisodicMemory)
          AND (b:Concept OR b:SemanticMemory OR b:EpisodicMemory)
        WITH a.level2 as from_id, b.level2 as to_id, count(r) as weight
        RETURN from_id, to_id, weight
        ORDER BY weight DESC
        LIMIT 200
        """,
        l0=l0_id, l1=l1_id
    )

    return {
        "clusters": clusters,
        "edges": [{"from": e["from_id"], "to": e["to_id"], "weight": e["weight"]} for e in edges],
        "parent": {"l0": l0_id, "l1": l1_id}
    }


@router.get("/admin/graph/clusters/nodes/{l0_id}/{l1_id}/{l2_id}")
async def get_cluster_nodes(request: Request, l0_id: int, l1_id: int, l2_id: int) -> dict:
    """Get actual nodes within an L2 cluster + their edges."""
    db = get_db(request)

    # Get nodes in this L2 cluster
    all_nodes = await db.execute_query(
        """
        MATCH (n)
        WHERE n.layout_x IS NOT NULL
          AND n.level0 = $l0 AND n.level1 = $l1 AND n.level2 = $l2
          AND (n:Concept OR n:SemanticMemory OR n:EpisodicMemory)
        RETURN n.id as id,
               COALESCE(n.name, LEFT(n.content, 50), LEFT(n.query, 40)) as name,
               CASE
                   WHEN n:Concept THEN 'concept'
                   WHEN n:SemanticMemory THEN 'semantic'
                   ELSE 'episodic'
               END as type,
               n.layout_x as x, n.layout_y as y,
               n.conn as conn
        """,
        l0=l0_id, l1=l1_id, l2=l2_id
    )

    nodes = [{
        "id": n["id"],
        "name": (n["name"] or "")[:40],
        "type": n["type"],
        "x": n["x"],
        "y": n["y"],
        "conn": n["conn"] or 0
    } for n in all_nodes]

    node_ids = {n["id"] for n in nodes}

    # Get edges between these nodes
    rels = await db.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.level0 = $l0 AND a.level1 = $l1 AND a.level2 = $l2
          AND b.level0 = $l0 AND b.level1 = $l1 AND b.level2 = $l2
          AND (a:Concept OR a:SemanticMemory OR a:EpisodicMemory)
          AND (b:Concept OR b:SemanticMemory OR b:EpisodicMemory)
        RETURN a.id as source, b.id as target
        """,
        l0=l0_id, l1=l1_id, l2=l2_id
    )

    edges = [{"from": r["source"], "to": r["target"]}
             for r in rels if r["source"] in node_ids and r["target"] in node_ids]

    return {
        "nodes": nodes,
        "edges": edges,
        "parent": {"l0": l0_id, "l1": l1_id, "l2": l2_id}
    }



from engram.api.graph_template import GRAPH_HTML



@router.get("/admin/graph", response_class=HTMLResponse)
async def graph_view() -> str:
    return GRAPH_HTML
