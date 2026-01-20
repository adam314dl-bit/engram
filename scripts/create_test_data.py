#!/usr/bin/env python3
"""Create test concepts and memories for local debugging."""

import asyncio
import random
import uuid

from engram.storage.neo4j_client import Neo4jClient


# Sample domains and concepts
DOMAINS = {
    "machine_learning": [
        ("neural_network", "tool", "Computational model inspired by biological neurons"),
        ("gradient_descent", "action", "Optimization algorithm for minimizing loss"),
        ("backpropagation", "action", "Algorithm for computing gradients"),
        ("transformer", "tool", "Attention-based architecture for sequences"),
        ("embedding", "resource", "Dense vector representation of data"),
        ("attention", "action", "Mechanism for weighing input importance"),
        ("loss_function", "tool", "Measures prediction error"),
        ("overfitting", "state", "Model memorizes training data"),
        ("regularization", "action", "Prevents overfitting"),
        ("batch_normalization", "tool", "Normalizes layer inputs"),
    ],
    "databases": [
        ("neo4j", "tool", "Graph database for connected data"),
        ("postgresql", "tool", "Relational database system"),
        ("index", "resource", "Data structure for fast lookups"),
        ("query", "action", "Request for data retrieval"),
        ("transaction", "action", "Atomic unit of database operations"),
        ("replication", "action", "Copying data across nodes"),
        ("sharding", "action", "Distributing data across servers"),
        ("acid", "config", "Database transaction properties"),
        ("deadlock", "error", "Circular wait for resources"),
        ("connection_pool", "resource", "Reusable database connections"),
    ],
    "devops": [
        ("docker", "tool", "Container platform"),
        ("kubernetes", "tool", "Container orchestration"),
        ("ci_cd", "action", "Continuous integration and deployment"),
        ("terraform", "tool", "Infrastructure as code"),
        ("prometheus", "tool", "Monitoring and alerting"),
        ("grafana", "tool", "Visualization and dashboards"),
        ("load_balancer", "resource", "Distributes traffic"),
        ("autoscaling", "action", "Automatic resource adjustment"),
        ("service_mesh", "tool", "Microservice communication layer"),
        ("helm", "tool", "Kubernetes package manager"),
    ],
    "python": [
        ("asyncio", "tool", "Asynchronous I/O framework"),
        ("pydantic", "tool", "Data validation library"),
        ("fastapi", "tool", "Modern web framework"),
        ("pytest", "tool", "Testing framework"),
        ("decorator", "action", "Function wrapper pattern"),
        ("generator", "tool", "Lazy evaluation iterator"),
        ("context_manager", "tool", "Resource management pattern"),
        ("type_hints", "config", "Static type annotations"),
        ("virtualenv", "tool", "Isolated Python environment"),
        ("pip", "tool", "Package installer"),
    ],
    "security": [
        ("jwt", "tool", "JSON Web Token for auth"),
        ("oauth", "tool", "Authorization framework"),
        ("encryption", "action", "Data protection"),
        ("hashing", "action", "One-way data transformation"),
        ("firewall", "tool", "Network security barrier"),
        ("ssl_tls", "tool", "Secure communication protocol"),
        ("xss", "error", "Cross-site scripting attack"),
        ("sql_injection", "error", "Database attack vector"),
        ("rate_limiting", "action", "Request throttling"),
        ("cors", "config", "Cross-origin resource sharing"),
    ],
}

# Memory templates
MEMORY_TEMPLATES = [
    ("fact", "{concept} is used for {description}"),
    ("fact", "{concept} helps with {related} operations"),
    ("procedure", "To use {concept}, first configure {related}"),
    ("procedure", "When {concept} fails, check {related} settings"),
    ("relationship", "{concept} works together with {related}"),
    ("relationship", "{concept} depends on {related} for proper operation"),
]


async def create_test_data(num_concepts: int = 100, num_memories: int = 200):
    """Create test concepts and memories."""
    db = Neo4jClient()
    await db.connect()

    print(f"Creating {num_concepts} concepts and {num_memories} memories...")

    # Flatten all concepts
    all_concepts = []
    for domain, concepts in DOMAINS.items():
        for name, ctype, desc in concepts:
            all_concepts.append((f"{domain}_{name}", ctype, desc, domain))

    # Create more by combining
    while len(all_concepts) < num_concepts:
        domain = random.choice(list(DOMAINS.keys()))
        base = random.choice(DOMAINS[domain])
        suffix = random.choice(["_v2", "_advanced", "_basic", "_pro", "_lite"])
        all_concepts.append((
            f"{domain}_{base[0]}{suffix}",
            base[1],
            f"Extended: {base[2]}",
            domain
        ))

    all_concepts = all_concepts[:num_concepts]

    # Create concepts with embeddings
    concept_ids = []
    for name, ctype, desc, domain in all_concepts:
        concept_id = f"concept_{uuid.uuid4().hex[:8]}"
        concept_ids.append((concept_id, name, desc))

        # Random position for layout
        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)
        cluster = random.randint(0, 20)

        # Create concept
        await db.execute_query(
            """
            CREATE (c:Concept {
                id: $id,
                name: $name,
                type: $type,
                description: $desc,
                domain: $domain,
                activation_count: $act,
                layout_x: $x,
                layout_y: $y,
                cluster: $cluster
            })
            """,
            id=concept_id,
            name=name,
            type=ctype,
            desc=desc,
            domain=domain,
            act=random.randint(1, 100),
            x=x,
            y=y,
            cluster=cluster
        )

    print(f"Created {len(concept_ids)} concepts")

    # Create some relationships between concepts
    num_relations = num_concepts * 3
    for _ in range(num_relations):
        c1 = random.choice(concept_ids)
        c2 = random.choice(concept_ids)
        if c1[0] != c2[0]:
            await db.execute_query(
                """
                MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                MERGE (a)-[:RELATED_TO {weight: $weight}]->(b)
                """,
                id1=c1[0],
                id2=c2[0],
                weight=random.uniform(0.3, 1.0)
            )

    print(f"Created ~{num_relations} concept relationships")

    # Create memories
    memory_ids = []
    for i in range(num_memories):
        c1 = random.choice(concept_ids)
        c2 = random.choice(concept_ids)
        template = random.choice(MEMORY_TEMPLATES)

        content = template[1].format(
            concept=c1[1],
            description=c1[2][:50],
            related=c2[1]
        )

        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        memory_ids.append(memory_id)

        # Random position near related concept
        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)
        cluster = random.randint(0, 20)

        await db.execute_query(
            """
            CREATE (m:SemanticMemory {
                id: $id,
                content: $content,
                memory_type: $mtype,
                importance: $importance,
                strength: $strength,
                access_count: $access,
                layout_x: $x,
                layout_y: $y,
                cluster: $cluster
            })
            """,
            id=memory_id,
            content=content,
            mtype=template[0],
            importance=random.uniform(3, 10),
            strength=random.uniform(0.5, 1.0),
            access=random.randint(1, 50),
            x=x,
            y=y,
            cluster=cluster
        )

        # Link to concepts
        await db.execute_query(
            """
            MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid})
            CREATE (m)-[:ABOUT]->(c)
            """,
            mid=memory_id,
            cid=c1[0]
        )

        if random.random() > 0.5:
            await db.execute_query(
                """
                MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid})
                CREATE (m)-[:ABOUT]->(c)
                """,
                mid=memory_id,
                cid=c2[0]
            )

    print(f"Created {len(memory_ids)} memories")

    # Create some episodic memories
    num_episodes = num_memories // 10
    for i in range(num_episodes):
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        query = random.choice([
            "How do I use {concept}?",
            "What is {concept}?",
            "Why is {concept} failing?",
            "How to configure {concept}?",
        ]).format(concept=random.choice(concept_ids)[1])

        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)
        cluster = random.randint(0, 20)

        await db.execute_query(
            """
            CREATE (e:EpisodicMemory {
                id: $id,
                query: $q,
                behavior_name: $behavior,
                success_count: $success,
                failure_count: $failure,
                layout_x: $x,
                layout_y: $y,
                cluster: $cluster
            })
            """,
            id=episode_id,
            q=query,
            behavior=random.choice(["explain", "troubleshoot", "configure", "compare"]),
            success=random.randint(0, 10),
            failure=random.randint(0, 3),
            x=x,
            y=y,
            cluster=cluster
        )

        # Link to some concepts
        for _ in range(random.randint(1, 4)):
            cid = random.choice(concept_ids)[0]
            await db.execute_query(
                """
                MATCH (e:EpisodicMemory {id: $eid}), (c:Concept {id: $cid})
                CREATE (e)-[:ACTIVATED]->(c)
                """,
                eid=episode_id,
                cid=cid
            )

    print(f"Created {num_episodes} episodic memories")

    await db.close()
    print("Done! Refresh the constellation page to see new data.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create test data for debugging")
    parser.add_argument("--concepts", type=int, default=100, help="Number of concepts")
    parser.add_argument("--memories", type=int, default=200, help="Number of memories")
    args = parser.parse_args()

    asyncio.run(create_test_data(args.concepts, args.memories))
