#!/usr/bin/env python3
"""Create large test dataset with super nodes for debugging."""

import asyncio
import random
import uuid

from engram.storage.neo4j_client import Neo4jClient


# Comprehensive domains with realistic concepts
DOMAINS = {
    "machine_learning": {
        "core": ["neural_network", "deep_learning", "supervised_learning", "unsupervised_learning",
                 "reinforcement_learning", "transfer_learning", "feature_engineering", "model_training"],
        "architectures": ["transformer", "cnn", "rnn", "lstm", "gru", "autoencoder", "gan", "vae",
                         "resnet", "bert", "gpt", "diffusion_model", "unet", "yolo"],
        "techniques": ["gradient_descent", "backpropagation", "dropout", "batch_norm", "layer_norm",
                      "attention", "self_attention", "cross_attention", "positional_encoding"],
        "optimization": ["adam", "sgd", "rmsprop", "learning_rate", "weight_decay", "momentum",
                        "lr_scheduler", "warmup", "gradient_clipping"],
        "evaluation": ["loss_function", "accuracy", "precision", "recall", "f1_score", "auc_roc",
                      "confusion_matrix", "cross_validation", "overfitting", "underfitting"],
    },
    "databases": {
        "systems": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "neo4j", "cassandra",
                   "dynamodb", "sqlite", "cockroachdb", "timescaledb"],
        "concepts": ["index", "query", "transaction", "acid", "base", "cap_theorem", "normalization",
                    "denormalization", "sharding", "replication", "partitioning"],
        "operations": ["select", "insert", "update", "delete", "join", "aggregation", "grouping",
                      "filtering", "sorting", "pagination", "full_text_search"],
        "optimization": ["query_plan", "execution_plan", "index_scan", "table_scan", "buffer_pool",
                        "connection_pool", "prepared_statement", "batch_insert"],
    },
    "devops": {
        "containers": ["docker", "kubernetes", "podman", "containerd", "helm", "kustomize",
                      "docker_compose", "dockerfile", "container_registry"],
        "ci_cd": ["jenkins", "github_actions", "gitlab_ci", "circleci", "travis", "argocd",
                 "tekton", "spinnaker", "pipeline", "deployment"],
        "monitoring": ["prometheus", "grafana", "datadog", "newrelic", "jaeger", "zipkin",
                      "elk_stack", "loki", "alertmanager", "pagerduty"],
        "infrastructure": ["terraform", "ansible", "pulumi", "cloudformation", "vagrant",
                          "load_balancer", "reverse_proxy", "cdn", "dns"],
    },
    "python": {
        "core": ["asyncio", "threading", "multiprocessing", "gil", "decorator", "generator",
                "context_manager", "metaclass", "descriptor", "abc"],
        "web": ["fastapi", "django", "flask", "starlette", "uvicorn", "gunicorn", "wsgi", "asgi",
               "middleware", "cors", "authentication"],
        "data": ["pandas", "numpy", "scipy", "polars", "dask", "vaex", "pyarrow", "parquet"],
        "tools": ["pytest", "mypy", "ruff", "black", "poetry", "pip", "virtualenv", "conda"],
    },
    "cloud": {
        "aws": ["ec2", "s3", "lambda", "rds", "dynamodb", "sqs", "sns", "cloudwatch", "iam",
               "vpc", "cloudfront", "route53", "eks", "ecs"],
        "gcp": ["compute_engine", "cloud_storage", "bigquery", "cloud_functions", "gke",
               "cloud_run", "pub_sub", "cloud_sql"],
        "azure": ["virtual_machines", "blob_storage", "cosmos_db", "azure_functions", "aks",
                 "azure_devops", "active_directory"],
        "concepts": ["iaas", "paas", "saas", "serverless", "auto_scaling", "high_availability",
                    "disaster_recovery", "multi_region", "edge_computing"],
    },
    "security": {
        "auth": ["jwt", "oauth2", "openid_connect", "saml", "ldap", "mfa", "sso", "rbac", "abac"],
        "crypto": ["encryption", "hashing", "tls", "ssl", "certificate", "pki", "aes", "rsa", "sha256"],
        "attacks": ["xss", "csrf", "sql_injection", "ddos", "mitm", "phishing", "ransomware"],
        "defense": ["firewall", "waf", "ids", "ips", "siem", "penetration_testing", "vulnerability_scan"],
    },
    "networking": {
        "protocols": ["tcp", "udp", "http", "https", "websocket", "grpc", "graphql", "rest",
                     "mqtt", "amqp", "dns", "dhcp"],
        "concepts": ["ip_address", "subnet", "routing", "nat", "vpn", "proxy", "gateway",
                    "bandwidth", "latency", "throughput", "packet"],
        "tools": ["nginx", "haproxy", "envoy", "istio", "consul", "traefik"],
    },
}

# Super node topics (will have 2000+ connections)
SUPER_NODES = [
    ("api", "tool", "Application Programming Interface - core integration point"),
    ("data", "resource", "Information processed and stored by systems"),
    ("server", "resource", "Machine that processes requests and serves responses"),
    ("config", "config", "System configuration and settings"),
    ("error", "error", "System failures and exceptions"),
]

# Memory content templates
FACT_TEMPLATES = [
    "{concept} is a {type} used in {domain} for {purpose}",
    "{concept} provides {benefit} when working with {related}",
    "{concept} was designed to solve {problem} in {domain}",
    "The main advantage of {concept} is {benefit}",
    "{concept} integrates with {related} through {mechanism}",
]

PROCEDURE_TEMPLATES = [
    "To configure {concept}, first set up {related} then adjust {setting}",
    "When using {concept}, always ensure {related} is properly configured",
    "Debug {concept} issues by checking {related} logs and {setting}",
    "Optimize {concept} performance by tuning {setting} and monitoring {metric}",
    "Deploy {concept} by configuring {related} and validating {check}",
]

RELATIONSHIP_TEMPLATES = [
    "{concept} depends on {related} for {purpose}",
    "{concept} works together with {related} to achieve {goal}",
    "{concept} can replace {related} in {scenario}",
    "{concept} extends {related} with additional {feature}",
    "{related} is required before using {concept} for {purpose}",
]


async def create_large_dataset():
    """Create large dataset with super nodes."""
    db = Neo4jClient()
    await db.connect()

    print("Clearing existing test data...")
    # Don't clear - add to existing

    all_concept_ids = []
    super_node_ids = []

    # Create super nodes first
    print("Creating super nodes...")
    for name, ctype, desc in SUPER_NODES:
        concept_id = f"super_{name}_{uuid.uuid4().hex[:6]}"
        super_node_ids.append(concept_id)
        all_concept_ids.append((concept_id, name, desc, "core"))

        x = random.uniform(-10000, 10000)  # Center area
        y = random.uniform(-10000, 10000)

        await db.execute_query(
            """
            CREATE (c:Concept {
                id: $id, name: $name, type: $type, description: $desc,
                domain: 'core', activation_count: $act,
                layout_x: $x, layout_y: $y, cluster: 0
            })
            """,
            id=concept_id, name=name, type=ctype, desc=desc,
            act=random.randint(500, 1000), x=x, y=y
        )
    print(f"Created {len(super_node_ids)} super nodes")

    # Create domain concepts
    print("Creating domain concepts...")
    concept_count = 0
    for domain, categories in DOMAINS.items():
        for category, concepts in categories.items():
            for concept_name in concepts:
                concept_id = f"{domain}_{concept_name}_{uuid.uuid4().hex[:4]}"
                desc = f"{concept_name.replace('_', ' ').title()} - {category} component in {domain}"
                all_concept_ids.append((concept_id, concept_name, desc, domain))

                # Position by domain (clusters)
                domain_idx = list(DOMAINS.keys()).index(domain)
                angle = (domain_idx / len(DOMAINS)) * 6.28
                radius = random.uniform(15000, 45000)
                x = radius * random.uniform(0.8, 1.2) * (1 if random.random() > 0.5 else -1) * abs(random.gauss(0, 1))
                y = radius * random.uniform(0.8, 1.2) * (1 if random.random() > 0.5 else -1) * abs(random.gauss(0, 1))
                cluster = domain_idx + 1

                await db.execute_query(
                    """
                    CREATE (c:Concept {
                        id: $id, name: $name, type: $type, description: $desc,
                        domain: $domain, activation_count: $act,
                        layout_x: $x, layout_y: $y, cluster: $cluster
                    })
                    """,
                    id=concept_id, name=concept_name, type=category, desc=desc,
                    domain=domain, act=random.randint(1, 100), x=x, y=y, cluster=cluster
                )
                concept_count += 1

                # Connect to super nodes (this creates the 2000+ connections)
                for super_id in super_node_ids:
                    if random.random() < 0.7:  # 70% chance to connect to each super node
                        await db.execute_query(
                            """
                            MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                            CREATE (a)-[:RELATED_TO {weight: $w}]->(b)
                            """,
                            id1=concept_id, id2=super_id, w=random.uniform(0.3, 1.0)
                        )

                if concept_count % 50 == 0:
                    print(f"  Created {concept_count} concepts...")

    print(f"Created {concept_count} domain concepts")

    # Add more concepts to reach 1000+
    print("Creating additional concepts...")
    additional_needed = max(0, 1000 - len(all_concept_ids))
    for i in range(additional_needed):
        domain = random.choice(list(DOMAINS.keys()))
        base_name = random.choice(list(DOMAINS[domain].values())[0])
        variant = random.choice(["_v2", "_pro", "_lite", "_advanced", "_experimental", "_legacy"])
        concept_name = f"{base_name}{variant}"
        concept_id = f"{domain}_{concept_name}_{uuid.uuid4().hex[:4]}"
        desc = f"Variant of {base_name} for specialized use cases"
        all_concept_ids.append((concept_id, concept_name, desc, domain))

        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)
        cluster = random.randint(1, 10)

        await db.execute_query(
            """
            CREATE (c:Concept {
                id: $id, name: $name, type: 'variant', description: $desc,
                domain: $domain, activation_count: $act,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=concept_id, name=concept_name, type="variant", desc=desc,
            domain=domain, act=random.randint(1, 50), x=x, y=y, cluster=cluster
        )

        # Connect to super nodes
        for super_id in super_node_ids:
            if random.random() < 0.5:
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                    CREATE (a)-[:RELATED_TO {weight: $w}]->(b)
                    """,
                    id1=concept_id, id2=super_id, w=random.uniform(0.2, 0.8)
                )

    print(f"Total concepts: {len(all_concept_ids)}")

    # Create inter-concept relationships (within domains)
    print("Creating concept relationships...")
    rel_count = 0
    for i, (id1, name1, _, domain1) in enumerate(all_concept_ids):
        # Connect to 3-8 random concepts, prefer same domain
        num_connections = random.randint(3, 8)
        for _ in range(num_connections):
            # 70% same domain, 30% cross-domain
            if random.random() < 0.7:
                same_domain = [(id2, name2) for id2, name2, _, d in all_concept_ids if d == domain1 and id2 != id1]
                if same_domain:
                    id2, _ = random.choice(same_domain)
                else:
                    id2 = random.choice(all_concept_ids)[0]
            else:
                id2 = random.choice(all_concept_ids)[0]

            if id1 != id2:
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                    MERGE (a)-[:RELATED_TO {weight: $w}]->(b)
                    """,
                    id1=id1, id2=id2, w=random.uniform(0.3, 1.0)
                )
                rel_count += 1

        if i % 100 == 0:
            print(f"  Processed {i}/{len(all_concept_ids)} concepts for relationships...")

    print(f"Created {rel_count} relationships")

    # Create semantic memories
    print("Creating semantic memories...")
    purposes = ["data processing", "system integration", "performance optimization",
                "error handling", "security", "monitoring", "scaling"]
    benefits = ["improved performance", "better reliability", "easier maintenance",
                "reduced complexity", "enhanced security", "faster development"]
    settings = ["timeout", "buffer_size", "max_connections", "retry_count", "log_level"]
    metrics = ["latency", "throughput", "error_rate", "cpu_usage", "memory_usage"]

    memory_count = 0
    for _ in range(1200):  # Create 1200 memories
        c1 = random.choice(all_concept_ids)
        c2 = random.choice(all_concept_ids)

        template_type = random.choice(["fact", "procedure", "relationship"])
        if template_type == "fact":
            template = random.choice(FACT_TEMPLATES)
            content = template.format(
                concept=c1[1], type=c1[3], domain=c1[3],
                purpose=random.choice(purposes), benefit=random.choice(benefits),
                related=c2[1], problem=f"{c1[1]} issues", mechanism="standard protocols"
            )
        elif template_type == "procedure":
            template = random.choice(PROCEDURE_TEMPLATES)
            content = template.format(
                concept=c1[1], related=c2[1],
                setting=random.choice(settings), metric=random.choice(metrics),
                check="health endpoints"
            )
        else:
            template = random.choice(RELATIONSHIP_TEMPLATES)
            content = template.format(
                concept=c1[1], related=c2[1],
                purpose=random.choice(purposes), goal=random.choice(benefits),
                scenario=f"{c1[3]} environments", feature=random.choice(purposes)
            )

        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)
        cluster = random.randint(0, 10)

        await db.execute_query(
            """
            CREATE (m:SemanticMemory {
                id: $id, content: $content, memory_type: $mtype,
                importance: $importance, strength: $strength, access_count: $access,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=memory_id, content=content, mtype=template_type,
            importance=random.uniform(3, 10), strength=random.uniform(0.5, 1.0),
            access=random.randint(1, 50), x=x, y=y, cluster=cluster
        )

        # Link to concepts
        await db.execute_query(
            "MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
            mid=memory_id, cid=c1[0]
        )
        if random.random() > 0.3:
            await db.execute_query(
                "MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                mid=memory_id, cid=c2[0]
            )

        memory_count += 1
        if memory_count % 100 == 0:
            print(f"  Created {memory_count} memories...")

    print(f"Created {memory_count} semantic memories")

    # Create episodic memories
    print("Creating episodic memories...")
    query_templates = [
        "How do I configure {concept}?",
        "What is the best practice for {concept}?",
        "Why is {concept} not working?",
        "How to optimize {concept} performance?",
        "What are alternatives to {concept}?",
        "How does {concept} integrate with {related}?",
    ]
    behaviors = ["explain", "troubleshoot", "configure", "compare", "optimize", "debug"]

    for i in range(100):
        c1 = random.choice(all_concept_ids)
        c2 = random.choice(all_concept_ids)
        query = random.choice(query_templates).format(concept=c1[1], related=c2[1])

        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)
        cluster = random.randint(0, 10)

        await db.execute_query(
            """
            CREATE (e:EpisodicMemory {
                id: $id, query: $q, behavior_name: $behavior,
                success_count: $success, failure_count: $failure,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=episode_id, q=query, behavior=random.choice(behaviors),
            success=random.randint(0, 20), failure=random.randint(0, 5),
            x=x, y=y, cluster=cluster
        )

        # Link to concepts
        for cid in random.sample([c[0] for c in all_concept_ids], min(5, len(all_concept_ids))):
            await db.execute_query(
                "MATCH (e:EpisodicMemory {id: $eid}), (c:Concept {id: $cid}) CREATE (e)-[:ACTIVATED]->(c)",
                eid=episode_id, cid=cid
            )

    print("Created 100 episodic memories")

    # Check super node connections
    print("\nChecking super node connections...")
    for super_id in super_node_ids:
        result = await db.execute_query(
            "MATCH (c:Concept {id: $id})-[r]-() RETURN count(r) as cnt",
            id=super_id
        )
        count = result[0]["cnt"] if result else 0
        name = super_id.split("_")[1]
        print(f"  {name}: {count} connections")

    await db.close()
    print("\nDone! Restart API and refresh /constellation")


if __name__ == "__main__":
    asyncio.run(create_large_dataset())
