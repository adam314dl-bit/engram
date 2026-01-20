#!/usr/bin/env python3
"""Create semantically connected test data."""

import asyncio
import random
import uuid

from engram.storage.neo4j_client import Neo4jClient


# Concepts with their actual semantic relationships
KNOWLEDGE_GRAPH = {
    # Machine Learning core
    "neural_network": {
        "type": "tool", "domain": "ml",
        "connects": ["deep_learning", "backpropagation", "gradient_descent", "activation_function",
                     "layer", "weights", "bias", "loss_function", "optimizer", "training"],
        "desc": "Computational model with interconnected nodes that learns patterns"
    },
    "deep_learning": {
        "type": "tool", "domain": "ml",
        "connects": ["neural_network", "cnn", "rnn", "transformer", "gpu", "tensorflow", "pytorch"],
        "desc": "Neural networks with many layers for complex pattern recognition"
    },
    "transformer": {
        "type": "architecture", "domain": "ml",
        "connects": ["attention", "self_attention", "bert", "gpt", "positional_encoding",
                     "encoder", "decoder", "embedding", "tokenizer"],
        "desc": "Attention-based architecture for sequence modeling"
    },
    "attention": {
        "type": "mechanism", "domain": "ml",
        "connects": ["transformer", "self_attention", "query", "key", "value", "softmax", "weights"],
        "desc": "Mechanism to focus on relevant parts of input"
    },
    "backpropagation": {
        "type": "algorithm", "domain": "ml",
        "connects": ["neural_network", "gradient_descent", "chain_rule", "loss_function", "weights"],
        "desc": "Algorithm for computing gradients through the network"
    },
    "gradient_descent": {
        "type": "algorithm", "domain": "ml",
        "connects": ["backpropagation", "learning_rate", "optimizer", "adam", "sgd", "loss_function"],
        "desc": "Optimization algorithm that minimizes loss by following gradients"
    },
    "cnn": {
        "type": "architecture", "domain": "ml",
        "connects": ["convolution", "pooling", "image_classification", "deep_learning", "resnet", "filter"],
        "desc": "Convolutional Neural Network for image processing"
    },
    "rnn": {
        "type": "architecture", "domain": "ml",
        "connects": ["lstm", "gru", "sequence", "hidden_state", "time_series", "nlp"],
        "desc": "Recurrent Neural Network for sequential data"
    },
    "lstm": {
        "type": "architecture", "domain": "ml",
        "connects": ["rnn", "gru", "forget_gate", "memory_cell", "sequence", "vanishing_gradient"],
        "desc": "Long Short-Term Memory network with gating mechanism"
    },
    "bert": {
        "type": "model", "domain": "ml",
        "connects": ["transformer", "nlp", "pretraining", "fine_tuning", "masked_language_model", "embedding"],
        "desc": "Bidirectional transformer for NLP tasks"
    },
    "gpt": {
        "type": "model", "domain": "ml",
        "connects": ["transformer", "language_model", "text_generation", "autoregressive", "decoder"],
        "desc": "Generative Pre-trained Transformer for text generation"
    },
    "embedding": {
        "type": "concept", "domain": "ml",
        "connects": ["vector", "word2vec", "transformer", "bert", "semantic", "dimension"],
        "desc": "Dense vector representation of discrete items"
    },
    "loss_function": {
        "type": "concept", "domain": "ml",
        "connects": ["neural_network", "cross_entropy", "mse", "gradient_descent", "optimization"],
        "desc": "Function that measures prediction error"
    },
    "optimizer": {
        "type": "tool", "domain": "ml",
        "connects": ["adam", "sgd", "gradient_descent", "learning_rate", "momentum"],
        "desc": "Algorithm that updates model parameters"
    },
    "adam": {
        "type": "algorithm", "domain": "ml",
        "connects": ["optimizer", "gradient_descent", "momentum", "learning_rate", "adaptive"],
        "desc": "Adaptive Moment Estimation optimizer"
    },
    "overfitting": {
        "type": "problem", "domain": "ml",
        "connects": ["regularization", "dropout", "validation", "training", "generalization"],
        "desc": "Model memorizes training data instead of learning patterns"
    },
    "dropout": {
        "type": "technique", "domain": "ml",
        "connects": ["regularization", "overfitting", "neural_network", "training"],
        "desc": "Randomly drops neurons during training to prevent overfitting"
    },

    # Databases
    "postgresql": {
        "type": "database", "domain": "db",
        "connects": ["sql", "relational", "acid", "index", "query", "transaction", "table"],
        "desc": "Open-source relational database system"
    },
    "mongodb": {
        "type": "database", "domain": "db",
        "connects": ["nosql", "document", "json", "schema_less", "replica_set", "sharding"],
        "desc": "Document-oriented NoSQL database"
    },
    "redis": {
        "type": "database", "domain": "db",
        "connects": ["cache", "key_value", "in_memory", "pub_sub", "session", "queue"],
        "desc": "In-memory data structure store"
    },
    "neo4j": {
        "type": "database", "domain": "db",
        "connects": ["graph", "cypher", "node", "relationship", "traversal", "pattern_matching"],
        "desc": "Native graph database"
    },
    "elasticsearch": {
        "type": "database", "domain": "db",
        "connects": ["search", "full_text", "index", "lucene", "aggregation", "kibana"],
        "desc": "Distributed search and analytics engine"
    },
    "sql": {
        "type": "language", "domain": "db",
        "connects": ["postgresql", "mysql", "query", "select", "join", "where", "index"],
        "desc": "Structured Query Language for relational databases"
    },
    "index": {
        "type": "concept", "domain": "db",
        "connects": ["query", "performance", "btree", "hash", "postgresql", "elasticsearch"],
        "desc": "Data structure for fast lookups"
    },
    "transaction": {
        "type": "concept", "domain": "db",
        "connects": ["acid", "commit", "rollback", "isolation", "postgresql", "consistency"],
        "desc": "Atomic unit of database work"
    },
    "acid": {
        "type": "concept", "domain": "db",
        "connects": ["transaction", "atomicity", "consistency", "isolation", "durability", "postgresql"],
        "desc": "Database transaction properties"
    },
    "sharding": {
        "type": "technique", "domain": "db",
        "connects": ["horizontal_scaling", "partition", "distributed", "mongodb", "cassandra"],
        "desc": "Distributing data across multiple servers"
    },
    "replication": {
        "type": "technique", "domain": "db",
        "connects": ["high_availability", "replica", "primary", "secondary", "failover"],
        "desc": "Copying data to multiple nodes"
    },

    # DevOps & Infrastructure
    "docker": {
        "type": "tool", "domain": "devops",
        "connects": ["container", "image", "dockerfile", "kubernetes", "registry", "compose"],
        "desc": "Platform for containerizing applications"
    },
    "kubernetes": {
        "type": "tool", "domain": "devops",
        "connects": ["docker", "container", "pod", "deployment", "service", "helm", "orchestration"],
        "desc": "Container orchestration platform"
    },
    "container": {
        "type": "concept", "domain": "devops",
        "connects": ["docker", "image", "isolation", "process", "namespace", "cgroup"],
        "desc": "Isolated environment for running applications"
    },
    "pod": {
        "type": "concept", "domain": "devops",
        "connects": ["kubernetes", "container", "deployment", "service", "node"],
        "desc": "Smallest deployable unit in Kubernetes"
    },
    "helm": {
        "type": "tool", "domain": "devops",
        "connects": ["kubernetes", "chart", "deployment", "template", "release"],
        "desc": "Kubernetes package manager"
    },
    "terraform": {
        "type": "tool", "domain": "devops",
        "connects": ["infrastructure_as_code", "aws", "gcp", "azure", "state", "provider"],
        "desc": "Infrastructure as Code tool"
    },
    "ci_cd": {
        "type": "concept", "domain": "devops",
        "connects": ["pipeline", "jenkins", "github_actions", "deployment", "testing", "automation"],
        "desc": "Continuous Integration and Deployment"
    },
    "prometheus": {
        "type": "tool", "domain": "devops",
        "connects": ["monitoring", "metrics", "alerting", "grafana", "time_series", "scraping"],
        "desc": "Monitoring and alerting system"
    },
    "grafana": {
        "type": "tool", "domain": "devops",
        "connects": ["prometheus", "dashboard", "visualization", "metrics", "alerting"],
        "desc": "Visualization and dashboarding tool"
    },

    # Python & Web
    "fastapi": {
        "type": "framework", "domain": "python",
        "connects": ["python", "api", "async", "pydantic", "openapi", "uvicorn", "rest"],
        "desc": "Modern Python web framework"
    },
    "asyncio": {
        "type": "library", "domain": "python",
        "connects": ["python", "async", "await", "coroutine", "event_loop", "concurrency"],
        "desc": "Python asynchronous I/O framework"
    },
    "pydantic": {
        "type": "library", "domain": "python",
        "connects": ["fastapi", "validation", "schema", "model", "python", "type_hints"],
        "desc": "Data validation using Python type hints"
    },
    "pytest": {
        "type": "tool", "domain": "python",
        "connects": ["testing", "python", "fixture", "assertion", "coverage", "mock"],
        "desc": "Python testing framework"
    },

    # Security
    "jwt": {
        "type": "standard", "domain": "security",
        "connects": ["authentication", "token", "oauth", "api", "claims", "signature"],
        "desc": "JSON Web Token for stateless authentication"
    },
    "oauth": {
        "type": "protocol", "domain": "security",
        "connects": ["authentication", "authorization", "jwt", "token", "scope", "client"],
        "desc": "Authorization framework"
    },
    "encryption": {
        "type": "concept", "domain": "security",
        "connects": ["aes", "rsa", "tls", "key", "cipher", "decrypt"],
        "desc": "Converting data to unreadable format"
    },
    "tls": {
        "type": "protocol", "domain": "security",
        "connects": ["https", "certificate", "encryption", "handshake", "ssl"],
        "desc": "Transport Layer Security for encrypted communication"
    },

    # API & Networking
    "api": {
        "type": "concept", "domain": "core",
        "connects": ["rest", "graphql", "endpoint", "request", "response", "fastapi", "authentication"],
        "desc": "Application Programming Interface"
    },
    "rest": {
        "type": "architecture", "domain": "api",
        "connects": ["api", "http", "json", "endpoint", "crud", "stateless"],
        "desc": "Representational State Transfer architectural style"
    },
    "graphql": {
        "type": "language", "domain": "api",
        "connects": ["api", "query", "mutation", "schema", "resolver", "apollo"],
        "desc": "Query language for APIs"
    },
    "http": {
        "type": "protocol", "domain": "networking",
        "connects": ["rest", "api", "request", "response", "status_code", "header", "tls"],
        "desc": "HyperText Transfer Protocol"
    },
    "websocket": {
        "type": "protocol", "domain": "networking",
        "connects": ["real_time", "bidirectional", "connection", "http", "socket"],
        "desc": "Full-duplex communication protocol"
    },

    # Cloud
    "aws": {
        "type": "platform", "domain": "cloud",
        "connects": ["ec2", "s3", "lambda", "rds", "terraform", "cloud"],
        "desc": "Amazon Web Services cloud platform"
    },
    "ec2": {
        "type": "service", "domain": "cloud",
        "connects": ["aws", "virtual_machine", "instance", "ami", "auto_scaling"],
        "desc": "AWS Elastic Compute Cloud"
    },
    "s3": {
        "type": "service", "domain": "cloud",
        "connects": ["aws", "storage", "bucket", "object", "cdn"],
        "desc": "AWS Simple Storage Service"
    },
    "lambda": {
        "type": "service", "domain": "cloud",
        "connects": ["aws", "serverless", "function", "event", "trigger"],
        "desc": "AWS serverless compute service"
    },

    # Universal super nodes
    "data": {
        "type": "concept", "domain": "core",
        "connects": ["database", "storage", "processing", "pipeline", "analytics", "ml", "json", "api"],
        "desc": "Information processed by systems"
    },
    "server": {
        "type": "resource", "domain": "core",
        "connects": ["api", "database", "docker", "kubernetes", "http", "request", "response", "load_balancer"],
        "desc": "Machine that processes requests"
    },
    "config": {
        "type": "concept", "domain": "core",
        "connects": ["environment", "yaml", "json", "settings", "terraform", "kubernetes", "docker"],
        "desc": "System configuration and settings"
    },
    "error": {
        "type": "concept", "domain": "core",
        "connects": ["exception", "logging", "debugging", "stack_trace", "handling", "monitoring", "alerting"],
        "desc": "System failures and exceptions"
    },
}

# Memory templates with semantic meaning
MEMORIES = [
    # ML
    ("fact", "neural_network", "Neural networks learn by adjusting weights through backpropagation"),
    ("fact", "transformer", "Transformers use self-attention to process sequences in parallel"),
    ("fact", "bert", "BERT is pre-trained on masked language modeling and next sentence prediction"),
    ("fact", "gradient_descent", "Gradient descent updates weights in the opposite direction of the gradient"),
    ("procedure", "overfitting", "To prevent overfitting, use dropout, regularization, and early stopping"),
    ("procedure", "neural_network", "Train a neural network by feeding batches and computing gradients"),

    # Database
    ("fact", "postgresql", "PostgreSQL supports ACID transactions and complex queries"),
    ("fact", "index", "B-tree indexes speed up equality and range queries"),
    ("fact", "redis", "Redis stores data in memory for sub-millisecond response times"),
    ("procedure", "sharding", "Implement sharding by partitioning data on a shard key"),
    ("procedure", "transaction", "Use transactions to ensure data consistency across multiple operations"),

    # DevOps
    ("fact", "docker", "Docker containers share the host kernel but have isolated filesystems"),
    ("fact", "kubernetes", "Kubernetes manages container deployment, scaling, and networking"),
    ("procedure", "ci_cd", "Set up CI/CD by defining build, test, and deploy stages in a pipeline"),
    ("procedure", "prometheus", "Configure Prometheus scraping to collect metrics from targets"),

    # Security
    ("fact", "jwt", "JWTs contain header, payload, and signature sections"),
    ("fact", "tls", "TLS encrypts data in transit using symmetric encryption after handshake"),
    ("procedure", "oauth", "Implement OAuth by registering client, getting authorization, exchanging tokens"),

    # API
    ("fact", "rest", "REST APIs use HTTP methods to perform CRUD operations on resources"),
    ("fact", "graphql", "GraphQL allows clients to request exactly the data they need"),
    ("procedure", "api", "Design APIs by defining resources, endpoints, and request/response schemas"),
]


async def create_semantic_data():
    """Create semantically connected data."""
    db = Neo4jClient()
    await db.connect()

    print("Creating semantically connected knowledge graph...")

    concept_ids = {}

    # Create all concepts
    print("Creating concepts...")
    for name, info in KNOWLEDGE_GRAPH.items():
        concept_id = f"c_{name}_{uuid.uuid4().hex[:4]}"
        concept_ids[name] = concept_id

        # Position by domain
        domains = ["ml", "db", "devops", "python", "security", "api", "networking", "cloud", "core"]
        domain_idx = domains.index(info["domain"]) if info["domain"] in domains else 0
        angle = (domain_idx / len(domains)) * 6.28 + random.uniform(-0.3, 0.3)
        radius = random.uniform(20000, 40000) if info["domain"] != "core" else random.uniform(5000, 15000)
        x = radius * (0.5 + random.random()) * (1 if random.random() > 0.5 else -1)
        y = radius * (0.5 + random.random()) * (1 if random.random() > 0.5 else -1)
        cluster = domain_idx

        await db.execute_query(
            """
            CREATE (c:Concept {
                id: $id, name: $name, type: $type, description: $desc,
                domain: $domain, activation_count: $act,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=concept_id, name=name, type=info["type"], desc=info["desc"],
            domain=info["domain"], act=random.randint(10, 200),
            x=x, y=y, cluster=cluster
        )

    print(f"Created {len(concept_ids)} concepts")

    # Create semantic relationships
    print("Creating semantic relationships...")
    rel_count = 0
    for name, info in KNOWLEDGE_GRAPH.items():
        for target in info["connects"]:
            if target in concept_ids:
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                    CREATE (a)-[:RELATED_TO {weight: $w}]->(b)
                    """,
                    id1=concept_ids[name], id2=concept_ids[target],
                    w=random.uniform(0.6, 1.0)
                )
                rel_count += 1

    print(f"Created {rel_count} semantic relationships")

    # Boost core concepts (super nodes) with more connections
    print("Boosting super nodes...")
    super_nodes = ["api", "data", "server", "config", "error"]
    all_concept_ids = list(concept_ids.values())

    for super_name in super_nodes:
        super_id = concept_ids[super_name]
        # Connect to 80% of all concepts
        for cid in all_concept_ids:
            if cid != super_id and random.random() < 0.8:
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                    MERGE (a)-[:RELATED_TO {weight: $w}]->(b)
                    """,
                    id1=cid, id2=super_id, w=random.uniform(0.3, 0.7)
                )

    # Add more concepts to reach 1000+
    print("Adding more concepts...")
    extra_concepts = []
    for base_name, info in list(KNOWLEDGE_GRAPH.items())[:50]:
        for suffix in ["_v2", "_advanced", "_config", "_utils", "_core", "_client", "_server",
                       "_handler", "_manager", "_service", "_module", "_plugin", "_extension"]:
            new_name = f"{base_name}{suffix}"
            concept_id = f"c_{new_name}_{uuid.uuid4().hex[:4]}"
            extra_concepts.append((new_name, concept_id, info["domain"]))

            domain_idx = ["ml", "db", "devops", "python", "security", "api", "networking", "cloud", "core"].index(info["domain"]) if info["domain"] in ["ml", "db", "devops", "python", "security", "api", "networking", "cloud", "core"] else 0
            x = random.uniform(-50000, 50000)
            y = random.uniform(-50000, 50000)

            await db.execute_query(
                """
                CREATE (c:Concept {
                    id: $id, name: $name, type: 'variant', description: $desc,
                    domain: $domain, activation_count: $act,
                    layout_x: $x, layout_y: $y, cluster: $cluster
                })
                """,
                id=concept_id, name=new_name, desc=f"Extended {base_name} functionality",
                domain=info["domain"], act=random.randint(1, 50),
                x=x, y=y, cluster=domain_idx
            )

            # Connect to base concept
            await db.execute_query(
                """
                MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                CREATE (a)-[:RELATED_TO {weight: $w}]->(b)
                """,
                id1=concept_id, id2=concept_ids[base_name], w=random.uniform(0.7, 1.0)
            )

            # Connect to super nodes
            for super_name in super_nodes:
                if random.random() < 0.6:
                    await db.execute_query(
                        """
                        MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                        MERGE (a)-[:RELATED_TO {weight: $w}]->(b)
                        """,
                        id1=concept_id, id2=concept_ids[super_name], w=random.uniform(0.2, 0.5)
                    )

    print(f"Added {len(extra_concepts)} extra concepts")

    # Create semantic memories
    print("Creating memories...")
    memory_count = 0
    for mtype, concept, content in MEMORIES:
        if concept in concept_ids:
            memory_id = f"mem_{uuid.uuid4().hex[:8]}"
            x = random.uniform(-50000, 50000)
            y = random.uniform(-50000, 50000)

            await db.execute_query(
                """
                CREATE (m:SemanticMemory {
                    id: $id, content: $content, memory_type: $mtype,
                    importance: $importance, strength: $strength, access_count: $access,
                    layout_x: $x, layout_y: $y, cluster: $cluster
                })
                """,
                id=memory_id, content=content, mtype=mtype,
                importance=random.uniform(5, 10), strength=random.uniform(0.7, 1.0),
                access=random.randint(5, 50), x=x, y=y, cluster=random.randint(0, 8)
            )

            await db.execute_query(
                "MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                mid=memory_id, cid=concept_ids[concept]
            )
            memory_count += 1

    # Add more memories
    for _ in range(1000):
        concept_name = random.choice(list(KNOWLEDGE_GRAPH.keys()))
        info = KNOWLEDGE_GRAPH[concept_name]

        templates = [
            f"{concept_name} is essential for {info['domain']} applications",
            f"When using {concept_name}, consider {random.choice(info['connects']) if info['connects'] else 'performance'}",
            f"{concept_name} integrates well with {random.choice(info['connects']) if info['connects'] else 'other tools'}",
            f"Best practice: configure {concept_name} for production use",
            f"Debug {concept_name} issues by checking logs and metrics",
        ]

        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)

        await db.execute_query(
            """
            CREATE (m:SemanticMemory {
                id: $id, content: $content, memory_type: $mtype,
                importance: $importance, strength: $strength, access_count: $access,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=memory_id, content=random.choice(templates),
            mtype=random.choice(["fact", "procedure", "relationship"]),
            importance=random.uniform(3, 9), strength=random.uniform(0.5, 1.0),
            access=random.randint(1, 30), x=x, y=y, cluster=random.randint(0, 8)
        )

        if concept_name in concept_ids:
            await db.execute_query(
                "MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                mid=memory_id, cid=concept_ids[concept_name]
            )
        memory_count += 1

    print(f"Created {memory_count} memories")

    # Check super node connections
    print("\nSuper node connections:")
    for name in super_nodes:
        result = await db.execute_query(
            "MATCH (c:Concept {id: $id})-[r]-() RETURN count(r) as cnt",
            id=concept_ids[name]
        )
        print(f"  {name}: {result[0]['cnt']} connections")

    await db.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(create_semantic_data())
