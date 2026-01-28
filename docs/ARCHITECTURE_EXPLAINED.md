# Engram Architecture — Annotated Guide

A comprehensive guide for backend engineers new to RAG systems.
**This version includes inline explanations of concepts you may not know.**

---

## Table of Contents

1. [What is Engram?](#what-is-engram)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Data Model](#data-model)
5. [Retrieval Pipeline](#retrieval-pipeline)
6. [Further Reading](#further-reading)

---

## What is Engram?

> **engram** (noun): a hypothetical permanent change in the brain accounting for the existence of memory; a memory trace.

Engram is a **cognitive-inspired knowledge system** that transforms documentation into an intelligent assistant. It reasons like a brain — connecting concepts, learning from experience, and consolidating knowledge over time.

Unlike traditional RAG that retrieves document chunks, Engram uses a **brain-inspired dual memory architecture**:

- **Concept Network**: Atomic ideas connected by typed, weighted edges
- **Semantic Memory**: Facts and procedures linked to concepts
- **Episodic Memory**: Past reasoning traces with outcomes

<details>
<summary>📚 Deep Dive: What is RAG?</summary>

**RAG = Retrieval-Augmented Generation**

LLMs like GPT-4 or Claude have a knowledge cutoff and can hallucinate facts. RAG solves this by:

1. **Storing** your documents in a searchable format
2. **Retrieving** relevant pieces when a user asks a question  
3. **Augmenting** the LLM prompt with those pieces
4. **Generating** an answer grounded in real documents

**Simple RAG flow:**
```
User: "How do I deploy to staging?"
         │
         ▼
┌─────────────────────────┐
│ Search your docs for    │
│ "deploy staging"        │
└─────────────────────────┘
         │
         ▼
Found: "To deploy to staging, run `make deploy-staging`..."
         │
         ▼
┌─────────────────────────┐
│ LLM Prompt:             │
│ "Based on this context: │
│ [found doc]             │
│ Answer: How do I deploy │
│ to staging?"            │
└─────────────────────────┘
         │
         ▼
LLM: "Run `make deploy-staging` to deploy..."
```

**Why "chunks"?** Documents are too long to fit in the LLM context window, so we split them into smaller pieces (chunks) — typically 500-2000 characters each.

**Traditional RAG limitations:**
- Chunks are isolated — no relationships between them
- Can't answer "multi-hop" questions requiring info from multiple places
- Loses document structure and hierarchy

Engram addresses these by building a **knowledge graph** on top of chunks.

</details>

---

### Key Insight

**Graph + Hybrid Retrieval > Pure Vector Search**

By combining multiple retrieval methods (BM25, vector, graph traversal, path-based), Engram finds relevant information that any single method would miss.

<details>
<summary>📚 Deep Dive: Why Multiple Retrieval Methods?</summary>

Each retrieval method has strengths and blind spots:

| Method | Good At | Bad At |
|--------|---------|--------|
| **BM25** (keywords) | Exact matches, rare terms, specific names | Synonyms, paraphrasing |
| **Vector** (embeddings) | Semantic similarity, paraphrasing | Rare words, exact matches |
| **Graph** (spreading activation) | Multi-hop relationships, "connect the dots" | Direct simple questions |

**Example where BM25 wins:**
- Query: "JIRA-1234"
- Vector search might return tickets about "task tracking" (semantically similar)
- BM25 finds the exact ticket number

**Example where Vector wins:**
- Query: "How to free up disk space"
- Vector finds "Cleaning storage on servers" (same meaning, different words)
- BM25 misses it — no word overlap

**Example where Graph wins:**
- Query: "What tools help with X?"
- Graph traverses: Problem X → related concepts → tools that solve them
- Direct search might not find the connection

Engram combines all three, then merges results using RRF (explained below).

</details>

---

## Core Concepts

### Embeddings

**Text → Vector** (list of numbers representing meaning)

```
"Docker container" → [0.12, -0.34, 0.56, ..., 0.78]  (1024 dimensions)
```

Similar texts have similar vectors. We use **BGE-M3** (multilingual, 1024 dimensions).

<details>
<summary>📚 Deep Dive: What Are Embeddings?</summary>

**The problem:** Computers don't understand text. They understand numbers.

**The solution:** Convert text to a list of numbers (a "vector") that captures its *meaning*.

**How it works (simplified):**
1. A neural network reads your text
2. It outputs a fixed-size array of floats (e.g., 1024 numbers)
3. Texts with similar meanings produce similar arrays

**Visual intuition:**
```
"king"   → [0.8, 0.2, 0.9, ...]
"queen"  → [0.7, 0.3, 0.9, ...]  ← similar!
"banana" → [0.1, 0.9, 0.1, ...]  ← very different
```

**Why 1024 dimensions?** More dimensions = more nuance captured. But also more memory and compute. 1024 is a common sweet spot.

**Python example:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

# Get embeddings
vec1 = model.encode("Docker container")  # shape: (1024,)
vec2 = model.encode("контейнер Docker")  # Russian — same meaning!

# Compare similarity (explained next)
similarity = cosine_similarity(vec1, vec2)  # ~0.95 (very similar)
```

**Key models:**
- **BGE-M3**: Multilingual, good for Russian. 1024 dims.
- **OpenAI text-embedding-3-small**: 1536 dims. Requires API.
- **GigaEmbeddings**: Russian-optimized (Sber).

</details>

<details>
<summary>📚 Deep Dive: Cosine Similarity</summary>

**How do we compare two vectors?**

The most common method is **cosine similarity** — measuring the angle between vectors.

**Formula:**
```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
```

- `A · B` = dot product (element-wise multiply, then sum)
- `||A||` = magnitude (sqrt of sum of squares)

**Result range:** -1 to +1
- **1.0** = identical direction (same meaning)
- **0.0** = perpendicular (unrelated)  
- **-1.0** = opposite (rarely happens with text)

**Python:**
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Or use sklearn
from sklearn.metrics.pairwise import cosine_similarity
```

**Why cosine, not Euclidean distance?**

Cosine ignores vector magnitude (length), only considers direction. This matters because:
- Longer documents might have "stronger" embeddings
- We care about *what* the text means, not *how much* of it there is

**Typical thresholds:**
- `> 0.85`: Very similar (likely same topic)
- `0.6–0.85`: Related  
- `< 0.5`: Probably unrelated

</details>

---

### Vector Search

Find similar content by comparing vector distances:

```
Query: "container platform"  →  embedding  →  [0.11, -0.33, 0.55, ...]
                                                    │
                                                    ▼
                                            FAISS index search
                                                    │
                                                    ▼
                                            Top-K similar memories
```

**FAISS** is a fast vector index library from Meta.

<details>
<summary>📚 Deep Dive: How FAISS Works</summary>

**The naive approach:**

To find the 10 most similar documents to a query:
```python
# O(n) — compare every document
similarities = [cosine_sim(query, doc) for doc in all_docs]
top_10 = sorted(similarities)[-10:]
```

With 1 million documents, this is too slow (~seconds per query).

**FAISS solution: Approximate Nearest Neighbors (ANN)**

Instead of checking every vector, FAISS uses clever data structures to find *approximately* the best matches in milliseconds.

**Common FAISS index types:**

| Index | Speed | Accuracy | Memory | Use When |
|-------|-------|----------|--------|----------|
| `Flat` | Slow (exact) | 100% | High | < 100K docs |
| `IVF` | Fast | 95%+ | Medium | 100K–10M docs |
| `HNSW` | Very fast | 98%+ | High | Need speed |

**How IVF works (simplified):**
1. **Training**: Cluster all vectors into ~1000 groups
2. **Searching**: Only check clusters near the query vector
3. **Result**: Check 1% of data, find 95% of true matches

**Python example:**
```python
import faiss
import numpy as np

# Create index
dimension = 1024
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (similar to cosine)

# Add vectors
vectors = np.random.random((10000, dimension)).astype('float32')
faiss.normalize_L2(vectors)  # Normalize for cosine similarity
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
faiss.normalize_L2(query)
distances, indices = index.search(query, k=10)  # Top 10
```

**Engram uses:** `IndexIVFFlat` with `nprobe=10` for speed/accuracy balance.

</details>

---

### BM25

Traditional keyword search (like Elasticsearch's default algorithm):

- Matches exact terms
- Considers term frequency and document length
- Works well for specific terminology

We enhance BM25 with **Russian lemmatization** (PyMorphy3):
```
"контейнеры" → "контейнер" (normalized form)
```

<details>
<summary>📚 Deep Dive: BM25 Algorithm</summary>

**BM25** = "Best Match 25" — a ranking function from the 1990s that's still state-of-the-art for keyword search.

**Core idea:** Score documents by how well they match query terms, adjusted for:
1. **Term frequency (TF)**: More occurrences = higher score (with diminishing returns)
2. **Inverse document frequency (IDF)**: Rare terms are more important
3. **Document length**: Don't penalize long documents unfairly

**Simplified formula:**
```
score(D, Q) = Σ IDF(qi) × (TF(qi, D) × (k1 + 1)) / (TF(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

Where:
- `k1` ≈ 1.5 (term saturation — diminishing returns)
- `b` ≈ 0.75 (document length normalization)
- `avgdl` = average document length

**Python example (using rank_bm25):**
```python
from rank_bm25 import BM25Okapi

documents = [
    "Docker uses containers for isolation",
    "Kubernetes orchestrates container deployments",
    "Python is a programming language",
]

# Tokenize
tokenized = [doc.lower().split() for doc in documents]

# Create index
bm25 = BM25Okapi(tokenized)

# Search
query = "container deployment"
scores = bm25.get_scores(query.lower().split())
# [0.43, 0.87, 0.0]  ← doc 2 wins
```

**Why lemmatization matters for Russian:**

Russian is morphologically rich — one word has many forms:
- "контейнер" (nominative)
- "контейнера" (genitive)
- "контейнеру" (dative)
- "контейнеры" (plural)
- etc.

Without lemmatization, "контейнеры" won't match "контейнер". PyMorphy3 normalizes all forms to the base word.

```python
import pymorphy3
morph = pymorphy3.MorphAnalyzer()

morph.parse("контейнеры")[0].normal_form  # → "контейнер"
```

</details>

---

### Knowledge Graph

Concepts connected by relationships:

```
     ┌─────────┐                    ┌─────────┐
     │ Docker  │───RELATES_TO──────►│ container│
     └─────────┘                    └─────────┘
          │                              │
          │                              │
    LINKED_TO                      LINKED_TO
          │                              │
          ▼                              ▼
    ┌─────────────────────────────────────────┐
    │ Memory: "Docker uses containers to      │
    │ isolate applications"                   │
    └─────────────────────────────────────────┘
```

<details>
<summary>📚 Deep Dive: Graph Databases and Neo4j</summary>

**What is a graph database?**

Unlike relational databases (tables with rows), graph databases store:
- **Nodes**: Entities (like a row)
- **Edges**: Relationships between entities
- **Properties**: Data on nodes/edges

**Why graphs for knowledge?**

| Task | SQL | Graph |
|------|-----|-------|
| Find direct relations | Easy JOIN | Easy |
| Find 3-hop connections | 3 nested JOINs | Same query |
| Find *any* path | Recursive CTE (complex) | Built-in |
| Add new relationship type | ALTER TABLE | Just add edge |

**Neo4j basics:**

```cypher
-- Create nodes
CREATE (d:Concept {name: "Docker", type: "tool"})
CREATE (c:Concept {name: "container", type: "resource"})

-- Create relationship
MATCH (d:Concept {name: "Docker"}), (c:Concept {name: "container"})
CREATE (d)-[:RELATES_TO {weight: 0.9}]->(c)

-- Query: Find all concepts related to Docker
MATCH (d:Concept {name: "Docker"})-[:RELATES_TO]->(related)
RETURN related.name

-- Query: Find paths between two concepts
MATCH path = shortestPath(
  (a:Concept {name: "Docker"})-[*..5]-(b:Concept {name: "Kubernetes"})
)
RETURN path
```

**Python driver:**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("""
        MATCH (c:Concept)-[:LINKED_TO]->(m:SemanticMemory)
        WHERE c.name = $concept
        RETURN m.content
    """, concept="Docker")
    
    for record in result:
        print(record["m.content"])
```

**Engram's graph structure:**
- `Concept` nodes: Atomic ideas ("Docker", "контейнер", "deploy")
- `SemanticMemory` nodes: Facts and procedures
- `RELATES_TO` edges: Concept ↔ Concept (weighted by co-occurrence)
- `LINKED_TO` edges: Memory ↔ Concept

</details>

---

### Spreading Activation

<details>
<summary>📚 Deep Dive: Spreading Activation Algorithm</summary>

**Inspiration:** How your brain finds related memories.

When you think "coffee", related concepts automatically activate:
- "morning" (you drink it then)
- "caffeine" (ingredient)
- "mug" (container)
- "tired" (why you drink it)

**Algorithm:**

1. **Seed**: Start with query concepts (activation = 1.0)
2. **Spread**: Transfer activation to neighbors through edges
3. **Decay**: Reduce activation at each hop (× 0.8)
4. **Threshold**: Stop spreading when activation < 0.3
5. **Collect**: Return all activated concepts

**Example:**
```
Query: "How to free disk space?"
Extracted concepts: ["disk", "space", "free"]

Hop 0: disk=1.0, space=1.0, free=1.0

Hop 1 (decay=0.8):
  disk → storage (0.8), partition (0.72), cleanup (0.64)
  space → memory (0.8), quota (0.72)
  free → delete (0.8), prune (0.64)

Hop 2 (decay=0.64):
  storage → Docker volumes (0.51)
  cleanup → docker system prune (0.41)
  prune → unused images (0.35)

Final activated concepts: disk, space, free, storage, cleanup, 
                          Docker volumes, docker system prune...
```

**Python implementation (simplified):**
```python
async def spread_activation(
    seed_concepts: list[str],
    graph,
    decay: float = 0.8,
    threshold: float = 0.3,
    max_hops: int = 3
) -> dict[str, float]:
    """Return {concept_id: activation_score}"""
    
    activation = {cid: 1.0 for cid in seed_concepts}
    
    for hop in range(max_hops):
        new_activation = {}
        
        for concept_id, current_act in activation.items():
            if current_act < threshold:
                continue
            
            # Get neighbors from graph
            neighbors = graph.get_neighbors(concept_id)
            
            for neighbor in neighbors:
                # Transfer activation through edge
                transfer = current_act * neighbor.edge_weight * decay
                
                # Accumulate (if multiple paths lead here)
                new_activation[neighbor.id] = (
                    new_activation.get(neighbor.id, 0) + transfer
                )
        
        # Merge new activations
        for cid, act in new_activation.items():
            activation[cid] = max(activation.get(cid, 0), act)
    
    return activation
```

**Key parameters (from cognitive science research):**

| Parameter | Value | Why |
|-----------|-------|-----|
| Decay | 0.7–0.9 | Prevents runaway activation |
| Threshold | 0.3–0.5 | Stops noise propagation |
| Max hops | 2–4 | Deeper rarely helps |

</details>

---

### RRF Fusion (Reciprocal Rank Fusion)

<details>
<summary>📚 Deep Dive: How RRF Merges Multiple Rankings</summary>

**The problem:** We have 4 different rankings (BM25, vector, graph, path). How do we combine them?

**Naive approach:** Average the scores?
- ❌ Scores aren't comparable (BM25: 0-20, cosine: 0-1, activation: 0-1)
- ❌ Different distributions

**RRF solution:** Use *ranks*, not scores.

**Formula:**
```
RRF_score(doc) = Σ (1 / (k + rank_i(doc)))
```

Where:
- `k` = smoothing constant (usually 60)
- `rank_i(doc)` = position in ranker i's list (1 = first)

**Example:**
```
Document "docker-cleanup.md" appears at:
- BM25: rank 3
- Vector: rank 12  
- Graph: rank 1
- Path: rank 8

RRF = 1/(60+3) + 1/(60+12) + 1/(60+1) + 1/(60+8)
    = 0.0159 + 0.0139 + 0.0164 + 0.0147
    = 0.0609
```

**Why RRF works:**
- Rewards documents that appear in *multiple* lists (even if not #1)
- Robust to different score scales
- Simple and effective

**Python implementation:**
```python
def reciprocal_rank_fusion(
    rankings: list[list[dict]],  # List of ranked results from each method
    k: int = 60,
    weights: list[float] = None  # Optional per-method weights
) -> list[dict]:
    """
    Merge multiple rankings using RRF.
    
    Args:
        rankings: [[{id, score}, ...], ...]  # One list per method
        k: Smoothing constant
        weights: [0.25, 0.25, 0.25, 0.25]  # Weight per method
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    
    scores = {}  # doc_id → RRF score
    
    for method_idx, ranked_list in enumerate(rankings):
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item["id"]
            rrf_contribution = weights[method_idx] / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_contribution
    
    # Sort by RRF score descending
    sorted_results = sorted(scores.items(), key=lambda x: -x[1])
    return [{"id": doc_id, "rrf_score": score} for doc_id, score in sorted_results]
```

**Engram's weights:**
```bash
RRF_BM25_WEIGHT=0.25
RRF_VECTOR_WEIGHT=0.25
RRF_GRAPH_WEIGHT=0.20
RRF_PATH_WEIGHT=0.30  # Path gets slightly more — good for multi-hop
```

</details>

---

### Reranker (Cross-Encoder)

<details>
<summary>📚 Deep Dive: What is a Reranker?</summary>

**The problem:** Initial retrieval is fast but imprecise. We retrieve 200 candidates but only need the top 10.

**Solution:** Use a more powerful (slower) model to re-score the top candidates.

**Bi-encoder vs Cross-encoder:**

| | Bi-encoder (embeddings) | Cross-encoder (reranker) |
|---|---|---|
| Input | Query and doc separately | Query + doc together |
| Output | Two vectors, compare with cosine | Single relevance score |
| Speed | Fast (pre-compute doc embeddings) | Slow (run for each pair) |
| Accuracy | Good | Better |
| Use case | Initial retrieval (millions of docs) | Reranking (top 100) |

**How cross-encoder works:**
```
Input: "[CLS] How to free disk space? [SEP] Docker uses disk for images and containers. Run `docker system prune` to clean up. [SEP]"
         │
         ▼
   BERT-like model
         │
         ▼
   Relevance score: 0.94
```

The model sees query AND document together, so it can do fine-grained matching.

**Python example:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

query = "How to free disk space?"
candidates = [
    "Docker uses disk for images. Run docker system prune.",
    "Python is a programming language.",
    "Disk partitions can be resized with fdisk.",
]

# Score each pair
pairs = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)
# [0.94, 0.02, 0.67]

# Rerank
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
```

**Engram's reranking flow:**
```
BM25 (200) + Vector (200) + Graph (200) + Path (200)
                    │
                    ▼
            RRF Fusion (merge)
                    │
                    ▼
            Top 64 candidates
                    │
                    ▼
        Cross-encoder reranking
                    │
                    ▼
            Top 10 to LLM
```

**Why BGE-reranker-v2-m3?**
- Multilingual (works with Russian)
- Fast enough for real-time (64 candidates in ~100ms on GPU)
- Strong accuracy

</details>

---

## Data Model

### Node Types

#### Concept

Atomic ideas/entities in the knowledge graph.

```python
Concept:
    id: str              # "concept-docker-abc123"
    name: str            # "docker" (normalized, lowercase)
    type: str            # "tool" | "resource" | "action" | "state" | "config" | "error" | "general"
    description: str     # LLM-generated definition (optional)
    aliases: list[str]   # ["докер", "Docker"] - alternative names
    embedding: list[float]       # BGE-M3 embedding (1024-dim)
    labse_embedding: list[float] # LaBSE embedding (for deduplication)
```

<details>
<summary>📚 Deep Dive: Why Two Embeddings?</summary>

**BGE-M3 embedding:** Used for semantic search. When you query, we compare your query embedding to concept embeddings.

**LaBSE embedding:** Used for *deduplication*. 

**Problem:** During ingestion, we might extract:
- "Docker" from doc A
- "докер" from doc B  
- "docker" from doc C

These are the same concept! We need to merge them.

**Solution:** LaBSE (Language-agnostic BERT Sentence Embeddings) is specifically trained for cross-lingual similarity. "Docker" and "докер" have nearly identical LaBSE embeddings.

```python
# Deduplication logic
if cosine_sim(new_concept.labse_embedding, existing.labse_embedding) > 0.92:
    # Same concept — merge aliases
    existing.aliases.append(new_concept.name)
else:
    # New concept — create node
    create_concept(new_concept)
```

</details>

#### Memory (SemanticMemory)

Units of knowledge: facts, procedures, relationships.

```python
SemanticMemory:
    id: str              # "mem-abc123"
    content: str         # "Docker использует контейнеры для изоляции приложений"
    search_content: str  # Optimized text for vector search (summary + keywords)
    memory_type: str     # "fact" | "procedure" | "relationship"
    concept_ids: list[str]    # Links to concepts
    source_url: str      # Original document URL
    source_title: str    # Document title
    confidence: float    # 0.0-1.0
    status: str          # "active" | "deprioritized" | "archived"
```

<details>
<summary>📚 Deep Dive: Memory Types</summary>

**Fact:** A declarative statement.
```
"Docker images are layered, with each layer representing a filesystem change."
```

**Procedure:** Step-by-step instructions.
```
"To remove unused Docker images: 1) Run `docker images` 2) Run `docker rmi <image_id>`"
```

**Relationship:** How concepts connect.
```
"Docker Compose uses YAML files to define multi-container applications."
```

**Why distinguish them?**

Different types get different treatment during retrieval and answer generation:
- **Facts** → Good for "what is X?" questions
- **Procedures** → Good for "how do I X?" questions  
- **Relationships** → Good for "what connects X and Y?" questions

</details>

---

## Retrieval Pipeline

### Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL (runtime)                          │
│                                                                     │
│   User Query: "Как освободить место на диске?"                      │
│        │                                                            │
│        ▼                                                            │
│   1. Concept Extraction (LLM) ───► ["диск", "место", "освободить"]  │
│        │                                                            │
│        ▼                                                            │
│   2. Spreading Activation ───► +["storage", "cleanup", "prune"...]  │
│        │                                                            │
│        ├──────────────┬──────────────┬──────────────┐               │
│        ▼              ▼              ▼              ▼               │
│   3. BM25         Vector        Graph          Path                 │
│    (keywords)   (embeddings)  (activated)   (bridging)              │
│     200 results   200 results   200 results   200 results           │
│        │              │              │              │               │
│        └──────────────┴──────────────┴──────────────┘               │
│                           │                                         │
│                           ▼                                         │
│   4. RRF Fusion (merge by rank)                                     │
│                           │                                         │
│                           ▼                                         │
│   5. Reranker (cross-encoder, top 64 → top 10)                      │
│                           │                                         │
│                           ▼                                         │
│   6. Top-K Memories ───► LLM ───► Response                          │
└─────────────────────────────────────────────────────────────────────┘
```

<details>
<summary>📚 Deep Dive: Step-by-Step Explanation</summary>

**Step 1: Concept Extraction**

LLM extracts key concepts from the query:
```
Query: "Как освободить место на диске?"
→ Concepts: ["диск", "место", "освободить"]
```

This uses a prompt like:
```
Extract the key concepts from this query. 
Return only the main nouns, verbs, and entities.
Query: {query}
```

**Step 2: Spreading Activation**

Start from extracted concepts, spread through the graph:
```
диск (1.0) → storage (0.8) → Docker volumes (0.64)
место (1.0) → quota (0.8)  
освободить (1.0) → delete (0.8) → prune (0.64) → docker system prune (0.51)
```

Now we have a broader set of activated concepts to search with.

**Step 3: Four Parallel Searches**

Each returns ~200 candidates:

1. **BM25**: Keyword match against memory content (lemmatized)
2. **Vector**: Cosine similarity between query embedding and memory embeddings
3. **Graph**: Memories linked to activated concepts
4. **Path**: Memories on shortest paths between query concepts in the graph

**Step 4: RRF Fusion**

Merge all results by rank:
```python
# Document appears at:
# BM25: rank 5, Vector: rank 3, Graph: rank 1, Path: rank 10
rrf_score = 1/(60+5) + 1/(60+3) + 1/(60+1) + 1/(60+10) = 0.062
```

**Step 5: Reranking**

Take top 64, run through cross-encoder for precise scoring:
```python
pairs = [(query, memory.content) for memory in top_64]
scores = reranker.predict(pairs)
top_10 = sorted(zip(top_64, scores), key=lambda x: -x[1])[:10]
```

**Step 6: Response Generation**

Send top 10 memories to LLM as context:
```
Based on the following information:

[Memory 1] Docker images take up disk space. Run `docker system prune` to clean up...
[Memory 2] Check disk usage with `df -h`...
...

Answer this question: Как освободить место на диске?
```

</details>

---

## Path-Based Retrieval

<details>
<summary>📚 Deep Dive: What is Path Retrieval?</summary>

**The problem:** Multi-hop questions require connecting concepts that aren't directly linked.

**Question:** "What tools can help with deployment issues in Kubernetes?"

The answer might involve:
```
Kubernetes → deployment → rollout → kubectl rollout status
```

Neither "Kubernetes" nor "tools" directly connects to the answer, but there's a *path*.

**How path retrieval works:**

1. Extract query concepts: ["Kubernetes", "deployment", "tools"]
2. Find shortest paths between all pairs in the graph
3. Collect memories along those paths
4. These "bridge" memories often contain the answer

**Cypher query:**
```cypher
MATCH path = shortestPath(
  (c1:Concept)-[*..4]-(c2:Concept)
)
WHERE c1.name IN $query_concepts 
  AND c2.name IN $query_concepts
  AND c1 <> c2
WITH path, nodes(path) AS path_nodes
UNWIND path_nodes AS node
MATCH (node)-[:LINKED_TO]->(m:SemanticMemory)
RETURN DISTINCT m
```

**Why it helps:**

| Retrieval Method | Finds |
|------------------|-------|
| BM25 | Docs with "Kubernetes" or "deployment" |
| Vector | Semantically similar docs |
| Graph | Docs linked to activated concepts |
| **Path** | Docs that *bridge* between concepts |

Path retrieval often finds the "glue" information that other methods miss.

</details>

---

## Further Reading

### Embeddings & Vector Search
- [Sentence Transformers Documentation](https://www.sbert.net/) — The library Engram uses
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) — Deep dive into vector indexing
- [What are Embeddings?](https://vickiboykis.com/what_are_embeddings/) — Excellent visual explainer
- [Understanding ANN Algorithms](https://www.pinecone.io/learn/series/faiss/vector-indexes/) — Pinecone's FAISS guide

### BM25 & Information Retrieval
- [BM25: The Next Generation](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — Original paper (accessible)
- [Lucene BM25 Explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — Elasticsearch deep dive
- [PyMorphy3 Documentation](https://pymorphy2.readthedocs.io/en/stable/) — Russian morphological analysis

### Knowledge Graphs
- [Neo4j GraphAcademy](https://graphacademy.neo4j.com/) — Free courses
- [Knowledge Graphs: Fundamentals, Techniques, and Applications](https://kgbook.org/) — Free online textbook
- [Building Knowledge Graphs with LLMs](https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/) — Neo4j blog

### Spreading Activation
- [Spreading Activation (Wikipedia)](https://en.wikipedia.org/wiki/Spreading_activation) — Overview
- [Collins & Loftus 1975](https://psycnet.apa.org/record/1975-22282-001) — Original cognitive science paper
- [HippoRAG Paper](https://arxiv.org/abs/2405.14831) — Modern RAG using hippocampal indexing theory

### RAG Systems
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) — Original RAG paper
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag-techniques/) — Pinecone guide
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — Practical implementation

### Reranking
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html) — Sentence Transformers docs
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) — Model card with usage examples
- [When to Use Rerankers](https://www.pinecone.io/learn/rerankers/) — Practical guide

### Fusion Methods
- [RRF Paper](https://dl.acm.org/doi/10.1145/1571941.1572114) — Original Reciprocal Rank Fusion paper
- [Hybrid Search Best Practices](https://weaviate.io/blog/hybrid-search-explained) — Weaviate guide

### Russian NLP
- [DeepPavlov](http://docs.deeppavlov.ai/) — Russian NLP library
- [Russian Transformers](https://huggingface.co/ai-forever) — Sber's Russian models
- [Natasha](https://github.com/natasha/natasha) — Rule-based Russian NLP
