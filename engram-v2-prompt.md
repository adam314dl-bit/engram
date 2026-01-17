# Engram v2 — Implementation Prompt

You are implementing **Engram**, a cognitive-inspired knowledge system with dual memory architecture. It transforms documentation into an intelligent assistant that reasons like a brain — connecting concepts, learning from experience, and consolidating knowledge over time.

> **engram** (noun): a hypothetical permanent change in the brain accounting for the existence of memory; a memory trace.

---

## Core Architecture: Concept-Centric Dual Memory

Unlike traditional RAG that retrieves document chunks, Engram uses a brain-inspired architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONCEPT NETWORK                            │
│  Atomic ideas connected by typed, weighted edges                │
│                                                                 │
│     Docker ──uses──► container ──stores──► image                │
│        │                 │                   │                  │
│        └──────needs──────┴───────────────────┘                  │
│                          │                                      │
│                     disk_space                                  │
└─────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  SEMANTIC   │  │  SEMANTIC   │  │  EPISODIC   │
   │  MEMORY     │  │  MEMORY     │  │  MEMORY     │
   │             │  │             │  │             │
   │ "Docker is  │  │ "prune      │  │ query: ...  │
   │ a container │  │ removes     │  │ behavior:   │
   │ platform"   │  │ unused      │  │  "check_    │
   │             │  │ resources"  │  │   disk_     │
   │ concepts:   │  │             │  │   usage"    │
   │ [Docker]    │  │ concepts:   │  │ outcome: +1 │
   │             │  │ [Docker,    │  │             │
   │ type: fact  │  │  prune,     │  │             │
   │             │  │  disk]      │  │             │
   └─────────────┘  └─────────────┘  └─────────────┘
                           │
                           │ crystallize on success
                           ▼
                    ┌─────────────┐
                    │ NEW SEMANTIC│
                    │ MEMORY      │
                    └─────────────┘
```

### The Three Node Types

| Node Type | Purpose | Example |
|-----------|---------|---------|
| **Concept** | Atomic idea/entity | "Docker", "disk space", "prune" |
| **SemanticMemory** | Fact or procedure linking concepts | "prune removes unused Docker resources" |
| **EpisodicMemory** | Past reasoning trace with outcome | Query + behavior + answer + feedback |

### Why This Works

- **Facts** ("What is Docker?") → activate Docker concept → retrieve connected semantic memories
- **Procedures** ("Docker full, what do?") → activate Docker + disk → find memories at intersection
- **Learning** → successful episodes crystallize into semantic memories
- **Re-reasoning** → failed answers use past episodes as templates for new approaches

---

## Technical Stack

| Component | Choice |
|-----------|--------|
| Python | 3.11+ |
| Package manager | uv |
| Graph DB | Neo4j 5.15+ (Docker) |
| API | FastAPI |
| Testing | pytest + pytest-asyncio |
| Type checking | mypy (strict) |
| Config | Pydantic Settings |

**Hardware:** Mac Studio M3 Ultra, 512GB RAM  
**LLM:** OpenAI-compatible endpoint (configurable)  
**Embeddings:** `all-MiniLM-L6-v2` for dev, `ai-sage/Giga-Embeddings-instruct` for production

---

## Project Structure

```
engram/
├── pyproject.toml
├── README.md
├── .env.example
├── docker-compose.yml
├── Dockerfile
│
├── src/
│   └── engram/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── models/
│       │   ├── concept.py          # Concept node
│       │   ├── semantic_memory.py  # Facts, procedures
│       │   ├── episodic_memory.py  # Reasoning traces
│       │   └── document.py         # Source documents
│       │
│       ├── ingestion/
│       │   ├── parser.py           # Document parser
│       │   ├── concept_extractor.py
│       │   ├── memory_extractor.py
│       │   └── prompts.py
│       │
│       ├── storage/
│       │   ├── neo4j_client.py
│       │   └── schema.py           # Graph schema setup
│       │
│       ├── retrieval/
│       │   ├── embeddings.py
│       │   ├── spreading_activation.py
│       │   ├── hybrid_search.py    # BM25 + vector + rerank
│       │   └── pipeline.py
│       │
│       ├── reasoning/
│       │   ├── synthesizer.py      # LLM response generation
│       │   ├── behavior_extractor.py  # Metacognitive reuse
│       │   └── re_reasoner.py      # Alternative path finding
│       │
│       ├── learning/
│       │   ├── feedback_handler.py
│       │   ├── consolidation.py    # Episode → semantic
│       │   ├── memory_strength.py  # SM-2 algorithm
│       │   └── reflection.py       # Threshold-triggered
│       │
│       ├── gaps/
│       │   └── detector.py
│       │
│       └── api/
│           ├── main.py
│           └── routes.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── mock_docs/
│   ├── unit/
│   └── integration/
│
└── scripts/
    ├── seed_test_data.py
    └── run_ingestion.py
```

---

## Data Models

### Concept

```python
@dataclass
class Concept:
    id: str
    name: str                           # "Docker", "disk space"
    type: str                           # "tool", "resource", "action", "state"
    description: str | None             # Brief definition if available
    embedding: list[float] | None
    
    # Hierarchy (optional)
    parent_id: str | None               # For is-a relationships
    level: int                          # 0=root, 1=domain, 2=concept, 3=instance
    
    # Stats
    activation_count: int = 0           # How often retrieved
    last_activated: datetime | None
```

### SemanticMemory

```python
@dataclass
class SemanticMemory:
    id: str
    content: str                        # The knowledge itself
    
    # Structure (optional, for structured facts)
    subject: str | None                 # "Docker"
    predicate: str | None               # "uses"
    object: str | None                  # "containers"
    
    # Links
    concept_ids: list[str]              # Connected concepts
    source_doc_ids: list[str]           # Provenance
    source_episode_ids: list[str]       # If crystallized from episodes
    
    # Type
    memory_type: Literal["fact", "procedure", "relationship"]
    
    # Confidence & strength
    importance: float                   # 1-10 scale, LLM-assigned
    confidence: float                   # 0-1
    strength: float                     # SM-2 easiness factor (1.3-2.5)
    
    # Temporal
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    # Validity
    valid_from: datetime | None
    valid_until: datetime | None
    status: Literal["active", "superseded", "invalid"] = "active"
    
    embedding: list[float] | None
```

### EpisodicMemory

```python
@dataclass
class EpisodicMemory:
    id: str
    
    # The episode
    query: str                          # Original user query
    concepts_activated: list[str]       # Concept IDs that fired
    memories_used: list[str]            # SemanticMemory IDs used
    
    # Reasoning (metacognitive reuse format)
    behavior_name: str                  # "check_disk_usage", "prune_resources"
    behavior_instruction: str           # One-line reusable pattern
    domain: str                         # "docker", "kubernetes", etc.
    
    # Outcome
    answer_summary: str                 # Brief summary of response
    feedback: Literal["positive", "negative", "correction"] | None
    correction_text: str | None         # If user provided correction
    
    # Stats
    importance: float                   # 1-10
    repetition_count: int = 1           # Times similar query succeeded
    success_count: int = 0
    failure_count: int = 0
    
    # Temporal
    created_at: datetime
    last_used: datetime
    
    # Consolidation tracking
    consolidated: bool = False          # Has become semantic memory?
    consolidated_memory_id: str | None
    
    embedding: list[float] | None       # On behavior_instruction
```

---

## Neo4j Schema

```cypher
// Constraints
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT semantic_id IF NOT EXISTS FOR (s:SemanticMemory) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT episodic_id IF NOT EXISTS FOR (e:EpisodicMemory) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Vector indexes
CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX semantic_embeddings IF NOT EXISTS
FOR (s:SemanticMemory) ON (s.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX episodic_embeddings IF NOT EXISTS
FOR (e:EpisodicMemory) ON (e.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

// Full-text index for BM25
CREATE FULLTEXT INDEX semantic_content IF NOT EXISTS
FOR (s:SemanticMemory) ON EACH [s.content];

// Concept relationships (weighted, typed)
// (c1:Concept)-[:RELATED_TO {weight: 0.8, type: "uses"}]->(c2:Concept)
// (c:Concept)-[:IS_A]->(parent:Concept)

// Memory relationships
// (s:SemanticMemory)-[:ABOUT]->(c:Concept)
// (s:SemanticMemory)-[:EXTRACTED_FROM]->(d:Document)
// (e:EpisodicMemory)-[:ACTIVATED]->(c:Concept)
// (e:EpisodicMemory)-[:USED]->(s:SemanticMemory)
// (e:EpisodicMemory)-[:CRYSTALLIZED_TO]->(s:SemanticMemory)
```

---

## Spreading Activation Algorithm

Based on research: decay 0.85, max 3 hops, query-oriented edge filtering.

```python
async def spread_activation(
    seed_concepts: list[str],
    query_embedding: list[float],
    decay: float = 0.85,
    threshold: float = 0.3,           # Firing threshold
    max_hops: int = 3,
    rescale: float = 0.4              # Prevents hub explosion
) -> dict[str, float]:
    """
    Spread activation through concept network.
    
    1. Initialize seeds with activation = 1.0
    2. For each hop:
       a. Get neighbors of active concepts
       b. Filter edges by query relevance (cosine sim > 0.3)
       c. Transfer: neighbor += current × edge_weight × decay × rescale
       d. Apply lateral inhibition (top-k per hop)
    3. Return concept_id -> activation mapping
    """
    activation = {cid: 1.0 for cid in seed_concepts}
    
    for hop in range(max_hops):
        next_activation = {}
        
        for concept_id, current_act in activation.items():
            if current_act < threshold:
                continue
            
            # Get neighbors with edge info
            neighbors = await get_concept_neighbors(concept_id)
            
            for neighbor in neighbors:
                # Query-oriented filtering
                edge_relevance = cosine_sim(query_embedding, neighbor.edge_embedding)
                if edge_relevance < 0.3:
                    continue
                
                # Transfer activation
                transfer = current_act * neighbor.edge_weight * edge_relevance * decay * rescale
                next_activation[neighbor.id] = next_activation.get(neighbor.id, 0) + transfer
        
        # Lateral inhibition: keep top 20 per hop
        sorted_acts = sorted(next_activation.items(), key=lambda x: x[1], reverse=True)
        next_activation = dict(sorted_acts[:20])
        
        # Merge with existing (don't lose seeds)
        for cid, act in next_activation.items():
            activation[cid] = max(activation.get(cid, 0), act)
    
    return activation
```

---

## Retrieval Pipeline

Three-stage hybrid retrieval:

```python
async def retrieve(query: str) -> RetrievalResult:
    """
    1. Extract concepts from query
    2. Spread activation through concept network
    3. Retrieve memories at activated concepts
    4. Hybrid search (graph + vector + BM25)
    5. Rerank by recency + importance + relevance
    6. Check for similar episodes (reasoning templates)
    """
    
    # 1. Concept extraction
    query_concepts = await extract_concepts(query)
    query_embedding = await embed(query)
    
    # 2. Spreading activation
    activated = await spread_activation(
        seed_concepts=[c.id for c in query_concepts],
        query_embedding=query_embedding
    )
    
    # 3. Get memories connected to activated concepts
    semantic_memories = await get_memories_for_concepts(
        concept_ids=list(activated.keys()),
        min_activation=0.3
    )
    
    # 4. Hybrid search for additional memories
    vector_results = await vector_search(query_embedding, k=20)
    bm25_results = await bm25_search(query, k=20)
    
    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([
        semantic_memories,
        vector_results,
        bm25_results
    ])
    
    # 5. Rerank using Generative Agents formula
    scored = []
    for memory in fused[:50]:
        recency = 0.995 ** hours_since_access(memory)
        importance = memory.importance / 10
        relevance = cosine_sim(query_embedding, memory.embedding)
        
        score = recency + importance + relevance
        scored.append((memory, score))
    
    ranked_memories = sorted(scored, key=lambda x: x[1], reverse=True)[:10]
    
    # 6. Find similar episodes (for reasoning templates)
    similar_episodes = await find_similar_episodes(query_embedding, k=3)
    
    return RetrievalResult(
        concepts=query_concepts,
        activated_concepts=activated,
        memories=[m for m, _ in ranked_memories],
        episodes=similar_episodes
    )
```

---

## Reasoning & Response Generation

### Response Synthesis

```python
async def synthesize_response(
    query: str,
    retrieval: RetrievalResult
) -> SynthesisResult:
    """Generate response using retrieved context."""
    
    # Format context for LLM
    context = format_context(
        memories=retrieval.memories,
        episodes=retrieval.episodes  # Past reasoning as templates
    )
    
    prompt = f"""
    Вопрос пользователя: {query}
    
    Релевантные знания:
    {context}
    
    Прошлые похожие рассуждения (используй как шаблон если применимо):
    {format_episodes(retrieval.episodes)}
    
    Дай полезный ответ. Если это вопрос "что такое X" — объясни концепцию.
    Если это проблема — предложи решение пошагово.
    Упомяни если уверенность в информации низкая.
    
    Также кратко опиши свою стратегию рассуждения одним предложением
    в формате: СТРАТЕГИЯ: [название_поведения] — [краткое описание]
    """
    
    response = await llm.generate(prompt)
    
    # Extract behavior for episodic memory
    behavior = extract_behavior(response)
    
    return SynthesisResult(
        answer=response.answer,
        behavior_name=behavior.name,
        behavior_instruction=behavior.instruction,
        memories_used=[m.id for m in retrieval.memories],
        concepts_activated=list(retrieval.activated_concepts.keys())
    )
```

### Metacognitive Behavior Extraction

Store abstracted patterns, not verbatim traces:

```python
@dataclass
class Behavior:
    name: str           # "check_disk_usage", "explain_concept"
    instruction: str    # "Check disk usage with df -h, then identify largest consumers"
    domain: str         # "docker", "general"

def extract_behavior(response: str) -> Behavior:
    """Extract reusable behavior pattern from response."""
    # Parse СТРАТЕГИЯ line from response
    # Or use LLM to extract if not explicit
    ...
```

---

## Learning System

### Feedback Handler

```python
async def handle_feedback(
    episode_id: str,
    feedback: Literal["positive", "negative", "correction"],
    correction_text: str | None = None
):
    episode = await get_episode(episode_id)
    
    if feedback == "positive":
        episode.success_count += 1
        episode.feedback = "positive"
        
        # Strengthen used memories (Hebbian)
        for memory_id in episode.memories_used:
            await strengthen_memory(memory_id)
        
        # Strengthen concept connections
        await strengthen_concept_links(episode.concepts_activated)
        
        # Check consolidation criteria
        await maybe_consolidate(episode)
        
        # Check reflection trigger
        await maybe_reflect()
        
    elif feedback == "negative":
        episode.failure_count += 1
        episode.feedback = "negative"
        
        # Weaken used memories slightly
        for memory_id in episode.memories_used:
            await weaken_memory(memory_id, factor=0.95)
        
        # Trigger re-reasoning
        return await re_reason(episode)
        
    elif feedback == "correction":
        episode.feedback = "correction"
        episode.correction_text = correction_text
        
        # Create new semantic memory from correction
        await create_memory_from_correction(correction_text, episode)
```

### Consolidation (Episode → Semantic)

Trigger when 3 of 4 criteria met:

```python
async def maybe_consolidate(episode: EpisodicMemory):
    """Check if episode should become semantic memory."""
    
    # Find similar episodes
    similar = await find_similar_episodes(episode.embedding, k=10)
    successful = [e for e in similar if e.success_count > e.failure_count]
    
    criteria_met = 0
    
    # 1. Repetition: 3+ successful uses
    if len(successful) >= 3:
        criteria_met += 1
    
    # 2. Success rate: 85%+
    total_success = sum(e.success_count for e in similar)
    total_failure = sum(e.failure_count for e in similar)
    if total_success / max(total_success + total_failure, 1) >= 0.85:
        criteria_met += 1
    
    # 3. Importance: 7+
    avg_importance = sum(e.importance for e in similar) / len(similar)
    if avg_importance >= 7:
        criteria_met += 1
    
    # 4. Cross-domain: used in 2+ domains
    domains = set(e.domain for e in similar)
    if len(domains) >= 2:
        criteria_met += 1
    
    if criteria_met >= 3:
        await crystallize(episode, similar)

async def crystallize(episode: EpisodicMemory, similar: list[EpisodicMemory]):
    """Transform successful episode pattern into semantic memory."""
    
    # Use LLM to generalize the pattern
    prompt = f"""
    Эти похожие вопросы успешно решались одинаковым подходом:
    {format_episodes(similar)}
    
    Извлеки общий паттерн знания в формате:
    "Когда [ситуация], [действие] потому что [причина]"
    
    Будь кратким и общим, не привязывайся к конкретным деталям.
    """
    
    generalized = await llm.generate(prompt)
    
    # Create new semantic memory
    memory = SemanticMemory(
        id=generate_id(),
        content=generalized,
        concept_ids=episode.concepts_activated,
        source_episode_ids=[e.id for e in similar],
        memory_type="procedure",
        importance=sum(e.importance for e in similar) / len(similar),
        confidence=0.8,  # High confidence from successful use
        ...
    )
    
    await save_memory(memory)
    
    # Mark episodes as consolidated
    for e in similar:
        e.consolidated = True
        e.consolidated_memory_id = memory.id
        await save_episode(e)
```

### Threshold-Triggered Reflection

When accumulated importance exceeds 150, generate higher-level abstractions:

```python
async def maybe_reflect():
    """Check if reflection should be triggered."""
    
    recent_episodes = await get_recent_episodes(hours=24)
    importance_sum = sum(e.importance for e in recent_episodes)
    
    if importance_sum >= 150:
        await generate_reflections(recent_episodes)
        await reset_importance_accumulator()

async def generate_reflections(episodes: list[EpisodicMemory]):
    """Generate higher-level insights from recent episodes."""
    
    prompt = f"""
    Проанализируй недавние взаимодействия и извлеки высокоуровневые инсайты:
    
    {format_episodes(episodes)}
    
    Сгенерируй 2-3 обобщения в формате:
    - Какие темы/проблемы повторяются?
    - Какие подходы работают лучше всего?
    - Какие пробелы в знаниях выявились?
    
    Каждый инсайт должен быть связан с конкретными концептами.
    """
    
    reflections = await llm.generate(prompt)
    
    # Store reflections as high-importance semantic memories
    for reflection in parse_reflections(reflections):
        memory = SemanticMemory(
            content=reflection.content,
            concept_ids=reflection.concepts,
            memory_type="fact",  # Meta-knowledge
            importance=9,  # High importance
            ...
        )
        await save_memory(memory)
```

### Memory Strength (SM-2 Algorithm)

```python
async def update_memory_strength(memory_id: str, quality: int):
    """
    Update memory strength using SM-2 algorithm.
    quality: 0-5 (0=complete failure, 5=perfect recall)
    """
    memory = await get_memory(memory_id)
    
    if quality >= 3:  # Successful recall
        memory.access_count += 1
        # Increase easiness factor
        memory.strength = max(
            1.3,
            memory.strength + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )
    else:  # Failed recall
        # Decrease easiness factor
        memory.strength = max(1.3, memory.strength - 0.2)
    
    memory.last_accessed = datetime.utcnow()
    await save_memory(memory)
```

---

## Re-Reasoning on Failure

When answer doesn't help, try alternative path:

```python
async def re_reason(failed_episode: EpisodicMemory) -> str:
    """Find alternative approach after failure."""
    
    # 1. Get the failed reasoning path
    failed_memories = failed_episode.memories_used
    failed_concepts = failed_episode.concepts_activated
    
    # 2. Find alternative memories (exclude failed ones)
    alternative_memories = await get_memories_for_concepts(
        concept_ids=failed_concepts,
        exclude_ids=failed_memories
    )
    
    # 3. Find successful episodes with different approaches
    successful_alternatives = await find_alternative_episodes(
        concepts=failed_concepts,
        exclude_behavior=failed_episode.behavior_name
    )
    
    # 4. Generate new response with alternatives
    prompt = f"""
    Предыдущий подход не помог: {failed_episode.behavior_instruction}
    
    Попробуй альтернативный подход используя эти знания:
    {format_memories(alternative_memories)}
    
    Успешные альтернативные подходы в похожих ситуациях:
    {format_episodes(successful_alternatives)}
    
    Предложи другое решение или задай уточняющий вопрос.
    """
    
    return await llm.generate(prompt)
```

---

## Gap Detection (6 Signals)

Same as before, but adapted for concept-centric model:

1. **Undefined concepts** — concept referenced but has no description or memories
2. **Contradictions** — memories with same concepts but conflicting content
3. **Missing rationale** — procedural memory without "because" explanation
4. **Dead ends** — "check X" memories without follow-up options
5. **Implicit prerequisites** — technical terms without linked concept definitions
6. **Missing relationships** — concepts co-occur in memories but no edge in graph

---

## Implementation Phases

### Phase 1: Foundation

**Goal:** Concept extraction, basic storage, simple retrieval.

**Deliverables:**
1. Project setup, Docker compose for Neo4j
2. Data models (Concept, SemanticMemory, EpisodicMemory)
3. Document parser
4. Concept extractor (LLM-based)
5. Memory extractor (facts and procedures from docs)
6. Neo4j schema and storage
7. Basic vector search

**Exit:** Can ingest 20 mock docs, extract concepts/memories, query by concept.

---

### Phase 2: Spreading Activation

**Goal:** Implement brain-like associative retrieval.

**Deliverables:**
1. Concept relationship extraction (edges with types/weights)
2. Spreading activation algorithm
3. Query-oriented edge filtering
4. Lateral inhibition
5. Concept activation tracking

**Exit:** Query activates related concepts, finds memories at intersections.

---

### Phase 3: Hybrid Retrieval

**Goal:** Production-quality retrieval combining multiple signals.

**Deliverables:**
1. BM25 full-text search
2. Reciprocal Rank Fusion
3. Reranking with recency + importance + relevance
4. Similar episode retrieval
5. Full retrieval pipeline

**Exit:** Retrieval finds relevant memories even for indirect queries.

---

### Phase 4: Reasoning & Synthesis

**Goal:** Generate responses using retrieved context.

**Deliverables:**
1. Response synthesizer
2. Behavior extraction (metacognitive reuse)
3. Episode creation and storage
4. Re-reasoning on failure

**Exit:** System answers questions, stores reasoning traces, can try alternatives.

---

### Phase 5: Learning System

**Goal:** Learn from feedback, consolidate knowledge.

**Deliverables:**
1. Feedback handler (positive/negative/correction)
2. Memory strength updates (SM-2)
3. Concept link strengthening (Hebbian)
4. Consolidation (3-of-4 criteria)
5. Threshold-triggered reflection

**Exit:** Positive feedback strengthens system, episodes crystallize into facts.

---

### Phase 6: API & Integration

**Goal:** OpenAI-compatible API for Open WebUI.

**Deliverables:**
1. `/v1/chat/completions` endpoint
2. `/v1/feedback` endpoint
3. Admin endpoints (calibration, review)
4. Docker Compose (Neo4j + API)
5. Health checks

**Exit:** Full system runs in Docker, API works with Open WebUI.

---

## Configuration

### Development

```python
class Settings(BaseSettings):
    # LLM
    llm_base_url: str = "http://localhost:8888/v1"
    llm_model: str = "kimi-k2"
    llm_max_concurrent: int = 16
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Embeddings (fast for dev)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    
    # Spreading activation
    activation_decay: float = 0.85
    activation_threshold: float = 0.3
    activation_max_hops: int = 3
    activation_rescale: float = 0.4
    
    # Consolidation
    consolidation_min_repetitions: int = 3
    consolidation_min_success_rate: float = 0.85
    consolidation_min_importance: float = 7.0
    reflection_importance_threshold: float = 150.0
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
```

### Production

```bash
# .env.production

# Embeddings (Russian-optimized)
EMBEDDING_MODEL=ai-sage/Giga-Embeddings-instruct
EMBEDDING_DIMENSIONS=1024

# LLM
LLM_BASE_URL=http://your-host:8888/v1

# Neo4j
NEO4J_URI=bolt://your-neo4j:7687
NEO4J_PASSWORD=secure-password
```

When switching to production:
1. Update embedding model and dimensions
2. Drop and recreate vector indexes with new dimensions
3. Re-run ingestion to regenerate embeddings

---

## LLM Prompts

### Concept Extraction

```
Извлеки концепты (сущности, идеи, термины) из текста.

Текст:
{content}

Для каждого концепта укажи:
- name: название (нормализованное, в нижнем регистре)
- type: tool | resource | action | state | config | error
- description: краткое определение если понятно из текста

Также укажи связи между концептами:
- source → target, type (uses | needs | causes | contains | is_a)

Выведи JSON.
```

### Memory Extraction

```
Извлеки единицы знаний из документа.

Документ: {title}
Содержимое:
{content}

Извлеки:
1. ФАКТЫ — определения, описания ("X это Y")
2. ПРОЦЕДУРЫ — инструкции ("чтобы сделать X, нужно Y")
3. СВЯЗИ — зависимости ("X требует Y", "X влияет на Y")

Для каждой единицы укажи:
- content: текст знания
- type: fact | procedure | relationship
- concepts: список связанных концептов
- importance: 1-10 (насколько критично знать)

Выведи JSON.
```

---

## Testing Guidelines

- Unit test each algorithm (spreading activation, consolidation criteria, SM-2)
- Mock LLM for deterministic tests
- Integration tests with real Neo4j (use testcontainers or require running instance)
- Test consolidation by simulating feedback sequences
- Test re-reasoning by simulating failures

---

## Quality Checklist

Each phase:
- [ ] All tests pass
- [ ] mypy passes
- [ ] ruff check passes
- [ ] README updated
- [ ] Can demonstrate feature working

---

## Final Notes

1. **Keep it simple** — Start minimal, add complexity only when needed
2. **Test the learning** — Simulate feedback loops, verify consolidation works
3. **Russian content, English code** — Prompts and user content in Russian
4. **Brain-inspired, not brain-identical** — Use cognitive science as guide, not constraint

Start with Phase 1. Report progress after each phase.
