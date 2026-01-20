# Engram Guide

Quick reference for using the Engram knowledge system.

---

## System Overview

Engram is a brain-inspired knowledge system that transforms documents into an intelligent assistant.

### Memory Types

| Type | Color | Description |
|------|-------|-------------|
| **Concept** | Cyan `#5eead4` | Atomic ideas extracted from documents |
| **Semantic Memory** | Purple `#a78bfa` | Facts and procedures linked to concepts |
| **Episodic Memory** | Pink `#f472b6` | Past reasoning traces with outcomes |

### How It Works

```
Documents → Concepts + Memories → Graph Database (Neo4j)
                                         ↓
User Query → Spreading Activation → Retrieve Relevant Memories
                                         ↓
                              LLM Synthesizes Answer
                                         ↓
                              Episode Created (learns)
```

**Key difference from RAG**: Instead of retrieving document chunks, Engram activates concepts through a network and retrieves connected memories.

---

## Quick Start

### Prerequisites

- Docker (for Neo4j)
- Python 3.11+
- uv package manager

### Start the System

```bash
# 1. Start Neo4j
docker start engram-neo4j

# 2. Start API server
uv run python -m engram.api.main
```

### Access Points

| URL | Description |
|-----|-------------|
| `http://localhost:8000/constellation` | Graph visualization |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/v1/chat/completions` | Chat API |

---

## Constellation Mode

Interactive graph visualization at `/constellation`.

### Navigation

| Action | How |
|--------|-----|
| **Pan** | Click and drag |
| **Zoom** | Mouse wheel |
| **Select node** | Click on node |
| **Deselect** | Press `Escape` |
| **Search** | Type in search box, press Enter |

### Control Buttons

| Button | Function |
|--------|----------|
| **Clusters** | Toggle cluster-based coloring |
| **Constellation** | Toggle constellation view mode |

### Legend (Bottom Left)

Click on a legend item to filter by that type:
- **Concept** (cyan) - Shows only concepts
- **Semantic** (purple) - Shows only semantic memories
- **Episodic** (pink) - Shows only episodic memories

Click again to clear filter.

### Node Selection

When you click a node:
1. Info panel opens on the right
2. Connected nodes are highlighted
3. Panel shows node details and connections

---

## Chat Panel

Click **💬** button (bottom right) to open chat.

### Basic Usage

1. Type question in input field
2. Press Enter or click Send
3. Response appears with confidence score
4. Click "Activated: X concepts, Y memories" to highlight used nodes

### Debug Mode

Click **🔍 Debug** button to enable debug mode.

When enabled, after each query you see:

#### Retrieved Memories (Purple bars)
- Memory content (truncated)
- Score bar (0-1, higher = more relevant)
- Sources: `V`=Vector, `B`=BM25, `G`=Graph, `F`=Forced

#### Activated Concepts (Cyan bars)
- Concept name
- Activation level (0-1)

### Force Include/Exclude

Test "what if" scenarios by forcing nodes in or out of retrieval:

| Button | Action |
|--------|--------|
| **+** | Force this node INTO next query (score=1.0, source=F) |
| **−** | Force this node OUT of next query |

**Workflow:**
1. Send a query
2. See results in debug panel
3. Click **+** or **−** on nodes you want to test
4. Send another query (same or different)
5. Results will include/exclude those nodes

Click node name in debug panel to highlight it in graph.

### Other Chat Controls

| Button | Function |
|--------|----------|
| **Show Activation** | Re-highlight nodes from last response |
| **Clear** | Clear chat history |

---

## API Reference

### Chat Endpoint

```
POST /v1/chat/completions
```

**Basic request:**
```json
{
  "messages": [{"role": "user", "content": "What is Docker?"}],
  "model": "engram"
}
```

**With debug mode:**
```json
{
  "messages": [{"role": "user", "content": "What is Docker?"}],
  "debug": true,
  "force_include_nodes": ["mem_abc123"],
  "force_exclude_nodes": ["c_xyz789"]
}
```

**Response includes:**
- `choices[0].message.content` - The answer
- `confidence` - Confidence score (0-1)
- `concepts_activated` - List of concept IDs used
- `memories_used` - List of memory IDs used
- `debug_info` - Detailed retrieval info (when debug=true)

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/feedback` | POST | Submit feedback for episode |
| `/admin/stats` | GET | System statistics |
| `/admin/reset?confirm=true` | POST | Reset database |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Escape` | Clear selection, filters, highlights |
| `Enter` | Send chat message (in chat input) |

---

## Tips

1. **Low confidence?** The system may not have relevant knowledge. Try rephrasing or check if documents were ingested.

2. **No memories retrieved?** Concepts may not be connected to memories. Check the graph structure.

3. **Testing retrieval?** Use debug mode with +/− buttons to see how including/excluding nodes affects answers.

4. **Large graph slow?** Use search instead of panning. Zoom out to see clusters.

5. **Node not found in debug?** Click the node name - if it's not loaded in current view, it won't highlight but the ID is still valid for force include/exclude.
