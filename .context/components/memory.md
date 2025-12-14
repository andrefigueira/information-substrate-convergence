# Conversation Memory Component

> Location: `isc_ai_system/src/isc_ai/memory.py`

## Overview

The Conversation Memory system provides persistent storage and retrieval for conversation history. It enables contextual awareness by finding relevant past interactions.

## Architecture

### Storage Layer

```
┌─────────────────────────────────────────────────┐
│               ConversationMemory                │
├─────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────┐  │
│  │         In-Memory Cache (LRU)             │  │
│  │         Recent interactions               │  │
│  │         Fast retrieval                    │  │
│  └───────────────────────────────────────────┘  │
│                      │                          │
│                      ▼                          │
│  ┌───────────────────────────────────────────┐  │
│  │           SQLite Database                  │  │
│  │    ┌─────────────┬───────────────┐        │  │
│  │    │interactions │   sessions    │        │  │
│  │    └─────────────┴───────────────┘        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Database Schema

```sql
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_input TEXT NOT NULL,
    system_response TEXT NOT NULL,
    embedding BLOB,                    -- Serialized 384-dim vector
    metadata TEXT,                     -- JSON metadata
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    interaction_count INTEGER DEFAULT 0,
    metadata TEXT                      -- JSON session metadata
);

-- Indexes for efficient queries
CREATE INDEX idx_interactions_session ON interactions(session_id);
CREATE INDEX idx_interactions_timestamp ON interactions(timestamp);
```

## Class Structure

```python
class ConversationMemory:
    """
    Persistent conversation memory with semantic retrieval.

    Features:
    - SQLite backend for durability
    - In-memory LRU cache for performance
    - Embedding-based similarity search
    - Session management
    """

    def __init__(
        self,
        db_path: str = "conversation_memory.db",
        cache_size: int = 1000
    ):
        self.db_path = db_path
        self.cache_size = cache_size
        self.cache: OrderedDict = OrderedDict()
        self._init_database()
```

## Key Methods

### add_interaction

```python
def add_interaction(
    self,
    user_input: str,
    system_response: str,
    embedding: np.ndarray,
    metadata: Optional[Dict] = None,
    session_id: Optional[str] = None
) -> int:
    """
    Store new interaction in memory.

    Args:
        user_input: User's message
        system_response: System's response
        embedding: 384-dim semantic embedding of input
        metadata: Optional metadata (phi, concepts, etc.)
        session_id: Session identifier (created if None)

    Returns:
        int: Interaction ID
    """
    # Serialize embedding
    embedding_blob = embedding.tobytes()
    metadata_json = json.dumps(metadata) if metadata else None

    # Insert into database
    cursor = self.conn.execute("""
        INSERT INTO interactions (session_id, user_input, system_response, embedding, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, user_input, system_response, embedding_blob, metadata_json))

    interaction_id = cursor.lastrowid

    # Update cache
    self._cache_interaction({
        'id': interaction_id,
        'user_input': user_input,
        'system_response': system_response,
        'embedding': embedding,
        'metadata': metadata
    })

    return interaction_id
```

### get_similar_interactions

```python
def get_similar_interactions(
    self,
    query: str,
    k: int = 5,
    encoder=None
) -> List[Dict]:
    """
    Find interactions similar to the query.

    Args:
        query: Text query to search for
        k: Maximum results to return
        encoder: Function to encode text to embeddings (required)

    Returns:
        List of interaction dicts sorted by similarity

    Note: Returns empty list if no encoder provided or no interactions.
    """
    if not self.interactions or not encoder:
        return []

    # Encode query
    query_embedding = encoder(query)
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().cpu().numpy()

    # Calculate similarities
    similarities = []
    for i, interaction in enumerate(self.interactions):
        if interaction.get("embedding") is not None:
            similarity = cosine_similarity(query_embedding, interaction["embedding"])
            similarities.append((i, similarity))

    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [self.interactions[i] for i, _ in similarities[:k]]
```

### get_recent_interactions

```python
def get_recent_interactions(
    self,
    limit: int = 10,
    session_id: Optional[str] = None
) -> List[Dict]:
    """
    Get most recent interactions.

    Args:
        limit: Maximum results
        session_id: Filter by session (optional)

    Returns:
        List of interaction dicts, most recent first
    """
    if session_id:
        query = """
            SELECT * FROM interactions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        rows = self.conn.execute(query, (session_id, limit)).fetchall()
    else:
        query = """
            SELECT * FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
        """
        rows = self.conn.execute(query, (limit,)).fetchall()

    return [self._row_to_dict(row) for row in rows]
```

### Session Management

```python
def start_session(self, metadata: Optional[Dict] = None) -> str:
    """
    Start new conversation session.

    Returns:
        str: New session ID
    """
    session_id = str(uuid.uuid4())
    metadata_json = json.dumps(metadata) if metadata else None

    self.conn.execute("""
        INSERT INTO sessions (id, metadata)
        VALUES (?, ?)
    """, (session_id, metadata_json))

    return session_id

def end_session(self, session_id: str) -> None:
    """
    Mark session as ended.
    """
    self.conn.execute("""
        UPDATE sessions
        SET end_time = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (session_id,))
```

## Similarity Search

### Cosine Similarity

```python
def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between embeddings.

    Returns:
        float: Similarity score in range [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)
```

### Database Search

```python
def _search_database(
    self,
    query_embedding: np.ndarray,
    limit: int,
    threshold: float
) -> List[Dict]:
    """
    Search database for similar interactions.

    Note: Full scan required without vector index.
    Consider using FAISS for larger databases.
    """
    rows = self.conn.execute("""
        SELECT * FROM interactions
        WHERE embedding IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 10000
    """).fetchall()

    results = []
    for row in rows:
        embedding = np.frombuffer(row['embedding'], dtype=np.float32)
        similarity = self._cosine_similarity(query_embedding, embedding)
        if similarity >= threshold:
            results.append({
                **self._row_to_dict(row),
                'similarity': similarity
            })

    return results
```

## Cache Management

### LRU Cache

```python
def _cache_interaction(self, interaction: Dict) -> None:
    """
    Add interaction to LRU cache.

    Evicts oldest entry if cache is full.
    """
    interaction_id = interaction['id']

    # Move to end if exists
    if interaction_id in self.cache:
        self.cache.move_to_end(interaction_id)
    else:
        self.cache[interaction_id] = interaction

    # Evict oldest if over limit
    while len(self.cache) > self.cache_size:
        self.cache.popitem(last=False)
```

## Import/Export

```python
def export(self) -> Dict:
    """
    Export all memory data for persistence.

    Returns:
        dict: {
            'interactions': [...],
            'sessions': [...],
            'metadata': {...}
        }
    """
    interactions = self.conn.execute("SELECT * FROM interactions").fetchall()
    sessions = self.conn.execute("SELECT * FROM sessions").fetchall()

    return {
        'interactions': [self._row_to_dict(r, include_blob=True) for r in interactions],
        'sessions': [dict(s) for s in sessions],
        'metadata': {
            'export_time': datetime.now().isoformat(),
            'count': len(interactions)
        }
    }

def import_data(self, data: Dict) -> int:
    """
    Import memory data from export.

    Returns:
        int: Number of interactions imported
    """
    count = 0
    for interaction in data.get('interactions', []):
        self._import_interaction(interaction)
        count += 1
    return count
```

## Usage Examples

### Basic Usage

```python
from isc_ai.memory import ConversationMemory

memory = ConversationMemory()

# Start session
session_id = memory.start_session({'user': 'researcher'})

# Store interaction
memory.add_interaction(
    user_input="What is consciousness?",
    system_response="Consciousness is...",
    embedding=encoder.encode("What is consciousness?"),
    metadata={'phi': 0.67, 'concepts': ['consciousness']},
    session_id=session_id
)
```

### Retrieval

```python
# Find similar past conversations
query_embedding = encoder.encode("How does awareness emerge?")
similar = memory.get_similar_interactions(query_embedding, limit=3)

for interaction in similar:
    print(f"Similarity: {interaction['similarity']:.3f}")
    print(f"Q: {interaction['user_input']}")
    print(f"A: {interaction['system_response'][:100]}...")
```

### Session Analysis

```python
# Get session history
history = memory.get_recent_interactions(limit=20, session_id=session_id)

# Analyze conversation flow
for i, interaction in enumerate(reversed(history)):
    print(f"{i+1}. User: {interaction['user_input']}")
    phi = interaction.get('metadata', {}).get('phi', 'N/A')
    print(f"   Phi: {phi}")
```

## Performance Considerations

| Operation | Cache Hit | Cache Miss |
|-----------|-----------|------------|
| Add interaction | <1ms | ~5ms |
| Get recent | <1ms | ~10ms |
| Similarity search | ~50ms (1000 cached) | ~500ms (10000 DB) |

### Scaling Recommendations

For >10,000 interactions:
1. Use FAISS for vector similarity search
2. Implement embedding quantization
3. Add date-range filtering to queries
4. Consider periodic archiving

## Related Components

- [isc-core.md](isc-core.md) - System orchestrator
- [knowledge-graph.md](knowledge-graph.md) - Concept storage
- [../architecture/overview.md](../architecture/overview.md) - System architecture
