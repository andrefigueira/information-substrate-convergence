# System Architecture Overview

## High-Level Structure

The ISC project consists of two complementary systems exploring consciousness emergence:

```
information-substrate-convergence/
├── isc_ai_system/           # Interactive AI implementing ISC principles
│   ├── src/isc_ai/          # Core modules
│   ├── scripts/             # Training and utilities
│   └── examples/            # Usage demonstrations
├── ca_experiment/           # Cellular automata evolution experiments
│   ├── ca/                  # CA simulation modules
│   └── tests/               # Unit tests
├── results/                 # Experiment outputs
├── PAPER.md                 # Full theoretical paper
└── .context/                # This documentation
```

## ISC AI System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         ISCCore                                  │
│                    (Orchestrator)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Self-        │  │ Knowledge    │  │ Information          │  │
│  │ Modifying    │  │ Graph        │  │ Integrator           │  │
│  │ Network      │  │              │  │ (Phi Calculator)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └────────────┬────┴──────────────────────┘              │
│                      │                                          │
│  ┌──────────────┐   │    ┌──────────────┐  ┌───────────────┐  │
│  │ Conversation │◄──┴───►│ Response     │  │ Learning      │  │
│  │ Memory       │        │ Generator    │  │ Engine        │  │
│  └──────────────┘        └──────────────┘  └───────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    Persistence Layer                            │
│              (SQLite + PyTorch State Files)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ Text Encoding   │  ◄── Sentence-Transformers (all-MiniLM-L6-v2)
│ (384-dim)       │      Produces semantic embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Self-Modifying  │  ◄── Observer layers monitor activations
│ Network         │      Meta-weights enable self-modification
└────────┬────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│ Concept         │                   │ Information     │
│ Extraction      │                   │ Integration     │
│ (NLTK)          │                   │ (Phi Calc)      │
└────────┬────────┘                   └────────┬────────┘
         │                                      │
         ▼                                      │
┌─────────────────┐                             │
│ Knowledge Graph │  ◄── NetworkX graph of      │
│ Update          │      concepts & relations    │
└────────┬────────┘                             │
         │                                      │
         ├──────────────────────────────────────┤
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│ Memory          │                   │ Metrics         │
│ Integration     │                   │ Update          │
└────────┬────────┘                   └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Response        │  ◄── Combines graph traversals,
│ Generation      │      templates, and context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Learning        │  ◄── Self-supervised + feedback
│ Update          │      Meta-weight adjustment
└────────┬────────┘
         │
         ▼
    System Output
    + State Persistence
```

## Cellular Automata System Architecture

### Processing Pipeline

```
Initial Population (Random 18-bit rules)
         │
         ▼
┌─────────────────┐
│ Evolution Loop  │
│                 │
│  ┌───────────┐  │
│  │ Fitness   │  │  ◄── Self-modeling capability score
│  │ Evaluation│  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Selection │  │  ◄── Tournament selection
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Crossover │  │  ◄── Combine rule bits
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Mutation  │  │  ◄── Adaptive mutation rate
│  └───────────┘  │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis        │  ◄── Self-modeling metrics
│ & Export        │      Convergence tracking
└─────────────────┘
```

## Storage Architecture

### SQLite Schema

```sql
-- Conversation memory
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    timestamp DATETIME,
    user_input TEXT,
    system_response TEXT,
    embedding BLOB,      -- Binary serialized 384-dim vector
    metadata TEXT        -- JSON metadata
);

-- Session tracking
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    start_time DATETIME,
    end_time DATETIME,
    interaction_count INTEGER,
    metadata TEXT
);
```

### State Persistence Format

PyTorch checkpoint containing:
```python
{
    'model_state': network.state_dict(),      # Neural network weights
    'knowledge_graph': graph.to_dict(),       # NetworkX node_link format
    'memory': memory.export(),                # Serialized interactions
    'metrics': {
        'phi_history': [...],
        'coherence_scores': [...],
        'learning_rates': [...],
    },
    'config': {...}                           # System configuration
}
```

## Integration Points

### Cross-System Communication

| System | Shared Concepts | Integration Method |
|--------|-----------------|-------------------|
| AI + CA | Self-reference metrics | Information integration measures |
| AI + CA | Pattern emergence | Convergence tracking |
| Components | Embeddings | 384-dim semantic vectors |
| Components | State | Shared persistence layer |

### Module Dependencies

```
core.py
├── information_integration.py
├── knowledge_graph.py
├── memory.py
├── response_generator.py
├── learning.py
└── persistence.py
```

## Performance Considerations

### Optimization Strategies

1. **Phi Caching**: Partition calculations cached to avoid recomputation
2. **Lazy Evaluation**: Heavy computations deferred until needed
3. **Incremental Updates**: Knowledge graph updates incrementally
4. **Batch Processing**: Learning engine uses batch updates for stability

### Bottlenecks

| Operation | Typical Time | Mitigation |
|-----------|-------------|------------|
| Phi calculation | 50-200ms | Caching, approximation |
| Embedding generation | 20-50ms | Model warm-up |
| Graph traversal | <10ms | NetworkX optimization |
| Response generation | 100-500ms | Template caching |

## Related Files

- [dependencies.md](dependencies.md) - Dependency injection patterns
- [ca-system.md](ca-system.md) - CA system details
- [../components/](../components/) - Individual component documentation
