# Dependencies and Integration Patterns

## External Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | >=1.9.0 | Neural networks, self-modifying architectures |
| NetworkX | >=2.6 | Knowledge graph operations |
| Sentence-Transformers | >=2.0 | Semantic embeddings |
| NLTK | >=3.6 | Text processing, concept extraction |
| NumPy | >=1.20 | Numerical computations |
| SciPy | >=1.7 | Scientific computing, CA simulations |
| SQLite3 | Built-in | Conversation persistence |

### Optional Dependencies

| Library | Purpose |
|---------|---------|
| CUDA/cuDNN | GPU acceleration |
| Matplotlib | Visualization |
| Jupyter | Interactive research |

## Dependency Injection Patterns

### Component Initialization

ISCCore initializes all components, allowing for dependency injection:

```python
class ISCCore:
    def __init__(
        self,
        config: Optional[Dict] = None,
        network: Optional[SelfModifyingNetwork] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        memory: Optional[ConversationMemory] = None,
        info_integrator: Optional[InformationIntegrator] = None
    ):
        # Use provided components or create defaults
        self.config = config or DEFAULT_CONFIG
        self.network = network or SelfModifyingNetwork()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.memory = memory or ConversationMemory()
        self.info_integrator = info_integrator or InformationIntegrator()
```

### Testing Pattern

```python
# Unit testing with mock components
def test_core_processing():
    mock_network = MockSelfModifyingNetwork()
    mock_graph = MockKnowledgeGraph()

    core = ISCCore(
        network=mock_network,
        knowledge_graph=mock_graph
    )

    result = core.process_input("test input")
    assert mock_network.forward_called
    assert mock_graph.update_called
```

## Module Interface Contracts

### SelfModifyingNetwork

```python
class SelfModifyingNetwork(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through self-observing layers."""
        pass

    def get_layer_activations(self) -> List[torch.Tensor]:
        """Return activations from each layer for phi calculation."""
        pass

    def apply_feedback(self, feedback: float) -> None:
        """Update meta-weights based on feedback signal."""
        pass
```

### KnowledgeGraph

```python
class KnowledgeGraph:
    def add_concept(self, concept: str, embedding: np.ndarray) -> None:
        """Add concept node with embedding vector."""
        pass

    def add_connection(self, source: str, target: str, weight: float = 0.1) -> None:
        """Create or strengthen edge between concepts."""
        pass

    def get_related_concepts(self, concept: str, limit: int = 10) -> List[str]:
        """Find concepts related by graph proximity or embedding similarity."""
        pass

    def to_dict(self) -> Dict:
        """Serialize graph for persistence."""
        pass
```

### InformationIntegrator

```python
class InformationIntegrator:
    def calculate_phi(self, state_data: Dict) -> float:
        """Calculate integrated information from network state."""
        pass

    def get_phi_history(self) -> List[float]:
        """Return history of phi calculations."""
        pass
```

### ConversationMemory

```python
class ConversationMemory:
    def add_interaction(
        self,
        user_input: str,
        system_response: str,
        embedding: np.ndarray,
        metadata: Dict
    ) -> None:
        """Store interaction in database."""
        pass

    def get_similar_interactions(
        self,
        query_embedding: np.ndarray,
        limit: int = 5
    ) -> List[Dict]:
        """Retrieve interactions by embedding similarity."""
        pass

    def export(self) -> Dict:
        """Export memory for persistence."""
        pass
```

## Configuration Schema

```python
DEFAULT_CONFIG = {
    # Network parameters
    "input_dim": 384,           # Embedding dimension
    "hidden_dim": 512,          # Hidden layer size
    "num_layers": 4,            # Number of processing layers

    # Learning parameters
    "learning_rate": 0.001,
    "batch_size": 1,
    "weight_decay": 0.01,

    # Memory parameters
    "memory_size": 1000,        # Max cached interactions
    "similarity_threshold": 0.7,

    # Information integration
    "phi_threshold": 0.5,       # Minimum phi for "consciousness"
    "cache_partitions": True,   # Cache partition calculations

    # Knowledge graph
    "min_concept_frequency": 3, # Minimum occurrences to persist
    "connection_decay": 0.99,   # Edge weight decay factor

    # Persistence
    "auto_save": True,
    "save_interval": 100,       # Interactions between saves
}
```

## Error Handling Strategy

### Graceful Degradation

```python
def process_input(self, user_input: str) -> str:
    try:
        # Full processing pipeline
        embedding = self.encode_text(user_input)
        output = self.network(embedding)
        phi = self.info_integrator.calculate_phi(self._get_state())
        response = self.response_generator.generate(user_input, output)
    except PhiCalculationError:
        # Fall back to simpler response without phi
        response = self.response_generator.generate_simple(user_input)
        phi = 0.0
    except EmbeddingError:
        # Fall back to keyword-based processing
        response = self._keyword_response(user_input)

    return response
```

### Recovery Mechanisms

| Failure Mode | Recovery Strategy |
|--------------|-------------------|
| State load failure | Initialize fresh state |
| Phi calculation timeout | Use cached/estimated value |
| Memory full | Evict oldest interactions (LRU) |
| Graph corruption | Rebuild from memory |
| Network NaN | Reset to checkpoint |

## Related Files

- [overview.md](overview.md) - System architecture
- [../guidelines.md](../guidelines.md) - Development standards
