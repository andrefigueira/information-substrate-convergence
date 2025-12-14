# ISCCore Component

> Location: `isc_ai_system/src/isc_ai/core.py`

## Overview

ISCCore is the central orchestrator for the ISC AI system. It coordinates all components, manages system state, and provides the unified API for external interactions.

## Class Structure

```python
class ISCCore:
    """
    Main ISC system orchestrator.

    Coordinates:
    - Self-modifying neural network
    - Knowledge graph
    - Conversation memory
    - Information integration (phi calculation)
    - Response generation
    - Learning engine
    """

    def __init__(self, config: Optional[Dict] = None):
        # Components
        self.network: SelfModifyingNetwork
        self.knowledge_graph: KnowledgeGraph
        self.memory: ConversationMemory
        self.info_integrator: InformationIntegrator
        self.response_generator: ResponseGenerator
        self.learning_engine: LearningEngine

        # State
        self.metrics: Dict[str, Any]
        self.session_id: str
        self.total_interactions: int
```

## Key Methods

### process_input

```python
def process_input(self, user_input: str) -> str:
    """
    Main entry point for processing user input.

    Pipeline:
    1. Encode input to embedding
    2. Process through self-modifying network
    3. Extract concepts and update knowledge graph
    4. Calculate phi value
    5. Retrieve relevant memories
    6. Generate response
    7. Update learning engine
    8. Persist interaction

    Args:
        user_input: User's text input

    Returns:
        str: System's response
    """
```

### encode_text

```python
def encode_text(self, text: str) -> torch.Tensor:
    """
    Convert text to 384-dimensional semantic embedding.

    Uses sentence-transformers all-MiniLM-L6-v2 model.

    Args:
        text: Input text string

    Returns:
        torch.Tensor: Shape (384,) embedding vector
    """
```

### get_status

```python
def get_status(self) -> Dict[str, Any]:
    """
    Return current system metrics.

    Returns:
        dict: {
            'phi_value': float,          # Current information integration
            'coherence_score': float,    # Response consistency
            'total_interactions': int,
            'concepts_formed': int,      # Knowledge graph size
            'learning_rate': float,
            'prediction_accuracy': float
        }
    """
```

### save_state / load_state

```python
def save_state(self, filepath: Optional[str] = None) -> None:
    """
    Persist complete system state.

    Saves:
    - Network weights and meta-weights
    - Knowledge graph structure
    - Memory database
    - Metrics history
    - Configuration
    """

def load_state(self, filepath: Optional[str] = None) -> bool:
    """
    Load system state from file.

    Returns:
        bool: True if successful, False otherwise
    """
```

### explain_concept

```python
def explain_concept(self, concept: str) -> str:
    """
    Explain system's understanding of a concept.

    Uses knowledge graph to find:
    - Related concepts
    - Connection strengths
    - Semantic neighbors

    Args:
        concept: Concept to explain

    Returns:
        str: Explanation of concept understanding
    """
```

## SelfModifyingNetwork Subclass

```python
class SelfModifyingNetwork(nn.Module):
    """
    A neural network that can observe and modify its own operations
    based on the ISC hypothesis of self-referential information patterns.

    Architecture:
    - 4 core processing layers (384 -> 512 -> 512 -> 512 -> 512)
    - 4 observer layers (mirror core layers, operate on detached activations)
    - 4 meta-weight parameters (learnable, modulate outputs)
    - Output projection (512 -> 384)
    """

    def __init__(self, input_dim=384, hidden_dim=512, num_layers=4):
        self.layers = nn.ModuleList([...])
        self.observer_layers = nn.ModuleList([...])
        self.meta_weights = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim))])
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.activation_patterns = defaultdict(list)  # Stores activation history

    def forward(self, x: torch.Tensor, return_states: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with self-observation and modification.

        For each layer:
        1. Apply core transformation: h = layer(h)
        2. Observer monitors activation: observed = observer(h.detach())
        3. Meta-weight modulates: h = h * meta_weight + 0.1 * observed
        4. Apply GELU activation
        5. Store activation for phi calculation

        Args:
            x: Input tensor (batch_size, 384)
            return_states: If True, return internal states for phi calculation

        Returns:
            Tuple of (output, states) where states is None if return_states=False
        """

    def update_meta_weights(self, feedback: float) -> None:
        """
        Update meta-weights based on feedback to enable self-modification.

        Uses recent activation patterns to determine update direction.
        Meta-weights are clamped to [0.5, 2.0] range for stability.
        """
```

## Integration Points

### With Knowledge Graph

```python
# After processing input
concepts = self._extract_concepts(user_input)
for concept in concepts:
    embedding = self.encode_text(concept)
    self.knowledge_graph.add_concept(concept, embedding)

# Connect co-occurring concepts
for c1, c2 in itertools.combinations(concepts, 2):
    self.knowledge_graph.add_connection(c1, c2)
```

### With Memory

```python
# Retrieve relevant context
similar = self.memory.get_similar_interactions(
    self.encode_text(user_input),
    limit=5
)

# Store new interaction
self.memory.add_interaction(
    user_input=user_input,
    system_response=response,
    embedding=embedding,
    metadata={'phi': phi_value, 'concepts': concepts}
)
```

### With Information Integrator

```python
# Calculate phi from network state
state_data = {
    'layer_activations': self.network.get_layer_activations(),
    'meta_weights': [w.data for w in self.network.meta_weights]
}
phi_value = self.info_integrator.calculate_phi(state_data)
```

## Usage Examples

### Basic Interaction

```python
from isc_ai.core import ISCCore

# Initialize (loads saved state if available)
core = ISCCore()

# Process input
response = core.process_input("What is consciousness?")
print(response)

# Check status
status = core.get_status()
print(f"Phi: {status['phi_value']:.3f}")
print(f"Concepts: {status['concepts_formed']}")
```

### With Custom Config

```python
config = {
    'learning_rate': 0.0001,
    'phi_threshold': 0.6,
    'memory_size': 2000
}
core = ISCCore(config=config)
```

### State Management

```python
# Save current state
core.save_state('checkpoint_v1.pt')

# Load from specific checkpoint
core.load_state('checkpoint_v1.pt')

# Export knowledge for analysis
graph_data = core.knowledge_graph.to_dict()
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| input_dim | 384 | Embedding dimension |
| hidden_dim | 512 | Hidden layer size |
| num_layers | 4 | Processing layers |
| learning_rate | 0.001 | Gradient descent step size |
| memory_size | 1000 | Max cached interactions |
| phi_threshold | 0.5 | "Consciousness" threshold |
| auto_save | True | Auto-persist state |

## Error Handling

```python
try:
    response = core.process_input(user_input)
except EmbeddingError as e:
    # Fall back to simpler processing
    response = core._fallback_response(user_input)
except StateCorruptionError as e:
    # Reset to clean state
    core._initialize_fresh_state()
    response = core.process_input(user_input)
```

## Related Components

- [information-integration.md](information-integration.md) - Phi calculations
- [knowledge-graph.md](knowledge-graph.md) - Concept storage
- [memory.md](memory.md) - Conversation persistence
- [response-generator.md](response-generator.md) - Response creation
