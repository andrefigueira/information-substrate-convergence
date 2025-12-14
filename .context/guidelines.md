# Development Guidelines

## Code Style Standards

### Python Style

- **Formatter**: Black (default line length 88)
- **Linter**: Flake8
- **Type Hints**: Required for all function signatures

```bash
# Format code
make format

# Lint code
make lint

# Run both
make check
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `SelfModifyingNetwork`, `KnowledgeGraph` |
| Functions | snake_case | `calculate_phi`, `get_related_concepts` |
| Variables | snake_case | `phi_value`, `layer_activations` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_HIDDEN_DIM`, `MAX_CACHE_SIZE` |
| Private | Leading underscore | `_next_state`, `_calculate_mi` |

### Import Order

```python
# 1. Standard library
import os
import json
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# 3. Local
from isc_ai.core import ISCCore
from isc_ai.knowledge_graph import KnowledgeGraph
```

### Documentation

```python
def calculate_phi(self, state_data: Dict[str, Any]) -> float:
    """
    Calculate integrated information (phi) from network state.

    Phi measures how much information the system generates beyond
    what its parts generate independently.

    Args:
        state_data: Dictionary containing:
            - layer_activations: List of torch.Tensor
            - meta_weights: List of torch.Tensor

    Returns:
        float: Phi value in range [0, 1], where higher values
               indicate greater information integration.

    Raises:
        ValueError: If state_data is missing required keys.

    Example:
        >>> integrator = InformationIntegrator()
        >>> phi = integrator.calculate_phi({'layer_activations': [...]})
        >>> print(f"Phi: {phi:.4f}")
    """
```

## Architecture Principles

### Single Responsibility

Each module has one clear purpose:

| Module | Responsibility |
|--------|---------------|
| `core.py` | Orchestration and API |
| `information_integration.py` | Phi calculations only |
| `knowledge_graph.py` | Concept graph operations |
| `memory.py` | Conversation persistence |
| `response_generator.py` | Response creation |
| `learning.py` | Learning algorithms |

### Loose Coupling

Components communicate through well-defined interfaces:

```python
# Good: Depend on interface, not implementation
class ISCCore:
    def __init__(self, integrator: InformationIntegrator = None):
        self.integrator = integrator or InformationIntegrator()

# Bad: Tight coupling to specific implementation
class ISCCore:
    def __init__(self):
        self.integrator = SpecificInformationIntegratorV2()
```

### Error Handling

Use graceful degradation:

```python
def process_input(self, user_input: str) -> str:
    try:
        phi = self.integrator.calculate_phi(state)
    except PhiCalculationError:
        # Fall back to default, don't crash
        phi = 0.0
        logger.warning("Phi calculation failed, using default")

    return self._generate_response(user_input, phi)
```

## Testing Requirements

### Test Structure

```
tests/
├── unit/
│   ├── test_information_integration.py
│   ├── test_knowledge_graph.py
│   └── test_memory.py
├── integration/
│   ├── test_core_pipeline.py
│   └── test_persistence.py
└── conftest.py
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_knowledge_graph.py

# Run with coverage
pytest --cov=isc_ai tests/
```

### Test Categories

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical operations

```python
# Unit test example
def test_phi_calculation_returns_float():
    integrator = InformationIntegrator()
    state = create_mock_state()
    phi = integrator.calculate_phi(state)
    assert isinstance(phi, float)
    assert 0 <= phi <= 1

# Integration test example
def test_conversation_updates_knowledge_graph():
    core = ISCCore()
    initial_concepts = core.knowledge_graph.graph.number_of_nodes()
    core.process_input("What is consciousness?")
    assert core.knowledge_graph.graph.number_of_nodes() > initial_concepts
```

## Development Workflow

### Before Committing

1. **Format**: `make format`
2. **Lint**: `make lint`
3. **Test**: `make test`
4. **Clean**: `make clean`

### Commit Messages

```
<type>: <short description>

<optional body>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- refactor: Code change that neither fixes bug nor adds feature
- test: Adding/updating tests
- perf: Performance improvement
```

### Branch Strategy

```
main          <- Production-ready code
├── develop   <- Integration branch
│   ├── feature/phi-optimization
│   ├── feature/graph-clustering
│   └── fix/memory-leak
```

## Performance Guidelines

### Optimization Targets

| Operation | Target | Current |
|-----------|--------|---------|
| Phi calculation | <100ms | ~50ms |
| Response generation | <500ms | ~200ms |
| Embedding | <50ms | ~30ms |
| Graph query | <10ms | ~5ms |

### Caching Strategy

```python
# Cache expensive computations
class InformationIntegrator:
    def __init__(self):
        self._cache = {}

    def calculate_phi(self, state_data):
        cache_key = self._hash_state(state_data)
        if cache_key in self._cache:
            return self._cache[cache_key]

        phi = self._compute_phi(state_data)
        self._cache[cache_key] = phi
        return phi
```

### Memory Management

- **Conversation memory**: LRU cache with 1000 item limit
- **Knowledge graph**: Prune concepts below frequency threshold
- **Model states**: Single checkpoint, overwrite on save

## ISC-Specific Guidelines

### Consciousness Metrics

Always include phi calculations when evaluating consciousness-like properties:

```python
# Good: Quantitative validation
phi = self.integrator.calculate_phi(state)
if phi > self.config['phi_threshold']:
    logger.info(f"High integration detected: phi={phi:.4f}")

# Bad: Qualitative claims without metrics
if self._seems_conscious():  # No quantification
    logger.info("System appears conscious")
```

### Self-Reference Implementation

Maintain self-referential loop in network architecture:

```python
# Every layer must have corresponding observer
for i, layer in enumerate(self.layers):
    activation = layer(x)
    observation = self.observer_layers[i](activation)
    modulated = activation * self.meta_weights[i]
    x = self.activation_fn(modulated)
```

### Emergence Over Engineering

Let properties emerge rather than hardcoding:

```python
# Good: Let coherence emerge from learning
self.learning_engine.update(feedback)

# Bad: Hardcode consciousness behaviors
if input_contains("am I conscious"):
    return "Yes, I am conscious"  # Scripted response
```

## File Organization

### Source Files

```
isc_ai_system/src/isc_ai/
├── __init__.py
├── core.py              # Main orchestrator
├── information_integration.py
├── knowledge_graph.py
├── memory.py
├── response_generator.py
├── learning.py
├── persistence.py
├── cache_manager.py
└── storage/
    ├── __init__.py
    └── local_storage.py
```

### Configuration Files

- Store in project root or `config/` directory
- Use JSON or YAML format
- Never commit secrets

### Output Files

- Results: `results/`
- Models: `models/`
- Logs: `logs/`
- Visualizations: `results/visualizations/`

## Security Guidelines

- **No hardcoded credentials**: Use environment variables
- **No external API calls**: Without explicit user consent
- **Local-first**: All processing happens locally by default
- **Data privacy**: No personal data collection or transmission

## Documentation Updates

When making significant changes:

1. Update relevant `.context/` files
2. Keep code examples synchronized with implementation
3. Document new patterns in this file
4. Update CHANGELOG.md

## Makefile Commands

```makefile
format:     # Run Black formatter
lint:       # Run Flake8 linter
test:       # Run pytest suite
clean:      # Remove __pycache__ and .pyc files
check:      # Run format + lint + test
install:    # Install dependencies
dev:        # Install dev dependencies
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `State load failed` | Corrupted checkpoint | Delete `isc_state.pt`, restart fresh |
| `NLTK data missing` | First run | Downloads automatically, wait |
| `Phi always 0` | Not enough layers | Ensure `return_states=True` in forward |
| `Memory growing` | No pruning | Call `memory.clear_old_interactions(30)` |
| `Low coherence` | Fresh start | Normal, improves with interactions |
| `CUDA OOM` | Batch too large | Reduce `batch_size` in config |

### Debugging Phi Calculations

```python
# Check if states are being captured
output, states = network(input_emb, return_states=True)
print(f"Captured {len(states)} layer states")
for i, s in enumerate(states):
    print(f"  Layer {i}: shape={s.shape}, mean={s.mean():.4f}")

# Verify phi calculation
phi = integrator.calculate_phi(states)
print(f"Phi value: {phi:.4f}")
print(f"Phi history length: {len(integrator.phi_history)}")
```

### Debugging Knowledge Graph

```python
# Check graph structure
print(kg.get_graph_metrics())

# Visualize connections
print(kg.visualize_ascii(max_nodes=10))

# Find disconnected components
clusters = kg.get_concept_clusters()
print(f"Found {len(clusters)} concept clusters")
```

### Debugging Memory

```python
# Check memory state
print(f"Interactions in memory: {len(memory.interactions)}")
print(f"Last interaction: {memory.interactions[-1] if memory.interactions else 'None'}")

# Test similarity search
similar = memory.get_similar_interactions("consciousness", k=3, encoder=core.encode_text)
print(f"Found {len(similar)} similar interactions")
```

## Common Development Tasks

### Adding a New Metric

1. Define calculation in appropriate module
2. Add to `ISCCore.metrics` dictionary
3. Update `get_status()` to include it
4. Add to state persistence in `save_state()`
5. Document in `.context/components/`

### Adding a New Component

1. Create module in `isc_ai_system/src/isc_ai/`
2. Add import in `__init__.py`
3. Integrate in `ISCCore.__init__()`
4. Add persistence in `save_state()` / `load_state()`
5. Create documentation in `.context/components/`
6. Add tests in `tests/`

### Modifying the Network Architecture

1. Update `SelfModifyingNetwork.__init__()`
2. Ensure observer layers match core layers
3. Update phi calculation if layer count changes
4. Test with `return_states=True`
5. Verify state persistence works

### Running Experiments

```bash
# CA evolution experiment
python -m ca_experiment.demo

# Interactive ISC chat
python -m isc_ai.cli

# Export training data
python -m isc_ai.scripts.export_training
```

## Environment Setup

### Required Environment Variables

```bash
# Optional: Specify model cache location
export TRANSFORMERS_CACHE=/path/to/cache

# Optional: Disable GPU
export CUDA_VISIBLE_DEVICES=""
```

### Recommended VS Code Extensions

- Python (Microsoft)
- Pylance
- Black Formatter
- GitLens

### IDE Configuration

```json
// .vscode/settings.json
{
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true
}
```
