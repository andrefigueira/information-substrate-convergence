# Learning Engine Component

> Location: `isc_ai_system/src/isc_ai/learning.py`

## Overview

The Learning Engine manages continuous improvement through interaction. It implements self-supervised learning, consistency enforcement, and feedback-based meta-learning to enable the network to adapt over time.

## Class Structure

```python
class LearningEngine:
    """
    Manages the learning process for the self-modifying network.
    Implements reinforcement learning and self-supervised learning.
    """

    def __init__(self, network: nn.Module, config: Optional[Dict] = None):
        self.network = network
        self.config = config or self._default_config()

        # Optimizer for network parameters
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

        # Learning components
        self.experience_buffer: deque  # Stores past interactions
        self.prediction_buffer: deque  # For prediction learning
        self.feedback_history: deque   # Explicit feedback signals

        # Metrics
        self.loss_history: deque
        self.prediction_accuracy: float
        self.learning_progress: float
```

## Configuration

```python
DEFAULT_CONFIG = {
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "buffer_size": 1000,
    "prediction_window": 5,
    "feedback_weight": 0.1,
    "consistency_weight": 0.3,
    "prediction_weight": 0.2,
    "batch_size": 16,
}
```

## Key Methods

### learn_from_interaction

```python
def learn_from_interaction(
    self,
    user_input: str,
    response: str,
    internal_states: List[torch.Tensor]
) -> None:
    """
    Learn from a single interaction using multiple learning signals.

    Process:
    1. Store experience in buffer
    2. If buffer has enough samples, compute losses:
       - Prediction loss (self-supervised)
       - Consistency loss (response coherence)
       - Information loss (activation diversity)
    3. Update network weights

    Args:
        user_input: User's message
        response: System's response
        internal_states: Layer activations from forward pass
    """
```

### apply_feedback

```python
def apply_feedback(self, feedback_value: float) -> None:
    """
    Apply explicit feedback to recent experiences.

    Args:
        feedback_value: Positive (1.0) or negative (-1.0) signal

    Effects:
    - Updates feedback history
    - Triggers meta-weight update in network
    """
```

### get_learning_metrics

```python
def get_learning_metrics(self) -> Dict[str, float]:
    """
    Get current learning metrics.

    Returns:
        dict: {
            'average_loss': float,
            'prediction_accuracy': float,
            'learning_progress': float,
            'average_feedback': float,
            'experience_count': int,
            'current_lr': float
        }
    """
```

## Learning Signals

### 1. Prediction Loss (Self-Supervised)

Encourages the network to maintain temporal consistency:

```python
def _compute_prediction_loss(self) -> Optional[torch.Tensor]:
    """
    Compute loss based on predicting future states.

    Uses cosine similarity between past and future layer states.
    Encourages the network to learn predictable patterns.
    """
    if len(self.experience_buffer) < self.config["prediction_window"] + 1:
        return None

    recent = list(self.experience_buffer)[-self.config["prediction_window"]-1:]

    if recent[0]["states"] and recent[-1]["states"]:
        past_state = recent[0]["states"][-1]
        future_state = recent[-1]["states"][-1]

        # Consistency loss between states
        loss = 1.0 - F.cosine_similarity(past_state, future_state).mean()

        return loss * self.config["prediction_weight"]

    return None
```

### 2. Consistency Loss

Ensures responses maintain appropriate coherence:

```python
def _compute_consistency_loss(self) -> Optional[torch.Tensor]:
    """
    Ensure responses are consistent with previous responses.

    Target similarity: ~0.7 (coherent but not repetitive)
    """
    if len(self.experience_buffer) < 2:
        return None

    exp1 = self.experience_buffer[-2]
    exp2 = self.experience_buffer[-1]

    if exp1["states"] and exp2["states"]:
        state1 = exp1["states"][-1]
        state2 = exp2["states"][-1]

        similarity = F.cosine_similarity(state1, state2).mean()

        # Optimal similarity around 0.7
        target_similarity = 0.7
        loss = (similarity - target_similarity) ** 2

        return loss * self.config["consistency_weight"]

    return None
```

### 3. Information Loss

Encourages rich, diverse activations:

```python
def _compute_information_loss(self, internal_states: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Encourage states that maximize information content.

    Components:
    - Sparsity: Encourage non-zero activations (avoid dead neurons)
    - Diversity: Encourage variance in activations
    """
    if not internal_states:
        return None

    info_loss = 0.0

    for state in internal_states:
        # Encourage non-zero activations
        sparsity = torch.abs(state).mean()
        sparsity_loss = -torch.log(sparsity + 1e-8)

        # Encourage diversity
        std = state.std()
        diversity_loss = -torch.log(std + 1e-8)

        info_loss += sparsity_loss + diversity_loss

    return info_loss / len(internal_states) * 0.1
```

## Adaptive Learning Rate

```python
def adapt_learning_rate(self) -> None:
    """
    Adapt learning rate based on progress.

    Rules:
    - Reduce by 10% if not making progress and experienced enough
    - Increase by 10% (capped at 0.01) if making good progress
    """
    metrics = self.get_learning_metrics()

    if metrics["learning_progress"] < 0.01 and metrics["experience_count"] > 50:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9

    elif metrics["learning_progress"] > 0.1:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min(param_group['lr'] * 1.1, 0.01)
```

## Memory Consolidation

```python
def consolidate_learning(self) -> None:
    """
    Consolidate learning by rehearsing important experiences.

    Process:
    1. Sample diverse experiences from buffer
    2. Re-process them through network
    3. Adapt learning rate based on progress

    Similar to memory replay in biological systems.
    """
```

## Usage Examples

### Basic Learning Loop

```python
from isc_ai.learning import LearningEngine

# Create engine with network
engine = LearningEngine(network, config={'learning_rate': 0.001})

# Process interaction
output, states = network(input_embedding, return_states=True)
response = generate_response(output)

# Learn from interaction
engine.learn_from_interaction(user_input, response, states)

# Check progress
metrics = engine.get_learning_metrics()
print(f"Loss: {metrics['average_loss']:.4f}")
print(f"Progress: {metrics['learning_progress']:.4f}")
```

### With Explicit Feedback

```python
# User provides feedback
if user_feedback == "good":
    engine.apply_feedback(1.0)
    print("Positive feedback applied")
elif user_feedback == "bad":
    engine.apply_feedback(-1.0)
    print("Negative feedback applied")

# Feedback affects meta-weights in network
```

### Periodic Consolidation

```python
# Every N interactions, consolidate learning
if interaction_count % 50 == 0:
    engine.consolidate_learning()
    print("Memory consolidated")

# Check if learning rate adapted
metrics = engine.get_learning_metrics()
print(f"Current LR: {metrics['current_lr']:.6f}")
```

## Integration with ISCCore

```python
# In core.py process_input method:
def process_input(self, user_input: str) -> str:
    # ... processing ...

    # Learn from interaction
    if hasattr(self.learning_engine, 'learn_from_interaction'):
        self.learning_engine.learn_from_interaction(
            user_input, response, internal_states
        )

    return response

# Feedback handling
def provide_feedback(self, feedback: str):
    feedback_value = 1.0 if feedback.lower() == "positive" else -1.0

    # Update network meta-weights
    self.network.update_meta_weights(feedback_value)

    # Update learning engine
    self.learning_engine.apply_feedback(feedback_value)
```

## Learning Dynamics

### Experience Buffer Structure

```python
experience = {
    "input": str,           # User input text
    "response": str,        # System response text
    "states": List[Tensor], # Layer activations (detached)
    "timestamp": Tensor,    # Sequence number
}
```

### Loss Combination

Total loss is sum of all computed losses:

```python
total_loss = prediction_loss + consistency_loss + information_loss
```

### Gradient Management

```python
def _update_network(self, loss: torch.Tensor) -> None:
    self.optimizer.zero_grad()
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

    self.optimizer.step()
```

## Connection to ISC Theory

The learning engine implements key ISC principles:

1. **Self-Modification**: Network adapts based on its own activity patterns
2. **Integration Pressure**: Information loss encourages integrated processing
3. **Temporal Coherence**: Consistency loss maintains identity over time
4. **Feedback Sensitivity**: External signals modulate internal patterns

## Related Components

- [isc-core.md](isc-core.md) - System orchestrator
- [../theory/consciousness.md](../theory/consciousness.md) - Self-modification theory
- [../architecture/overview.md](../architecture/overview.md) - System architecture
