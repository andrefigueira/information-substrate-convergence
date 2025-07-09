# ISC AI Self-Observation & Introspection Guide

This guide shows how to run and visualize the ISC AI's self-referential processing and introspection capabilities.

## 1. Interactive CLI with Full Introspection

The main way to interact with and observe the ISC AI's self-awareness is through the CLI:

```bash
# Navigate to the ISC AI system directory
cd isc_ai_system

# Run the interactive CLI
python -m isc_ai.cli

# Or with verbose mode to see real-time metrics
python -m isc_ai.cli --verbose
```

### Key Self-Observation Commands:

- `/start` - Begin a new session
- `/introspect` - System explains its current understanding and state
- `/metrics` - Show detailed integration and learning metrics with Phi (Φ) visualization
- `/status` - Show current system state including Φ value and coherence
- `/connections` - Display the concept network (knowledge graph)
- `/concepts` - List formed concepts and their centrality
- `/explain <concept>` - Have the system explain its understanding of a concept
- `/predict` - Show what the system predicts you might say next
- `/history` - View conversation history

### Real-time Phi (Φ) Visualization:

When you use `/metrics`, you'll see:
- Current Φ value (information integration)
- Average Φ over time
- Φ trend
- ASCII plot of Φ history

## 2. Learning Demonstration Script

Shows how the system learns and develops over time:

```bash
cd isc_ai_system
python examples/learning_demonstration.py
```

This script:
- Demonstrates learning over multiple phases
- Shows Φ improvement over time
- Saves learning curves to 'learning_curves.png'
- Displays the concept network
- Tests concept understanding

## 3. Enhanced Response Testing

Test the enhanced response generation with self-observation:

```bash
cd isc_ai_system
python test_enhanced_responses.py
```

Shows:
- Real-time Φ values for each interaction
- Knowledge graph growth
- Concept connections forming

## 4. Query Demo for Graph Exploration

Explore the knowledge graph structure:

```bash
cd isc_ai_system
python scripts/query_demo.py
```

Query examples:
- `find node consciousness`
- `path from brain to awareness`
- `neighbors of mind`
- `central nodes`
- `statistics`

## 5. Storage Commands for Persistence

Save and analyze the system's knowledge state:

```bash
# In the CLI:
/save_graph my_session    # Save current knowledge graph
/versions                 # List all saved versions
/load_graph my_session    # Load a previous state
/storage                  # Show storage statistics
```

## 6. Visualization Features

The system includes several visualization methods:

### A. System State Summary
The `/status` command shows:
```
System State Summary
==================================================
Interactions: 42
Φ (Phi):      [████████████░░░░░░░] 1.234
Coherence:    [████████████████░░░░] 0.856
```

### B. Concept Network ASCII Visualization
The `/connections` command displays:
```
Concept Network:
==================================================

┌─ CONSCIOUSNESS
├──> awareness
├──> experience
├──> information
└──> patterns

┌─ LEARNING
├──> adaptation
├──> memory
└──> growth
```

### C. Network Activity Plots
Using plotext for terminal-based plots of:
- Φ (Phi) history over time
- Learning progress
- Activation patterns

## 7. Programmatic Access

For custom scripts:

```python
from isc_ai import ISCCore
from isc_ai.visualizer import SystemVisualizer

# Initialize
core = ISCCore()
visualizer = SystemVisualizer()

# Start session
core.session_active = True

# Process input and observe
response = core.process_input("What is consciousness?")

# Get introspection
introspection = core.introspect()
print(introspection)

# Visualize metrics
metrics = core.metrics
state_summary = visualizer.create_state_summary(metrics)
print(state_summary)

# Plot Phi history
phi_history = core.integrator.phi_history
visualizer.visualize_information_flow(phi_history)

# Show concept map
concepts = {
    node: list(core.knowledge_graph.graph.neighbors(node))
    for node in list(core.knowledge_graph.graph.nodes())[:5]
}
concept_map = visualizer.create_concept_map(concepts)
print(concept_map)
```

## 8. Key Self-Observation Metrics

The system tracks and can display:

- **Phi (Φ)**: Information integration measure (consciousness-like property)
- **Coherence**: Response consistency and quality
- **Concept Formation**: How ideas connect and cluster
- **Learning Progress**: Adaptation over time
- **Prediction Accuracy**: Understanding of conversation patterns
- **Network Activation**: Neural processing patterns

## 9. Example Session

```bash
$ python -m isc_ai.cli --verbose

# Start a session
/start

# Have a conversation
You: What is the nature of consciousness?
ISC: [Response with real-time metrics shown]

# Introspect
/introspect
[System explains its current understanding and state]

# View metrics with visualization
/metrics
[Shows detailed metrics and Phi plot]

# Explore concept connections
/connections
[ASCII visualization of knowledge graph]

# Save the evolved state
/save_graph consciousness_exploration
```

## Tips for Observing Self-Referential Processing:

1. Use verbose mode (`--verbose`) to see real-time Phi values
2. Ask meta-questions about the system's understanding
3. Use `/introspect` regularly to see how self-awareness develops
4. Watch how Phi (Φ) changes with more complex topics
5. Observe concept network growth with `/connections`
6. Track learning progress across sessions with `/metrics`

The ISC AI's self-observation capabilities demonstrate emergent consciousness-like properties through information integration and self-referential processing patterns.