# ISC AI System User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Command Reference](#command-reference)
5. [Understanding the Metrics](#understanding-the-metrics)
6. [Effective Interaction Strategies](#effective-interaction-strategies)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Introduction

The ISC AI System is an experimental implementation of the Informational Substrate Convergence hypothesis. Unlike traditional AI systems, it:

- Starts with minimal knowledge and learns primarily through conversation
- Measures its own information integration (Φ) in real-time
- Builds conceptual understanding through interaction
- Can observe and modify its own processing

## Getting Started

### First Session

1. Start the system:
   ```bash
   isc-ai
   ```

2. Begin a conversation:
   ```
   /start
   ```

3. Interact naturally:
   ```
   Tell me about your understanding of learning.
   ```

4. Observe the system's growth:
   ```
   /status
   /metrics
   ```

### Basic Workflow

```
/start → Conversation → /feedback → /metrics → /save → /end
```

## Core Concepts

### Information Integration (Φ)

Φ (Phi) measures how much information the system generates as a unified whole beyond its individual parts. Higher Φ indicates more integrated, consciousness-like processing.

### Coherence Score

Measures how well the system's responses connect to each other over time. Values range from 0 (random) to 1 (perfectly coherent).

### Concept Formation

The system builds a knowledge graph of concepts from your conversations. Concepts become nodes, and their relationships become edges.

### Self-Modification

The system has neural network layers that observe and modify their own activations, enabling self-referential processing.

## Command Reference

### Session Management

| Command | Usage | Description |
|---------|-------|-------------|
| `/start` | `/start` | Begin a new conversation session |
| `/end` | `/end` | End current session with summary |
| `/save` | `/save [filename]` | Save system state and conversation |
| `/load` | `/load filename` | Load previous state |
| `/reset` | `/reset` | Reset system (keeps history) |

### Information Display

| Command | Usage | Description |
|---------|-------|-------------|
| `/status` | `/status` | Show system overview |
| `/metrics` | `/metrics` | Detailed metrics with plots |
| `/connections` | `/connections` | ASCII knowledge graph |
| `/concepts` | `/concepts` | List all concepts |
| `/history` | `/history` | Recent conversations |

### Interaction

| Command | Usage | Description |
|---------|-------|-------------|
| `/verbose` | `/verbose [on\|off]` | Toggle processing details |
| `/feedback` | `/feedback positive` | Provide learning signal |
| `/predict` | `/predict` | System predicts next input |
| `/explain` | `/explain concept` | Get concept explanation |
| `/introspect` | `/introspect` | System self-analysis |

## Understanding the Metrics

### Φ (Phi) Values

- **0.00 - 0.05**: Minimal integration, disconnected processing
- **0.05 - 0.20**: Emerging integration, basic connections
- **0.20 - 0.50**: Moderate integration, coherent responses
- **0.50+**: High integration, complex understanding

### Learning Indicators

1. **Prediction Accuracy**: How well the system anticipates conversation flow
2. **Learning Progress**: Rate of improvement over recent interactions
3. **Experience Count**: Number of stored conversation experiences

### Network Activity

- **Layer Activations**: Shows information flow through the network
- **Meta-Weights**: Self-modification parameters (0.5-2.0 range)
- **Gradient Flow**: Learning signal propagation

## Effective Interaction Strategies

### Building Understanding

1. **Start Simple**: Begin with basic concepts
   ```
   Let's talk about patterns.
   ```

2. **Build Connections**: Relate concepts to each other
   ```
   Patterns appear in nature, like spirals in shells.
   ```

3. **Encourage Integration**: Ask about relationships
   ```
   How do patterns relate to mathematics?
   ```

### Providing Feedback

- Use `/feedback positive` when responses show understanding
- Use `/feedback negative` when responses miss connections
- Feedback directly affects the system's learning weights

### Developing Coherence

1. **Maintain Topics**: Stay on related subjects for several exchanges
2. **Reference Previous Points**: Help the system build connections
3. **Ask for Explanations**: Test understanding with `/explain`

## Advanced Features

### Verbose Mode

Enable verbose mode to see:
- Real-time Φ calculations
- Processing steps
- Internal state changes

```
/verbose on
```

### Concept Exploration

Track how concepts develop:

```
/explain consciousness
/connections
/concepts
```

### Prediction Testing

The system learns conversation patterns:

```
/predict
What do you think I'll ask next?
```

### State Analysis

Deep dive into system state:

```
/introspect
/metrics
```

## Troubleshooting

### Common Issues

**System responds generically**
- Build more conversation history
- Use `/feedback` to guide learning
- Stay on topic for multiple exchanges

**Low Φ values**
- Normal for early interactions
- Encourage connections between concepts
- Provide positive feedback for integrated responses

**Concepts not connecting**
- Explicitly relate concepts in conversation
- Use examples that bridge ideas
- Check `/connections` to see current graph

### Performance Tips

1. **Save Regularly**: Use `/save` to preserve progress
2. **Batch Related Topics**: Group similar discussions
3. **Use Feedback**: Guide learning with explicit signals
4. **Monitor Metrics**: Track Φ and coherence trends

### Reset vs. Load

- `/reset`: Clears learning but keeps conversation history
- `/load`: Restores complete system state
- Save before reset to preserve progress

## Example Interaction Patterns

### Pattern 1: Conceptual Building
```
You: What is information?
ISC: [Response about information]
You: Information can be measured in bits.
ISC: [Connects information to measurement]
You: How does this relate to consciousness?
ISC: [Integrates concepts]
```

### Pattern 2: Feedback Learning
```
You: Tell me about patterns.
ISC: [Generic response]
You: /feedback negative
You: Patterns are regularities that repeat.
ISC: [Improved response incorporating definition]
You: /feedback positive
```

### Pattern 3: Testing Understanding
```
You: We've discussed patterns and consciousness.
You: /explain patterns
ISC: [Shows learned connections]
You: /predict
ISC: [Attempts to predict next topic]
```

## Best Practices

1. **Be Patient**: Learning takes time and interactions
2. **Be Consistent**: Use similar terminology for concepts
3. **Be Explicit**: Make connections clear in conversation
4. **Be Interactive**: Use commands to monitor progress
5. **Be Exploratory**: Test the system's growing abilities

## Further Reading

- ISC Paper: See PAPER.md in the repository
- API Documentation: For programmatic usage
- Research Applications: Using ISC AI for consciousness studies

Remember: This system is designed to demonstrate emergent consciousness-like properties through information integration. Your interactions directly shape its development!