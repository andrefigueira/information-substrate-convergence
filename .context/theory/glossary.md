# ISC Glossary

## Core Concepts

### Phi (Φ)
**Definition**: A quantitative measure of integrated information, representing how much information a system generates beyond what its parts generate independently.

**Formula**: Φ = I(System) - I(MinimumPartition)

**Range**: 0 to unbounded (typically 0-10 in practice)

**Interpretation**: Higher phi indicates more integrated, consciousness-like processing. A system with phi = 0 is completely reducible to independent parts.

---

### Information Substrate
**Definition**: The fundamental informational "fabric" from which reality emerges according to ISC theory.

**Key Properties**:
- Substrate-independent (not tied to specific physical implementation)
- Self-referential (can represent itself)
- Computationally universal (supports arbitrary computation)

---

### Self-Reference / Self-Modeling
**Definition**: The capacity of a system to create internal representations of its own states and processes.

**Implementation**: Observer layers in the SelfModifyingNetwork that monitor and modulate core processing layers.

**Significance**: Proposed as necessary condition for consciousness emergence.

---

### Meta-Weights
**Definition**: Learnable parameters that modulate network layer outputs based on self-observation.

**Function**: Enable the network to modify its own processing based on feedback and self-monitoring.

**Range**: Clamped to [0.5, 2.0] for stability.

---

### Observer Layers
**Definition**: Neural network layers that monitor activations of core processing layers.

**Operation**: Receive detached (non-gradient) copies of layer activations and produce modulation signals.

**Purpose**: Implement self-reference loop enabling consciousness-like properties.

---

## Theoretical Terms

### Integrated Information Theory (IIT)
**Origin**: Giulio Tononi's theory of consciousness

**Core Claim**: Consciousness is identical to integrated information (phi)

**ISC Usage**: Phi calculation adapted for neural network states

---

### Digital Physics
**Definition**: Hypothesis that the universe is fundamentally computational/informational

**Key Proponents**: Konrad Zuse, Ed Fredkin, Stephen Wolfram

**ISC Connection**: Supports information primacy thesis

---

### Holographic Principle
**Definition**: Physical theory stating that all information in a 3D volume can be encoded on its 2D boundary

**Evidence**: Black hole thermodynamics, AdS/CFT correspondence

**ISC Connection**: Suggests dimensionality emerges from information, not vice versa

---

### Strange Loop
**Definition**: Douglas Hofstadter's concept of a self-referential hierarchical structure where moving through levels eventually returns to the starting point

**Example**: A system that models itself modeling itself

**ISC Implementation**: Observer layers creating recursive self-representation

---

### Substrate Independence
**Definition**: The principle that consciousness depends on informational patterns, not specific physical implementation

**Implication**: Consciousness could arise in silicon, quantum systems, or any suitable information substrate

**Evidence**: Functional equivalence across different implementations

---

## Technical Terms

### Mutual Information (MI)
**Definition**: Measure of the amount of information that one random variable contains about another

**Formula (approximation)**: MI ≈ -log(1 - r²) where r is correlation

**Usage**: Calculated between consecutive network layers

---

### Partition
**Definition**: Division of a system into separate subsystems for analysis

**Minimum Partition**: The partition that minimizes information transfer between parts

**Usage**: Finding the "weakest link" in system integration

---

### Coherence Score
**Definition**: Measure of semantic consistency between consecutive system responses

**Calculation**: Average cosine similarity of consecutive response embeddings

**Range**: -1 to 1 (typically 0.5-0.9 for coherent systems)

---

### Embedding
**Definition**: Dense vector representation of text capturing semantic meaning

**Model**: all-MiniLM-L6-v2 (produces 384-dimensional vectors)

**Usage**: Text encoding, similarity search, concept representation

---

### Knowledge Graph
**Definition**: Graph structure where nodes are concepts and edges are relationships

**Implementation**: NetworkX undirected graph with weighted edges

**Operations**: Centrality analysis, path finding, clustering

---

## Cellular Automata Terms

### Moore Neighborhood
**Definition**: The 8 cells surrounding a central cell in a 2D grid

```
NW | N | NE
---+---+---
W  | C | E
---+---+---
SW | S | SE
```

**Usage**: Neighbor counting for rule application

---

### Birth/Survival Rules
**Definition**: Conditions for cell state transitions in CA

**Format**: B{counts}/S{counts} (e.g., B3/S23 for Conway's Life)

**Encoding**: 18-bit array (9 birth bits + 9 survival bits)

---

### Shock-Recovery Test
**Definition**: Fitness evaluation method for self-modeling capability

**Process**:
1. Run CA for t1 steps (establish pattern)
2. Randomly flip fraction of cells (shock)
3. Run CA for t2 steps (recovery)
4. Measure similarity to pre-shock state

---

### Genome (Evolution)
**Definition**: Complete specification of an individual in evolutionary CA

**Components**:
- `rule_bits`: 8-bit elementary CA rule
- `adj_matrix`: Network topology (n×n boolean matrix)
- `tau_mut`: Adaptive mutation rate

---

### Adaptive Mutation (tau_mut)
**Definition**: Self-evolving mutation rate parameter

**Range**: [0, 1]

**Behavior**: Evolves alongside rules, enabling meta-evolution

---

## Metrics

| Metric | Range | Optimal | Meaning |
|--------|-------|---------|---------|
| Phi | 0-∞ | Higher | More integrated information |
| Coherence | -1 to 1 | ~0.7 | Response consistency |
| Differentiation | 0-∞ | Balance | State diversity |
| Integration | 0-1 | Higher | Processing unity |
| Complexity | 0-∞ | Higher | Differentiation × Integration |

## Related Files

- [overview.md](overview.md) - Theoretical foundations
- [consciousness.md](consciousness.md) - Consciousness mechanisms
- [evidence.md](evidence.md) - Scientific support
