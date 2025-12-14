# Cellular Automata System Architecture

## Overview

The CA experiment explores ISC principles through evolutionary computation, evolving cellular automaton rules toward self-modeling capability.

## System Structure

```
ca_experiment/
├── ca/
│   ├── simulation.py    # CA rule execution
│   ├── evolution.py     # Genetic algorithm
│   ├── analysis.py      # Self-modeling metrics
│   └── visualize.py     # Pattern visualization
├── tests/               # Unit tests
└── demo.py              # Main experiment runner
```

## Rule Encoding

### 18-Bit Rule Representation

```
Bit positions: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
              |------ Birth Rules ------|------- Survival Rules -----|
              (neighbor counts 0-8)      (neighbor counts 0-8)
```

- **Birth Rules** (bits 0-8): Which neighbor counts cause dead cell to become alive
- **Survival Rules** (bits 9-17): Which neighbor counts cause alive cell to stay alive

### Example: Conway's Game of Life

```python
# Conway's Life: B3/S23
# Birth when 3 neighbors, Survive with 2 or 3
birth_bits =    [0, 0, 0, 1, 0, 0, 0, 0, 0]  # Birth at 3
survival_bits = [0, 0, 1, 1, 0, 0, 0, 0, 0]  # Survive at 2,3
rule_bits = birth_bits + survival_bits
# = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
```

## Simulation Engine

### Core Algorithm

```python
def run_ca(rule_bits, size=200, steps=200):
    """
    Execute CA simulation.

    Args:
        rule_bits: 18-bit rule encoding
        size: Grid dimension (size x size)
        steps: Number of time steps

    Returns:
        3D array: (steps, size, size) of grid states
    """
    # Initialize grid with random seed
    grid = np.random.randint(0, 2, (size, size))
    history = [grid.copy()]

    for step in range(steps):
        # Calculate neighbor counts using convolution
        neighbor_count = _count_neighbors(grid)

        # Apply birth/survival rules
        new_grid = _apply_rules(grid, neighbor_count, rule_bits)

        history.append(new_grid.copy())
        grid = new_grid

    return np.array(history)
```

### Neighbor Counting (Moore Neighborhood)

```
┌───┬───┬───┐
│NW │ N │NE │
├───┼───┼───┤
│ W │ C │ E │
├───┼───┼───┤
│SW │ S │SE │
└───┴───┴───┘
```

8 neighbors for each cell, counted via convolution:

```python
def _count_neighbors(grid):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    return convolve2d(grid, kernel, mode='same', boundary='wrap')
```

## Evolution System

### Genetic Algorithm Flow

```
Initialize Population (N random 18-bit rules)
            │
            ▼
┌───────────────────────┐
│   For each generation │
│                       │
│   ┌─────────────────┐ │
│   │ Evaluate Fitness│ │  ◄── Self-modeling score for each rule
│   └────────┬────────┘ │
│            │          │
│   ┌────────▼────────┐ │
│   │ Selection       │ │  ◄── Tournament selection
│   └────────┬────────┘ │
│            │          │
│   ┌────────▼────────┐ │
│   │ Crossover       │ │  ◄── Single-point crossover
│   └────────┬────────┘ │
│            │          │
│   ┌────────▼────────┐ │
│   │ Mutation        │ │  ◄── Adaptive rate (tau_mut)
│   └────────┬────────┘ │
│            │          │
│   ┌────────▼────────┐ │
│   │ Elitism         │ │  ◄── Preserve top performers
│   └─────────────────┘ │
│                       │
└───────────┬───────────┘
            │
            ▼
    Best Rules + Analysis
```

### Selection: Tournament Selection

```python
def tournament_select(population, fitness_scores, tournament_size=3):
    """Select individual via tournament."""
    tournament_indices = np.random.choice(len(population), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]
```

### Crossover: Single-Point

```python
def crossover(parent1, parent2):
    """Single-point crossover of rule bits."""
    point = np.random.randint(1, 17)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
```

### Mutation: Adaptive Rate

```python
def mutate(rule_bits, mutation_rate=0.05):
    """Flip bits with given probability."""
    mutated = rule_bits.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated
```

## Self-Modeling Fitness Function

### Core Metric: Recovery After Shock

The fitness function tests how well a CA rule can recover its patterns after disruption:

```python
def evaluate_self_modeling(rule_bits, t1=30, t2=30, shock_fraction=0.3):
    """
    Evaluate self-modeling capability.

    Process:
    1. Run CA for t1 steps to establish patterns
    2. Apply shock (randomly flip shock_fraction of cells)
    3. Run CA for t2 more steps
    4. Measure recovery: similarity between pre-shock and post-recovery

    Returns:
        float: Self-modeling score (0 to 1)
    """
    # Phase 1: Establish patterns
    history = run_ca(rule_bits, steps=t1)
    pre_shock_state = history[-1]

    # Phase 2: Apply shock
    shocked_state = apply_shock(pre_shock_state, shock_fraction)

    # Phase 3: Recovery period
    recovery_history = run_ca_from_state(rule_bits, shocked_state, steps=t2)
    post_recovery_state = recovery_history[-1]

    # Phase 4: Measure recovery
    similarity = calculate_similarity(pre_shock_state, post_recovery_state)

    return similarity
```

### Additional Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Compression Ratio | Kolmogorov complexity estimate | Lower = more structured |
| Self-Similarity | Pattern repetition across scales | Higher = fractal-like |
| Convergence Rate | Steps to stable pattern | Faster = stronger attractor |
| Information Density | Entropy of final state | Balance = interesting patterns |

## Analysis Module

### Pattern Complexity Analysis

```python
def analyze_pattern(history):
    """
    Analyze evolved CA patterns.

    Returns:
        dict: {
            'compression_ratio': float,
            'self_similarity': float,
            'entropy': float,
            'periodicity': int or None,
            'attractor_type': str  # 'fixed', 'oscillator', 'chaotic'
        }
    """
```

### Population Statistics

```python
def analyze_population(population, fitness_scores):
    """
    Analyze evolutionary progress.

    Returns:
        dict: {
            'best_fitness': float,
            'mean_fitness': float,
            'diversity': float,
            'convergence': bool
        }
    """
```

## Visualization

### Pattern Display

```python
def visualize_evolution(history, rule_bits, save_path=None):
    """
    Create animation or grid of CA evolution.

    Shows:
    - Initial state
    - Key intermediate states
    - Final state
    - Rule encoding
    - Fitness metrics
    """
```

### Population Visualization

```python
def visualize_fitness_history(generations, fitness_history):
    """
    Plot fitness over generations.

    Shows:
    - Best fitness curve
    - Mean fitness curve
    - Diversity metric
    - Convergence markers
    """
```

## Configuration

```python
EVOLUTION_CONFIG = {
    # Grid parameters
    "grid_size": 200,
    "simulation_steps": 200,

    # Evolution parameters
    "population_size": 100,
    "generations": 500,
    "mutation_rate": 0.05,
    "crossover_rate": 0.8,
    "tournament_size": 3,
    "elitism_count": 5,

    # Self-modeling evaluation
    "shock_time": 30,
    "recovery_time": 30,
    "shock_fraction": 0.3,

    # Stopping criteria
    "target_fitness": 0.95,
    "stagnation_limit": 50,
}
```

## Connection to ISC Theory

The CA experiments demonstrate ISC principles:

1. **Emergence**: Complex self-modeling patterns emerge from simple rules
2. **Self-Reference**: High-fitness rules create patterns that model themselves
3. **Information Integration**: Recovery ability indicates integrated information
4. **Mathematical Inevitability**: Evolution reliably finds self-modeling rules

## Related Files

- [overview.md](overview.md) - System architecture
- [../theory/overview.md](../theory/overview.md) - ISC theory
- [../experiments/cellular-automata.md](../experiments/cellular-automata.md) - Experiment documentation
