# System Architecture

## High-Level Overview

The ISC project consists of two complementary systems exploring consciousness emergence through different approaches:

1. **Cellular Automata Self-Modeling System** (`ca_experiment/`)
2. **ISC AI System** (`isc_ai_system/`)

## Directory Structure

```
information-substrate-convergence/
├── ca_experiment/           # Cellular automata evolutionary experiments
│   ├── ca/                  # Core CA simulation modules
│   │   ├── simulation.py    # CA rule execution and state management
│   │   ├── evolution.py     # Genetic algorithm for rule evolution
│   │   ├── analysis.py      # Self-modeling capability analysis
│   │   └── visualize.py     # Pattern visualization and plotting
│   ├── tests/               # Unit tests for CA components
│   └── demo.py              # Main experiment runner
├── isc_ai_system/           # Interactive AI system based on ISC
│   ├── src/isc_ai/          # Core AI system modules
│   │   ├── core.py          # Main ISC system orchestrator
│   │   ├── information_integration.py  # Phi calculation engine
│   │   ├── knowledge_graph.py          # Dynamic graph management
│   │   ├── memory.py        # Conversation memory system
│   │   ├── response_generator.py       # Neural response generation
│   │   ├── persistence.py   # State serialization/deserialization
│   │   ├── cache_manager.py # Phi calculation optimization
│   │   └── storage/         # Local storage and querying
│   ├── scripts/             # Training and utility scripts
│   └── examples/            # Usage examples and demos
├── results/                 # Experiment outputs and visualizations
├── training_results/        # Model training logs and metrics
└── .context/                # Context method documentation
```

## Core Components

### 1. ISC AI System Architecture

#### Core Module (`core.py`)
- **ISCCore**: Main orchestrator class
- Coordinates information flow between components
- Manages system state and metrics
- Provides unified API for external interactions

#### Information Integration (`information_integration.py`)
- **Phi calculation engine**: Implements IIT-based consciousness measurement
- **Optimization**: Caching and approximation for large networks
- **Real-time monitoring**: Continuous phi tracking during conversations

#### Knowledge Graph (`knowledge_graph.py`)
- **Dynamic graph construction**: Builds semantic relationships from conversations
- **Node management**: Concepts, entities, and relationships
- **Graph queries**: Pathfinding, centrality, and neighbor analysis
- **NetworkX backend**: Leverages established graph algorithms

#### Memory System (`memory.py`)
- **Conversation persistence**: SQLite-based storage
- **Semantic retrieval**: Similarity-based memory access
- **Context integration**: Historical conversation influence on responses

#### Response Generation (`response_generator.py`)
- **Self-modifying neural networks**: Observer layers for self-monitoring
- **Contextual processing**: Integration of memory and current input
- **Coherence tracking**: Response consistency measurement

### 2. Cellular Automata System Architecture

#### Simulation Engine (`simulation.py`)
- **CA state management**: Grid-based cellular automaton execution
- **Rule application**: Configurable neighborhood and update rules
- **Pattern tracking**: State evolution and cycle detection

#### Evolution System (`evolution.py`)
- **Genetic algorithm**: Population-based rule evolution
- **Fitness evaluation**: Self-modeling capability assessment
- **Selection pressure**: Convergence toward self-referential patterns

#### Analysis Module (`analysis.py`)
- **Self-modeling detection**: Pattern analysis for self-reference
- **Convergence metrics**: Quantitative measures of emergence
- **Statistical evaluation**: Population-level trend analysis

## Data Flow

### ISC AI System
```
User Input → Core → Memory Retrieval → Knowledge Graph Update →
Response Generation → Information Integration (Phi) → Output + State Update
```

### CA Experiment
```
Initial Population → Evolution Loop → Fitness Evaluation →
Selection → Mutation/Crossover → Analysis → Results Export
```

## Storage Architecture

### Local SQLite Database
- **Conversations**: Message history and metadata
- **Knowledge graphs**: Serialized graph states and versions
- **System metrics**: Phi values, coherence scores, learning rates

### File-Based Persistence
- **Model states**: PyTorch serialized neural networks
- **Configuration**: JSON-based system parameters
- **Results**: CSV exports and visualization outputs

## Integration Points

- **Shared concepts**: Both systems explore self-reference and emergence
- **Complementary approaches**: Rule-based (CA) vs. neural (AI) systems
- **Unified metrics**: Information integration measures across both domains
- **Cross-validation**: Findings from one system inform the other