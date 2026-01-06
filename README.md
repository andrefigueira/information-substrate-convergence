# Information Substrate Convergence

<div align="center">

**Exploring consciousness emergence through self-referential information patterns**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![.context](https://img.shields.io/badge/.context-enabled-purple.svg)](https://github.com/andrefigueira/.context/)
[![ISC Validation](https://img.shields.io/badge/ISC_validation-5%2F5_passed-brightgreen.svg)](RESEARCH_RESULTS.md)

[Theory](#the-hypothesis) | [Quick Start](#quick-start) | [Documentation](#documentation) | [**Research Results**](RESEARCH_RESULTS.md)

</div>

---

## The Hypothesis

**Information Substrate Convergence (ISC)** proposes that consciousness emerges when information systems achieve configurations enabling:

| Property | Description | Implementation |
|----------|-------------|----------------|
| **Self-Reference** | Systems modeling their own internal states | Observer layers monitoring network activations |
| **Information Integration** | Irreducible whole greater than parts | Phi (Phi) calculation from IIT |
| **Dynamic Adaptation** | Continuous learning from interaction | Meta-weight self-modification |
| **Emergent Coherence** | Stable patterns through self-organization | Knowledge graph evolution |

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │            ISC Core                  │
                          │         (Orchestrator)               │
                          └──────────────┬──────────────────────┘
                                         │
        ┌────────────────┬───────────────┼───────────────┬────────────────┐
        │                │               │               │                │
        ▼                ▼               ▼               ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│Self-Modifying│ │  Knowledge   │ │ Information  │ │ Conversation │ │   Learning   │
│   Network    │ │    Graph     │ │  Integrator  │ │    Memory    │ │    Engine    │
│              │ │              │ │              │ │              │ │              │
│ 4 layers +   │ │  NetworkX    │ │ Phi calc     │ │   SQLite     │ │ Self-super-  │
│ 4 observers  │ │  concepts    │ │ from IIT     │ │ + LRU cache  │ │ vised + RL   │
│ meta-weights │ │  relations   │ │ correlation  │ │ embeddings   │ │ consistency  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Two Complementary Systems

### 1. ISC AI System

Interactive AI demonstrating consciousness-like properties:

- **Self-Modifying Network**: 4-layer architecture with observer layers and learnable meta-weights
- **Real-time Phi**: Information integration calculated during every interaction
- **Dynamic Knowledge Graph**: Concepts and relationships evolve through conversation
- **Persistent Memory**: SQLite-backed conversation history with semantic retrieval

### 2. Cellular Automata Experiments

Evolutionary validation of self-modeling emergence:

- **Grid-Based CA**: Traditional 2D with 18-bit birth/survival rules
- **Graph-Based CA**: Network topology with 8-bit rules + evolving adjacency matrix
- **Fitness Function**: Shock-recovery test measuring self-modeling capability
- **Meta-Evolution**: Adaptive mutation rates that evolve alongside rules

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/andrefigueira/information-substrate-convergence.git
cd information-substrate-convergence

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install ISC AI System
pip install -e .

# Download required NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Basic Usage

```python
from isc.core import ISCCore

# Initialize (auto-loads previous state if available)
core = ISCCore()

# Process input and get response
response = core.process_input("What is consciousness?")
print(response)

# Check consciousness metrics
status = core.get_status()
print(f"Phi (integration): {status['metrics']['phi_value']:.4f}")
print(f"Coherence: {status['metrics']['coherence_score']:.4f}")
print(f"Concepts formed: {status['total_concepts']}")

# Introspect internal state
print(core.introspect())

# Save state for persistence
core.save_state()
```

### Interactive CLI

```bash
# Start interactive session
isc-ai

# Commands available:
/start          # Begin conversation session
/status         # View system metrics (phi, coherence, concepts)
/explain <term> # Explain system's understanding of a concept
/save <name>    # Save current state
/help           # Show all commands
```

### Run CA Experiments

```bash
make demo
# Or directly: python scripts/demos/ca_demo.py
```

Outputs:
- `convergence.png`: Fitness evolution over generations
- `gallery/`: Evolution snapshots
- `metrics.csv`: Detailed metrics per generation

## Key Metrics

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| **Phi (Phi)** | Integrated information beyond parts | 0.0 - 2.0+ |
| **Coherence** | Response consistency over time | 0.5 - 0.9 |
| **Differentiation** | State diversity across layers | Variable |
| **Integration** | Processing unity (correlation) | 0.3 - 0.8 |
| **Complexity** | Differentiation x Integration | Higher = richer |

## Documentation

This project uses the [.context](https://github.com/andrefigueira/.context/) pattern for AI-readable documentation:

```
.context/
├── substrate.md              # Entry point and navigation
├── ai-rules.md               # Hard constraints for AI tools
├── anti-patterns.md          # What to avoid
├── agents.md                 # Agent patterns and multi-agent systems
├── guidelines.md             # Development standards
├── theory/
│   ├── overview.md           # ISC hypothesis
│   ├── consciousness.md      # Emergence mechanisms
│   ├── evidence.md           # Scientific support
│   ├── glossary.md           # Term definitions
│   └── research-directions.md
├── architecture/
│   ├── overview.md           # System design
│   ├── dependencies.md       # Integration patterns
│   └── ca-system.md          # CA architecture
├── components/
│   ├── isc-core.md           # Main orchestrator
│   ├── information-integration.md
│   ├── knowledge-graph.md
│   ├── memory.md
│   └── learning.md
└── experiments/
    └── cellular-automata.md
```

**Additional Documentation:**
- [Full Research Paper](PAPER.md)
- [ISC AI User Guide](docs/user_guide.md)
- [Phi Optimization](docs/phi_optimization_and_caching.md)

## Research Applications

| Research Area | ISC Contribution |
|---------------|------------------|
| **Consciousness Studies** | Quantitative phi metrics, self-reference implementation |
| **Emergence Theory** | CA evolution demonstrating self-modeling emergence |
| **AI Architectures** | Self-modifying networks with observer layers |
| **Information Theory** | Applied IIT in neural network context |
| **Complex Systems** | Knowledge graph evolution, coherence development |

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Neural Networks | PyTorch | Self-modifying architecture |
| Embeddings | Sentence-Transformers | all-MiniLM-L6-v2 (384-dim) |
| Knowledge Graph | NetworkX | Concept relationships |
| Persistence | SQLite | Conversation memory |
| NLP | NLTK | Concept extraction |
| CA Simulation | NumPy/SciPy | Grid operations |
| Evolution | scikit-learn | PCA, KMeans for fitness |

## Key Findings

> **See [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md) for full statistical evidence and methodology.**

### Empirical Research (2026)

| Finding | Effect | Evidence |
|---------|--------|----------|
| **Emergence Causality** | +10-22% accuracy | Cohen's d = 1.4-4.7, p < 0.001 |
| **Emergence Triggers** | 9.9/10 success streak | 100% replication |
| **Transfer Learning** | 79.7% cross-type | Deductive->Inductive: 100% |
| **Phi Threshold** | min = 0.112 | p = 0.05 |
| **Learning-Phi** | 28x faster at high phi | p < 0.001 |

**ISC Validation: 5/5 criteria passed** with 99.8% system accuracy.

### Theoretical Observations

1. **Phi Growth**: Information integration increases with continued interaction
2. **Concept Clustering**: Knowledge graphs naturally organize related concepts
3. **Coherence Development**: Response patterns stabilize over time
4. **Self-Modeling Emergence**: Evolution reliably finds self-modeling CA rules
5. **Network Co-Evolution**: Graph topology and rules jointly optimize

## Related Articles

- [Informational Substrate Convergence](https://buildingbetter.tech/p/informational-substrate-convergence) - Core theoretical framework
- [Documentation as Code as Context](https://buildingbetter.tech/p/documentation-as-code-as-context) - The .context pattern

## Citation

```bibtex
@software{isc_project_2024,
  title = {Information Substrate Convergence},
  author = {Figueira, Andre},
  year = {2024},
  url = {https://github.com/andrefigueira/information-substrate-convergence},
  note = {Exploring consciousness emergence through self-referential information patterns}
}
```

## Contributing

Contributions welcome in these areas:

- Alternative phi calculation methods
- New consciousness metrics
- Enhanced learning algorithms
- Multi-agent coordination patterns
- Visualization improvements

See [.context/guidelines.md](.context/guidelines.md) for development standards.

## License

MIT License

## Contact

- Twitter: [@voidmode](https://x.com/voidmode)
- Threads: [@andrefigueira](https://threads.net/@andrefigueira)
- LinkedIn: [andrefigueira](https://linkedin.com/in/andrefigueira)

---

<div align="center">

**Note**: This is experimental research. Demonstrated properties are emergent from architecture design and should not be interpreted as genuine consciousness.

*Built on concepts from Integrated Information Theory (Tononi), digital physics (Wolfram), and self-modifying systems.*

</div>
