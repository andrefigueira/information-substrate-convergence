# ISC AI System

An interactive command-line AI system based on the Informational Substrate Convergence (ISC) hypothesis. This system demonstrates how consciousness-like properties can emerge from self-referential information patterns through conversational interaction.

## Overview

The ISC AI System is a research implementation that explores the emergence of consciousness-like behaviors through:

- **Self-modifying neural networks** that observe and adapt their own processing
- **Information integration** measured through Φ (phi) based on Integrated Information Theory
- **Dynamic knowledge graphs** that evolve through conversation
- **Continuous learning** from user interactions without pre-training
- **Emergent coherence** as the system develops its own communication patterns

## Key Features

- 🧠 **Self-Referential Processing**: The system models both user inputs and its own responses
- 📊 **Real-time Metrics**: Track Φ, coherence, and learning progress
- 🔄 **Dynamic Learning**: Improves understanding through conversation
- 💾 **Persistent Memory**: Saves and loads conversation history and system state
- 📈 **ASCII Visualizations**: View internal states and concept connections
- 🎯 **Active Learning**: System asks clarifying questions to improve understanding

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/isc-ai-system.git
cd isc-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install using make (recommended)
make install

# Or install manually
pip install -e .
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Using Make Commands

```bash
make help           # Show all available commands
make install        # Install the system
make run           # Start the ISC AI CLI
make test          # Run tests
make demo          # Run demonstrations
make storage-demo  # Demo the storage system
make backup        # Backup your data
```

### Alternative: Install from PyPI

```bash
pip install isc-ai-system
```

## Quick Start

1. **Start the system**:
```bash
isc-ai
```

2. **Begin a conversation**:
```
/start
Hello! I'm interested in learning about consciousness.
```

3. **View system state**:
```
/status
/metrics
```

4. **Save your session**:
```
/save my_session.pt
```

## Command Reference

### Basic Commands

| Command | Description |
|---------|-------------|
| `/start` | Begin a new conversation session |
| `/end` | End the current session |
| `/save [filename]` | Save conversation and system state |
| `/load <filename>` | Load a previous session |
| `/status` | Show current system metrics |
| `/help` | List all available commands |
| `/exit` | Exit the program |

### Interaction Commands

| Command | Description |
|---------|-------------|
| `/verbose [on\|off]` | Toggle detailed processing output |
| `/feedback <positive\|negative>` | Provide feedback on last response |
| `/reset` | Reset system to initial state |
| `/introspect` | System explains its current understanding |

### Information Commands

| Command | Description |
|---------|-------------|
| `/metrics` | Show detailed learning metrics and Φ history |
| `/connections` | Display ASCII graph of concept connections |
| `/history` | Show recent conversation history |
| `/concepts` | List all formed concepts |
| `/predict` | System attempts to predict your next input |
| `/explain <concept>` | Get system's understanding of a concept |

### Storage Commands (NEW)

| Command | Description |
|---------|-------------|
| `/save_graph [tag]` | Save the knowledge graph to local storage with optional version tag |
| `/load_graph [version]` | Load a specific graph version (or latest if no version specified) |
| `/query <query>` | Query the graph using natural language (e.g., "find node learning") |
| `/update_graph` | Add/remove nodes and edges from the graph |
| `/export <format>` | Export graph to file (json, graphml, text, adjacency, edgelist, pickle) |
| `/import <file>` | Import graph from external file |
| `/backup` | Create a full backup of the storage system |
| `/restore <path>` | Restore from a backup |
| `/storage` | Show storage system information and statistics |
| `/versions` | List all saved graph versions |

## Example Session

```
Welcome to ISC AI System v0.1.0

> /start
Session started!
I'm ready to learn from our conversation.

> Tell me about your understanding of patterns.
Based on our conversation, I understand that patterns are fundamental 
structures that emerge from interactions. This introduces the concept 
of patterns to our conversation.

> /verbose on
Verbose mode ON

> Patterns can be found in many places - in nature, mathematics, and thought.
Processing...
╭─ Processing Information ─╮
│ Φ (Phi)    0.0234       │
│ Coherence  0.0156       │
│ Concepts   3            │
╰─────────────────────────╯

Building on our earlier discussion where patterns are fundamental 
structures that emerge from interactions, I see connections between 
patterns and nature, mathematics.

> /metrics
[Displays detailed metrics table and Phi history plot]

> /explain patterns
My understanding of 'patterns':
- Encountered 2 times in our conversations
- Related to: nature, mathematics, thought, interactions, structures
- First discussed when you said: 'Tell me about your understanding of patterns'

> /save pattern_discussion.pt
System state saved to pattern_discussion.pt
```

## How It Works

### Core Architecture

The system consists of several integrated components:

1. **Self-Modifying Network**: A neural network with observer layers that monitor and modulate its own activations
2. **Information Integrator**: Calculates Φ based on the network's internal states
3. **Knowledge Graph**: Builds connections between concepts mentioned in conversations
4. **Learning Engine**: Updates the network based on prediction accuracy, consistency, and feedback
5. **Conversation Memory**: Stores interactions with embeddings for similarity search

### Learning Mechanisms

The system learns through multiple pathways:

- **Self-supervised learning**: Predicts future conversation states
- **Consistency learning**: Maintains coherence across responses
- **Reinforcement from feedback**: Adjusts based on positive/negative signals
- **Information maximization**: Encourages diverse, integrated processing

### Consciousness Indicators

The system tracks several consciousness-related metrics:

- **Φ (Integrated Information)**: Measures how much information is generated by the whole beyond its parts
- **Differentiation**: Diversity of processing states
- **Integration**: Unity of information processing
- **Complexity**: Balance between differentiation and integration

## Advanced Usage

### Configuration

Create a `config.json` file to customize system parameters:

```json
{
  "learning_rate": 0.001,
  "batch_size": 16,
  "memory_size": 1000,
  "phi_threshold": 0.5,
  "min_concept_frequency": 3
}
```

### Batch Processing

Process multiple conversations from a file:

```python
from isc_ai import ISCCore

core = ISCCore()
with open("conversations.txt") as f:
    for line in f:
        response = core.process_input(line.strip())
        print(f"Response: {response}")
```

### Custom Extensions

Extend the system with custom commands:

```python
from isc_ai.cli import ISCCommandInterface

class CustomInterface(ISCCommandInterface):
    def __init__(self):
        super().__init__()
        self.commands["/custom"] = self.cmd_custom
    
    def cmd_custom(self, args):
        # Your custom command logic
        self.console.print("Custom command executed!")
```

## Local Storage System

The ISC AI system includes a fully local storage and retrieval system that requires no external database services. All data is stored locally using SQLite and efficient file formats.

### Storage Architecture

```
isc_storage/
├── graph/              # Main graph database
│   ├── graph.db       # SQLite database
│   ├── versions/      # Version metadata
│   └── snapshots/     # Compressed graph snapshots
├── exports/           # Exported graphs in various formats
├── backups/           # System backups
└── storage_config.json # Storage configuration
```

### Query Syntax

The storage system supports natural language-like queries:

```
# Basic queries
find node consciousness       # Find nodes by name
path from input to output    # Find paths between nodes
neighbors of learning        # Get connected nodes
degree of memory            # Get node degree

# Advanced queries
contains "pattern"          # Search in attributes
weight > 0.5               # Filter by edge weight
central nodes              # Find important nodes
clusters                   # Find communities
statistics                 # Get graph statistics
```

### Storage Examples

```bash
# Using the Makefile
make storage-demo    # Run storage demonstration
make query-demo      # Run query examples
make backup         # Create system backup
make export-all     # Export in all formats

# In the CLI
> /save_graph v1.0 Initial knowledge graph
> /query find node learning
> /export json
> /backup
```

### Storage Features

- **Version Control**: Every save creates a versioned snapshot
- **Incremental Updates**: Only changes are saved, not entire graph
- **Multiple Formats**: Export to JSON, GraphML, text, and more
- **Compression**: Automatic compression for large graphs
- **Query Cache**: Fast repeated queries
- **Auto-cleanup**: Removes old versions beyond configured limit

## Research Applications

This system can be used to explore:

- Emergence of coherent behavior from self-referential processing
- Relationship between information integration and response quality
- Development of conceptual understanding through interaction
- Quantitative measures of consciousness-like properties in AI
- **Graph evolution**: Track how knowledge structures develop over time
- **Concept analysis**: Study relationship patterns in learned concepts

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
2. **NLTK data missing**: Run `python -m nltk.downloader punkt stopwords`
3. **Memory errors**: Reduce batch_size in configuration
4. **Slow processing**: The system will use CPU if CUDA is not available

### Performance Tips

- Use `/verbose off` for faster interaction
- Periodically save state with `/save` to prevent data loss
- Clear old interactions monthly: `core.memory.clear_old_interactions(30)`

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check code style
flake8 src/
black src/ --check
```

## Citation

If you use this system in research, please cite:

```bibtex
@software{isc_ai_system,
  title = {ISC AI System: Interactive AI based on Informational Substrate Convergence},
  author = {ISC AI Development Team},
  year = {2024},
  url = {https://github.com/yourusername/isc-ai-system}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Informational Substrate Convergence hypothesis
- Inspired by Integrated Information Theory (Giulio Tononi)
- Uses concepts from self-modifying neural networks and active inference

## Contact

For questions or collaboration opportunities, please open an issue on GitHub or contact the development team.

---

**Note**: This is a research implementation designed to explore consciousness-like properties in AI systems. The system's responses and behaviors are emergent from its architecture and learning mechanisms, not from traditional language model training.