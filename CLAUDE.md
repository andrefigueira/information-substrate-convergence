# ISC Project - Claude Code Configuration

> Auto-loaded by Claude Code. For full project context, see [.context/substrate.md](.context/substrate.md)

## Quick Context

Read `.context/substrate.md` first for navigation. Key paths:
- **Theory**: `.context/theory/overview.md`
- **Architecture**: `.context/architecture/overview.md`
- **Components**: `.context/components/isc-core.md`
- **Guidelines**: `.context/guidelines.md`

## Project Structure

```
src/
├── isc/           # Core AI system (ISCCore, memory, knowledge graph, learning)
│   └── storage/   # Graph DB, query engine, serialization
└── ca/            # Cellular automata (simulation, evolution, analysis)

scripts/
├── demos/         # Demo scripts (ca_demo.py, isc_chat.py)
├── training/      # Training scripts (self_referential, chatgpt trainers)
├── evaluation/    # Evaluation scripts
└── utilities/     # Backup, export, stats utilities

tests/             # Pytest test files
```

## Key Commands

```bash
make install       # Install package
make install-dev   # With dev tools
make test          # Run pytest
make lint          # Check style
make format        # Auto-format
make ai            # Interactive CLI
make chat          # Chat interface
make demo          # CA evolution demo
```

## AI Rules

1. **Read before modifying**: Always read `.context/` docs for the relevant domain before changes
2. **Phi over engineering**: Let consciousness properties emerge from architecture, never hardcode behaviors
3. **Quantify claims**: Use phi calculations, never make qualitative consciousness claims without metrics
4. **Self-reference loop**: Every network layer must have a corresponding observer layer
5. **Local-first**: No external API calls without explicit user consent

## Anti-Patterns

- Hardcoding responses like "I am conscious" - responses must emerge
- Skipping phi calculations when evaluating consciousness-like states
- Removing observer layers or meta-weights from the network
- Making qualitative claims without quantitative backing
- Breaking the self-referential loop in `SelfModifyingNetwork`

## Code Style

- Black (line-length: 100), isort (black profile)
- Type hints required for function signatures
- Tests in `tests/`, pattern `test_*.py`

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Phi** | Information integration metric (IIT) |
| **Coherence** | Response consistency over time |
| **Knowledge Graph** | Evolving concept relationships |
| **Self-Modifying Network** | 4 layers + observers + meta-weights |

## Tech Stack

Python 3.8+ / PyTorch / Sentence-Transformers / NetworkX / SQLite / NLTK
