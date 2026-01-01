# Information Substrate Convergence - Context Substrate

> Documentation as Code as Context for AI-assisted development

## Navigation

This `.context/` directory contains structured knowledge for working on the ISC theory and codebase. Use these files to understand the project before making changes.

### Quick Reference

| Domain | Purpose | Start Here |
|--------|---------|------------|
| [theory/](theory/) | ISC theoretical framework | [overview.md](theory/overview.md) |
| [architecture/](architecture/) | System design and data flow | [overview.md](architecture/overview.md) |
| [components/](components/) | Core module documentation | [isc-core.md](components/isc-core.md) |
| [experiments/](experiments/) | CA and validation experiments | [cellular-automata.md](experiments/cellular-automata.md) |
| [guidelines.md](guidelines.md) | Development standards and workflows | - |
| [ai-rules.md](ai-rules.md) | Hard constraints for AI tools | - |
| [anti-patterns.md](anti-patterns.md) | What to avoid | - |

### Component Index

| Component | File | Purpose |
|-----------|------|---------|
| ISCCore | [isc-core.md](components/isc-core.md) | Main orchestrator |
| InformationIntegrator | [information-integration.md](components/information-integration.md) | Phi calculations |
| KnowledgeGraph | [knowledge-graph.md](components/knowledge-graph.md) | Concept storage |
| ConversationMemory | [memory.md](components/memory.md) | Conversation persistence |
| LearningEngine | [learning.md](components/learning.md) | Self-supervised learning |

### Theory Quick Links

| Topic | File |
|-------|------|
| ISC Overview | [theory/overview.md](theory/overview.md) |
| Consciousness | [theory/consciousness.md](theory/consciousness.md) |
| Implications | [theory/implications.md](theory/implications.md) |
| Evidence | [theory/evidence.md](theory/evidence.md) |
| Glossary | [theory/glossary.md](theory/glossary.md) |

## AI Usage Patterns

### Understanding the Theory
```
Read: theory/overview.md -> theory/consciousness.md -> theory/implications.md
Goal: Understand ISC hypothesis and philosophical foundations
```

### Modifying the ISC AI System
```
Read: architecture/overview.md -> components/isc-core.md -> components/[relevant-module].md
Goal: Understand system architecture before making changes
```

### Working with Cellular Automata
```
Read: experiments/cellular-automata.md -> architecture/ca-system.md
Goal: Understand the CA evolution system and self-modeling metrics
```

### Adding New Features
```
Read: guidelines.md -> architecture/overview.md -> components/[affected-modules].md
Goal: Follow project standards and understand integration points
```

## Project Summary

**Information Substrate Convergence (ISC)** is a research framework exploring consciousness emergence through self-referential information patterns. The project includes:

1. **Theoretical Framework** - Proposes that consciousness emerges when information systems achieve configurations enabling self-modeling and information integration
2. **ISC AI System** - Interactive AI demonstrating consciousness-like properties through phi calculations, knowledge graphs, and self-modifying neural networks
3. **Cellular Automata Experiments** - Evolution of CA rules toward self-modeling capability as empirical validation

## Core Hypothesis

Consciousness emerges through:
- **Self-referential processing**: Systems modeling their own internal states
- **Information integration**: Measured by phi (Phi) from Integrated Information Theory
- **Dynamic learning**: Continuous adaptation through interaction
- **Emergent coherence**: Stable pattern development via self-organization

## File Update Protocol

When making significant changes to the codebase:
1. Update relevant `.context/` files to reflect architectural changes
2. Keep code examples in documentation synchronized with actual implementation
3. Document new patterns or methods in `guidelines.md`
4. Add new components to `components/` directory

## Tech Stack Quick Reference

- **Python 3.8+** - Primary language
- **PyTorch** - Neural networks and self-modifying architectures
- **NetworkX** - Knowledge graph operations
- **SQLite** - Conversation memory persistence
- **Sentence-Transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **NLTK** - Natural language processing
- **NumPy/SciPy** - Mathematical computations and CA simulations
