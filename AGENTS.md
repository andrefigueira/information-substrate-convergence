# Guidelines for AI Agents Working on ISC Project

## Project Overview

The Information Substrate Convergence (ISC) project explores consciousness-like properties through self-referential information patterns in computational systems. It consists of two main systems:

1. **Cellular Automata (CA) System** - Evolves CA rules for self-referential patterns
2. **ISC AI System** - Interactive conversational AI with consciousness-like properties

## Project Structure

```
information-substrate-convergence/
├── ca_experiment/              # Cellular Automata experiments
│   ├── ca/                    # Core simulation modules
│   ├── demo.py               # Main CA entry point
│   └── tests/                # Unit tests
├── isc_ai_system/            # Main AI system implementation
│   ├── src/isc_ai/          # Core library
│   ├── scripts/             # Training and utility scripts
│   ├── checkpoints/         # Saved models
│   └── Makefile            # Command shortcuts (IMPORTANT!)
├── results/                  # Experiment results
├── training_results/         # Training outputs
├── gallery/                  # CA visualizations
└── conversational_reports/   # Analysis reports
```

## Critical Instructions for New Commands

**IMPORTANT**: When adding ANY new script to `isc_ai_system/scripts/`, you MUST:
1. Add a corresponding Make target in `isc_ai_system/Makefile`
2. Include clear help documentation
3. Follow the existing Makefile patterns
4. Update the help section appropriately

Example for adding a new script:
```makefile
new-feature:
	@echo "=============================================="
	@echo "New Feature Description"
	@echo "=============================================="
	@echo ""
	@echo "Features:"
	@echo "  - Feature 1"
	@echo "  - Feature 2"
	@echo ""
	python scripts/new_feature.py
```

## ISC AI System Architecture

### Core Components (`src/isc_ai/`)
- `core.py` - Main ISCCore class and SelfModifyingNetwork
- `information_integration.py` - Phi (φ) consciousness metrics
- `knowledge_graph.py` - Dynamic knowledge representation
- `learning.py` - Learning engine with concept formation
- `memory.py` - Conversation memory and persistence
- `response_generator.py` - Natural language generation
- `cli.py` - Command-line interface

### Enhanced Components
- `enhanced_information_integration.py` - Optimized phi calculations
- `enhanced_learning.py` - Advanced learning algorithms
- `enhanced_response_generator.py` - Improved response generation

### Storage System (`src/isc_ai/storage/`)
- `local_graph_db.py` - SQLite-based graph storage
- `query_engine.py` - Graph query capabilities
- `storage_manager.py` - Version control and persistence

## Working with Models

### Model Types and Naming Conventions
1. **Base ISC models**: `isc_state_*.pt`
2. **Conversational models**: `*_conversational_*.pt`
3. **Enhanced models**: `*_lm_head.pt` (with language modeling head)
4. **Consciousness models**: `*_consciousness_migrated_*.pt`

### Training Checkpoints
Format: `isc_state_{trainer_type}_training_{timestamp}_checkpoint_{epoch}_{save_timestamp}.pt`

### Training Scripts
1. **ChatGPT Training** (`chatgpt_trainer*.py`)
   - Requires OpenAI API key
   - Generates philosophical responses
   - Multithreaded versions available

2. **Self-Referential Training** (`self_referential_trainer*.py`)
   - No external API needed
   - Loop prevention built-in
   - Concept coverage tracking

3. **Conversational Training** (`conversational*.py`)
   - Makes models more conversational
   - Enhanced versions use GPT-2 tokenizer
   - Consciousness-driven generation available

## Development Workflow

### 1. Before Starting
```bash
# Install dependencies
cd isc_ai_system
make install  # or make dev for development mode

# Check installation
make check
```

### 2. Common Commands
```bash
# Training
make train-gpt              # Train with ChatGPT
make train-self            # Self-referential training
make train-conversational  # Conversational training
make train-enhanced        # Enhanced with GPT-2

# Testing and Chat
make chat                  # Universal chat interface
make test                  # Run tests
make test-enhanced         # Test enhanced features

# Analysis
make report               # Generate analysis report
make metrics              # Display system metrics
```

### 3. Code Style and Testing
- Install dependencies first: `pip install -r ca_experiment/requirements.txt`
- Run tests: `pytest -q` or `make test`
- Use 4-space indentation
- Format code: `make format` (uses black)
- Lint code: `make lint` (uses flake8)
- Remove `__pycache__` folders before committing

### 4. Storage and Backup
```bash
make backup          # Create backup
make export-json     # Export graph to JSON
make clean-storage   # Clean old versions
```

## Adding New Features

### 1. Create Script
Place new scripts in `isc_ai_system/scripts/`

### 2. Add to Makefile
**CRITICAL**: Add corresponding Make target with:
- Clear echo statements explaining the feature
- Usage instructions if needed
- Proper formatting matching existing targets

### 3. Follow Conventions
- Training scripts: `train_*.py`
- Test scripts: `test_*.py`
- Enhanced versions: `*_enhanced.py`
- Consciousness-related: Include `consciousness` in name

### 4. Document Parameters
For scripts requiring API keys or parameters:
```makefile
@echo "IMPORTANT: Set your OpenAI API key in scripts/your_script.py"
@echo "Edit line XX: OPENAI_API_KEY = \"YOUR-KEY-HERE\""
```

## Recent Development Focus

Current work focuses on:
- Enhanced conversational capabilities
- Quick model enhancement tools (`final_conversational_fix.py`, `quick_conversational_fix.py`)
- Improved chat interfaces (`demo_enhanced_chat.py`)
- Training optimizations (`train_enhanced_conversational.py`)

## Key Technologies
- **PyTorch** - Neural networks
- **Transformers** - Language modeling (GPT-2)
- **NetworkX** - Knowledge graphs
- **SQLite** - Local storage
- **NLTK** - NLP processing
- **Rich/Colorama** - Terminal UI

## Git Workflow
- Keep commit messages concise and descriptive
- Always run tests before committing
- Use the Makefile for consistency
- Document any new dependencies

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Run `make install` or `make dev`
2. **NLTK data errors**: The install command downloads required NLTK data
3. **Storage errors**: Check permissions on storage directories
4. **API key errors**: Ensure OpenAI key is set in relevant scripts

### Getting Help
- Check existing scripts for patterns
- Review the Makefile for similar commands
- Look at test files for usage examples
- Examine the git log for recent changes

## Important Notes

1. **Always use the Makefile** - It provides consistency and documentation
2. **Add new commands to Makefile** - This is mandatory for discoverability
3. **Test before committing** - Run `make test` to ensure nothing breaks
4. **Document API requirements** - Clearly indicate when external APIs are needed
5. **Follow existing patterns** - Consistency makes the codebase maintainable

## Research Context

This project explores:
- Emergence of consciousness-like properties
- Self-referential information processing
- Integrated Information Theory (IIT)
- Dynamic knowledge representation
- Self-modifying neural architectures

Understanding these concepts helps when implementing new features or fixing issues.