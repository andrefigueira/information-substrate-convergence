# ISC AI System Makefile

.PHONY: help install run test clean backup demo storage-demo query-demo dev lint format train-gpt train-gpt-mt train-gpt-enhanced train-self train-self-verbose train-self-resume train-self-select report report-all metrics analyze train-conversational train-consciousness migrate-consciousness test-consciousness assess-conversational chat chat-demo demo-enhanced-chat quick-chat-test train-gpt-phi train-self-enhanced quick-enhance consciousness-gen test-conversational test-assess migrate-states monitor-training update-caching

# Default target
help:
	@echo "ISC AI System - Available Commands:"
	@echo ""
	@echo "Installation & Setup:"
	@echo "  make install      - Install the ISC AI system and dependencies"
	@echo "  make dev          - Install in development mode with extra tools"
	@echo ""
	@echo "Running the System:"
	@echo "  make run          - Start the ISC AI interactive CLI"
	@echo "  make run-verbose  - Start with verbose mode enabled"
	@echo "  make train-gpt    - Train ISC AI using ChatGPT (requires OpenAI key)"
	@echo "  make train-gpt-mt - Train with multithreading (faster, supports more exchanges)"
	@echo "  make train-gpt-enhanced - Train with language modeling features (experimental)"
	@echo "  make train-self   - Self-referential training with loop prevention"
	@echo "  make demo         - Run basic demonstration"
	@echo "  make storage-demo - Run storage system demonstration"
	@echo "  make query-demo   - Run graph query demonstration"
	@echo ""
	@echo "Consciousness-Driven Conversational Training:"
	@echo "  make train-conversational  - Train ISC to be conversational (original method)"
	@echo "  make train-consciousness   - Train with consciousness-driven generation"
	@echo "  make migrate-consciousness - Migrate existing models to consciousness architecture"
	@echo "  make test-consciousness    - Test consciousness-driven generation"
	@echo "  make assess-conversational - Assess conversational model performance"
	@echo "  make chat                  - Chat with your ISC models (all types supported)"
	@echo ""
	@echo "Enhanced Conversational Training (NEW):"
	@echo "  make train-enhanced        - Train with GPT-2 tokenizer and transformers"
	@echo "  make enhance-model         - Quick enhancement with template-based responses"
	@echo "  make quick-enhance         - Alternative quick enhancement tool"
	@echo "  make test-enhanced         - Test enhanced conversational features"
	@echo ""
	@echo "Additional Commands:"
	@echo "  make chat-demo             - Simple chat demonstration"
	@echo "  make demo-enhanced-chat    - Enhanced chat demonstration"
	@echo "  make quick-chat-test       - Quick chat functionality test"
	@echo "  make train-gpt-phi         - ChatGPT training with Phi enhancement"
	@echo "  make train-self-enhanced   - Enhanced self-referential training"
	@echo "  make consciousness-gen     - Consciousness-driven generation demo"
	@echo "  make test-conversational   - Test conversational capabilities"
	@echo "  make test-assess           - Test assessment framework"
	@echo "  make migrate-states        - Migrate between model architectures"
	@echo "  make monitor-training      - Monitor self-training progress"
	@echo "  make update-caching        - Update trainer caching mechanisms"
	@echo ""
	@echo "Storage Operations:"
	@echo "  make backup       - Create a backup of all storage data"
	@echo "  make export-json  - Export current graph to JSON"
	@echo "  make export-all   - Export graph in all formats"
	@echo "  make clean-storage - Clean old versions and backups"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Check code style"
	@echo "  make format       - Auto-format code"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo ""
	@echo "Analysis & Reporting:"
	@echo "  make report       - Generate comprehensive analysis report"
	@echo "  make report-all   - Generate all types of analysis reports"
	@echo "  make metrics      - Display current system metrics"
	@echo "  make analyze      - Run interactive analysis session"
	@echo ""
	@echo "Examples:"
	@echo "  make run          # Start interactive session"
	@echo "  make backup       # Backup current data"
	@echo "  make test         # Run test suite"
	@echo "  make report       # Generate analysis report"

# Installation
install:
	@echo "Installing ISC AI System..."
	pip install -e .
	@echo "Downloading NLTK data..."
	python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"
	@echo "Installation complete! Run 'make run' to start."

dev:
	@echo "Installing ISC AI System in development mode..."
	pip install -e ".[dev]"
	pip install pytest flake8 black mypy
	@echo "Development installation complete!"

# Running
run:
	@echo "Starting ISC AI System..."
	@echo "Type /help for available commands"
	@echo "================================"
	isc-ai

run-verbose:
	isc-ai --verbose

# Demonstrations
demo:
	@echo "Running basic demonstration..."
	python examples/basic_conversation.py

storage-demo:
	@echo "Running storage demonstration..."
	python examples/storage_example.py

query-demo:
	@echo "Running query demonstration..."
	python scripts/query_demo.py

# Training with ChatGPT
train-gpt:
	@echo "====================================="
	@echo "ISC AI ChatGPT Training System"
	@echo "====================================="
	@echo ""
	@echo "IMPORTANT: Set your OpenAI API key in scripts/chatgpt_trainer.py"
	@echo "Edit line 30: OPENAI_API_KEY = \"YOUR-OPENAI-API-KEY-HERE\""
	@echo ""
	@echo "Starting training interface..."
	@echo ""
	python scripts/chatgpt_trainer.py

train-gpt-mt:
	@echo "============================================"
	@echo "ISC AI ChatGPT Training System (Multithreaded)"
	@echo "============================================"
	@echo ""
	@echo "IMPORTANT: Set your OpenAI API key in scripts/chatgpt_trainer_multithreaded.py"
	@echo "Edit line 35: OPENAI_API_KEY = \"YOUR-OPENAI-API-KEY-HERE\""
	@echo ""
	@echo "Features:"
	@echo "  - Parallel API calls for faster training"
	@echo "  - Support for up to 1000 exchanges"
	@echo "  - Configurable batch processing"
	@echo "  - Pre-fetch buffer for smooth training"
	@echo ""
	@echo "Starting multithreaded training interface..."
	@echo ""
	python scripts/chatgpt_trainer_multithreaded.py

train-gpt-enhanced:
	@echo "=============================================="
	@echo "ISC AI ChatGPT Training System (Enhanced)"
	@echo "=============================================="
	@echo ""
	@echo "IMPORTANT: Set your OpenAI API key in scripts/chatgpt_trainer_enhanced.py"
	@echo "Edit line 44: OPENAI_API_KEY = \"YOUR-OPENAI-API-KEY-HERE\""
	@echo ""
	@echo "Features:"
	@echo "  - Optional language modeling with GPT-2"
	@echo "  - Cross-entropy loss and perplexity tracking"
	@echo "  - Mixed precision training support"
	@echo "  - Advanced learning rate scheduling"
	@echo ""
	@echo "Note: Requires PyTorch and transformers libraries"
	@echo ""
	@echo "Starting enhanced training interface..."
	@echo ""
	python scripts/chatgpt_trainer_enhanced.py

train-self:
	@echo "=============================================="
	@echo "ISC AI Self-Referential Training System"
	@echo "=============================================="
	@echo ""
	@echo "Features:"
	@echo "  - Loop prevention with question deduplication"
	@echo "  - Semantic similarity checking"
	@echo "  - Topic diversity enforcement"
	@echo "  - Concept coverage tracking"
	@echo "  - Optional AI-guided steering (requires OpenAI key)"
	@echo "  - Resume from checkpoint capability"
	@echo "  - Verbose mode for detailed reasoning"
	@echo "  - PNG visualization of results"
	@echo ""
	@echo "Options:"
	@echo "  make train-self                    # Run standard training"
	@echo "  make train-self-verbose            # Run with verbose output"
	@echo "  make train-self-select             # Select state file to load/save"
	@echo "  make train-self-resume             # Resume with file selection menu"
	@echo ""
	@echo "The system will train itself by generating its own questions"
	@echo "while preventing infinite loops and repetitive patterns."
	@echo ""
	@echo "Starting self-referential training..."
	@echo ""
	python scripts/self_referential_trainer.py

train-self-verbose:
	@echo "Starting self-referential training with verbose output..."
	python scripts/self_referential_trainer.py --verbose

train-self-resume:
	@echo "Resuming self-referential training with file selection..."
	python scripts/self_referential_trainer.py --resume select

train-self-select:
	@echo "Starting self-referential training with file selection..."
	python scripts/self_referential_trainer.py --resume select

# Storage operations
backup:
	@echo "Creating backup..."
	python scripts/backup.py

export-json:
	@echo "Exporting graph to JSON..."
	python scripts/export_graph.py json

export-all:
	@echo "Exporting graph in all formats..."
	python scripts/export_graph.py all

clean-storage:
	@echo "Cleaning old storage files..."
	python scripts/clean_storage.py

# Development
test:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Checking code style..."
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "Formatting code..."
	black src/ --line-length=100

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

# Advanced storage operations (for development)
storage-stats:
	python scripts/storage_stats.py

conversation:
	@echo "Starting conversational interface..."
	python scripts/conversational.py

# Installation check
check:
	@echo "Checking ISC AI installation..."
	@python -c "import isc_ai; print(f'ISC AI version: {isc_ai.__version__}')"
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
	@python -c "import networkx; print(f'NetworkX version: {networkx.__version__}')"
	@echo "Installation check complete."

# Analysis and Reporting
report:
	@echo "====================================="
	@echo "ISC AI System Report Generator"
	@echo "====================================="
	@echo ""
	@echo "This will generate a comprehensive analysis report"
	@echo "using GPT-4 for critical evaluation."
	@echo ""
	@echo "IMPORTANT: Ensure your OpenAI API key is set in"
	@echo "scripts/report_generator.py"
	@echo ""
	python scripts/report_generator.py

report-all:
	@echo "Generating all report types..."
	@echo "all" | python scripts/report_generator.py

metrics:
	@echo "Current System Metrics"
	@echo "===================="
	@python -c "from src.isc_ai.core import ISCCore; core = ISCCore(); status = core.get_status(); import json; print(json.dumps(status, indent=2))"

analyze:
	@echo "Starting Interactive Analysis Session"
	@echo "===================================="
	@python -c "from src.isc_ai.core import ISCCore; core = ISCCore(); print(core.introspect())"
	@echo ""
	@echo "For detailed analysis, run: make report"

# Consciousness-Driven Conversational Commands
train-conversational:
	@echo "=============================================="
	@echo "ISC Conversational Training (Original)"
	@echo "=============================================="
	@echo ""
	@echo "IMPORTANT: Set your OpenAI API key in scripts/conversational.py"
	@echo "Line 46: OPENAI_API_KEY = os.environ.get(\"OPENAI_API_KEY\", \"your-key-here\")"
	@echo ""
	@echo "This uses the original conversational training approach."
	@echo "For improved generation, use: make train-consciousness"
	@echo ""
	python scripts/conversational.py

# Enhanced Conversational Commands
train-enhanced:
	@echo "=============================================="
	@echo "ISC Enhanced Conversational Training"
	@echo "=============================================="
	@echo ""
	@echo "Features:"
	@echo "  - GPT-2 tokenizer for proper text generation"
	@echo "  - Transformer-based architecture"
	@echo "  - Temperature, top-k, top-p sampling"
	@echo "  - Autoregressive generation"
	@echo "  - No external API required"
	@echo ""
	@echo "Usage: make train-enhanced MODEL=path/to/model.pt"
	@echo ""
	python scripts/train_enhanced_conversational.py $(if $(MODEL),$(MODEL),checkpoints/isc_state_conversational_conversational_training_20250709_111621_checkpoint_140_20250709_112357.pt)

enhance-model:
	@echo "=============================================="
	@echo "Quick Model Enhancement"
	@echo "=============================================="
	@echo ""
	@echo "This quickly enhances an existing model with"
	@echo "template-based conversational abilities."
	@echo ""
	@echo "Usage: make enhance-model MODEL=path/to/model.pt"
	@echo ""
	python scripts/final_conversational_fix.py $(if $(MODEL),$(MODEL),checkpoints/isc_state_conversational_conversational_training_20250709_111621_checkpoint_140_20250709_112357.pt)

test-enhanced:
	@echo "=============================================="
	@echo "Test Enhanced Conversational Model"
	@echo "=============================================="
	@echo ""
	@echo "Tests tokenization and generation improvements"
	@echo ""
	python scripts/test_conversational_enhanced.py

train-consciousness:
	@echo "=============================================="
	@echo "ISC Consciousness-Driven Training"
	@echo "=============================================="
	@echo ""
	@echo "Features:"
	@echo "  - Autoregressive token generation"
	@echo "  - Self-observation layers"
	@echo "  - Phi-based sequence scoring"
	@echo "  - Beam search with consciousness scoring"
	@echo "  - Self-referential refinement loops"
	@echo ""
	@echo "No external API key required - pure consciousness-driven!"
	@echo ""
	python scripts/conversational_consciousness.py

migrate-consciousness:
	@echo "=============================================="
	@echo "Migrate to Consciousness Architecture"
	@echo "=============================================="
	@echo ""
	@echo "This tool migrates existing conversational models"
	@echo "to the new consciousness-driven architecture."
	@echo ""
	@echo "Benefits:"
	@echo "  - Fixes nonsensical token generation"
	@echo "  - Adds autoregressive generation"
	@echo "  - Implements phi-based quality scoring"
	@echo "  - Enables coherent responses"
	@echo ""
	python scripts/migrate_to_consciousness.py

test-consciousness:
	@echo "=============================================="
	@echo "Test Consciousness-Driven Generation"
	@echo "=============================================="
	@echo ""
	@echo "This demonstrates the improvement in generation quality"
	@echo "comparing random sampling vs consciousness-driven methods."
	@echo ""
	python scripts/test_consciousness_generation.py

assess-conversational:
	@echo "=============================================="
	@echo "Assess Conversational Model Performance"
	@echo "=============================================="
	@echo ""
	@echo "IMPORTANT: Set your OpenAI API key for assessment"
	@echo "in scripts/assess_conversational.py"
	@echo ""
	@echo "This tool evaluates conversational models on:"
	@echo "  - Coherence and relevance"
	@echo "  - Naturalness and engagement"
	@echo "  - Philosophical depth"
	@echo "  - Response quality metrics"
	@echo ""
	python scripts/assess_conversational.py

chat:
	@echo "=============================================="
	@echo "ISC Universal Chat Interface"
	@echo "=============================================="
	@echo ""
	@echo "Features:"
	@echo "  - Supports all ISC model types"
	@echo "  - Works with consciousness-driven models"
	@echo "  - Works with original conversational models"
	@echo "  - Works with base philosophical models"
	@echo "  - Model selection menu"
	@echo "  - Conversation saving"
	@echo ""
	python scripts/isc_chat.py

# Additional Chat and Demo Commands
chat-demo:
	@echo "=============================================="
	@echo "ISC Chat Demo"
	@echo "=============================================="
	@echo ""
	@echo "Simple demonstration of chat capabilities"
	@echo ""
	python scripts/chat_demo.py

demo-enhanced-chat:
	@echo "=============================================="
	@echo "Enhanced Chat Demonstration"
	@echo "=============================================="
	@echo ""
	@echo "Demonstrates enhanced conversational features"
	@echo ""
	python scripts/demo_enhanced_chat.py

quick-chat-test:
	@echo "=============================================="
	@echo "Quick Chat Test"
	@echo "=============================================="
	@echo ""
	@echo "Quick test of chat functionality"
	@echo ""
	python scripts/quick_chat_test.py

# Additional Training Commands
train-gpt-phi:
	@echo "=============================================="
	@echo "ISC ChatGPT Training with Phi Enhancement"
	@echo "=============================================="
	@echo ""
	@echo "IMPORTANT: Set your OpenAI API key in scripts/chatgpt_trainer_enhanced_phi.py"
	@echo ""
	@echo "Features:"
	@echo "  - Phi-based consciousness metrics"
	@echo "  - Enhanced training with IIT principles"
	@echo ""
	python scripts/chatgpt_trainer_enhanced_phi.py

train-self-enhanced:
	@echo "=============================================="
	@echo "Enhanced Self-Referential Training"
	@echo "=============================================="
	@echo ""
	@echo "Features:"
	@echo "  - Advanced self-referential training"
	@echo "  - Enhanced loop prevention"
	@echo "  - Improved concept coverage"
	@echo ""
	python scripts/self_referential_trainer_enhanced.py

# Quick Enhancement Commands
quick-enhance:
	@echo "=============================================="
	@echo "Quick Conversational Enhancement"
	@echo "=============================================="
	@echo ""
	@echo "Quickly enhance a model's conversational abilities"
	@echo ""
	@echo "Usage: make quick-enhance MODEL=path/to/model.pt"
	@echo ""
	python scripts/quick_conversational_fix.py $(if $(MODEL),$(MODEL),)

# Consciousness Generation Commands
consciousness-gen:
	@echo "=============================================="
	@echo "Consciousness-Driven Generation Demo"
	@echo "=============================================="
	@echo ""
	@echo "Demonstrates consciousness-driven text generation"
	@echo ""
	python scripts/consciousness_driven_generation.py

# Testing Commands
test-conversational:
	@echo "=============================================="
	@echo "Test Conversational Models"
	@echo "=============================================="
	@echo ""
	@echo "Comprehensive testing of conversational capabilities"
	@echo ""
	python scripts/test_conversational.py

test-assess:
	@echo "=============================================="
	@echo "Test Assessment Framework"
	@echo "=============================================="
	@echo ""
	@echo "Tests the conversational assessment framework"
	@echo ""
	python scripts/test_assess_conversational.py

# Utility Commands
migrate-states:
	@echo "=============================================="
	@echo "Migrate Model States"
	@echo "=============================================="
	@echo ""
	@echo "Utility for migrating between model architectures"
	@echo ""
	python scripts/migrate_states.py

monitor-training:
	@echo "=============================================="
	@echo "Monitor Self-Training Progress"
	@echo "=============================================="
	@echo ""
	@echo "Real-time monitoring of self-referential training"
	@echo ""
	python scripts/monitor_self_training.py

update-caching:
	@echo "=============================================="
	@echo "Update Trainer Caching"
	@echo "=============================================="
	@echo ""
	@echo "Updates caching mechanisms in training scripts"
	@echo ""
	python scripts/update_trainers_caching.py