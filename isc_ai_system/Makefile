# ISC AI System Makefile

.PHONY: help install run test clean backup demo storage-demo query-demo dev lint format train-gpt train-gpt-mt train-gpt-enhanced train-self train-self-verbose train-self-resume train-self-select report report-all metrics analyze

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