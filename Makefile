.PHONY: install install-dev install-training install-all test test-cov lint format clean help
.PHONY: ai ai-cli chat chat-verbose chat-philosophical chat-technical
.PHONY: demo ca-evolve train-enhanced train-enhanced-chat train-enhanced-verbose train-enhanced-quick
.PHONY: train-chatgpt train-chatgpt-realtime train-chatgpt-quick
.PHONY: evaluate evaluate-report evaluate-compare evaluate-full evaluate-simple
.PHONY: backup export-graph storage-stats status

# ===================================================================
# INSTALLATION
# ===================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-training:
	pip install -e ".[training]"

install-all:
	pip install -e ".[dev,training]"

# ===================================================================
# DEVELOPMENT
# ===================================================================

test:
	PYTHONPATH=src python run_tests.py tests/ -v

test-cov:
	PYTHONPATH=src python run_tests.py tests/ --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -name '__pycache__' -type d -exec rm -r {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -name '*.pyo' -delete 2>/dev/null || true
	find . -name '.pytest_cache' -type d -exec rm -r {} + 2>/dev/null || true
	find . -name '*.egg-info' -type d -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true

# ===================================================================
# CELLULAR AUTOMATA EXPERIMENTS
# ===================================================================

demo:
	python scripts/demos/ca_demo.py

ca-evolve:
	python -c "from ca.evolution import run_evolution; metrics, best = run_evolution(generations=500); print(f'Best fitness: {metrics[-1][\"max_fitness\"]:.4f}')"

# ===================================================================
# ISC AI SYSTEM
# ===================================================================

# Interactive CLI
ai:
	isc-ai

# Alternative: direct Python invocation
ai-cli:
	python -m isc.cli

# ===================================================================
# TRAINING COMMANDS
# ===================================================================

train-enhanced:
	python scripts/training/self_referential_trainer_enhanced.py

train-enhanced-chat:
	python scripts/training/self_referential_trainer_enhanced.py --chat

train-enhanced-verbose:
	python scripts/training/self_referential_trainer_enhanced.py --verbose --exchanges 50

train-enhanced-quick:
	python scripts/training/self_referential_trainer_enhanced.py --exchanges 20 --verbose --chat

# ChatGPT training
train-chatgpt:
	python scripts/training/chatgpt_trainer.py

train-chatgpt-realtime:
	python scripts/training/realtime_chatgpt_trainer.py

train-chatgpt-quick:
	python scripts/training/quick_chatgpt_training.py

# ===================================================================
# CHAT COMMANDS
# ===================================================================

chat:
	python scripts/demos/isc_chat.py

chat-verbose:
	python scripts/demos/isc_chat.py --verbose

chat-philosophical:
	python scripts/demos/isc_chat.py --style philosophical

chat-technical:
	python scripts/demos/isc_chat.py --style technical --verbose

# ===================================================================
# EVALUATION COMMANDS
# ===================================================================

evaluate:
	python scripts/evaluation/conversation_evaluator.py test

evaluate-report:
	python scripts/evaluation/conversation_evaluator.py report

evaluate-compare:
	python scripts/evaluation/conversation_evaluator.py compare

evaluate-full:
	python scripts/evaluation/conversation_evaluator.py full

evaluate-simple:
	python scripts/evaluation/simple_evaluator.py

# ===================================================================
# UTILITIES
# ===================================================================

backup:
	python scripts/utilities/backup.py

export-graph:
	python scripts/utilities/export_graph.py

storage-stats:
	python scripts/utilities/storage_stats.py

# ===================================================================
# STATUS & INFO
# ===================================================================

status:
	@echo "ISC Project Status"
	@echo "=================="
	@python -c "from isc.core import ISCCore; c = ISCCore(); s = c.get_status(); print(f'Phi: {s[\"metrics\"][\"phi_value\"]:.4f}'); print(f'Concepts: {s[\"total_concepts\"]}'); print(f'Interactions: {s[\"metrics\"][\"total_interactions\"]}')" 2>/dev/null || echo "No saved state found"

# ===================================================================
# HELP
# ===================================================================

help:
	@echo "ISC Project Commands"
	@echo "===================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install package"
	@echo "  make install-dev      Install with dev dependencies"
	@echo "  make install-training Install with training dependencies (OpenAI)"
	@echo "  make install-all      Install all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage"
	@echo "  make lint             Check code style"
	@echo "  make format           Format code"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "CA Experiments:"
	@echo "  make demo             Run CA evolution demo"
	@echo "  make ca-evolve        Run evolution experiment"
	@echo ""
	@echo "ISC AI:"
	@echo "  make ai               Start interactive CLI"
	@echo "  make chat             Start chat interface"
	@echo "  make chat-verbose     Chat with detailed metrics"
	@echo "  make status           Show system status"
	@echo ""
	@echo "Training:"
	@echo "  make train-enhanced        Self-referential training"
	@echo "  make train-enhanced-chat   Training with chat mode"
	@echo "  make train-chatgpt         ChatGPT-based training"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate         Run evaluation tests"
	@echo "  make evaluate-full    Complete evaluation cycle"
	@echo ""
	@echo "Utilities:"
	@echo "  make backup           Backup storage"
	@echo "  make export-graph     Export knowledge graph"
	@echo "  make storage-stats    Show storage statistics"
