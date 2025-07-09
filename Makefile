.PHONY: install test lint format clean demo train-enhanced train-enhanced-chat train-enhanced-verbose

install:
	pip install -r ca_experiment/requirements.txt

test:
	pytest -q

lint:
	black --check .
	flake8 .

format:
	black .

clean:
	find . -name '__pycache__' -type d -exec rm -r {} +

demo:
	python ca_experiment/demo.py

# Enhanced self-referential training with proper CE loss
train-enhanced:
	python isc_ai_system/scripts/self_referential_trainer_enhanced.py

# Enhanced training with immediate chat mode
train-enhanced-chat:
	python isc_ai_system/scripts/self_referential_trainer_enhanced.py --chat

# Enhanced training with verbose output
train-enhanced-verbose:
	python isc_ai_system/scripts/self_referential_trainer_enhanced.py --verbose --exchanges 50

# Resume enhanced training and enter chat
chat-enhanced:
	python isc_ai_system/scripts/self_referential_trainer_enhanced.py --resume select --chat

# Direct chat mode with file selection (no training)
chat-select:
	python isc_ai_system/scripts/self_referential_trainer_enhanced.py --chat

# Quick enhanced training test (20 exchanges)
train-enhanced-quick:
	python isc_ai_system/scripts/self_referential_trainer_enhanced.py --exchanges 20 --verbose --chat

# ISC Chat Interface Commands
# Interactive chat with model selection
chat:
	python isc_ai_system/scripts/isc_chat.py

# Chat with verbose mode (shows detailed metrics)
chat-verbose:
	python isc_ai_system/scripts/isc_chat.py --verbose

# Chat with philosophical response style
chat-philosophical:
	python isc_ai_system/scripts/isc_chat.py --style philosophical

# Chat with technical response style
chat-technical:
	python isc_ai_system/scripts/isc_chat.py --style technical --verbose

# Chat with specific model file
chat-model:
	@echo "Usage: make chat-model MODEL=path/to/model.pt"
	@if [ -z "$(MODEL)" ]; then echo "Error: MODEL parameter required"; exit 1; fi
	python isc_ai_system/scripts/isc_chat.py --model $(MODEL)

