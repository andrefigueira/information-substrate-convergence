.PHONY: install test lint format clean demo

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

