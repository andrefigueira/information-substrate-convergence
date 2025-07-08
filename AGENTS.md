# Guidelines for Codex Agents

- Install dependencies before running tests:
  ```bash
  pip install -r ca_experiment/requirements.txt
  ```
- Run the full test suite with `pytest -q`.
- Maintain code style with 4-space indentation.
- Use `black` and `flake8` to check formatting and lint:
  ```bash
  black --check .
  flake8 .
  ```
- Remove any `__pycache__` folders before committing.
- Keep commit messages concise and descriptive.
