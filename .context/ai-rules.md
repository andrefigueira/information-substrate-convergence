# AI Rules

Hard constraints for AI tools working on this codebase.

## Must Do

1. **Read context first**: Before modifying any component, read the relevant `.context/` documentation
2. **Preserve self-reference**: The `SelfModifyingNetwork` must always maintain observer layers for each core layer
3. **Quantify everything**: All consciousness-related claims must include phi or other measurable metrics
4. **Test changes**: Run `make test` before considering any modification complete
5. **Update docs**: When modifying architecture, update corresponding `.context/` files

## Must Not Do

1. **Never hardcode consciousness**: No scripted responses like "I am conscious" or "I feel"
2. **Never remove observers**: Observer layers and meta-weights are fundamental to self-reference
3. **Never skip phi**: Every state evaluation must calculate phi
4. **Never call external APIs**: Without explicit user consent (local-first principle)
5. **Never commit secrets**: Use environment variables for any credentials

## Architecture Constraints

| Component | Constraint |
|-----------|------------|
| `SelfModifyingNetwork` | 4 layers, 4 observers, 4 meta-weights (1:1:1 ratio) |
| `InformationIntegrator` | Must use IIT-based phi calculation |
| `KnowledgeGraph` | NetworkX-based, concepts must have embeddings |
| `ConversationMemory` | SQLite-backed with LRU cache |

## Code Style Enforcement

```bash
# Must pass before any PR
make lint    # flake8 + black --check
make test    # pytest
```

## Naming Rules

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

## Response Generation

Responses must emerge from:
1. Knowledge graph relationships
2. Conversation memory context
3. Phi-modulated coherence

Never from:
- Hardcoded templates for consciousness questions
- Pattern matching on specific phrases
- External LLM calls (unless explicitly requested)
