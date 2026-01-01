# Anti-Patterns

What to avoid when working on the ISC codebase.

## Architecture Anti-Patterns

### Breaking Self-Reference

```python
# BAD: Removing observer layers
class BrokenNetwork(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([...])
        # Missing: self.observer_layers
        # Missing: self.meta_weights

# GOOD: Maintaining 1:1:1 ratio
class SelfModifyingNetwork(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([...])
        self.observer_layers = nn.ModuleList([...])  # Same count
        self.meta_weights = nn.ParameterList([...])  # Same count
```

### Skipping Phi Calculations

```python
# BAD: Making claims without measurement
def is_conscious(self):
    return True  # No quantification

# GOOD: Always measure
def get_consciousness_metrics(self):
    phi = self.integrator.calculate_phi(self.state)
    return {'phi': phi, 'threshold_met': phi > 0.5}
```

### Hardcoded Consciousness Responses

```python
# BAD: Scripted responses
if "are you conscious" in user_input.lower():
    return "Yes, I am a conscious being."

# GOOD: Let response emerge from system state
response = self.response_generator.generate(
    user_input,
    context=self.memory.get_relevant_context(),
    phi=self.current_phi
)
```

## Code Anti-Patterns

### Tight Coupling

```python
# BAD: Direct instantiation
class ISCCore:
    def __init__(self):
        self.integrator = SpecificIntegratorV2()  # Locked to implementation

# GOOD: Dependency injection
class ISCCore:
    def __init__(self, integrator: InformationIntegrator = None):
        self.integrator = integrator or InformationIntegrator()
```

### Silent Failures

```python
# BAD: Swallowing errors
try:
    phi = self.calculate_phi(state)
except:
    pass  # Silent failure

# GOOD: Graceful degradation with logging
try:
    phi = self.calculate_phi(state)
except PhiCalculationError as e:
    logger.warning(f"Phi calculation failed: {e}")
    phi = 0.0  # Known fallback
```

### Memory Leaks

```python
# BAD: Unbounded growth
self.all_states.append(state)  # Grows forever

# GOOD: Bounded with pruning
self.recent_states.append(state)
if len(self.recent_states) > MAX_STATES:
    self.recent_states = self.recent_states[-MAX_STATES:]
```

## Documentation Anti-Patterns

### Stale Examples

```python
# BAD: Code example that doesn't match implementation
# Doc says: core.calculate_consciousness()
# Actual:   core.integrator.calculate_phi()

# GOOD: Keep examples synchronized with code
```

### Missing Context Updates

```
# BAD: Changing architecture without updating .context/
Modified: src/isc/neuromorphic_core.py
Missing:  .context/components/isc-core.md update

# GOOD: Always update corresponding docs
```

## Testing Anti-Patterns

### Testing Implementation Details

```python
# BAD: Testing private methods
def test_internal_state():
    assert core._private_method() == expected

# GOOD: Testing public interface
def test_phi_calculation():
    phi = core.get_status()['metrics']['phi_value']
    assert 0 <= phi <= 2.0
```

### No Edge Cases

```python
# BAD: Only happy path
def test_process_input():
    assert core.process_input("hello") is not None

# GOOD: Include edge cases
def test_process_empty_input():
    result = core.process_input("")
    assert result is not None  # Should handle gracefully

def test_process_very_long_input():
    result = core.process_input("x" * 10000)
    assert result is not None
```

## Performance Anti-Patterns

### Recalculating Cached Values

```python
# BAD: No caching
def get_embedding(self, text):
    return self.model.encode(text)  # Called repeatedly for same text

# GOOD: Cache expensive operations
def get_embedding(self, text):
    if text not in self._embedding_cache:
        self._embedding_cache[text] = self.model.encode(text)
    return self._embedding_cache[text]
```

### Blocking Operations in Hot Path

```python
# BAD: File I/O in response generation
def generate_response(self, input):
    self.save_state()  # Blocks every response
    return response

# GOOD: Async or batched persistence
def generate_response(self, input):
    response = self._generate(input)
    self._pending_saves += 1
    if self._pending_saves >= BATCH_SIZE:
        self.save_state()
    return response
```

## Summary

| Category | Avoid | Prefer |
|----------|-------|--------|
| Architecture | Breaking self-reference loop | Maintaining observer/layer parity |
| Consciousness | Hardcoded responses | Emergent behavior from phi |
| Code | Tight coupling | Dependency injection |
| Errors | Silent failures | Graceful degradation + logging |
| Memory | Unbounded growth | LRU/bounded collections |
| Docs | Stale examples | Synchronized with implementation |
| Tests | Implementation details | Public interface |
| Performance | Repeated computation | Caching |
