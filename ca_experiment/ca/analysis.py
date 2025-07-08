"""Analysis utilities for cellular automaton experiments."""

from __future__ import annotations

from typing import List, Tuple
import zlib

import numpy as np

from .simulation import run_ca


def compressibility(history: np.ndarray) -> float:
    """Compute compression ratio of CA history using zlib."""
    bits = np.packbits(history.astype(np.uint8).ravel())
    data = bits.tobytes()
    raw_size = len(data)
    compressed_size = len(zlib.compress(data))
    return compressed_size / raw_size if raw_size else 0.0


def evolve_population(
    pop: List[np.ndarray], generations: int, target: float = 0.45
) -> Tuple[List[float], np.ndarray]:
    """Evolve population of CA rules toward a target compressibility."""
    scores: List[float] = []
    best_history: np.ndarray | None = None

    for _ in range(generations):
        results = []
        for rule in pop:
            history = run_ca(rule, size=30, steps=30)
            score = abs(compressibility(history) - target)
            results.append((score, history, rule))
        results.sort(key=lambda x: x[0])
        best_score, best_history, best_rule = results[0]
        scores.append(best_score)

        # Create next generation by mutating best rule
        pop = [best_rule.copy() for _ in pop]
        for rule in pop:
            idx = np.random.randint(rule.size)
            rule[idx] = ~rule[idx]

    assert best_history is not None
    return scores, best_history
