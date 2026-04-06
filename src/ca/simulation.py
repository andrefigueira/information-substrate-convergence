"""Cellular automaton simulation utilities."""

from typing import Sequence

import numpy as np


def step(grid: np.ndarray, birth_mask: Sequence[bool], survival_mask: Sequence[bool]) -> np.ndarray:
    """Perform one CA step using Moore neighborhood.

    Args:
        grid: 2D boolean array representing the current state.
        birth_mask: Boolean sequence of length 9 for dead cell birth counts.
        survival_mask: Boolean sequence of length 9 for alive cell survival counts.

    Returns:
        Next grid state as a new array.
    """
    padded = np.pad(grid, 1, mode="constant").astype(int)
    neighbors = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )

    birth_mask = np.asarray(birth_mask, dtype=bool)
    survival_mask = np.asarray(survival_mask, dtype=bool)
    birth = birth_mask[neighbors]
    survive = survival_mask[neighbors]

    return np.where(grid, survive, birth)


def run_ca(rule_bits: np.ndarray, size: int = 200, steps: int = 200) -> np.ndarray:
    """Run cellular automaton simulation.

    Args:
        rule_bits: Boolean array of length 18 (birth bits followed by survival).
        size: Grid size.
        steps: Number of steps to simulate.

    Returns:
        History array of shape ``(steps, size, size)``.
    """
    rule_bits = np.asarray(rule_bits, dtype=bool).ravel()
    assert rule_bits.size == 18, "rule_bits must have length 18"
    birth_mask = rule_bits[:9]
    survival_mask = rule_bits[9:]

    grid = np.random.rand(size, size) < 0.5
    history = np.empty((steps, size, size), dtype=bool)
    for t in range(steps):
        history[t] = grid
        grid = step(grid, birth_mask, survival_mask)
    return history
