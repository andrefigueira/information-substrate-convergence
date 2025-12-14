"""Visualization utilities for cellular automaton experiments."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(scores: List[float], out_path: Path) -> None:
    """Plot convergence of scores over generations."""
    plt.figure()
    plt.plot(range(len(scores)), scores, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Score")
    plt.title("Evolution Convergence")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_snapshots(history: np.ndarray, out_dir: Path, per_step: int = 1) -> None:
    """Save snapshots of CA history to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = history.shape[0]
    for i in range(0, steps, per_step):
        plt.imsave(out_dir / f"step_{i:03d}.png", history[i], cmap="binary")
