"""Demo script for CA self-modeling experiment."""

from pathlib import Path
import csv
import numpy as np

from ca.analysis import evolve_population
from ca.visualize import plot_convergence, save_snapshots


DEFAULT_GENERATIONS = 10


def main() -> int:
    pop_size = 4
    pop = [np.random.randint(0, 2, 18, dtype=bool) for _ in range(pop_size)]

    scores, history = evolve_population(pop, generations=DEFAULT_GENERATIONS)

    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_score"])
        for i, score in enumerate(scores):
            writer.writerow([i, score])

    plot_convergence(scores, Path("convergence.png"))

    per_step = max(1, history.shape[0] // 10)
    save_snapshots(history, Path("gallery"), per_step=per_step)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
