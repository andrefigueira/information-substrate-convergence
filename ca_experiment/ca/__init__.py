"""CA experiment package."""

from .analysis import compressibility, evolve_population
from .simulation import run_ca, step
from .visualize import plot_convergence, save_snapshots
from .evolution import Genome, evaluate_individual, mutate_individual, run_evolution

__all__ = [
    "compressibility",
    "evolve_population",
    "run_ca",
    "step",
    "plot_convergence",
    "save_snapshots",
    "Genome",
    "evaluate_individual",
    "mutate_individual",
    "run_evolution",
]
