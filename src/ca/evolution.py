"""Evolutionary cellular automaton experiment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


@dataclass
class Genome:
    """Genome for the evolutionary CA."""

    rule_bits: np.ndarray  # shape (8,)
    adj_matrix: np.ndarray  # shape (n, n)
    tau_mut: float


def _next_state(
    state: np.ndarray, rule_bits: np.ndarray, neighbors: List[np.ndarray]
) -> np.ndarray:
    """Compute next CA state for a given graph."""
    n = state.size
    new_state = state.copy()
    for i in range(n):
        neigh = neighbors[i]
        left = state[neigh[0]] if len(neigh) > 0 else False
        right = state[neigh[1]] if len(neigh) > 1 else False
        pattern = (left << 2) | (state[i] << 1) | right
        new_state[i] = bool(rule_bits[7 - pattern])
    return new_state


def _run_ca(rule_bits: np.ndarray, adj: np.ndarray, t1: int = 30, t2: int = 30) -> np.ndarray:
    """Run CA with a shock after ``t1`` steps."""
    n = adj.shape[0]
    state = np.random.rand(n) < 0.5
    history = []
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    for _ in range(t1):
        history.append(state.copy())
        state = _next_state(state, rule_bits, neighbors)
    history.append(state.copy())
    # Shock
    idx = np.random.choice(n, int(0.15 * n), replace=False)
    state[idx] = ~state[idx]
    for _ in range(t2):
        history.append(state.copy())
        state = _next_state(state, rule_bits, neighbors)
    history.append(state.copy())
    return np.array(history)


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    xf = x.astype(float)
    yf = y.astype(float)
    bins_x = np.digitize(xf, np.histogram_bin_edges(xf, bins=10)) - 1
    bins_y = np.digitize(yf, np.histogram_bin_edges(yf, bins=10)) - 1
    return float(normalized_mutual_info_score(bins_x, bins_y))


def evaluate_individual(genome: Genome, t1: int = 30, t2: int = 30) -> float:
    """Evaluate an individual's fitness."""
    history = _run_ca(genome.rule_bits, genome.adj_matrix, t1=t1, t2=t2)
    counts = history.sum(axis=1)
    pre = counts[1 : t1 + 1]
    post = counts[t1 + 1 : t1 + 1 + t2]
    i12 = _mutual_info(pre, post)

    early = history[:16].T
    final_state = history[t1 + t2]
    compressed = PCA(n_components=2).fit_transform(early)
    labels = KMeans(n_clusters=5, n_init=5, random_state=0).fit_predict(compressed)
    cols = early.shape[1]
    powers = 1 << np.arange(cols - 1, -1, -1)
    early_int = (early * powers).sum(axis=1)
    ib = _mutual_info(labels, final_state) - 0.5 * _mutual_info(labels, early_int)

    penalty = genome.tau_mut * (1.0 - genome.tau_mut)
    return 0.5 * i12 + 0.5 * ib - 0.1 * penalty


def mutate_individual(genome: Genome) -> Genome:
    """Return a mutated copy of ``genome``."""
    rule = genome.rule_bits.copy()
    mask = np.random.rand(rule.size) < genome.tau_mut
    rule[mask] = ~rule[mask]

    adj = genome.adj_matrix.copy()
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < genome.tau_mut:
                adj[i, j] = ~adj[i, j]
                adj[j, i] = adj[i, j]
    np.fill_diagonal(adj, 0)

    tau = float(np.clip(genome.tau_mut + np.random.normal(scale=0.05), 0.0, 1.0))
    return Genome(rule_bits=rule, adj_matrix=adj, tau_mut=tau)


def run_evolution(
    generations: int = 1000,
    population_size: int = 50,
    elite_frac: float = 0.1,
    t1: int = 30,
    t2: int = 30,
) -> Tuple[List[dict], Genome]:
    """Run evolutionary search for the CA."""
    n = 150
    pop = [
        Genome(
            rule_bits=np.random.randint(0, 2, 8, dtype=bool),
            adj_matrix=(np.random.rand(n, n) < 0.05).astype(bool),
            tau_mut=float(np.random.rand()),
        )
        for _ in range(population_size)
    ]
    for g in pop:
        np.fill_diagonal(g.adj_matrix, 0)
        g.adj_matrix = np.logical_or(g.adj_matrix, g.adj_matrix.T)

    metrics: List[dict] = []
    best: Genome | None = None

    for _ in range(generations):
        fitness = np.array([evaluate_individual(ind, t1, t2) for ind in pop])
        order = np.argsort(-fitness)
        pop = [pop[i] for i in order]
        fitness = fitness[order]
        best = pop[0]

        Gs = [nx.from_numpy_array(ind.adj_matrix) for ind in pop]
        avg_degrees = [np.mean([d for _, d in g.degree()]) for g in Gs]
        clusterings = [nx.average_clustering(g) for g in Gs]

        metrics.append(
            {
                "min_fitness": float(fitness.min()),
                "max_fitness": float(fitness.max()),
                "mean_fitness": float(fitness.mean()),
                "avg_degree": float(np.mean(avg_degrees)),
                "clustering": float(np.mean(clusterings)),
            }
        )

        elite_count = max(1, int(elite_frac * population_size))
        elites = pop[:elite_count]
        new_pop = elites.copy()
        while len(new_pop) < population_size:
            idx = np.random.choice(population_size, 3, replace=False)
            candidate = pop[idx[np.argmax(fitness[idx])]]
            new_pop.append(mutate_individual(candidate))
        pop = new_pop

    assert best is not None
    return metrics, best
