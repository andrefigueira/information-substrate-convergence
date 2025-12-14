import numpy as np

from ca.evolution import Genome, evaluate_individual, mutate_individual, run_evolution


def test_mutate_individual_bounds():
    n = 10
    genome = Genome(
        rule_bits=np.zeros(8, dtype=bool),
        adj_matrix=np.zeros((n, n), dtype=bool),
        tau_mut=0.5,
    )
    mutated = mutate_individual(genome)
    assert mutated.rule_bits.shape == (8,)
    assert mutated.adj_matrix.shape == (n, n)
    assert 0 <= mutated.tau_mut <= 1


def test_evaluate_individual_runs():
    n = 10
    genome = Genome(
        rule_bits=np.random.randint(0, 2, 8, dtype=bool),
        adj_matrix=np.eye(n, dtype=bool),
        tau_mut=0.2,
    )
    fitness = evaluate_individual(genome, t1=2, t2=2)
    assert isinstance(fitness, float)


def test_run_evolution_small():
    metrics, best = run_evolution(generations=2, population_size=4, t1=2, t2=2)
    assert len(metrics) == 2
    assert isinstance(best, Genome)
