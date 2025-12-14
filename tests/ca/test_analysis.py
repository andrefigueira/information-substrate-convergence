import numpy as np

from ca.analysis import compressibility, evolve_population


def test_compressibility_limits():
    frozen = np.zeros((20, 20, 20), dtype=bool)
    random = np.random.randint(0, 2, (20, 20, 20), dtype=bool)
    comp_frozen = compressibility(frozen)
    comp_random = compressibility(random)
    assert comp_frozen < 0.2
    assert comp_random > 0.8


def test_evolve_population_outputs():
    pop = [np.random.randint(0, 2, 18, dtype=bool) for _ in range(3)]
    scores, history = evolve_population(pop, generations=3)
    assert len(scores) == 3
    assert history.shape == (30, 30, 30)


def test_evolve_population_improves_score():
    np.random.seed(0)
    pop = [np.random.randint(0, 2, 18, dtype=bool) for _ in range(4)]
    scores, history = evolve_population(pop, generations=5, target=0.45)
    assert scores[-1] <= scores[0]
    final_comp = compressibility(history)
    assert abs(final_comp - 0.45) < 0.2
