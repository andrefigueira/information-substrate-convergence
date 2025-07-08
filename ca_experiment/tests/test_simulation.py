import numpy as np

from ca.simulation import run_ca, step


def test_step_blinker_block():
    birth = [False] * 9
    birth[3] = True
    survival = [False] * 9
    survival[2] = True
    survival[3] = True

    blinker = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)
    blinker_next = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool)
    result = step(blinker, birth, survival)
    assert np.array_equal(result, blinker_next)

    block = np.array([[1, 1], [1, 1]], dtype=bool)
    result_block = step(block, birth, survival)
    assert np.array_equal(result_block, block)


def test_run_ca_shape():
    rule = np.random.randint(0, 2, 18, dtype=bool)
    hist = run_ca(rule, size=10, steps=5)
    assert hist.shape == (5, 10, 10)
