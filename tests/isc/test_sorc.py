"""
Unit tests for the SORC metric (src/isc/sorc.py).

These tests run without a real transformer model — they verify the mathematical
properties of the metric using synthetic hidden states constructed to produce
known outputs.

Properties tested:
    1. Degenerate input (identical hidden states) → low SORC
    2. Random input (scattered hidden states) → high SORC
    3. Structured input scores between degenerate and random
    4. Output is bounded in [0, 1]
    5. SORCResult accessors are correct
    6. Single-layer and multi-layer inputs both work
    7. Layer depth-weighting increases score when later layers are richer
"""

import numpy as np
import pytest
import torch

from src.isc.sorc import compute_sorc


def _make_hidden_states(
    n_tokens: int,
    hidden_dim: int,
    mode: str,
    n_layers: int = 4,
    seed: int = 0,
) -> list[torch.Tensor]:
    """
    Construct synthetic hidden states for testing.

    mode:
        "degenerate"  — all token vectors identical (near-rank-1 S matrix)
        "random"      — vectors drawn from standard normal
        "structured"  — vectors with moderate relational structure
    """
    rng = np.random.default_rng(seed)

    layers = []
    for _ in range(n_layers):
        if mode == "degenerate":
            base = rng.standard_normal(hidden_dim).astype(np.float32)
            H = np.tile(base, (n_tokens, 1))
            # Add tiny noise so normalisation doesn't divide by zero
            H += rng.standard_normal(H.shape).astype(np.float32) * 1e-6
        elif mode == "random":
            H = rng.standard_normal((n_tokens, hidden_dim)).astype(np.float32)
        elif mode == "structured":
            # k cluster centres, tokens assigned to clusters with noise
            k = max(2, n_tokens // 4)
            centres = rng.standard_normal((k, hidden_dim)).astype(np.float32)
            assignments = rng.integers(0, k, size=n_tokens)
            H = (
                centres[assignments]
                + rng.standard_normal((n_tokens, hidden_dim)).astype(np.float32) * 0.3
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        layers.append(torch.tensor(H))

    return layers


# ---------------------------------------------------------------------------
# Core property tests
# ---------------------------------------------------------------------------


class TestSORCBounds:
    def test_output_bounded_degenerate(self):
        hs = _make_hidden_states(20, 64, "degenerate")
        result = compute_sorc(hs)
        assert 0.0 <= result.sorc <= 1.0

    def test_output_bounded_random(self):
        hs = _make_hidden_states(20, 64, "random")
        result = compute_sorc(hs)
        assert 0.0 <= result.sorc <= 1.0

    def test_output_bounded_structured(self):
        hs = _make_hidden_states(20, 64, "structured")
        result = compute_sorc(hs)
        assert 0.0 <= result.sorc <= 1.0

    def test_layer_scores_all_bounded(self):
        hs = _make_hidden_states(20, 64, "random", n_layers=6)
        result = compute_sorc(hs)
        for s in result.layer_scores:
            assert 0.0 <= s <= 1.0


class TestSORCRanking:
    """
    The key prediction: degenerate < structured < random.
    Random produces the highest SORC because near-uniform eigenvalue distributions
    maximise spectral entropy. This is by design — absolute SORC values are only
    meaningful relative to the model-specific random baseline.
    """

    def test_degenerate_lower_than_random(self):
        hs_deg = _make_hidden_states(30, 64, "degenerate", seed=1)
        hs_rnd = _make_hidden_states(30, 64, "random", seed=1)
        result_deg = compute_sorc(hs_deg)
        result_rnd = compute_sorc(hs_rnd)
        assert (
            result_deg.sorc < result_rnd.sorc
        ), f"Expected degenerate ({result_deg.sorc:.4f}) < random ({result_rnd.sorc:.4f})"

    def test_structured_between_degenerate_and_random(self):
        hs_deg = _make_hidden_states(30, 64, "degenerate", seed=2)
        hs_str = _make_hidden_states(30, 64, "structured", seed=2)
        hs_rnd = _make_hidden_states(30, 64, "random", seed=2)
        r_deg = compute_sorc(hs_deg).sorc
        r_str = compute_sorc(hs_str).sorc
        r_rnd = compute_sorc(hs_rnd).sorc
        assert r_deg < r_str, f"Expected degenerate ({r_deg:.4f}) < structured ({r_str:.4f})"
        assert r_str < r_rnd, f"Expected structured ({r_str:.4f}) < random ({r_rnd:.4f})"

    def test_more_degenerate_layers_lower_score(self):
        """Adding degenerate layers to a structured set should lower mean score."""
        hs_structured = _make_hidden_states(20, 32, "structured", n_layers=4, seed=3)
        hs_degenerate_extra = _make_hidden_states(20, 32, "degenerate", n_layers=4, seed=3)
        # All structured
        r_all_structured = compute_sorc(hs_structured).sorc
        # Mix: structured + degenerate
        mixed = hs_structured[:2] + hs_degenerate_extra[:2]
        r_mixed = compute_sorc(mixed).sorc
        assert r_mixed < r_all_structured


class TestSORCDepthWeighting:
    """Later layers receive higher depth weights. Richer later layers → higher SORC."""

    def test_rich_later_layers_score_higher(self):
        # early layers degenerate, late layers rich
        early = _make_hidden_states(20, 32, "degenerate", n_layers=2, seed=4)
        late = _make_hidden_states(20, 32, "random", n_layers=2, seed=4)
        hs_front_rich = late + early  # rich early, degenerate late
        hs_back_rich = early + late  # degenerate early, rich late
        r_front = compute_sorc(hs_front_rich).sorc
        r_back = compute_sorc(hs_back_rich).sorc
        # Back-rich should score higher because later layers have larger depth weights
        assert r_back > r_front, f"Expected back-rich ({r_back:.4f}) > front-rich ({r_front:.4f})"


# ---------------------------------------------------------------------------
# SORCResult accessor tests
# ---------------------------------------------------------------------------


class TestSORCResult:
    def test_n_tokens_correct(self):
        n = 15
        hs = _make_hidden_states(n, 32, "random", n_layers=3)
        result = compute_sorc(hs, context_label="test")
        assert result.n_tokens == n

    def test_n_layers_correct(self):
        hs = _make_hidden_states(15, 32, "random", n_layers=5)
        result = compute_sorc(hs)
        assert result.n_layers == 5
        assert len(result.layer_scores) == 5

    def test_context_label_stored(self):
        hs = _make_hidden_states(10, 32, "random")
        result = compute_sorc(hs, context_label="my_context")
        assert result.context_label == "my_context"

    def test_deep_layer_mean_is_mean_of_top_half(self):
        hs = _make_hidden_states(20, 32, "random", n_layers=6)
        result = compute_sorc(hs)
        expected = float(np.mean(result.layer_scores[3:]))
        assert abs(result.deep_layer_mean - expected) < 1e-6

    def test_token_variance_non_negative(self):
        hs = _make_hidden_states(20, 32, "structured")
        result = compute_sorc(hs)
        assert result.token_variance >= 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSORCEdgeCases:
    def test_single_layer(self):
        hs = _make_hidden_states(10, 32, "random", n_layers=1)
        result = compute_sorc(hs)
        assert 0.0 <= result.sorc <= 1.0
        assert len(result.layer_scores) == 1

    def test_two_tokens(self):
        hs = _make_hidden_states(2, 32, "random", n_layers=2)
        result = compute_sorc(hs)
        assert 0.0 <= result.sorc <= 1.0

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError):
            compute_sorc([])

    def test_large_hidden_dim(self):
        hs = _make_hidden_states(10, 4096, "random", n_layers=2)
        result = compute_sorc(hs)
        assert 0.0 <= result.sorc <= 1.0

    def test_reproducible_with_same_input(self):
        hs = _make_hidden_states(20, 64, "structured", seed=99)
        r1 = compute_sorc(hs)
        r2 = compute_sorc(hs)
        assert r1.sorc == r2.sorc
