"""
Second-Order Relational Coherence (SORC) metric.

Operationalizes the contextual activation mechanism from Informational Substrate
Convergence theory. Measures the structural depth of a context by computing the
spectral entropy of the second-order relational matrix across transformer layers.

References:
    Figueira, A. P. (2026). Informational Substrate Convergence. PAPER.md.
    Kriegeskorte, N., et al. (2008). Representational Similarity Analysis. Frontiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SORCResult:
    """Result from a single SORC computation."""

    sorc: float
    layer_scores: list[float]
    n_tokens: int
    n_layers: int
    context_label: str = ""

    @property
    def deep_layer_mean(self) -> float:
        """Mean SORC score over the top half of layers (more informative)."""
        half = len(self.layer_scores) // 2
        return float(np.mean(self.layer_scores[half:]))

    @property
    def token_variance(self) -> float:
        """Variance across per-layer scores — proxy for structural heterogeneity."""
        return float(np.var(self.layer_scores))


def compute_sorc(
    hidden_states: list[torch.Tensor],
    context_label: str = "",
    eps: float = 1e-8,
) -> SORCResult:
    """
    Compute SORC from a list of per-layer hidden state tensors.

    Args:
        hidden_states: List of tensors, one per layer, each shape (n_tokens, hidden_dim).
                       Pass only the non-embedding layers (transformer blocks).
        context_label:  Optional label for logging and result identification.
        eps:            Small value to avoid division by zero.

    Returns:
        SORCResult with the scalar SORC score and per-layer breakdown.

    Notes:
        SORC is bounded in [0, 1].
        Degenerate/repetitive inputs produce near-rank-1 S matrices → low entropy → low SORC.
        Random inputs produce near-uniform eigenvalue distributions → high entropy → high SORC.
        A model-specific random baseline must be established before interpreting absolute values.
        The meaningful comparison is always SORC(context) vs SORC(random baseline).

        Length sensitivity: SORC normalises spectral entropy by log(n) where n is the
        token count. Contexts of different lengths are not directly comparable without
        controlling for length. Always compare contexts at matched token counts, or use
        the domain-matched comparison (expert vs shallow on same topic at same length).

    Complexity:
        O(n² * d) to build relational matrices, O(n³) for eigendecomposition per layer.
        Tractable for n ≤ 512 tokens. Subsample longer contexts before calling.
    """
    L = len(hidden_states)
    if L == 0:
        raise ValueError("hidden_states must contain at least one layer.")

    layer_sd_scores: list[float] = []

    for l_idx, H in enumerate(hidden_states):
        H = H.float()
        n = H.shape[0]

        if n < 2:
            layer_sd_scores.append(0.0)
            continue

        # --- Step 1: normalise hidden states ---
        norms = torch.norm(H, dim=1, keepdim=True).clamp(min=eps)
        H_hat = H / norms  # (n, d)

        # --- Step 2: first-order relational matrix R ---
        R = H_hat @ H_hat.T  # (n, n), values in [-1, 1]

        # --- Step 3: normalise rows of R to get relational profiles ---
        R_norms = torch.norm(R, dim=1, keepdim=True).clamp(min=eps)
        R_hat = R / R_norms  # (n, n)

        # --- Step 4: second-order matrix S ---
        S = R_hat @ R_hat.T  # (n, n)

        # --- Step 5: eigenvalue distribution ---
        # eigvalsh returns real eigenvalues for symmetric matrices, ascending order
        eigenvalues = torch.linalg.eigvalsh(S)
        eigenvalues = eigenvalues.clamp(min=0)
        total = eigenvalues.sum().clamp(min=eps)
        p = eigenvalues / total  # probability distribution over eigenvalues

        # --- Step 6: normalised spectral entropy ---
        # Shannon entropy normalised by log(n) so SD ∈ [0, 1]
        log_n = np.log(n)
        if log_n < eps:
            layer_sd_scores.append(0.0)
            continue

        entropy = -(p * (p + eps).log()).sum().item()
        sd = float(entropy / log_n)
        sd = max(0.0, min(1.0, sd))  # clip numerical noise
        layer_sd_scores.append(sd)

    # --- Step 7: depth-weighted integration ---
    # Weights: l / sum(1..L) = l * 2 / (L(L+1)), sums to 1
    normalizer = 2.0 / (L * (L + 1))
    sorc = normalizer * sum((l + 1) * sd for l, sd in enumerate(layer_sd_scores))
    sorc = max(0.0, min(1.0, sorc))

    return SORCResult(
        sorc=sorc,
        layer_scores=layer_sd_scores,
        n_tokens=hidden_states[0].shape[0],
        n_layers=L,
        context_label=context_label,
    )


def extract_hidden_states(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_tokens: int = 512,
) -> list[torch.Tensor]:
    """
    Run a HuggingFace causal LM forward pass and return per-layer hidden states.

    Args:
        model:      HuggingFace AutoModelForCausalLM (must be loaded with
                    output_hidden_states=True or config set accordingly).
        tokenizer:  Matching tokenizer.
        text:       Input context string.
        device:     Torch device string.
        max_tokens: Truncation limit. SORC is O(n³) per layer; keep ≤ 512.

    Returns:
        List of tensors (n_tokens, hidden_dim), one per transformer block layer.
        The embedding layer (index 0 in model output) is excluded.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple: (embedding, layer_1, ..., layer_L)
    # We drop the embedding layer (index 0) — it has no relational computation yet.
    hidden_states = [h.squeeze(0).cpu() for h in outputs.hidden_states[1:]]
    return hidden_states


def random_baseline(
    model,
    tokenizer,
    n_tokens: int = 200,
    n_samples: int = 5,
    device: str = "cpu",
    seed: int = 42,
) -> float:
    """
    Estimate the model-specific SORC baseline on random token sequences.

    Random inputs produce near-uniform eigenvalue distributions and therefore
    high SORC. All context SORC values must be compared against this baseline,
    not against zero.

    Args:
        model:     HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        n_tokens:  Length of random sequences to generate.
        n_samples: Number of random sequences to average over.
        device:    Torch device string.
        seed:      RNG seed for reproducibility.

    Returns:
        Mean SORC score across random sequences — the upper reference point.
    """
    rng = np.random.default_rng(seed)
    vocab_size = tokenizer.vocab_size
    scores = []

    for _ in range(n_samples):
        token_ids = rng.integers(low=0, high=vocab_size, size=n_tokens).tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        try:
            hs = extract_hidden_states(model, tokenizer, text, device=device, max_tokens=n_tokens)
            result = compute_sorc(hs, context_label="random_baseline")
            scores.append(result.sorc)
        except Exception as e:
            logger.warning("Baseline sample failed: %s", e)

    if not scores:
        raise RuntimeError("All random baseline samples failed.")

    return float(np.mean(scores))
