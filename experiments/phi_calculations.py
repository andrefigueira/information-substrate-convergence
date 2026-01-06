"""
Multiple Phi (Integrated Information) Calculation Methods

This module implements several approximations of phi to compare which
best predicts reasoning capability.

Methods:
1. Simple variance-based (baseline)
2. Mutual Information based
3. Effective Information (EI)
4. Geometric Integrated Information (Phi-G)
5. Stochastic Interaction (SI)

References:
- Tononi et al. (2008) - Original IIT phi
- Oizumi et al. (2014) - Phi geometric
- Barrett & Seth (2011) - Practical measures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from collections import defaultdict


class PhiCalculator:
    """Base class for phi calculations"""

    def __init__(self, name: str):
        self.name = name
        self.history: List[float] = []

    def calculate(self, activations: np.ndarray, connectivity: np.ndarray) -> float:
        """Calculate phi from activation pattern and connectivity matrix"""
        raise NotImplementedError

    def record(self, phi: float):
        """Record phi value to history"""
        self.history.append(phi)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about phi history"""
        if not self.history:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
        return {
            'mean': np.mean(self.history),
            'std': np.std(self.history),
            'min': np.min(self.history),
            'max': np.max(self.history),
            'n': len(self.history)
        }


class SimpleVariancePhi(PhiCalculator):
    """
    Baseline: Phi based on variance of activations and connectivity.

    This is our original simple measure.
    """

    def __init__(self):
        super().__init__("simple_variance")

    def calculate(self, activations: np.ndarray, connectivity: np.ndarray) -> float:
        if len(activations) < 2:
            return 0.0

        # Whole system variance
        whole_var = np.var(activations) + 1e-8

        # Partition and calculate parts variance
        n = len(activations)
        mid = n // 2
        part1_var = np.var(activations[:mid]) + 1e-8 if mid > 0 else 1e-8
        part2_var = np.var(activations[mid:]) + 1e-8 if n - mid > 0 else 1e-8

        parts_sum = part1_var + part2_var

        # Integration measure
        if parts_sum > 0:
            integration = 1.0 - (whole_var / parts_sum)
            phi = max(0.0, min(1.0, integration))
        else:
            phi = 0.0

        # Add connectivity factor
        if connectivity.size > 0:
            avg_conn = np.mean(np.abs(connectivity))
            phi = 0.7 * phi + 0.3 * avg_conn

        self.record(phi)
        return phi


class MutualInformationPhi(PhiCalculator):
    """
    Phi based on mutual information between partitions.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)

    High MI = partitions share information = high integration
    """

    def __init__(self):
        super().__init__("mutual_information")

    def _entropy(self, x: np.ndarray, bins: int = 10) -> float:
        """Calculate entropy of a distribution"""
        if len(x) < 2:
            return 0.0
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist + 1e-10)) / bins

    def _joint_entropy(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Calculate joint entropy H(X,Y)"""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        hist, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist + 1e-10)) / (bins * bins)

    def calculate(self, activations: np.ndarray, connectivity: np.ndarray) -> float:
        n = len(activations)
        if n < 4:
            return 0.0

        # Partition into two halves of equal length
        mid = n // 2
        part1 = activations[:mid]
        part2 = activations[mid:mid + len(part1)]  # Ensure equal length

        if len(part1) != len(part2) or len(part1) < 2:
            return 0.0

        # Calculate mutual information
        h1 = self._entropy(part1)
        h2 = self._entropy(part2)
        h_joint = self._joint_entropy(part1, part2)

        mi = h1 + h2 - h_joint

        # Normalize to [0, 1]
        max_mi = min(h1, h2) if min(h1, h2) > 0 else 1.0
        phi = mi / max_mi if max_mi > 0 else 0.0
        phi = max(0.0, min(1.0, phi))

        self.record(phi)
        return phi


class EffectiveInformationPhi(PhiCalculator):
    """
    Effective Information (EI) based phi.

    EI measures how much information a system generates above chance.
    EI = H(effect | do(cause=max_entropy)) - H(effect | do(cause=observed))

    Approximated as: how much the output distribution differs from uniform
    given the connectivity structure.
    """

    def __init__(self):
        super().__init__("effective_information")

    def calculate(self, activations: np.ndarray, connectivity: np.ndarray) -> float:
        if len(activations) < 2 or connectivity.size == 0:
            return 0.0

        # Simulate "do" intervention: what would outputs be with max entropy input?
        n = len(activations)

        # Current output distribution (from activations)
        current_dist = activations / (np.sum(activations) + 1e-8)

        # Max entropy distribution (uniform)
        uniform_dist = np.ones(n) / n

        # KL divergence from uniform (how far from max entropy)
        # Higher = more structured = more effective information
        kl_div = np.sum(current_dist * np.log2((current_dist + 1e-10) / (uniform_dist + 1e-10)))

        # Normalize by max possible KL (log(n))
        max_kl = np.log2(n) if n > 1 else 1.0
        ei = kl_div / max_kl if max_kl > 0 else 0.0

        # Weight by connectivity strength
        conn_strength = np.mean(np.abs(connectivity)) if connectivity.size > 0 else 0.5
        phi = ei * conn_strength

        phi = max(0.0, min(1.0, phi))
        self.record(phi)
        return phi


class GeometricPhi(PhiCalculator):
    """
    Geometric Integrated Information (Phi-G).

    Based on Oizumi et al. (2014) - measures the geometric distance
    between the whole system and its disconnected parts.

    Phi-G = D(p(whole) || product of p(parts))

    where D is a divergence measure.
    """

    def __init__(self):
        super().__init__("geometric_phi")

    def calculate(self, activations: np.ndarray, connectivity: np.ndarray) -> float:
        n = len(activations)
        if n < 4:
            return 0.0

        # Partition into two halves of equal length
        mid = n // 2
        part1 = activations[:mid]
        part2 = activations[mid:mid + len(part1)]  # Ensure equal length

        if len(part1) != len(part2) or len(part1) < 2:
            return 0.0

        # Calculate covariance structure
        if len(part1) > 1 and len(part2) > 1:
            # Cross-correlation between partitions
            cross_corr = np.corrcoef(part1, part2)[0, 1]

            if np.isnan(cross_corr):
                cross_corr = 0

            # Phi-G approximation: how much correlation exists between partitions
            # that would be lost if we disconnected them
            phi_g = np.abs(cross_corr)

            # Weight by connectivity between partitions
            if connectivity.size > 0:
                conn_matrix = connectivity.reshape(int(np.sqrt(connectivity.size)), -1) if connectivity.ndim == 1 else connectivity
                if conn_matrix.shape[0] > mid and conn_matrix.shape[1] > mid:
                    cross_conn = np.mean(np.abs(conn_matrix[:mid, mid:]))
                    phi_g = 0.6 * phi_g + 0.4 * cross_conn
        else:
            phi_g = 0.0

        phi_g = max(0.0, min(1.0, phi_g))
        self.record(phi_g)
        return phi_g


class StochasticInteractionPhi(PhiCalculator):
    """
    Stochastic Interaction (SI) based phi.

    SI measures the average pairwise interaction between elements,
    weighted by how much they deviate from independence.

    SI = sum_ij |p(xi, xj) - p(xi)p(xj)|
    """

    def __init__(self):
        super().__init__("stochastic_interaction")

    def calculate(self, activations: np.ndarray, connectivity: np.ndarray) -> float:
        n = len(activations)
        if n < 2:
            return 0.0

        # Calculate pairwise interactions
        total_interaction = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # "Probability" approximated by activation
                pi = activations[i]
                pj = activations[j]

                # Joint "probability" - use connectivity as proxy
                if connectivity.size > 0:
                    conn_flat = connectivity.flatten()
                    idx = i * n + j if i * n + j < len(conn_flat) else 0
                    pij = np.abs(conn_flat[idx]) * (pi + pj) / 2
                else:
                    pij = pi * pj  # Independence assumption

                # Interaction = deviation from independence
                interaction = np.abs(pij - pi * pj)
                total_interaction += interaction
                count += 1

        if count > 0:
            phi = total_interaction / count
        else:
            phi = 0.0

        phi = max(0.0, min(1.0, phi))
        self.record(phi)
        return phi


class CompositePhiCalculator:
    """
    Calculates multiple phi measures and tracks correlations with accuracy.
    """

    def __init__(self):
        self.calculators = {
            'simple': SimpleVariancePhi(),
            'mutual_info': MutualInformationPhi(),
            'effective_info': EffectiveInformationPhi(),
            'geometric': GeometricPhi(),
            'stochastic': StochasticInteractionPhi()
        }
        self.accuracy_records: List[Tuple[Dict[str, float], bool]] = []

    def calculate_all(self, activations: np.ndarray, connectivity: np.ndarray) -> Dict[str, float]:
        """Calculate all phi measures"""
        results = {}
        for name, calc in self.calculators.items():
            results[name] = calc.calculate(activations, connectivity)
        return results

    def record_outcome(self, phi_values: Dict[str, float], was_correct: bool):
        """Record phi values and whether reasoning was correct"""
        self.accuracy_records.append((phi_values.copy(), was_correct))

    def get_correlations(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation between each phi measure and accuracy"""
        if len(self.accuracy_records) < 10:
            return {name: {'correlation': 0, 'p_value': 1, 'n': 0}
                    for name in self.calculators}

        results = {}
        accuracies = np.array([1.0 if r[1] else 0.0 for r in self.accuracy_records])

        for name in self.calculators:
            phi_values = np.array([r[0].get(name, 0) for r in self.accuracy_records])

            if np.std(phi_values) > 0 and np.std(accuracies) > 0:
                corr, p_value = stats.pearsonr(phi_values, accuracies)
            else:
                corr, p_value = 0.0, 1.0

            results[name] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'n': len(self.accuracy_records),
                'mean_phi': float(np.mean(phi_values)),
                'std_phi': float(np.std(phi_values))
            }

        return results

    def get_best_predictor(self) -> Tuple[str, float]:
        """Return the phi measure that best predicts accuracy"""
        correlations = self.get_correlations()
        best_name = max(correlations.keys(),
                       key=lambda k: abs(correlations[k]['correlation']))
        return best_name, correlations[best_name]['correlation']
