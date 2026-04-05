"""
Bayesian Uncertainty Quantification for Reasoning

Provides calibrated confidence estimates for reasoning outputs using:
1. Bayesian inference for belief updates
2. Ensemble disagreement for epistemic uncertainty
3. Calibration metrics (Brier score, ECE)
4. Confidence intervals for predictions

Based on:
- Bayesian Deep Learning (Gal & Ghahramani, 2016)
- Uncertainty Calibration (Guo et al., 2017)
- Conformal Prediction (Shafer & Vovk, 2008)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class UncertaintyEstimate:
    """Complete uncertainty estimate for a prediction"""
    point_estimate: Any  # The prediction
    confidence: float  # Overall confidence 0-1
    aleatoric_uncertainty: float  # Data/irreducible uncertainty
    epistemic_uncertainty: float  # Model/knowledge uncertainty
    credible_interval: Tuple[float, float]  # Bayesian credible interval
    calibrated_confidence: float  # After calibration adjustment
    reasoning_trace_confidence: float  # Confidence in the reasoning itself


class BayesianBeliefUpdater:
    """
    Updates beliefs using Bayes' theorem.

    P(H|E) = P(E|H) * P(H) / P(E)

    Maintains calibrated probability estimates.
    """

    def __init__(self, prior: float = 0.5):
        self.beliefs: Dict[str, float] = {}  # hypothesis -> probability
        self.evidence_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.default_prior = prior

    def update_belief(
        self,
        hypothesis: str,
        evidence: str,
        likelihood_ratio: float,  # P(E|H) / P(E|not H)
        evidence_strength: float = 1.0
    ) -> float:
        """
        Update belief in hypothesis given new evidence.

        Uses log-odds form for numerical stability:
        log_odds_posterior = log_odds_prior + log(likelihood_ratio)
        """
        # Get current belief (prior)
        prior = self.beliefs.get(hypothesis, self.default_prior)

        # Convert to log-odds
        prior_odds = prior / (1 - prior + 1e-10)
        log_prior_odds = math.log(prior_odds + 1e-10)

        # Update with evidence (scaled by strength)
        effective_lr = likelihood_ratio ** evidence_strength
        log_posterior_odds = log_prior_odds + math.log(effective_lr + 1e-10)

        # Convert back to probability
        posterior_odds = math.exp(min(log_posterior_odds, 20))  # Cap to prevent overflow
        posterior = posterior_odds / (1 + posterior_odds)

        # Clamp to valid range
        posterior = max(0.001, min(0.999, posterior))

        # Store update
        self.beliefs[hypothesis] = posterior
        self.evidence_history[hypothesis].append((evidence, likelihood_ratio))

        return posterior

    def get_belief(self, hypothesis: str) -> float:
        """Get current belief probability"""
        return self.beliefs.get(hypothesis, self.default_prior)

    def get_credible_interval(
        self,
        hypothesis: str,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute Bayesian credible interval using Beta distribution approximation.

        Based on number of supporting/opposing evidence items.
        """
        history = self.evidence_history.get(hypothesis, [])

        if not history:
            # Uninformative prior
            alpha = 1
            beta = 1
        else:
            # Count effective evidence
            supporting = sum(1 for _, lr in history if lr > 1)
            opposing = sum(1 for _, lr in history if lr < 1)

            # Beta distribution parameters
            alpha = supporting + 1
            beta = opposing + 1

        # Compute credible interval using Beta quantiles
        from scipy import stats
        beta_dist = stats.beta(alpha, beta)

        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q

        return (beta_dist.ppf(lower_q), beta_dist.ppf(upper_q))


class EnsembleUncertainty:
    """
    Estimates uncertainty through ensemble disagreement.

    Epistemic uncertainty = disagreement between ensemble members
    Aleatoric uncertainty = average uncertainty of each member
    """

    def __init__(self, n_members: int = 5):
        self.n_members = n_members
        self.ensemble_predictions: List[List[float]] = []

    def compute_uncertainty(
        self,
        member_predictions: List[float]
    ) -> Tuple[float, float, float]:
        """
        Compute uncertainty from ensemble predictions.

        Returns: (mean_prediction, epistemic_uncertainty, aleatoric_uncertainty)
        """
        predictions = np.array(member_predictions)

        # Mean prediction
        mean_pred = np.mean(predictions)

        # Epistemic uncertainty: variance across ensemble
        epistemic = np.var(predictions)

        # Aleatoric uncertainty: average entropy of predictions
        # For binary: H = -p*log(p) - (1-p)*log(1-p)
        def entropy(p):
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -p * np.log(p) - (1 - p) * np.log(1 - p)

        aleatoric = np.mean([entropy(p) for p in predictions])

        return mean_pred, epistemic, aleatoric

    def sample_predictions(
        self,
        base_prediction: float,
        noise_scale: float = 0.1
    ) -> List[float]:
        """
        Generate ensemble predictions by adding calibrated noise.

        In practice, this would come from multiple model runs.
        """
        predictions = []
        for _ in range(self.n_members):
            noisy = base_prediction + np.random.normal(0, noise_scale)
            noisy = np.clip(noisy, 0, 1)
            predictions.append(noisy)
        return predictions


class CalibrationMetrics:
    """
    Measures and improves confidence calibration.

    A well-calibrated model's confidence should match accuracy:
    - 80% confident predictions should be correct 80% of the time
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.predictions: List[float] = []
        self.actuals: List[int] = []  # 1 = correct, 0 = incorrect

    def add_prediction(self, confidence: float, correct: bool):
        """Record a prediction for calibration analysis"""
        self.predictions.append(confidence)
        self.actuals.append(1 if correct else 0)

    def compute_ece(self) -> float:
        """
        Compute Expected Calibration Error.

        ECE = Σ (n_b / N) * |accuracy_b - confidence_b|

        Lower is better. 0 = perfectly calibrated.
        """
        if not self.predictions:
            return 0.0

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                avg_confidence = np.mean(predictions[in_bin])
                avg_accuracy = np.mean(actuals[in_bin])
                ece += (n_in_bin / len(predictions)) * abs(avg_accuracy - avg_confidence)

        return ece

    def compute_brier_score(self) -> float:
        """
        Compute Brier Score.

        Brier = (1/N) * Σ (confidence - actual)²

        Lower is better. 0 = perfect predictions.
        """
        if not self.predictions:
            return 0.0

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)

        return np.mean((predictions - actuals) ** 2)

    def get_calibration_curve(self) -> Tuple[List[float], List[float]]:
        """
        Get calibration curve data for plotting.

        Returns: (mean_confidence_per_bin, mean_accuracy_per_bin)
        """
        if not self.predictions:
            return [], []

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        confidences = []
        accuracies = []

        for i in range(self.n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                confidences.append(np.mean(predictions[in_bin]))
                accuracies.append(np.mean(actuals[in_bin]))

        return confidences, accuracies

    def compute_calibration_adjustment(self) -> Callable[[float], float]:
        """
        Learn a calibration adjustment function using isotonic regression.

        Returns a function that maps raw confidence to calibrated confidence.
        """
        if len(self.predictions) < 10:
            return lambda x: x  # Not enough data

        try:
            from sklearn.isotonic import IsotonicRegression

            predictions = np.array(self.predictions)
            actuals = np.array(self.actuals)

            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(predictions, actuals)

            return lambda x: float(iso_reg.predict([[x]])[0])

        except ImportError:
            # Fallback: simple linear adjustment
            mean_conf = np.mean(self.predictions)
            mean_acc = np.mean(self.actuals)

            # Adjust toward mean accuracy
            def linear_adjust(x):
                adjusted = x + (mean_acc - mean_conf)
                return max(0, min(1, adjusted))

            return linear_adjust


class ReasoningUncertaintyEstimator:
    """
    Complete uncertainty estimation for reasoning tasks.

    Combines:
    - Bayesian belief updating
    - Ensemble uncertainty
    - Calibration correction
    - Reasoning trace confidence
    """

    def __init__(self):
        self.belief_updater = BayesianBeliefUpdater()
        self.ensemble = EnsembleUncertainty()
        self.calibrator = CalibrationMetrics()

    def estimate_uncertainty(
        self,
        prediction: Any,
        raw_confidence: float,
        reasoning_steps: List[Dict[str, Any]],
        problem_type: str,
        historical_accuracy: Optional[float] = None
    ) -> UncertaintyEstimate:
        """
        Compute comprehensive uncertainty estimate for a reasoning prediction.
        """
        # 1. Compute reasoning trace confidence
        trace_confidence = self._compute_trace_confidence(reasoning_steps)

        # 2. Adjust for problem type (some are inherently harder)
        type_adjustment = self._get_type_adjustment(problem_type)
        adjusted_confidence = raw_confidence * type_adjustment

        # 3. Ensemble uncertainty
        ensemble_preds = self.ensemble.sample_predictions(adjusted_confidence, noise_scale=0.15)
        mean_pred, epistemic, aleatoric = self.ensemble.compute_uncertainty(ensemble_preds)

        # 4. Compute credible interval
        # Use reasoning steps as pseudo-evidence
        evidence_count = len(reasoning_steps)
        ci_width = 0.2 / (1 + evidence_count * 0.1)  # Narrower with more evidence
        credible_interval = (
            max(0, mean_pred - ci_width),
            min(1, mean_pred + ci_width)
        )

        # 5. Apply calibration if we have historical data
        if historical_accuracy is not None:
            calibrated = self._apply_historical_calibration(mean_pred, historical_accuracy)
        else:
            calibrated = mean_pred

        # 6. Final confidence combines all factors
        final_confidence = (
            calibrated * 0.5 +
            trace_confidence * 0.3 +
            (1 - epistemic) * 0.2
        )

        return UncertaintyEstimate(
            point_estimate=prediction,
            confidence=final_confidence,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            credible_interval=credible_interval,
            calibrated_confidence=calibrated,
            reasoning_trace_confidence=trace_confidence
        )

    def _compute_trace_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """Compute confidence based on the quality of reasoning trace"""
        if not reasoning_steps:
            return 0.5

        # Factors that increase confidence
        factors = []

        # 1. Number of steps (more complete reasoning)
        step_factor = min(len(reasoning_steps) / 5, 1.0)
        factors.append(step_factor)

        # 2. Step confidence (if provided)
        step_confidences = [s.get('confidence', 0.5) for s in reasoning_steps]
        avg_step_conf = np.mean(step_confidences)
        factors.append(avg_step_conf)

        # 3. Presence of key reasoning types
        step_types = [s.get('type', '') for s in reasoning_steps]
        has_extract = any('extract' in t for t in step_types)
        has_derive = any('apply' in t or 'derive' in t for t in step_types)
        has_conclude = any('conclude' in t for t in step_types)

        completeness = (has_extract + has_derive + has_conclude) / 3
        factors.append(completeness)

        return np.mean(factors)

    def _get_type_adjustment(self, problem_type: str) -> float:
        """Get confidence adjustment based on problem type"""
        # Some problem types are inherently harder
        adjustments = {
            'deductive': 1.0,  # Rule-based, high confidence possible
            'inductive': 0.8,  # Pattern-based, some uncertainty
            'abductive': 0.7,  # Best explanation, inherent uncertainty
            'analogical': 0.75,
            'causal': 0.7,  # Causation is hard
            'relational': 0.85,
            'temporal': 0.9,
            'spatial': 0.9,
            'probabilistic': 0.65,  # Probability is tricky
        }
        return adjustments.get(problem_type, 0.8)

    def _apply_historical_calibration(
        self,
        raw_confidence: float,
        historical_accuracy: float
    ) -> float:
        """Adjust confidence based on historical accuracy"""
        # If we've been overconfident, reduce confidence
        # If we've been underconfident, increase confidence
        adjustment = historical_accuracy / (raw_confidence + 1e-10)
        adjustment = np.clip(adjustment, 0.5, 2.0)

        calibrated = raw_confidence * adjustment
        return np.clip(calibrated, 0, 1)

    def record_outcome(self, predicted_confidence: float, was_correct: bool):
        """Record an outcome for calibration tracking"""
        self.calibrator.add_prediction(predicted_confidence, was_correct)

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration metrics report"""
        return {
            'ece': self.calibrator.compute_ece(),
            'brier_score': self.calibrator.compute_brier_score(),
            'n_predictions': len(self.calibrator.predictions),
            'mean_confidence': np.mean(self.calibrator.predictions) if self.calibrator.predictions else 0,
            'mean_accuracy': np.mean(self.calibrator.actuals) if self.calibrator.actuals else 0
        }
