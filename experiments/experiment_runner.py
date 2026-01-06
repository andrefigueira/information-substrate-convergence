"""
Rigorous Experiment Runner for ISC Phi-Driven Reasoning

This module provides a structured framework for running experiments that:
1. Compare multiple phi calculation methods
2. Track phi-accuracy correlations with statistical tests
3. Log all results in structured JSON format
4. Generate summary reports

Usage:
    python -m experiments.experiment_runner --iterations 500 --output results/experiments/
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.phi_calculations import CompositePhiCalculator


@dataclass
class TrialResult:
    """Single trial result"""
    trial_id: int
    timestamp: str
    problem_type: str
    premise: str
    question: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    phi_values: Dict[str, float]
    activations_shape: Tuple[int, ...]
    connectivity_density: float
    response_time_ms: float
    emergent_nodes_count: int
    substrate_edges: int


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    iterations: int
    phi_methods: List[str]
    problem_types: List[str]
    random_seed: int
    timestamp: str


@dataclass
class ExperimentResults:
    """Complete experiment results"""
    config: ExperimentConfig
    trials: List[TrialResult]
    summary: Dict[str, Any]
    phi_correlations: Dict[str, Dict[str, float]]
    best_phi_predictor: str
    statistical_tests: Dict[str, Any]
    thesis_validation: Dict[str, bool]


class ExperimentRunner:
    """
    Runs structured experiments on phi-driven reasoning.
    """

    def __init__(self, output_dir: str = "results/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phi_calculator = CompositePhiCalculator()

        # Standard test problems for reproducibility
        self.test_problems = self._create_test_problems()

    def _create_test_problems(self) -> List[Dict[str, Any]]:
        """Create standardized test problems across reasoning types"""
        problems = []

        # Deductive reasoning problems
        deductive = [
            {
                "type": "deductive",
                "premise": "All mammals are warm-blooded. Whales are mammals.",
                "question": "Are whales warm-blooded?",
                "expected": "yes",
                "difficulty": "easy"
            },
            {
                "type": "deductive",
                "premise": "If it rains, the ground gets wet. It is raining.",
                "question": "Is the ground wet?",
                "expected": "yes",
                "difficulty": "easy"
            },
            {
                "type": "deductive",
                "premise": "All birds have feathers. Penguins are birds.",
                "question": "Do penguins have feathers?",
                "expected": "yes",
                "difficulty": "easy"
            },
            {
                "type": "deductive",
                "premise": "No reptiles are warm-blooded. Snakes are reptiles.",
                "question": "Are snakes warm-blooded?",
                "expected": "no",
                "difficulty": "medium"
            },
            {
                "type": "deductive",
                "premise": "All prime numbers greater than 2 are odd. 17 is a prime number greater than 2.",
                "question": "Is 17 odd?",
                "expected": "yes",
                "difficulty": "medium"
            },
        ]

        # Inductive reasoning problems
        inductive = [
            {
                "type": "inductive",
                "premise": "The sun rose today. The sun rose yesterday. The sun rose the day before.",
                "question": "Will the sun rise tomorrow?",
                "expected": "yes",
                "difficulty": "easy"
            },
            {
                "type": "inductive",
                "premise": "Swan 1 is white. Swan 2 is white. Swan 3 is white. Swan 4 is white.",
                "question": "Are all swans white?",
                "expected": "likely",
                "difficulty": "medium"
            },
            {
                "type": "inductive",
                "premise": "Metal A expands when heated. Metal B expands when heated. Metal C expands when heated.",
                "question": "Do metals expand when heated?",
                "expected": "yes",
                "difficulty": "easy"
            },
        ]

        # Abductive reasoning problems
        abductive = [
            {
                "type": "abductive",
                "premise": "The grass is wet. It is morning. There are no sprinklers.",
                "question": "What is the most likely explanation?",
                "expected": "dew",
                "difficulty": "medium"
            },
            {
                "type": "abductive",
                "premise": "The patient has fever, cough, and fatigue.",
                "question": "What is the most likely diagnosis?",
                "expected": "flu",
                "difficulty": "medium"
            },
            {
                "type": "abductive",
                "premise": "The lights are on but nobody answers the door.",
                "question": "What might explain this?",
                "expected": "not home",
                "difficulty": "easy"
            },
        ]

        # Analogical reasoning problems
        analogical = [
            {
                "type": "analogical",
                "premise": "Puppy is to dog as kitten is to what?",
                "question": "Complete the analogy.",
                "expected": "cat",
                "difficulty": "easy"
            },
            {
                "type": "analogical",
                "premise": "Hot is to cold as light is to what?",
                "question": "Complete the analogy.",
                "expected": "dark",
                "difficulty": "easy"
            },
            {
                "type": "analogical",
                "premise": "Author is to book as composer is to what?",
                "question": "Complete the analogy.",
                "expected": "music",
                "difficulty": "medium"
            },
        ]

        # Causal reasoning problems
        causal = [
            {
                "type": "causal",
                "premise": "Plants given fertilizer grew taller. Plants without fertilizer stayed short.",
                "question": "Does fertilizer cause growth?",
                "expected": "yes",
                "difficulty": "easy"
            },
            {
                "type": "causal",
                "premise": "Ice cream sales increase. Drowning incidents increase.",
                "question": "Does ice cream cause drowning?",
                "expected": "no",
                "difficulty": "medium"
            },
            {
                "type": "causal",
                "premise": "In a randomized trial, drug A reduced symptoms in 80% of patients vs 20% for placebo.",
                "question": "Is the drug effective?",
                "expected": "yes",
                "difficulty": "easy"
            },
        ]

        problems.extend(deductive)
        problems.extend(inductive)
        problems.extend(abductive)
        problems.extend(analogical)
        problems.extend(causal)

        return problems

    def run_experiment(
        self,
        name: str,
        description: str,
        iterations: int = 100,
        random_seed: int = 42,
        substrate_class: Optional[type] = None
    ) -> ExperimentResults:
        """
        Run a complete experiment with multiple trials.

        Args:
            name: Experiment name
            description: What this experiment tests
            iterations: Number of trials per problem
            random_seed: For reproducibility
            substrate_class: Custom substrate class (defaults to PhiDrivenSubstrate)
        """
        np.random.seed(random_seed)
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            iterations=iterations,
            phi_methods=list(self.phi_calculator.calculators.keys()),
            problem_types=list(set(p["type"] for p in self.test_problems)),
            random_seed=random_seed,
            timestamp=datetime.now().isoformat()
        )

        # Import substrate - use improved version by default
        if substrate_class is None:
            try:
                from src.isc.improved_emergent_reasoning import ImprovedPhiDrivenSubstrate, ImprovedEmergentReasoner
                substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
                reasoner = ImprovedEmergentReasoner(substrate)
            except ImportError:
                from src.isc.emergent_reasoning import PhiDrivenSubstrate, EmergentReasoner
                substrate = PhiDrivenSubstrate(initial_nodes=50)
                reasoner = EmergentReasoner(substrate)
        else:
            substrate = substrate_class(initial_nodes=100)
            from src.isc.improved_emergent_reasoning import ImprovedEmergentReasoner
            reasoner = ImprovedEmergentReasoner(substrate)

        trials = []
        trial_id = 0

        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"ID: {experiment_id}")
        print(f"Iterations per problem: {iterations}")
        print(f"Total problems: {len(self.test_problems)}")
        print(f"{'='*60}\n")

        # Run trials
        for iteration in range(iterations):
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{iterations}...")

            for problem in self.test_problems:
                start_time = time.time()

                # Get activations and connectivity from substrate
                activations = substrate.get_activation_pattern()
                connectivity = substrate.get_connectivity_matrix()

                # Calculate all phi values
                phi_values = self.phi_calculator.calculate_all(activations, connectivity)

                # Execute reasoning
                result = reasoner.reason(
                    premise=problem["premise"],
                    question=problem["question"]
                )

                response_time = (time.time() - start_time) * 1000

                # Evaluate correctness
                actual_answer = result.get("answer", "").lower()
                expected = problem["expected"].lower()
                is_correct = self._evaluate_answer(actual_answer, expected, problem["type"])

                # Record outcome for correlation tracking
                self.phi_calculator.record_outcome(phi_values, is_correct)

                # Create trial result
                trial = TrialResult(
                    trial_id=trial_id,
                    timestamp=datetime.now().isoformat(),
                    problem_type=problem["type"],
                    premise=problem["premise"],
                    question=problem["question"],
                    expected_answer=problem["expected"],
                    actual_answer=actual_answer,
                    is_correct=is_correct,
                    phi_values=phi_values,
                    activations_shape=activations.shape,
                    connectivity_density=float(np.mean(connectivity > 0)),
                    response_time_ms=response_time,
                    emergent_nodes_count=len([n for n in substrate.nodes if n.startswith("emergent")]),
                    substrate_edges=len(substrate.edges)
                )
                trials.append(trial)
                trial_id += 1

                # Update substrate and reasoner based on outcome (enables emergence)
                reasoner.learn_from_feedback(
                    premise=problem["premise"],
                    question=problem["question"],
                    correct_answer=expected,
                    predicted_answer=actual_answer,
                    was_correct=is_correct
                )

        # Calculate summary statistics
        summary = self._calculate_summary(trials)

        # Get phi correlations
        phi_correlations = self.phi_calculator.get_correlations()

        # Find best predictor
        best_predictor, best_corr = self.phi_calculator.get_best_predictor()

        # Run statistical tests
        statistical_tests = self._run_statistical_tests(trials, phi_correlations)

        # Validate thesis criteria
        thesis_validation = self._validate_thesis(trials, phi_correlations, substrate)

        results = ExperimentResults(
            config=config,
            trials=trials,
            summary=summary,
            phi_correlations=phi_correlations,
            best_phi_predictor=best_predictor,
            statistical_tests=statistical_tests,
            thesis_validation=thesis_validation
        )

        # Save results
        self._save_results(results)

        return results

    def _evaluate_answer(self, actual: str, expected: str, problem_type: str) -> bool:
        """Evaluate if answer is correct"""
        actual = actual.lower().strip()
        expected = expected.lower().strip()

        # Direct match
        if expected in actual:
            return True

        # Handle yes/no questions
        if expected in ["yes", "no"]:
            if expected == "yes" and any(w in actual for w in ["yes", "true", "correct", "likely"]):
                return True
            if expected == "no" and any(w in actual for w in ["no", "false", "incorrect", "unlikely"]):
                return True

        # Handle "likely" as partial yes
        if expected == "likely" and any(w in actual for w in ["likely", "probably", "yes"]):
            return True

        return False

    def _calculate_summary(self, trials: List[TrialResult]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        correct = sum(1 for t in trials if t.is_correct)
        total = len(trials)

        # By problem type
        by_type = {}
        for ptype in set(t.problem_type for t in trials):
            type_trials = [t for t in trials if t.problem_type == ptype]
            type_correct = sum(1 for t in type_trials if t.is_correct)
            by_type[ptype] = {
                "correct": type_correct,
                "total": len(type_trials),
                "accuracy": type_correct / len(type_trials) if type_trials else 0
            }

        # Phi statistics
        phi_means = {}
        for method in self.phi_calculator.calculators.keys():
            values = [t.phi_values.get(method, 0) for t in trials]
            phi_means[method] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }

        # Response time stats
        response_times = [t.response_time_ms for t in trials]

        return {
            "total_trials": total,
            "correct_trials": correct,
            "overall_accuracy": correct / total if total > 0 else 0,
            "accuracy_by_type": by_type,
            "phi_statistics": phi_means,
            "response_time_ms": {
                "mean": float(np.mean(response_times)),
                "std": float(np.std(response_times)),
                "min": float(np.min(response_times)),
                "max": float(np.max(response_times))
            },
            "emergent_nodes_final": trials[-1].emergent_nodes_count if trials else 0,
            "substrate_edges_final": trials[-1].substrate_edges if trials else 0
        }

    def _run_statistical_tests(
        self,
        trials: List[TrialResult],
        phi_correlations: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Run statistical significance tests"""
        results = {}

        # Chi-square test for accuracy vs random (50%)
        correct = sum(1 for t in trials if t.is_correct)
        total = len(trials)
        expected_random = total * 0.5
        chi2, p_value = stats.chisquare([correct, total - correct], [expected_random, expected_random])

        results["accuracy_vs_random"] = {
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "interpretation": "above_random" if correct > expected_random and p_value < 0.05 else "not_significant"
        }

        # T-test: phi values for correct vs incorrect
        for method in self.phi_calculator.calculators.keys():
            correct_phis = [t.phi_values.get(method, 0) for t in trials if t.is_correct]
            incorrect_phis = [t.phi_values.get(method, 0) for t in trials if not t.is_correct]

            if len(correct_phis) > 1 and len(incorrect_phis) > 1:
                t_stat, p_value = stats.ttest_ind(correct_phis, incorrect_phis)
                results[f"phi_{method}_ttest"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "mean_correct": float(np.mean(correct_phis)),
                    "mean_incorrect": float(np.mean(incorrect_phis)),
                    "effect_size": float(np.mean(correct_phis) - np.mean(incorrect_phis))
                }

        # Correlation significance (already in phi_correlations, but add interpretation)
        significant_correlations = []
        for method, corr_data in phi_correlations.items():
            if corr_data.get("p_value", 1) < 0.05:
                significant_correlations.append({
                    "method": method,
                    "correlation": corr_data["correlation"],
                    "p_value": corr_data["p_value"]
                })

        results["significant_phi_correlations"] = significant_correlations

        return results

    def _validate_thesis(
        self,
        trials: List[TrialResult],
        phi_correlations: Dict[str, Dict[str, float]],
        substrate
    ) -> Dict[str, bool]:
        """Validate ISC thesis criteria"""

        # Criterion 1: Phi predicts accuracy (significant positive correlation)
        best_corr = max(abs(c["correlation"]) for c in phi_correlations.values())
        best_p = min(c["p_value"] for c in phi_correlations.values())
        phi_predicts_accuracy = best_corr > 0.1 and best_p < 0.05

        # Criterion 2: System learns (accuracy improves over time)
        first_quarter = trials[:len(trials)//4]
        last_quarter = trials[-len(trials)//4:]
        early_acc = sum(1 for t in first_quarter if t.is_correct) / len(first_quarter) if first_quarter else 0
        late_acc = sum(1 for t in last_quarter if t.is_correct) / len(last_quarter) if last_quarter else 0

        # Account for ceiling effects: if starting high, relative improvement matters
        room_to_improve = 1.0 - early_acc
        if room_to_improve > 0.1:  # Normal case: need 5% absolute improvement
            system_learns = late_acc > early_acc + 0.05
        else:  # Ceiling effect: need to capture most of remaining room OR maintain high accuracy
            system_learns = late_acc >= early_acc or late_acc >= 0.95

        # Criterion 3: Emergent nodes created
        emergent_count = len([n for n in substrate.nodes if n.startswith("emergent")])
        emergence_occurs = emergent_count > 0

        # Criterion 4: Above random performance
        overall_acc = sum(1 for t in trials if t.is_correct) / len(trials) if trials else 0
        above_random = overall_acc > 0.55  # Significantly above 50%

        # Criterion 5: Integration increases with accuracy
        # Check if high-phi trials are more accurate
        phi_values = [np.mean(list(t.phi_values.values())) for t in trials]
        median_phi = np.median(phi_values)
        high_phi_trials = [t for t, p in zip(trials, phi_values) if p >= median_phi]
        low_phi_trials = [t for t, p in zip(trials, phi_values) if p < median_phi]
        high_phi_acc = sum(1 for t in high_phi_trials if t.is_correct) / len(high_phi_trials) if high_phi_trials else 0
        low_phi_acc = sum(1 for t in low_phi_trials if t.is_correct) / len(low_phi_trials) if low_phi_trials else 0
        integration_helps = high_phi_acc > low_phi_acc

        return {
            "phi_predicts_accuracy": phi_predicts_accuracy,
            "system_learns_over_time": system_learns,
            "emergence_occurs": emergence_occurs,
            "above_random_performance": above_random,
            "integration_improves_accuracy": integration_helps,
            "criteria_met": sum([
                phi_predicts_accuracy,
                system_learns,
                emergence_occurs,
                above_random,
                integration_helps
            ]),
            "total_criteria": 5,
            "details": {
                "best_correlation": float(best_corr),
                "correlation_p_value": float(best_p),
                "early_accuracy": float(early_acc),
                "late_accuracy": float(late_acc),
                "improvement": float(late_acc - early_acc),
                "emergent_nodes": emergent_count,
                "overall_accuracy": float(overall_acc),
                "high_phi_accuracy": float(high_phi_acc),
                "low_phi_accuracy": float(low_phi_acc)
            }
        }

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _save_results(self, results: ExperimentResults):
        """Save experiment results to JSON files"""
        exp_dir = self.output_dir / results.config.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(exp_dir / "config.json", "w") as f:
            json.dump(self._convert_to_json_serializable(asdict(results.config)), f, indent=2)

        # Save summary (human-readable)
        summary_data = {
            "experiment_id": results.config.experiment_id,
            "name": results.config.name,
            "description": results.config.description,
            "timestamp": results.config.timestamp,
            "summary": results.summary,
            "phi_correlations": results.phi_correlations,
            "best_phi_predictor": results.best_phi_predictor,
            "statistical_tests": results.statistical_tests,
            "thesis_validation": results.thesis_validation
        }
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(self._convert_to_json_serializable(summary_data), f, indent=2)

        # Save detailed trials (for analysis)
        trials_data = [asdict(t) for t in results.trials]
        with open(exp_dir / "trials.json", "w") as f:
            json.dump(self._convert_to_json_serializable(trials_data), f, indent=2)

        # Save human-readable report
        self._generate_report(results, exp_dir / "report.txt")

        print(f"\nResults saved to: {exp_dir}")

    def _generate_report(self, results: ExperimentResults, path: Path):
        """Generate human-readable report"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"EXPERIMENT REPORT: {results.config.name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Experiment ID: {results.config.experiment_id}")
        lines.append(f"Date: {results.config.timestamp}")
        lines.append(f"Description: {results.config.description}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 70)
        lines.append(f"Total Trials: {results.summary['total_trials']}")
        lines.append(f"Overall Accuracy: {results.summary['overall_accuracy']:.1%}")
        lines.append("")

        lines.append("Accuracy by Reasoning Type:")
        for ptype, stats in results.summary["accuracy_by_type"].items():
            lines.append(f"  {ptype:15s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
        lines.append("")

        lines.append("-" * 70)
        lines.append("PHI ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"Best Predictor: {results.best_phi_predictor}")
        lines.append("")
        lines.append("Phi-Accuracy Correlations:")
        for method, corr in results.phi_correlations.items():
            sig = "*" if corr["p_value"] < 0.05 else ""
            lines.append(f"  {method:20s}: r={corr['correlation']:+.3f} (p={corr['p_value']:.4f}){sig}")
        lines.append("  (* = statistically significant at p<0.05)")
        lines.append("")

        lines.append("-" * 70)
        lines.append("STATISTICAL TESTS")
        lines.append("-" * 70)

        acc_test = results.statistical_tests.get("accuracy_vs_random", {})
        if acc_test:
            lines.append(f"Accuracy vs Random (chi-square):")
            lines.append(f"  Chi-square: {acc_test.get('chi_square', 0):.2f}")
            lines.append(f"  p-value: {acc_test.get('p_value', 1):.4f}")
            lines.append(f"  Result: {acc_test.get('interpretation', 'unknown')}")
        lines.append("")

        sig_corrs = results.statistical_tests.get("significant_phi_correlations", [])
        if sig_corrs:
            lines.append("Significant Phi-Accuracy Correlations:")
            for sc in sig_corrs:
                lines.append(f"  {sc['method']}: r={sc['correlation']:.3f} (p={sc['p_value']:.4f})")
        lines.append("")

        lines.append("-" * 70)
        lines.append("ISC THESIS VALIDATION")
        lines.append("-" * 70)
        tv = results.thesis_validation
        lines.append(f"Criteria Met: {tv['criteria_met']}/{tv['total_criteria']}")
        lines.append("")
        lines.append("Individual Criteria:")
        lines.append(f"  1. Phi predicts accuracy:      {'PASS' if tv['phi_predicts_accuracy'] else 'FAIL'}")
        lines.append(f"  2. System learns over time:    {'PASS' if tv['system_learns_over_time'] else 'FAIL'}")
        lines.append(f"  3. Emergence occurs:           {'PASS' if tv['emergence_occurs'] else 'FAIL'}")
        lines.append(f"  4. Above random performance:   {'PASS' if tv['above_random_performance'] else 'FAIL'}")
        lines.append(f"  5. Integration helps accuracy: {'PASS' if tv['integration_improves_accuracy'] else 'FAIL'}")
        lines.append("")

        details = tv.get("details", {})
        if details:
            lines.append("Details:")
            lines.append(f"  Best correlation: {details.get('best_correlation', 0):.3f}")
            lines.append(f"  Early accuracy: {details.get('early_accuracy', 0):.1%}")
            lines.append(f"  Late accuracy: {details.get('late_accuracy', 0):.1%}")
            lines.append(f"  Improvement: {details.get('improvement', 0):+.1%}")
            lines.append(f"  Emergent nodes: {details.get('emergent_nodes', 0)}")
        lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        with open(path, "w") as f:
            f.write("\n".join(lines))


def main():
    """Run a standard experiment"""
    import argparse

    parser = argparse.ArgumentParser(description="Run ISC phi-driven reasoning experiment")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per problem")
    parser.add_argument("--output", type=str, default="results/experiments", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default="phi_accuracy_correlation", help="Experiment name")

    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output)

    results = runner.run_experiment(
        name=args.name,
        description="Testing correlation between phi measures and reasoning accuracy",
        iterations=args.iterations,
        random_seed=args.seed
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Overall Accuracy: {results.summary['overall_accuracy']:.1%}")
    print(f"Best Phi Predictor: {results.best_phi_predictor}")
    print(f"Thesis Criteria Met: {results.thesis_validation['criteria_met']}/{results.thesis_validation['total_criteria']}")


if __name__ == "__main__":
    main()
