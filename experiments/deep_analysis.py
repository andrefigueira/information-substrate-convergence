"""
Deep Analysis for Novel Findings

This module runs comprehensive experiments to discover novel patterns in:
1. Phi measure comparison - which best predicts accuracy for each reasoning type
2. Emergence dynamics - when and why emergent nodes form
3. Phase transitions - critical phi values where behavior changes
4. Substrate size effects - relationship between network size and reasoning
5. Learning curves - how accuracy evolves with experience

Scientific Method:
- Each experiment is run multiple times with different seeds
- Statistical tests validate significance of findings
- Results are logged in structured format for reproducibility
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from scipy import stats
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.isc.improved_emergent_reasoning import (
    ImprovedPhiDrivenSubstrate, ImprovedEmergentReasoner
)
from experiments.phi_calculations import CompositePhiCalculator


class DeepAnalysis:
    """Run deep analysis experiments for novel findings"""

    def __init__(self, output_dir: str = "results/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.findings: List[Dict[str, Any]] = []

    def run_all_analyses(self, n_seeds: int = 5):
        """Run all analysis experiments"""
        print("=" * 70)
        print("DEEP ANALYSIS FOR NOVEL FINDINGS")
        print("=" * 70)

        # Analysis 1: Phi measure comparison by problem type
        print("\n[1/5] Phi Measure Comparison by Problem Type...")
        phi_comparison = self._analyze_phi_by_problem_type(n_seeds)
        self.findings.append(phi_comparison)

        # Analysis 2: Emergence dynamics
        print("\n[2/5] Emergence Dynamics Analysis...")
        emergence = self._analyze_emergence_dynamics(n_seeds)
        self.findings.append(emergence)

        # Analysis 3: Phase transitions
        print("\n[3/5] Phase Transition Detection...")
        phase_trans = self._analyze_phase_transitions(n_seeds)
        self.findings.append(phase_trans)

        # Analysis 4: Substrate size effects
        print("\n[4/5] Substrate Size Effects...")
        size_effects = self._analyze_substrate_size(n_seeds)
        self.findings.append(size_effects)

        # Analysis 5: Learning curve analysis
        print("\n[5/5] Learning Curve Analysis...")
        learning = self._analyze_learning_curves(n_seeds)
        self.findings.append(learning)

        # Save all findings
        self._save_findings()

        # Generate summary report
        self._generate_findings_report()

        return self.findings

    def _analyze_phi_by_problem_type(self, n_seeds: int) -> Dict[str, Any]:
        """Find which phi measure best predicts accuracy for each problem type"""
        results_by_type = defaultdict(lambda: defaultdict(list))

        test_problems = self._get_test_problems()

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)
            phi_calc = CompositePhiCalculator()

            # Track phi-accuracy by problem type
            type_records = defaultdict(list)  # type -> [(phi_values, correct)]

            for _ in range(200):  # 200 problems
                problem = random.choice(test_problems)

                activations = substrate.get_activation_pattern()
                connectivity = substrate.get_connectivity_matrix()
                phi_values = phi_calc.calculate_all(activations, connectivity)

                result = reasoner.reason(problem["premise"], problem["question"])
                is_correct = self._check_correct(result["answer"], problem["expected"])

                type_records[problem["type"]].append((phi_values.copy(), is_correct))

                reasoner.learn_from_feedback(
                    problem["premise"], problem["question"],
                    problem["expected"], result["answer"], is_correct
                )

            # Calculate correlations for each type
            for ptype, records in type_records.items():
                if len(records) < 20:
                    continue

                accuracies = np.array([1.0 if r[1] else 0.0 for r in records])

                for phi_method in ['simple', 'mutual_info', 'effective_info', 'geometric', 'stochastic']:
                    phi_vals = np.array([r[0].get(phi_method, 0) for r in records])

                    if np.std(phi_vals) > 0 and np.std(accuracies) > 0:
                        corr, p_val = stats.pearsonr(phi_vals, accuracies)
                        results_by_type[ptype][phi_method].append({
                            'correlation': corr,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        })

        # Aggregate results
        aggregated = {}
        best_predictors = {}

        for ptype, method_results in results_by_type.items():
            aggregated[ptype] = {}
            best_corr = 0
            best_method = None

            for method, corr_list in method_results.items():
                if not corr_list:
                    continue

                mean_corr = np.mean([c['correlation'] for c in corr_list])
                std_corr = np.std([c['correlation'] for c in corr_list])
                sig_rate = np.mean([c['significant'] for c in corr_list])

                aggregated[ptype][method] = {
                    'mean_correlation': float(mean_corr),
                    'std_correlation': float(std_corr),
                    'significance_rate': float(sig_rate),
                    'n_experiments': len(corr_list)
                }

                if abs(mean_corr) > abs(best_corr):
                    best_corr = mean_corr
                    best_method = method

            best_predictors[ptype] = {
                'method': best_method,
                'correlation': float(best_corr)
            }

        finding = {
            'name': 'Phi Measure Comparison by Problem Type',
            'hypothesis': 'Different phi measures predict accuracy differently across reasoning types',
            'results': aggregated,
            'best_predictors': best_predictors,
            'novel_finding': self._interpret_phi_comparison(best_predictors)
        }

        return finding

    def _interpret_phi_comparison(self, best_predictors: Dict) -> str:
        """Interpret the phi comparison results"""
        methods = [bp['method'] for bp in best_predictors.values() if bp['method']]
        unique_methods = set(methods)

        if len(unique_methods) == 1:
            return f"FINDING: Single phi measure ({list(unique_methods)[0]}) best predicts all reasoning types"
        elif len(unique_methods) > 1:
            return f"NOVEL FINDING: Different phi measures optimal for different reasoning - {best_predictors}"
        return "Insufficient data for conclusive finding"

    def _analyze_emergence_dynamics(self, n_seeds: int) -> Dict[str, Any]:
        """Analyze when and why emergent nodes form"""
        emergence_events = []

        test_problems = self._get_test_problems()

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

            prev_emergent_count = 0

            for iteration in range(300):
                problem = random.choice(test_problems)

                activations = substrate.get_activation_pattern()
                phi_before = float(np.mean(activations))

                result = reasoner.reason(problem["premise"], problem["question"])
                is_correct = self._check_correct(result["answer"], problem["expected"])

                reasoner.learn_from_feedback(
                    problem["premise"], problem["question"],
                    problem["expected"], result["answer"], is_correct
                )

                curr_emergent_count = len([n for n in substrate.nodes if n.startswith('emergent')])

                if curr_emergent_count > prev_emergent_count:
                    emergence_events.append({
                        'iteration': iteration,
                        'phi_at_emergence': phi_before,
                        'was_correct': is_correct,
                        'problem_type': problem["type"],
                        'total_emergent': curr_emergent_count,
                        'seed': seed
                    })
                    prev_emergent_count = curr_emergent_count

        # Analyze emergence patterns
        if not emergence_events:
            return {'name': 'Emergence Dynamics', 'finding': 'No emergence observed'}

        iterations = [e['iteration'] for e in emergence_events]
        correct_at_emerge = [e['was_correct'] for e in emergence_events]
        types_at_emerge = [e['problem_type'] for e in emergence_events]

        finding = {
            'name': 'Emergence Dynamics Analysis',
            'hypothesis': 'Emergent nodes form after successful reasoning on repeated patterns',
            'results': {
                'total_emergence_events': len(emergence_events),
                'mean_iteration_at_emergence': float(np.mean(iterations)),
                'std_iteration': float(np.std(iterations)),
                'correct_rate_at_emergence': float(np.mean(correct_at_emerge)),
                'emergence_by_type': {t: types_at_emerge.count(t) for t in set(types_at_emerge)},
                'first_emergence_iteration': int(min(iterations)),
                'emergence_rate': len(emergence_events) / (n_seeds * 300)
            },
            'novel_finding': f"FINDING: Emergence occurs at {np.mean(correct_at_emerge)*100:.0f}% success rate, "
                           f"primarily for {max(set(types_at_emerge), key=types_at_emerge.count)} problems"
        }

        return finding

    def _analyze_phase_transitions(self, n_seeds: int) -> Dict[str, Any]:
        """Look for critical phi values where accuracy behavior changes"""
        phi_accuracy_data = []

        test_problems = self._get_test_problems()

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

            for _ in range(300):
                problem = random.choice(test_problems)

                activations = substrate.get_activation_pattern()
                phi = float(substrate.calculate_phi())

                result = reasoner.reason(problem["premise"], problem["question"])
                is_correct = self._check_correct(result["answer"], problem["expected"])

                phi_accuracy_data.append({'phi': phi, 'correct': is_correct})

                reasoner.learn_from_feedback(
                    problem["premise"], problem["question"],
                    problem["expected"], result["answer"], is_correct
                )

        # Bin phi values and calculate accuracy per bin
        phis = np.array([d['phi'] for d in phi_accuracy_data])
        corrects = np.array([d['correct'] for d in phi_accuracy_data])

        # Find optimal number of bins
        bins = np.linspace(min(phis), max(phis), 11)
        bin_accuracies = []
        bin_centers = []

        for i in range(len(bins) - 1):
            mask = (phis >= bins[i]) & (phis < bins[i + 1])
            if np.sum(mask) >= 10:
                bin_acc = np.mean(corrects[mask])
                bin_accuracies.append(bin_acc)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)

        # Look for sharp transitions
        transitions = []
        if len(bin_accuracies) >= 3:
            for i in range(1, len(bin_accuracies) - 1):
                diff_before = bin_accuracies[i] - bin_accuracies[i - 1]
                diff_after = bin_accuracies[i + 1] - bin_accuracies[i]

                if abs(diff_before) > 0.1 or abs(diff_after) > 0.1:
                    transitions.append({
                        'phi_value': float(bin_centers[i]),
                        'accuracy_jump': float(max(abs(diff_before), abs(diff_after))),
                        'direction': 'up' if diff_after > 0 else 'down'
                    })

        finding = {
            'name': 'Phase Transition Analysis',
            'hypothesis': 'There exist critical phi values where accuracy behavior changes sharply',
            'results': {
                'bin_centers': [float(b) for b in bin_centers],
                'bin_accuracies': [float(a) for a in bin_accuracies],
                'transitions_found': len(transitions),
                'transitions': transitions,
                'phi_range': [float(min(phis)), float(max(phis))],
                'overall_correlation': float(np.corrcoef(phis, corrects)[0, 1]) if len(phis) > 10 else 0
            },
            'novel_finding': f"FINDING: {'Phase transition detected' if transitions else 'Gradual relationship'} - "
                           f"{len(transitions)} sharp transitions in phi-accuracy curve"
        }

        return finding

    def _analyze_substrate_size(self, n_seeds: int) -> Dict[str, Any]:
        """Analyze how substrate size affects reasoning capability"""
        sizes = [25, 50, 100, 200]
        size_results = {}

        test_problems = self._get_test_problems()

        for size in sizes:
            accuracies = []
            emergent_counts = []
            phi_values = []

            for seed in range(n_seeds):
                random.seed(seed)
                np.random.seed(seed)

                substrate = ImprovedPhiDrivenSubstrate(initial_nodes=size)
                reasoner = ImprovedEmergentReasoner(substrate)

                correct = 0
                total = 0

                for _ in range(150):
                    problem = random.choice(test_problems)

                    result = reasoner.reason(problem["premise"], problem["question"])
                    is_correct = self._check_correct(result["answer"], problem["expected"])

                    if is_correct:
                        correct += 1
                    total += 1

                    reasoner.learn_from_feedback(
                        problem["premise"], problem["question"],
                        problem["expected"], result["answer"], is_correct
                    )

                accuracies.append(correct / total)
                emergent_counts.append(len([n for n in substrate.nodes if n.startswith('emergent')]))
                phi_values.append(float(substrate.global_phi))

            size_results[size] = {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'mean_emergent': float(np.mean(emergent_counts)),
                'mean_phi': float(np.mean(phi_values)),
                'n_seeds': n_seeds
            }

        # Check for significant size effect
        size_list = list(sizes)
        acc_list = [size_results[s]['mean_accuracy'] for s in size_list]

        corr, p_val = stats.pearsonr(size_list, acc_list)

        finding = {
            'name': 'Substrate Size Effects',
            'hypothesis': 'Larger substrates enable better reasoning through more integration',
            'results': size_results,
            'size_accuracy_correlation': float(corr),
            'size_effect_p_value': float(p_val),
            'novel_finding': f"FINDING: Size-accuracy correlation r={corr:.3f} (p={p_val:.4f}) - "
                           f"{'significant' if p_val < 0.05 else 'not significant'} size effect"
        }

        return finding

    def _analyze_learning_curves(self, n_seeds: int) -> Dict[str, Any]:
        """Analyze learning curve shapes across problem types"""
        test_problems = self._get_test_problems()

        type_learning_curves = defaultdict(lambda: defaultdict(list))

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

            type_history = defaultdict(lambda: {'correct': 0, 'total': 0})
            checkpoints = [10, 25, 50, 100, 150, 200]
            checkpoint_idx = 0

            for iteration in range(max(checkpoints) + 1):
                problem = random.choice(test_problems)

                result = reasoner.reason(problem["premise"], problem["question"])
                is_correct = self._check_correct(result["answer"], problem["expected"])

                type_history[problem["type"]]['total'] += 1
                if is_correct:
                    type_history[problem["type"]]['correct'] += 1

                reasoner.learn_from_feedback(
                    problem["premise"], problem["question"],
                    problem["expected"], result["answer"], is_correct
                )

                if checkpoint_idx < len(checkpoints) and iteration == checkpoints[checkpoint_idx]:
                    for ptype, data in type_history.items():
                        if data['total'] > 0:
                            acc = data['correct'] / data['total']
                            type_learning_curves[ptype][iteration].append(acc)
                    checkpoint_idx += 1

        # Aggregate learning curves
        aggregated_curves = {}
        for ptype, iterations in type_learning_curves.items():
            aggregated_curves[ptype] = {
                iter_n: {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs))
                }
                for iter_n, accs in iterations.items()
            }

        # Calculate learning rate (slope of accuracy curve)
        learning_rates = {}
        for ptype, curve in aggregated_curves.items():
            iters = sorted(curve.keys())
            if len(iters) >= 3:
                x = np.array(iters)
                y = np.array([curve[i]['mean'] for i in iters])
                slope, intercept = np.polyfit(x, y, 1)
                learning_rates[ptype] = float(slope * 100)  # % per iteration

        finding = {
            'name': 'Learning Curve Analysis',
            'hypothesis': 'Different reasoning types have different learning curve shapes',
            'results': {
                'curves': aggregated_curves,
                'learning_rates': learning_rates
            },
            'novel_finding': f"FINDING: Learning rates vary by type - "
                           f"fastest: {max(learning_rates.items(), key=lambda x: x[1]) if learning_rates else 'N/A'}, "
                           f"slowest: {min(learning_rates.items(), key=lambda x: x[1]) if learning_rates else 'N/A'}"
        }

        return finding

    def _get_test_problems(self) -> List[Dict[str, Any]]:
        """Get test problems matching experiment runner"""
        return [
            {"type": "deductive", "premise": "All mammals are warm-blooded. Whales are mammals.",
             "question": "Are whales warm-blooded?", "expected": "yes"},
            {"type": "deductive", "premise": "If it rains, the ground gets wet. It is raining.",
             "question": "Is the ground wet?", "expected": "yes"},
            {"type": "deductive", "premise": "No reptiles are warm-blooded. Snakes are reptiles.",
             "question": "Are snakes warm-blooded?", "expected": "no"},
            {"type": "inductive", "premise": "The sun rose today. The sun rose yesterday. The sun rose the day before.",
             "question": "Will the sun rise tomorrow?", "expected": "likely"},
            {"type": "inductive", "premise": "Metal A expands when heated. Metal B expands when heated.",
             "question": "Do metals expand when heated?", "expected": "yes"},
            {"type": "abductive", "premise": "The grass is wet. It is morning. There are no sprinklers.",
             "question": "What is the most likely explanation?", "expected": "dew"},
            {"type": "abductive", "premise": "The patient has fever, cough, and fatigue.",
             "question": "What is the most likely diagnosis?", "expected": "flu"},
            {"type": "analogical", "premise": "Puppy is to dog as kitten is to what?",
             "question": "Complete the analogy.", "expected": "cat"},
            {"type": "analogical", "premise": "Hot is to cold as light is to what?",
             "question": "Complete the analogy.", "expected": "dark"},
            {"type": "causal", "premise": "Plants given fertilizer grew taller. Plants without fertilizer stayed short.",
             "question": "Does fertilizer cause growth?", "expected": "yes"},
            {"type": "causal", "premise": "Ice cream sales increase. Drowning incidents increase.",
             "question": "Does ice cream cause drowning?", "expected": "no"},
        ]

    def _check_correct(self, predicted: str, expected: str) -> bool:
        """Check if answer is correct"""
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()

        if expected in predicted:
            return True
        if expected in ['yes', 'no', 'likely']:
            if expected == 'yes' and predicted in ['yes', 'likely_yes', 'true']:
                return True
            if expected == 'no' and predicted in ['no', 'likely_no', 'false']:
                return True
            if expected == 'likely' and predicted in ['likely', 'yes', 'probably']:
                return True
        return False

    def _save_findings(self):
        """Save all findings to JSON"""
        output_file = self.output_dir / f"findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(output_file, 'w') as f:
            json.dump(convert(self.findings), f, indent=2)

        print(f"\nFindings saved to: {output_file}")

    def _generate_findings_report(self):
        """Generate human-readable findings report"""
        lines = []
        lines.append("=" * 70)
        lines.append("NOVEL FINDINGS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        for i, finding in enumerate(self.findings, 1):
            lines.append(f"\n{'='*70}")
            lines.append(f"FINDING {i}: {finding.get('name', 'Unknown')}")
            lines.append("=" * 70)
            lines.append(f"\nHypothesis: {finding.get('hypothesis', 'N/A')}")
            lines.append(f"\n{finding.get('novel_finding', 'No specific finding')}")

            if 'results' in finding and isinstance(finding['results'], dict):
                lines.append("\nKey Results:")
                for key, value in list(finding['results'].items())[:5]:
                    if isinstance(value, dict):
                        lines.append(f"  {key}:")
                        for k, v in list(value.items())[:3]:
                            lines.append(f"    {k}: {v}")
                    else:
                        lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 70)
        lines.append("END OF FINDINGS REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)
        print(report_text)

        report_file = self.output_dir / f"findings_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)


if __name__ == "__main__":
    analysis = DeepAnalysis()
    findings = analysis.run_all_analyses(n_seeds=5)
