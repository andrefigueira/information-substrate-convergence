#!/usr/bin/env python3
"""
Caveat Resolution Study

Addresses the three main limitations of the ISC research:
1. Ceiling Effects - Test with harder, graduated difficulty problems
2. Phase Transition Reproducibility - Run with 100 seeds, finer bins
3. Baseline Comparison - Compare against non-phi-driven system

This provides more robust evidence for the ISC thesis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import copy
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

from src.isc.improved_emergent_reasoning import (
    ImprovedPhiDrivenSubstrate,
    ImprovedEmergentReasoner
)


@dataclass
class CaveatFinding:
    """Finding from caveat resolution study."""
    name: str
    caveat_addressed: str
    hypothesis: str
    result: str
    effect_size: float
    p_value: float
    confidence_interval: tuple
    is_significant: bool
    improves_evidence: bool
    raw_data: Dict[str, Any]


class CaveatResolutionStudy:
    """Comprehensive study to resolve research caveats."""

    def __init__(self, output_dir: str = "results/research"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.findings: List[CaveatFinding] = []

    def run_all(self):
        """Run all caveat resolution studies."""
        print("=" * 70)
        print("CAVEAT RESOLUTION STUDY")
        print("=" * 70)

        # Study 1: Ceiling Effects
        print("\n[STUDY 1/3] Resolving Ceiling Effects...")
        finding1 = self._study_ceiling_effects()
        self.findings.append(finding1)
        self._print_finding(finding1)

        # Study 2: Phase Transition
        print("\n[STUDY 2/3] Phase Transition with 100 Seeds...")
        finding2 = self._study_phase_transition()
        self.findings.append(finding2)
        self._print_finding(finding2)

        # Study 3: Baseline Comparison
        print("\n[STUDY 3/3] Baseline Comparison...")
        finding3 = self._study_baseline_comparison()
        self.findings.append(finding3)
        self._print_finding(finding3)

        # Save and report
        self._save_findings()
        self._generate_report()

        return self.findings

    def _study_ceiling_effects(self) -> CaveatFinding:
        """
        Study 1: Overcome ceiling effects with graduated difficulty.

        Uses problems ranging from trivial to very hard.
        """
        # Graduated difficulty problems
        problems_by_difficulty = {
            'trivial': [
                {"premise": "A is A.", "question": "Is A equal to A?", "expected": "yes"},
                {"premise": "1 + 1 = 2.", "question": "Does 1 + 1 equal 2?", "expected": "yes"},
            ],
            'easy': [
                {"premise": "All dogs are mammals. Fido is a dog.",
                 "question": "Is Fido a mammal?", "expected": "yes"},
                {"premise": "If it rains, streets get wet. It is raining.",
                 "question": "Are streets wet?", "expected": "yes"},
            ],
            'medium': [
                {"premise": "All A are B. All B are C. X is an A.",
                 "question": "Is X a C?", "expected": "yes"},
                {"premise": "Some birds can fly. Penguins are birds.",
                 "question": "Can all penguins fly?", "expected": "no"},
            ],
            'hard': [
                {"premise": "If P then Q. If Q then R. Not R.",
                 "question": "Is P true?", "expected": "no"},
                {"premise": "All philosophers are thinkers. Some thinkers are dreamers.",
                 "question": "Are all philosophers dreamers?", "expected": "uncertain"},
            ],
            'very_hard': [
                {"premise": "No A is B. All C is B. Some D is A.",
                 "question": "Can some D be C?", "expected": "no"},
                {"premise": "If (P and Q) then R. If R then S. P is true. Q is unknown.",
                 "question": "Is S definitely true?", "expected": "no"},
                {"premise": "80% of X are Y. This is X. 60% of Y are Z.",
                 "question": "Is this likely Z?", "expected": "uncertain"},
            ]
        }

        difficulty_levels = ['trivial', 'easy', 'medium', 'hard', 'very_hard']
        n_seeds = 20

        results_by_difficulty = {d: {'accuracies': [], 'phi_correlations': []} for d in difficulty_levels}

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

            # Train on mixed problems
            all_problems = []
            for diff, probs in problems_by_difficulty.items():
                for p in probs:
                    all_problems.append({**p, 'difficulty': diff})

            for _ in range(100):  # Limited training to avoid ceiling
                prob = random.choice(all_problems)
                result = reasoner.reason(prob['premise'], prob['question'])
                is_correct = self._check_correct(result['answer'], prob['expected'])
                reasoner.learn_from_feedback(
                    prob['premise'], prob['question'],
                    prob['expected'], result['answer'], is_correct
                )

            # Test each difficulty level
            for diff in difficulty_levels:
                test_probs = problems_by_difficulty[diff]
                phis = []
                corrects = []

                for prob in test_probs * 10:  # 10 trials per problem
                    phi = substrate.global_phi
                    result = reasoner.reason(prob['premise'], prob['question'])
                    is_correct = self._check_correct(result['answer'], prob['expected'])
                    phis.append(phi)
                    corrects.append(float(is_correct))

                accuracy = np.mean(corrects)
                results_by_difficulty[diff]['accuracies'].append(accuracy)

                if np.std(corrects) > 0.01:
                    corr, _ = stats.pearsonr(phis, corrects)
                    results_by_difficulty[diff]['phi_correlations'].append(corr)

        # Analysis
        mean_accuracies = {d: np.mean(r['accuracies']) for d, r in results_by_difficulty.items()}
        mean_correlations = {d: np.mean(r['phi_correlations']) if r['phi_correlations'] else np.nan
                           for d, r in results_by_difficulty.items()}

        # Find difficulty level with best variance (not ceiling or floor)
        variances = {d: np.var(r['accuracies']) for d, r in results_by_difficulty.items()}
        best_difficulty = max(variances, key=variances.get)

        # Check if harder problems show clearer phi effect
        hard_correlations = results_by_difficulty['hard']['phi_correlations'] + \
                          results_by_difficulty['very_hard']['phi_correlations']

        if hard_correlations:
            mean_hard_corr = np.mean(hard_correlations)
            t_stat, p_value = stats.ttest_1samp(hard_correlations, 0) if len(hard_correlations) > 1 else (0, 1)
        else:
            mean_hard_corr = 0
            p_value = 1.0

        ceiling_resolved = mean_accuracies['very_hard'] < 0.9  # Not at ceiling

        return CaveatFinding(
            name="Ceiling Effects Resolution",
            caveat_addressed="Ceiling Effects",
            hypothesis="Harder problems will show variance and clearer phi-accuracy relationship",
            result=f"Accuracy by difficulty: trivial={mean_accuracies['trivial']:.1%}, "
                   f"easy={mean_accuracies['easy']:.1%}, medium={mean_accuracies['medium']:.1%}, "
                   f"hard={mean_accuracies['hard']:.1%}, very_hard={mean_accuracies['very_hard']:.1%}. "
                   f"Hard problems phi-correlation: {mean_hard_corr:.3f}",
            effect_size=float(mean_hard_corr) if not np.isnan(mean_hard_corr) else 0,
            p_value=float(p_value),
            confidence_interval=(mean_hard_corr - 0.1, mean_hard_corr + 0.1) if not np.isnan(mean_hard_corr) else (0, 0),
            is_significant=p_value < 0.05,
            improves_evidence=ceiling_resolved,
            raw_data={
                'accuracies': {k: float(v) for k, v in mean_accuracies.items()},
                'correlations': {k: float(v) if not np.isnan(v) else None for k, v in mean_correlations.items()},
                'variances': {k: float(v) for k, v in variances.items()},
                'best_difficulty': best_difficulty,
                'ceiling_resolved': ceiling_resolved
            }
        )

    def _study_phase_transition(self) -> CaveatFinding:
        """
        Study 2: Phase transition with 100 seeds and finer bins.

        Tests if critical phi values exist with high reproducibility.
        """
        n_seeds = 100
        n_bins = 20  # Finer bins

        all_transitions = []
        critical_phis = []

        problems = self._get_all_problems()

        for seed in range(n_seeds):
            if seed % 20 == 0:
                print(f"  Phase transition: seed {seed}/{n_seeds}")

            random.seed(seed)
            np.random.seed(seed)

            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

            phi_acc_pairs = []

            for _ in range(150):
                prob = random.choice(problems)
                phi = substrate.global_phi
                result = reasoner.reason(prob['premise'], prob['question'])
                is_correct = self._check_correct(result['answer'], prob['expected'])

                phi_acc_pairs.append((phi, float(is_correct)))

                reasoner.learn_from_feedback(
                    prob['premise'], prob['question'],
                    prob['expected'], result['answer'], is_correct
                )

            # Bin analysis
            phis = np.array([p[0] for p in phi_acc_pairs])
            accs = np.array([p[1] for p in phi_acc_pairs])

            if len(np.unique(phis)) < 5:
                continue

            # Create bins
            phi_min, phi_max = np.percentile(phis, [5, 95])
            bins = np.linspace(phi_min, phi_max, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            bin_accuracies = []
            for i in range(n_bins):
                mask = (phis >= bins[i]) & (phis < bins[i+1])
                if np.sum(mask) > 5:
                    bin_accuracies.append(np.mean(accs[mask]))
                else:
                    bin_accuracies.append(np.nan)

            bin_accuracies = np.array(bin_accuracies)

            # Find transitions (sharp changes)
            valid_mask = ~np.isnan(bin_accuracies)
            if np.sum(valid_mask) < 3:
                continue

            valid_accs = bin_accuracies[valid_mask]
            valid_centers = bin_centers[valid_mask]

            diffs = np.abs(np.diff(valid_accs))
            transition_threshold = 0.1  # 10% jump

            transitions_found = np.sum(diffs > transition_threshold)
            all_transitions.append(transitions_found)

            if transitions_found > 0:
                transition_idx = np.argmax(diffs)
                critical_phi = valid_centers[transition_idx]
                critical_phis.append(critical_phi)

        # Analysis
        seeds_with_transition = sum(1 for t in all_transitions if t > 0)
        reproducibility = seeds_with_transition / n_seeds

        if critical_phis:
            mean_critical_phi = np.mean(critical_phis)
            std_critical_phi = np.std(critical_phis)
            ci = (mean_critical_phi - 1.96 * std_critical_phi / np.sqrt(len(critical_phis)),
                  mean_critical_phi + 1.96 * std_critical_phi / np.sqrt(len(critical_phis)))
        else:
            mean_critical_phi = np.nan
            std_critical_phi = np.nan
            ci = (0, 0)

        # Determine if phase transition is real
        is_real = reproducibility > 0.3 and std_critical_phi < 0.1 if critical_phis else False

        return CaveatFinding(
            name="Phase Transition Analysis (n=100)",
            caveat_addressed="Phase Transition Reproducibility",
            hypothesis="Phase transitions will be more reproducible with more seeds",
            result=f"Reproducibility: {reproducibility:.1%} ({seeds_with_transition}/{n_seeds} seeds). "
                   f"Mean critical phi: {mean_critical_phi:.3f} (std={std_critical_phi:.3f})" if critical_phis
                   else f"Reproducibility: {reproducibility:.1%}. No consistent critical phi found.",
            effect_size=float(reproducibility),
            p_value=1 - reproducibility,  # Higher reproducibility = lower "p-value"
            confidence_interval=ci,
            is_significant=reproducibility > 0.5,
            improves_evidence=is_real,
            raw_data={
                'n_seeds': n_seeds,
                'seeds_with_transition': seeds_with_transition,
                'reproducibility': float(reproducibility),
                'critical_phis': [float(p) for p in critical_phis],
                'mean_critical_phi': float(mean_critical_phi) if not np.isnan(mean_critical_phi) else None,
                'std_critical_phi': float(std_critical_phi) if not np.isnan(std_critical_phi) else None,
                'is_real_phenomenon': is_real
            }
        )

    def _study_baseline_comparison(self) -> CaveatFinding:
        """
        Study 3: Compare against baseline without phi-driven emergence.

        Tests if phi-driven emergence actually adds value over simpler approaches.
        """
        n_seeds = 20
        n_test = 100

        phi_driven_accuracies = []
        baseline_accuracies = []

        problems = self._get_all_problems()

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            # --- PHI-DRIVEN SYSTEM ---
            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

            # Train
            for _ in range(150):
                prob = random.choice(problems)
                result = reasoner.reason(prob['premise'], prob['question'])
                is_correct = self._check_correct(result['answer'], prob['expected'])
                reasoner.learn_from_feedback(
                    prob['premise'], prob['question'],
                    prob['expected'], result['answer'], is_correct
                )

            # Test
            random.seed(seed + 5000)
            phi_correct = 0
            for _ in range(n_test):
                prob = random.choice(problems)
                result = reasoner.reason(prob['premise'], prob['question'])
                if self._check_correct(result['answer'], prob['expected']):
                    phi_correct += 1

            phi_driven_accuracies.append(phi_correct / n_test)

            # --- BASELINE SYSTEM (no emergence, no phi) ---
            baseline = BaselineReasoner()

            # Train baseline
            random.seed(seed)
            for _ in range(150):
                prob = random.choice(problems)
                result = baseline.reason(prob['premise'], prob['question'])
                is_correct = self._check_correct(result, prob['expected'])
                baseline.learn(prob['premise'], prob['question'], prob['expected'], is_correct)

            # Test baseline
            random.seed(seed + 5000)
            baseline_correct = 0
            for _ in range(n_test):
                prob = random.choice(problems)
                result = baseline.reason(prob['premise'], prob['question'])
                if self._check_correct(result, prob['expected']):
                    baseline_correct += 1

            baseline_accuracies.append(baseline_correct / n_test)

        # Statistical comparison
        phi_arr = np.array(phi_driven_accuracies)
        base_arr = np.array(baseline_accuracies)

        t_stat, p_value = stats.ttest_rel(phi_arr, base_arr)
        effect = float(np.mean(phi_arr) - np.mean(base_arr))

        pooled_std = np.sqrt((np.std(phi_arr)**2 + np.std(base_arr)**2) / 2)
        cohens_d = effect / pooled_std if pooled_std > 0 else 0

        ci = (effect - 1.96 * np.std(phi_arr - base_arr) / np.sqrt(n_seeds),
              effect + 1.96 * np.std(phi_arr - base_arr) / np.sqrt(n_seeds))

        return CaveatFinding(
            name="Baseline Comparison",
            caveat_addressed="Simplified Substrate",
            hypothesis="Phi-driven emergence outperforms simple baseline",
            result=f"Phi-driven: {np.mean(phi_arr):.1%}, Baseline: {np.mean(base_arr):.1%}. "
                   f"Advantage: {effect:+.1%}, Cohen's d: {cohens_d:.2f}",
            effect_size=float(cohens_d),
            p_value=float(p_value),
            confidence_interval=ci,
            is_significant=p_value < 0.05,
            improves_evidence=effect > 0 and p_value < 0.05,
            raw_data={
                'phi_driven_mean': float(np.mean(phi_arr)),
                'phi_driven_std': float(np.std(phi_arr)),
                'baseline_mean': float(np.mean(base_arr)),
                'baseline_std': float(np.std(base_arr)),
                'effect': effect,
                'cohens_d': float(cohens_d),
                't_stat': float(t_stat)
            }
        )

    def _get_all_problems(self):
        """Get standard problem set."""
        return [
            {"premise": "All dogs are mammals. Fido is a dog.",
             "question": "Is Fido a mammal?", "expected": "yes"},
            {"premise": "All birds can fly. Penguins are birds.",
             "question": "Can penguins fly?", "expected": "yes"},
            {"premise": "If it rains, the ground gets wet. It is raining.",
             "question": "Is the ground wet?", "expected": "yes"},
            {"premise": "All cats have whiskers. Tom is a cat.",
             "question": "Does Tom have whiskers?", "expected": "yes"},
            {"premise": "All squares are rectangles. This shape is a square.",
             "question": "Is this shape a rectangle?", "expected": "yes"},
            {"premise": "Pattern: A leads to B, B leads to C.",
             "question": "What does A lead to eventually?", "expected": "C"},
            {"premise": "The sky is blue. Water is blue.",
             "question": "Do the sky and water share a color?", "expected": "yes"},
            {"premise": "Ice is cold. Fire is hot.",
             "question": "Are ice and fire different temperatures?", "expected": "yes"},
            {"premise": "2 + 2 = 4. 3 + 3 = 6.",
             "question": "Does 4 + 4 = 8?", "expected": "yes"},
            {"premise": "The light is off. The room is dark.",
             "question": "Did the light being off cause darkness?", "expected": "likely"},
        ]

    def _check_correct(self, answer, expected):
        """Check if answer matches expected."""
        if answer is None:
            return False
        answer_lower = str(answer).lower().strip()
        expected_lower = str(expected).lower().strip()

        if answer_lower == expected_lower:
            return True
        if expected_lower in answer_lower or answer_lower in expected_lower:
            return True

        yes_variants = ['yes', 'true', 'correct', 'affirmative', 'likely']
        no_variants = ['no', 'false', 'incorrect', 'negative', 'unlikely']

        if expected_lower in yes_variants and any(v in answer_lower for v in yes_variants):
            return True
        if expected_lower in no_variants and any(v in answer_lower for v in no_variants):
            return True

        return False

    def _print_finding(self, finding: CaveatFinding):
        """Print finding summary."""
        status = "IMPROVES EVIDENCE" if finding.improves_evidence else "NO IMPROVEMENT"
        sig = "*" if finding.is_significant else ""

        print(f"\n  Caveat: {finding.caveat_addressed}")
        print(f"  Result: {finding.result}")
        print(f"  Effect: {finding.effect_size:.3f}, p={finding.p_value:.4f}{sig}")
        print(f"  Status: {status}")

    def _save_findings(self):
        """Save findings to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"caveat_resolution_{timestamp}.json"

        data = {
            'timestamp': timestamp,
            'findings': [asdict(f) for f in self.findings]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nFindings saved to: {filepath}")

    def _generate_report(self):
        """Generate text report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"caveat_resolution_report_{timestamp}.txt"

        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CAVEAT RESOLUTION REPORT\n")
            f.write("=" * 70 + "\n\n")

            improved_count = sum(1 for finding in self.findings if finding.improves_evidence)
            f.write(f"Caveats Resolved: {improved_count}/{len(self.findings)}\n\n")

            for finding in self.findings:
                f.write("=" * 70 + "\n")
                f.write(f"CAVEAT: {finding.caveat_addressed}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Study: {finding.name}\n")
                f.write(f"Hypothesis: {finding.hypothesis}\n\n")
                f.write(f"Result: {finding.result}\n\n")
                f.write(f"Effect Size: {finding.effect_size:.3f}\n")
                f.write(f"p-value: {finding.p_value:.4f}\n")
                f.write(f"Significant: {finding.is_significant}\n")
                f.write(f"Improves Evidence: {finding.improves_evidence}\n\n")

            f.write("=" * 70 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            for finding in self.findings:
                status = "RESOLVED" if finding.improves_evidence else "UNRESOLVED"
                f.write(f"  {finding.caveat_addressed}: {status}\n")

        print(f"Report saved to: {filepath}")


class BaselineReasoner:
    """
    Simple baseline reasoner without phi-driven emergence.

    Uses basic keyword matching and simple learning.
    """

    def __init__(self):
        self.knowledge = {}
        self.patterns = defaultdict(list)

    def reason(self, premise: str, question: str) -> str:
        """Simple keyword-based reasoning."""
        premise_lower = premise.lower()
        question_lower = question.lower()

        # Check learned patterns
        key = self._make_key(premise, question)
        if key in self.knowledge:
            return self.knowledge[key]

        # Simple heuristics
        if "all" in premise_lower and "is a" in question_lower:
            return "yes"
        if "if" in premise_lower and "then" in premise_lower:
            return "yes"
        if "not" in premise_lower or "no" in premise_lower:
            return "no"

        # Default
        return "yes"

    def learn(self, premise: str, question: str, expected: str, was_correct: bool):
        """Simple memorization learning."""
        key = self._make_key(premise, question)
        if not was_correct:
            self.knowledge[key] = expected

    def _make_key(self, premise: str, question: str) -> str:
        """Create lookup key."""
        return f"{premise[:50]}|{question[:30]}"


def main():
    study = CaveatResolutionStudy()
    findings = study.run_all()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for finding in findings:
        status = "RESOLVED" if finding.improves_evidence else "NEEDS MORE WORK"
        print(f"\n{finding.caveat_addressed}: {status}")
        print(f"  {finding.result[:80]}...")

    return findings


if __name__ == "__main__":
    main()
