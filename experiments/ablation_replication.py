#!/usr/bin/env python3
"""
Ablation Replication Study

Goal: Understand why we got contradictory results:
- Original study: +22% effect (100% vs 78%)
- Graded study: -4% effect (73% vs 77%)

This script runs both methodologies side-by-side to identify the discrepancy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import copy
import numpy as np
from scipy import stats
from collections import defaultdict
from src.isc.improved_emergent_reasoning import (
    ImprovedPhiDrivenSubstrate,
    ImprovedEmergentReasoner
)


def get_all_problems():
    """Standard problem set."""
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


def check_correct(answer, expected):
    """Check if answer matches expected."""
    if answer is None:
        return False
    answer_lower = str(answer).lower().strip()
    expected_lower = str(expected).lower().strip()

    # Exact match
    if answer_lower == expected_lower:
        return True
    # Partial match
    if expected_lower in answer_lower or answer_lower in expected_lower:
        return True
    # Yes/no variants
    yes_variants = ['yes', 'true', 'correct', 'affirmative', 'likely']
    no_variants = ['no', 'false', 'incorrect', 'negative', 'unlikely']

    if expected_lower in yes_variants and any(v in answer_lower for v in yes_variants):
        return True
    if expected_lower in no_variants and any(v in answer_lower for v in no_variants):
        return True

    return False


def run_original_methodology(n_seeds=10):
    """Exact replication of original ablation study."""
    print("\n" + "="*70)
    print("ORIGINAL METHODOLOGY (research_suite.py)")
    print("="*70)

    with_emergence_acc = []
    without_emergence_acc = []
    problems = get_all_problems()

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        # Train with emergence enabled
        substrate1 = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner1 = ImprovedEmergentReasoner(substrate1)

        # Train phase (150 iterations as in original)
        for _ in range(150):
            prob = random.choice(problems)
            result = reasoner1.reason(prob['premise'], prob['question'])
            is_correct = check_correct(result['answer'], prob['expected'])
            reasoner1.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], is_correct
            )

        # Count emergent nodes
        n_emergent = len([n for n in substrate1.nodes if n.startswith('emergent')])

        # Test phase WITH emergence
        correct_with = 0
        total_with = 0
        random.seed(seed + 1000)  # Different test problems
        for _ in range(100):
            prob = random.choice(problems)
            result = reasoner1.reason(prob['premise'], prob['question'])
            if check_correct(result['answer'], prob['expected']):
                correct_with += 1
            total_with += 1

        with_emergence_acc.append(correct_with / total_with)

        # Now ablate: remove emergent nodes and test
        substrate2 = copy.deepcopy(substrate1)
        reasoner2 = ImprovedEmergentReasoner(substrate2)

        # Remove emergent nodes
        emergent_nodes = [n for n in substrate2.nodes if n.startswith('emergent')]
        for node_id in emergent_nodes:
            del substrate2.nodes[node_id]
        # Remove connections to emergent nodes
        for node in substrate2.nodes.values():
            to_remove = [k for k in node.connection_weights if k.startswith('emergent')]
            for k in to_remove:
                del node.connection_weights[k]

        # Test WITHOUT emergence (same seed!)
        correct_without = 0
        total_without = 0
        random.seed(seed + 1000)  # Same test problems
        for _ in range(100):
            prob = random.choice(problems)
            result = reasoner2.reason(prob['premise'], prob['question'])
            if check_correct(result['answer'], prob['expected']):
                correct_without += 1
            total_without += 1

        without_emergence_acc.append(correct_without / total_without)

        print(f"  Seed {seed}: WITH={correct_with}%, WITHOUT={correct_without}%, "
              f"emergent_nodes={n_emergent}")

    # Statistics
    with_arr = np.array(with_emergence_acc)
    without_arr = np.array(without_emergence_acc)

    effect = float(np.mean(with_arr) - np.mean(without_arr))
    t_stat, p_value = stats.ttest_rel(with_arr, without_arr)
    pooled_std = np.sqrt((np.std(with_arr)**2 + np.std(without_arr)**2) / 2)
    cohens_d = effect / pooled_std if pooled_std > 0 else 0

    print(f"\nResults:")
    print(f"  WITH emergence:    {np.mean(with_arr):.1%} (std={np.std(with_arr):.3f})")
    print(f"  WITHOUT emergence: {np.mean(without_arr):.1%} (std={np.std(without_arr):.3f})")
    print(f"  Effect: {effect:+.1%}")
    print(f"  Cohen's d: {cohens_d:.2f}")
    print(f"  p-value: {p_value:.4f}")

    return {
        'method': 'original',
        'with_emergence': float(np.mean(with_arr)),
        'without_emergence': float(np.mean(without_arr)),
        'effect': effect,
        'cohens_d': cohens_d,
        'p_value': float(p_value)
    }


def run_graded_methodology(n_seeds=10):
    """Exact replication of graded ablation study."""
    print("\n" + "="*70)
    print("GRADED METHODOLOGY (advanced_research.py)")
    print("="*70)

    ablation_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    accuracies_by_level = {level: [] for level in ablation_levels}
    problems = get_all_problems()

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        # Train a system (200 iterations as in graded)
        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        for _ in range(200):
            prob = random.choice(problems)
            result = reasoner.reason(prob['premise'], prob['question'])
            is_correct = check_correct(result['answer'], prob['expected'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], is_correct
            )

        # Count emergent nodes
        emergent_nodes = [n for n in substrate.nodes if n.startswith('emergent')]
        n_emergent = len(emergent_nodes)

        # Test at each ablation level
        for level in ablation_levels:
            test_substrate = copy.deepcopy(substrate)
            test_reasoner = ImprovedEmergentReasoner(test_substrate)

            # Remove fraction of emergent nodes
            n_to_remove = int(level * n_emergent)
            random.seed(seed + int(level * 100))
            nodes_to_remove = random.sample(emergent_nodes, n_to_remove) if n_to_remove > 0 else []

            for node_id in nodes_to_remove:
                if node_id in test_substrate.nodes:
                    del test_substrate.nodes[node_id]
            for node in test_substrate.nodes.values():
                for k in list(node.connection_weights.keys()):
                    if k in nodes_to_remove:
                        del node.connection_weights[k]

            # Test
            correct = 0
            total = 0
            random.seed(seed + 5000)
            for _ in range(100):
                prob = random.choice(problems)
                result = test_reasoner.reason(prob['premise'], prob['question'])
                if check_correct(result['answer'], prob['expected']):
                    correct += 1
                total += 1

            accuracies_by_level[level].append(correct / total)

        print(f"  Seed {seed}: 0%={accuracies_by_level[0.0][-1]:.0%}, "
              f"100%={accuracies_by_level[1.0][-1]:.0%}, emergent_nodes={n_emergent}")

    # Analysis
    means = {level: np.mean(accs) for level, accs in accuracies_by_level.items()}

    effect = means[0.0] - means[1.0]  # Positive = emergence helps

    print(f"\nResults:")
    for level in ablation_levels:
        print(f"  {level:.0%} removal: {means[level]:.1%} (std={np.std(accuracies_by_level[level]):.3f})")
    print(f"  Effect (0% - 100%): {effect:+.1%}")

    return {
        'method': 'graded',
        'with_emergence': means[0.0],
        'without_emergence': means[1.0],
        'effect': effect,
        'all_means': means
    }


def run_controlled_comparison(n_seeds=10):
    """
    Run both methods with IDENTICAL conditions to isolate the discrepancy.

    Key variables to control:
    - Same training iterations
    - Same test seed
    - Same problem set
    """
    print("\n" + "="*70)
    print("CONTROLLED COMPARISON (identical conditions)")
    print("="*70)

    problems = get_all_problems()

    original_effects = []
    graded_effects = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        # Train once
        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        for _ in range(150):  # Same as original
            prob = random.choice(problems)
            result = reasoner.reason(prob['premise'], prob['question'])
            is_correct = check_correct(result['answer'], prob['expected'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], is_correct
            )

        emergent_nodes = [n for n in substrate.nodes if n.startswith('emergent')]
        n_emergent = len(emergent_nodes)

        # Test WITH all emergence (both methods)
        test_sub_with = copy.deepcopy(substrate)
        test_reasoner_with = ImprovedEmergentReasoner(test_sub_with)

        random.seed(seed + 1000)
        correct_with = sum(
            1 for _ in range(100)
            if check_correct(
                test_reasoner_with.reason(
                    (p := random.choice(problems))['premise'],
                    p['question']
                )['answer'],
                p['expected']
            )
        )

        # Test WITHOUT emergence (original method: remove ALL)
        test_sub_without = copy.deepcopy(substrate)
        test_reasoner_without = ImprovedEmergentReasoner(test_sub_without)

        for node_id in emergent_nodes:
            if node_id in test_sub_without.nodes:
                del test_sub_without.nodes[node_id]
        for node in test_sub_without.nodes.values():
            for k in list(node.connection_weights.keys()):
                if k in emergent_nodes:
                    del node.connection_weights[k]

        random.seed(seed + 1000)  # Same test seed
        correct_without = sum(
            1 for _ in range(100)
            if check_correct(
                test_reasoner_without.reason(
                    (p := random.choice(problems))['premise'],
                    p['question']
                )['answer'],
                p['expected']
            )
        )

        effect = (correct_with - correct_without) / 100
        original_effects.append(effect)

        print(f"  Seed {seed}: WITH={correct_with}%, WITHOUT={correct_without}%, "
              f"effect={effect:+.0%}, n_emergent={n_emergent}")

    mean_effect = np.mean(original_effects)
    t_stat, p_value = stats.ttest_1samp(original_effects, 0)

    print(f"\nControlled Results:")
    print(f"  Mean effect: {mean_effect:+.1%}")
    print(f"  Std: {np.std(original_effects):.3f}")
    print(f"  t-stat: {t_stat:.2f}, p-value: {p_value:.4f}")

    return {
        'method': 'controlled',
        'effects': original_effects,
        'mean_effect': float(mean_effect),
        'p_value': float(p_value)
    }


def main():
    print("="*70)
    print("ABLATION REPLICATION STUDY")
    print("="*70)
    print("\nGoal: Understand discrepancy between +22% and -4% effects")

    n_seeds = 15

    # Run all three approaches
    original = run_original_methodology(n_seeds)
    graded = run_graded_methodology(n_seeds)
    controlled = run_controlled_comparison(n_seeds)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: ABLATION DISCREPANCY ANALYSIS")
    print("="*70)

    print(f"\nOriginal methodology:")
    print(f"  Effect: {original['effect']:+.1%}")
    print(f"  Cohen's d: {original.get('cohens_d', 'N/A')}")

    print(f"\nGraded methodology:")
    print(f"  Effect: {graded['effect']:+.1%}")

    print(f"\nControlled comparison:")
    print(f"  Effect: {controlled['mean_effect']:+.1%}")
    print(f"  p-value: {controlled['p_value']:.4f}")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if controlled['mean_effect'] > 0.05:
        print("\nFINDING: Emergence HELPS reasoning (controlled replication confirms)")
        print("The graded methodology may have a bug or different test conditions.")
    elif controlled['mean_effect'] < -0.05:
        print("\nFINDING: Emergence HURTS reasoning (unexpected!)")
        print("The original positive result may have been due to methodological artifact.")
    else:
        print("\nFINDING: Emergence has MINIMAL effect (-5% to +5%)")
        print("Both original (+22%) and graded (-4%) may be sampling noise.")

    return {
        'original': original,
        'graded': graded,
        'controlled': controlled
    }


if __name__ == "__main__":
    main()
