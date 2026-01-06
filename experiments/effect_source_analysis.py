#!/usr/bin/env python3
"""
Effect Source Analysis - Where does the emergence effect actually come from?

Tests multiple hypotheses:
1. Negation-specific (fails on "no" problems)
2. General accuracy boost
3. Variance reduction
4. Test set composition artifact
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import copy
from collections import defaultdict

from src.isc.improved_emergent_reasoning import (
    ImprovedPhiDrivenSubstrate,
    ImprovedEmergentReasoner
)


def check_correct(answer, expected):
    """Check if answer matches expected."""
    if answer is None:
        return False
    a, e = str(answer).lower().strip(), str(expected).lower().strip()
    if a == e:
        return True
    if e in a or a in e:
        return True
    if e in ['yes', 'true'] and any(v in a for v in ['yes', 'true']):
        return True
    if e in ['no', 'false'] and any(v in a for v in ['no', 'false']):
        return True
    return False


PROBLEMS = [
    {'premise': 'All A are B. X is A.', 'question': 'Is X B?', 'expected': 'yes', 'id': 'yes_1'},
    {'premise': 'If P then Q. P is true.', 'question': 'Is Q true?', 'expected': 'yes', 'id': 'yes_2'},
    {'premise': 'All C are D. Y is C.', 'question': 'Is Y D?', 'expected': 'yes', 'id': 'yes_3'},
    {'premise': 'No E is F. Z is E.', 'question': 'Is Z F?', 'expected': 'no', 'id': 'no_1'},
]


def analyze_effect_source(n_seeds=20):
    """Analyze where the effect comes from across many seeds."""

    print("=" * 70)
    print("EFFECT SOURCE ANALYSIS")
    print("=" * 70)
    print(f"Testing {n_seeds} seeds to find true source of emergence effect")
    print()

    # Track per-problem performance
    with_correct_by_problem = defaultdict(list)
    without_correct_by_problem = defaultdict(list)

    # Track overall
    all_effects = []
    with_accs = []
    without_accs = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        # Train
        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        for _ in range(50):
            prob = random.choice(PROBLEMS)
            result = reasoner.reason(prob['premise'], prob['question'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], True
            )

        # Create without version
        substrate2 = copy.deepcopy(substrate)
        emergent_nodes = [n for n in list(substrate2.nodes.keys()) if n.startswith('emergent')]
        for n in emergent_nodes:
            del substrate2.nodes[n]
        reasoner2 = ImprovedEmergentReasoner(substrate2)

        # Test on EVERY problem (not random sample) - deterministic test
        with_correct = 0
        without_correct = 0

        for prob in PROBLEMS:
            r1 = reasoner.reason(prob['premise'], prob['question'])
            r2 = reasoner2.reason(prob['premise'], prob['question'])

            c1 = check_correct(r1['answer'], prob['expected'])
            c2 = check_correct(r2['answer'], prob['expected'])

            with_correct += c1
            without_correct += c2

            with_correct_by_problem[prob['id']].append(c1)
            without_correct_by_problem[prob['id']].append(c2)

        with_acc = with_correct / len(PROBLEMS)
        without_acc = without_correct / len(PROBLEMS)
        effect = with_acc - without_acc

        with_accs.append(with_acc)
        without_accs.append(without_acc)
        all_effects.append(effect)

        print(f"Seed {seed:2d}: WITH={with_acc:.0%}, WITHOUT={without_acc:.0%}, effect={effect:+.0%}")

    # Analysis
    print()
    print("-" * 70)
    print("PER-PROBLEM ACCURACY")
    print("-" * 70)
    print()
    print(f"{'Problem':<10} {'WITH':<15} {'WITHOUT':<15} {'Difference':<15}")
    print("-" * 55)

    problem_diffs = {}
    for prob in PROBLEMS:
        pid = prob['id']
        with_rate = np.mean(with_correct_by_problem[pid])
        without_rate = np.mean(without_correct_by_problem[pid])
        diff = with_rate - without_rate
        problem_diffs[pid] = diff
        print(f"{pid:<10} {with_rate:<15.1%} {without_rate:<15.1%} {diff:+.1%}")

    print()
    print("-" * 70)
    print("FINDINGS")
    print("-" * 70)

    # Check if effect is concentrated in specific problems
    max_diff_prob = max(problem_diffs, key=problem_diffs.get)
    min_diff_prob = min(problem_diffs, key=problem_diffs.get)

    if problem_diffs[max_diff_prob] > 0.3:
        print(f"\nFINDING: Effect concentrated in problem '{max_diff_prob}'")
        print(f"  This problem shows {problem_diffs[max_diff_prob]:+.0%} difference")
    else:
        print(f"\nFINDING: Effect is distributed across problems")

    # Check overall statistics
    mean_effect = np.mean(all_effects)
    std_effect = np.std(all_effects)

    print()
    print(f"Overall mean effect: {mean_effect:+.1%} (std={std_effect:.1%})")
    print(f"Mean WITH accuracy: {np.mean(with_accs):.1%}")
    print(f"Mean WITHOUT accuracy: {np.mean(without_accs):.1%}")

    # Check if WITHOUT has systematic bias
    yes_problems = [p for p in PROBLEMS if p['expected'] == 'yes']
    no_problems = [p for p in PROBLEMS if p['expected'] == 'no']

    without_yes_rate = np.mean([np.mean(without_correct_by_problem[p['id']]) for p in yes_problems])
    without_no_rate = np.mean([np.mean(without_correct_by_problem[p['id']]) for p in no_problems])

    print()
    print(f"WITHOUT emergence on YES problems: {without_yes_rate:.1%}")
    print(f"WITHOUT emergence on NO problems: {without_no_rate:.1%}")

    if without_yes_rate - without_no_rate > 0.3:
        print("\nFINDING: WITHOUT emergence has YES bias (struggles with negation)")
    elif without_no_rate - without_yes_rate > 0.3:
        print("\nFINDING: WITHOUT emergence has NO bias")
    else:
        print("\nFINDING: WITHOUT emergence has no strong bias by answer type")

    # Return data for further analysis
    return {
        'with_accs': with_accs,
        'without_accs': without_accs,
        'effects': all_effects,
        'by_problem': {
            'with': dict(with_correct_by_problem),
            'without': dict(without_correct_by_problem)
        }
    }


def test_with_balanced_problems(n_seeds=20):
    """Test with balanced YES/NO problem set."""

    print()
    print("=" * 70)
    print("BALANCED PROBLEM SET TEST")
    print("=" * 70)

    # Add more NO problems for balance
    balanced_problems = [
        {'premise': 'All A are B. X is A.', 'question': 'Is X B?', 'expected': 'yes'},
        {'premise': 'If P then Q. P is true.', 'question': 'Is Q true?', 'expected': 'yes'},
        {'premise': 'No E is F. Z is E.', 'question': 'Is Z F?', 'expected': 'no'},
        {'premise': 'No G is H. W is G.', 'question': 'Is W H?', 'expected': 'no'},
    ]

    print(f"Testing with 50/50 YES/NO balance: {len(balanced_problems)} problems")
    print()

    all_effects = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        # Train on balanced problems
        for _ in range(50):
            prob = random.choice(balanced_problems)
            result = reasoner.reason(prob['premise'], prob['question'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], True
            )

        # Create without version
        substrate2 = copy.deepcopy(substrate)
        for n in [n for n in list(substrate2.nodes.keys()) if n.startswith('emergent')]:
            del substrate2.nodes[n]
        reasoner2 = ImprovedEmergentReasoner(substrate2)

        # Test
        with_correct = sum(
            1 for prob in balanced_problems
            if check_correct(reasoner.reason(prob['premise'], prob['question'])['answer'], prob['expected'])
        )
        without_correct = sum(
            1 for prob in balanced_problems
            if check_correct(reasoner2.reason(prob['premise'], prob['question'])['answer'], prob['expected'])
        )

        effect = (with_correct - without_correct) / len(balanced_problems)
        all_effects.append(effect)

    mean_effect = np.mean(all_effects)
    std_effect = np.std(all_effects)

    print(f"Mean effect with balanced problems: {mean_effect:+.1%} (std={std_effect:.1%})")

    if abs(mean_effect) < 0.05:
        print("\nFINDING: Effect disappears with balanced problems!")
        print("This suggests the effect was an artifact of unbalanced test set.")
    elif mean_effect > 0.1:
        print("\nFINDING: Effect persists with balanced problems!")
        print("This is genuine emergence benefit.")
    else:
        print("\nFINDING: Effect is reduced but still present.")


if __name__ == "__main__":
    import contextlib
    import io

    # Suppress emergent node output
    with contextlib.redirect_stdout(io.StringIO()):
        data = analyze_effect_source(n_seeds=20)

    # Print the captured analysis
    analyze_effect_source(n_seeds=20)
    test_with_balanced_problems(n_seeds=20)
