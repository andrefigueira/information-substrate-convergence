#!/usr/bin/env python3
"""
Baseline Comparison - Is the emergence effect unique to ISC architecture?

Compares ISC's phi-driven substrate against:
1. Simple pattern matching (no integration)
2. Random guessing baseline
3. Majority class baseline (always "yes")

If ISC's effect is unique, only ISC should show the +80% negation handling.
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
    {'premise': 'All A are B. X is A.', 'question': 'Is X B?', 'expected': 'yes'},
    {'premise': 'If P then Q. P is true.', 'question': 'Is Q true?', 'expected': 'yes'},
    {'premise': 'All C are D. Y is C.', 'question': 'Is Y D?', 'expected': 'yes'},
    {'premise': 'No E is F. Z is E.', 'question': 'Is Z F?', 'expected': 'no'},
]


class RandomBaseline:
    """Random guessing - 50% expected accuracy."""
    def reason(self, premise, question):
        return {'answer': random.choice(['yes', 'no']), 'confidence': 0.5}

    def learn_from_feedback(self, *args):
        pass  # No learning


class MajorityBaseline:
    """Always answers "yes" - matches training distribution."""
    def reason(self, premise, question):
        return {'answer': 'yes', 'confidence': 1.0}

    def learn_from_feedback(self, *args):
        pass  # No learning


class SimplePatternMatcher:
    """
    Simple pattern matching without phi-driven integration.
    Learns direct premise->answer mappings.
    """
    def __init__(self):
        self.patterns = {}  # premise pattern -> answer

    def reason(self, premise, question):
        # Check for exact match
        key = premise.lower().strip()
        if key in self.patterns:
            return {'answer': self.patterns[key], 'confidence': 0.9}

        # Check for partial matches
        for pattern, answer in self.patterns.items():
            if pattern in key or key in pattern:
                return {'answer': answer, 'confidence': 0.7}

        # Default to "yes" (majority class)
        return {'answer': 'yes', 'confidence': 0.3}

    def learn_from_feedback(self, premise, question, expected, actual, correct):
        key = premise.lower().strip()
        self.patterns[key] = expected


class NegationAwareBaseline:
    """
    Pattern matcher that explicitly checks for "No" in premise.
    This is the "cheating" baseline - hardcodes the rule we want to learn.
    """
    def __init__(self):
        self.patterns = {}

    def reason(self, premise, question):
        # Explicit negation detection
        if premise.lower().startswith('no '):
            return {'answer': 'no', 'confidence': 0.9}

        # Check patterns
        key = premise.lower().strip()
        if key in self.patterns:
            return {'answer': self.patterns[key], 'confidence': 0.9}

        return {'answer': 'yes', 'confidence': 0.5}

    def learn_from_feedback(self, premise, question, expected, actual, correct):
        key = premise.lower().strip()
        self.patterns[key] = expected


def test_system(name, system_factory, n_seeds=10, n_train=50):
    """Test a system across multiple seeds."""

    yes_correct = []
    no_correct = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        system = system_factory()

        # Train
        for _ in range(n_train):
            prob = random.choice(PROBLEMS)
            result = system.reason(prob['premise'], prob['question'])
            system.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'],
                check_correct(result['answer'], prob['expected'])
            )

        # Test on each problem type
        for prob in PROBLEMS:
            result = system.reason(prob['premise'], prob['question'])
            correct = check_correct(result['answer'], prob['expected'])

            if prob['expected'] == 'yes':
                yes_correct.append(correct)
            else:
                no_correct.append(correct)

    return {
        'yes_acc': np.mean(yes_correct),
        'no_acc': np.mean(no_correct),
        'overall': np.mean(yes_correct + no_correct),
        'negation_gap': np.mean(yes_correct) - np.mean(no_correct)
    }


def test_isc_with_emergence(n_seeds=10, n_train=50):
    """Test ISC WITH emergent nodes."""

    yes_correct = []
    no_correct = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        # Train
        for _ in range(n_train):
            prob = random.choice(PROBLEMS)
            result = reasoner.reason(prob['premise'], prob['question'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], True
            )

        # Test
        for prob in PROBLEMS:
            result = reasoner.reason(prob['premise'], prob['question'])
            correct = check_correct(result['answer'], prob['expected'])

            if prob['expected'] == 'yes':
                yes_correct.append(correct)
            else:
                no_correct.append(correct)

    return {
        'yes_acc': np.mean(yes_correct),
        'no_acc': np.mean(no_correct),
        'overall': np.mean(yes_correct + no_correct),
        'negation_gap': np.mean(yes_correct) - np.mean(no_correct)
    }


def test_isc_without_emergence(n_seeds=10, n_train=50):
    """Test ISC WITHOUT emergent nodes (ablated)."""

    yes_correct = []
    no_correct = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        # Train
        for _ in range(n_train):
            prob = random.choice(PROBLEMS)
            result = reasoner.reason(prob['premise'], prob['question'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], True
            )

        # Remove emergent nodes
        emergent_nodes = [n for n in list(substrate.nodes.keys()) if n.startswith('emergent')]
        for n in emergent_nodes:
            del substrate.nodes[n]

        # Create new reasoner with ablated substrate
        reasoner = ImprovedEmergentReasoner(substrate)

        # Test
        for prob in PROBLEMS:
            result = reasoner.reason(prob['premise'], prob['question'])
            correct = check_correct(result['answer'], prob['expected'])

            if prob['expected'] == 'yes':
                yes_correct.append(correct)
            else:
                no_correct.append(correct)

    return {
        'yes_acc': np.mean(yes_correct),
        'no_acc': np.mean(no_correct),
        'overall': np.mean(yes_correct + no_correct),
        'negation_gap': np.mean(yes_correct) - np.mean(no_correct)
    }


def main():
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print()
    print("Question: Is the emergence effect unique to ISC architecture?")
    print()
    print("Testing 5 systems:")
    print("  1. ISC WITH emergence (phi-driven)")
    print("  2. ISC WITHOUT emergence (ablated)")
    print("  3. Simple pattern matcher (no integration)")
    print("  4. Majority baseline (always 'yes')")
    print("  5. Negation-aware baseline (hardcoded rule)")
    print()

    n_seeds = 20

    # Suppress emergent node output
    import io
    import contextlib

    print("Running tests (20 seeds each)...")
    print()

    results = {}

    # Test baselines
    print("Testing Random baseline...")
    results['Random'] = test_system('Random', RandomBaseline, n_seeds)

    print("Testing Majority baseline...")
    results['Majority (yes)'] = test_system('Majority', MajorityBaseline, n_seeds)

    print("Testing Simple Pattern Matcher...")
    results['Pattern Matcher'] = test_system('Pattern', SimplePatternMatcher, n_seeds)

    print("Testing Negation-Aware baseline...")
    results['Negation-Aware'] = test_system('Negation', NegationAwareBaseline, n_seeds)

    # Test ISC
    print("Testing ISC WITH emergence...")
    with contextlib.redirect_stdout(io.StringIO()):
        results['ISC + Emergence'] = test_isc_with_emergence(n_seeds)

    print("Testing ISC WITHOUT emergence...")
    with contextlib.redirect_stdout(io.StringIO()):
        results['ISC - Emergence'] = test_isc_without_emergence(n_seeds)

    # Results table
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'System':<20} {'YES acc':<12} {'NO acc':<12} {'Overall':<12} {'Gap':<10}")
    print("-" * 66)

    for name, r in results.items():
        gap_str = f"{r['negation_gap']:+.0%}" if r['negation_gap'] != 0 else "0%"
        print(f"{name:<20} {r['yes_acc']:<12.0%} {r['no_acc']:<12.0%} {r['overall']:<12.0%} {gap_str:<10}")

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    isc_with = results['ISC + Emergence']
    isc_without = results['ISC - Emergence']
    pattern = results['Pattern Matcher']

    print()
    print("Key comparisons:")
    print()

    # ISC emergence effect
    isc_effect = isc_with['no_acc'] - isc_without['no_acc']
    print(f"1. ISC Emergence Effect on NO problems: {isc_effect:+.0%}")

    # Does pattern matcher learn negation?
    pattern_no = pattern['no_acc']
    print(f"2. Simple Pattern Matcher on NO problems: {pattern_no:.0%}")

    # Compare to hardcoded
    negation_aware = results['Negation-Aware']['no_acc']
    print(f"3. Hardcoded Negation Rule on NO problems: {negation_aware:.0%}")

    print()
    print("-" * 70)
    print("CONCLUSION")
    print("-" * 70)

    if isc_effect > 0.5 and pattern_no < 0.5:
        print()
        print("The emergence effect IS UNIQUE to ISC architecture!")
        print()
        print("Evidence:")
        print(f"  - ISC learns negation through emergence: {isc_with['no_acc']:.0%} accuracy")
        print(f"  - Simple pattern matching fails: {pattern_no:.0%} accuracy")
        print(f"  - The phi-driven integration enables this capability")
        print()
        print("Without phi-driven emergence, systems default to majority class ('yes')")
        print("and cannot learn to handle negation from examples alone.")
    else:
        print()
        print("The effect may not be unique to ISC - further investigation needed.")


if __name__ == "__main__":
    main()
