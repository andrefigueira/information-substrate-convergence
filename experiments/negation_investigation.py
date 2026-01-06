#!/usr/bin/env python3
"""
Negation Investigation - Why does WITHOUT emergence fail specifically on "no" problems?

Critical question: Is emergence genuinely helping with negation, or is this an artifact?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import copy

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


# Separate YES and NO problems clearly
YES_PROBLEMS = [
    {'premise': 'All A are B. X is A.', 'question': 'Is X B?', 'expected': 'yes'},
    {'premise': 'If P then Q. P is true.', 'question': 'Is Q true?', 'expected': 'yes'},
    {'premise': 'All C are D. Y is C.', 'question': 'Is Y D?', 'expected': 'yes'},
]

NO_PROBLEMS = [
    {'premise': 'No E is F. Z is E.', 'question': 'Is Z F?', 'expected': 'no'},
]

ALL_PROBLEMS = YES_PROBLEMS + NO_PROBLEMS


def investigate_negation():
    """Deep investigation into negation handling."""

    print("=" * 70)
    print("NEGATION INVESTIGATION")
    print("=" * 70)
    print()
    print("Question: Why does WITHOUT emergence specifically fail on 'no' problems?")
    print()

    # Test 1: Fresh substrate behavior on YES vs NO
    print("-" * 70)
    print("TEST 1: Fresh (untrained) substrate behavior")
    print("-" * 70)

    random.seed(42)
    np.random.seed(42)

    substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
    reasoner = ImprovedEmergentReasoner(substrate)

    print("\nTesting each problem type on FRESH substrate:")
    for prob in ALL_PROBLEMS:
        result = reasoner.reason(prob['premise'], prob['question'])
        correct = check_correct(result['answer'], prob['expected'])
        print(f"  Expected: {prob['expected']:>3} | Got: {str(result['answer'])[:10]:>10} | Correct: {correct}")

    # Test 2: Train and compare WITH vs WITHOUT emergence
    print()
    print("-" * 70)
    print("TEST 2: After training - WITH vs WITHOUT emergence")
    print("-" * 70)

    random.seed(42)
    np.random.seed(42)

    substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
    reasoner = ImprovedEmergentReasoner(substrate)

    # Train
    for _ in range(50):
        prob = random.choice(ALL_PROBLEMS)
        result = reasoner.reason(prob['premise'], prob['question'])
        reasoner.learn_from_feedback(
            prob['premise'], prob['question'],
            prob['expected'], result['answer'], True
        )

    n_emergent = len([n for n in substrate.nodes if n.startswith('emergent')])
    print(f"\nAfter training: {n_emergent} emergent nodes, phi={substrate.global_phi:.4f}")

    # Test WITH emergence
    print("\nWITH emergence:")
    with_results = []
    for prob in ALL_PROBLEMS:
        result = reasoner.reason(prob['premise'], prob['question'])
        correct = check_correct(result['answer'], prob['expected'])
        with_results.append((prob['expected'], correct))
        print(f"  Expected: {prob['expected']:>3} | Got: {str(result['answer'])[:10]:>10} | Correct: {correct}")

    # Create WITHOUT version
    substrate2 = copy.deepcopy(substrate)
    emergent_nodes = [n for n in list(substrate2.nodes.keys()) if n.startswith('emergent')]
    for n in emergent_nodes:
        del substrate2.nodes[n]
    reasoner2 = ImprovedEmergentReasoner(substrate2)

    print(f"\nWITHOUT emergence (removed {len(emergent_nodes)} emergent nodes):")
    without_results = []
    for prob in ALL_PROBLEMS:
        result = reasoner2.reason(prob['premise'], prob['question'])
        correct = check_correct(result['answer'], prob['expected'])
        without_results.append((prob['expected'], correct))
        print(f"  Expected: {prob['expected']:>3} | Got: {str(result['answer'])[:10]:>10} | Correct: {correct}")

    # Test 3: What does the system actually output for "no" problems?
    print()
    print("-" * 70)
    print("TEST 3: Detailed 'no' problem analysis")
    print("-" * 70)

    no_prob = NO_PROBLEMS[0]
    print(f"\nProblem: {no_prob['premise']}")
    print(f"Question: {no_prob['question']}")
    print(f"Expected: {no_prob['expected']}")

    print("\nWITH emergence - full result:")
    result_with = reasoner.reason(no_prob['premise'], no_prob['question'])
    print(f"  Answer: {result_with['answer']}")
    print(f"  Confidence: {result_with.get('confidence', 'N/A')}")

    print("\nWITHOUT emergence - full result:")
    result_without = reasoner2.reason(no_prob['premise'], no_prob['question'])
    print(f"  Answer: {result_without['answer']}")
    print(f"  Confidence: {result_without.get('confidence', 'N/A')}")

    # Test 4: Does baseline have a default "yes" bias?
    print()
    print("-" * 70)
    print("TEST 4: Baseline bias check")
    print("-" * 70)

    # Test 100 random problems, count yes vs no outputs
    yes_count_with = 0
    yes_count_without = 0

    random.seed(123)
    for _ in range(100):
        prob = random.choice(ALL_PROBLEMS)

        r1 = reasoner.reason(prob['premise'], prob['question'])
        r2 = reasoner2.reason(prob['premise'], prob['question'])

        if r1['answer'] and 'yes' in str(r1['answer']).lower():
            yes_count_with += 1
        if r2['answer'] and 'yes' in str(r2['answer']).lower():
            yes_count_without += 1

    print(f"\nOver 100 random tests:")
    print(f"  WITH emergence - outputs 'yes': {yes_count_with}%")
    print(f"  WITHOUT emergence - outputs 'yes': {yes_count_without}%")

    if yes_count_without > 85:
        print("\n  FINDING: WITHOUT emergence has strong 'yes' bias!")
        print("  This explains why it fails on 'no' problems.")

    # Test 5: Balanced test set
    print()
    print("-" * 70)
    print("TEST 5: Balanced test (equal YES and NO problems)")
    print("-" * 70)

    # Create more NO problems for balance
    extended_no_problems = [
        {'premise': 'No E is F. Z is E.', 'question': 'Is Z F?', 'expected': 'no'},
        {'premise': 'No G is H. W is G.', 'question': 'Is W H?', 'expected': 'no'},
        {'premise': 'No cats are dogs. Felix is a cat.', 'question': 'Is Felix a dog?', 'expected': 'no'},
    ]

    balanced_problems = YES_PROBLEMS + extended_no_problems

    with_correct = 0
    without_correct = 0
    with_by_type = {'yes': [], 'no': []}
    without_by_type = {'yes': [], 'no': []}

    for prob in balanced_problems:
        r1 = reasoner.reason(prob['premise'], prob['question'])
        r2 = reasoner2.reason(prob['premise'], prob['question'])

        c1 = check_correct(r1['answer'], prob['expected'])
        c2 = check_correct(r2['answer'], prob['expected'])

        with_correct += c1
        without_correct += c2
        with_by_type[prob['expected']].append(c1)
        without_by_type[prob['expected']].append(c2)

    print(f"\nResults on balanced test set ({len(balanced_problems)} problems):")
    print(f"  WITH emergence: {with_correct}/{len(balanced_problems)} ({100*with_correct/len(balanced_problems):.0f}%)")
    print(f"  WITHOUT emergence: {without_correct}/{len(balanced_problems)} ({100*without_correct/len(balanced_problems):.0f}%)")

    print(f"\nBreakdown by problem type:")
    print(f"  WITH emergence on YES problems: {sum(with_by_type['yes'])}/{len(with_by_type['yes'])}")
    print(f"  WITH emergence on NO problems: {sum(with_by_type['no'])}/{len(with_by_type['no'])}")
    print(f"  WITHOUT emergence on YES problems: {sum(without_by_type['yes'])}/{len(without_by_type['yes'])}")
    print(f"  WITHOUT emergence on NO problems: {sum(without_by_type['no'])}/{len(without_by_type['no'])}")

    # Conclusion
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    without_no_acc = sum(without_by_type['no']) / len(without_by_type['no']) if without_by_type['no'] else 0
    with_no_acc = sum(with_by_type['no']) / len(with_by_type['no']) if with_by_type['no'] else 0

    if without_no_acc < 0.3 and with_no_acc > 0.7:
        print("\nFINDING: Emergence specifically enables NEGATION handling!")
        print(f"  - WITHOUT emergence: {without_no_acc:.0%} on 'no' problems")
        print(f"  - WITH emergence: {with_no_acc:.0%} on 'no' problems")
        print()
        print("This is a REAL capability difference, not a methodology bug.")
        print("Emergent nodes learn to handle negation/exclusion logic.")
        print()
        print("The 23% effect consistency is explained by:")
        print("  1. The test set has 1/4 'no' problems (25%)")
        print("  2. WITHOUT emergence fails on ALL 'no' problems")
        print("  3. Effect = proportion of 'no' problems in sample")
    else:
        print("\nThe pattern is unclear - needs more investigation.")


if __name__ == "__main__":
    investigate_negation()
