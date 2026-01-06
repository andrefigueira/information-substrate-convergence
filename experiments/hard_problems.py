#!/usr/bin/env python3
"""
Genuinely Hard Problems for Breaking Ceiling Effects

These problems are designed to:
1. Require real reasoning (not keyword matching)
2. Include adversarial cases that trick naive approaches
3. Test at different learning stages (before convergence)
4. Include genuinely ambiguous cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from scipy import stats
from collections import defaultdict

from src.isc.improved_emergent_reasoning import (
    ImprovedPhiDrivenSubstrate,
    ImprovedEmergentReasoner
)


# Adversarial problems designed to trick keyword-based approaches
ADVERSARIAL_PROBLEMS = [
    # Negation traps
    {"premise": "All birds can fly. Penguins are birds that cannot fly.",
     "question": "Can penguins fly?", "expected": "no", "trap": "keyword 'can fly' misleads"},

    {"premise": "Not all that glitters is gold. This glitters.",
     "question": "Is this gold?", "expected": "uncertain", "trap": "double negation"},

    # Scope ambiguity
    {"premise": "Every student passed a test. There were 5 tests.",
     "question": "Did all students pass the same test?", "expected": "uncertain", "trap": "scope ambiguity"},

    # Contraposition failures
    {"premise": "All dogs bark. Rex does not bark.",
     "question": "Is Rex a dog?", "expected": "no", "trap": "must use contraposition"},

    # Hidden assumptions
    {"premise": "Socrates is a man. All men are mortal.",
     "question": "Is Socrates immortal?", "expected": "no", "trap": "requires negation of conclusion"},

    # Statistical vs absolute
    {"premise": "90% of swans are white. This is a swan.",
     "question": "Is this swan definitely white?", "expected": "no", "trap": "probabilistic vs certain"},

    # Temporal reasoning
    {"premise": "John was rich. John lost his money.",
     "question": "Is John rich now?", "expected": "no", "trap": "temporal change"},

    # Multiple quantifiers
    {"premise": "Some A are B. Some B are C.",
     "question": "Are some A definitely C?", "expected": "no", "trap": "invalid syllogism"},

    # Exclusive vs inclusive or
    {"premise": "The suspect was in Paris or London on Monday. Evidence shows Paris.",
     "question": "Could the suspect have been in London?", "expected": "no", "trap": "exclusive location"},

    # False dichotomy
    {"premise": "If it's raining, bring an umbrella. It's not raining.",
     "question": "Should you not bring an umbrella?", "expected": "uncertain", "trap": "affirming consequent"},
]

# Multi-step reasoning problems
MULTISTEP_PROBLEMS = [
    {"premise": "A implies B. B implies C. C implies D. A is true.",
     "question": "Is D true?", "expected": "yes", "steps": 3},

    {"premise": "If X then Y. If Y then Z. If Z then W. Not W.",
     "question": "Is X true?", "expected": "no", "steps": 3},

    {"premise": "All A are B. All B are C. All C are D. X is an A.",
     "question": "Is X a D?", "expected": "yes", "steps": 3},

    {"premise": "P or Q. If P then R. If Q then R. Not both P and Q.",
     "question": "Is R true?", "expected": "yes", "steps": 2},

    {"premise": "A is north of B. B is north of C. C is north of D.",
     "question": "Is A north of D?", "expected": "yes", "steps": 3},
]

# Genuinely ambiguous problems (no clear right answer)
AMBIGUOUS_PROBLEMS = [
    {"premise": "The bank was closed. John went to the bank.",
     "question": "Did John go to a financial institution?", "expected": "uncertain"},

    {"premise": "Flying planes can be dangerous.",
     "question": "Is piloting or the planes themselves dangerous?", "expected": "uncertain"},

    {"premise": "I saw her duck.",
     "question": "Did I see a bird or an action?", "expected": "uncertain"},
]


def check_correct(answer, expected):
    """Check if answer matches expected."""
    if answer is None:
        return False
    answer_lower = str(answer).lower().strip()
    expected_lower = str(expected).lower().strip()

    if answer_lower == expected_lower:
        return True
    if expected_lower in answer_lower or answer_lower in expected_lower:
        return True

    yes_variants = ['yes', 'true', 'correct', 'affirmative']
    no_variants = ['no', 'false', 'incorrect', 'negative']
    uncertain_variants = ['uncertain', 'maybe', 'possibly', 'unknown', 'cannot determine']

    if expected_lower in yes_variants and any(v in answer_lower for v in yes_variants):
        return True
    if expected_lower in no_variants and any(v in answer_lower for v in no_variants):
        return True
    if expected_lower in uncertain_variants and any(v in answer_lower for v in uncertain_variants):
        return True

    return False


def test_at_learning_stages():
    """
    Test accuracy at different stages of learning to find when ceiling hits.
    """
    print("=" * 70)
    print("TESTING AT DIFFERENT LEARNING STAGES")
    print("=" * 70)

    all_problems = ADVERSARIAL_PROBLEMS + MULTISTEP_PROBLEMS
    stages = [0, 10, 25, 50, 100, 150, 200]
    n_seeds = 10

    results_by_stage = {stage: [] for stage in stages}

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        # Test at stage 0 (no training)
        correct = 0
        for prob in all_problems:
            result = reasoner.reason(prob['premise'], prob['question'])
            if check_correct(result['answer'], prob['expected']):
                correct += 1
        results_by_stage[0].append(correct / len(all_problems))

        # Train and test at each stage
        training_problems = [
            {"premise": "All X are Y. Z is an X.", "question": "Is Z a Y?", "expected": "yes"},
            {"premise": "If A then B. A is true.", "question": "Is B true?", "expected": "yes"},
            {"premise": "No A is B. X is A.", "question": "Is X a B?", "expected": "no"},
        ]

        for stage_idx, stage in enumerate(stages[1:], 1):
            prev_stage = stages[stage_idx - 1]
            iterations_needed = stage - prev_stage

            for _ in range(iterations_needed):
                prob = random.choice(training_problems)
                result = reasoner.reason(prob['premise'], prob['question'])
                is_correct = check_correct(result['answer'], prob['expected'])
                reasoner.learn_from_feedback(
                    prob['premise'], prob['question'],
                    prob['expected'], result['answer'], is_correct
                )

            # Test
            correct = 0
            for prob in all_problems:
                result = reasoner.reason(prob['premise'], prob['question'])
                if check_correct(result['answer'], prob['expected']):
                    correct += 1
            results_by_stage[stage].append(correct / len(all_problems))

    print("\nAccuracy by training stage:")
    for stage in stages:
        mean_acc = np.mean(results_by_stage[stage])
        std_acc = np.std(results_by_stage[stage])
        print(f"  Stage {stage:3d}: {mean_acc:.1%} (std={std_acc:.3f})")

    return results_by_stage


def test_adversarial_only():
    """
    Test only on adversarial problems to see if they break the ceiling.
    """
    print("\n" + "=" * 70)
    print("ADVERSARIAL PROBLEMS ONLY")
    print("=" * 70)

    n_seeds = 20
    n_train = 50  # Limited training

    accuracies = []
    problem_results = defaultdict(list)

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        # Limited training
        training_problems = [
            {"premise": "All X are Y. Z is an X.", "question": "Is Z a Y?", "expected": "yes"},
            {"premise": "If A then B. A is true.", "question": "Is B true?", "expected": "yes"},
        ]

        for _ in range(n_train):
            prob = random.choice(training_problems)
            result = reasoner.reason(prob['premise'], prob['question'])
            is_correct = check_correct(result['answer'], prob['expected'])
            reasoner.learn_from_feedback(
                prob['premise'], prob['question'],
                prob['expected'], result['answer'], is_correct
            )

        # Test adversarial
        correct = 0
        for i, prob in enumerate(ADVERSARIAL_PROBLEMS):
            result = reasoner.reason(prob['premise'], prob['question'])
            is_correct = check_correct(result['answer'], prob['expected'])
            problem_results[i].append(is_correct)
            if is_correct:
                correct += 1

        accuracies.append(correct / len(ADVERSARIAL_PROBLEMS))

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"\nOverall adversarial accuracy: {mean_acc:.1%} (std={std_acc:.3f})")
    print("\nPer-problem accuracy:")
    for i, prob in enumerate(ADVERSARIAL_PROBLEMS):
        prob_acc = np.mean(problem_results[i])
        trap = prob.get('trap', 'N/A')
        print(f"  {i+1}. {prob_acc:.0%} - Trap: {trap}")

    return mean_acc, problem_results


def test_phi_correlation_on_hard():
    """
    Test phi-accuracy correlation specifically on hard problems.
    """
    print("\n" + "=" * 70)
    print("PHI-ACCURACY CORRELATION ON HARD PROBLEMS")
    print("=" * 70)

    n_seeds = 15
    all_problems = ADVERSARIAL_PROBLEMS + MULTISTEP_PROBLEMS

    all_phis = []
    all_corrects = []

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
        reasoner = ImprovedEmergentReasoner(substrate)

        # No training - test raw reasoning
        for prob in all_problems:
            phi = substrate.global_phi
            result = reasoner.reason(prob['premise'], prob['question'])
            is_correct = check_correct(result['answer'], prob['expected'])

            all_phis.append(phi)
            all_corrects.append(float(is_correct))

    # Correlation
    if np.std(all_corrects) > 0.01:
        corr, p_value = stats.pearsonr(all_phis, all_corrects)
        print(f"\nPhi-accuracy correlation: r={corr:.3f}, p={p_value:.4f}")
        print(f"Accuracy variance: {np.var(all_corrects):.4f}")
        print(f"Mean accuracy: {np.mean(all_corrects):.1%}")

        if p_value < 0.05:
            print("SIGNIFICANT phi-accuracy relationship found on hard problems!")
        else:
            print("No significant relationship.")
    else:
        print("\nInsufficient variance for correlation (still at ceiling)")

    return all_phis, all_corrects


def main():
    print("=" * 70)
    print("HARD PROBLEMS STUDY")
    print("Breaking the ceiling effect with genuinely difficult problems")
    print("=" * 70)

    # Test 1: Learning stages
    stage_results = test_at_learning_stages()

    # Test 2: Adversarial only
    adv_acc, adv_results = test_adversarial_only()

    # Test 3: Phi correlation on hard
    phis, corrects = test_phi_correlation_on_hard()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    initial_acc = np.mean(stage_results[0])
    final_acc = np.mean(stage_results[200])

    print(f"\nLearning curve:")
    print(f"  Initial (untrained): {initial_acc:.1%}")
    print(f"  Final (200 iters):   {final_acc:.1%}")
    print(f"  Improvement: {final_acc - initial_acc:+.1%}")

    print(f"\nAdversarial accuracy: {adv_acc:.1%}")

    if adv_acc < 0.9:
        print("\nCEILING BROKEN: Adversarial problems reveal system limitations")
    else:
        print("\nCeiling persists even on adversarial problems")

    mean_correct = np.mean(corrects)
    if np.std(corrects) > 0.01:
        corr, _ = stats.pearsonr(phis, corrects)
        print(f"\nPhi-accuracy on hard: r={corr:.3f}, mean acc={mean_correct:.1%}")
    else:
        print(f"\nMean accuracy on hard: {mean_correct:.1%} (no variance for correlation)")


if __name__ == "__main__":
    main()
