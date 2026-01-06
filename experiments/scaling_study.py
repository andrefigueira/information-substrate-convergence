#!/usr/bin/env python3
"""
Scaling Study: Testing emergence effect at large substrate sizes.

Validates that ISC findings are not toy artifacts by testing at:
- 100 nodes (baseline)
- 500 nodes
- 1000 nodes
- 5000 nodes
- 100000 nodes (target)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from scipy import stats
import copy
import time
import json
from datetime import datetime
from pathlib import Path

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


def run_scaling_study(scales, n_seeds=10, verbose=True):
    """
    Run ablation study at multiple scales.

    Args:
        scales: List of substrate sizes to test
        n_seeds: Number of random seeds per scale
        verbose: Print progress

    Returns:
        Dictionary of results by scale
    """
    problems = [
        {'premise': 'All A are B. X is A.', 'question': 'Is X B?', 'expected': 'yes'},
        {'premise': 'If P then Q. P is true.', 'question': 'Is Q true?', 'expected': 'yes'},
        {'premise': 'All C are D. Y is C.', 'question': 'Is Y D?', 'expected': 'yes'},
        {'premise': 'No E is F. Z is E.', 'question': 'Is Z F?', 'expected': 'no'},
    ]

    results = {}

    for scale in scales:
        if verbose:
            print(f'\n[SCALE: {scale:,} nodes]')

        start_time = time.time()

        with_accs = []
        without_accs = []
        effects = []
        emergent_counts = []
        phi_values = []

        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            # Create substrate at scale
            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=scale)
            reasoner = ImprovedEmergentReasoner(substrate)

            # Scale training iterations with substrate size (with cap)
            n_train = min(200, 50 + scale // 100)

            for i in range(n_train):
                prob = random.choice(problems)
                result = reasoner.reason(prob['premise'], prob['question'])
                reasoner.learn_from_feedback(
                    prob['premise'], prob['question'],
                    prob['expected'], result['answer'], True
                )

                # Progress for large scales
                if verbose and scale >= 10000 and i % 50 == 0:
                    print(f'    Training: {i}/{n_train}', end='\r')

            n_emergent = len([n for n in substrate.nodes if n.startswith('emergent')])
            emergent_counts.append(n_emergent)
            phi_values.append(substrate.global_phi)

            # Test WITH emergence
            random.seed(seed + 5000)
            with_correct = sum(
                1 for _ in range(30)
                if check_correct(
                    reasoner.reason(
                        (p := random.choice(problems))['premise'],
                        p['question']
                    )['answer'],
                    p['expected']
                )
            )

            # Test WITHOUT emergence (ablate)
            substrate2 = copy.deepcopy(substrate)
            emergent_nodes = [n for n in list(substrate2.nodes.keys()) if n.startswith('emergent')]
            for n in emergent_nodes:
                del substrate2.nodes[n]
            reasoner2 = ImprovedEmergentReasoner(substrate2)

            random.seed(seed + 5000)  # Same test seed for fair comparison
            without_correct = sum(
                1 for _ in range(30)
                if check_correct(
                    reasoner2.reason(
                        (p := random.choice(problems))['premise'],
                        p['question']
                    )['answer'],
                    p['expected']
                )
            )

            with_accs.append(with_correct / 30)
            without_accs.append(without_correct / 30)
            effects.append((with_correct - without_correct) / 30)

            if verbose:
                print(f'  Seed {seed}: emergent={n_emergent}, phi={substrate.global_phi:.4f}, effect={effects[-1]:+.0%}')

        elapsed = time.time() - start_time

        # Statistics
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        if std_effect > 0:
            t_stat, p_value = stats.ttest_rel(with_accs, without_accs)
        else:
            t_stat, p_value = 0, 1.0

        results[scale] = {
            'with_acc': float(np.mean(with_accs)),
            'without_acc': float(np.mean(without_accs)),
            'effect': float(mean_effect),
            'std': float(std_effect),
            'p_value': float(p_value),
            't_stat': float(t_stat),
            'emergent_mean': float(np.mean(emergent_counts)),
            'emergent_std': float(np.std(emergent_counts)),
            'phi_mean': float(np.mean(phi_values)),
            'phi_std': float(np.std(phi_values)),
            'time_seconds': elapsed,
            'n_seeds': n_seeds
        }

        if verbose:
            print(f'  ---')
            print(f'  Mean effect: {mean_effect:+.1%} (std={std_effect:.1%})')
            print(f'  p-value: {p_value:.4f}')
            print(f'  Emergent nodes: {np.mean(emergent_counts):.0f}')
            print(f'  Mean phi: {np.mean(phi_values):.4f}')
            print(f'  Time: {elapsed:.1f}s')

    return results


def save_results(results, output_dir="results/research"):
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_path / f"scaling_study_{timestamp}.json"

    data = {
        'timestamp': timestamp,
        'results': {str(k): v for k, v in results.items()}
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def print_summary(results):
    """Print summary table."""
    print('\n' + '=' * 70)
    print('SCALING SUMMARY')
    print('=' * 70)
    print('Scale      Effect     p-value    Emergent   Phi        Time')
    print('-' * 65)

    for scale in sorted(results.keys()):
        r = results[scale]
        sig = '*' if r['p_value'] < 0.05 else ''
        time_str = f"{r['time_seconds']:.0f}s" if r['time_seconds'] < 3600 else f"{r['time_seconds']/3600:.1f}h"
        print(f"{scale:<10} {r['effect']:+.1%}{sig:<4} {r['p_value']:<10.4f} {r['emergent_mean']:<10.0f} {r['phi_mean']:<10.4f} {time_str}")


def main():
    print('=' * 70)
    print('ISC SCALING STUDY')
    print('=' * 70)

    # Default scales
    scales = [100, 500, 1000, 5000]

    # Check for command line args
    if len(sys.argv) > 1:
        scales = [int(s) for s in sys.argv[1:]]

    print(f'Testing scales: {scales}')

    results = run_scaling_study(scales, n_seeds=10)
    print_summary(results)
    save_results(results)

    # Conclusion
    print('\n' + '=' * 70)
    print('CONCLUSION')
    print('=' * 70)

    max_scale = max(results.keys())
    if results[max_scale]['effect'] > 0 and results[max_scale]['p_value'] < 0.05:
        print(f"Emergence effect CONFIRMED at {max_scale:,} nodes")
        print(f"Effect: {results[max_scale]['effect']:+.1%}, p={results[max_scale]['p_value']:.4f}")
    else:
        print(f"Emergence effect not confirmed at {max_scale:,} nodes")


if __name__ == "__main__":
    main()
