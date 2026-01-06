#!/usr/bin/env python3
"""
Parallel Scaling Study - Uses multiprocessing for faster execution.

Each seed runs in a separate process for ~N-fold speedup on N cores.
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
from multiprocessing import Pool, cpu_count
from functools import partial

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


def run_single_seed(args):
    """Run ablation study for a single seed. Designed for parallel execution."""
    seed, scale, n_train, n_test = args

    random.seed(seed)
    np.random.seed(seed)

    # Suppress emergent node output
    import io
    import contextlib

    # Create and train substrate
    substrate = ImprovedPhiDrivenSubstrate(initial_nodes=scale)
    reasoner = ImprovedEmergentReasoner(substrate)

    for _ in range(n_train):
        prob = random.choice(PROBLEMS)
        result = reasoner.reason(prob['premise'], prob['question'])
        reasoner.learn_from_feedback(
            prob['premise'], prob['question'],
            prob['expected'], result['answer'], True
        )

    n_emergent = len([n for n in substrate.nodes if n.startswith('emergent')])
    phi = substrate.global_phi

    # Test WITH emergence
    random.seed(seed + 5000)
    with_correct = sum(
        1 for _ in range(n_test)
        if check_correct(
            reasoner.reason(
                (p := random.choice(PROBLEMS))['premise'],
                p['question']
            )['answer'],
            p['expected']
        )
    )

    # Test WITHOUT emergence
    substrate2 = copy.deepcopy(substrate)
    emergent_nodes = [n for n in list(substrate2.nodes.keys()) if n.startswith('emergent')]
    for n in emergent_nodes:
        del substrate2.nodes[n]
    reasoner2 = ImprovedEmergentReasoner(substrate2)

    random.seed(seed + 5000)
    without_correct = sum(
        1 for _ in range(n_test)
        if check_correct(
            reasoner2.reason(
                (p := random.choice(PROBLEMS))['premise'],
                p['question']
            )['answer'],
            p['expected']
        )
    )

    return {
        'seed': seed,
        'with_acc': with_correct / n_test,
        'without_acc': without_correct / n_test,
        'effect': (with_correct - without_correct) / n_test,
        'n_emergent': n_emergent,
        'phi': phi
    }


def run_parallel_scaling(scale, n_seeds=10, n_workers=None):
    """
    Run scaling study with parallel execution across seeds.

    Args:
        scale: Number of nodes in substrate
        n_seeds: Number of random seeds
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        Results dictionary
    """
    if n_workers is None:
        n_workers = min(cpu_count(), n_seeds)

    n_train = min(200, 50 + scale // 100)
    n_test = 30

    print(f'Scale: {scale:,} nodes')
    print(f'Seeds: {n_seeds}')
    print(f'Workers: {n_workers}')
    print(f'Training iterations: {n_train}')
    print()

    # Prepare arguments for each seed
    args_list = [(seed, scale, n_train, n_test) for seed in range(n_seeds)]

    start_time = time.time()

    # Run in parallel
    with Pool(n_workers) as pool:
        results_list = pool.map(run_single_seed, args_list)

    elapsed = time.time() - start_time

    # Aggregate results
    with_accs = [r['with_acc'] for r in results_list]
    without_accs = [r['without_acc'] for r in results_list]
    effects = [r['effect'] for r in results_list]
    emergent_counts = [r['n_emergent'] for r in results_list]
    phi_values = [r['phi'] for r in results_list]

    # Statistics
    mean_effect = np.mean(effects)
    std_effect = np.std(effects)
    if std_effect > 0:
        t_stat, p_value = stats.ttest_rel(with_accs, without_accs)
    else:
        t_stat, p_value = 0, 1.0

    results = {
        'scale': scale,
        'n_seeds': n_seeds,
        'n_workers': n_workers,
        'with_acc': float(np.mean(with_accs)),
        'without_acc': float(np.mean(without_accs)),
        'effect': float(mean_effect),
        'std': float(std_effect),
        'p_value': float(p_value),
        't_stat': float(t_stat),
        'emergent_mean': float(np.mean(emergent_counts)),
        'phi_mean': float(np.mean(phi_values)),
        'time_seconds': elapsed,
        'seed_results': results_list
    }

    # Print results
    print('Results:')
    for r in results_list:
        print(f"  Seed {r['seed']}: emergent={r['n_emergent']}, phi={r['phi']:.4f}, effect={r['effect']:+.0%}")

    print()
    print(f'WITH emergence:    {np.mean(with_accs):.1%}')
    print(f'WITHOUT emergence: {np.mean(without_accs):.1%}')
    print(f'Mean effect: {mean_effect:+.1%} (std={std_effect:.1%})')
    print(f'p-value: {p_value:.4f}')
    print(f'Time: {elapsed:.1f}s ({elapsed/60:.1f} min)')

    return results


def main():
    print('=' * 70)
    print('PARALLEL SCALING STUDY')
    print('=' * 70)
    print(f'Available CPUs: {cpu_count()}')

    # Parse command line args
    if len(sys.argv) > 1:
        scales = [int(s) for s in sys.argv[1:]]
    else:
        scales = [10000, 50000, 100000]

    all_results = {}

    for scale in scales:
        print()
        print('=' * 70)
        print(f'TESTING {scale:,} NODES')
        print('=' * 70)

        n_seeds = 5 if scale >= 50000 else 10
        results = run_parallel_scaling(scale, n_seeds=n_seeds)
        all_results[scale] = results

        if results['effect'] > 0 and results['p_value'] < 0.05:
            print(f'\nEMERGENCE EFFECT CONFIRMED at {scale:,} nodes')
        elif results['effect'] > 0:
            print(f'\nPositive effect at {scale:,} but needs more seeds for significance')

    # Summary
    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print('Scale      Effect     p-value    Time')
    print('-' * 45)
    for scale in sorted(all_results.keys()):
        r = all_results[scale]
        sig = '*' if r['p_value'] < 0.05 else ''
        time_str = f"{r['time_seconds']:.0f}s" if r['time_seconds'] < 3600 else f"{r['time_seconds']/3600:.1f}h"
        print(f"{scale:<10,} {r['effect']:+.1%}{sig:<4} {r['p_value']:<10.4f} {time_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/research")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"parallel_scaling_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2, default=str)

    print(f'\nResults saved to: {filepath}')


if __name__ == "__main__":
    main()
