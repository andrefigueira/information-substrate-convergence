"""
Visualization Script for ISC Experiment Results

Generates charts and figures from experiment JSON files.

Usage:
    python -m experiments.visualize_results results/experiments/exp_XXXXX
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Text-based visualization only.")


def load_experiment(exp_dir: str) -> Dict[str, Any]:
    """Load experiment data from directory"""
    exp_path = Path(exp_dir)

    with open(exp_path / "summary.json") as f:
        summary = json.load(f)

    with open(exp_path / "trials.json") as f:
        trials = json.load(f)

    return {
        "summary": summary,
        "trials": trials,
        "exp_dir": exp_path
    }


def generate_text_report(data: Dict[str, Any]) -> str:
    """Generate a text-based visualization"""
    summary = data["summary"]
    trials = data["trials"]

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("ISC EXPERIMENT VISUALIZATION (Text Mode)")
    lines.append("=" * 70)

    # Accuracy bar chart (ASCII)
    lines.append("\n--- ACCURACY BY REASONING TYPE ---\n")
    acc_by_type = summary["summary"]["accuracy_by_type"]
    max_bar = 40

    for ptype, stats in sorted(acc_by_type.items(), key=lambda x: -x[1]["accuracy"]):
        acc = stats["accuracy"]
        bar_len = int(acc * max_bar)
        bar = "#" * bar_len + "." * (max_bar - bar_len)
        lines.append(f"{ptype:15s} |{bar}| {acc*100:5.1f}%")

    # Phi correlation chart (ASCII)
    lines.append("\n--- PHI-ACCURACY CORRELATIONS ---\n")
    phi_corrs = summary["phi_correlations"]

    for method, corr_data in sorted(phi_corrs.items(), key=lambda x: -abs(x[1]["correlation"])):
        corr = corr_data["correlation"]
        p_val = corr_data["p_value"]
        sig = "*" if p_val < 0.05 else " "

        # Bar from -1 to +1
        center = 20
        bar_len = int(abs(corr) * 20)

        if corr >= 0:
            bar = " " * center + "+" * bar_len + " " * (20 - bar_len)
        else:
            bar = " " * (center - bar_len) + "-" * bar_len + " " * 20

        lines.append(f"{method:20s} |{bar}| r={corr:+.3f}{sig}")

    lines.append("\n(* = p < 0.05)")

    # Phi values over trials
    lines.append("\n--- PHI VALUES OVER TIME ---\n")

    # Sample trials at intervals
    n_trials = len(trials)
    sample_points = min(10, n_trials)
    interval = max(1, n_trials // sample_points)

    for i in range(0, n_trials, interval):
        trial = trials[i]
        phi_simple = trial["phi_values"]["simple"]
        phi_bar_len = int(phi_simple * 30)
        phi_bar = "#" * phi_bar_len

        correct_marker = "OK" if trial["is_correct"] else "X"
        lines.append(f"Trial {i:4d}: |{phi_bar:30s}| phi={phi_simple:.3f} [{correct_marker}]")

    # Thesis validation summary
    lines.append("\n--- ISC THESIS VALIDATION ---\n")
    tv = summary["thesis_validation"]

    criteria = [
        ("Phi predicts accuracy", tv["phi_predicts_accuracy"]),
        ("System learns over time", tv["system_learns_over_time"]),
        ("Emergence occurs", tv["emergence_occurs"]),
        ("Above random performance", tv["above_random_performance"]),
        ("Integration helps accuracy", tv["integration_improves_accuracy"]),
    ]

    for name, passed in criteria:
        status = "[PASS]" if passed else "[FAIL]"
        lines.append(f"  {status} {name}")

    lines.append(f"\n  TOTAL: {tv['criteria_met']}/{tv['total_criteria']} criteria met")

    # Key finding
    lines.append("\n" + "=" * 70)
    lines.append("KEY FINDING")
    lines.append("=" * 70)

    best_predictor = summary["best_phi_predictor"]
    best_corr = phi_corrs[best_predictor]["correlation"]
    best_p = phi_corrs[best_predictor]["p_value"]

    if best_p < 0.05 and best_corr > 0:
        lines.append(f"\nStatistically significant positive correlation found!")
        lines.append(f"Best predictor: {best_predictor}")
        lines.append(f"Correlation: r = {best_corr:.3f} (p = {best_p:.6f})")
        lines.append(f"\nThis supports the ISC thesis: integrated information")
        lines.append(f"is associated with better reasoning performance.")
    elif best_p < 0.05:
        lines.append(f"\nStatistically significant correlation found (but negative)")
        lines.append(f"This is unexpected and warrants further investigation.")
    else:
        lines.append(f"\nNo statistically significant correlation found.")
        lines.append(f"More data or different phi measures may be needed.")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def generate_matplotlib_figures(data: Dict[str, Any], output_dir: Path):
    """Generate matplotlib figures"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping figure generation")
        return

    summary = data["summary"]
    trials = data["trials"]

    # Figure 1: Accuracy by reasoning type
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    acc_by_type = summary["summary"]["accuracy_by_type"]
    types = list(acc_by_type.keys())
    accuracies = [acc_by_type[t]["accuracy"] * 100 for t in types]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(types)))
    bars = ax1.bar(types, accuracies, color=colors)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Reasoning Type")
    ax1.set_title("Accuracy by Reasoning Type")
    ax1.axhline(y=50, color='r', linestyle='--', label='Random baseline')
    ax1.set_ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig1.savefig(output_dir / "accuracy_by_type.png", dpi=150)
    plt.close(fig1)

    # Figure 2: Phi correlations
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    phi_corrs = summary["phi_correlations"]
    methods = list(phi_corrs.keys())
    correlations = [phi_corrs[m]["correlation"] for m in methods]
    p_values = [phi_corrs[m]["p_value"] for m in methods]

    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    bars = ax2.barh(methods, correlations, color=colors)

    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel("Correlation with Accuracy (r)")
    ax2.set_title("Phi Measures: Correlation with Reasoning Accuracy")

    # Add significance markers
    for i, (method, corr, p) in enumerate(zip(methods, correlations, p_values)):
        if p < 0.05:
            ax2.text(corr + 0.01 if corr > 0 else corr - 0.01,
                    i, '*', fontsize=14, va='center',
                    ha='left' if corr > 0 else 'right')

    green_patch = mpatches.Patch(color='green', label='p < 0.05 (significant)')
    gray_patch = mpatches.Patch(color='gray', label='p >= 0.05 (not significant)')
    ax2.legend(handles=[green_patch, gray_patch], loc='lower right')

    plt.tight_layout()
    fig2.savefig(output_dir / "phi_correlations.png", dpi=150)
    plt.close(fig2)

    # Figure 3: Phi values over time
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Extract phi values and correctness
    trial_nums = list(range(len(trials)))
    phi_simple = [t["phi_values"]["simple"] for t in trials]
    is_correct = [t["is_correct"] for t in trials]

    # Smooth phi with rolling average
    window = min(50, len(phi_simple) // 10) or 1
    phi_smooth = np.convolve(phi_simple, np.ones(window)/window, mode='valid')
    trial_smooth = trial_nums[:len(phi_smooth)]

    ax3.plot(trial_smooth, phi_smooth, 'b-', linewidth=2, label='Phi (smoothed)')
    ax3.fill_between(trial_smooth, 0, phi_smooth, alpha=0.3)

    # Mark correct answers
    correct_trials = [i for i, c in enumerate(is_correct) if c]
    correct_phis = [phi_simple[i] for i in correct_trials]
    ax3.scatter(correct_trials, correct_phis, c='green', alpha=0.5, s=20, label='Correct', zorder=5)

    ax3.set_xlabel("Trial Number")
    ax3.set_ylabel("Phi Value")
    ax3.set_title("Integrated Information (Phi) Over Time")
    ax3.legend()
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    fig3.savefig(output_dir / "phi_over_time.png", dpi=150)
    plt.close(fig3)

    # Figure 4: Thesis validation
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    tv = summary["thesis_validation"]

    criteria_names = [
        "Phi predicts\naccuracy",
        "System learns\nover time",
        "Emergence\noccurs",
        "Above random\nperformance",
        "Integration helps\naccuracy"
    ]
    criteria_values = [
        tv["phi_predicts_accuracy"],
        tv["system_learns_over_time"],
        tv["emergence_occurs"],
        tv["above_random_performance"],
        tv["integration_improves_accuracy"]
    ]

    colors = ['green' if v else 'red' for v in criteria_values]
    bars = ax4.bar(criteria_names, [1]*len(criteria_values), color=colors)

    ax4.set_ylim(0, 1.2)
    ax4.set_ylabel("")
    ax4.set_title(f"ISC Thesis Validation: {tv['criteria_met']}/{tv['total_criteria']} Criteria Met")
    ax4.set_yticks([])

    for bar, passed in zip(bars, criteria_values):
        label = "PASS" if passed else "FAIL"
        ax4.text(bar.get_x() + bar.get_width()/2, 0.5, label,
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white')

    plt.tight_layout()
    fig4.savefig(output_dir / "thesis_validation.png", dpi=150)
    plt.close(fig4)

    print(f"Figures saved to {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.visualize_results <experiment_dir>")
        print("Example: python -m experiments.visualize_results results/experiments/exp_20260106_125049")
        sys.exit(1)

    exp_dir = sys.argv[1]

    if not Path(exp_dir).exists():
        # Try to find the latest experiment
        results_dir = Path("results/experiments")
        if results_dir.exists():
            experiments = sorted(results_dir.iterdir(), key=lambda x: x.name, reverse=True)
            if experiments:
                exp_dir = str(experiments[0])
                print(f"Using latest experiment: {exp_dir}")
            else:
                print("No experiments found")
                sys.exit(1)
        else:
            print(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)

    data = load_experiment(exp_dir)

    # Generate text report
    text_report = generate_text_report(data)
    print(text_report)

    # Save text report
    report_path = data["exp_dir"] / "visualization.txt"
    with open(report_path, "w") as f:
        f.write(text_report)
    print(f"\nText report saved to: {report_path}")

    # Generate matplotlib figures if available
    if HAS_MATPLOTLIB:
        generate_matplotlib_figures(data, data["exp_dir"])


if __name__ == "__main__":
    main()
