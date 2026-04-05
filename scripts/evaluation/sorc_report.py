#!/usr/bin/env python3
"""
SORC Experiment Report Generator
==================================

Reads a results JSON file produced by run_sorc_experiment.py and prints a
clean, publication-ready summary including:

  - Per-category statistics (mean, std, min, max, mean token count)
  - Prediction evaluation (degenerate < shallow < expert < multi_turn)
  - Layer-depth profile comparison (early vs late layer scores)
  - Pairwise significance notes (effect size, not statistical test)

Usage
-----
    python scripts/evaluation/sorc_report.py results/sorc/exp_004/results.json
    python scripts/evaluation/sorc_report.py results/sorc/exp_004/results.json --latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CATEGORY_ORDER = ["degenerate", "shallow", "expert_short", "expert", "multi_turn", "multi_turn_unmarked"]
CATEGORY_LABELS = {
    "degenerate":          "Degenerate",
    "shallow":             "Shallow",
    "expert_short":        "Expert (short)",
    "expert":              "Expert (long)",
    "multi_turn":          "Multi-turn (marked)",
    "multi_turn_unmarked": "Multi-turn (unmarked)",
}

PREDICTION_PAIRS = [
    ("degenerate", "shallow",             "degenerate < shallow"),
    ("degenerate", "expert_short",        "degenerate < expert_short"),
    ("degenerate", "expert",              "degenerate < expert"),
    ("degenerate", "multi_turn",          "degenerate < multi_turn"),
    ("degenerate", "multi_turn_unmarked", "degenerate < multi_turn_unmarked"),
    ("shallow",    "expert_short",        "shallow ≈ expert_short (length-matched)"),
    ("expert",     "multi_turn",          "expert < multi_turn  [exp_004 ISC prediction]"),
    ("expert",     "multi_turn_unmarked", "expert < multi_turn_unmarked  [exp_005 KEY TEST]"),
    ("multi_turn_unmarked", "multi_turn", "unmarked ≈ marked  [marker isolation check]"),
    ("shallow",    "multi_turn",          "shallow < multi_turn"),
]


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def category_stats(cat_results: list[dict]) -> dict:
    scores = [r["sorc"] for r in cat_results]
    tokens = [r["n_tokens"] for r in cat_results]
    layers = [r["layer_scores"] for r in cat_results]

    # Split layer scores into early half and late half
    n_layers = len(layers[0])
    half = n_layers // 2
    early = [np.mean(ls[:half]) for ls in layers]
    late  = [np.mean(ls[half:]) for ls in layers]

    return {
        "n": len(scores),
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "min":  float(np.min(scores)),
        "max":  float(np.max(scores)),
        "mean_tokens": float(np.mean(tokens)),
        "early_layer_mean": float(np.mean(early)),
        "late_layer_mean":  float(np.mean(late)),
        "late_early_ratio": float(np.mean(late) / max(np.mean(early), 1e-8)),
    }


def cohens_d(a_scores: list[float], b_scores: list[float]) -> float:
    """Pooled Cohen's d effect size."""
    na, nb = len(a_scores), len(b_scores)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a_scores, ddof=1) + (nb - 1) * np.var(b_scores, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std < 1e-10:
        return float("inf")
    return float((np.mean(b_scores) - np.mean(a_scores)) / pooled_std)


def print_report(data: dict, latex: bool = False) -> None:
    model    = data.get("model", "unknown")
    baseline = data.get("random_baseline", float("nan"))
    results  = data.get("results", {})

    present = [c for c in CATEGORY_ORDER if c in results and results[c]]
    stats   = {c: category_stats(results[c]) for c in present}

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  SORC EXPERIMENT REPORT")
    print("=" * 70)
    print(f"  Model:           {model}")
    print(f"  Random baseline: {baseline:.4f}")
    print(f"  Categories:      {', '.join(present)}")
    print()

    # -------------------------------------------------------------------------
    # Per-category table
    # -------------------------------------------------------------------------
    col_w = [18, 7, 7, 7, 7, 8, 8, 8, 9]
    headers = ["Category", "Mean", "Std", "Min", "Max", "Tokens", "Early L", "Late L", "Late/Early"]
    row_fmt = "{:<18} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>9}"

    print(row_fmt.format(*headers))
    print("-" * 73)

    for cat in present:
        s = stats[cat]
        print(row_fmt.format(
            CATEGORY_LABELS.get(cat, cat),
            f"{s['mean']:.4f}",
            f"{s['std']:.4f}",
            f"{s['min']:.4f}",
            f"{s['max']:.4f}",
            f"{s['mean_tokens']:.0f}",
            f"{s['early_layer_mean']:.4f}",
            f"{s['late_layer_mean']:.4f}",
            f"{s['late_early_ratio']:.3f}",
        ))

    print()

    # -------------------------------------------------------------------------
    # Prediction evaluation
    # -------------------------------------------------------------------------
    print("PREDICTION EVALUATION")
    print("-" * 70)

    for (a, b, label) in PREDICTION_PAIRS:
        if a not in stats or b not in stats:
            continue
        ma, mb = stats[a]["mean"], stats[b]["mean"]
        a_scores = [r["sorc"] for r in results[a]]
        b_scores = [r["sorc"] for r in results[b]]
        d = cohens_d(a_scores, b_scores)
        direction = ">" if mb > ma else "<" if mb < ma else "="
        supported = mb > ma
        flag = "SUPPORTED    " if supported else "NOT SUPPORTED"
        tok_a = stats[a]["mean_tokens"]
        tok_b = stats[b]["mean_tokens"]
        tok_note = f"  [{tok_a:.0f} vs {tok_b:.0f} tok]"
        d_note = f"  d={d:.2f}" if not np.isnan(d) else ""
        print(f"  {flag}  {label}{tok_note}{d_note}")
        print(f"             {CATEGORY_LABELS.get(a, a)}: {ma:.4f}  {direction}  {CATEGORY_LABELS.get(b, b)}: {mb:.4f}")
        print()

    # -------------------------------------------------------------------------
    # Layer depth profile narrative
    # -------------------------------------------------------------------------
    print("LAYER DEPTH PROFILE")
    print("-" * 70)
    print("  Late/Early ratio: fraction of early-layer SORC retained at deep layers.")
    print("  Low ratio = semantic clustering at deep layers (coherent domain text).")
    print("  High ratio = distributed structure maintained through depth.")
    print()
    for cat in present:
        s = stats[cat]
        ratio = s["late_early_ratio"]
        note = "deep collapse" if ratio < 0.6 else "moderate decay" if ratio < 0.85 else "stable"
        print(f"  {CATEGORY_LABELS.get(cat, cat):<20} {ratio:.3f}  ({note})")
    print()

    # -------------------------------------------------------------------------
    # Key ISC prediction summary
    # -------------------------------------------------------------------------
    if "expert" in stats:
        ex = stats["expert"]["mean"]
        ex_tok = stats["expert"]["mean_tokens"]
        print("KEY ISC PREDICTIONS")
        print("-" * 70)

        for cat, label in [("multi_turn", "exp_004 (marked)"), ("multi_turn_unmarked", "exp_005 (unmarked)")]:
            if cat not in stats:
                continue
            mt = stats[cat]["mean"]
            mt_tok = stats[cat]["mean_tokens"]
            if mt > ex:
                pct = (mt - ex) / ex * 100
                verdict = "SUPPORTED    "
                arrow = ">"
                sign = f"+{pct:.1f}%"
            else:
                pct = (ex - mt) / ex * 100
                verdict = "NOT SUPPORTED"
                arrow = "<"
                sign = f"-{pct:.1f}%"
            print(f"  {verdict}  {label}: multi_turn ({mt_tok:.0f} tok) {mt:.4f}  {arrow}  expert ({ex_tok:.0f} tok) {ex:.4f}  ({sign})")

        if "multi_turn" in stats and "multi_turn_unmarked" in stats:
            mt_m = stats["multi_turn"]["mean"]
            mt_u = stats["multi_turn_unmarked"]["mean"]
            diff_pct = (mt_m - mt_u) / mt_m * 100 if mt_m > 0 else 0
            print(f"\n  MARKER EFFECT: marked={mt_m:.4f}  unmarked={mt_u:.4f}  diff={diff_pct:.1f}%")
            # Also compute marker contribution as fraction of the expert-to-multi_turn gap
            if "expert" in stats and mt_m > 0:
                ex_mean = stats["expert"]["mean"]
                gap_total = mt_m - ex_mean
                gap_marker = mt_m - mt_u
                marker_pct_of_gap = (gap_marker / gap_total * 100) if gap_total > 1e-10 else float("nan")
                print(f"  Marker contribution to expert→multi_turn gap: {marker_pct_of_gap:.0f}%")
            if abs(diff_pct) < 5:
                print("  Marker effect <5% of marked SORC score. Conversational structure is the driver.")
            elif diff_pct > 0:
                print(f"  Markers amplify the effect ({diff_pct:.1f}% of marked score).")
            else:
                print("  Unmarked scores higher than marked. Markers may suppress structure.")
        print()

    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SORC experiment report")
    parser.add_argument("results_file", type=Path, help="Path to results JSON")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table (stub)")
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Error: {args.results_file} not found", file=sys.stderr)
        sys.exit(1)

    data = load_results(args.results_file)
    print_report(data, latex=args.latex)


if __name__ == "__main__":
    main()
