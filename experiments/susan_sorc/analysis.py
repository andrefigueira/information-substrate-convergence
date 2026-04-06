"""
Statistical analysis for exp_007.

Loads results JSON and reports:
  1. Descriptive stats: SORC mean ± SD per condition
  2. Primary test: Mann-Whitney U, self_referential vs control (one-tailed)
  3. All pairwise Mann-Whitney U comparisons
  4. Effect sizes: Cliff's delta for each pair
  5. Length covariate: Spearman correlation between n_tokens and SORC
  6. Linear regression: SORC ~ condition + n_tokens
  7. Within-condition SORC trajectory across turns (mean per turn)

Usage:
    python -m experiments.susan_sorc.analysis --input results/exp_007_results.json
    python -m experiments.susan_sorc.analysis --input results/exp_007_results.json --plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

CONDITION_ORDER = ["control", "history_only", "feedback_blind", "self_referential"]


def load_results(path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text())
    return raw, raw["results"]


def group_by_condition(results: list[dict]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in results:
        groups[r["condition"]].append(r["sorc"])
    return dict(groups)


def cliff_delta(x: list[float], y: list[float]) -> float:
    """
    Cliff's delta: proportion of pairs where x > y minus proportion where y > x.
    Range [-1, 1]. Positive means x tends to be larger.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    greater = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (greater - less) / (nx * ny)


def interpret_delta(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def spearman_length_sorc(results: list[dict]) -> dict[str, float]:
    """Spearman correlation between n_tokens and SORC, per condition."""
    by_condition: dict[str, list] = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append((r["n_tokens"], r["sorc"]))

    out = {}
    for cond, pairs in by_condition.items():
        tokens = [p[0] for p in pairs]
        sorc = [p[1] for p in pairs]
        rho, pval = stats.spearmanr(tokens, sorc)
        out[cond] = {"rho": round(float(rho), 3), "p": round(float(pval), 4)}
    return out


def regression_sorc_condition_tokens(results: list[dict]) -> dict:
    """
    OLS: SORC ~ condition_dummies + n_tokens.
    Condition dummies: control=0, history_only=1, feedback_blind=2, self_referential=3.
    Returns coefficients and R².
    """
    condition_to_int = {c: i for i, c in enumerate(CONDITION_ORDER)}
    X_raw = np.array(
        [[condition_to_int.get(r["condition"], 0), r["n_tokens"]] for r in results],
        dtype=float,
    )
    y = np.array([r["sorc"] for r in results])

    # Add intercept
    X = np.column_stack([np.ones(len(X_raw)), X_raw])
    # OLS: (X'X)^-1 X'y
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            "intercept": round(float(coeffs[0]), 4),
            "beta_condition": round(float(coeffs[1]), 4),
            "beta_n_tokens": round(float(coeffs[2]), 6),
            "r_squared": round(r2, 4),
            "note": (
                "beta_condition: SORC change per step up in condition order "
                "(control→history→feedback→self_ref). "
                "beta_n_tokens: SORC change per additional context token."
            ),
        }
    except np.linalg.LinAlgError:
        return {"error": "OLS failed"}


def turn_trajectory(results: list[dict]) -> dict[str, list[float]]:
    """Mean SORC per turn index, per condition."""
    by_cond_turn: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_cond_turn[r["condition"]][r["turn"]].append(r["sorc"])

    out = {}
    for cond in CONDITION_ORDER:
        if cond not in by_cond_turn:
            continue
        turns = sorted(by_cond_turn[cond].keys())
        out[cond] = [round(float(np.mean(by_cond_turn[cond][t])), 3) for t in turns]
    return out


def print_report(path: Path) -> None:
    meta, results = load_results(path)
    groups = group_by_condition(results)

    print(f"\n{'=' * 70}")
    print("exp_007 — SORC across SUSAN conditions")
    print(f"Model: {meta.get('model', 'unknown')}  Device: {meta.get('device', 'unknown')}")
    print(f"Random baseline SORC: {meta.get('random_baseline_sorc', 'N/A'):.3f}")
    print(f"Total measurements: {meta.get('n_results', len(results))}")
    print(f"{'=' * 70}")

    # --- 1. Descriptive stats ---
    print("\n1. DESCRIPTIVE STATS (SORC per condition)")
    print(f"  {'Condition':<22} {'N':>4}  {'Mean':>6}  {'SD':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-' * 60}")
    for cond in CONDITION_ORDER:
        vals = groups.get(cond, [])
        if not vals:
            continue
        print(
            f"  {cond:<22} {len(vals):>4}  {np.mean(vals):>6.3f}  "
            f"{np.std(vals):>6.3f}  {np.min(vals):>6.3f}  {np.max(vals):>6.3f}"
        )

    # --- 2. Primary test ---
    print("\n2. PRIMARY TEST: self_referential vs control (one-tailed, H1: SR > C)")
    sr = groups.get("self_referential", [])
    ctrl = groups.get("control", [])
    if sr and ctrl:
        u, p_two = stats.mannwhitneyu(sr, ctrl, alternative="greater")
        d = cliff_delta(sr, ctrl)
        print(f"  Mann-Whitney U = {u:.0f},  p (one-tailed) = {p_two:.4f}")
        print(f"  Cliff's delta = {d:+.3f} ({interpret_delta(d)})")
        sig = "SIGNIFICANT (p < 0.05)" if p_two < 0.05 else "NOT significant"
        direction = (
            "self_referential > control"
            if np.mean(sr) > np.mean(ctrl)
            else "self_referential <= control"
        )
        print(f"  {sig} — direction: {direction}")

    # --- 3. Clean architecture test ---
    print("\n3. CLEAN ARCHITECTURE TEST: self_referential vs feedback_blind")
    print("   (Same feedback loop, same history — only difference: model sees its own metrics)")
    fb = groups.get("feedback_blind", [])
    if sr and fb:
        u, p = stats.mannwhitneyu(sr, fb, alternative="greater")
        d = cliff_delta(sr, fb)
        print(f"  Mann-Whitney U = {u:.0f},  p = {p:.4f}")
        print(f"  Cliff's delta = {d:+.3f} ({interpret_delta(d)})")

    # --- 4. All pairwise comparisons ---
    print("\n4. ALL PAIRWISE COMPARISONS (two-tailed)")
    print(f"  {'Pair':<40} {'U':>8}  {'p':>7}  {'delta':>7}  {'effect'}")
    print(f"  {'-' * 75}")
    conds_with_data = [c for c in CONDITION_ORDER if groups.get(c)]
    for a, b in combinations(conds_with_data, 2):
        xa, xb = groups[a], groups[b]
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        d = cliff_delta(xa, xb)
        pair_label = f"{a} vs {b}"
        print(f"  {pair_label:<40} {u:>8.0f}  {p:>7.4f}  {d:>+7.3f}  {interpret_delta(d)}")

    # --- 5. Length covariate ---
    print("\n5. LENGTH COVARIATE: Spearman rho(n_tokens, SORC) per condition")
    length_corr = spearman_length_sorc(results)
    for cond in CONDITION_ORDER:
        if cond in length_corr:
            info = length_corr[cond]
            print(f"  {cond:<22}  rho = {info['rho']:+.3f}  p = {info['p']:.4f}")

    # --- 6. Regression ---
    print("\n6. OLS REGRESSION: SORC ~ condition_order + n_tokens")
    reg = regression_sorc_condition_tokens(results)
    if "error" not in reg:
        print(f"  Intercept:        {reg['intercept']:.4f}")
        print(f"  beta_condition:   {reg['beta_condition']:+.4f}  (per step: ctrl→hist→fb→sr)")
        print(f"  beta_n_tokens:    {reg['beta_n_tokens']:+.6f}  (per token)")
        print(f"  R²:               {reg['r_squared']:.4f}")
        print(f"  Note: {reg['note']}")

    # --- 7. Turn trajectory ---
    print("\n7. SORC TRAJECTORY ACROSS TURNS (mean per turn)")
    trajectory = turn_trajectory(results)
    n_turns = max(len(v) for v in trajectory.values()) if trajectory else 0
    header = f"  {'Turn':<5}" + "".join(
        f"  {c[:12]:<14}" for c in CONDITION_ORDER if c in trajectory
    )
    print(header)
    for t in range(n_turns):
        row = f"  {t:<5}"
        for cond in CONDITION_ORDER:
            if cond in trajectory and t < len(trajectory[cond]):
                row += f"  {trajectory[cond][t]:<14.3f}"
        print(row)

    print(f"\n{'=' * 70}")
    print("Interpretation guide:")
    print(
        "  Cliff's delta: |d| < 0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium, >0.474 large"
    )
    print("  SESOI (pre-registered): Cliff's delta >= 0.33 (medium effect)")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse exp_007 results")
    parser.add_argument(
        "--input",
        default="results/exp_007_results.json",
        help="Path to results JSON",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Results file not found: {path}")
        print("Run the experiment first: python -m experiments.susan_sorc.run")
        raise SystemExit(1)

    print_report(path)


if __name__ == "__main__":
    main()
