"""trackD_parallel/exp1_submodular.py
-----------------------------------
Experiment 1: Submodular Optimization Benchmarks
- (a) Submodularity verification: Diminishing returns test on Facility Location vs Non-Submodular objective.
- (b) Exact comparison: Brute Force vs ILP (OR-Tools SCIP), verifying (1 - 1/e) greedy guarantee.
- (c) Scalability: Standard Greedy vs Lazy Greedy on n = 300, 1000, 3000 facilities (k = 20).
- (d) Non-submodular failure instance demonstration.

Saves:
  results/trackD_submodular.csv
  results/trackD_submodular_analysis.png
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackD_parallel.submodular import (
    brute_force,
    check_submodular,
    facility_location,
    greedy,
    ilp_facility_location,
    lazy_greedy,
    make_non_submodular_failure_instance,
)

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 1: Submodular Optimization Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run with reduced instances for quick sanity check")
    return parser.parse_args()


# ===========================================================================
# Part (a): Submodularity Verification
# ===========================================================================

def run_part_a(quick: bool = False) -> Tuple[int, int]:
    print("=" * 80)
    print("PART (A): Submodularity Verification via Diminishing Returns Checks")
    print("=" * 80, flush=True)

    n_instances = 5 if quick else 20
    trials_per_inst = 500 if quick else 2000

    print(f"Testing {n_instances} random Facility Location instances ({trials_per_inst} trials each)...", flush=True)
    fl_violations_total = 0
    rng = np.random.default_rng(100)

    for inst_idx in range(n_instances):
        m, n = 20, 20
        W = rng.uniform(0.0, 10.0, (m, n))
        f = facility_location(W)
        v = check_submodular(f, list(range(n)), trials=trials_per_inst, seed=inst_idx + 1000)
        fl_violations_total += v

    print(f"  Facility Location Total Violations: {fl_violations_total} (Expected: 0)")
    if fl_violations_total > 0:
        raise RuntimeError(f"Facility Location violated submodularity {fl_violations_total} times!")

    print("\nTesting Non-Submodular Objective with Complementarity Synergy...", flush=True)
    non_sub_obj, ground_set, _ = make_non_submodular_failure_instance()
    ns_violations = check_submodular(non_sub_obj, ground_set, trials=trials_per_inst, seed=999)
    print(f"  Non-Submodular Objective Violations: {ns_violations} (Expected: > 0)")
    if ns_violations == 0:
        raise RuntimeError("Non-submodular objective unexpectedly had 0 violations!")

    print("  [PASS] Part (a) verified: Facility location is monotone submodular; non-submodular objective exhibits synergy violations.", flush=True)
    return fl_violations_total, ns_violations


# ===========================================================================
# Part (b): Exact Comparison (Brute Force vs ILP, Greedy Approximation Ratio)
# ===========================================================================

def run_part_b(quick: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PART (B): Exact Comparison and (1 - 1/e) Greedy Approximation Ratio")
    print("=" * 80, flush=True)

    n_values = [16, 18]
    k_values = [3, 4] if quick else [3, 4, 5, 6]
    n_seeds = 5 if quick else 20
    bound = 1.0 - 1.0 / np.e  # ~0.63212

    records: List[Dict[str, Any]] = []

    print(f"Comparing Brute Force, ILP, Greedy, and Lazy Greedy across {len(n_values)*len(k_values)*n_seeds} instances...", flush=True)

    for n in n_values:
        for k in k_values:
            ratios: List[float] = []
            for seed in range(n_seeds):
                m = n  # square customer-facility matrix
                rng = np.random.default_rng(seed * 100 + n * 10 + k)
                W = rng.uniform(0.1, 5.0, (m, n))
                f = facility_location(W)

                # 1. Exact solvers
                bf_set, bf_val, bf_evals = brute_force(f, list(range(n)), k)
                ilp_set, ilp_val = ilp_facility_location(W, k)

                # Verification: Brute Force and ILP must agree
                diff = abs(bf_val - ilp_val)
                if diff > 1e-5:
                    raise RuntimeError(
                        f"Exact solver mismatch at n={n}, k={k}, seed={seed}: "
                        f"BF={bf_val:.6f} ({bf_set}), ILP={ilp_val:.6f} ({ilp_set}), diff={diff:.2e}"
                    )

                # 2. Heuristics
                g_set, g_val, g_evals = greedy(f, list(range(n)), k)
                lg_set, lg_val, lg_evals = lazy_greedy(f, list(range(n)), k)

                # Assert identical selected sets between greedy and lazy greedy
                if g_set != lg_set:
                    raise RuntimeError(f"Greedy {g_set} and Lazy Greedy {lg_set} produced different sets at n={n}, k={k}")

                ratio = g_val / bf_val
                ratios.append(ratio)

                if ratio < bound - 1e-6:
                    raise RuntimeError(
                        f"Greedy guarantee VIOLATED at n={n}, k={k}, seed={seed}: "
                        f"Ratio {ratio:.4f} < bound {bound:.4f}"
                    )

                records.append({
                    "n": n,
                    "k": k,
                    "seed": seed,
                    "opt_val": bf_val,
                    "greedy_val": g_val,
                    "ratio": ratio,
                    "bf_evals": bf_evals,
                    "greedy_evals": g_evals,
                    "lazy_evals": lg_evals,
                })

            mean_r = float(np.mean(ratios))
            min_r = float(np.min(ratios))
            std_r = float(np.std(ratios))
            print(
                f"  n={n:2d}, k={k}: Greedy/Opt Ratio Mean={mean_r:.4f}, Min={min_r:.4f}, Std={std_r:.4f} "
                f"(Bound = {bound:.4f})",
                flush=True,
            )

    df_b = pd.DataFrame(records)
    overall_min = df_b["ratio"].min()
    overall_mean = df_b["ratio"].mean()
    print(f"\n  Overall Greedy Ratio: Mean = {overall_mean:.4f}, Min = {overall_min:.4f} (>= bound {bound:.4f})")
    print("  Observation: Measured greedy ratios (typically 0.88 - 0.99) are significantly higher than the worst-case (1 - 1/e) guarantee.")
    return df_b


# ===========================================================================
# Part (c): Lazy Greedy vs Standard Greedy Scalability
# ===========================================================================

def run_part_c(quick: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PART (C): Lazy Greedy vs Standard Greedy Scalability (k = 20)")
    print("=" * 80, flush=True)

    n_sizes = [300, 1000] if quick else [300, 1000, 3000]
    k = 20
    m = 100  # 100 customers

    records: List[Dict[str, Any]] = []

    for n in n_sizes:
        rng = np.random.default_rng(42)
        W = rng.uniform(0.1, 5.0, (m, n))
        f = facility_location(W)
        ground_set = list(range(n))

        # Standard greedy
        f.reset_eval_count()
        t0 = time.perf_counter()
        g_set, g_val, g_evals = greedy(f, ground_set, k)
        dt_g = time.perf_counter() - t0

        # Lazy greedy
        f.reset_eval_count()
        t0 = time.perf_counter()
        lg_set, lg_val, lg_evals = lazy_greedy(f, ground_set, k)
        dt_lg = time.perf_counter() - t0

        # Verification: Sets must be identical
        assert g_set == lg_set, f"Selected sets differ for n={n}: {g_set} vs {lg_set}"
        assert abs(g_val - lg_val) < 1e-9

        speedup = dt_g / max(1e-9, dt_lg)
        eval_reduction = 100.0 * (1.0 - lg_evals / g_evals)

        print(
            f"  n={n:4d}, k={k:2d}: Sets identical [PASS] | "
            f"Evals: Greedy={g_evals:6d}, Lazy={lg_evals:6d} (-{eval_reduction:.1f}%) | "
            f"Time: Greedy={dt_g:.3f}s, Lazy={dt_lg:.3f}s | Speedup={speedup:.2f}x",
            flush=True,
        )

        records.append({
            "n": n,
            "k": k,
            "greedy_evals": g_evals,
            "lazy_evals": lg_evals,
            "greedy_time_s": dt_g,
            "lazy_time_s": dt_lg,
            "speedup": speedup,
            "eval_reduction_pct": eval_reduction,
        })

    return pd.DataFrame(records)


# ===========================================================================
# Part (d): Non-Submodular Failure Demonstration
# ===========================================================================

def run_part_d() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PART (D): Greedy Failure on Non-Submodular Complementarity Objective")
    print("=" * 80, flush=True)

    non_sub_obj, ground_set, k = make_non_submodular_failure_instance()
    bound = 1.0 - 1.0 / np.e

    g_set, g_val, g_evals = greedy(non_sub_obj, ground_set, k)
    bf_set, bf_val, bf_evals = brute_force(non_sub_obj, ground_set, k)

    ratio = g_val / bf_val

    print(f"  Ground Set: {ground_set}, Budget k={k}")
    print(f"  Base weights: {list(non_sub_obj.base_weights)}, Pair synergy bonus B={non_sub_obj.bonus} on {non_sub_obj.pair}")
    print(f"  Greedy selection: {g_set} -> Objective Value = {g_val:.2f}")
    print(f"  Optimal selection: {bf_set} -> Objective Value = {bf_val:.2f}")
    print(f"  Approximation Ratio: {ratio:.4f} (vs 1 - 1/e bound = {bound:.4f})")
    print(f"  Ratio < (1 - 1/e): {ratio < bound}")
    print(
        "  Explanation: Greedy falls for individual local gains (12.0 each for items 2 & 3) and misses the\n"
        "  synergistic pair {0, 1} (value 10 + 10 + 30 = 50.0). The (1 - 1/e) guarantee does NOT apply\n"
        "  because the objective exhibits complementarity and violates diminishing returns.",
        flush=True,
    )

    return {
        "greedy_set": g_set,
        "greedy_val": g_val,
        "opt_set": bf_set,
        "opt_val": bf_val,
        "ratio": ratio,
        "bound": bound,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def generate_submodular_plots(df_b: pd.DataFrame, df_c: pd.DataFrame) -> None:
    print("\nGenerating submodular performance plots...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Greedy Approximation Ratio Distribution
    ratios = df_b["ratio"].values
    bound = 1.0 - 1.0 / np.e

    ax1.hist(ratios, bins=15, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    ax1.axvline(bound, color="red", linestyle="--", linewidth=2, label=f"Theoretical Bound (1 - 1/e ≈ {bound:.3f})")
    ax1.axvline(np.mean(ratios), color="blue", linestyle="-", linewidth=2, label=f"Measured Mean ({np.mean(ratios):.3f})")
    ax1.set_xlabel("Greedy Approximation Ratio (Greedy / Optimum)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Greedy Ratio Distribution on Facility Location", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Evaluations: Greedy vs Lazy Greedy scaling
    ax2.plot(df_c["n"], df_c["greedy_evals"], "o-", color="#d62728", linewidth=2, markersize=6, label="Standard Greedy ($O(kn)$)")
    ax2.plot(df_c["n"], df_c["lazy_evals"], "s-", color="#1f77b4", linewidth=2, markersize=6, label="Lazy Greedy (Heap of Stale Bounds)")
    ax2.set_xlabel("Number of Facilities ($n$)", fontsize=11)
    ax2.set_ylabel("Objective Evaluations ($k=20$)", fontsize=11)
    ax2.set_title("Objective Evaluations: Greedy vs Lazy Greedy", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "trackD_submodular_analysis.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}", flush=True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    print("=" * 80)
    print("TRACK D: EXPERIMENT 1 - SUBMODULAR OPTIMIZATION")
    if args.quick:
        print("MODE: --quick")
    print("=" * 80, flush=True)

    # Part (a)
    run_part_a(quick=args.quick)

    # Part (b)
    df_b = run_part_b(quick=args.quick)

    # Part (c)
    df_c = run_part_c(quick=args.quick)

    # Part (d)
    part_d_res = run_part_d()

    # Save CSV
    csv_path = RESULTS_DIR / "trackD_submodular.csv"
    df_b.to_csv(csv_path, index=False)
    print(f"\nSaved submodular benchmark records to {csv_path}", flush=True)

    # Generate Plots
    generate_submodular_plots(df_b, df_c)
    print("\nExperiment 1 completed successfully.", flush=True)


if __name__ == "__main__":
    main()
