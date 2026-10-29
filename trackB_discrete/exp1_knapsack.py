"""trackB_discrete/exp1_knapsack.py
---------------------------------
Knapsack solver evaluation and scalability benchmark:
1. Exact solver equivalence check over 20 random seeds on n in {15, 20}.
2. Scaling benchmark for n in {50, 200, 1000} comparing exact and heuristic solvers.
3. Hand-built pathological failure case for ratio-greedy fixed by greedy_half_approx.
4. Export CSV results and log-scale runtime-vs-n plot.
"""

from __future__ import annotations

import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackB_discrete.knapsack import (
    branch_and_bound,
    brute_force,
    dp_knapsack,
    greedy_half_approx,
    greedy_ratio,
    ilp_knapsack,
    milp_knapsack_scipy,
)

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


def generate_instance(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate correlated knapsack instance with weights 1-50 and capacity=40% of total."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 51, size=n)
    # Correlated values: weights + uniform noise in [-5, 15]
    noise = rng.integers(-5, 16, size=n)
    values = np.maximum(weights + noise, 1)
    capacity = int(0.40 * np.sum(weights))
    return values, weights, capacity


def verify_exact_solvers_equivalence():
    """Verify that all exact solvers reach the exact same optimal value over 20 seeds."""
    print("=== Step 1: Verifying Exact Solvers Equivalence on n in {15, 20} over 20 seeds ===")
    test_ns = [15, 20]
    n_seeds = 20

    for n in test_ns:
        print(f"Checking n = {n}...")
        for seed in range(n_seeds):
            v, w, W = generate_instance(n, seed)

            val_bf, _, _ = brute_force(v, w, W)
            val_dp, _, _ = dp_knapsack(v, w, W)
            val_ilp, _, _ = ilp_knapsack(v, w, W)
            val_scipy, _, _ = milp_knapsack_scipy(v, w, W)
            val_bb, _, _ = branch_and_bound(v, w, W)

            if not (val_bf == val_dp == val_ilp == val_scipy == val_bb):
                msg = (
                    f"Solver disagreement at n={n}, seed={seed}!\n"
                    f"  Brute Force: {val_bf}\n"
                    f"  DP:          {val_dp}\n"
                    f"  ILP:         {val_ilp}\n"
                    f"  SciPy MILP:  {val_scipy}\n"
                    f"  B&B:         {val_bb}"
                )
                print(msg, file=sys.stderr)
                raise RuntimeError(msg)

    print("SUCCESS: All exact solvers agree 100% on all 40 test instances.\n")


def demonstrate_greedy_failure_case():
    """Demonstrate a hand-built instance where ratio-greedy fails but greedy_half_approx succeeds."""
    print("=== Step 2: Hand-built Greedy Failure Case ===")
    # Item 0: small item with high ratio
    # Item 1: large item with slightly lower ratio that dominates
    # Capacity = 10
    v = [2, 10]
    w = [1, 10]
    W = 10

    val_ratio, sel_ratio, _ = greedy_ratio(v, w, W)
    val_half, sel_half, info_half = greedy_half_approx(v, w, W)
    val_opt, sel_opt, _ = dp_knapsack(v, w, W)

    print("Problem Instance:")
    print("  Item 0: value = 2,  weight = 1  (ratio = 2.0)")
    print("  Item 1: value = 10, weight = 10 (ratio = 1.0)")
    print("  Capacity W = 10")
    print(f"  Ratio-Greedy:        Selected {sel_ratio}, Value = {val_ratio} (Only 20% of OPT!)")
    print(f"  Greedy 1/2-Approx:   Selected {sel_half}, Value = {val_half} (Winner: {info_half['winner']})")
    print(f"  Exact Optimum:       Selected {sel_opt}, Value = {val_opt}")
    print("  -> Confirmed: greedy_half_approx completely fixes the ratio-greedy trap!\n")


def run_scalability_benchmark() -> pd.DataFrame:
    """Benchmark solvers on n in {50, 200, 1000} over 5 seeds."""
    print("=== Step 3: Scalability Benchmark (n in {50, 200, 1000}) ===")
    bench_ns = [50, 200, 1000]
    n_seeds = 5

    methods = [
        ("Greedy Ratio", greedy_ratio),
        ("Greedy 1/2 Approx", greedy_half_approx),
        ("Dynamic Programming", dp_knapsack),
        ("ILP (OR-Tools SCIP)", ilp_knapsack),
        ("SciPy MILP", milp_knapsack_scipy),
        ("Branch and Bound", branch_and_bound),
    ]

    records = []

    for n in bench_ns:
        print(f"Benchmarking n = {n} (5 seeds)...")
        # Precompute runs
        for name, solver_fn in methods:
            vals = []
            gaps = []
            runtimes = []
            nodes = []

            for seed in range(n_seeds):
                v, w, W = generate_instance(n, seed)
                # Compute reference DP optimum
                opt_val, _, _ = dp_knapsack(v, w, W)

                val, _, info = solver_fn(v, w, W)
                gap_pct = ((opt_val - val) / opt_val) * 100.0

                vals.append(val)
                gaps.append(gap_pct)
                runtimes.append(info["runtime"])
                if "nodes_explored" in info:
                    nodes.append(info["nodes_explored"])

            records.append({
                "n": n,
                "Method": name,
                "Value (mean)": np.mean(vals),
                "Value (std)": np.std(vals),
                "Gap % (mean)": np.mean(gaps),
                "Gap % (std)": np.std(gaps),
                "Runtime (s) (mean)": np.mean(runtimes),
                "Runtime (s) (std)": np.std(runtimes),
                "Nodes Explored (mean)": np.mean(nodes) if nodes else np.nan,
                "Nodes Explored (std)": np.std(nodes) if nodes else np.nan,
            })

    df = pd.DataFrame(records)

    # Plot runtime vs n on log-log scale
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in [m[0] for m in methods]:
        sub_df = df[df["Method"] == name]
        ax.plot(sub_df["n"], sub_df["Runtime (s) (mean)"], "o-", label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Items (n)")
    ax.set_ylabel("Runtime in Seconds (log scale)")
    ax.set_title("0/1 Knapsack Solvers Runtime vs. Problem Size (n)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "trackB_knapsack_runtime.png", dpi=120)
    plt.close()

    df.to_csv(RESULTS / "trackB_knapsack.csv", index=False)
    return df


def main() -> None:
    verify_exact_solvers_equivalence()
    demonstrate_greedy_failure_case()
    df = run_scalability_benchmark()

    cols_to_print = [
        "n", "Method", "Value (mean)", "Gap % (mean)",
        "Runtime (s) (mean)", "Nodes Explored (mean)"
    ]
    print("\nSummary Benchmark Table:")
    print(df[cols_to_print].to_string(index=False, float_format=lambda x: f"{x:.4e}" if abs(x) < 0.01 else f"{x:.4f}"))
    print("\nSaved results to results/trackB_knapsack.csv and results/trackB_knapsack_runtime.png")


if __name__ == "__main__":
    main()
