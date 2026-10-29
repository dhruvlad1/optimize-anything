"""run_comparison.py - Comprehensive comparison and scaling experiments for Track E CVRP.

Executes:
1. Exact & ILP Comparison (n in {8, 10, 12}, seeds 0-4):
   - Solves instances with exact MTZ ILP (60s limit).
   - Reports optimality gaps for driver_order, nearest_neighbour, clarke_wright,
     clarke_wright + local_search, and ortools_routing (10s).
   - Asserts that no heuristic beats proven-optimal ILP by > 1e-6.
2. Scaling Experiments (n in {20, 50, 100, 200}, seeds 0-4):
   - Compares all heuristics on the exact benchmark instances as trackE_baseline.csv.
   - Evaluates ILP on n = 20 (120s limit) to test exact scalability boundary.
   - Computes mean, std of distance, vehicle counts, runtimes, and % improvements.
3. Multi-Core Parallelism (n = 100, clarke_wright + local_search, seeds 0-4):
   - Measures wall-clock speedup and efficiency for sequential vs 2, 4, and max CPU workers.
   - Highlights embarrassing parallelism across instances vs single-solve thread scaling.
4. Generates Visualizations:
   - results/trackE_scaling.png: Distance and runtime scaling vs n.
   - results/trackE_routes_n50.png: Route map comparison (NN vs CW+LS) for n=50, seed 0.
   - results/trackE_comparison.csv: Full benchmark records.

Usage:
  python -m trackE_mumbai_routing.run_comparison
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackE_mumbai_routing.baselines import driver_order, nearest_neighbour
from trackE_mumbai_routing.routing_common import (
    DEPOT_COORDS,
    CVRPInstance,
    generate_instance,
    solution_cost,
    validate_solution,
)
from trackE_mumbai_routing.solvers import (
    clarke_wright,
    ilp_cvrp,
    local_search,
    ortools_routing,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Multiprocessing worker functions (must be top-level for pickling on Windows)
# ---------------------------------------------------------------------------

def _worker_cw_ls(n_seed: tuple[int, int]) -> tuple[int, float, int, float]:
    """Worker for parallelism test: runs CW + LS on given (n, seed)."""
    n, seed = n_seed
    inst = generate_instance(n, seed)
    t0 = time.perf_counter()
    cw_r = clarke_wright(inst)
    ls_r, _ = local_search(inst, cw_r)
    runtime = time.perf_counter() - t0
    cost = solution_cost(ls_r, inst.dist_matrix)
    return seed, cost, len(ls_r), runtime


def _worker_ilp_n20(seed: int) -> dict[str, Any]:
    """Worker for n=20 ILP with 120s limit."""
    inst = generate_instance(20, seed)
    t0 = time.perf_counter()
    res = ilp_cvrp(inst, time_limit_s=120.0)
    runtime = time.perf_counter() - t0
    valid, _ = validate_solution(inst, res.routes) if res.routes else (False, [])
    return {
        "method": "ilp_cvrp",
        "n": 20,
        "seed": seed,
        "distance_km": round(res.cost, 3),
        "vehicles": len(res.routes),
        "runtime_s": round(runtime, 4),
        "valid": valid,
        "proved_optimal": res.proved_optimal,
        "best_bound": round(res.best_bound, 3),
        "status": res.status,
    }


# ---------------------------------------------------------------------------
# Experiment 1: Exact ILP vs Heuristics (n in {8, 10, 12}, seeds 0-4)
# ---------------------------------------------------------------------------

def run_exact_comparison() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Exact ILP & Optimality Gap Comparison (n in {8, 10, 12})")
    print("=" * 80)

    n_values = [8, 10, 12]
    seeds = list(range(5))
    records: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for n in n_values:
        for seed in seeds:
            inst = generate_instance(n, seed)

            # 1. Exact ILP
            t0 = time.perf_counter()
            ilp_res = ilp_cvrp(inst, time_limit_s=60.0)
            t_ilp = time.perf_counter() - t0

            ilp_dist = ilp_res.cost
            ilp_opt = ilp_res.proved_optimal
            ilp_veh = len(ilp_res.routes)
            valid_ilp, _ = validate_solution(inst, ilp_res.routes)

            records.append({
                "method": "ilp_cvrp",
                "n": n,
                "seed": seed,
                "distance_km": round(ilp_dist, 3),
                "vehicles": ilp_veh,
                "runtime_s": round(t_ilp, 4),
                "valid": valid_ilp,
                "proved_optimal": ilp_opt,
            })

            # Evaluate heuristics on this instance
            # Driver order
            t0 = time.perf_counter()
            r_do = driver_order(inst)
            t_do = time.perf_counter() - t0
            c_do = solution_cost(r_do, inst.dist_matrix)
            records.append({
                "method": "driver_order",
                "n": n,
                "seed": seed,
                "distance_km": round(c_do, 3),
                "vehicles": len(r_do),
                "runtime_s": round(t_do, 6),
                "valid": validate_solution(inst, r_do)[0],
                "proved_optimal": False,
            })

            # Nearest neighbour
            t0 = time.perf_counter()
            r_nn = nearest_neighbour(inst)
            t_nn = time.perf_counter() - t0
            c_nn = solution_cost(r_nn, inst.dist_matrix)
            records.append({
                "method": "nearest_neighbour",
                "n": n,
                "seed": seed,
                "distance_km": round(c_nn, 3),
                "vehicles": len(r_nn),
                "runtime_s": round(t_nn, 6),
                "valid": validate_solution(inst, r_nn)[0],
                "proved_optimal": False,
            })

            # Clarke-Wright
            t0 = time.perf_counter()
            r_cw = clarke_wright(inst)
            t_cw = time.perf_counter() - t0
            c_cw = solution_cost(r_cw, inst.dist_matrix)
            records.append({
                "method": "clarke_wright",
                "n": n,
                "seed": seed,
                "distance_km": round(c_cw, 3),
                "vehicles": len(r_cw),
                "runtime_s": round(t_cw, 6),
                "valid": validate_solution(inst, r_cw)[0],
                "proved_optimal": False,
            })

            # Clarke-Wright + Local Search
            t0 = time.perf_counter()
            r_cw_init = clarke_wright(inst)
            r_ls, _ = local_search(inst, r_cw_init)
            t_ls = time.perf_counter() - t0
            c_ls = solution_cost(r_ls, inst.dist_matrix)
            records.append({
                "method": "clarke_wright_local_search",
                "n": n,
                "seed": seed,
                "distance_km": round(c_ls, 3),
                "vehicles": len(r_ls),
                "runtime_s": round(t_ls, 6),
                "valid": validate_solution(inst, r_ls)[0],
                "proved_optimal": False,
            })

            # OR-Tools Routing (10s)
            t0 = time.perf_counter()
            r_or, c_or = ortools_routing(inst, time_limit_s=10.0)
            t_or = time.perf_counter() - t0
            records.append({
                "method": "ortools_routing",
                "n": n,
                "seed": seed,
                "distance_km": round(c_or, 3),
                "vehicles": len(r_or),
                "runtime_s": round(t_or, 4),
                "valid": validate_solution(inst, r_or)[0],
                "proved_optimal": False,
            })

            # Check if any heuristic beat proven optimal ILP
            if ilp_opt:
                heur_costs = {
                    "driver_order": c_do,
                    "nearest_neighbour": c_nn,
                    "clarke_wright": c_cw,
                    "clarke_wright_local_search": c_ls,
                    "ortools_routing": c_or,
                }
                for h_name, h_val in heur_costs.items():
                    if h_val < ilp_dist - 1e-6:
                        raise RuntimeError(
                            f"VALIDATION ERROR: Heuristic {h_name} achieved {h_val:.4f} km, "
                            f"strictly beating proven-optimal ILP value {ilp_dist:.4f} km on n={n}, seed={seed}!"
                        )

                # Optimality gaps in percent
                gap_rows.append({
                    "n": n,
                    "seed": seed,
                    "ilp_opt_km": round(ilp_dist, 3),
                    "driver_order_gap%": round(100.0 * (c_do - ilp_dist) / ilp_dist, 2),
                    "nearest_neighbour_gap%": round(100.0 * (c_nn - ilp_dist) / ilp_dist, 2),
                    "clarke_wright_gap%": round(100.0 * (c_cw - ilp_dist) / ilp_dist, 2),
                    "cw_ls_gap%": round(100.0 * (c_ls - ilp_dist) / ilp_dist, 2),
                    "ortools_gap%": round(100.0 * (c_or - ilp_dist) / ilp_dist, 2),
                })

    df_gaps = pd.DataFrame(gap_rows)
    print("\n--- Per-Instance Optimality Gaps (%) vs Proven Optimal ILP ---")
    print(df_gaps.to_string(index=False))

    # Mean gaps summary
    mean_gaps = df_gaps.groupby("n")[
        [
            "driver_order_gap%",
            "nearest_neighbour_gap%",
            "clarke_wright_gap%",
            "cw_ls_gap%",
            "ortools_gap%",
        ]
    ].mean()
    print("\n--- Average Optimality Gap (%) by Problem Size n ---")
    print(mean_gaps.round(2).to_string())

    return records, df_gaps


# ---------------------------------------------------------------------------
# Experiment 2: Scaling Benchmark (n in {20, 50, 100, 200}, seeds 0-4)
# ---------------------------------------------------------------------------

def run_scaling_benchmark() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Scaling Benchmark (n in {20, 50, 100, 200}, seeds 0-4)")
    print("=" * 80)

    n_values = [20, 50, 100, 200]
    seeds = list(range(5))
    records: list[dict[str, Any]] = []

    # 1. Run n=20 ILP in parallel across the 5 seeds with 120s limit
    print("Running exact ILP on n=20 (5 seeds, 120s limit) in parallel...")
    with ProcessPoolExecutor(max_workers=min(5, os.cpu_count() or 1)) as executor:
        ilp_n20_results = list(executor.map(_worker_ilp_n20, seeds))

    for rec in ilp_n20_results:
        records.append({
            "method": rec["method"],
            "n": rec["n"],
            "seed": rec["seed"],
            "distance_km": rec["distance_km"],
            "vehicles": rec["vehicles"],
            "runtime_s": rec["runtime_s"],
            "valid": rec["valid"],
            "proved_optimal": rec["proved_optimal"],
        })
        print(
            f"  n=20 seed={rec['seed']}: status={rec['status']}, "
            f"dist={rec['distance_km']} km, proved_optimal={rec['proved_optimal']}, "
            f"best_bound={rec['best_bound']} km, runtime={rec['runtime_s']}s"
        )

    # 2. Run heuristics across all scaling sizes
    for n in n_values:
        or_time_limit = 30.0 if n == 200 else 10.0
        print(f"\nProcessing n = {n} (seeds 0-4)...")

        for seed in seeds:
            inst = generate_instance(n, seed)

            # Driver order
            t0 = time.perf_counter()
            r_do = driver_order(inst)
            t_do = time.perf_counter() - t0
            c_do = solution_cost(r_do, inst.dist_matrix)
            records.append({
                "method": "driver_order",
                "n": n,
                "seed": seed,
                "distance_km": round(c_do, 3),
                "vehicles": len(r_do),
                "runtime_s": round(t_do, 6),
                "valid": validate_solution(inst, r_do)[0],
                "proved_optimal": False,
            })

            # Nearest neighbour
            t0 = time.perf_counter()
            r_nn = nearest_neighbour(inst)
            t_nn = time.perf_counter() - t0
            c_nn = solution_cost(r_nn, inst.dist_matrix)
            records.append({
                "method": "nearest_neighbour",
                "n": n,
                "seed": seed,
                "distance_km": round(c_nn, 3),
                "vehicles": len(r_nn),
                "runtime_s": round(t_nn, 6),
                "valid": validate_solution(inst, r_nn)[0],
                "proved_optimal": False,
            })

            # Clarke-Wright
            t0 = time.perf_counter()
            r_cw = clarke_wright(inst)
            t_cw = time.perf_counter() - t0
            c_cw = solution_cost(r_cw, inst.dist_matrix)
            records.append({
                "method": "clarke_wright",
                "n": n,
                "seed": seed,
                "distance_km": round(c_cw, 3),
                "vehicles": len(r_cw),
                "runtime_s": round(t_cw, 6),
                "valid": validate_solution(inst, r_cw)[0],
                "proved_optimal": False,
            })

            # Clarke-Wright + Local Search
            t0 = time.perf_counter()
            r_cw_init = clarke_wright(inst)
            r_ls, _ = local_search(inst, r_cw_init)
            t_ls = time.perf_counter() - t0
            c_ls = solution_cost(r_ls, inst.dist_matrix)
            records.append({
                "method": "clarke_wright_local_search",
                "n": n,
                "seed": seed,
                "distance_km": round(c_ls, 3),
                "vehicles": len(r_ls),
                "runtime_s": round(t_ls, 6),
                "valid": validate_solution(inst, r_ls)[0],
                "proved_optimal": False,
            })

            # OR-Tools Routing
            t0 = time.perf_counter()
            r_or, c_or = ortools_routing(inst, time_limit_s=or_time_limit)
            t_or = time.perf_counter() - t0
            records.append({
                "method": "ortools_routing",
                "n": n,
                "seed": seed,
                "distance_km": round(c_or, 3),
                "vehicles": len(r_or),
                "runtime_s": round(t_or, 4),
                "valid": validate_solution(inst, r_or)[0],
                "proved_optimal": False,
            })

    df_scaling = pd.DataFrame(records)

    # Compute aggregation table (mean, std, improvements vs NN and vs DO)
    summary_rows: list[dict[str, Any]] = []
    methods_order = [
        "driver_order",
        "nearest_neighbour",
        "clarke_wright",
        "clarke_wright_local_search",
        "ortools_routing",
        "ilp_cvrp",
    ]

    for n in sorted(df_scaling["n"].unique()):
        sub_n = df_scaling[df_scaling["n"] == n]

        # Get pivot for percentage improvement on each seed
        pivot_dist = sub_n.pivot(index="seed", columns="method", values="distance_km")

        for m in methods_order:
            m_data = sub_n[sub_n["method"] == m]
            if m_data.empty:
                continue

            dist_mean = m_data["distance_km"].mean()
            dist_std = m_data["distance_km"].std()
            veh_mean = m_data["vehicles"].mean()
            time_mean = m_data["runtime_s"].mean()

            # Improvement vs Nearest Neighbour (%)
            if "nearest_neighbour" in pivot_dist.columns:
                nn_diffs = (pivot_dist["nearest_neighbour"] - pivot_dist[m]) / pivot_dist["nearest_neighbour"] * 100.0
                imp_nn_mean = nn_diffs.mean()
            else:
                imp_nn_mean = 0.0

            # Improvement vs Driver Order (%)
            if "driver_order" in pivot_dist.columns:
                do_diffs = (pivot_dist["driver_order"] - pivot_dist[m]) / pivot_dist["driver_order"] * 100.0
                imp_do_mean = do_diffs.mean()
            else:
                imp_do_mean = 0.0

            summary_rows.append({
                "n": n,
                "method": m,
                "dist_mean_km": round(dist_mean, 2),
                "dist_std_km": round(dist_std, 2),
                "vehicles_mean": round(veh_mean, 1),
                "runtime_mean_s": round(time_mean, 4),
                "imp_vs_NN_%": round(imp_nn_mean, 2),
                "imp_vs_DO_%": round(imp_do_mean, 2),
            })

    df_summary = pd.DataFrame(summary_rows)
    print("\n=== Track E Scaling Performance Summary Table ===")
    print(df_summary.to_string(index=False))

    return records, df_summary


# ---------------------------------------------------------------------------
# Experiment 3: Multiprocessing Scaling (n = 100, CW+LS, seeds 0-4)
# ---------------------------------------------------------------------------

def run_parallelism_experiment() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Multi-Core Parallelism (n = 100, CW+LS, 5 seeds)")
    print("=" * 80)

    n = 100
    seeds = list(range(5))
    tasks = [(n, s) for s in seeds]

    # 1. Sequential run
    t0 = time.perf_counter()
    for t in tasks:
        _worker_cw_ls(t)
    t_seq = time.perf_counter() - t0
    print(f"Sequential wall-clock time (1 process): {t_seq:.4f} s")

    # Worker pool counts to evaluate: 2, 4, and max CPU count
    max_cpus = os.cpu_count() or 4
    worker_counts = sorted(list({2, 4, min(len(seeds), max_cpus)}))

    parallel_rows = [{
        "processes": 1,
        "wall_time_s": round(t_seq, 4),
        "speedup": 1.00,
        "efficiency": 1.00,
    }]

    for p in worker_counts:
        t0 = time.perf_counter()
        with multiprocessing.Pool(processes=p) as pool:
            pool.map(_worker_cw_ls, tasks)
        t_par = time.perf_counter() - t0
        speedup = t_seq / t_par if t_par > 0 else 1.0
        efficiency = speedup / p
        parallel_rows.append({
            "processes": p,
            "wall_time_s": round(t_par, 4),
            "speedup": round(speedup, 2),
            "efficiency": round(efficiency, 2),
        })
        print(f"Multiprocessing with {p} processes: {t_par:.4f} s (speedup: {speedup:.2f}x, eff: {efficiency:.2f})")

    df_par = pd.DataFrame(parallel_rows)
    print("\n=== Parallel Scaling Summary Table ===")
    print(df_par.to_string(index=False))
    print("\n[NOTE] The workload is embarrassingly parallel across instances, but strictly single-threaded within each solver run.")

    return df_par


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_scaling_curves(df_all: pd.DataFrame) -> None:
    """Plot distance and runtime scaling versus n for all methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = [
        ("driver_order", "Driver Order (Naive)", "gray", "--", "x"),
        ("nearest_neighbour", "Nearest Neighbour", "#1f77b4", "-.", "o"),
        ("clarke_wright", "Clarke-Wright", "#ff7f0e", ":", "s"),
        ("clarke_wright_local_search", "CW + Local Search", "#2ca02c", "-", "^"),
        ("ortools_routing", "OR-Tools GLS (Reference)", "#d62728", "-", "*"),
        ("ilp_cvrp", "ILP (SCIP Exact)", "#9467bd", "-", "D"),
    ]

    for m_id, m_label, color, ls, marker in methods:
        sub = df_all[df_all["method"] == m_id]
        if sub.empty:
            continue
        agg = sub.groupby("n").agg({"distance_km": "mean", "runtime_s": "mean"}).reset_index()

        ax1.plot(
            agg["n"],
            agg["distance_km"],
            label=m_label,
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            agg["n"],
            agg["runtime_s"],
            label=m_label,
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("Problem Size n (Customers)", fontsize=11)
    ax1.set_ylabel("Total Route Distance (km)", fontsize=11)
    ax1.set_title("Track E: Solution Quality Scaling", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Problem Size n (Customers)", fontsize=11)
    ax2.set_ylabel("Wall Runtime (s, log scale)", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_title("Track E: Solver Runtime Scaling", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = RESULTS_DIR / "trackE_scaling.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nScaling plot saved to {out_path}")


def plot_routes_side_by_side(n: int = 50, seed: int = 0) -> None:
    """Plot side-by-side route maps for nearest_neighbour and CW+LS on n=50, seed=0."""
    inst = generate_instance(n, seed)

    # 1. Nearest Neighbour routes
    r_nn = nearest_neighbour(inst)
    cost_nn = solution_cost(r_nn, inst.dist_matrix)

    # 2. Clarke-Wright + Local Search routes
    r_cw = clarke_wright(inst)
    r_ls, _ = local_search(inst, r_cw)
    cost_ls = solution_cost(r_ls, inst.dist_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    depot_lat, depot_lon = DEPOT_COORDS

    cmap = plt.cm.tab20

    # Subplot 1: Nearest Neighbour
    for idx, r in enumerate(r_nn):
        pts = np.array([inst.coords[node] for node in r])
        # Plot route path (lon = x, lat = y)
        color = cmap(idx % 20)
        ax1.plot(pts[:, 1], pts[:, 0], color=color, linewidth=1.5, alpha=0.85)
        ax1.scatter(pts[1:-1, 1], pts[1:-1, 0], color=color, s=25, zorder=3)

    # Plot depot
    ax1.scatter(depot_lon, depot_lat, color="red", marker="s", s=120, edgecolors="black", zorder=5, label="Depot")
    ax1.set_title(f"Nearest Neighbour (Baseline)\nDistance: {cost_nn:.1f} km | Vehicles: {len(r_nn)}", fontsize=12)
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Clarke-Wright + Local Search
    for idx, r in enumerate(r_ls):
        pts = np.array([inst.coords[node] for node in r])
        color = cmap(idx % 20)
        ax2.plot(pts[:, 1], pts[:, 0], color=color, linewidth=1.5, alpha=0.85)
        ax2.scatter(pts[1:-1, 1], pts[1:-1, 0], color=color, s=25, zorder=3)

    ax2.scatter(depot_lon, depot_lat, color="red", marker="s", s=120, edgecolors="black", zorder=5, label="Depot")
    ax2.set_title(
        f"Clarke-Wright + Local Search (Optimized)\nDistance: {cost_ls:.1f} km | Vehicles: {len(r_ls)} "
        f"({(cost_nn - cost_ls) / cost_nn * 100:.1f}% shorter)",
        fontsize=12,
    )
    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(f"DabbaDash Mumbai CVRP Route Map Comparison (n = {n}, seed = {seed})", fontsize=14, y=0.98)
    plt.tight_layout()
    out_path = RESULTS_DIR / "trackE_routes_n50.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Route map comparison saved to {out_path}")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Exact comparison (n in {8, 10, 12})
    exact_records, df_gaps = run_exact_comparison()

    # 2. Scaling benchmark (n in {20, 50, 100, 200})
    scaling_records, df_scaling = run_scaling_benchmark()

    # 3. Parallelism experiment
    df_par = run_parallelism_experiment()

    # 4. Save combined comparison CSV
    all_records = exact_records + scaling_records
    df_all = pd.DataFrame(all_records)
    csv_out = RESULTS_DIR / "trackE_comparison.csv"
    df_all.to_csv(csv_out, index=False)
    print(f"\nAll benchmark records written to {csv_out}")

    # 5. Visualizations
    plot_scaling_curves(df_all)
    plot_routes_side_by_side(n=50, seed=0)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
