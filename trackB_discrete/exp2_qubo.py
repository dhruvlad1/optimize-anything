"""trackB_discrete/exp2_qubo.py
----------------------------
QUBO and Ising benchmarks:
  (a) Max-Cut on random graphs (n in {10, 12, 14}):
      - Solvers: Brute force, ILP (SCIP), Greedy Local Search (20 restarts), Simulated Annealing (10 runs).
      - Cross-solver verification: Brute force == ILP == QUBO Brute force.
      - Save instances to results/trackB_maxcut_instances.json for Track C reuse.
  (b) Binary-to-spin exact equivalence check across all 2^n assignments.
  (c) Cardinality-constrained penalty study (P in {0.1, 0.5, 1, 2, 5} * max(v)).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from trackB_discrete.qubo import (
    brute_force_qubo,
    cardinality_qubo,
    evaluate_qubo,
    ising_energy,
    maxcut_to_qubo,
    qubo_to_ising,
    simulated_annealing_qubo,
)

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


# ===========================================================================
# Max-Cut ILP and Local Search Solvers
# ===========================================================================

def maxcut_ilp(W: np.ndarray) -> Tuple[int, np.ndarray, float]:
    """Solve Max-Cut via Integer Linear Programming using OR-Tools SCIP."""
    t0 = time.perf_counter()
    n = W.shape[0]
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver not available in ortools.")

    # Node assignment variables: x_i in {0, 1}
    x = [solver.BoolVar(f"x_{i}") for i in range(n)]

    # Edge cut indicators: y_{ij} in {0, 1} for i < j where W[i, j] > 0
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                y[i, j] = solver.BoolVar(f"y_{i}_{j}")
                # Linearization constraints for XOR: y_ij == x_i XOR x_j
                # y_ij <= x_i + x_j
                solver.Add(y[i, j] <= x[i] + x[j])
                # y_ij <= 2 - (x_i + x_j)
                solver.Add(y[i, j] <= 2 - x[i] - x[j])

    # Objective: maximize sum_{(i, j)} w_{ij} * y_{ij}
    obj_expr = solver.Sum(int(W[i, j]) * y[i, j] for (i, j) in y)
    solver.Maximize(obj_expr)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Max-Cut ILP failed, status: {status}")

    best_cut = int(round(solver.Objective().Value()))
    best_x = np.array([int(round(x[i].solution_value())) for i in range(n)], dtype=int)
    runtime = time.perf_counter() - t0
    return best_cut, best_x, runtime


def maxcut_brute_force(W: np.ndarray) -> Tuple[int, np.ndarray, float]:
    """Solve Max-Cut by exhaustive enumeration over 2^n assignments."""
    t0 = time.perf_counter()
    n = W.shape[0]
    best_cut = -1
    best_x = np.zeros(n, dtype=int)

    # Fix node 0 = 0 to break binary symmetry and evaluate 2^{n-1} cuts
    for mask in range(1 << (n - 1)):
        # x[0] = 0, x[1..n-1] from mask
        x = np.zeros(n, dtype=int)
        for i in range(1, n):
            if (mask >> (i - 1)) & 1:
                x[i] = 1

        # Compute cut weight
        cut = 0
        for i in range(n):
            for j in range(i + 1, n):
                if x[i] != x[j]:
                    cut += int(W[i, j])

        if cut > best_cut:
            best_cut = cut
            best_x = x.copy()

    runtime = time.perf_counter() - t0
    return best_cut, best_x, runtime


def maxcut_greedy_local_search(
    W: np.ndarray,
    restarts: int = 20,
    seed: int = 0,
) -> Tuple[int, np.ndarray, float]:
    """Greedy local search (steepest ascent) with multiple random restarts."""
    t0 = time.perf_counter()
    n = W.shape[0]
    rng = np.random.default_rng(seed)

    def compute_cut(x_arr: np.ndarray) -> int:
        c = 0
        for i in range(n):
            for j in range(i + 1, n):
                if x_arr[i] != x_arr[j]:
                    c += int(W[i, j])
        return c

    global_best_cut = -1
    global_best_x = np.zeros(n, dtype=int)

    for _ in range(restarts):
        # Random initial partition
        x = rng.integers(0, 2, size=n).astype(int)
        cur_cut = compute_cut(x)

        # Iterative steepest ascent
        while True:
            best_gain = 0
            best_flip_node = -1

            for node in range(n):
                # Gain from flipping node: sum_{j in N(node)} w_{node, j} * (1 if x[node]==x[j] else -1)
                gain = 0
                for j in range(n):
                    if j != node and W[node, j] > 0:
                        if x[node] == x[j]:
                            gain += int(W[node, j])
                        else:
                            gain -= int(W[node, j])

                if gain > best_gain:
                    best_gain = gain
                    best_flip_node = node

            if best_gain > 0:
                x[best_flip_node] = 1 - x[best_flip_node]
                cur_cut += best_gain
            else:
                break  # Local optimum reached

        if cur_cut > global_best_cut:
            global_best_cut = cur_cut
            global_best_x = x.copy()

    runtime = time.perf_counter() - t0
    return global_best_cut, global_best_x, runtime


# ===========================================================================
# Part (a): Max-Cut Benchmark
# ===========================================================================

def exp2_part_a_maxcut():
    print("=" * 75)
    print("PART (A): Max-Cut Optimization (n in {10, 12, 14})")
    print("=" * 75)

    graph_ns = [10, 12, 14]
    n_seeds = 5
    saved_instances = []

    rows = []

    for n in graph_ns:
        print(f"\nProcessing n = {n} nodes...")

        # Containers for metrics across instances
        method_names = ["Brute Force", "ILP (SCIP)", "Greedy LS (20 restarts)", "Simulated Annealing (10 runs)"]
        cuts_dict = {m: [] for m in method_names}
        pct_opt_dict = {m: [] for m in method_names}
        exact_hits_dict = {m: [] for m in method_names}
        times_dict = {m: [] for m in method_names}

        for seed in range(n_seeds):
            # Generate random Erdos-Renyi graph with p=0.5
            G = nx.erdos_renyi_graph(n=n, p=0.5, seed=seed)
            rng_w = np.random.default_rng(seed + 1000)

            # Assign integer weights 1-5
            W = np.zeros((n, n), dtype=int)
            edge_list = []
            for u, v in G.edges():
                w_val = int(rng_w.integers(1, 6))
                W[u, v] = w_val
                W[v, u] = w_val
                edge_list.append([int(u), int(v), w_val])

            # 1. Brute Force
            cut_bf, x_bf, t_bf = maxcut_brute_force(W)

            # 2. ILP (SCIP)
            cut_ilp, x_ilp, t_ilp = maxcut_ilp(W)

            # 3. QUBO Brute Force Check
            Q, off = maxcut_to_qubo(W)
            e_qubo_bf, x_qubo_bf = brute_force_qubo(Q, off)
            cut_qubo_bf = int(round(-e_qubo_bf))

            # Cross-solver exact equivalence verification
            if not (cut_bf == cut_ilp == cut_qubo_bf):
                raise RuntimeError(
                    f"Disagreement at n={n}, seed={seed}: BF={cut_bf}, ILP={cut_ilp}, QUBO_BF={cut_qubo_bf}"
                )

            opt_cut = cut_bf

            # 4. Greedy Local Search (20 restarts)
            cut_ls, x_ls, t_ls = maxcut_greedy_local_search(W, restarts=20, seed=seed)

            # 5. Simulated Annealing (10 independent runs per instance)
            sa_cuts = []
            t0_sa = time.perf_counter()
            for sa_run in range(10):
                e_sa, x_sa, _ = simulated_annealing_qubo(
                    Q, off, seed=seed * 100 + sa_run, steps=1500, t_init=10.0, t_final=0.01
                )
                sa_cuts.append(int(round(-e_sa)))
            t_sa = time.perf_counter() - t0_sa

            mean_sa_cut = float(np.mean(sa_cuts))
            best_sa_cut = max(sa_cuts)
            sa_hit_rate = float(np.mean([c == opt_cut for c in sa_cuts]))

            # Record metrics
            # Brute Force
            cuts_dict["Brute Force"].append(cut_bf)
            pct_opt_dict["Brute Force"].append(100.0)
            exact_hits_dict["Brute Force"].append(1.0)
            times_dict["Brute Force"].append(t_bf)

            # ILP
            cuts_dict["ILP (SCIP)"].append(cut_ilp)
            pct_opt_dict["ILP (SCIP)"].append(100.0)
            exact_hits_dict["ILP (SCIP)"].append(1.0)
            times_dict["ILP (SCIP)"].append(t_ilp)

            # Greedy LS
            cuts_dict["Greedy LS (20 restarts)"].append(cut_ls)
            pct_opt_dict["Greedy LS (20 restarts)"].append((cut_ls / opt_cut) * 100.0)
            exact_hits_dict["Greedy LS (20 restarts)"].append(1.0 if cut_ls == opt_cut else 0.0)
            times_dict["Greedy LS (20 restarts)"].append(t_ls)

            # Simulated Annealing
            cuts_dict["Simulated Annealing (10 runs)"].append(mean_sa_cut)
            pct_opt_dict["Simulated Annealing (10 runs)"].append((mean_sa_cut / opt_cut) * 100.0)
            exact_hits_dict["Simulated Annealing (10 runs)"].append(sa_hit_rate)
            times_dict["Simulated Annealing (10 runs)"].append(t_sa / 10.0)

            # Save instance for Track C reuse
            saved_instances.append({
                "instance_id": f"maxcut_n{n}_seed{seed}",
                "n": n,
                "seed": seed,
                "edges": edge_list,
                "optimal_cut_value": opt_cut,
                "optimal_assignment": [int(v) for v in x_bf],
            })

        for m in method_names:
            rows.append({
                "n": n,
                "Method": m,
                "Cut Value (mean)": np.mean(cuts_dict[m]),
                "Pct of Optimum (%)": np.mean(pct_opt_dict[m]),
                "Exact Optimum Hit Rate (%)": np.mean(exact_hits_dict[m]) * 100.0,
                "Runtime (s) (mean)": np.mean(times_dict[m]),
            })

    # Save instances to JSON for Track C
    with open(RESULTS / "trackB_maxcut_instances.json", "w") as fh:
        json.dump(saved_instances, fh, indent=2)
    print(f"Saved {len(saved_instances)} verified Max-Cut instances to results/trackB_maxcut_instances.json")

    df_a = pd.DataFrame(rows)
    print("\nMax-Cut Solvers Performance Summary:")
    print(df_a.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df_a


# ===========================================================================
# Part (b): Binary-to-Spin Exact Equivalence Check
# ===========================================================================

def exp2_part_b_spin_check():
    print("\n" + "=" * 75)
    print("PART (B): Binary-to-Spin Exact Energy Equivalence Check")
    print("=" * 75)

    check_ns = [6, 8]
    for n in check_ns:
        # Generate random graph
        G = nx.erdos_renyi_graph(n, p=0.6, seed=42 + n)
        rng = np.random.default_rng(100 + n)
        W = np.zeros((n, n))
        for u, v in G.edges():
            w = rng.integers(1, 10)
            W[u, v] = w
            W[v, u] = w

        Q, qubo_off = maxcut_to_qubo(W)
        J, h, ising_off = qubo_to_ising(Q, qubo_off)

        max_diff = 0.0
        for mask in range(1 << n):
            x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
            s = 2.0 * x - 1.0  # spin transformation: s in {-1, +1}

            e_qubo = evaluate_qubo(Q, x, qubo_off)
            e_ising = ising_energy(s, J, h, ising_off)

            diff = abs(e_qubo - e_ising)
            if diff > max_diff:
                max_diff = diff

        print(f"Graph n = {n} (evaluated all {1 << n} configurations):")
        print(f"  Maximum absolute difference |E_QUBO(x) - E_Ising(s)| = {max_diff:.4e}")
        assert max_diff < 1e-12, f"Discrepancy between QUBO and Ising energy on n={n}: {max_diff}"


# ===========================================================================
# Part (c): Cardinality Penalty Sweep
# ===========================================================================

def exp2_part_c_penalties():
    print("\n" + "=" * 75)
    print("PART (C): Cardinality Penalty Calibration (n=14, k=5)")
    print("=" * 75)

    n = 14
    k = 5
    n_seeds = 10
    multipliers = [0.1, 0.5, 1.0, 2.0, 5.0]

    penalty_results = []

    for mult in multipliers:
        feasible_count = 0
        optimal_count = 0

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            values = rng.integers(1, 101, size=n).astype(float)
            max_v = float(np.max(values))
            P = mult * max_v

            # Ground truth optimum: select top k largest values
            sorted_indices = np.argsort(values)[::-1]
            true_opt_val = float(np.sum(values[sorted_indices[:k]]))

            # Formulate and solve QUBO
            Q, off = cardinality_qubo(values, k, penalty=P)
            best_e, best_x = brute_force_qubo(Q, off)

            chosen_count = int(np.sum(best_x))
            chosen_val = float(np.sum(values * best_x))

            is_feasible = (chosen_count == k)
            is_optimal = is_feasible and (abs(chosen_val - true_opt_val) < 1e-6)

            if is_feasible:
                feasible_count += 1
            if is_optimal:
                optimal_count += 1

        pct_feasible = (feasible_count / n_seeds) * 100.0
        pct_optimal = (optimal_count / n_seeds) * 100.0

        penalty_results.append({
            "Penalty Multiplier": f"{mult} * max(v)",
            "Feasible Seeds (%)": pct_feasible,
            "Optimal Seeds (%)": pct_optimal,
            "Observation": (
                "Too small: constraint violated (infeasible)"
                if pct_feasible < 50
                else ("Marginal" if pct_optimal < 100 else "Robust: 100% feasible and optimal")
            ),
        })

    df_c = pd.DataFrame(penalty_results)
    print("Penalty Multiplier Study Table:")
    print(df_c.to_string(index=False))
    df_c.to_csv(RESULTS / "trackB_penalty.csv", index=False)
    print("\nSaved penalty results to results/trackB_penalty.csv")
    return df_c


def main() -> None:
    exp2_part_a_maxcut()
    exp2_part_b_spin_check()
    exp2_part_c_penalties()


if __name__ == "__main__":
    main()
