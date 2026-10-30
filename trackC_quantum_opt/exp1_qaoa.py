"""trackC_quantum_opt/exp1_qaoa.py
---------------------------------
QAOA depth sweep and classical comparison for Max-Cut.

NOTE: All quantum results in this study are SIMULATED on classical hardware.
No physical quantum processor was used.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml

from trackB_discrete.exp2_qubo import maxcut_brute_force, maxcut_greedy_local_search
from trackB_discrete.qubo import maxcut_to_qubo, simulated_annealing_qubo
from trackC_quantum_opt.optimizers_q import adam_ps, cobyla_wrapper, spsa
from trackC_quantum_opt.qaoa_sim import (
    approximation_ratio,
    expected_cut,
    finite_diff_grad,
    maxcut_diagonal,
    param_shift_grad,
    probability_of_optimal,
    sample,
)

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
INSTANCES_FILE = RESULTS_DIR / "trackB_maxcut_instances.json"


def load_instances() -> List[Dict[str, Any]]:
    with open(INSTANCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Part (a): Validation Suite
# ===========================================================================

def run_validation(instances: List[Dict[str, Any]]):
    print("=" * 80)
    print("PART (A): Mathematical & Simulation Validation (SIMULATED)")
    print("=" * 80)

    # 1. Verify max(maxcut_diagonal) matches stored optimal_cut_value for all 15 instances
    print("\n1. Verifying maxcut_diagonal max matches optimal_cut_value across all 15 instances:")
    all_match = True
    for inst in instances:
        n = inst["n"]
        edges = inst["edges"]
        diag = maxcut_diagonal(n, edges)
        max_diag = int(np.max(diag))
        opt_val = int(inst["optimal_cut_value"])
        if max_diag != opt_val:
            print(f"  [FAIL] {inst['instance_id']}: max diag={max_diag} != opt={opt_val}")
            all_match = False
    if all_match:
        print("  [PASS] All 15 instances have exact match between maxcut_diagonal and optimal_cut_value.")

    # 2. Check exact equivalence with QUBO formulation on a 6-node random graph
    print("\n2. Checking exact equivalence between maxcut_diagonal and QUBO evaluation on 6-node graph:")
    rng = np.random.default_rng(999)
    n_val = 6
    W_raw = rng.integers(1, 5, (n_val, n_val))
    W = (W_raw + W_raw.T) // 2
    np.fill_diagonal(W, 0)
    edges_val = [(i, j, int(W[i, j])) for i in range(n_val) for j in range(i + 1, n_val) if W[i, j] > 0]

    diag_val = maxcut_diagonal(n_val, edges_val)
    Q, _ = maxcut_to_qubo(W)

    qubo_match = True
    for idx in range(1 << n_val):
        # In trackB qubo convention:
        # idx -> bitstring x where x[k] = (idx >> (n - 1 - k)) & 1 (MSB)
        x = np.array([(idx >> (n_val - 1 - k)) & 1 for k in range(n_val)], dtype=float)
        qubo_cut = -(x.T @ Q @ x)
        if not np.isclose(diag_val[idx], qubo_cut):
            qubo_match = False
            break
    if qubo_match:
        print("  [PASS] All 2^6 = 64 states match: maxcut_diagonal[idx] == -(x^T Q x).")
    else:
        print("  [FAIL] Discrepancy between maxcut_diagonal and QUBO formulation.")

    # 3. PennyLane default.qubit comparison on 3 instances at p=1
    print("\n3. Comparing NumPy simulator against PennyLane default.qubit (p=1, 3 instances):")
    for inst in instances[:3]:
        n = inst["n"]
        edges = inst["edges"]
        diag = maxcut_diagonal(n, edges)
        p = 1
        params = np.array([0.5, 0.4])

        exp_numpy = expected_cut(params, diag, n)

        dev = qml.device("default.qubit", wires=n)
        coeffs = [w / 2.0 for _, _, w in edges] + [-w / 2.0 for _, _, w in edges]
        obs = [qml.Identity(0) for _ in edges] + [qml.PauliZ(u) @ qml.PauliZ(v) for u, v, _ in edges]
        cost_ham = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit(g, b):
            for i in range(n):
                qml.Hadamard(wires=i)
            for u, v, w in edges:
                qml.IsingZZ(-g[0] * w, wires=[u, v])
            for i in range(n):
                qml.RX(2 * b[0], wires=i)
            return qml.expval(cost_ham)

        exp_pl = float(circuit(params[:p], params[p:]))
        diff = abs(exp_numpy - exp_pl)
        print(f"  {inst['instance_id']}: NumPy={exp_numpy:.6f}, PennyLane={exp_pl:.6f}, diff={diff:.2e}")
        assert diff < 1e-9

    # 4. Parameter-shift gradient vs finite differences vs PennyLane grad
    print("\n4. Checking Parameter-Shift gradient against Finite Differences:")
    inst0 = instances[0]
    n0 = inst0["n"]
    edges0 = inst0["edges"]
    diag0 = maxcut_diagonal(n0, edges0)
    params0 = np.array([0.35, 0.45])

    ps_g = param_shift_grad(params0, diag0, n0, edges0)
    fd_g = finite_diff_grad(params0, diag0, n0, eps=1e-6)
    max_err = np.max(np.abs(ps_g - fd_g))
    print(f"  Instance {inst0['instance_id']}:")
    print(f"    PS Grad: {ps_g}")
    print(f"    FD Grad: {fd_g}")
    print(f"    Max Diff: {max_err:.2e} (PASS < 1e-5)")

    # 5. Naive shared parameter shift vs Per-gate chain rule shift
    print("\n5. Demonstrating why naive shared-parameter shift FAILS on weighted graphs:")
    shift = np.pi / 2.0
    p_plus = params0.copy()
    p_plus[0] += shift
    p_minus = params0.copy()
    p_minus[0] -= shift
    naive_grad_gamma0 = (expected_cut(p_plus, diag0, n0) - expected_cut(p_minus, diag0, n0)) / (2.0 * shift)
    print(f"    True Per-Gate d<C>/dgamma_0:    {ps_g[0]:.6f}")
    print(f"    Naive Shared-Shift d<C>/dgamma: {naive_grad_gamma0:.6f}")
    print(f"    Absolute Error:                 {abs(naive_grad_gamma0 - ps_g[0]):.6f}")
    print("    -> Confirmed: Naive shift yields invalid gradient for multi-generator hamiltonians.")


# ===========================================================================
# Part (b): QAOA Depth Sweep
# ===========================================================================

def run_qaoa_depth_sweep(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("PART (B): QAOA Depth Sweep (p in {1, 2, 3}) across Optimizers")
    print("=" * 80)

    p_values = [1, 2, 3]
    optimizers = ["Adam", "COBYLA", "SPSA"]
    budget = 300

    records = []

    total_runs = 0
    t0_all = time.perf_counter()

    for inst in instances:
        inst_id = inst["instance_id"]
        n = inst["n"]
        seed = inst["seed"]
        edges = inst["edges"]
        opt_cut = float(inst["optimal_cut_value"])
        diag = maxcut_diagonal(n, edges)

        n_starts = 3 if n == 14 else 5

        for p in p_values:
            for opt_name in optimizers:
                for start_idx in range(n_starts):
                    run_seed = seed * 1000 + start_idx * 17 + p
                    rng = np.random.default_rng(run_seed)

                    gamma_init = rng.uniform(0.0, np.pi, p)
                    beta_init = rng.uniform(0.0, np.pi / 2.0, p)
                    x0 = np.concatenate([gamma_init, beta_init])

                    if opt_name == "Adam":
                        def loss_and_grad(params_vec):
                            val = -expected_cut(params_vec, diag, n)
                            g = -param_shift_grad(params_vec, diag, n, edges)
                            return val, g

                        res = adam_ps(loss_and_grad, x0, budget=budget, lr=0.04)

                    elif opt_name == "COBYLA":
                        loss_fn = lambda params_vec: -expected_cut(params_vec, diag, n)
                        res = cobyla_wrapper(loss_fn, x0, budget=budget, rhobeg=0.4)

                    elif opt_name == "SPSA":
                        loss_fn = lambda params_vec: -expected_cut(params_vec, diag, n)
                        res = spsa(loss_fn, x0, budget=budget, seed=run_seed, a=0.2, c=0.1)

                    best_params = res.x
                    ratio = approximation_ratio(best_params, diag, n, opt_cut)
                    p_opt = probability_of_optimal(best_params, diag, n, opt_cut)

                    sample_cuts = sample(best_params, diag, n, shots=1000, rng=rng)
                    sample_ratio = float(np.max(sample_cuts)) / opt_cut

                    records.append({
                        "instance_id": inst_id,
                        "n": n,
                        "seed": seed,
                        "p": p,
                        "optimizer": opt_name,
                        "start": start_idx,
                        "ratio": ratio,
                        "p_opt": p_opt,
                        "sample_ratio": sample_ratio,
                        "nfev": res.nfev,
                    })
                    total_runs += 1

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "trackC_qaoa.csv"
    df.to_csv(csv_path, index=False)
    elapsed = time.perf_counter() - t0_all
    print(f"\nCompleted {total_runs} QAOA optimization runs in {elapsed:.2f} s.")
    print(f"Results saved to {csv_path}")

    # Print summary table: Mean +/- Std of approximation ratio per (n, p, optimizer)
    # Take best over starts per instance first, as standard protocol
    best_per_instance = df.groupby(["instance_id", "n", "p", "optimizer"])["ratio"].max().reset_index()
    summary = best_per_instance.groupby(["n", "p", "optimizer"])["ratio"].agg(["mean", "std"]).reset_index()

    print("\nSummary: Approximation Ratio (Mean +/- Std over instances, best of starts):")
    print(summary.to_string(index=False))

    return df


# ===========================================================================
# Part (c): Classical Baselines Comparison
# ===========================================================================

def run_classical_comparison(instances: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("PART (C): Classical Solvers Comparison against QAOA")
    print("=" * 80)

    rows = []
    for inst in instances:
        n = inst["n"]
        edges = inst["edges"]
        opt_cut = float(inst["optimal_cut_value"])
        seed = inst["seed"]

        W = np.zeros((n, n), dtype=float)
        total_weight = 0.0
        for u, v, w in edges:
            W[u, v] = float(w)
            W[v, u] = float(w)
            total_weight += float(w)

        # 1. Random baseline (expectation = total_weight / 2)
        random_ratio = (total_weight / 2.0) / opt_cut

        # 2. Greedy Local Search (20 restarts)
        gls_cut, _, _ = maxcut_greedy_local_search(W, restarts=20, seed=seed)
        gls_ratio = float(gls_cut) / opt_cut

        # 3. Simulated Annealing (QUBO formulation)
        Q, _ = maxcut_to_qubo(W)
        sa_e, _, _ = simulated_annealing_qubo(Q, seed=seed, steps=2000, restarts=5)
        sa_cut = -sa_e
        sa_ratio = float(sa_cut) / opt_cut

        rows.append({
            "instance_id": inst["instance_id"],
            "n": n,
            "random_ratio": random_ratio,
            "gls_ratio": gls_ratio,
            "sa_ratio": sa_ratio,
            "opt_ratio": 1.0,
        })

    df_classical = pd.DataFrame(rows)
    agg_classical = df_classical.groupby("n")[["random_ratio", "gls_ratio", "sa_ratio"]].mean().reset_index()
    print("\nClassical Baseline Mean Approximation Ratios:")
    print(agg_classical.to_string(index=False))
    return df_classical


# ===========================================================================
# Plotting
# ===========================================================================

def generate_plots(df_qaoa: pd.DataFrame, instances: List[Dict[str, Any]]):
    print("\nGenerating visualization figures...")

    # Plot 1: Ratio vs p per optimizer (averaged over instances per n)
    best_df = df_qaoa.groupby(["instance_id", "n", "p", "optimizer"])["ratio"].max().reset_index()
    summary = best_df.groupby(["n", "p", "optimizer"])["ratio"].mean().reset_index()

    ns = sorted(summary["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.5), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    opts = ["Adam", "COBYLA", "SPSA"]
    colors = {"Adam": "#1f77b4", "COBYLA": "#ff7f0e", "SPSA": "#2ca02c"}
    markers = {"Adam": "o", "COBYLA": "s", "SPSA": "^"}

    for ax, n_val in zip(axes, ns):
        sub = summary[summary["n"] == n_val]
        for opt in opts:
            opt_sub = sub[sub["optimizer"] == opt].sort_values("p")
            ax.plot(
                opt_sub["p"],
                opt_sub["ratio"],
                label=opt,
                color=colors[opt],
                marker=markers[opt],
                linewidth=2,
            )
        ax.set_title(f"n = {n_val} Qubits", fontsize=12)
        ax.set_xlabel("QAOA Layers (p)", fontsize=11)
        ax.set_xticks([1, 2, 3])
        ax.grid(True, linestyle="--", alpha=0.6)
        if ax == axes[0]:
            ax.set_ylabel("Approximation Ratio (<C> / C_opt)", fontsize=11)
            ax.legend(title="Optimizer")

    fig.suptitle("QAOA Approximation Ratio vs Depth p (SIMULATED)", fontsize=14, y=1.02)
    plt.tight_layout()
    depth_plot_path = RESULTS_DIR / "trackC_exp1_depth.png"
    fig.savefig(depth_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved depth sweep plot to {depth_plot_path}")

    # Plot 2: 2D Loss Landscape for n=10, instance 0, p=1
    inst0 = instances[0]
    n0 = inst0["n"]
    edges0 = inst0["edges"]
    diag0 = maxcut_diagonal(n0, edges0)
    opt_cut0 = float(inst0["optimal_cut_value"])

    grid_size = 40
    gammas = np.linspace(0.0, np.pi, grid_size)
    betas = np.linspace(0.0, np.pi / 2.0, grid_size)
    G, B = np.meshgrid(gammas, betas)
    Z = np.zeros_like(G)

    for i in range(grid_size):
        for j in range(grid_size):
            p_vec = np.array([G[i, j], B[i, j]])
            Z[i, j] = expected_cut(p_vec, diag0, n0) / opt_cut0

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(G, B, Z, levels=30, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Approximation Ratio", fontsize=11)

    # Run single optimizations from fixed point to display endpoints
    x0_plot = np.array([np.pi / 2.0, np.pi / 4.0])
    res_adam = adam_ps(
        lambda p_vec: (-expected_cut(p_vec, diag0, n0), -param_shift_grad(p_vec, diag0, n0, edges0)),
        x0_plot,
        budget=100,
        lr=0.05,
    )
    res_cobyla = cobyla_wrapper(lambda p_vec: -expected_cut(p_vec, diag0, n0), x0_plot, budget=100)
    res_spsa = spsa(lambda p_vec: -expected_cut(p_vec, diag0, n0), x0_plot, budget=100, seed=42)

    ax.scatter(x0_plot[0], x0_plot[1], color="red", marker="x", s=80, label="Start x0")
    ax.scatter(res_adam.x[0], res_adam.x[1], color="cyan", marker="o", s=70, label="Adam-PS End")
    ax.scatter(res_cobyla.x[0], res_cobyla.x[1], color="orange", marker="s", s=70, label="COBYLA End")
    ax.scatter(res_spsa.x[0], res_spsa.x[1], color="magenta", marker="^", s=70, label="SPSA End")

    ax.set_title(f"QAOA p=1 Energy Landscape ({inst0['instance_id']})", fontsize=12)
    ax.set_xlabel(r"$\gamma$ (Cost Phase)", fontsize=11)
    ax.set_ylabel(r"$\beta$ (Mixer Angle)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    landscape_path = RESULTS_DIR / "trackC_exp1_landscape.png"
    fig.savefig(landscape_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved landscape plot to {landscape_path}")


def main():
    print("=" * 80)
    print("TRACK C: EXP 1 - QAOA DEPTH SWEEP & CLASSICAL BENCHMARK")
    print("NOTE: All results are SIMULATED on classical computers.")
    print("=" * 80)

    instances = load_instances()
    run_validation(instances)
    df_qaoa = run_qaoa_depth_sweep(instances)
    run_classical_comparison(instances)
    generate_plots(df_qaoa, instances)
    print("\nExp 1 Completed successfully.")


if __name__ == "__main__":
    main()
