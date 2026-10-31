"""trackC_quantum_opt/exp4_vqe.py
---------------------------------
Variational Quantum Eigensolver (VQE) and Quantum Natural Gradient (QNG) experiments.

NOTE: All results in this study are SIMULATED on classical hardware using PennyLane default.qubit
      for validation and the pure-numpy fast backend for all optimization sweeps.
      No physical quantum processor was used.

Usage
-----
  python -m trackC_quantum_opt.exp4_vqe           # full run (< 10 min)
  python -m trackC_quantum_opt.exp4_vqe --quick   # quick run: 3 starts, 50 iters, small grids
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

from trackC_quantum_opt.vqe import (
    ansatz_param_count,
    exact_ground_energy,
    heisenberg_chain_hamiltonian,
    ising_chain_hamiltonian,
    make_state_qnode,
    make_vqe_qnode,
    metric_tensor_vqe,
    param_shift_grad_vqe,
)
from trackC_quantum_opt.vqe_fast import (
    build_dense_hamiltonian,
    energy_fast,
    metric_tensor_fast,
    param_shift_grad_fast,
)

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 4: VQE & QNG benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 starts, 50 iters, smaller tuning grids",
    )
    return parser.parse_args()


# ===========================================================================
# Part (a): Timing the fast backend and estimating total runtime
# ===========================================================================

def measure_fast_backend_timing(n: int = 4, layers: int = 3) -> Tuple[float, float, float]:
    """Measure wall-clock time for one energy eval, one gradient, one metric tensor.

    Returns (t_energy_ms, t_grad_ms, t_metric_ms).
    """
    H = ising_chain_hamiltonian(n, J=1.0, h=1.0)
    dense_H = build_dense_hamiltonian(H, n)
    d = ansatz_param_count(n, layers)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 2 * np.pi, d)

    # Warm-up
    energy_fast(p, dense_H, n, layers)
    param_shift_grad_fast(p, dense_H, n, layers)
    metric_tensor_fast(p, n, layers)

    REPS = 5

    t0 = time.perf_counter()
    for _ in range(REPS):
        energy_fast(p, dense_H, n, layers)
    t_energy = (time.perf_counter() - t0) / REPS * 1e3  # ms

    t0 = time.perf_counter()
    for _ in range(REPS):
        param_shift_grad_fast(p, dense_H, n, layers)
    t_grad = (time.perf_counter() - t0) / REPS * 1e3

    t0 = time.perf_counter()
    for _ in range(REPS):
        metric_tensor_fast(p, n, layers)
    t_metric = (time.perf_counter() - t0) / REPS * 1e3

    return t_energy, t_grad, t_metric


def print_timing_and_estimate(
    t_energy_ms: float,
    t_grad_ms: float,
    t_metric_ms: float,
    n_iters: int,
    n_starts: int,
    gd_lr_grid: List[float],
    qng_lr_grid: List[float],
    qng_lam_grid: List[float],
) -> float:
    """Print per-op timings and estimated total runtime. Return estimate in seconds."""
    n_problems = 4  # TFIM L2, TFIM L3, Heis L2, Heis L3
    n_opts = 4      # GD, Adam, QNG, COBYLA

    # Tuning cost (run once per problem on start 0, 25 iters GD + 20 iters QNG * grid)
    n_gd_tune = len(gd_lr_grid)
    n_qng_tune = len(qng_lr_grid) * len(qng_lam_grid)
    tune_gd_ms = n_gd_tune * 25 * t_grad_ms
    tune_qng_ms = n_qng_tune * 20 * (t_grad_ms + t_metric_ms)
    tune_total_ms = n_problems * (tune_gd_ms + tune_qng_ms)

    # Optimization sweeps
    gd_ms   = n_starts * n_iters * t_grad_ms          # GD: 1 grad per iter
    adam_ms = n_starts * n_iters * t_grad_ms           # Adam: 1 grad per iter
    qng_ms  = n_starts * n_iters * (t_grad_ms + t_metric_ms)   # QNG: grad + metric per iter
    # COBYLA: same budget as GD in circuit evals, each eval costs t_energy_ms
    # For TFIM_L2 (d=16): cobyla_budget = 100*2*16=3200 evals; L3 (d=24): 4800 evals
    # Heisenberg same. Average d ~ 20.
    avg_d = 20
    cobyla_ms = n_starts * (n_iters * 2 * avg_d) * t_energy_ms

    sweeps_ms = n_problems * (gd_ms + adam_ms + qng_ms + cobyla_ms)
    total_ms = tune_total_ms + sweeps_ms
    total_s = total_ms / 1e3

    print("\n" + "=" * 60)
    print("TIMING: Fast numpy backend (n=4, L=3)")
    print("=" * 60)
    print(f"  Energy eval:        {t_energy_ms:.3f} ms")
    print(f"  Param-shift grad:   {t_grad_ms:.3f} ms  (2*d energy evals)")
    print(f"  Full metric tensor: {t_metric_ms:.3f} ms  (d+1 state evals)")
    print(f"  Estimated total runtime: {total_s:.1f} s ({total_s/60:.1f} min)")
    print("=" * 60)

    return total_s


# ===========================================================================
# Part (a): Mathematical & Simulation Validation
# ===========================================================================

def run_validation():
    print("=" * 80)
    print("EXP 4: PART (A) - VQE Validation Suite (SIMULATED)")
    print("=" * 80)

    n = 3
    layers = 2
    d = ansatz_param_count(n, layers)
    H = ising_chain_hamiltonian(n, J=1.0, h=1.0)
    E0 = exact_ground_energy(H, n)
    dense_H = build_dense_hamiltonian(H, n)

    # Devices
    dev_aux = qml.device("default.qubit", wires=n + 1)
    dev = qml.device("default.qubit", wires=n)

    qnode_exp = make_vqe_qnode(H, n, layers, dev=dev)
    qnode_exp_aux = make_vqe_qnode(H, n, layers, dev=dev_aux)

    rng = np.random.default_rng(42)
    p = rng.uniform(0.1, 2.0 * np.pi, d)
    p_autograd = pnp.array(p, requires_grad=True)

    # 1. Parameter-shift gradient against PennyLane autograd and finite differences
    print("\n1. Verifying parameter-shift gradient against PennyLane and finite differences:")
    grad_hand = param_shift_grad_vqe(qnode_exp, p)
    grad_pl = np.asarray(qml.grad(qnode_exp)(p_autograd))

    # Central finite differences
    eps = 1e-6
    grad_fd = np.zeros(d, dtype=np.float64)
    for i in range(d):
        p_plus = p.copy(); p_plus[i] += eps
        p_minus = p.copy(); p_minus[i] -= eps
        grad_fd[i] = (float(qnode_exp(p_plus)) - float(qnode_exp(p_minus))) / (2.0 * eps)

    diff_pl = float(np.max(np.abs(grad_hand - grad_pl)))
    diff_fd = float(np.max(np.abs(grad_hand - grad_fd)))
    norm_pl = float(np.linalg.norm(grad_pl))
    rel_err_pl = diff_pl / max(1e-12, norm_pl)

    print(f"  Hand vs PennyLane Max Abs Diff: {diff_pl:.2e} (Rel Error: {rel_err_pl:.2e})")
    print(f"  Hand vs Finite Diff Max Abs Diff: {diff_fd:.2e}")
    if rel_err_pl >= 1e-6:
        raise RuntimeError(f"Validation FAILED: Gradient rel error {rel_err_pl:.2e} >= 1e-6")
    print("  [PASS] Parameter-shift gradient matches PennyLane (< 1e-6) and finite differences.")

    # 2. Metric tensors (Full and Block-Diagonal) against qml.metric_tensor
    print("\n2. Verifying hand-written metric tensors against qml.metric_tensor:")
    g_full_hand = metric_tensor_vqe(p, n, layers, approx=None, dev_state=dev)
    g_block_hand = metric_tensor_vqe(p, n, layers, approx="block-diag", dev_state=dev)

    g_full_pl = np.asarray(qml.metric_tensor(qnode_exp_aux)(p_autograd))
    g_block_pl = np.asarray(qml.metric_tensor(qnode_exp_aux, approx="block-diag")(p_autograd))

    err_full = float(np.max(np.abs(g_full_hand - g_full_pl)))
    err_block = float(np.max(np.abs(g_block_hand - g_block_pl)))
    print(f"  Full Metric Max Abs Error:       {err_full:.2e}")
    print(f"  Block-Diag Metric Max Abs Error: {err_block:.2e}")
    if err_full >= 1e-6 or err_block >= 1e-6:
        raise RuntimeError("Validation FAILED: Metric tensor error >= 1e-6")
    print("  [PASS] Both metric tensors match PennyLane reference (< 1e-6).")

    # 3. Variational principle across 200 random points
    print("\n3. Testing Variational Principle E(theta) >= E_0 across 200 random points:")
    violated = 0
    min_observed = float("inf")
    for _ in range(200):
        p_rand = rng.uniform(0.0, 2.0 * np.pi, d)
        e_val = float(qnode_exp(p_rand))
        min_observed = min(min_observed, e_val)
        if e_val < E0 - 1e-9:
            violated += 1

    print(f"  Exact ground energy E_0: {E0:.6f}")
    print(f"  Minimum observed E:     {min_observed:.6f} (Gap: {min_observed - E0:.6f})")
    print(f"  Violations (E < E0):     {violated}")
    if violated > 0:
        raise RuntimeError(f"Validation FAILED: Variational principle violated {violated} times!")
    print("  [PASS] Variational principle confirmed: E(theta) >= E_0 strictly holds.")

    # 1b. Verify fast numpy backend against PennyLane (5 random points, n=3, L=2)
    print("\n1b. Verifying fast numpy backend against PennyLane (5 random points, n=3, L=2):")
    for i_pt in range(5):
        p_test = rng.uniform(0.1, 2 * np.pi, d)
        # Energy
        e_pl = float(qnode_exp(p_test))
        e_fast = energy_fast(p_test, dense_H, n, layers)
        # Gradient
        g_pl_vec = param_shift_grad_vqe(qnode_exp, p_test)
        g_fast_vec = param_shift_grad_fast(p_test, dense_H, n, layers)
        # Metric tensor
        m_pl = metric_tensor_vqe(p_test, n, layers, approx=None, dev_state=dev)
        m_fast = metric_tensor_fast(p_test, n, layers)

        diff_e = abs(e_pl - e_fast)
        diff_g = float(np.max(np.abs(g_pl_vec - g_fast_vec)))
        diff_m = float(np.max(np.abs(m_pl - m_fast)))
        print(f"  Point {i_pt+1}: |dE|={diff_e:.2e}  |dg|_max={diff_g:.2e}  |dm|_max={diff_m:.2e}")
        if diff_e >= 1e-8 or diff_g >= 1e-8 or diff_m >= 1e-8:
            raise RuntimeError(
                f"Fast backend validation FAILED on point {i_pt}: "
                f"|dE|={diff_e:.2e}, |dg|={diff_g:.2e}, |dm|={diff_m:.2e}"
            )
    print("  [PASS] Fast backend matches PennyLane to < 1e-8 on all 5 points.")

    # Also verify for n=4, L=3 (the actual benchmark size)
    print("\n1c. Verifying fast numpy backend for n=4, L=3 (benchmark size, 3 random points):")
    n4, l3 = 4, 3
    d4 = ansatz_param_count(n4, l3)
    H4 = ising_chain_hamiltonian(n4, J=1.0, h=1.0)
    dense_H4 = build_dense_hamiltonian(H4, n4)
    dev4 = qml.device("default.qubit", wires=n4)
    qnode4 = make_vqe_qnode(H4, n4, l3, dev=dev4)
    for i_pt in range(3):
        p_test = rng.uniform(0.1, 2 * np.pi, d4)
        e_pl = float(qnode4(p_test))
        e_fast = energy_fast(p_test, dense_H4, n4, l3)
        g_fast_vec = param_shift_grad_fast(p_test, dense_H4, n4, l3)
        g_pl_vec = param_shift_grad_vqe(qnode4, p_test)
        m_fast = metric_tensor_fast(p_test, n4, l3)
        m_pl = metric_tensor_vqe(p_test, n4, l3, approx=None, dev_state=dev4)
        diff_e = abs(e_pl - e_fast)
        diff_g = float(np.max(np.abs(g_pl_vec - g_fast_vec)))
        diff_m = float(np.max(np.abs(m_pl - m_fast)))
        print(f"  Point {i_pt+1}: |dE|={diff_e:.2e}  |dg|_max={diff_g:.2e}  |dm|_max={diff_m:.2e}")
        if diff_e >= 1e-8 or diff_g >= 1e-8 or diff_m >= 1e-8:
            raise RuntimeError(
                f"Fast backend validation FAILED (n=4,L=3) on point {i_pt}: "
                f"|dE|={diff_e:.2e}, |dg|={diff_g:.2e}, |dm|={diff_m:.2e}"
            )
    print("  [PASS] Fast backend matches PennyLane to < 1e-8 on n=4, L=3 points.")


# ===========================================================================
# Part (b): VQE Benchmarking on 4-Qubit Spin Chains
#
# NOTE: Circuit evaluation counts are a cost model, not wall-clock time.
#       One "circuit evaluation" = one statevector evolution through the full ansatz.
#       The fast numpy backend is used for all optimization sweeps; the count
#       accounting is identical to the original PennyLane-based counting.
# ===========================================================================

def tune_learning_rates(
    H: qml.Hamiltonian,
    dense_H: np.ndarray,
    n: int,
    layers: int,
    gd_lr_grid: List[float],
    qng_lr_grid: List[float],
    qng_lam_grid: List[float],
) -> Tuple[float, float, float]:
    """Small grid tuning of learning rates for GD and QNG on start 0, using fast backend."""
    d = ansatz_param_count(n, layers)
    p0 = np.random.default_rng(100).uniform(0.0, 2.0 * np.pi, d)

    # Tune GD lr (25 short iterations)
    best_gd_lr = gd_lr_grid[0]
    best_gd_e = float("inf")
    for lr in gd_lr_grid:
        p = p0.copy()
        for _ in range(25):
            g = param_shift_grad_fast(p, dense_H, n, layers)
            p -= lr * g
        e = energy_fast(p, dense_H, n, layers)
        if e < best_gd_e:
            best_gd_e = e
            best_gd_lr = lr

    # Tune QNG (lr and regularisation lambda, 20 short iterations)
    best_qng_lr = qng_lr_grid[0]
    best_qng_lam = qng_lam_grid[0]
    best_qng_e = float("inf")
    for lr in qng_lr_grid:
        for lam in qng_lam_grid:
            p = p0.copy()
            for _ in range(20):
                g = param_shift_grad_fast(p, dense_H, n, layers)
                mt = metric_tensor_fast(p, n, layers)
                step = np.linalg.solve(mt + lam * np.eye(d), g)
                p -= lr * step
            e = energy_fast(p, dense_H, n, layers)
            if e < best_qng_e:
                best_qng_e = e
                best_qng_lr = lr
                best_qng_lam = lam

    return best_gd_lr, best_qng_lr, best_qng_lam


def run_vqe_benchmarks(
    n_starts: int,
    n_iters: int,
    gd_lr_grid: List[float],
    qng_lr_grid: List[float],
    qng_lam_grid: List[float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("EXP 4: PART (B) - VQE & QNG Optimization Benchmarks (Fast numpy backend)")
    print("=" * 80)

    n = 4
    problems = [
        ("TFIM",       2, ising_chain_hamiltonian(n, J=1.0, h=1.0)),
        ("TFIM",       3, ising_chain_hamiltonian(n, J=1.0, h=1.0)),
        ("Heisenberg", 2, heisenberg_chain_hamiltonian(n)),
        ("Heisenberg", 3, heisenberg_chain_hamiltonian(n)),
    ]

    tol = 1e-2

    all_records: List[Dict[str, Any]] = []
    convergence_histories: Dict[str, Any] = {}

    for prob_name, layers, H in problems:
        E0 = exact_ground_energy(H, n)
        d = ansatz_param_count(n, layers)
        prob_id = f"{prob_name}_L{layers}"
        # Build dense Hamiltonian once per problem (used by fast backend)
        dense_H = build_dense_hamiltonian(H, n)

        print(f"\n--- Problem: {prob_id} (n={n}, layers={layers}, d={d}, E_0={E0:.6f}) ---")

        # Circuit evaluation accounting note (cost model, not wall-clock):
        # - Parameter-shift gradient: 2 * d circuit evaluations per step.
        # - QNG step: 2 * d (gradient) + d (metric tensor pi-shifts) = 3 * d evals.
        # - COBYLA budget: same number of evaluations as GD uses over n_iters iters.
        cobyla_budget = n_iters * 2 * d

        gd_lr, qng_lr, qng_lam = tune_learning_rates(
            H, dense_H, n, layers, gd_lr_grid, qng_lr_grid, qng_lam_grid
        )
        print(f"  Tuned parameters: GD lr={gd_lr}, QNG lr={qng_lr}, QNG lambda={qng_lam}")

        optimizers = ["GD", "Adam", "QNG", "COBYLA"]
        for opt in optimizers:
            for s_idx in range(n_starts):
                seed = s_idx * 100 + 42
                rng = np.random.default_rng(seed)
                x0 = rng.uniform(0.0, 2.0 * np.pi, d)

                energy_history: List[float] = []
                evals_history: List[int] = []
                total_evals = 0

                t0 = time.perf_counter()

                if opt == "GD":
                    p = x0.copy()
                    for it in range(n_iters):
                        e_cur = energy_fast(p, dense_H, n, layers)
                        energy_history.append(e_cur)
                        evals_history.append(total_evals)
                        grad = param_shift_grad_fast(p, dense_H, n, layers)
                        total_evals += 2 * d   # cost model: 2*d per param-shift grad
                        p -= gd_lr * grad
                    final_e = energy_fast(p, dense_H, n, layers)

                elif opt == "Adam":
                    p = x0.copy()
                    m = np.zeros(d, dtype=float)
                    v = np.zeros(d, dtype=float)
                    adam_lr = 0.05
                    for it in range(n_iters):
                        e_cur = energy_fast(p, dense_H, n, layers)
                        energy_history.append(e_cur)
                        evals_history.append(total_evals)
                        grad = param_shift_grad_fast(p, dense_H, n, layers)
                        total_evals += 2 * d   # cost model: 2*d per param-shift grad
                        t_step = it + 1
                        m = 0.9 * m + 0.1 * grad
                        v = 0.999 * v + 0.001 * (grad ** 2)
                        m_hat = m / (1.0 - 0.9 ** t_step)
                        v_hat = v / (1.0 - 0.999 ** t_step)
                        p -= adam_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    final_e = energy_fast(p, dense_H, n, layers)

                elif opt == "QNG":
                    p = x0.copy()
                    for it in range(n_iters):
                        e_cur = energy_fast(p, dense_H, n, layers)
                        energy_history.append(e_cur)
                        evals_history.append(total_evals)
                        grad = param_shift_grad_fast(p, dense_H, n, layers)
                        mt = metric_tensor_fast(p, n, layers)
                        # Cost model: 2*d for grad + d for metric pi-shifts = 3*d per step
                        total_evals += 3 * d
                        step = np.linalg.solve(mt + qng_lam * np.eye(d), grad)
                        p -= qng_lr * step
                    final_e = energy_fast(p, dense_H, n, layers)

                elif opt == "COBYLA":
                    eval_cnt = 0
                    cob_best_e = float("inf")

                    def cobyla_cost(p_cand):
                        nonlocal eval_cnt, cob_best_e
                        val = energy_fast(p_cand, dense_H, n, layers)
                        eval_cnt += 1
                        if val < cob_best_e:
                            cob_best_e = val
                        # Sample history every 2*d evals to align with iteration curves
                        if eval_cnt % (2 * d) == 0 or eval_cnt == 1:
                            energy_history.append(cob_best_e)
                            evals_history.append(eval_cnt)
                        return val

                    res_cob = minimize(
                        cobyla_cost,
                        x0.copy(),
                        method="COBYLA",
                        options={"maxiter": cobyla_budget, "rhobeg": 0.4},
                    )
                    final_e = energy_fast(res_cob.x, dense_H, n, layers)
                    total_evals = eval_cnt

                elapsed = time.perf_counter() - t0
                err_final = max(0.0, final_e - E0)

                # Iterations to reach error < tol
                iters_to_tol = n_iters
                for it_idx, e_val in enumerate(energy_history):
                    if (e_val - E0) < tol:
                        iters_to_tol = it_idx
                        break

                # Stalled check: if final error is > 0.20
                stalled = err_final > 0.20

                all_records.append({
                    "problem":      prob_id,
                    "layers":       layers,
                    "optimizer":    opt,
                    "start":        s_idx,
                    "final_energy": final_e,
                    "ground_energy": E0,
                    "energy_error": err_final,
                    "iters_to_tol": iters_to_tol,
                    "circuit_evals": total_evals,
                    "stalled":      stalled,
                    "runtime_s":    elapsed,
                })

                # Store history of start 0 for plotting
                if s_idx == 0:
                    convergence_histories[f"{prob_id}_{opt}"] = {
                        "iters":  list(range(len(energy_history))),
                        "evals":  evals_history,
                        "errors": [max(1e-6, e - E0) for e in energy_history],
                    }

                # Progress output
                print(
                    f"  [{prob_id}] {opt} start {s_idx+1}/{n_starts}: "
                    f"final_err={err_final:.4f}, evals={total_evals}, time={elapsed:.2f}s",
                    flush=True,
                )

    df = pd.DataFrame(all_records)
    csv_path = RESULTS_DIR / "trackC_vqe.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved VQE benchmark records to {csv_path}")

    # ===========================================================================
    # Summary Tables
    # ===========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE 1: Equal Iterations Comparison (n_iters iters / Equal Evals Budget)")
    print("=" * 80)
    summary_iters = df.groupby(["problem", "optimizer"]).agg(
        mean_error=("energy_error", "mean"),
        std_error=("energy_error", "std"),
        mean_iters_tol=("iters_to_tol", "mean"),
        mean_evals=("circuit_evals", "mean"),
        stalled_runs=("stalled", "sum"),
    ).reset_index()
    print(summary_iters.to_string(index=False))

    return df, convergence_histories


# ===========================================================================
# Plotting
# ===========================================================================

def generate_vqe_plots(convergence_histories: Dict[str, Any]):
    print("\nGenerating VQE convergence figures...")
    problems = ["TFIM_L2", "TFIM_L3", "Heisenberg_L2", "Heisenberg_L3"]
    opts = ["GD", "Adam", "QNG", "COBYLA"]
    colors = {"GD": "#1f77b4", "Adam": "#ff7f0e", "QNG": "#2ca02c", "COBYLA": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, prob in enumerate(problems):
        ax = axes[idx]
        for opt in opts:
            key = f"{prob}_{opt}"
            if key in convergence_histories:
                hist = convergence_histories[key]
                ax.plot(hist["iters"], hist["errors"], label=opt, color=colors[opt], linewidth=2)
        ax.set_yscale("log")
        ax.set_title(f"Convergence: {prob}", fontsize=12)
        ax.set_xlabel("Iterations", fontsize=10)
        ax.set_ylabel(r"Energy Error ($E - E_0$)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("VQE Optimization Convergence across Spin Chains (SIMULATED)", fontsize=14, y=0.99)
    plt.tight_layout()
    out_path = RESULTS_DIR / "trackC_exp4_vqe_convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence plot to {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    print("=" * 80)
    print("TRACK C: EXP 4 - VARIATIONAL QUANTUM EIGENSOLVER & QNG")
    print("NOTE: All results are SIMULATED on classical computers.")
    if args.quick:
        print("MODE: --quick (3 starts, 50 iters, small grids)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Tuning grids
    # ------------------------------------------------------------------
    if args.quick:
        n_starts   = 3
        n_iters    = 50
        gd_lr_grid  = [0.05, 0.1]
        qng_lr_grid = [0.05, 0.1]
        qng_lam_grid = [0.01]
    else:
        n_starts   = 5
        n_iters    = 100
        gd_lr_grid  = [0.02, 0.05, 0.1, 0.2]
        qng_lr_grid = [0.03, 0.1]
        qng_lam_grid = [0.001, 0.05]

    # ------------------------------------------------------------------
    # Step 0: measure fast backend timing and estimate runtime
    # ------------------------------------------------------------------
    print("\nStep 0: Measuring fast numpy backend timings (n=4, L=3)...")
    t_energy_ms, t_grad_ms, t_metric_ms = measure_fast_backend_timing(n=4, layers=3)
    est_s = print_timing_and_estimate(
        t_energy_ms, t_grad_ms, t_metric_ms,
        n_iters, n_starts, gd_lr_grid, qng_lr_grid, qng_lam_grid,
    )

    if not args.quick and est_s > 600:
        # Reduce QNG grid further to stay under 10 min
        qng_lr_grid  = [0.05]
        qng_lam_grid = [0.01]
        print(f"  [NOTE] Estimated runtime {est_s:.0f}s > 600s — reducing QNG grid to "
              f"lr={qng_lr_grid}, lam={qng_lam_grid} to keep run under 10 min.")
        est_s = print_timing_and_estimate(
            t_energy_ms, t_grad_ms, t_metric_ms,
            n_iters, n_starts, gd_lr_grid, qng_lr_grid, qng_lam_grid,
        )

    # ------------------------------------------------------------------
    # Step 1: Validation (PennyLane reference vs fast backend)
    # ------------------------------------------------------------------
    run_validation()

    # ------------------------------------------------------------------
    # Step 2: Benchmarks using fast backend
    # ------------------------------------------------------------------
    df, histories = run_vqe_benchmarks(
        n_starts=n_starts,
        n_iters=n_iters,
        gd_lr_grid=gd_lr_grid,
        qng_lr_grid=qng_lr_grid,
        qng_lam_grid=qng_lam_grid,
    )
    generate_vqe_plots(histories)
    print("\nExp 4 Completed successfully.")


if __name__ == "__main__":
    main()
