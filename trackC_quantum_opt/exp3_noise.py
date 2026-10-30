"""trackC_quantum_opt/exp3_noise.py
----------------------------------
Depolarizing noise experiment using PennyLane default.mixed density matrix simulator.

NOTE: All results in this experiment are SIMULATED using a depolarizing noise model
on a classical computer. No physical quantum processor was used.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml

from trackC_quantum_opt.optimizers_q import adam_ps, cobyla_wrapper
from trackC_quantum_opt.qaoa_sim import (
    approximation_ratio,
    expected_cut,
    maxcut_diagonal,
    param_shift_grad,
)

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def generate_8node_graphs() -> List[Dict[str, Any]]:
    graphs = []
    seeds = [42, 43, 44]
    n = 8
    for s in seeds:
        rng = np.random.default_rng(s)
        W_half = rng.integers(1, 4, (n, n))
        W = (W_half + W_half.T) // 2
        np.fill_diagonal(W, 0)
        edges = [(i, j, int(W[i, j])) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]
        diag = maxcut_diagonal(n, edges)
        opt_cut = float(np.max(diag))
        graphs.append({
            "graph_idx": len(graphs),
            "seed": s,
            "n": n,
            "edges": edges,
            "diag": diag,
            "opt_cut": opt_cut,
        })
    return graphs


def make_pennylane_circuit(n: int, edges: List[Tuple[int, int, int]], p: int):
    dev = qml.device("default.mixed", wires=n)
    coeffs = [w / 2.0 for _, _, w in edges] + [-w / 2.0 for _, _, w in edges]
    obs = [qml.Identity(0) for _ in edges] + [qml.PauliZ(u) @ qml.PauliZ(v) for u, v, _ in edges]
    cost_ham = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev)
    def circuit(gammas, betas, q):
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p):
            for u, v, w in edges:
                qml.IsingZZ(-gammas[l] * w, wires=[u, v])
            if q > 0:
                for i in range(n):
                    qml.DepolarizingChannel(q, wires=i)
            for i in range(n):
                qml.RX(2 * betas[l], wires=i)
            if q > 0:
                for i in range(n):
                    qml.DepolarizingChannel(q, wires=i)
        return qml.expval(cost_ham)

    return circuit


def run_noise_study(graphs: List[Dict[str, Any]]) -> pd.DataFrame:
    print("=" * 80)
    print("EXP 3: Depolarizing Noise Study on 8-Qubit Graphs (SIMULATED)")
    print("=" * 80)

    q_values = [0.0, 0.001, 0.01, 0.05]
    p_values = [1, 2]
    records = []

    for g_info in graphs:
        g_idx = g_info["graph_idx"]
        seed = g_info["seed"]
        n = g_info["n"]
        edges = g_info["edges"]
        diag = g_info["diag"]
        opt_cut = g_info["opt_cut"]

        print(f"\n--- Graph {g_idx + 1}/3 (Seed {seed}, n={n}, opt_cut={opt_cut:.1f}) ---")

        for p in p_values:
            # 1. Pre-optimize noiseless parameters using Adam-PS on fast NumPy sim
            print(f"  Optimizing noiseless baseline for p={p}...")
            best_noiseless_params = None
            best_noiseless_cut = -1.0

            for s_idx in range(3):
                rng = np.random.default_rng(seed * 10 + s_idx + p)
                x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2.0, p)])

                def loss_and_grad(params_vec):
                    val = -expected_cut(params_vec, diag, n)
                    grad = -param_shift_grad(params_vec, diag, n, edges)
                    return val, grad

                res = adam_ps(loss_and_grad, x0, budget=150, lr=0.05)
                cut_val = expected_cut(res.x, diag, n)
                if cut_val > best_noiseless_cut:
                    best_noiseless_cut = cut_val
                    best_noiseless_params = res.x

            noiseless_ratio = best_noiseless_cut / opt_cut
            print(f"  Noiseless p={p} best ratio: {noiseless_ratio:.4f}")

            circuit = make_pennylane_circuit(n, edges, p)
            g_opt = best_noiseless_params[:p]
            b_opt = best_noiseless_params[p:]

            for q in q_values:
                t0 = time.perf_counter()
                # Fixed parameters under noise
                noisy_exp = float(circuit(g_opt, b_opt, q))
                ratio_fixed = noisy_exp / opt_cut

                if q == 0.0:
                    assert np.isclose(noisy_exp, best_noiseless_cut, atol=1e-8)

                # Re-optimization under noise: only for p=1 and higher noise (q in [0.01, 0.05])
                ratio_reopt = np.nan
                if p == 1 and q in [0.01, 0.05]:
                    def noisy_loss(p_vec):
                        return -float(circuit(p_vec[:p], p_vec[p:], q))

                    res_noisy = cobyla_wrapper(noisy_loss, best_noiseless_params.copy(), budget=25, rhobeg=0.2)
                    reopt_cut = -res_noisy.fun
                    ratio_reopt = reopt_cut / opt_cut

                elapsed = time.perf_counter() - t0
                print(f"    p={p}, q={q:0.3f} | Fixed Ratio={ratio_fixed:.4f} | Reopt Ratio={ratio_reopt:.4f} ({elapsed:.2f} s)")

                records.append({
                    "graph_idx": g_idx,
                    "seed": seed,
                    "n": n,
                    "p": p,
                    "q": q,
                    "ratio_fixed": ratio_fixed,
                    "ratio_reopt": ratio_reopt,
                })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "trackC_noise.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved noise study results to {csv_path}")
    return df


def plot_noise_results(df_noise: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    summary = df_noise.groupby(["p", "q"])["ratio_fixed"].mean().reset_index()

    colors = {1: "#1f77b4", 2: "#ff7f0e"}
    markers = {1: "o", 2: "s"}

    for p in [1, 2]:
        sub = summary[summary["p"] == p].sort_values("q")
        ax.plot(
            sub["q"],
            sub["ratio_fixed"],
            label=f"p = {p} (Fixed Noiseless Angles)",
            color=colors[p],
            marker=markers[p],
            linewidth=2,
        )

    # Re-optimized points for p=1
    sub_reopt = df_noise[df_noise["p"] == 1].dropna(subset=["ratio_reopt"])
    if not sub_reopt.empty:
        reopt_summary = sub_reopt.groupby("q")["ratio_reopt"].mean().reset_index()
        ax.scatter(
            reopt_summary["q"],
            reopt_summary["ratio_reopt"],
            color="#2ca02c",
            marker="^",
            s=80,
            zorder=5,
            label="p = 1 (Re-optimized under noise)",
        )

    ax.set_xlabel("Depolarizing Error Rate per Gate (q)", fontsize=11)
    ax.set_ylabel("Approximation Ratio (<C> / C_opt)", fontsize=11)
    ax.set_title("QAOA Robustness to Depolarizing Noise (SIMULATED)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = RESULTS_DIR / "trackC_exp3_noise.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved noise plot to {out_path}")


def main():
    print("=" * 80)
    print("TRACK C: EXP 3 - DEPOLARIZING NOISE SIMULATION")
    print("NOTE: All results are SIMULATED on classical computers.")
    print("=" * 80)

    graphs = generate_8node_graphs()
    df_noise = run_noise_study(graphs)
    plot_noise_results(df_noise)
    print("\nExp 3 Completed successfully.")


if __name__ == "__main__":
    main()
