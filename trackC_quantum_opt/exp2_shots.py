"""trackC_quantum_opt/exp2_shots.py
----------------------------------
Quantifying shot noise in QAOA expectation values and shot-based optimization.

NOTE: All quantum results in this study are SIMULATED on classical hardware.
No physical quantum processor was used.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackC_quantum_opt.optimizers_q import adam_ps, cobyla_wrapper, spsa
from trackC_quantum_opt.qaoa_sim import (
    approximation_ratio,
    expected_cut,
    maxcut_diagonal,
    probabilities,
    sample,
)

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
INSTANCES_FILE = RESULTS_DIR / "trackB_maxcut_instances.json"


def load_first_instance() -> Dict[str, Any]:
    with open(INSTANCES_FILE, "r", encoding="utf-8") as f:
        instances = json.load(f)
    for inst in instances:
        if inst["n"] == 10:
            return inst
    return instances[0]


def run_shot_noise_scaling(inst: Dict[str, Any]) -> Tuple[pd.DataFrame, float, np.ndarray, np.ndarray]:
    print("=" * 80)
    print("EXP 2: PART 1 - Finite Shot Noise Scaling Study (SIMULATED)")
    print("=" * 80)

    n = inst["n"]
    edges = inst["edges"]
    diag = maxcut_diagonal(n, edges)
    opt_cut = float(inst["optimal_cut_value"])

    # 1. Optimize p=1 parameters using noiseless COBYLA to get a fixed target state
    print(f"Target instance: {inst['instance_id']} (n={n}, opt_cut={opt_cut})")
    x0 = np.array([0.5, 0.4])
    res_cobyla = cobyla_wrapper(lambda p: -expected_cut(p, diag, n), x0, budget=150)
    best_params = res_cobyla.x

    exact_val = expected_cut(best_params, diag, n)
    print(f"Optimized p=1 parameters: gamma={best_params[0]:.4f}, beta={best_params[1]:.4f}")
    print(f"Exact expected cut <C>: {exact_val:.6f} (Ratio: {exact_val / opt_cut:.4f})")

    # 2. Shot sweep
    shot_levels = [100, 1000, 10000, 100000]
    n_reps = 200
    rows = []

    print(f"\nEvaluating empirical statistics across {n_reps} repetitions per shot level...")
    probs = probabilities(best_params, diag, n)
    dim = len(probs)

    rng = np.random.default_rng(2026)

    for shots in shot_levels:
        estimates = []
        for rep in range(n_reps):
            sampled_idx = rng.choice(dim, size=shots, p=probs)
            emp_mean = float(np.mean(diag[sampled_idx]))
            estimates.append(emp_mean)

        estimates = np.array(estimates)
        mean_est = float(np.mean(estimates))
        std_est = float(np.std(estimates, ddof=1))
        se_mean = std_est / np.sqrt(n_reps)
        bias = mean_est - exact_val

        rows.append({
            "shots": shots,
            "mean_est": mean_est,
            "std_est": std_est,
            "exact": exact_val,
            "bias": bias,
            "se_mean": se_mean,
            "unbiased_within_4se": abs(bias) < 4.0 * se_mean,
        })
        print(f"  Shots={shots:6d}: Mean={mean_est:.4f}, Std={std_est:.4f}, Bias={bias:+.4f}, Within 4*SE: {abs(bias) < 4.0 * se_mean}")

    df_shots = pd.DataFrame(rows)

    # Fit scaling law log(std) = slope * log(shots) + const
    log_s = np.log10(df_shots["shots"].values)
    log_std = np.log10(df_shots["std_est"].values)
    slope, intercept = np.polyfit(log_s, log_std, deg=1)
    print(f"\nLog-Log Empirical Scaling Slope: {slope:.4f} (Theoretical: -0.5000)")
    assert np.isclose(slope, -0.5, atol=0.08), f"Expected slope around -0.5, got {slope}"

    return df_shots, slope, best_params, diag


def run_shot_based_optimization(inst: Dict[str, Any], diag: np.ndarray):
    print("\n" + "=" * 80)
    print("EXP 2: PART 2 - Optimization with Finite-Shot Evaluations (p=2)")
    print("=" * 80)

    n = inst["n"]
    opt_cut = float(inst["optimal_cut_value"])
    p = 2
    budget = 300
    shots_per_eval = 1000
    n_starts = 5

    results = []

    for start_idx in range(n_starts):
        rng = np.random.default_rng(start_idx * 101 + 42)
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2.0, p)])

        # Noiseless SPSA baseline
        res_spsa_exact = spsa(
            lambda p_vec: -expected_cut(p_vec, diag, n),
            x0,
            budget=budget,
            seed=start_idx,
            a=0.2,
            c=0.1,
        )
        exact_ratio_noiseless = approximation_ratio(res_spsa_exact.x, diag, n, opt_cut)

        # Finite-shot SPSA
        def shot_noisy_objective(p_vec):
            samples = sample(p_vec, diag, n, shots=shots_per_eval, rng=rng)
            return -float(np.mean(samples))

        res_spsa_shot = spsa(
            shot_noisy_objective,
            x0,
            budget=budget,
            seed=start_idx,
            a=0.2,
            c=0.1,
        )
        # Evaluate final parameters EXACTLY
        exact_ratio_shot_opt = approximation_ratio(res_spsa_shot.x, diag, n, opt_cut)

        results.append({
            "start": start_idx,
            "ratio_noiseless_opt": exact_ratio_noiseless,
            "ratio_shot_opt": exact_ratio_shot_opt,
        })
        print(f"  Start {start_idx + 1}: Noiseless Opt Ratio={exact_ratio_noiseless:.4f}, 1k-Shot Opt Ratio={exact_ratio_shot_opt:.4f}")

    df_opt = pd.DataFrame(results)
    print(f"\nMean Exact Ratio with Noiseless SPSA: {df_opt['ratio_noiseless_opt'].mean():.4f}")
    print(f"Mean Exact Ratio with 1000-Shot SPSA: {df_opt['ratio_shot_opt'].mean():.4f}")
    return df_opt


def plot_shot_results(df_shots: pd.DataFrame, slope: float):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    shots = df_shots["shots"].values
    stds = df_shots["std_est"].values

    ax.scatter(shots, stds, color="#d62728", s=60, zorder=5, label=f"Empirical Std (slope={slope:.3f})")

    # Reference theoretical line
    ref_x = np.array([shots[0], shots[-1]])
    ref_y = stds[0] * np.sqrt(shots[0] / ref_x)
    ax.plot(ref_x, ref_y, "k--", alpha=0.7, label=r"Theory: $\propto 1/\sqrt{S}$ (slope=-0.5)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measurement Shots (S)", fontsize=11)
    ax.set_ylabel(r"Standard Deviation $\sigma_{\langle C \rangle}$", fontsize=11)
    ax.set_title("Shot Noise Scaling in QAOA (SIMULATED)", fontsize=13)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = RESULTS_DIR / "trackC_exp2_shots.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved shot scaling plot to {out_path}")


def main():
    print("=" * 80)
    print("TRACK C: EXP 2 - SHOT NOISE QUANTIFICATION")
    print("NOTE: All results are SIMULATED on classical computers.")
    print("=" * 80)

    inst = load_first_instance()
    df_shots, slope, _, diag = run_shot_noise_scaling(inst)
    csv_path = RESULTS_DIR / "trackC_shots.csv"
    df_shots.to_csv(csv_path, index=False)
    print(f"\nSaved shot noise statistics to {csv_path}")

    plot_shot_results(df_shots, slope)
    run_shot_based_optimization(inst, diag)
    print("\nExp 2 Completed successfully.")


if __name__ == "__main__":
    main()
