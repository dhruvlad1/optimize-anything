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
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackC_quantum_opt.optimizers_q import adam_ps, cobyla_wrapper, spsa
from trackC_quantum_opt.qaoa_sim import (
    _parse_edges,
    approximation_ratio,
    expected_cut,
    maxcut_diagonal,
    parse_params,
    probabilities,
    qaoa_state,
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


def _make_param_shift_grad_shots(n: int, edges: List[Any], diag: np.ndarray, shots: int):
    dim = 1 << n
    edge_list = _parse_edges(edges, None)
    indices = np.arange(dim, dtype=np.int64)
    edge_masks = []
    for u, v, w in edge_list:
        bu = (indices >> (n - 1 - u)) & 1
        bv = (indices >> (n - 1 - v)) & 1
        edge_masks.append((bu != bv).astype(np.float64))

    def grad_fn(params: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gammas, betas = parse_params(params)
        p = len(gammas)
        prefix_states = [np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)]
        for l in range(p - 1):
            psi_next = prefix_states[-1].copy()
            g = gammas[l]
            b = betas[l]
            if g != 0.0:
                psi_next *= np.exp(-1j * g * diag)
            if b != 0.0:
                c = np.cos(b)
                s = -1j * np.sin(b)
                for j in range(n):
                    sh = (1 << j, 2, 1 << (n - 1 - j))
                    p_sh = psi_next.reshape(sh)
                    psi_next = np.stack(
                        [c * p_sh[:, 0, :] + s * p_sh[:, 1, :], s * p_sh[:, 0, :] + c * p_sh[:, 1, :]],
                        axis=1,
                    ).reshape(-1)
            prefix_states.append(psi_next)

        def evolve_and_sample(psi_start, start_layer, layer_mod_phase, mod_qubit):
            psi = psi_start.copy()
            for l in range(start_layer, p):
                if l == start_layer and layer_mod_phase is not None:
                    psi *= np.exp(-1j * layer_mod_phase)
                else:
                    g = gammas[l]
                    if g != 0.0:
                        psi *= np.exp(-1j * g * diag)
                b = betas[l]
                for j in range(n):
                    bj = b
                    if l == start_layer and mod_qubit is not None and mod_qubit[0] == j:
                        bj += mod_qubit[1]
                    if bj != 0.0:
                        c = np.cos(bj)
                        s = -1j * np.sin(bj)
                        sh = (1 << j, 2, 1 << (n - 1 - j))
                        p_sh = psi.reshape(sh)
                        psi = np.stack(
                            [c * p_sh[:, 0, :] + s * p_sh[:, 1, :], s * p_sh[:, 0, :] + c * p_sh[:, 1, :]],
                            axis=1,
                        ).reshape(-1)
            probs = np.abs(psi) ** 2
            s_sum = np.sum(probs)
            if s_sum > 0:
                probs /= s_sum
            sampled = rng.choice(dim, size=shots, p=probs)
            return float(np.mean(diag[sampled]))

        grad_gamma = np.zeros(p, dtype=np.float64)
        grad_beta = np.zeros(p, dtype=np.float64)
        for l in range(p):
            psi_pre = prefix_states[l]
            base_phase = gammas[l] * diag
            for e_idx, (u, v, w) in enumerate(edge_list):
                mask = edge_masks[e_idx]
                f_plus = evolve_and_sample(psi_pre, l, base_phase + (np.pi / 2.0) * mask, None)
                f_minus = evolve_and_sample(psi_pre, l, base_phase - (np.pi / 2.0) * mask, None)
                grad_gamma[l] += w * (f_plus - f_minus) / 2.0
            for j in range(n):
                f_plus = evolve_and_sample(psi_pre, l, base_phase, (j, np.pi / 4.0))
                f_minus = evolve_and_sample(psi_pre, l, base_phase, (j, -np.pi / 4.0))
                grad_beta[l] += (f_plus - f_minus)
        return np.concatenate([grad_gamma, grad_beta])

    return grad_fn


def run_shot_based_optimization(inst: Dict[str, Any], diag: np.ndarray):
    print("\n" + "=" * 80)
    print("EXP 2: PART 2 - Optimization with Finite-Shot Evaluations (p=2)")
    print("=" * 80)

    n = inst["n"]
    edges = inst["edges"]
    opt_cut = float(inst["optimal_cut_value"])
    p = 2
    budget = 300
    shots_per_eval = 1000
    n_starts = 5
    dim = 1 << n

    # Compute reference level: mean exact ratio over 100 random starts
    rng_ref = np.random.default_rng(12345)
    random_ratios = []
    for _ in range(100):
        p_rand = np.concatenate([rng_ref.uniform(0, np.pi, p), rng_ref.uniform(0, np.pi / 2.0, p)])
        random_ratios.append(approximation_ratio(p_rand, diag, n, opt_cut))
    ref_mean = float(np.mean(random_ratios))
    ref_std = float(np.std(random_ratios))

    print(f"Reference Level (100 random parameter starts): {ref_mean:.4f} +/- {ref_std:.4f}")

    # Build shot-based parameter-shift gradient calculator
    shot_grad_fn = _make_param_shift_grad_shots(n, edges, diag, shots_per_eval)

    # Existing SPSA records (retained for backward compatibility)
    existing_spsa_records = []

    # Full multi-optimizer records
    opt_rows = []

    for start_idx in range(n_starts):
        rng = np.random.default_rng(start_idx * 101 + 42)
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2.0, p)])
        init_exact_ratio = approximation_ratio(x0, diag, n, opt_cut)

        # -------------------------------------------------------------------
        # 0. Noiseless SPSA Baseline
        # -------------------------------------------------------------------
        res_spsa_exact = spsa(
            lambda p_vec: -expected_cut(p_vec, diag, n),
            x0,
            budget=budget,
            seed=start_idx,
            a=0.2,
            c=0.1,
        )
        exact_ratio_noiseless = approximation_ratio(res_spsa_exact.x, diag, n, opt_cut)

        # -------------------------------------------------------------------
        # 1. 1000-Shot SPSA Arm
        # -------------------------------------------------------------------
        spsa_path_exact = [init_exact_ratio]

        def shot_noisy_objective_spsa(p_vec):
            spsa_path_exact.append(approximation_ratio(p_vec, diag, n, opt_cut))
            g, b = parse_params(p_vec)
            probs = np.abs(qaoa_state(g, b, diag, n)) ** 2
            s_sum = np.sum(probs)
            if s_sum > 0:
                probs /= s_sum
            samples = diag[rng.choice(dim, size=shots_per_eval, p=probs)]
            return -float(np.mean(samples))

        res_spsa_shot = spsa(
            shot_noisy_objective_spsa,
            x0,
            budget=budget,
            seed=start_idx,
            a=0.2,
            c=0.1,
        )
        spsa_final_exact = approximation_ratio(res_spsa_shot.x, diag, n, opt_cut)
        spsa_best_path = max(spsa_path_exact)

        existing_spsa_records.append({
            "start": start_idx,
            "ratio_noiseless_opt": exact_ratio_noiseless,
            "ratio_shot_opt": spsa_final_exact,
        })
        print(f"  Start {start_idx + 1}: Noiseless Opt Ratio={exact_ratio_noiseless:.4f}, 1k-Shot Opt Ratio={spsa_final_exact:.4f}")

        opt_rows.append({
            "optimizer": "SPSA",
            "start": start_idx + 1,
            "init_ratio": init_exact_ratio,
            "final_exact_ratio": spsa_final_exact,
            "best_path_exact_ratio": spsa_best_path,
            "nfev": res_spsa_shot.nfev,
            "total_shots": res_spsa_shot.nfev * shots_per_eval,
            "improved": spsa_final_exact > init_exact_ratio,
        })

        # -------------------------------------------------------------------
        # 2. 1000-Shot COBYLA Arm
        # -------------------------------------------------------------------
        cobyla_path_exact = [init_exact_ratio]

        def shot_noisy_objective_cobyla(p_vec):
            cobyla_path_exact.append(approximation_ratio(p_vec, diag, n, opt_cut))
            g, b = parse_params(p_vec)
            probs = np.abs(qaoa_state(g, b, diag, n)) ** 2
            s_sum = np.sum(probs)
            if s_sum > 0:
                probs /= s_sum
            samples = diag[rng.choice(dim, size=shots_per_eval, p=probs)]
            return -float(np.mean(samples))

        res_cobyla_shot = cobyla_wrapper(
            shot_noisy_objective_cobyla,
            x0,
            budget=budget,
            rhobeg=0.4,
        )
        cobyla_final_exact = approximation_ratio(res_cobyla_shot.x, diag, n, opt_cut)
        cobyla_best_path = max(cobyla_path_exact)

        opt_rows.append({
            "optimizer": "COBYLA",
            "start": start_idx + 1,
            "init_ratio": init_exact_ratio,
            "final_exact_ratio": cobyla_final_exact,
            "best_path_exact_ratio": cobyla_best_path,
            "nfev": res_cobyla_shot.nfev,
            "total_shots": res_cobyla_shot.nfev * shots_per_eval,
            "improved": cobyla_final_exact > init_exact_ratio,
        })

        # -------------------------------------------------------------------
        # 3. 1000-Shot Adam with Parameter-Shift Gradient Arm
        # -------------------------------------------------------------------
        adam_path_exact = [init_exact_ratio]

        def shot_loss_and_grad_adam(p_vec):
            adam_path_exact.append(approximation_ratio(p_vec, diag, n, opt_cut))
            g, b = parse_params(p_vec)
            probs = np.abs(qaoa_state(g, b, diag, n)) ** 2
            s_sum = np.sum(probs)
            if s_sum > 0:
                probs /= s_sum
            samples = diag[rng.choice(dim, size=shots_per_eval, p=probs)]
            f_shot = float(np.mean(samples))
            grad_shot = shot_grad_fn(p_vec, rng)
            return -f_shot, -grad_shot

        res_adam_shot = adam_ps(
            shot_loss_and_grad_adam,
            x0,
            budget=budget,
            lr=0.04,
        )
        adam_final_exact = approximation_ratio(res_adam_shot.x, diag, n, opt_cut)
        adam_best_path = max(adam_path_exact)

        opt_rows.append({
            "optimizer": "Adam",
            "start": start_idx + 1,
            "init_ratio": init_exact_ratio,
            "final_exact_ratio": adam_final_exact,
            "best_path_exact_ratio": adam_best_path,
            "nfev": res_adam_shot.nfev,
            "total_shots": res_adam_shot.nfev * shots_per_eval,
            "improved": adam_final_exact > init_exact_ratio,
        })

    df_existing_spsa = pd.DataFrame(existing_spsa_records)
    print(f"\nMean Exact Ratio with Noiseless SPSA: {df_existing_spsa['ratio_noiseless_opt'].mean():.4f}")
    print(f"Mean Exact Ratio with 1000-Shot SPSA: {df_existing_spsa['ratio_shot_opt'].mean():.4f}")

    df_all_opts = pd.DataFrame(opt_rows)

    # =======================================================================
    # Table 1: Final Exact Ratio, Evaluations, and Total Shots
    # =======================================================================
    print("\n" + "=" * 80)
    print("TABLE 1: Final True (Exact) Expected-Cut Ratio, Circuit Evals & Total Shots")
    print("=" * 80)
    print(f"{'Optimizer':<10} {'Start':<6} {'Final Exact Ratio':<18} {'Evals (nfev)':<14} {'Total Shots':<12}")
    print("-" * 64)
    for opt_name in ["SPSA", "COBYLA", "Adam"]:
        sub = df_all_opts[df_all_opts["optimizer"] == opt_name]
        for _, r in sub.iterrows():
            print(f"{r['optimizer']:<10} {int(r['start']):<6} {r['final_exact_ratio']:<18.4f} {int(r['nfev']):<14} {int(r['total_shots']):<12}")
        mean_ratio = sub["final_exact_ratio"].mean()
        mean_nfev = sub["nfev"].mean()
        mean_shots = sub["total_shots"].mean()
        print(f"{opt_name + ' (Mean)':<10} {'--':<6} {mean_ratio:<18.4f} {mean_nfev:<14.1f} {mean_shots:<12.0f}")
        print("-" * 64)

    # =======================================================================
    # Table 2: Best Exact Ratio over Whole Optimization Path
    # =======================================================================
    print("\n" + "=" * 80)
    print("TABLE 2: Best Exact Ratio Over Whole Optimization Path (vs Random Reference)")
    print("=" * 80)
    print(f"Reference Level (100 Random Starts): {ref_mean:.4f} +/- {ref_std:.4f}\n")
    print(f"{'Optimizer':<10} {'Start':<6} {'Initial Ratio':<15} {'Best Path Ratio':<17} {'Final Ratio':<14} {'Improved?':<10}")
    print("-" * 74)
    for opt_name in ["SPSA", "COBYLA", "Adam"]:
        sub = df_all_opts[df_all_opts["optimizer"] == opt_name]
        for _, r in sub.iterrows():
            improved_str = "YES" if r["improved"] else "NO (Degraded)"
            print(f"{r['optimizer']:<10} {int(r['start']):<6} {r['init_ratio']:<15.4f} {r['best_path_exact_ratio']:<17.4f} {r['final_exact_ratio']:<14.4f} {improved_str:<10}")
        mean_init = sub["init_ratio"].mean()
        mean_best = sub["best_path_exact_ratio"].mean()
        mean_final = sub["final_exact_ratio"].mean()
        print(f"{opt_name + ' (Mean)':<10} {'--':<6} {mean_init:<15.4f} {mean_best:<17.4f} {mean_final:<14.4f}")
        print("-" * 74)

    # Plainly state results at random-parameter level or where optimizer did not improve
    print("\n" + "=" * 80)
    print("ANALYSIS OF PERFORMANCE & PROGRESS UNDER 1000-SHOT NOISE:")
    print("=" * 80)
    for _, r in df_all_opts.iterrows():
        opt = r["optimizer"]
        s = int(r["start"])
        fin = r["final_exact_ratio"]
        init_r = r["init_ratio"]
        best_p = r["best_path_exact_ratio"]

        # Check if at random-parameter level
        is_random = abs(fin - ref_mean) <= ref_std
        # Check if did not improve
        no_improvement = fin <= init_r

        if no_improvement:
            print(f"  - [{opt} Start {s}]: DID NOT IMPROVE. Initial ratio = {init_r:.4f}, Final ratio = {fin:.4f} (Peak on path = {best_p:.4f}).")
        if is_random:
            print(f"  - [{opt} Start {s}]: AT RANDOM-PARAMETER LEVEL. Final ratio {fin:.4f} is within 1 std ({ref_std:.4f}) of random reference ({ref_mean:.4f}).")

    # Save to results/trackC_shots_opt.csv
    out_csv = RESULTS_DIR / "trackC_shots_opt.csv"
    df_all_opts.to_csv(out_csv, index=False)
    print(f"\nSaved finite-shot optimization results to {out_csv}")

    return df_all_opts


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
