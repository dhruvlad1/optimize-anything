"""
trackA_continuous/exp4_noisy.py
---------------------------------
Noisy-gradient and finite-sum experiments.

(i)  Noisy quadratic: SGD constant, SGD decay (linear & sqrt), Momentum, Adam under 3 noise levels.
(ii) Finite-sum least squares (n=2000, d=20): GD, SGD(B=1), mini-batch(B=8,64), Adam(mini-batch).
     Compared at EQUAL numbers of sample-gradient evaluations (10n, 50n, 200n checkpoints).
     10 seeds; mean ± std of final gap; shaded convergence bands.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackA_continuous.objectives import quadratic

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

N_SEEDS = 10
MAX_GE = 2000      # gradient evaluations budget for noisy quadratic
N_SAMPLES = 2000
N_FEATURES = 20
SEED_BASE = 0


def load_lr(opt_name: str, obj_key: str = "quadratic") -> float:
    lr_path = RESULTS / "trackA_tuned_lr.json"
    if not lr_path.exists():
        raise FileNotFoundError("Run exp3_sensitivity.py first.")
    with open(lr_path) as fh:
        d = json.load(fh)
    return float(d.get(opt_name, {}).get(obj_key, 0.01))


# ---------------------------------------------------------------------------
# Item 2: Inspection of the previous sgd_decay schedule
# ---------------------------------------------------------------------------

def inspect_sgd_decay_schedule():
    obj = quadratic(kappa=10, dim=10)
    x0 = obj.x_star + np.ones(10)
    start_gap = float(obj.f(x0) - (obj.f_star or 0.0))
    lr0 = 0.015199110829529332
    decay = 1.0
    schedule = [lr0 / (1.0 + decay * k) for k in range(MAX_GE)]

    print("\n=== Item 2: Inspection of Previous sgd_decay Schedule ===")
    print(f"Starting gap: {start_gap:.4f}")
    print(f"First 5 learning rates: {[round(s, 6) for s in schedule[:5]]}")
    print(f"Learning rate at iteration 100:  {schedule[100]:.6e}")
    print(f"Learning rate at iteration 1000: {schedule[1000]:.6e}")
    print(f"Learning rate at last iter ({MAX_GE-1}): {schedule[-1]:.6e}")
    print("Explanation of failure:")
    print("  With decay = 1.0 (lr_k = lr0 / (1 + k)), the step size decays harmonically (sum of steps grows only like log k),")
    print("  falling from ~0.015 to 0.00015 by iteration 100 and to 1.5e-5 by iteration 1000.")
    print("  The cumulative step budget sum_{k=0}^{1999} lr_k is only ~0.124, which is far too small to traverse the initial distance to the minimum.")
    print("  Consequently, the optimizer 'freezes' almost immediately after a handful of steps, leaving the objective gap stalled near ~5.81.\n")


# ---------------------------------------------------------------------------
# (i) Noisy quadratic
# ---------------------------------------------------------------------------

def noisy_quad_experiment():
    obj = quadratic(kappa=10, dim=10)
    f_star = 0.0
    x0 = obj.x_star + np.ones(10)
    start_gap = float(obj.f(x0) - f_star)
    noise_levels = [0.01, 0.1, 1.0]

    # Tuned learning rates for the quadratic
    lr_const = load_lr("gd", "quadratic")
    lr_mom = load_lr("momentum", "quadratic")
    lr_adam = load_lr("adam", "quadratic")

    # Tuned schedules:
    # 1) lr_k = lr0 / (1 + k / tau) with lr0 = 0.05, tau = 500
    lr0_tau, tau = 0.05, 500.0
    # 2) lr_k = lr0 / sqrt(1 + k) with lr0 = 0.1
    lr0_sqrt = 0.1

    methods = [
        "sgd_const",
        "sgd_decay_tau",
        "sgd_decay_sqrt",
        "momentum",
        "adam",
    ]

    all_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax_i, noise_std in enumerate(noise_levels):
        ax = axes[ax_i]
        ax.set_title(f"Noisy Quadratic (σ={noise_std})")
        ax.set_xlabel("Gradient evaluations")
        ax.set_ylabel("Gap f(x) - f* (log scale)")
        ax.set_yscale("log")

        for method_name in methods:
            seed_curves = []
            final_vals = []

            for seed in range(N_SEEDS):
                rng = np.random.default_rng(SEED_BASE + seed)
                x = x0.copy().astype(float)
                m = np.zeros_like(x)
                v = np.zeros_like(x)
                fvals = [obj.f(x)]

                for k in range(MAX_GE):
                    g = obj.grad(x) + rng.normal(0, noise_std, size=x.shape)

                    if method_name == "sgd_const":
                        x = x - lr_const * g
                    elif method_name == "sgd_decay_tau":
                        step = lr0_tau / (1.0 + k / tau)
                        x = x - step * g
                    elif method_name == "sgd_decay_sqrt":
                        step = lr0_sqrt / np.sqrt(1.0 + k)
                        x = x - step * g
                    elif method_name == "momentum":
                        m = 0.9 * m + g
                        x = x - lr_mom * m
                    elif method_name == "adam":
                        b1, b2, eps = 0.9, 0.999, 1e-8
                        m = b1 * m + (1.0 - b1) * g
                        v = b2 * v + (1.0 - b2) * (g ** 2)
                        m_hat = m / (1.0 - b1 ** (k + 1))
                        v_hat = v / (1.0 - b2 ** (k + 1))
                        x = x - lr_adam * m_hat / (np.sqrt(v_hat) + eps)

                    fvals.append(obj.f(x))

                gaps = np.maximum(np.array(fvals) - f_star, 1e-25)
                seed_curves.append(gaps)
                final_vals.append(gaps[-1])

            mat = np.array(seed_curves)
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            ge_axis = np.arange(len(mean))

            ax.plot(ge_axis, mean, label=method_name)
            ax.fill_between(ge_axis, np.maximum(mean - std, 1e-25), mean + std, alpha=0.15)

            all_rows.append({
                "noise_std": noise_std,
                "method": method_name,
                "starting_gap": start_gap,
                "mean_final_gap": float(np.mean(final_vals)),
                "std_final_gap": float(np.std(final_vals)),
            })

        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    plt.suptitle("Noisy Quadratic Optimization – Mean Gap & Std Band across 10 Seeds", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS / "trackA_exp4_noisy_quad.png", dpi=120)
    plt.close()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# (ii) Finite-sum least squares
# ---------------------------------------------------------------------------

def finite_sum_experiment():
    """
    Finite-sum least squares: f(x) = (1/n) sum_{i=1}^n 0.5*(a_i^T x - b_i)^2
    n = 2000 samples, d = 20 features.
    Compare at EQUAL numbers of sample-gradient evaluations:
    Checkpoints at 10n (20,000), 50n (100,000), 200n (400,000) sample gradients.
    """
    rng_data = np.random.default_rng(99)
    A = rng_data.standard_normal((N_SAMPLES, N_FEATURES))
    x_true = rng_data.standard_normal(N_FEATURES)
    b = A @ x_true + 0.01 * rng_data.standard_normal(N_SAMPLES)

    # Compute exact least-squares minimizer on this data
    x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    def f_full(x):
        r = A @ x - b
        return float(0.5 * np.mean(r ** 2))

    def grad_full(x):
        r = A @ x - b
        return A.T @ r / N_SAMPLES

    f_star = f_full(x_lstsq)
    grad_at_star = grad_full(x_lstsq)
    norm_grad_star = float(np.linalg.norm(grad_at_star))

    print("=== Item 5: Finite-Sum Least Squares Ground Truth ===")
    print(f"Optimal objective value f*(lstsq):     {f_star:.10e}")
    print(f"Gradient norm at lstsq solution:        {norm_grad_star:.10e}")
    print(f"Confirmation: Reported gaps are strictly f(x) - f*(lstsq), NOT relative to generating parameters.\n")

    # Small grid tuning for each method on this problem:
    # GD: 0.5
    # SGD (B=1): 0.005
    # MB (B=8): 0.02
    # MB (B=64): 0.1
    # Adam (B=8): 0.005
    methods_config = [
        {"name": "GD",             "bsz": N_SAMPLES, "lr": 0.500},
        {"name": "SGD (batch=1)",   "bsz": 1,         "lr": 0.005},
        {"name": "MB-SGD (batch=8)","bsz": 8,         "lr": 0.020},
        {"name": "MB-SGD (batch=64)","bsz": 64,       "lr": 0.100},
        {"name": "Adam (batch=8)",  "bsz": 8,         "lr": 0.005},
    ]

    total_sample_budget = 200 * N_SAMPLES  # 200n = 400,000
    checkpoints = [10 * N_SAMPLES, 50 * N_SAMPLES, 200 * N_SAMPLES]

    table_rows = []
    curves_data = {}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("Sample-gradient evaluations")
    ax.set_ylabel("Gap f(x) - f*(lstsq) (log scale)")
    ax.set_yscale("log")
    ax.set_title("Finite-Sum Least Squares – Equal Sample-Gradient Budget Comparison")

    common_ge = np.linspace(0, total_sample_budget, 400)

    for cfg in methods_config:
        name = cfg["name"]
        bsz = cfg["bsz"]
        lr = cfg["lr"]

        n_iters = total_sample_budget // bsz
        cp_iters = {cp: cp // bsz for cp in checkpoints}

        seed_cp_gaps = {cp: [] for cp in checkpoints}
        seed_curves_interp = []

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            x = np.zeros(N_FEATURES)
            m = np.zeros(N_FEATURES)
            v = np.zeros(N_FEATURES)

            iter_ge = [0]
            iter_gaps = [max(f_full(x) - f_star, 1e-25)]

            for it in range(1, n_iters + 1):
                if bsz == N_SAMPLES:
                    g = grad_full(x)
                    x = x - lr * g
                elif "Adam" in name:
                    idx = rng.integers(0, N_SAMPLES, size=bsz)
                    Ai, bi = A[idx], b[idx]
                    g = Ai.T @ (Ai @ x - bi) / bsz
                    b1, b2, eps = 0.9, 0.999, 1e-8
                    m = b1 * m + (1.0 - b1) * g
                    v = b2 * v + (1.0 - b2) * (g ** 2)
                    mh = m / (1.0 - b1 ** it)
                    vh = v / (1.0 - b2 ** it)
                    x = x - lr * mh / (np.sqrt(vh) + eps)
                else:
                    idx = rng.integers(0, N_SAMPLES, size=bsz)
                    Ai, bi = A[idx], b[idx]
                    g = Ai.T @ (Ai @ x - bi) / bsz
                    x = x - lr * g

                # Record checkpoints
                for cp, cp_it in cp_iters.items():
                    if it == cp_it:
                        gap_val = max(f_full(x) - f_star, 1e-25)
                        seed_cp_gaps[cp].append(gap_val)

                # Track curve points
                if it % max(1, n_iters // 200) == 0 or it == n_iters:
                    iter_ge.append(it * bsz)
                    iter_gaps.append(max(f_full(x) - f_star, 1e-25))

            # Interpolate for plotting
            interp_gap = np.interp(common_ge, iter_ge, iter_gaps)
            seed_curves_interp.append(interp_gap)

        # Compute checkpoint stats
        cp_stats = {}
        for cp in checkpoints:
            vals = seed_cp_gaps[cp]
            cp_stats[cp] = (float(np.mean(vals)), float(np.std(vals)))

        table_rows.append({
            "Method": name,
            "Batch Size": bsz,
            "Tuned LR": lr,
            "Mean Gap @ 10n": cp_stats[10 * N_SAMPLES][0],
            "Std Gap @ 10n":  cp_stats[10 * N_SAMPLES][1],
            "Mean Gap @ 50n": cp_stats[50 * N_SAMPLES][0],
            "Std Gap @ 50n":  cp_stats[50 * N_SAMPLES][1],
            "Mean Gap @ 200n": cp_stats[200 * N_SAMPLES][0],
            "Std Gap @ 200n":  cp_stats[200 * N_SAMPLES][1],
        })

        mat = np.array(seed_curves_interp)
        mean_curve = mat.mean(axis=0)
        std_curve = mat.std(axis=0)
        ax.plot(common_ge, mean_curve, label=f"{name} (lr={lr})")
        ax.fill_between(common_ge, np.maximum(mean_curve - std_curve, 1e-25),
                        mean_curve + std_curve, alpha=0.15)

    ax.axvline(10 * N_SAMPLES, color="gray", linestyle="--", alpha=0.5, label="10n Checkpoint")
    ax.axvline(50 * N_SAMPLES, color="gray", linestyle=":", alpha=0.5, label="50n Checkpoint")
    ax.axvline(200 * N_SAMPLES, color="gray", linestyle="-.", alpha=0.5, label="200n Checkpoint")

    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "trackA_exp4_finitesum.png", dpi=120)
    plt.close()

    return pd.DataFrame(table_rows), norm_grad_star


def main() -> None:
    # Item 2 inspection
    inspect_sgd_decay_schedule()

    print("=== (i) Noisy Quadratic Optimization ===")
    df_noisy = noisy_quad_experiment()
    print(df_noisy.to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    if df_noisy["mean_final_gap"].isna().any():
        sys.exit("ERROR: NaN found in noisy-quadratic results where divergence was not expected.")

    print("\n=== (ii) Finite-Sum Least Squares (Checkpoints @ 10n, 50n, 200n) ===")
    df_fs, norm_grad = finite_sum_experiment()
    cols_display = [
        "Method", "Batch Size", "Tuned LR",
        "Mean Gap @ 10n", "Mean Gap @ 50n", "Mean Gap @ 200n"
    ]
    print(df_fs[cols_display].to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    if df_fs["Mean Gap @ 200n"].isna().any():
        sys.exit("ERROR: NaN found in finite-sum results where divergence was not expected.")

    df_noisy.to_csv(RESULTS / "trackA_exp4_noisy.csv", index=False)
    df_fs.to_csv(RESULTS / "trackA_exp4_finitesum.csv", index=False)

    print("\nSaved plots and CSV summaries to results/.")
    print("Note: Adam on mini-batches can get worse with more budget and that the cause has not been verified.\n")


if __name__ == "__main__":
    main()
