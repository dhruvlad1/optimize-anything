"""
trackA_continuous/exp1_convex.py
---------------------------------
Convex-function benchmark.

Objectives: quadratic(kappa=10), quadratic(kappa=100), quartic
Reads tuned LRs from results/trackA_tuned_lr.json (run exp3_sensitivity first).

Output:
  results/trackA_exp1_convex.csv
  results/trackA_exp1_convex_<obj>.png  (gap vs iteration, log-y)
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

from trackA_continuous.objectives import quadratic, quartic
from trackA_continuous.optimizers import gd, momentum, nag, adagrad, rmsprop, adam, newton

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

SEED = 42
MAX_ITER = 2000
GTOL = 1e-8
TARGET_GAP = 1e-6
N_STARTS = 10


def load_lr(opt_name: str, obj_key: str) -> float:
    lr_path = RESULTS / "trackA_tuned_lr.json"
    if not lr_path.exists():
        raise FileNotFoundError(
            "Run exp3_sensitivity.py first to generate results/trackA_tuned_lr.json"
        )
    with open(lr_path) as fh:
        d = json.load(fh)
    return float(d.get(opt_name, {}).get(obj_key, 0.01))


def iters_to_target(fvals, f_star, target=TARGET_GAP):
    gaps = np.array(fvals) - f_star
    hit = np.where(gaps <= target)[0]
    return int(hit[0]) if len(hit) else len(fvals)


def fit_gd_tail_rate(fvals: np.ndarray, f_star: float, kappa: float) -> tuple[float, float, tuple[int, int]]:
    """
    Fit empirical linear rate on the tail of GD on a quadratic:
    Tail window is defined as the last half of iterations before reaching 1e-12.
    Theoretical gap contraction rate is (1 - 1/kappa)^2.
    """
    gaps = np.array(fvals) - f_star
    hit = np.where(gaps <= 1e-12)[0]
    end_idx = int(hit[0]) if len(hit) > 0 else len(gaps)
    start_idx = end_idx // 2
    if end_idx - start_idx < 5:
        start_idx = max(0, end_idx - 10)

    ks = np.arange(start_idx, end_idx)
    log_gaps = np.log(np.maximum(gaps[ks], 1e-30))
    A = np.vstack([ks, np.ones_like(ks)]).T
    slope, _ = np.linalg.lstsq(A, log_gaps, rcond=None)[0]
    emp_rate = float(np.exp(slope))
    theory_rate = float((1.0 - 1.0 / kappa) ** 2)
    return emp_rate, theory_rate, (start_idx, end_idx)


def inspect_adagrad_artifact():
    obj_q10 = quadratic(kappa=10, dim=10)
    x0_ones = np.ones(10)
    lr = 1.0
    res = adagrad(obj_q10.f, obj_q10.grad, x0_ones, lr=lr, max_iter=2, gtol=0.0)
    x1 = res["path"][1]
    f1 = res["fvals"][1]
    print("\n=== Item 3: AdaGrad Artifact Inspection (quadratic kappa=10) ===")
    print("Start point x0 (all-ones):", x0_ones.tolist())
    print("Initial f(x0):", obj_q10.f(x0_ones))
    print("Gradient at x0:", obj_q10.grad(x0_ones).tolist())
    print("First AdaGrad step x1 with lr=1.0:", x1.tolist())
    print(f"Objective f(x1) after 1 step: {f1:.4e}")
    print("Explanation: Since g_0 = A * x0 > 0 component-wise, the coordinate update is:")
    print("  x_1 = x_0 - lr * g_0 / (sqrt(g_0^2) + eps) = 1.0 - lr * (1 - eps / g_0).")
    print("With lr = 1.0, x_1 = eps / g_0 ~ 1e-8 / lambda_i, which lands within ~1e-8 of the origin in exactly 1 step!")
    print("This confirms the 1-step convergence was an artifact of x0 = (1,...,1) with lr = 1.0.\n")


def run_single_start(obj, obj_key: str, x0: np.ndarray):
    """Run all optimizers on a single starting point."""
    res_dict = {}
    f_star = obj.f_star or 0.0

    # GD with 1/L step
    lr_gd = 1.0 / obj.L if obj.L is not None else load_lr("gd", obj_key)
    res_dict["gd"] = gd(obj.f, obj.grad, x0, lr=lr_gd, max_iter=MAX_ITER, gtol=GTOL)

    opts = {
        "momentum": lambda lr: momentum(obj.f, obj.grad, x0, lr=lr, beta=0.9, max_iter=MAX_ITER, gtol=GTOL),
        "nag":      lambda lr: nag(obj.f, obj.grad, x0, lr=lr, beta=0.9, max_iter=MAX_ITER, gtol=GTOL),
        "adagrad":  lambda lr: adagrad(obj.f, obj.grad, x0, lr=lr, max_iter=MAX_ITER, gtol=GTOL),
        "rmsprop":  lambda lr: rmsprop(obj.f, obj.grad, x0, lr=lr, max_iter=MAX_ITER, gtol=GTOL),
        "adam":     lambda lr: adam(obj.f, obj.grad, x0, lr=lr, max_iter=MAX_ITER, gtol=GTOL),
    }
    for name, fn in opts.items():
        lr = load_lr(name, obj_key)
        res_dict[name] = fn(lr)

    if obj.hess is not None:
        res_dict["newton_damped"] = newton(obj.f, obj.grad, obj.hess, x0, lr=1.0,
                                           mode="damped", max_iter=min(MAX_ITER, 200), gtol=GTOL)

    return res_dict


def run_benchmark(obj, obj_key: str, start_points: list[np.ndarray]):
    """Run benchmark across 10 random start points, returning stats and sample curves."""
    f_star = obj.f_star or 0.0
    opt_names = ["gd", "momentum", "nag", "adagrad", "rmsprop", "adam"]
    if obj.hess is not None:
        opt_names.append("newton_damped")

    iters_records = {name: [] for name in opt_names}
    final_gaps = {name: [] for name in opt_names}
    grad_evals = {name: [] for name in opt_names}
    runtimes = {name: [] for name in opt_names}
    first_res = None

    for i, x0 in enumerate(start_points):
        res = run_single_start(obj, obj_key, x0)
        if i == 0:
            first_res = res
        for name in opt_names:
            fvals = res[name]["fvals"]
            iters_records[name].append(iters_to_target(fvals, f_star))
            final_gaps[name].append(max(float(fvals[-1] - f_star), 1e-25))
            grad_evals[name].append(res[name]["n_grad_evals"])
            runtimes[name].append(res[name]["runtime"])

    summary_rows = []
    for name in opt_names:
        iters_arr = np.array(iters_records[name])
        med_iter = float(np.median(iters_arr))
        q25 = float(np.percentile(iters_arr, 25))
        q75 = float(np.percentile(iters_arr, 75))
        iqr_iter = q75 - q25

        summary_rows.append({
            "objective": obj.name,
            "optimizer": name,
            "median_iters_to_1e6": med_iter,
            "q25_iters": q25,
            "q75_iters": q75,
            "iqr_iters": iqr_iter,
            "mean_final_gap": float(np.mean(final_gaps[name])),
            "mean_grad_evals": float(np.mean(grad_evals[name])),
            "mean_runtime_s": float(np.mean(runtimes[name])),
        })

    # Rate comparison for GD if quadratic
    rate_info = None
    if obj.mu is not None and obj.L is not None and obj.mu > 0:
        emp_rate, theory_rate, window = fit_gd_tail_rate(
            first_res["gd"]["fvals"], f_star, kappa=obj.L / obj.mu
        )
        rate_info = (emp_rate, theory_rate, window)

    return summary_rows, first_res, rate_info


def plot_curves(curves_dict: dict, obj_name: str, path: pathlib.Path, rate_info=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    f_star = 0.0
    for name, res in curves_dict.items():
        gaps = np.maximum(np.array(res["fvals"]) - f_star, 1e-20)
        ax.semilogy(gaps, label=name)

    title = f"Gap vs Iteration – {obj_name}"
    if rate_info is not None:
        emp_rate, theory_rate, win = rate_info
        title += f"\nGD Tail Gap Rate: {emp_rate:.4f} vs (1-1/kappa)^2: {theory_rate:.4f} (window {win})"
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective gap (log scale)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    # First: Inspect AdaGrad artifact
    inspect_adagrad_artifact()

    # Generate 10 random start points rescaled to norm 5
    rng = np.random.default_rng(SEED)

    objectives = [
        (quadratic(kappa=10, dim=10),  "quadratic"),
        (quadratic(kappa=100, dim=10), "quadratic"),
        (quartic(dim=10),              "rosenbrock"),
    ]

    all_summary_rows = []

    for obj, key in objectives:
        print(f"--- Running Benchmark on {obj.name} (10 random starts, norm=5) ---")
        starts = []
        for _ in range(N_STARTS):
            v = rng.standard_normal(obj.dim)
            v = 5.0 * v / np.linalg.norm(v)
            starts.append(v)

        summary_rows, first_res, rate_info = run_benchmark(obj, key, starts)
        all_summary_rows.extend(summary_rows)

        if rate_info is not None:
            emp_rate, theory_rate, win = rate_info
            print(f"  GD linear tail gap rate ({obj.name}):")
            print(f"    Empirical gap rate:       {emp_rate:.5f}")
            print(f"    Theoretical (1-1/kappa)^2: {theory_rate:.5f}")
            print(f"    Difference:               {abs(emp_rate - theory_rate):.5f}")
            print(f"    Fitting window:           iterations {win[0]} to {win[1]}")

        safe_name = obj.name.replace("(", "_").replace(")", "").replace(",", "_").replace("=", "")
        plot_curves(first_res, obj.name, RESULTS / f"trackA_exp1_convex_{safe_name}.png", rate_info)

    df = pd.DataFrame(all_summary_rows)
    df.to_csv(RESULTS / "trackA_exp1_convex.csv", index=False)

    print("\n=== Item 3: Convex Benchmark Summary (10 Random Starts, norm=5) ===")
    cols_to_print = ["objective", "optimizer", "median_iters_to_1e6", "q25_iters", "q75_iters", "iqr_iters", "mean_final_gap"]
    print(df[cols_to_print].to_string(index=False, float_format=lambda x: f"{x:.4e}"))


if __name__ == "__main__":
    main()
