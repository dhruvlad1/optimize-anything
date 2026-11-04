"""trackD_parallel/exp4_multitask.py
-------------------------------------
Experiment 4: Multi-task optimization (equal weighting, MGDA, GradNorm).

Parts:
  (a) Cosine similarity at init and task loss scales.
  (b) lr grid tuning, 5 seeds, 500 steps per method.
  (c) MGDA direction verification at 50 random points.
  (d) Print tables, save results/trackD_multitask.csv and plots.

Usage:
    python -m trackD_parallel.exp4_multitask
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

from trackD_parallel.multitask import (
    init_params,
    make_data,
    forward,
    losses,
    backprop,
    mgda_alpha,
    mgda_step,
    gradnorm_step,
    train,
    _task_grads_shared,
)


# ---------------------------------------------------------------------------
# Runtime estimate
# ---------------------------------------------------------------------------

def estimate_runtime() -> float:
    """Calibrate and estimate total runtime."""
    params = init_params(d_in=10, hidden=8, seed=0)
    X, y1, y2 = make_data(n=500, d_in=10, seed=0)
    t0 = time.perf_counter()
    for _ in range(50):
        backprop(params, X, y1, y2, (1.0, 1.0))
    t_step = (time.perf_counter() - t0) / 50

    # 3 methods x 3 lr x 5 seeds x 500 steps
    n_runs = 3 * 3 * 5 * 500
    return n_runs * t_step


# ---------------------------------------------------------------------------
# Part (a): Init diagnostics
# ---------------------------------------------------------------------------

def part_a() -> None:
    print("\n" + "=" * 60, flush=True)
    print("Part (a): Init diagnostics", flush=True)
    print("=" * 60, flush=True)

    X, y1, y2 = make_data(n=500, d_in=10, seed=0)

    # Cosine similarity between task gradients on shared layer at random inits
    n_samples = 100
    cos_sims = []
    for seed in range(n_samples):
        params = init_params(d_in=10, hidden=8, seed=seed)
        g1, g2 = _task_grads_shared(params, X, y1, y2)
        n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
        cos = float(np.dot(g1, g2) / (n1 * n2 + 1e-30))
        cos_sims.append(cos)

    cos_arr = np.array(cos_sims)
    conflict_frac = float(np.mean(cos_arr < 0))
    print(f"  Cosine similarity (task 1 vs task 2 shared-layer grad) over {n_samples} random inits:", flush=True)
    print(f"    mean={np.mean(cos_arr):.3f}  std={np.std(cos_arr):.3f}  "
          f"fraction_negative={conflict_frac:.2f}", flush=True)

    # Loss scales at init
    params0 = init_params(d_in=10, hidden=8, seed=0)
    L1, L2 = losses(params0, X, y1, y2)
    print(f"\n  Loss scales at init: L1={L1:.4f}  L2={L2:.4f}  ratio L2/L1={L2/L1:.2f}", flush=True)
    if L2 / L1 > 5:
        print("  → Task 2 dominates at init (expected by design)", flush=True)


# ---------------------------------------------------------------------------
# Part (b): Training sweep
# ---------------------------------------------------------------------------

def part_b() -> pd.DataFrame:
    print("\n" + "=" * 60, flush=True)
    print("Part (b): Training sweep (equal / MGDA / GradNorm)", flush=True)
    print("=" * 60, flush=True)

    methods = ["equal", "mgda", "gradnorm"]
    lr_grid = [0.005, 0.01, 0.05]
    seeds = [0, 1, 2, 3, 4]
    steps = 500

    X, y1, y2 = make_data(n=500, d_in=10, seed=99)

    records = []
    t_start = time.perf_counter()

    for method in methods:
        for lr in lr_grid:
            for seed in seeds:
                params0 = init_params(d_in=10, hidden=8, seed=seed)
                L1_0, L2_0 = losses(params0, X, y1, y2)
                hist = train(method, params0, X, y1, y2,
                             steps=steps, lr=lr, lr_weights=lr * 0.1, alpha_gn=1.5)
                L1_f = hist["L1"][-1]
                L2_f = hist["L2"][-1]
                rel1 = L1_f / (L1_0 + 1e-30)
                rel2 = L2_f / (L2_0 + 1e-30)
                mean_alpha = float(np.nanmean(hist["alpha"])) if method == "mgda" else float("nan")
                mean_w1 = float(np.nanmean(hist["w1"])) if method == "gradnorm" else float("nan")
                mean_w2 = float(np.nanmean(hist["w2"])) if method == "gradnorm" else float("nan")
                conflict_frac = float(np.mean(np.array(hist["cosine_sim"]) < 0)) if method == "mgda" else float("nan")

                records.append({
                    "method": method,
                    "lr": lr,
                    "seed": seed,
                    "L1_init": L1_0,
                    "L2_init": L2_0,
                    "L1_final": L1_f,
                    "L2_final": L2_f,
                    "rel_L1": rel1,
                    "rel_L2": rel2,
                    "sum_rel": rel1 + rel2,
                    "worst_rel": max(rel1, rel2),
                    "mean_alpha": mean_alpha,
                    "mean_w1": mean_w1,
                    "mean_w2": mean_w2,
                    "conflict_frac": conflict_frac,
                })
                elapsed = time.perf_counter() - t_start
                print(f"  {method:8s} lr={lr:.3f} seed={seed} | "
                      f"L1={L1_f:.4f}({rel1:.3f}x) L2={L2_f:.4f}({rel2:.3f}x) "
                      f"worst={max(rel1,rel2):.3f}x | [{elapsed:.1f}s]", flush=True)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Part (c): MGDA verification
# ---------------------------------------------------------------------------

def part_c() -> None:
    print("\n" + "=" * 60, flush=True)
    print("Part (c): MGDA direction verification (50 random gradient pairs)", flush=True)
    print("=" * 60, flush=True)

    rng = np.random.default_rng(42)
    n_checks = 50
    violations = 0

    for i in range(n_checks):
        d = 20
        g1 = rng.standard_normal(d)
        g2 = rng.standard_normal(d)

        # Brute-force verify: alpha* minimizes ||alpha*g1 + (1-alpha)*g2||^2 over [0,1]
        alpha_star = mgda_alpha(g1, g2)
        d_star = alpha_star * g1 + (1.0 - alpha_star) * g2
        norm_star = float(np.dot(d_star, d_star))

        # Check non-negative inner product with both gradients
        inner1 = float(np.dot(d_star, g1))
        inner2 = float(np.dot(d_star, g2))
        if inner1 < -1e-12 or inner2 < -1e-12:
            violations += 1
            print(f"  VIOLATION at pair {i}: inner1={inner1:.2e} inner2={inner2:.2e}", flush=True)

        # Check it's truly the minimiser (sample a few other alphas)
        for a_test in np.linspace(0, 1, 101):
            d_test = a_test * g1 + (1.0 - a_test) * g2
            if float(np.dot(d_test, d_test)) < norm_star - 1e-10:
                violations += 1
                print(f"  VIOLATION: alpha*={alpha_star:.4f} is NOT the minimiser (pair {i})", flush=True)
                break

    if violations > 0:
        raise AssertionError(f"MGDA verification failed: {violations} violations in {n_checks} pairs")
    print(f"  ✓ All {n_checks} pairs verified: d* has non-neg inner product with both grads "
          f"and is the min-norm point.", flush=True)

    # Special case: exactly opposite equal-norm gradients → zero direction
    g = rng.standard_normal(20)
    g /= np.linalg.norm(g)
    alpha_opp = mgda_alpha(g, -g)
    d_opp = alpha_opp * g + (1.0 - alpha_opp) * (-g)
    norm_opp = float(np.linalg.norm(d_opp))
    print(f"\n  Opposite equal-norm gradients: alpha*={alpha_opp:.4f}  ||d*||={norm_opp:.2e}", flush=True)
    if norm_opp > 1e-10:
        print(f"  ⚠ ||d*|| = {norm_opp:.2e} (expected ~0 for opposite gradients; numerical tolerance)", flush=True)
    else:
        print("  ✓ Zero direction for opposite gradients", flush=True)


# ---------------------------------------------------------------------------
# Part (d): Summary tables and CSV
# ---------------------------------------------------------------------------

def part_d(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60, flush=True)
    print("Part (d): Summary tables", flush=True)
    print("=" * 60, flush=True)

    # Best lr per method (by worst_rel = max(rel_L1, rel_L2), lower is better)
    best_rows = []
    for method in ["equal", "mgda", "gradnorm"]:
        sub = df[df["method"] == method]
        best_lr = sub.groupby("lr")["worst_rel"].mean().idxmin()
        best = sub[sub["lr"] == best_lr]
        best_rows.append({
            "method": method,
            "best_lr": best_lr,
            "L1_final_mean": best["L1_final"].mean(),
            "L2_final_mean": best["L2_final"].mean(),
            "rel_L1_mean": best["rel_L1"].mean(),
            "rel_L2_mean": best["rel_L2"].mean(),
            "sum_rel_mean": best["sum_rel"].mean(),
            "worst_rel_mean": best["worst_rel"].mean(),
            "conflict_frac_mean": best["conflict_frac"].mean(),
            "mean_alpha_mean": best["mean_alpha"].mean(),
            "mean_w1_mean": best["mean_w1"].mean(),
            "mean_w2_mean": best["mean_w2"].mean(),
        })

    summary_df = pd.DataFrame(best_rows)
    print("\nTable 1: Best-lr summary (mean over 5 seeds, at best lr per method)")
    print(summary_df[["method", "best_lr", "L1_final_mean", "L2_final_mean",
                       "rel_L1_mean", "rel_L2_mean", "worst_rel_mean"]].to_string(
        index=False, float_format="{:.4f}".format))

    print("\n  Observations:", flush=True)
    baseline_worst = summary_df[summary_df["method"] == "equal"]["worst_rel_mean"].values[0]
    for _, row in summary_df.iterrows():
        m = row["method"]
        wr = row["worst_rel_mean"]
        flag = ""
        if wr >= 1.0:
            flag = "  ← DID NOT IMPROVE worst-task"
        elif m != "equal" and wr > baseline_worst:
            flag = "  ← WORSE than equal weighting on worst task"
        print(f"    {m:9s}: worst_rel={wr:.4f}{flag}", flush=True)

    # Save CSV
    csv_path = RESULTS / "trackD_multitask.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        X_data, y1_data, y2_data = make_data(n=500, d_in=10, seed=99)
        colors = {"equal": "blue", "mgda": "orange", "gradnorm": "green"}

        for method in ["equal", "mgda", "gradnorm"]:
            best_lr = df[df["method"] == method].groupby("lr")["worst_rel"].mean().idxmin()
            sub_best = df[(df["method"] == method) & (df["lr"] == best_lr)]
            best_seed = int(sub_best.loc[sub_best["worst_rel"].idxmin(), "seed"])

            params0 = init_params(d_in=10, hidden=8, seed=best_seed)
            hist = train(method, params0, X_data, y1_data, y2_data,
                         steps=500, lr=best_lr, lr_weights=best_lr * 0.1)
            axes[0].plot(hist["L1"], label=f"{method}", color=colors[method], alpha=0.8)
            axes[1].plot(hist["L2"], label=f"{method}", color=colors[method], alpha=0.8)
            axes[2].plot(hist["total_loss"], label=f"{method}", color=colors[method], alpha=0.8)

        for ax, title in zip(axes, ["L1 (Task 1)", "L2 (Task 2)", "L1+L2"]):
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        plt.suptitle("Multi-task Training: equal / MGDA / GradNorm", y=1.02)
        plt.tight_layout()
        plot_path = RESULTS / "trackD_multitask.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}", flush=True)
    except Exception as e:
        print(f"  Plot skipped: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60, flush=True)
    print("Track D Exp 4: Multi-task Optimization", flush=True)
    print("=" * 60, flush=True)

    t_est = estimate_runtime()
    print(f"\nRuntime estimate: {t_est:.0f}s ({t_est/60:.1f} min)", flush=True)
    if t_est > 300:
        print("  → Estimate exceeds 5 min; running as configured (steps=500 is minimal).", flush=True)

    t_total = time.perf_counter()
    part_a()
    df = part_b()
    part_c()
    part_d(df)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
