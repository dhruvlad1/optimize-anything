"""
trackA_continuous/exp3_sensitivity.py
--------------------------------------
LR sensitivity sweep for every optimizer (except Newton) on:
  - quadratic(kappa=100): 5 random starts (rescaled to norm 5)
  - rosenbrock: 5 random starts (uniform in [-2, 2]^2)

Chooses each optimizer's learning rate by the MEDIAN final gap over those starts.
Writes results/trackA_tuned_lr.json (used by exp1, exp2, exp4).
Run first:  python -m trackA_continuous.exp3_sensitivity
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

from trackA_continuous.objectives import quadratic, rosenbrock
from trackA_continuous.optimizers import gd, momentum, nag, adagrad, rmsprop, adam

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

SEED = 42
MAX_ITER = 500
LR_GRID = np.logspace(-5, 2, 16)          # 16 values: 1e-5 … 1e2
BUDGET_GE = MAX_ITER                       # gradient evaluations budget
N_STARTS = 5

OPTIMIZERS = {
    "gd":       lambda f, g, x0, lr: gd(f, g, x0, lr=lr, max_iter=MAX_ITER, gtol=0.0),
    "momentum": lambda f, g, x0, lr: momentum(f, g, x0, lr=lr, beta=0.9, max_iter=MAX_ITER, gtol=0.0),
    "nag":      lambda f, g, x0, lr: nag(f, g, x0, lr=lr, beta=0.9, max_iter=MAX_ITER, gtol=0.0),
    "adagrad":  lambda f, g, x0, lr: adagrad(f, g, x0, lr=lr, max_iter=MAX_ITER, gtol=0.0),
    "rmsprop":  lambda f, g, x0, lr: rmsprop(f, g, x0, lr=lr, max_iter=MAX_ITER, gtol=0.0),
    "adam":     lambda f, g, x0, lr: adam(f, g, x0, lr=lr, max_iter=MAX_ITER, gtol=0.0),
}

DIVERGED_THRESHOLD = 1e10


def run_multi_start_sweep(obj_name: str, obj, starts: list[np.ndarray]):
    init_gaps = [float(obj.f(s) - (obj.f_star or 0.0)) for s in starts]
    median_init_gap = float(np.median(init_gaps))

    all_detailed_rows = []
    summary_by_opt = {}

    for opt_name, opt_fn in OPTIMIZERS.items():
        best_lr = None
        best_med_gap = float("inf")
        best_iqr = float("inf")
        stable_count = 0
        converged_count = 0

        for lr in LR_GRID:
            start_gaps = []
            for s in starts:
                try:
                    res = opt_fn(obj.f, obj.grad, s, lr)
                    fg = float(res["fvals"][-1] - (obj.f_star or 0.0))
                    if not np.isfinite(fg) or fg > DIVERGED_THRESHOLD:
                        fg = np.nan
                except Exception:
                    fg = np.nan
                start_gaps.append(fg)

            finite_gaps = [g for g in start_gaps if np.isfinite(g)]
            is_stable = (len(finite_gaps) == len(starts)) and (float(np.median(finite_gaps)) < median_init_gap)
            med_g = float(np.median(finite_gaps)) if is_stable else np.nan
            is_conv = is_stable and (med_g < 1e-3)

            if is_stable:
                stable_count += 1
                q25 = float(np.percentile(finite_gaps, 25))
                q75 = float(np.percentile(finite_gaps, 75))
                iqr = q75 - q25
                if med_g < best_med_gap:
                    best_med_gap = med_g
                    best_lr = float(lr)
                    best_iqr = iqr

            if is_conv:
                converged_count += 1

            for s_idx, fg in enumerate(start_gaps):
                all_detailed_rows.append({
                    "objective": obj_name,
                    "optimizer": opt_name,
                    "lr": float(lr),
                    "start_idx": s_idx,
                    "final_gap": fg,
                    "stable": is_stable,
                    "converged": is_conv,
                })

        # Fallback if no LR was strictly stable
        if best_lr is None:
            best_lr = float(LR_GRID[len(LR_GRID) // 2])
            best_med_gap = np.nan
            best_iqr = np.nan

        summary_by_opt[opt_name] = {
            "best_lr": best_lr,
            "median_final_gap": best_med_gap,
            "iqr_final_gap": best_iqr,
            "stable_lrs": stable_count,
            "converged_lrs": converged_count,
        }

    return all_detailed_rows, summary_by_opt


def main() -> None:
    # 1. Print existing (old) start point(s)
    old_rng = np.random.default_rng(SEED)
    old_x0_quad = old_rng.standard_normal(10)
    old_x0_rosen = old_rng.uniform(-2, 2, size=2)

    print("=== Item 3: Existing Start Points Inspection ===")
    print("Previous start point for quadratic(kappa=100) (seed=42, unnormalized standard normal):")
    print("  x0_quad_old:", [round(v, 4) for v in old_x0_quad.tolist()], f"(norm = {np.linalg.norm(old_x0_quad):.4f})")
    print("Previous start point for rosenbrock (seed=42, uniform [-2, 2]^2):")
    print("  x0_rosen_old:", [round(v, 4) for v in old_x0_rosen.tolist()])
    print("================================================\n")

    # 2. Generate 5 random start points per problem
    rng_q = np.random.default_rng(SEED)
    starts_q = []
    for _ in range(N_STARTS):
        v = rng_q.standard_normal(10)
        v = 5.0 * v / np.linalg.norm(v)
        starts_q.append(v)

    rng_r = np.random.default_rng(SEED + 10)
    starts_r = [rng_r.uniform(-2, 2, size=2) for _ in range(N_STARTS)]

    print(f"Running sweep over 16 LR values [{LR_GRID[0]:.1e}, {LR_GRID[-1]:.1e}] across 5 random starts each...\n")

    obj_quad = quadratic(kappa=100, dim=10)
    rows_q, sum_q = run_multi_start_sweep("quadratic(kappa=100)", obj_quad, starts_q)

    obj_rosen = rosenbrock()
    rows_r, sum_r = run_multi_start_sweep("rosenbrock", obj_rosen, starts_r)

    all_detailed = rows_q + rows_r
    df = pd.DataFrame(all_detailed)
    df.to_csv(RESULTS / "trackA_exp3_sensitivity.csv", index=False)

    # Read old tuned LRs if present for comparison
    old_tuned = {}
    if (RESULTS / "trackA_tuned_lr.json").exists():
        try:
            with open(RESULTS / "trackA_tuned_lr.json") as fh:
                old_tuned = json.load(fh)
        except Exception:
            pass

    # 3. Build new tuned LR dictionary and rewrite
    new_tuned = {}
    for opt_name in OPTIMIZERS:
        new_tuned[opt_name] = {
            "quadratic": sum_q[opt_name]["best_lr"],
            "rosenbrock": sum_r[opt_name]["best_lr"],
        }
    new_tuned["newton"] = {"quadratic": 1.0, "rosenbrock": 1.0}

    with open(RESULTS / "trackA_tuned_lr.json", "w") as fh:
        json.dump(new_tuned, fh, indent=2)
    print(f"Rewrote results/trackA_tuned_lr.json with multi-start median-tuned rates.\n")

    # 4. Old vs New Tuned Learning Rates
    print("=== Old vs New Tuned Learning Rates ===")
    print(f"{'Optimizer':<12} | {'Old LR (quad)':>14} -> {'New LR (quad)':<14} | {'Old LR (rosen)':>14} -> {'New LR (rosen)':<14}")
    print("-" * 75)
    for opt_name in OPTIMIZERS:
        old_q = old_tuned.get(opt_name, {}).get("quadratic", float("nan"))
        new_q = new_tuned[opt_name]["quadratic"]
        old_r = old_tuned.get(opt_name, {}).get("rosenbrock", float("nan"))
        new_r = new_tuned[opt_name]["rosenbrock"]
        print(f"{opt_name:<12} | {old_q:>14.4e} -> {new_q:<14.4e} | {old_r:>14.4e} -> {new_r:<14.4e}")
    print()

    # 5. Summary table reporting median final gap & IQR at chosen rate
    print("=== Sensitivity Sweep Summary: Median Final Gap & IQR at Chosen Rate ===")
    hdr = (f"{'Problem':<15} | {'Optimizer':<10} | {'Tuned LR':>11} | {'Median Gap':>12} | "
           f"{'IQR':>12} | {'Stable':>7} | {'Conv (<1e-3)':>13} | {'Edge?':>6}")
    print(hdr)
    print("-" * len(hdr))

    zero_conv_pairs = 0

    for prob_name, summary in [("quadratic(k=100)", sum_q), ("rosenbrock", sum_r)]:
        for opt_name, s in summary.items():
            blr = s["best_lr"]
            med_g = s["median_final_gap"]
            iqr = s["iqr_final_gap"]
            st = s["stable_lrs"]
            cv = s["converged_lrs"]
            is_edge = "YES" if np.isclose(blr, LR_GRID[0]) or np.isclose(blr, LR_GRID[-1]) else "no"

            if cv == 0:
                zero_conv_pairs += 1

            print(f"{prob_name:<15} | {opt_name:<10} | {blr:>11.4e} | {med_g:>12.4e} | "
                  f"{iqr:>12.4e} | {st:>7d} | {cv:>13d} | {is_edge:>6}")

    print("-" * len(hdr))
    print(f"Total optimizer/problem pairs with ZERO converged runs (gap < 1e-3): {zero_conv_pairs} / 12.")
    print("Plain Statement:")
    print("  On Rosenbrock, the sweep ranks optimizers strictly by progress within the iteration budget (500 steps),")
    print("  not by convergence (most optimizers fail to reach a gap below 1e-3 within the budget due to the narrow curved valley).\n")

    # Grid edge check
    edge_cases = []
    for prob_key, summary in [("quadratic", sum_q), ("rosenbrock", sum_r)]:
        for opt_name, s in summary.items():
            if np.isclose(s["best_lr"], LR_GRID[0]):
                edge_cases.append(f"{opt_name} on {prob_key} at LOWER edge ({LR_GRID[0]:.1e})")
            elif np.isclose(s["best_lr"], LR_GRID[-1]):
                edge_cases.append(f"{opt_name} on {prob_key} at UPPER edge ({LR_GRID[-1]:.1e})")

    if edge_cases:
        for ec in edge_cases:
            print(f"  NOTICE: {ec}")
    else:
        print("  Notice: No best learning rate is at a grid edge.\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df_all = pd.DataFrame(all_detailed)
    for ax_idx, (p_name, title) in enumerate([("quadratic(kappa=100)", "quadratic(kappa=100)"), ("rosenbrock", "rosenbrock")]):
        ax = axes[ax_idx]
        df_p = df_all[df_all["objective"] == p_name]
        for opt_name in OPTIMIZERS:
            sub = df_p[df_p["optimizer"] == opt_name]
            lrs = []
            med_gaps = []
            for lr, grp in sub.groupby("lr"):
                lrs.append(lr)
                finite = [g for g in grp["final_gap"] if np.isfinite(g)]
                med_gaps.append(np.median(finite) if len(finite) == len(grp) else np.nan)
            ax.plot(lrs, med_gaps, marker="o", label=opt_name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Median final gap (5 starts)")
        ax.set_title(f"Sensitivity: {title}")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "trackA_exp3_sensitivity.png", dpi=120)
    plt.close()
    print("Plot saved to results/trackA_exp3_sensitivity.png")


if __name__ == "__main__":
    main()
