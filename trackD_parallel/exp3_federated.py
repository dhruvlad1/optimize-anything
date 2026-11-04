"""trackD_parallel/exp3_federated.py
------------------------------------
Experiment 3: Federated Learning (FedAvg vs centralised SGD).

Parts:
  (a) Gradient check vs finite differences; FedAvg E=1 full-batch == centralised GD.
  (b) FedAvg vs centralised SGD sweep: heterogeneity x E grid, participation.
  (c) Print tables, save results/trackD_federated.csv and plots.

Usage:
    python -m trackD_parallel.exp3_federated
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

from trackD_parallel.federated import (
    make_clients,
    logistic_loss_and_grad,
    fedavg,
    fedavg_full_batch,
    centralized_gd,
    centralized_gd_steps,
    centralized_sgd,
    accuracy,
    _sigmoid,
)


# ---------------------------------------------------------------------------
# Runtime estimate
# ---------------------------------------------------------------------------

def estimate_runtime() -> float:
    """Estimate total runtime in seconds and print it."""
    # Quick calibration: time one FedAvg run with realistic settings
    clients, _ = make_clients(K=10, d=20, n_per_client=200, heterogeneity=0.0, seed=42)
    t0 = time.perf_counter()
    fedavg(clients, rounds=10, E=1, batch_size=200, lr=0.01, participation=1.0, seed=7)
    t_one = time.perf_counter() - t0
    # Scale to full experiment:
    # 3 het x 3 E x 3 lr x 3 seeds (100 rounds each) + participation: 3 part x 3 lr x 3 seeds
    full_runs = (3 * 3 * 3 * 3 + 3 * 3 * 3)
    est = full_runs * (t_one * 10)  # 10x for 100 rounds
    return est


# ---------------------------------------------------------------------------
# Part (a): Gradient check + E=1 full-batch sanity check
# ---------------------------------------------------------------------------

def part_a() -> None:
    print("\n" + "=" * 60, flush=True)
    print("Part (a): Gradient correctness & E=1 sanity check", flush=True)
    print("=" * 60, flush=True)

    rng = np.random.default_rng(1)
    n, d = 50, 20
    X = rng.standard_normal((n, d))
    y = (rng.random(n) > 0.5).astype(float)
    w = rng.standard_normal(d) * 0.3
    eps = 1e-5

    loss, grad = logistic_loss_and_grad(w, X, y, l2=1e-3)
    fd_grad = np.zeros_like(w)
    for i in range(d):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        lp, _ = logistic_loss_and_grad(wp, X, y, l2=1e-3)
        lm, _ = logistic_loss_and_grad(wm, X, y, l2=1e-3)
        fd_grad[i] = (lp - lm) / (2 * eps)

    max_err = float(np.max(np.abs(grad - fd_grad)))
    rel_err = float(np.max(np.abs(grad - fd_grad) / (np.abs(fd_grad) + 1e-10)))
    print(f"  Gradient vs finite-diff: max_abs={max_err:.2e}  max_rel={rel_err:.2e}", flush=True)
    if max_err > 1e-5:
        raise AssertionError(f"Gradient check failed: max_abs={max_err:.2e}")
    print("  ✓ Gradient check passed", flush=True)

    # E=1, full-batch, full-participation FedAvg == centralised GD
    clients, _ = make_clients(K=5, d=10, n_per_client=50, heterogeneity=0.0, seed=0)
    rounds = 20
    lr_test = 0.05

    w_fed = fedavg_full_batch(clients, rounds=rounds, E=1, lr=lr_test)
    w_cen = centralized_gd_steps(clients, rounds=rounds, lr=lr_test)
    max_diff = float(np.max(np.abs(w_fed - w_cen)))
    print(f"\n  FedAvg(E=1, full-batch) vs centralised GD max diff: {max_diff:.2e}", flush=True)
    if max_diff > 1e-10:
        raise AssertionError(f"E=1 equivalence check failed: max_diff={max_diff:.2e}")
    print("  ✓ E=1 equivalence check passed (max diff < 1e-10)", flush=True)


# ---------------------------------------------------------------------------
# Part (b): FedAvg vs centralised SGD sweep
# ---------------------------------------------------------------------------

def part_b() -> pd.DataFrame:
    print("\n" + "=" * 60, flush=True)
    print("Part (b): FedAvg vs Centralised SGD sweep", flush=True)
    print("=" * 60, flush=True)

    heterogeneities = [0.0, 0.5, 1.0]
    E_values = [1, 5, 20]
    lr_grid = [0.05, 0.1, 0.2]
    seeds = [7, 13, 42]
    rounds = 100
    batch_size = 32

    records = []
    t_start = time.perf_counter()

    for het in heterogeneities:
        clients, _ = make_clients(K=10, d=20, n_per_client=200, heterogeneity=het, seed=42)
        # Best-lr centralised SGD as reference
        X_all = np.concatenate([c[0] for c in clients])
        y_all = np.concatenate([c[1] for c in clients])

        for E in E_values:
            for lr in lr_grid:
                for seed in seeds:
                    t0 = time.perf_counter()
                    hist = fedavg(clients, rounds=rounds, E=E, batch_size=batch_size,
                                  lr=lr, participation=1.0, seed=seed)
                    final_loss = hist[-1]["loss"]
                    final_acc = hist[-1]["accuracy"]
                    # Centralised SGD with same number of gradient steps
                    total_steps = rounds * 10 * E  # 10 clients * E local steps per round
                    cen_hist = centralized_sgd(clients, total_steps=total_steps,
                                               batch_size=batch_size, lr=lr, seed=seed)
                    cen_final_loss = cen_hist[-1]["loss"] if cen_hist else float("nan")
                    cen_final_acc = cen_hist[-1]["accuracy"] if cen_hist else float("nan")
                    elapsed = time.perf_counter() - t0

                    records.append({
                        "experiment": "het_E_sweep",
                        "heterogeneity": het,
                        "E": E,
                        "lr": lr,
                        "seed": seed,
                        "participation": 1.0,
                        "fedavg_final_loss": final_loss,
                        "fedavg_final_acc": final_acc,
                        "cen_sgd_final_loss": cen_final_loss,
                        "cen_sgd_final_acc": cen_final_acc,
                        "loss_diff": final_loss - cen_final_loss,
                    })
                    elapsed_total = time.perf_counter() - t_start
                    print(f"  het={het:.1f} E={E:2d} lr={lr:.2f} seed={seed} | "
                          f"FedAvg loss={final_loss:.4f} acc={final_acc:.3f} | "
                          f"CenSGD loss={cen_final_loss:.4f} acc={cen_final_acc:.3f} | "
                          f"[{elapsed_total:.1f}s]", flush=True)

    # Participation sweep: E=5, het=1.0
    print("\n  --- Participation fraction sweep (E=5, het=1.0) ---", flush=True)
    het = 1.0
    E = 5
    participation_vals = [0.3, 0.5, 1.0]
    clients, _ = make_clients(K=10, d=20, n_per_client=200, heterogeneity=het, seed=42)
    for part in participation_vals:
        for lr in lr_grid:
            for seed in seeds:
                hist = fedavg(clients, rounds=rounds, E=E, batch_size=batch_size,
                              lr=lr, participation=part, seed=seed)
                final_loss = hist[-1]["loss"]
                final_acc = hist[-1]["accuracy"]
                total_comm = hist[-1]["comm_volume"]
                records.append({
                    "experiment": "participation_sweep",
                    "heterogeneity": het,
                    "E": E,
                    "lr": lr,
                    "seed": seed,
                    "participation": part,
                    "fedavg_final_loss": final_loss,
                    "fedavg_final_acc": final_acc,
                    "cen_sgd_final_loss": float("nan"),
                    "cen_sgd_final_acc": float("nan"),
                    "loss_diff": float("nan"),
                })
                elapsed_total = time.perf_counter() - t_start
                print(f"  part={part:.1f} E={E:2d} lr={lr:.2f} seed={seed} | "
                      f"FedAvg loss={final_loss:.4f} acc={final_acc:.3f} comm={total_comm} | "
                      f"[{elapsed_total:.1f}s]", flush=True)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Part (c): Summary tables and CSV
# ---------------------------------------------------------------------------

def part_c(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60, flush=True)
    print("Part (c): Summary tables", flush=True)
    print("=" * 60, flush=True)

    # Table 1: het x E sweep (mean over lr and seeds)
    het_e = df[df["experiment"] == "het_E_sweep"].copy()
    summary1 = (
        het_e.groupby(["heterogeneity", "E"])[
            ["fedavg_final_loss", "fedavg_final_acc", "cen_sgd_final_loss", "cen_sgd_final_acc"]
        ]
        .mean()
        .reset_index()
    )
    summary1.columns = ["het", "E", "FedAvg_loss", "FedAvg_acc", "CenSGD_loss", "CenSGD_acc"]
    summary1["FedAvg_vs_CenSGD"] = summary1["FedAvg_loss"] - summary1["CenSGD_loss"]

    print("\nTable 1: FedAvg vs Centralised SGD (mean over lr grid and 3 seeds)")
    print(summary1.to_string(index=False, float_format="{:.4f}".format))

    # Highlight client drift
    print("\n  Observations:", flush=True)
    for _, row in summary1.iterrows():
        if row["FedAvg_vs_CenSGD"] > 0.01:
            print(f"    ⚠ het={row['het']:.1f} E={row['E']:2d}: FedAvg loss {row['FedAvg_vs_CenSGD']:+.4f} "
                  f"above CenSGD → client drift", flush=True)
        elif row["FedAvg_vs_CenSGD"] < -0.005:
            print(f"    ✓ het={row['het']:.1f} E={row['E']:2d}: FedAvg slightly better than CenSGD "
                  f"(diff {row['FedAvg_vs_CenSGD']:+.4f})", flush=True)
        else:
            print(f"    ~ het={row['het']:.1f} E={row['E']:2d}: FedAvg ≈ CenSGD "
                  f"(diff {row['FedAvg_vs_CenSGD']:+.4f})", flush=True)

    # Table 2: Participation sweep
    part_df = df[df["experiment"] == "participation_sweep"].copy()
    summary2 = (
        part_df.groupby(["participation"])[["fedavg_final_loss", "fedavg_final_acc"]]
        .mean()
        .reset_index()
    )
    summary2.columns = ["participation", "FedAvg_loss", "FedAvg_acc"]
    print("\nTable 2: Participation fraction (E=5, het=1.0, mean over lr and 3 seeds)")
    print(summary2.to_string(index=False, float_format="{:.4f}".format))

    # Save CSV
    csv_path = RESULTS / "trackD_federated.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for E_val in [1, 5, 20]:
            sub = summary1[summary1["E"] == E_val]
            axes[0].plot(sub["het"], sub["FedAvg_loss"], marker="o", label=f"FedAvg E={E_val}")
            axes[0].plot(sub["het"], sub["CenSGD_loss"], marker="s", linestyle="--",
                         label=f"CenSGD E={E_val}", alpha=0.6)
        axes[0].set_xlabel("Heterogeneity")
        axes[0].set_ylabel("Final loss (mean over lr, seeds)")
        axes[0].set_title("FedAvg vs Centralised SGD")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(summary2["participation"].astype(str), summary2["FedAvg_acc"])
        axes[1].set_xlabel("Participation fraction")
        axes[1].set_ylabel("Final accuracy")
        axes[1].set_title("Participation (het=1.0, E=5)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS / "trackD_federated.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        print(f"  Saved: {plot_path}", flush=True)
    except Exception as e:
        print(f"  Plot skipped: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60, flush=True)
    print("Track D Exp 3: Federated Learning", flush=True)
    print("=" * 60, flush=True)

    t_est = estimate_runtime()
    print(f"\nRuntime estimate: {t_est:.0f}s ({t_est/60:.1f} min)", flush=True)
    if t_est > 300:
        print("  → Estimate exceeds 5 min; sweep is already compact; proceeding.", flush=True)

    t_total = time.perf_counter()
    part_a()
    df = part_b()
    part_c(df)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
