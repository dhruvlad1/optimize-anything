"""trackA_continuous/exp5_backprop_ml.py
--------------------------------------
Machine Learning optimization benchmark on make_moons.

Trains a 2-layer neural network (hidden dimension 8) with tanh activation:
- Dataset: make_moons (n=300, noise=0.2, seed 0), train/test split 200/100.
- Optimizers: Full-batch Gradient Descent vs Adam (via adapter to trackA_continuous/optimizers.py).
- Learning rate tuning on a small grid.
- L2 regularization study: L2 in {0.0, 0.01, 0.1} for the better optimizer.
- Outputs: train loss curves, train/test accuracies, parameter norms, CSV and figures.
"""

from __future__ import annotations

import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

from trackA_continuous.compare.backprop_antigravity import init_params, loss_and_grads
from trackA_continuous.curvature import flatten_params, unflatten_params
from trackA_continuous.optimizers import adam, gd

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def get_data(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate make_moons dataset and split into 200 train and 100 test samples."""
    X, y = make_moons(n_samples=300, noise=0.2, random_state=seed)
    
    # Deterministic train/test split without external dependencies
    rng = np.random.default_rng(42)
    indices = rng.permutation(300)
    train_idx, test_idx = indices[:200], indices[200:]
    
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_accuracy(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy."""
    hidden = np.tanh(X @ params["W1"] + params["b1"])
    logits = hidden @ params["W2"] + params["b2"]
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def make_objective(X: np.ndarray, y: np.ndarray, l2: float, shapes: dict[str, tuple[int, ...]]):
    """Return flat scalar objective f(theta) and gradient grad(theta) for optimizers."""
    def f(theta: np.ndarray) -> float:
        params = unflatten_params(theta, shapes)
        loss, _ = loss_and_grads(params, X, y, l2=l2)
        return float(loss)

    def grad(theta: np.ndarray) -> np.ndarray:
        params = unflatten_params(theta, shapes)
        _, g_dict = loss_and_grads(params, X, y, l2=l2)
        g_flat, _ = flatten_params(g_dict)
        return g_flat

    return f, grad


# ---------------------------------------------------------------------------
# Main Training Benchmark
# ---------------------------------------------------------------------------

def run_ml_experiment(max_iter: int = 1500) -> pd.DataFrame:
    X_train, y_train, X_test, y_test = get_data(seed=0)
    
    # Model architecture: 2 features -> 8 hidden -> 2 classes
    d, h, k = 2, 8, 2
    init_p = init_params(d, h, k, seed=42)
    theta0, shapes = flatten_params(init_p)

    lr_grid = [0.005, 0.01, 0.03, 0.1, 0.3, 1.0]

    print("=== Step 1: Learning Rate Tuning on Training Set (L2=0.0) ===")
    
    # Tune GD
    best_gd_lr = None
    best_gd_loss = float("inf")
    best_gd_res = None
    for lr in lr_grid:
        f_fn, g_fn = make_objective(X_train, y_train, 0.0, shapes)
        res = gd(f_fn, g_fn, theta0, lr=lr, max_iter=max_iter, gtol=1e-8)
        final_l = float(res["fvals"][-1])
        if np.isfinite(final_l) and final_l < best_gd_loss:
            best_gd_loss = final_l
            best_gd_lr = lr
            best_gd_res = res

    print(f"Tuned GD:   best lr = {best_gd_lr:.4f}, final train loss = {best_gd_loss:.4e}")

    # Tune Adam
    best_adam_lr = None
    best_adam_loss = float("inf")
    best_adam_res = None
    for lr in lr_grid:
        f_fn, g_fn = make_objective(X_train, y_train, 0.0, shapes)
        res = adam(f_fn, g_fn, theta0, lr=lr, max_iter=max_iter, gtol=1e-8)
        final_l = float(res["fvals"][-1])
        if np.isfinite(final_l) and final_l < best_adam_loss:
            best_adam_loss = final_l
            best_adam_lr = lr
            best_adam_res = res

    print(f"Tuned Adam: best lr = {best_adam_lr:.4f}, final train loss = {best_adam_loss:.4e}")

    # Determine better optimizer
    better_opt_name = "Adam" if best_adam_loss <= best_gd_loss else "GD"
    better_opt_fn = adam if better_opt_name == "Adam" else gd
    better_opt_lr = best_adam_lr if better_opt_name == "Adam" else best_gd_lr
    print(f"\nBetter optimizer based on training loss: {better_opt_name} (lr={better_opt_lr})")

    # Regularization sweep on better optimizer: L2 in {0.0, 0.01, 0.1}
    l2_values = [0.0, 0.01, 0.1]
    records = []
    loss_curves = {}

    # Include GD baseline
    p_gd = unflatten_params(best_gd_res["path"][-1], shapes)
    train_acc_gd = compute_accuracy(p_gd, X_train, y_train)
    test_acc_gd = compute_accuracy(p_gd, X_test, y_test)
    param_norm_gd = float(np.linalg.norm(best_gd_res["path"][-1]))
    records.append({
        "Model / Config": f"GD (lr={best_gd_lr})",
        "L2 Penalty": 0.0,
        "Train Loss": best_gd_loss,
        "Train Acc": train_acc_gd,
        "Test Acc": test_acc_gd,
        "Param Norm": param_norm_gd,
        "Iterations": best_gd_res["n_iters"],
        "Runtime (s)": best_gd_res["runtime"],
    })
    loss_curves[f"GD (lr={best_gd_lr}, L2=0)"] = best_gd_res["fvals"]

    # Run regularization conditions
    for l2 in l2_values:
        f_fn, g_fn = make_objective(X_train, y_train, l2, shapes)
        res = better_opt_fn(f_fn, g_fn, theta0, lr=better_opt_lr, max_iter=max_iter, gtol=1e-8)
        final_theta = res["path"][-1]
        p_opt = unflatten_params(final_theta, shapes)
        train_acc = compute_accuracy(p_opt, X_train, y_train)
        test_acc = compute_accuracy(p_opt, X_test, y_test)
        param_norm = float(np.linalg.norm(final_theta))
        train_loss = float(res["fvals"][-1])

        label = f"{better_opt_name} (L2={l2})"
        loss_curves[label] = res["fvals"]
        records.append({
            "Model / Config": f"{better_opt_name} (lr={better_opt_lr})",
            "L2 Penalty": l2,
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Param Norm": param_norm,
            "Iterations": res["n_iters"],
            "Runtime (s)": res["runtime"],
        })

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Loss curves
    ax0 = axes[0]
    for label, curve in loss_curves.items():
        ax0.plot(curve, label=label, linewidth=1.8)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Loss (log scale)")
    ax0.set_yscale("log")
    ax0.set_title("Training Loss Convergence on make_moons")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # Accuracy vs Parameter Norm
    ax1 = axes[1]
    l2_rows = df[df["Model / Config"].str.contains(better_opt_name)]
    ax1.plot(l2_rows["L2 Penalty"], l2_rows["Test Acc"], "o-", color="royalblue", label="Test Accuracy")
    ax1.plot(l2_rows["L2 Penalty"], l2_rows["Train Acc"], "s--", color="forestgreen", label="Train Accuracy")
    ax1.set_xlabel("L2 Regularization Penalty")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy vs. L2 Penalty ({better_opt_name})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "trackA_ml_training.png", dpi=120)
    plt.close()

    df.to_csv(RESULTS / "trackA_ml_training.csv", index=False)
    return df


def main() -> None:
    df = run_ml_experiment(max_iter=1500)
    print("\n=== Experiment 5: ML Optimization on make_moons ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nResults and plots saved to results/trackA_ml_training.*")


if __name__ == "__main__":
    main()
