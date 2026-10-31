"""trackC_quantum_opt/exp5_qnn.py
---------------------------------
2-Qubit Data-Re-Uploading Quantum Classifier on make_moons.

NOTE: All results in this study are SIMULATED on classical hardware using PennyLane default.qubit.
No physical quantum processor was used.
Track I compares this kind of model with a classical model and no conclusion about quantum
advantage is drawn here.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_moons

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 2-Qubit Data-Re-Uploading Circuit
# ===========================================================================

def R_y(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def R_z(theta: float) -> np.ndarray:
    e1 = np.exp(-1j * theta / 2.0)
    e2 = np.exp(1j * theta / 2.0)
    return np.array([[e1, 0.0], [0.0, e2]], dtype=np.complex128)


CNOT_MAT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
], dtype=np.complex128)

Z0_MAT = np.kron(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128), np.eye(2, dtype=np.complex128))


def qnn_predict_batch(X: np.ndarray, weights: np.ndarray, n_layers: int = 3) -> np.ndarray:
    """Vectorized simulation of 2-qubit data-re-uploading circuit across a batch of samples."""
    B = len(X)
    psi = np.zeros((B, 4), dtype=np.complex128)
    psi[:, 0] = 1.0  # |00>

    idx = 0
    for l in range(n_layers):
        # Data encoding on both wires via RY and RZ
        for b in range(B):
            x = X[b]
            U0 = R_z(float(x[1])) @ R_y(float(x[0]))
            U1 = R_z(float(x[1])) @ R_y(float(x[0]))
            psi[b] = np.kron(U0, U1) @ psi[b]

        # Trainable rotations: wire 0: RY(w0), RZ(w1); wire 1: RY(w2), RZ(w3)
        w0, w1, w2, w3 = weights[idx:idx + 4]
        idx += 4
        V0 = R_z(float(w1)) @ R_y(float(w0))
        V1 = R_z(float(w3)) @ R_y(float(w2))
        V = np.kron(V0, V1)

        psi = psi @ V.T
        psi = psi @ CNOT_MAT.T

    # Measurement: Expectation of Z on qubit 0
    expvals = np.real(np.sum(np.conj(psi) * (psi @ Z0_MAT.T), axis=1))
    return np.asarray(expvals, dtype=np.float64)


def make_pennylane_qnode():
    dev = qml.device("default.qubit", wires=2)

    def circuit(x, weights):
        idx = 0
        for l in range(3):
            qml.RY(x[0], wires=0); qml.RZ(x[1], wires=0)
            qml.RY(x[0], wires=1); qml.RZ(x[1], wires=1)
            for i in range(2):
                qml.RY(weights[idx], wires=i); idx += 1
                qml.RZ(weights[idx], wires=i); idx += 1
            qml.CNOT(wires=[0, 1])

    @qml.qnode(dev, interface="autograd")
    def qnode(x, weights):
        circuit(x, weights)
        return qml.expval(qml.PauliZ(0))

    return qnode


# ===========================================================================
# Validation: Parameter-Shift vs PennyLane Autograd
# ===========================================================================

def verify_gradient(X_val: np.ndarray, y_val: np.ndarray):
    print("=" * 80)
    print("EXP 5: Gradient Verification against PennyLane Autograd (5 samples)")
    print("=" * 80)

    pl_qnode = make_pennylane_qnode()
    n_params = 12
    rng = np.random.default_rng(999)
    w_init = rng.uniform(0.1, 2.0 * np.pi, n_params)
    w_pl = pnp.array(w_init, requires_grad=True)

    # PennyLane autograd gradient on batch of 5
    def pl_loss(w_in):
        preds = pnp.array([pl_qnode(X_val[i], w_in) for i in range(len(X_val))])
        return pnp.mean((preds - y_val) ** 2)

    g_pl = np.asarray(qml.grad(pl_loss)(w_pl))

    # Parameter-shift gradient: d(MSE)/dw = (2/B) sum (pred_b - y_b) * d(pred_b)/dw
    B = len(X_val)
    preds = qnn_predict_batch(X_val, w_init)
    g_shift = np.zeros(n_params, dtype=np.float64)
    shift = np.pi / 2.0

    for i in range(n_params):
        w_plus = w_init.copy(); w_plus[i] += shift
        w_minus = w_init.copy(); w_minus[i] -= shift
        preds_plus = qnn_predict_batch(X_val, w_plus)
        preds_minus = qnn_predict_batch(X_val, w_minus)
        dpred_dwi = (preds_plus - preds_minus) / (2.0 * np.sin(shift))
        g_shift[i] = (2.0 / B) * np.sum((preds - y_val) * dpred_dwi)

    max_diff = float(np.max(np.abs(g_pl - g_shift)))
    print(f"  PL Autograd Grad:     {np.round(g_pl[:4], 5)}...")
    print(f"  Param-Shift Grad:     {np.round(g_shift[:4], 5)}...")
    print(f"  Max Absolute Diff:    {max_diff:.2e}")
    if max_diff >= 1e-6:
        raise RuntimeError(f"Gradient verification FAILED: diff {max_diff:.2e} >= 1e-6")
    print("  [PASS] Parameter-shift gradient matches PennyLane exactly (< 1e-6).")


# ===========================================================================
# Training Experiment
# ===========================================================================

def run_qnn_experiment():
    print("\n" + "=" * 80)
    print("EXP 5: 2-Qubit Data-Re-Uploading Classifier (make_moons)")
    print("=" * 80)

    # 1. Dataset generation
    X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100] * 2 - 1, y[100:] * 2 - 1  # map to {-1, +1}

    verify_gradient(X_train[:5], y_train[:5])

    n_params = 12  # 3 layers * 4 trainable parameters
    n_iters = 200
    seeds = [42, 43, 44]
    best_lr = 0.04

    results = []
    best_seed_model = None
    best_test_acc = -1.0

    print(f"\nTraining QNN with Adam ({n_iters} iterations, lr={best_lr}) across 3 random starts:")

    for s_idx, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        w = rng.uniform(-np.pi, np.pi, n_params)

        m = np.zeros(n_params, dtype=float)
        v = np.zeros(n_params, dtype=float)
        shift = np.pi / 2.0

        t0 = time.perf_counter()
        B = len(X_train)

        # Total circuit evaluations:
        # In each iteration: 1 forward pass (B evals) + 2*d shifted passes (2*d*B evals) = (2*d + 1)*B evals
        evals_per_iter = (2 * n_params + 1) * B
        total_evals = 0

        for it in range(1, n_iters + 1):
            preds = qnn_predict_batch(X_train, w)
            diff = preds - y_train
            loss = float(np.mean(diff ** 2))
            total_evals += B

            # Compute gradient via parameter shift across all samples
            grad = np.zeros(n_params, dtype=np.float64)
            for i in range(n_params):
                w_plus = w.copy(); w_plus[i] += shift
                w_minus = w.copy(); w_minus[i] -= shift
                preds_plus = qnn_predict_batch(X_train, w_plus)
                preds_minus = qnn_predict_batch(X_train, w_minus)
                dpred = (preds_plus - preds_minus) / 2.0
                grad[i] = (2.0 / B) * np.sum(diff * dpred)
                total_evals += 2 * B

            # Adam update
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad ** 2)
            m_hat = m / (1.0 - 0.9 ** it)
            v_hat = v / (1.0 - 0.999 ** it)
            w -= best_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        elapsed = time.perf_counter() - t0

        # Evaluate performance
        train_preds = qnn_predict_batch(X_train, w)
        test_preds = qnn_predict_batch(X_test, w)

        train_acc = float(np.mean((np.sign(train_preds) == y_train).astype(float)))
        test_acc = float(np.mean((np.sign(test_preds) == y_test).astype(float)))
        final_loss = float(np.mean((train_preds - y_train) ** 2))

        print(f"  Seed {s} | Train Loss={final_loss:.4f} | Train Acc={train_acc * 100:.1f}% | Test Acc={test_acc * 100:.1f}% | Evals={total_evals} ({elapsed:.2f} s)")

        results.append({
            "seed": s,
            "trainable_params": n_params,
            "iterations": n_iters,
            "train_loss": final_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "circuit_evals": total_evals,
            "runtime_s": elapsed,
        })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_seed_model = w.copy()

    df_qnn = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "trackC_qnn.csv"
    df_qnn.to_csv(csv_path, index=False)
    print(f"\nSaved QNN benchmark records to {csv_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: QNN Classification Performance on make_moons")
    print("=" * 80)
    print(df_qnn[["seed", "trainable_params", "train_acc", "test_acc", "circuit_evals", "runtime_s"]].to_string(index=False))
    print(f"Mean Train Accuracy: {df_qnn['train_acc'].mean() * 100:.2f}% +/- {df_qnn['train_acc'].std() * 100:.2f}%")
    print(f"Mean Test Accuracy:  {df_qnn['test_acc'].mean() * 100:.2f}% +/- {df_qnn['test_acc'].std() * 100:.2f}%")

    print("\nNOTE: Track I compares this kind of model with a classical model and no conclusion about quantum advantage is drawn here.")

    # Plot decision boundary for best seed
    plot_decision_boundary(X_train, y_train, X_test, y_test, best_seed_model)


def plot_decision_boundary(X_train, y_train, X_test, y_test, weights):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Grid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_preds = qnn_predict_batch(grid_points, weights)
    Z = grid_preds.reshape(xx.shape)

    cs = ax.contourf(xx, yy, Z, levels=25, cmap="RdBu_r", alpha=0.7)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"QNode Output $\langle Z_0 \rangle \in [-1, 1]$", fontsize=10)

    # Decision boundary contour
    ax.contour(xx, yy, Z, levels=[0.0], colors="black", linewidths=2, linestyles="--")

    # Scatter points
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#d62728", edgecolors="k", s=40, label="Train Class +1")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c="#1f77b4", edgecolors="k", s=40, label="Train Class -1")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#ff7f0e", marker="^", s=40, label="Test Class +1")
    ax.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], c="#2ca02c", marker="v", s=40, label="Test Class -1")

    ax.set_title("2-Qubit QNN Data Re-Uploading Decision Boundary (SIMULATED)", fontsize=12)
    ax.set_xlabel("Feature $x_0$", fontsize=11)
    ax.set_ylabel("Feature $x_1$", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_path = RESULTS_DIR / "trackC_exp5_qnn_decision.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved decision boundary plot to {out_path}")


def main():
    run_qnn_experiment()


if __name__ == "__main__":
    main()
