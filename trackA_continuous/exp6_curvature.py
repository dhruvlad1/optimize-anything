"""trackA_continuous/exp6_curvature.py
------------------------------------
Curvature experiments:
  (a) Hessian-Vector Product (HVP) validation and spectral analysis.
  (b) Gauss-Newton and Levenberg-Marquardt on non-linear least squares.
  (c) Empirical Fisher preconditioning on neural network training.
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
from scipy.optimize import least_squares
from sklearn.datasets import make_moons

from trackA_continuous.compare.backprop_antigravity import init_params, loss_and_grads
from trackA_continuous.curvature import (
    diag_fisher,
    empirical_fisher,
    flatten_params,
    gauss_newton,
    hvp,
    low_rank_fisher,
    per_sample_grads,
    power_iteration_top_eig,
    preconditioned_gradient_step,
    unflatten_params,
)

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


# ===========================================================================
# Part (a): HVP Check & Spectral Analysis
# ===========================================================================

def exp6_part_a_hvp_check():
    print("=" * 75)
    print("PART (A): HVP Check & Spectral Analysis")
    print("=" * 75)

    # Tiny network: d=2, h=3, k=2 on 8 synthetic samples
    d, h, k = 2, 3, 2
    n_samples = 8
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=1)
    
    # Initialize at a random point
    params = init_params(d, h, k, seed=123)
    theta, shapes = flatten_params(params)
    dim = theta.shape[0]  # (2*3 + 3 + 3*2 + 2) = 17

    def grad_fn(th: np.ndarray) -> np.ndarray:
        p = unflatten_params(th, shapes)
        _, g_dict = loss_and_grads(p, X, y, l2=0.01)
        g_flat, _ = flatten_params(g_dict)
        return g_flat

    # 1. Build explicit Hessian via central finite differences of the gradient
    eps = 1e-5
    H_explicit = np.zeros((dim, dim), dtype=float)
    for j in range(dim):
        ej = np.zeros(dim)
        ej[j] = 1.0
        g_plus = grad_fn(theta + eps * ej)
        g_minus = grad_fn(theta - eps * ej)
        H_explicit[:, j] = (g_plus - g_minus) / (2.0 * eps)
    # Symmetrize
    H_explicit = 0.5 * (H_explicit + H_explicit.T)

    # 2. Check HVP against H_explicit @ v for 5 random vectors
    rng = np.random.default_rng(42)
    rel_errors = []
    for i in range(5):
        v = rng.standard_normal(dim)
        v = v / np.linalg.norm(v)
        hv_approx = hvp(grad_fn, theta, v, eps=eps)
        hv_exact = H_explicit @ v
        err = np.linalg.norm(hv_approx - hv_exact) / max(1e-8, np.linalg.norm(hv_exact))
        rel_errors.append(err)

    max_rel_err = float(np.max(rel_errors))
    print(f"Dimension of parameter space: D = {dim}")
    print(f"HVP vs Explicit Hessian @ v (5 random v):")
    print(f"  Maximum relative error: {max_rel_err:.4e}")

    # 3. Power iteration top eigenvalue vs eigvalsh
    top_eig_power, _ = power_iteration_top_eig(
        lambda v: hvp(grad_fn, theta, v, eps=eps),
        dim=dim,
        iters=150,
        seed=42,
    )
    eigvals = np.linalg.eigvalsh(H_explicit)
    # The dominant eigenvalue by magnitude
    top_idx = np.argmax(np.abs(eigvals))
    top_eig_exact = float(eigvals[top_idx])
    top_rel_err = abs(top_eig_power - top_eig_exact) / abs(top_eig_exact)

    print(f"\nLeading Eigenvalue Estimation:")
    print(f"  Power iteration (HVP only): {top_eig_power:.6f}")
    print(f"  numpy.linalg.eigvalsh:      {top_eig_exact:.6f}")
    print(f"  Relative error:             {top_rel_err:.4e}")

    # 4. Most negative eigenvalue to demonstrate non-convexity
    most_neg_eig = float(np.min(eigvals))
    print(f"\nNon-convexity check at random point:")
    print(f"  Most negative eigenvalue:   {most_neg_eig:.6f}")
    if most_neg_eig < 0:
        print("  -> Confirmed: Hessian has negative eigenvalues, exhibiting non-convex curvature.")
    else:
        print("  -> Hessian is positive semi-definite at this local point.")

    return max_rel_err, top_eig_power, top_eig_exact, most_neg_eig


# ===========================================================================
# Part (b): Gauss-Newton & Levenberg-Marquardt
# ===========================================================================

def exp6_part_b_gauss_newton():
    print("\n" + "=" * 75)
    print("PART (B): Gauss-Newton & Levenberg-Marquardt")
    print("=" * 75)

    # Ground truth model: y = a * exp(b * x), true a=2.0, b=-1.3
    # 40 points in [0, 2], noise std 0.05, seed 0
    rng = np.random.default_rng(0)
    x_data = np.linspace(0.0, 2.0, 40)
    a_true, b_true = 2.0, -1.3
    y_clean = a_true * np.exp(b_true * x_data)
    y_data = y_clean + rng.normal(0.0, 0.05, size=40)

    def residual(th: np.ndarray) -> np.ndarray:
        a, b = th[0], th[1]
        return a * np.exp(b * x_data) - y_data

    def jacobian(th: np.ndarray) -> np.ndarray:
        a, b = th[0], th[1]
        exp_bx = np.exp(b * x_data)
        J = np.empty((40, 2), dtype=float)
        J[:, 0] = exp_bx
        J[:, 1] = a * x_data * exp_bx
        return J

    def exact_hessian(th: np.ndarray) -> np.ndarray:
        a, b = th[0], th[1]
        r = residual(th)
        J = jacobian(th)
        JTJ = J.T @ J
        # Second derivatives: d2r/da2=0, d2r/dadb=x*exp(bx), d2r/db2=a*x^2*exp(bx)
        exp_bx = np.exp(b * x_data)
        H2 = np.zeros((2, 2), dtype=float)
        H2[0, 1] = H2[1, 0] = np.sum(r * x_data * exp_bx)
        H2[1, 1] = np.sum(r * a * (x_data ** 2) * exp_bx)
        return JTJ + H2

    # Independent reference from scipy
    scipy_res = least_squares(residual, x0=[1.0, -0.5], jac=jacobian, ftol=1e-12, xtol=1e-12, gtol=1e-8)
    scipy_cost = 0.5 * float(np.sum(scipy_res.fun ** 2))
    print(f"scipy.optimize.least_squares ground truth cost: {scipy_cost:.10e}")
    print(f"scipy fitted parameters (a, b): ({scipy_res.x[0]:.4f}, {scipy_res.x[1]:.4f})\n")

    # Start 1: Standard start (a=1.0, b=-0.5)
    th0 = np.array([1.0, -0.5])
    tol = 1e-8
    max_iter = 2000

    results_b = []

    # 1. Gradient Descent
    t0 = time.perf_counter()
    th_gd = th0.copy()
    lr_gd = 0.02
    for it_gd in range(max_iter):
        r = residual(th_gd)
        J = jacobian(th_gd)
        g = J.T @ r
        if np.linalg.norm(g) < tol:
            break
        th_gd -= lr_gd * g
    t_gd = time.perf_counter() - t0
    cost_gd = 0.5 * float(np.sum(residual(th_gd) ** 2))
    results_b.append({
        "Method": "Gradient Descent",
        "Start": "(1.0, -0.5)",
        "Iterations": it_gd + 1,
        "Final Cost": cost_gd,
        "Grad Norm": float(np.linalg.norm(jacobian(th_gd).T @ residual(th_gd))),
        "Runtime (s)": t_gd,
        "Status": "Converged" if it_gd < max_iter - 1 else "Capped",
    })

    # 2. Exact Newton
    t0 = time.perf_counter()
    th_newton = th0.copy()
    for it_newton in range(max_iter):
        r = residual(th_newton)
        J = jacobian(th_newton)
        g = J.T @ r
        if np.linalg.norm(g) < tol:
            break
        H = exact_hessian(th_newton)
        delta = np.linalg.solve(H, -g)
        th_newton += delta
    t_newton = time.perf_counter() - t0
    cost_newton = 0.5 * float(np.sum(residual(th_newton) ** 2))
    results_b.append({
        "Method": "Exact Newton",
        "Start": "(1.0, -0.5)",
        "Iterations": it_newton + 1,
        "Final Cost": cost_newton,
        "Grad Norm": float(np.linalg.norm(jacobian(th_newton).T @ residual(th_newton))),
        "Runtime (s)": t_newton,
        "Status": "Converged" if it_newton < max_iter - 1 else "Capped",
    })

    # 3. Plain Gauss-Newton
    t0 = time.perf_counter()
    gn_res = gauss_newton(residual, jacobian, th0, lam=0.0, max_iter=max_iter, tol=tol, lm=False)
    t_gn = time.perf_counter() - t0
    results_b.append({
        "Method": "Gauss-Newton",
        "Start": "(1.0, -0.5)",
        "Iterations": gn_res["iters"] + 1,
        "Final Cost": gn_res["cost"],
        "Grad Norm": gn_res["grad_norm"],
        "Runtime (s)": t_gn,
        "Status": "Converged" if gn_res["converged"] else "Capped",
    })

    # 4. Levenberg-Marquardt
    t0 = time.perf_counter()
    lm_res = gauss_newton(residual, jacobian, th0, lam=1e-2, max_iter=max_iter, tol=tol, lm=True)
    t_lm = time.perf_counter() - t0
    results_b.append({
        "Method": "Levenberg-Marquardt",
        "Start": "(1.0, -0.5)",
        "Iterations": lm_res["iters"] + 1,
        "Final Cost": lm_res["cost"],
        "Grad Norm": lm_res["grad_norm"],
        "Runtime (s)": t_lm,
        "Status": "Converged" if lm_res["converged"] else "Capped",
    })

    # Start 2: Poor start (a=5.0, b=3.0)
    th_poor = np.array([5.0, 3.0])
    t0 = time.perf_counter()
    gn_poor = gauss_newton(residual, jacobian, th_poor, lam=0.0, max_iter=200, tol=tol, lm=False)
    t_gn_poor = time.perf_counter() - t0

    t0 = time.perf_counter()
    lm_poor = gauss_newton(residual, jacobian, th_poor, lam=1.0, max_iter=500, tol=tol, lm=True)
    t_lm_poor = time.perf_counter() - t0

    status_gn_poor = "Converged (Optimal)" if abs(gn_poor["cost"] - scipy_cost) < 1e-4 else "Stalled at plateau (sub-optimal)"
    status_lm_poor = "Converged (Optimal)" if abs(lm_poor["cost"] - scipy_cost) < 1e-4 else "Stalled / Failed"

    results_b.append({
        "Method": "Gauss-Newton (poor start)",
        "Start": "(5.0, 3.0)",
        "Iterations": gn_poor["iters"],
        "Final Cost": gn_poor["cost"],
        "Grad Norm": gn_poor["grad_norm"],
        "Runtime (s)": t_gn_poor,
        "Status": status_gn_poor,
    })
    results_b.append({
        "Method": "Levenberg-Marquardt (poor start)",
        "Start": "(5.0, 3.0)",
        "Iterations": lm_poor["iters"],
        "Final Cost": lm_poor["cost"],
        "Grad Norm": lm_poor["grad_norm"],
        "Runtime (s)": t_lm_poor,
        "Status": status_lm_poor,
    })

    df_b = pd.DataFrame(results_b)
    print("Non-Linear Least Squares Comparison:")
    print(df_b.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print(f"\nVerification against SciPy: abs(Gauss-Newton cost - SciPy cost) = {abs(gn_res['cost'] - scipy_cost):.4e}")

    return df_b


# ===========================================================================
# Part (c): Fisher Preconditioning on make_moons
# ===========================================================================

def exp6_part_c_fisher_preconditioning():
    print("\n" + "=" * 75)
    print("PART (C): Fisher Preconditioning on make_moons Network")
    print("=" * 75)

    X_all, y_all = make_moons(n_samples=300, noise=0.2, random_state=0)
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(300)
    X_train, y_train = X_all[perm[:200]], y_all[perm[:200]]
    X_test, y_test = X_all[perm[200:]], y_all[perm[200:]]

    d, h, k = 2, 8, 2
    params0 = init_params(d, h, k, seed=42)
    theta0, shapes = flatten_params(params0)
    dim = theta0.shape[0]

    # Warm-up: 100 steps of plain GD
    theta = theta0.copy()
    warmup_lr = 0.1
    for _ in range(100):
        p = unflatten_params(theta, shapes)
        _, g_dict = loss_and_grads(p, X_train, y_train, l2=0.0)
        g_flat, _ = flatten_params(g_dict)
        theta -= warmup_lr * g_flat

    theta_warmup = theta.copy()
    p_warmup = unflatten_params(theta_warmup, shapes)
    warmup_loss, _ = loss_and_grads(p_warmup, X_train, y_train, l2=0.0)
    print(f"End of 100-step GD warm-up: train loss = {warmup_loss:.4f}")

    # Compute empirical Fisher F at theta_warmup
    G_warmup = per_sample_grads(p_warmup, X_train, y_train, l2=0.0)
    F_full = empirical_fisher(per_sample_g=G_warmup)
    norm_F_full = float(np.linalg.norm(F_full, "fro"))
    rank_F = int(np.linalg.matrix_rank(F_full))

    print(f"Empirical Fisher Matrix: Size = ({dim}, {dim}), Numerical Rank = {rank_F}")
    print("NOTE: The EMPIRICAL Fisher is the sample covariance of gradients at current labels;")
    print("      it is NOT the true Fisher (expectation over model) nor the Gauss-Newton matrix,")
    print("      and serves strictly as an empirical curvature approximation.\n")

    # Construct Fisher approximations
    F_diag = np.diag(np.diag(F_full))
    F_rank3 = low_rank_fisher(F_full, r=3)
    F_rank8 = low_rank_fisher(F_full, r=8)

    preconditioners = {
        "Plain GD": None,
        "Diag-Fisher": np.diag(F_full),
        "Low-Rank (r=3)": F_rank3,
        "Low-Rank (r=8)": F_rank8,
        "Full Empirical Fisher": F_full,
    }

    # Sweep learning rate for each preconditioner over 200 further steps
    lr_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    eval_budget = 200
    damping = 1e-3

    results_c = []

    for name, F_approx in preconditioners.items():
        # Compute relative Frobenius error
        if F_approx is None:
            frob_rel_err = float("nan")
        elif F_approx.ndim == 1:
            frob_rel_err = float(np.linalg.norm(np.diag(F_approx) - F_full, "fro") / norm_F_full)
        else:
            frob_rel_err = float(np.linalg.norm(F_approx - F_full, "fro") / norm_F_full)

        best_loss = float("inf")
        best_lr = None
        best_theta = None
        best_step_time = 0.0

        for lr in lr_grid:
            th = theta_warmup.copy()
            t0 = time.perf_counter()
            for step_i in range(eval_budget):
                p = unflatten_params(th, shapes)
                _, g_dict = loss_and_grads(p, X_train, y_train, l2=0.0)
                g_flat, _ = flatten_params(g_dict)

                if F_approx is None:
                    th -= lr * g_flat
                else:
                    th = preconditioned_gradient_step(th, g_flat, F_approx, lr=lr, damping=damping)
            
            step_time = (time.perf_counter() - t0) / eval_budget
            p_final = unflatten_params(th, shapes)
            loss_final, _ = loss_and_grads(p_final, X_train, y_train, l2=0.0)

            if np.isfinite(loss_final) and loss_final < best_loss:
                best_loss = float(loss_final)
                best_lr = lr
                best_theta = th.copy()
                best_step_time = step_time

        # Test accuracy with best lr
        p_best = unflatten_params(best_theta, shapes)
        hidden_test = np.tanh(X_test @ p_best["W1"] + p_best["b1"])
        logits_test = hidden_test @ p_best["W2"] + p_best["b2"]
        test_acc = float(np.mean(np.argmax(logits_test, axis=1) == y_test))

        results_c.append({
            "Method": name,
            "Tuned LR": best_lr,
            "Train Loss": best_loss,
            "Test Accuracy": test_acc,
            "Step Time (s)": best_step_time,
            "Matrix Size": f"{dim}x{dim}",
            "Approx Rank": 1 if name == "Diag-Fisher" else (3 if "r=3" in name else (8 if "r=8" in name else (rank_F if "Full" in name else 0))),
            "Rel Frobenius Error": frob_rel_err,
        })

    df_c = pd.DataFrame(results_c)
    print("Fisher Preconditioning (200 steps after warm-up):")
    print(df_c.to_string(index=False, float_format=lambda x: f"{x:.4e}" if isinstance(x, float) and abs(x) < 1e-2 else f"{x:.4f}"))
    
    return df_c


def main() -> None:
    exp6_part_a_hvp_check()
    df_b = exp6_part_b_gauss_newton()
    df_c = exp6_part_c_fisher_preconditioning()

    df_b.to_csv(RESULTS / "trackA_exp6_gauss_newton.csv", index=False)
    df_c.to_csv(RESULTS / "trackA_exp6_fisher_precond.csv", index=False)
    print("\nSaved curvature experiment results to results/trackA_exp6_*")


if __name__ == "__main__":
    main()
