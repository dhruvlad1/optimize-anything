"""trackD_parallel/tests/test_d2.py
------------------------------------
Test suite for Track D Part 2: Federated Learning and Multi-task Optimization.

All tests are fast (small data, few iterations).
"""

from __future__ import annotations

import numpy as np
import pytest

from trackD_parallel.federated import (
    make_clients,
    logistic_loss_and_grad,
    fedavg_full_batch,
    centralized_gd_steps,
    _sigmoid,
)
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


# ===========================================================================
# 1. Federated: gradient vs finite differences
# ===========================================================================

def test_federated_gradient_finite_diff():
    """Hand-written logistic gradient matches central finite differences."""
    rng = np.random.default_rng(99)
    n, d = 30, 10
    X = rng.standard_normal((n, d))
    y = (rng.random(n) > 0.5).astype(float)
    w = rng.standard_normal(d) * 0.3
    eps = 1e-5
    l2 = 1e-3

    loss, grad = logistic_loss_and_grad(w, X, y, l2)
    fd = np.zeros(d)
    for i in range(d):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps; wm[i] -= eps
        lp, _ = logistic_loss_and_grad(wp, X, y, l2)
        lm, _ = logistic_loss_and_grad(wm, X, y, l2)
        fd[i] = (lp - lm) / (2 * eps)

    max_abs = float(np.max(np.abs(grad - fd)))
    assert max_abs < 1e-5, f"Max abs gradient error {max_abs:.2e} > 1e-5"


# ===========================================================================
# 2. FedAvg E=1 full-batch == centralised GD
# ===========================================================================

def test_fedavg_equals_centralized_gd():
    """FedAvg(E=1, full-batch, full participation) == centralised GD (max diff < 1e-10)."""
    clients, _ = make_clients(K=4, d=8, n_per_client=40, heterogeneity=0.0, seed=5)
    rounds = 10
    lr = 0.03

    w_fed = fedavg_full_batch(clients, rounds=rounds, E=1, lr=lr)
    w_cen = centralized_gd_steps(clients, rounds=rounds, lr=lr)

    max_diff = float(np.max(np.abs(w_fed - w_cen)))
    assert max_diff < 1e-10, (
        f"FedAvg(E=1) vs centralised GD max diff = {max_diff:.2e} >= 1e-10"
    )


# ===========================================================================
# 3. Multitask backprop vs finite differences
# ===========================================================================

def test_multitask_backprop_finite_diff():
    """Hand-written backprop matches finite differences for every parameter group."""
    rng = np.random.default_rng(7)
    params = init_params(d_in=6, hidden=4, seed=7)
    n = 30
    X = rng.standard_normal((n, 6))
    y1 = rng.standard_normal(n)
    y2 = rng.standard_normal(n) * 5

    task_weights = (1.0, 1.0)
    grad = backprop(params, X, y1, y2, task_weights)

    eps = 1e-5
    for key in params:
        p_flat = params[key].ravel()
        g_flat = grad[key].ravel()
        fd_flat = np.zeros_like(p_flat)
        for i in range(len(p_flat)):
            p_plus = params[key].ravel().copy(); p_plus[i] += eps
            p_minus = params[key].ravel().copy(); p_minus[i] -= eps

            def _loss(pval):
                pcopy = {k: v.copy() for k, v in params.items()}
                pcopy[key] = pval.reshape(params[key].shape)
                L1, L2 = losses(pcopy, X, y1, y2)
                return task_weights[0] * L1 + task_weights[1] * L2

            fd_flat[i] = (_loss(p_plus) - _loss(p_minus)) / (2 * eps)

        max_err = float(np.max(np.abs(g_flat - fd_flat)))
        assert max_err < 1e-5, f"Backprop FD error for '{key}': {max_err:.2e}"


# ===========================================================================
# 4. MGDA closed form is global minimum of ||alpha*g1+(1-alpha)*g2||^2
# ===========================================================================

def test_mgda_closed_form_optimal():
    """alpha* from closed form minimises ||alpha*g1+(1-alpha)*g2||^2 for 20 random pairs."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        g1 = rng.standard_normal(15)
        g2 = rng.standard_normal(15)
        alpha_star = mgda_alpha(g1, g2)
        d_star = alpha_star * g1 + (1.0 - alpha_star) * g2
        norm_star = float(np.dot(d_star, d_star))

        for a in np.linspace(0.0, 1.0, 1001):
            d_test = a * g1 + (1.0 - a) * g2
            norm_test = float(np.dot(d_test, d_test))
            assert norm_test >= norm_star - 1e-10, (
                f"alpha*={alpha_star:.4f} is NOT the min-norm: "
                f"alpha={a:.4f} gives norm {norm_test:.6f} < {norm_star:.6f}"
            )


# ===========================================================================
# 5. MGDA returns near-zero direction for opposite equal-norm gradients
# ===========================================================================

def test_mgda_opposite_gradients():
    """MGDA on g and -g should return near-zero direction (Pareto-stationary)."""
    rng = np.random.default_rng(1)
    g = rng.standard_normal(10)
    g /= np.linalg.norm(g)
    alpha = mgda_alpha(g, -g)
    d = alpha * g + (1.0 - alpha) * (-g)
    norm = float(np.linalg.norm(d))
    assert norm < 1e-10, f"||d|| = {norm:.2e} for opposite gradients (expected < 1e-10)"


# ===========================================================================
# 6. GradNorm weights stay positive and sum to num_tasks
# ===========================================================================

def test_gradnorm_weights_valid():
    """GradNorm task weights stay positive and sum to 2 (num_tasks) throughout training."""
    X, y1, y2 = make_data(n=100, d_in=10, seed=0)
    params = init_params(d_in=10, hidden=8, seed=0)
    task_weights = np.array([1.0, 1.0])
    L1_0, L2_0 = losses(params, X, y1, y2)

    for _ in range(50):
        params, task_weights, _, _ = gradnorm_step(
            params, task_weights, X, y1, y2,
            lr=0.01, lr_weights=0.001, L1_0=L1_0, L2_0=L2_0
        )
        assert task_weights[0] > 0 and task_weights[1] > 0, \
            f"Negative weight: {task_weights}"
        total = float(task_weights[0] + task_weights[1])
        assert abs(total - 2.0) < 1e-10, \
            f"Weights sum = {total:.6f} (expected 2.0)"


# ===========================================================================
# 7. All methods decrease total loss within 200 steps
# ===========================================================================

def test_methods_decrease_loss():
    """equal, mgda, and gradnorm all decrease the total loss L1+L2 within 200 steps."""
    X, y1, y2 = make_data(n=200, d_in=10, seed=0)

    for method in ("equal", "mgda", "gradnorm"):
        params0 = init_params(d_in=10, hidden=8, seed=0)
        L1_0, L2_0 = losses(params0, X, y1, y2)
        init_total = L1_0 + L2_0

        hist = train(method, params0, X, y1, y2, steps=200, lr=0.01, lr_weights=0.001)
        final_total = hist["L1"][-1] + hist["L2"][-1]

        assert final_total < init_total, (
            f"Method '{method}' did NOT decrease total loss: "
            f"init={init_total:.4f} final={final_total:.4f}"
        )
