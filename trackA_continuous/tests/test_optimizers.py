"""
trackA_continuous/tests/test_optimizers.py
------------------------------------------
Pytest tests for Track A objectives and optimizers.
All expected values are derived independently (hand computation, finite differences, scipy).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from trackA_continuous.objectives import (
    quadratic, quartic, saddle, himmelblau, rosenbrock, rastrigin,
)
from trackA_continuous.optimizers import (
    gd, sgd, momentum, nag, adagrad, rmsprop, adam, newton,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_grad(f, x, h=1e-5):
    """Central finite-difference gradient."""
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        e = np.zeros_like(x, dtype=float)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def numerical_hess(grad_fn, x, h=1e-5):
    """Finite-difference Hessian via central differences of gradient."""
    d = len(x)
    H = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        H[:, i] = (grad_fn(x + e) - grad_fn(x - e)) / (2 * h)
    return H


def make_noisy_grad(grad_fn, noise_std=0.1, seed=0):
    """Return a stochastic gradient oracle wrapping a deterministic gradient."""
    def sg(x, rng):
        return grad_fn(x) + rng.normal(0, noise_std, size=x.shape)
    return sg


# ---------------------------------------------------------------------------
# 1. Gradient checks for all objectives
# ---------------------------------------------------------------------------

OBJECTIVES_TO_CHECK = [
    quadratic(kappa=10, dim=5),
    quartic(dim=4),
    saddle(),
    himmelblau(),
    rosenbrock(),
    rastrigin(dim=2),
]

TEST_POINTS = [
    np.array([0.5, -0.3, 1.2, -0.8, 0.1]),
    np.array([0.5, -0.3, 1.2, -0.8]),
    np.array([0.3, 0.7]),
    np.array([1.0, 2.0]),
    np.array([-0.5, 0.8]),
    np.array([0.3, -0.5]),
]


@pytest.mark.parametrize("obj,x", list(zip(OBJECTIVES_TO_CHECK, TEST_POINTS)))
def test_gradient_check(obj, x):
    """Analytic gradient matches central finite differences."""
    ag = obj.grad(x)
    ng = numerical_grad(obj.f, x)
    np.testing.assert_allclose(ag, ng, rtol=1e-4, atol=1e-6,
                               err_msg=f"Gradient check failed for {obj.name}")


@pytest.mark.parametrize("obj,x", list(zip(OBJECTIVES_TO_CHECK, TEST_POINTS)))
def test_hessian_check(obj, x):
    """Analytic Hessian matches finite-difference Hessian (where available)."""
    if obj.hess is None:
        pytest.skip("No analytic Hessian")
    ah = obj.hess(x)
    nh = numerical_hess(obj.grad, x)
    np.testing.assert_allclose(ah, nh, rtol=1e-4, atol=1e-5,
                               err_msg=f"Hessian check failed for {obj.name}")


# Test at a second set of random points
@pytest.mark.parametrize("obj", OBJECTIVES_TO_CHECK)
def test_gradient_check_random(obj):
    """Gradient check at 3 random points."""
    rng = np.random.default_rng(7)
    for _ in range(3):
        x = rng.uniform(-1, 1, size=obj.dim)
        ag = obj.grad(x)
        ng = numerical_grad(obj.f, x)
        np.testing.assert_allclose(ag, ng, rtol=1e-4, atol=1e-6,
                                   err_msg=f"Random grad check failed for {obj.name} at x={x}")


# ---------------------------------------------------------------------------
# 2. Hand-computed one-step GD on a 1-D quadratic
#    f(x) = 0.5 * 4 * x^2,  grad = 4x,  x0 = 1.0,  lr = 0.1
#    x1 = x0 - lr * grad(x0) = 1.0 - 0.1 * 4.0 = 0.6
# ---------------------------------------------------------------------------

def test_gd_one_step_1d():
    obj = quadratic(kappa=4.0, dim=1)
    # eigenvalue of the 1-D quadratic is exactly 4 (log-space gives 4)
    x0 = np.array([1.0])
    lr = 0.1
    res = gd(obj.f, obj.grad, x0, lr=lr, max_iter=1, gtol=0.0)
    expected = x0[0] - lr * obj.grad(x0)[0]
    np.testing.assert_allclose(res["path"][1, 0], expected, rtol=1e-10,
                               err_msg="GD one-step 1-D hand check failed")


# ---------------------------------------------------------------------------
# 3. Pure Newton solves quadratic exactly in ONE step
#    For f(x)=0.5 x^T A x, Newton step: p = A^{-1} A x = x, so x1 = x0 - x0 = 0
# ---------------------------------------------------------------------------

def test_newton_quadratic_one_step():
    obj = quadratic(kappa=10, dim=5)
    x0 = np.array([1.0, 2.0, -1.0, 0.5, -0.3])
    res = newton(obj.f, obj.grad, obj.hess, x0, lr=1.0, mode="pure", max_iter=1, gtol=0.0)
    np.testing.assert_allclose(res["path"][-1], np.zeros(5), atol=1e-10,
                               err_msg="Newton did not solve quadratic in one step")


# ---------------------------------------------------------------------------
# 4. GD with step 1/L on quadratic(kappa=10) reaches minimizer
# ---------------------------------------------------------------------------

def test_gd_converges_quadratic():
    obj = quadratic(kappa=10, dim=10)
    x0 = np.ones(10)
    lr = 1.0 / obj.L
    res = gd(obj.f, obj.grad, x0, lr=lr, max_iter=5000, gtol=1e-8)
    assert res["fvals"][-1] < 1e-10, f"GD did not converge: f={res['fvals'][-1]}"


# ---------------------------------------------------------------------------
# 5. Every deterministic optimizer reduces the objective on quadratic(kappa=10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opt_fn,kwargs", [
    (gd,       {"lr": 0.05}),
    (momentum, {"lr": 0.01, "beta": 0.9}),
    (nag,      {"lr": 0.01, "beta": 0.9}),
    (adagrad,  {"lr": 0.5}),
    (rmsprop,  {"lr": 0.05}),
    (adam,     {"lr": 0.05}),
])
def test_optimizer_reduces_objective(opt_fn, kwargs):
    obj = quadratic(kappa=10, dim=5)
    x0 = np.array([1.0, -1.0, 2.0, -0.5, 0.3])
    res = opt_fn(obj.f, obj.grad, x0, max_iter=500, gtol=0.0, **kwargs)
    assert res["fvals"][-1] < res["fvals"][0], \
        f"{opt_fn.__name__} did not reduce objective"


def test_newton_reduces_objective():
    obj = quadratic(kappa=10, dim=5)
    x0 = np.array([1.0, -1.0, 2.0, -0.5, 0.3])
    res = newton(obj.f, obj.grad, obj.hess, x0, lr=1.0, mode="pure", max_iter=50)
    assert res["fvals"][-1] < res["fvals"][0]


# ---------------------------------------------------------------------------
# 6. SGD reproducibility: same seed -> same result; different seeds -> differ
# ---------------------------------------------------------------------------

def test_sgd_reproducibility():
    obj = quadratic(kappa=10, dim=5)
    x0 = np.ones(5)
    sg = make_noisy_grad(obj.grad, noise_std=0.5, seed=0)

    res1 = sgd(obj.f, sg, x0, lr=0.01, max_iter=100, seed=1)
    res2 = sgd(obj.f, sg, x0, lr=0.01, max_iter=100, seed=1)
    res3 = sgd(obj.f, sg, x0, lr=0.01, max_iter=100, seed=99)

    np.testing.assert_array_equal(res1["path"], res2["path"],
                                   err_msg="Same seed should give identical paths")
    assert not np.array_equal(res1["path"], res3["path"]), \
        "Different seeds should give different paths"


# ---------------------------------------------------------------------------
# 7. Scipy BFGS reaches the known Rosenbrock minimum (reference check)
# ---------------------------------------------------------------------------

def test_scipy_rosenbrock_reference():
    obj = rosenbrock()
    x0 = np.array([-0.5, 0.5])
    result = scipy_minimize(obj.f, x0, jac=obj.grad, method="BFGS",
                            options={"gtol": 1e-10, "maxiter": 10000})
    assert result.fun < 1e-10, f"Scipy BFGS did not reach Rosenbrock minimum: f={result.fun}"
    np.testing.assert_allclose(result.x, obj.x_star, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. GD tail gap contraction rate on quadratic(kappa=10) matches (1 - 1/10)^2
# ---------------------------------------------------------------------------

def test_gd_tail_rate_quadratic_10():
    """Measured GD tail gap rate on quadratic(kappa=10) matches (1-1/10)^2 within 0.03."""
    obj = quadratic(kappa=10, dim=10)
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(10)
    x0 = 5.0 * x0 / np.linalg.norm(x0)

    lr = 1.0 / obj.L
    res = gd(obj.f, obj.grad, x0, lr=lr, max_iter=2000, gtol=1e-15)
    fvals = np.array(res["fvals"])

    # Tail window: last half of iterations before reaching 1e-12
    hit = np.where(fvals <= 1e-12)[0]
    end_idx = hit[0] if len(hit) > 0 else len(fvals)
    start_idx = end_idx // 2

    ks = np.arange(start_idx, end_idx)
    log_gaps = np.log(fvals[ks])
    A = np.vstack([ks, np.ones_like(ks)]).T
    slope, _ = np.linalg.lstsq(A, log_gaps, rcond=None)[0]
    emp_rate = float(np.exp(slope))
    theory_rate = float((1.0 - 1.0 / 10.0) ** 2)  # 0.81

    assert abs(emp_rate - theory_rate) < 0.03, (
        f"Empirical rate {emp_rate:.4f} differs from theoretical {theory_rate:.4f} by more than 0.03"
    )


# ---------------------------------------------------------------------------
# 9. Gradient of finite-sum least-squares loss at lstsq solution is below 1e-8
# ---------------------------------------------------------------------------

def test_lstsq_solution_gradient():
    """Gradient of the finite-sum loss at the numpy lstsq solution is below 1e-8."""
    n_samples, n_features = 2000, 20
    rng_data = np.random.default_rng(99)
    A = rng_data.standard_normal((n_samples, n_features))
    x_true = rng_data.standard_normal(n_features)
    b = A @ x_true + 0.01 * rng_data.standard_normal(n_samples)

    x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    grad = A.T @ (A @ x_lstsq - b) / n_samples
    norm_grad = float(np.linalg.norm(grad))

    assert norm_grad < 1e-8, (
        f"Gradient norm at lstsq solution is {norm_grad:.3e}, expected < 1e-8"
    )


# ---------------------------------------------------------------------------
# 10. Momentum (beta > 0) produces a different path from GD on a 2-D quadratic
# ---------------------------------------------------------------------------

def test_momentum_path_differs_from_gd_on_quadratic():
    """Momentum (beta > 0) produces a different path from GD at the same learning rate on a 2-D quadratic."""
    obj = quadratic(kappa=10, dim=2)
    x0 = np.array([2.0, -1.5])
    lr = 0.05
    beta = 0.9
    res_gd = gd(obj.f, obj.grad, x0, lr=lr, max_iter=25, gtol=0.0)
    res_mom = momentum(obj.f, obj.grad, x0, lr=lr, beta=beta, max_iter=25, gtol=0.0)

    max_diff = float(np.max(np.linalg.norm(res_gd["path"] - res_mom["path"], axis=1)))
    assert max_diff > 0.05, f"Momentum path did not differ significantly from GD (max diff = {max_diff})"


# ---------------------------------------------------------------------------
# 11. Saddle escape iteration count increases as y0 decreases
# ---------------------------------------------------------------------------

def test_saddle_escape_time_increases_as_y0_decreases():
    """Saddle escape iteration count increases as y0 decreases."""
    obj = saddle()
    lr = 0.05
    y0_vals = [1e-2, 1e-4, 1e-6]
    escape_iters = []

    for y0 in y0_vals:
        x = np.array([0.0, y0])
        k_esc = None
        for k in range(1, 10000):
            x = x - lr * obj.grad(x)
            if obj.f(x) < -0.9:
                k_esc = k
                break
        assert k_esc is not None, f"Failed to escape from y0={y0}"
        escape_iters.append(k_esc)

    assert escape_iters[0] < escape_iters[1] < escape_iters[2], (
        f"Escape iterations not monotonically increasing with 1/y0: {escape_iters}"
    )


