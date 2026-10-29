"""Unit tests for curvature methods, HVP, spectral analysis, Fisher matrices, and Gauss-Newton."""

import numpy as np
import pytest

from trackA_continuous.compare.backprop_antigravity import init_params
from trackA_continuous.curvature import (
    empirical_fisher,
    flatten_params,
    gauss_newton,
    hvp,
    low_rank_fisher,
    per_sample_grads,
    power_iteration_top_eig,
    preconditioned_gradient_step,
)


def test_hvp_matches_explicit_hessian_on_quadratic():
    """HVP on a quadratic f(x) = 0.5 x^T A x + b^T x must equal A @ v exactly."""
    rng = np.random.default_rng(10)
    dim = 5
    M = rng.standard_normal((dim, dim))
    A = M.T @ M + np.eye(dim)  # Symmetric positive definite
    b = rng.standard_normal(dim)

    def grad_fn(x: np.ndarray) -> np.ndarray:
        return A @ x + b

    x0 = rng.standard_normal(dim)
    for _ in range(5):
        v = rng.standard_normal(dim)
        v = v / np.linalg.norm(v)

        approx_hv = hvp(grad_fn, x0, v, eps=1e-5)
        exact_hv = A @ v
        rel_err = np.linalg.norm(approx_hv - exact_hv) / np.linalg.norm(exact_hv)
        assert rel_err < 1e-6, f"HVP relative error {rel_err:.2e} exceeded tolerance."


def test_power_iteration_matches_eigvalsh():
    """Power iteration on a symmetric matrix must match top eigenvalue from eigvalsh."""
    rng = np.random.default_rng(20)
    dim = 6
    M = rng.standard_normal((dim, dim))
    S = M.T @ M + 2.0 * np.eye(dim)  # PSD symmetric matrix

    def hvp_fn(v: np.ndarray) -> np.ndarray:
        return S @ v

    top_eig_power, _ = power_iteration_top_eig(hvp_fn, dim=dim, iters=100, seed=42)
    exact_eigvals = np.linalg.eigvalsh(S)
    top_eig_exact = exact_eigvals[-1]

    rel_err = abs(top_eig_power - top_eig_exact) / top_eig_exact
    assert rel_err < 1e-4, f"Power iteration eigenvalue error {rel_err:.2e} exceeded tolerance."


def test_gauss_newton_solves_linear_least_squares_in_one_step():
    """Gauss-Newton on linear least squares ||A x - b||^2 must reach exact solution in 1 step."""
    rng = np.random.default_rng(30)
    m, d = 25, 4
    A = rng.standard_normal((m, d))
    b = rng.standard_normal(m)

    def residual(th: np.ndarray) -> np.ndarray:
        return A @ th - b

    def jacobian(th: np.ndarray) -> np.ndarray:
        return A

    theta0 = np.zeros(d)
    res = gauss_newton(residual, jacobian, theta0, lam=0.0, max_iter=5, tol=1e-12, lm=False)
    
    # Compare with independent numpy.linalg.lstsq
    expected_theta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Must converge in 1 iteration
    assert res["iters"] == 1
    assert np.allclose(res["theta"], expected_theta, atol=1e-9)


def test_empirical_fisher_is_symmetric_and_psd():
    """Empirical Fisher matrix must be strictly symmetric and positive semi-definite."""
    d, h, k = 2, 4, 2
    params = init_params(d, h, k, seed=40)
    rng = np.random.default_rng(41)
    X = rng.standard_normal((15, d))
    y = rng.integers(0, k, size=15)

    F = empirical_fisher(params, X, y, l2=0.0)
    
    # Symmetry check
    assert np.allclose(F, F.T, atol=1e-12)

    # Positive semi-definiteness: all eigenvalues >= -1e-12
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals >= -1e-12), f"Found negative eigenvalue in Fisher: {np.min(eigvals)}"


def test_low_rank_fisher_rank_and_monotone_error():
    """low_rank_fisher(F, r) must have numerical rank r, and approximation error must decrease as r increases."""
    rng = np.random.default_rng(50)
    dim = 10
    M = rng.standard_normal((dim, dim))
    F = M.T @ M + 0.1 * np.eye(dim)  # Full rank PSD

    ranks = [1, 2, 4, 7]
    errors = []
    norm_F = np.linalg.norm(F, "fro")

    for r in ranks:
        F_r = low_rank_fisher(F, r)
        # Check rank
        num_rank = np.linalg.matrix_rank(F_r, tol=1e-8)
        assert num_rank == r, f"Expected rank {r}, got {num_rank}"

        # Check PSD
        eigs_r = np.linalg.eigvalsh(F_r)
        assert np.all(eigs_r >= -1e-12)

        err = np.linalg.norm(F_r - F, "fro") / norm_F
        errors.append(err)

    # Error must be strictly decreasing
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1], f"Error did not decrease: {errors[i]} <= {errors[i+1]}"


def test_preconditioned_step_reduces_to_plain_gd():
    """When F_approx is Identity and damping is 0, preconditioned step must equal theta - lr * g."""
    dim = 6
    theta = np.array([1.0, -2.0, 3.5, 0.0, -1.2, 4.0])
    g = np.array([0.1, 0.5, -0.2, 0.8, -0.4, 0.3])
    lr = 0.05

    # Case 1: 1D identity diagonal
    F_diag = np.ones(dim)
    th_diag = preconditioned_gradient_step(theta, g, F_diag, lr=lr, damping=0.0)
    assert np.allclose(th_diag, theta - lr * g)

    # Case 2: 2D identity matrix
    F_eye = np.eye(dim)
    th_eye = preconditioned_gradient_step(theta, g, F_eye, lr=lr, damping=0.0)
    assert np.allclose(th_eye, theta - lr * g)
