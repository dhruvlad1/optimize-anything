"""trackC_quantum_opt/tests/test_qaoa.py
--------------------------------------
Unit test suite for QAOA statevector simulator, parameter-shift gradients,
sampling, and optimizers.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from trackB_discrete.exp2_qubo import maxcut_brute_force
from trackC_quantum_opt.optimizers_q import cobyla_wrapper, spsa
from trackC_quantum_opt.qaoa_sim import (
    approximation_ratio,
    expected_cut,
    finite_diff_grad,
    maxcut_diagonal,
    param_shift_grad,
    probabilities,
    qaoa_state,
    sample,
)


def test_triangle_diagonal_by_hand():
    """Triangle graph n=3, edges (0,1), (1,2), (0,2) all with weight 1."""
    edges = [(0, 1, 1), (1, 2, 1), (0, 2, 1)]
    diag = maxcut_diagonal(3, edges)
    # Bit strings (MSB):
    # 000 (idx 0): 0
    # 001 (idx 1): cut (0,2), (1,2) -> 2
    # 010 (idx 2): cut (0,1), (1,2) -> 2
    # 011 (idx 3): cut (0,1), (0,2) -> 2
    # 100 (idx 4): cut (0,1), (0,2) -> 2
    # 101 (idx 5): cut (0,1), (1,2) -> 2
    # 110 (idx 6): cut (0,2), (1,2) -> 2
    # 111 (idx 7): 0
    expected = np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(diag, expected)
    assert np.max(diag) == 2.0


def test_max_diagonal_matches_brute_force_5graphs():
    """Check max(maxcut_diagonal) matches Track B maxcut_brute_force across 5 random graphs."""
    rng = np.random.default_rng(12345)
    n = 6
    for seed in range(5):
        W_raw = rng.integers(1, 4, (n, n))
        W = (W_raw + W_raw.T) // 2
        np.fill_diagonal(W, 0)
        edges = [(i, j, int(W[i, j])) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]

        diag = maxcut_diagonal(n, edges)
        bf_cut, _, _ = maxcut_brute_force(W)
        assert np.isclose(np.max(diag), bf_cut)


def test_numpy_sim_matches_pennylane():
    """Verify numpy qaoa_sim expected cut matches PennyLane default.qubit."""
    n = 4
    edges = [(0, 1, 2), (1, 2, 1), (2, 3, 3), (0, 3, 1)]
    diag = maxcut_diagonal(n, edges)

    # Random parameters
    gammas = [0.42]
    betas = [0.35]
    params = np.concatenate([gammas, betas])

    exp_numpy = expected_cut(params, diag, n)

    # Build PennyLane circuit
    dev = qml.device("default.qubit", wires=n)
    coeffs = [w / 2.0 for _, _, w in edges] + [-w / 2.0 for _, _, w in edges]
    obs = [qml.Identity(0) for _ in edges] + [qml.PauliZ(u) @ qml.PauliZ(v) for u, v, _ in edges]
    cost_ham = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev)
    def pl_circuit(g, b):
        for i in range(n):
            qml.Hadamard(wires=i)
        for u, v, w in edges:
            qml.IsingZZ(-g[0] * w, wires=[u, v])
        for i in range(n):
            qml.RX(2 * b[0], wires=i)
        return qml.expval(cost_ham)

    exp_pl = float(pl_circuit(gammas, betas))
    assert np.isclose(exp_numpy, exp_pl, atol=1e-9)


def test_per_gate_ps_matches_fd():
    """Verify per-gate parameter-shift gradient matches finite difference and naive shift fails."""
    n = 4
    edges = [(0, 1, 2), (1, 2, 1), (2, 3, 3), (0, 3, 1)]
    diag = maxcut_diagonal(n, edges)
    p = 2
    params = np.array([0.3, 0.6, 0.4, 0.5])  # gamma_0, gamma_1, beta_0, beta_1

    ps_grad = param_shift_grad(params, diag, n, edges)
    fd_grad = finite_diff_grad(params, diag, n, eps=1e-6)

    np.testing.assert_allclose(ps_grad, fd_grad, atol=1e-5)

    # Naive parameter-shift on the first parameter (gamma_0 shifted as a whole):
    shift = np.pi / 2.0
    p_plus = params.copy()
    p_plus[0] += shift
    p_minus = params.copy()
    p_minus[0] -= shift
    naive_grad_gamma0 = (expected_cut(p_plus, diag, n) - expected_cut(p_minus, diag, n)) / (2.0 * shift)

    # Because edge weights are 1, 2, 3, naive shift does not equal true gradient
    assert abs(naive_grad_gamma0 - ps_grad[0]) > 0.5


def test_one_qubit_rx_ps_exact():
    """Verify parameter-shift gradient for 2-node single-edge system matches FD to high precision."""
    n = 2
    edges = [(0, 1, 1)]
    diag = maxcut_diagonal(n, edges)
    params = np.array([np.pi / 4.0, np.pi / 4.0])

    ps_grad = param_shift_grad(params, diag, n, edges)
    fd_grad = finite_diff_grad(params, diag, n, eps=1e-6)
    np.testing.assert_allclose(ps_grad, fd_grad, atol=1e-6)


def test_spsa_is_deterministic_and_reduces_objective():
    """SPSA should be strictly deterministic for a fixed seed and reduce the loss."""
    n = 4
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)]
    diag = maxcut_diagonal(n, edges)
    x0 = np.array([0.1, 0.1])

    loss_fn = lambda p: -expected_cut(p, diag, n)
    init_loss = loss_fn(x0)

    res1 = spsa(loss_fn, x0, budget=60, seed=42)
    res2 = spsa(loss_fn, x0, budget=60, seed=42)

    np.testing.assert_allclose(res1.x, res2.x)
    assert res1.fun == res2.fun
    assert res1.fun < init_loss


def test_shot_estimator_unbiased():
    """Sample mean across independent repetitions matches the exact expectation within 4 SEs."""
    n = 4
    edges = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (0, 3, 1)]
    diag = maxcut_diagonal(n, edges)
    params = np.array([0.5, 0.5])
    exact = expected_cut(params, diag, n)

    rng = np.random.default_rng(42)
    n_reps = 200
    shots_small = 20
    means = [float(np.mean(sample(params, diag, n, shots=shots_small, rng=rng))) for _ in range(n_reps)]
    overall_mean = float(np.mean(means))
    se = float(np.std(means, ddof=1) / np.sqrt(n_reps))

    assert abs(overall_mean - exact) < 4.0 * se

    # Variance decreases with shots:
    samples_100 = [float(np.mean(sample(params, diag, n, shots=100, rng=rng))) for _ in range(50)]
    samples_5000 = [float(np.mean(sample(params, diag, n, shots=5000, rng=rng))) for _ in range(50)]
    assert np.std(samples_5000) < np.std(samples_100)


def test_depolarizing_q0_matches_noiseless():
    """PennyLane default.mixed with q=0 matches noiseless simulator."""
    n = 4
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    diag = maxcut_diagonal(n, edges)
    p = 1
    params = np.array([0.4, 0.3])

    exp_noiseless = expected_cut(params, diag, n)

    dev_mixed = qml.device("default.mixed", wires=n)
    coeffs = [w / 2.0 for _, _, w in edges] + [-w / 2.0 for _, _, w in edges]
    obs = [qml.Identity(0) for _ in edges] + [qml.PauliZ(u) @ qml.PauliZ(v) for u, v, _ in edges]
    cost_ham = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev_mixed)
    def noisy_circuit(g, b, q):
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p):
            for u, v, w in edges:
                qml.IsingZZ(-g[l] * w, wires=[u, v])
            if q > 0:
                for i in range(n):
                    qml.DepolarizingChannel(q, wires=i)
            for i in range(n):
                qml.RX(2 * b[l], wires=i)
            if q > 0:
                for i in range(n):
                    qml.DepolarizingChannel(q, wires=i)
        return qml.expval(cost_ham)

    exp_noisy_0 = float(noisy_circuit(params[:p], params[p:], 0.0))
    assert np.isclose(exp_noiseless, exp_noisy_0, atol=1e-8)


def test_depolarizing_q005_lowers_ratio():
    """Depolarizing noise q=0.05 lowers the expected cut value."""
    n = 4
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    diag = maxcut_diagonal(n, edges)
    p = 1
    # Find optimized noiseless parameters
    res = cobyla_wrapper(lambda p_arr: -expected_cut(p_arr, diag, n), np.array([0.4, 0.3]), budget=50)
    best_params = res.x
    val_noiseless = expected_cut(best_params, diag, n)

    dev_mixed = qml.device("default.mixed", wires=n)
    coeffs = [w / 2.0 for _, _, w in edges] + [-w / 2.0 for _, _, w in edges]
    obs = [qml.Identity(0) for _ in edges] + [qml.PauliZ(u) @ qml.PauliZ(v) for u, v, _ in edges]
    cost_ham = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev_mixed)
    def noisy_circuit(g, b, q):
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p):
            for u, v, w in edges:
                qml.IsingZZ(-g[l] * w, wires=[u, v])
            if q > 0:
                for i in range(n):
                    qml.DepolarizingChannel(q, wires=i)
            for i in range(n):
                qml.RX(2 * b[l], wires=i)
            if q > 0:
                for i in range(n):
                    qml.DepolarizingChannel(q, wires=i)
        return qml.expval(cost_ham)

    val_noisy = float(noisy_circuit(best_params[:p], best_params[p:], 0.05))
    assert val_noisy < val_noiseless
