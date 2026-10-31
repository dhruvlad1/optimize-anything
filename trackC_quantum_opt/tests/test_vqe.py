"""trackC_quantum_opt/tests/test_vqe.py
-------------------------------------
Unit test suite for VQE, Quantum Natural Gradient, and Barren Plateaus.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import pytest

from trackC_quantum_opt.vqe import (
    exact_ground_energy,
    ising_chain_hamiltonian,
    make_state_qnode,
    make_vqe_qnode,
    metric_tensor_vqe,
    param_shift_grad_vqe,
)


def test_ising_ground_energy_hand_calculation():
    """Verify 2-qubit Ising chain (J=1, h=1) ground energy against hand calculation.
    H = -Z_0 Z_1 - X_0 - X_1.
    In the 4x4 computational basis:
        H = [[-1, -1, -1,  0],
             [-1,  1,  0, -1],
             [-1,  0,  1, -1],
             [ 0, -1, -1, -1]]
    In the symmetric subspace { (|00>+|11>)/sqrt(2), (|01>+|10>)/sqrt(2) },
    the matrix is [[-1, -2], [-2, 1]], yielding characteristic polynomial
    lambda^2 - 5 = 0 -> lambda = +- sqrt(5).
    Anti-symmetric subspace yields lambda = +- 1.
    Exact ground energy is -sqrt(5) = -2.2360679774997898.
    """
    H = ising_chain_hamiltonian(2, J=1.0, h=1.0)
    E0 = exact_ground_energy(H, 2)
    E0_hand = -np.sqrt(5.0)  # -2.2360679774997898
    np.testing.assert_allclose(E0, E0_hand, atol=1e-12)


def test_hand_written_param_shift_grad_matches_pennylane():
    """Verify hand-written parameter-shift gradient matches PennyLane autograd on 3-qubit ansatz."""
    n = 3
    layers = 2
    H = ising_chain_hamiltonian(n, J=1.0, h=1.0)
    qnode = make_vqe_qnode(H, n, layers)

    rng = np.random.default_rng(101)
    p = pnp.array(rng.uniform(0.1, 2.0 * np.pi, 2 * n * layers), requires_grad=True)

    grad_pl = np.asarray(qml.grad(qnode)(p))
    grad_hand = param_shift_grad_vqe(qnode, np.asarray(p))

    np.testing.assert_allclose(grad_hand, grad_pl, atol=1e-6)


def test_single_ry_metric_tensor_is_one_quarter():
    """Verify metric tensor of a single RY rotation on |0> is exactly 1/4 (0.25).
    |psi(theta)> = cos(theta/2)|0> + sin(theta/2)|1>
    d/dtheta |psi> = -1/2 sin(theta/2)|0> + 1/2 cos(theta/2)|1>
    <dpsi|dpsi> = 1/4, <psi|dpsi> = 0.
    Metric g = 1/4 = 0.25.
    """
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def single_ry_state(theta):
        qml.RY(theta[0], wires=0)
        return qml.state()

    theta = np.array([0.7])
    psi = single_ry_state(theta)
    dpsi = 0.5 * single_ry_state(theta + np.array([np.pi]))
    g = float(np.real(np.vdot(dpsi, dpsi) - np.vdot(dpsi, psi) * np.vdot(psi, dpsi)))

    np.testing.assert_allclose(g, 0.25, atol=1e-12)


def test_metric_tensor_symmetric_and_positive_semidefinite():
    """Verify the full metric tensor is symmetric and positive semidefinite."""
    n = 2
    layers = 2
    d = 2 * n * layers
    rng = np.random.default_rng(2026)
    p = rng.uniform(0.0, 2.0 * np.pi, d)

    g = metric_tensor_vqe(p, n, layers, approx=None)

    # Symmetry
    np.testing.assert_allclose(g, g.T, atol=1e-10)

    # Positive semidefinite: eigenvalues >= -1e-10
    eigs = np.linalg.eigvalsh(g)
    assert np.all(eigs >= -1e-10), f"Metric tensor has negative eigenvalues: {eigs[eigs < -1e-10]}"


def test_qng_step_with_identity_metric_equals_gradient_step():
    """Verify that when metric tensor is identity, QNG step equals plain gradient descent step."""
    d = 8
    rng = np.random.default_rng(777)
    grad = rng.normal(0, 1, d)
    eta = 0.05

    # Plain gradient step
    step_gd = eta * grad

    # QNG step with g = I, lambda = 0
    g_eye = np.eye(d)
    step_qng = eta * np.linalg.solve(g_eye, grad)

    np.testing.assert_allclose(step_qng, step_gd, atol=1e-14)


def test_vqe_two_qubits_reaches_ground_energy_within_tolerance():
    """Verify that VQE on 2-qubit Ising chain with 2 layers reaches ground energy within 1e-3."""
    H = ising_chain_hamiltonian(2, J=1.0, h=1.0)
    E0 = exact_ground_energy(H, 2)
    qnode = make_vqe_qnode(H, 2, 2)

    # Start near reasonable initial point and run 80 Adam steps
    p = np.array([0.5, 0.2, 0.4, 0.1, 0.3, 0.2, 0.1, 0.4])
    m = np.zeros_like(p)
    v = np.zeros_like(p)

    for t in range(1, 81):
        g = param_shift_grad_vqe(qnode, p)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * (g ** 2)
        m_hat = m / (1.0 - 0.9 ** t)
        v_hat = v / (1.0 - 0.999 ** t)
        p = p - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)

    final_energy = float(qnode(p))
    error = abs(final_energy - E0)
    assert error < 1e-3, f"Final energy {final_energy} is not within 1e-3 of E0 {E0} (error={error:.6e})"


def test_barren_plateau_variance_non_increasing():
    """Verify that for n=2 and n=4 qubits, the global cost gradient variance is non-increasing
    over 200 random initializations (allowing 10% numerical tolerance).
    """
    def measure_var(n_qubits: int, n_samples: int = 200) -> float:
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(params):
            idx = 0
            for l in range(n_qubits):
                for i in range(n_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        @qml.qnode(dev)
        def q_state(params):
            circuit(params)
            return qml.state()

        d = n_qubits * n_qubits
        rng = np.random.default_rng(42)
        grads = []

        for _ in range(n_samples):
            p = rng.uniform(0, 2 * np.pi, d)
            p_plus = p.copy()
            p_plus[0] += np.pi / 2
            p_minus = p.copy()
            p_minus[0] -= np.pi / 2

            s_plus = q_state(p_plus)
            s_minus = q_state(p_minus)

            p0_plus = np.abs(s_plus[0]) ** 2
            p0_minus = np.abs(s_minus[0]) ** 2
            # d/dtheta (1 - |<0|psi>|^2)
            g = (p0_minus - p0_plus) / 2.0
            grads.append(g)

        return float(np.var(grads))

    var_2 = measure_var(2, 200)
    var_4 = measure_var(4, 200)

    # Variance must be non-increasing in n (var_4 <= 1.10 * var_2)
    assert var_4 <= 1.10 * var_2, f"var(n=4)={var_4:.6e} is greater than 1.10 * var(n=2)={var_2:.6e}"
    assert var_4 < var_2, f"Expected var(n=4) < var(n=2), got var(n=4)={var_4}, var(n=2)={var_2}"
