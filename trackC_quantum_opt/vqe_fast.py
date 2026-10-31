"""trackC_quantum_opt/vqe_fast.py
---------------------------------
Pure-numpy statevector backend for the hardware-efficient VQE ansatz.

Implements the same ansatz as vqe.py (RY, RZ layers + CNOT ladder) but
uses a single complex128 statevector (shape 2^n) with explicit gate
applications instead of PennyLane QNodes. This avoids Python/JIT overhead
per QNode call and achieves ~20-80x speedups over the PennyLane backend for
small n (n <= 6).

NOTE: All quantum computations are SIMULATED on classical hardware.
      No physical quantum processor is used.

API
---
build_dense_hamiltonian(H, n) -> np.ndarray       : 2^n x 2^n dense matrix
statevec(params, n, layers)   -> np.ndarray       : final statevector |psi>
energy_fast(params, dense_H, n, layers) -> float  : <psi|H|psi>
param_shift_grad_fast(params, dense_H, n, layers) -> np.ndarray  : gradient
metric_tensor_fast(params, n, layers)   -> np.ndarray  : Fubini-Study tensor
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Dense Hamiltonian builder
# ---------------------------------------------------------------------------

def build_dense_hamiltonian(H: qml.Hamiltonian, n: int) -> np.ndarray:
    """Build the 2^n x 2^n dense Hamiltonian matrix from a PennyLane Hamiltonian.

    Uses qml.matrix with the canonical wire order [0, 1, ..., n-1] so that
    qubit indexing is consistent with the statevector convention used below.
    """
    mat = qml.matrix(H, wire_order=list(range(n)))
    return np.asarray(mat, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Gate primitives  (operate on a 1-D complex128 array of length 2^n)
# ---------------------------------------------------------------------------

def _apply_single_qubit(psi: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply an arbitrary 2x2 gate matrix to `qubit` of an n-qubit statevector.

    Reshape trick:
        psi shape (2^n,)  ->  (2^qubit, 2, 2^(n-1-qubit))
    The qubit axis is axis 1. Apply the 2x2 gate along that axis and
    reshape back.
    """
    dim_left = 1 << qubit             # 2^qubit
    dim_right = 1 << (n - 1 - qubit) # 2^(n-1-qubit)
    ps = psi.reshape(dim_left, 2, dim_right)
    # out[..., 0, :] = gate[0,0]*ps[...,0,:] + gate[0,1]*ps[...,1,:]
    # out[..., 1, :] = gate[1,0]*ps[...,0,:] + gate[1,1]*ps[...,1,:]
    out = np.einsum("ab,xby->xay", gate, ps)
    return out.reshape(-1)


def apply_RY(psi: np.ndarray, theta: float, qubit: int, n: int) -> np.ndarray:
    """Apply RY(theta) = [[cos t/2, -sin t/2],[sin t/2, cos t/2]] to `qubit`."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    return _apply_single_qubit(psi, gate, qubit, n)


def apply_RZ(psi: np.ndarray, theta: float, qubit: int, n: int) -> np.ndarray:
    """Apply RZ(theta) = diag(exp(-i t/2), exp(+i t/2)) to `qubit`."""
    gate = np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )
    return _apply_single_qubit(psi, gate, qubit, n)


def apply_CNOT(psi: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    """Apply CNOT(control, target) to an n-qubit statevector.

    Works for any distinct control and target pair (not just adjacent).
    Algorithm: for each basis state with control=1, XOR the target bit.
    This is a O(2^n) index permutation.
    """
    dim = 1 << n
    indices = np.arange(dim, dtype=np.int64)
    # Bit positions in the integer index (MSB = qubit 0)
    ctrl_pos = n - 1 - control
    tgt_pos  = n - 1 - target
    ctrl_bit = (indices >> ctrl_pos) & 1
    # Flip target bit where control=1
    flipped = indices ^ (ctrl_bit * (1 << tgt_pos))
    result = psi[flipped]
    return result


# ---------------------------------------------------------------------------
# Ansatz: identical structure to hardware_efficient_ansatz in vqe.py
#   For each layer l in 0..layers-1:
#     For each qubit i in 0..n-1:  RY(p[idx]) ; RZ(p[idx+1])  (idx advances)
#     For each pair (i, i+1) in 0..n-2: CNOT(i, i+1)
# ---------------------------------------------------------------------------

def statevec(params: np.ndarray, n: int, layers: int) -> np.ndarray:
    """Evolve the computational zero state |00...0> through the hardware-efficient ansatz.

    Returns the final complex128 statevector of shape (2^n,).

    Parameters
    ----------
    params : 1-D real array of length 2 * n * layers.
    n      : Number of qubits.
    layers : Number of ansatz layers.
    """
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0  # |00...0>

    idx = 0
    for _l in range(layers):
        for i in range(n):
            psi = apply_RY(psi, float(params[idx]), i, n)
            idx += 1
            psi = apply_RZ(psi, float(params[idx]), i, n)
            idx += 1
        for i in range(n - 1):
            psi = apply_CNOT(psi, i, i + 1, n)

    return psi


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def energy_fast(
    params: np.ndarray,
    dense_H: np.ndarray,
    n: int,
    layers: int,
) -> float:
    """Compute <psi(params)| H |psi(params)> using the numpy statevector backend.

    Parameters
    ----------
    params  : 1-D real array, length 2*n*layers.
    dense_H : 2^n x 2^n complex Hamiltonian matrix (from build_dense_hamiltonian).
    n       : Number of qubits.
    layers  : Number of ansatz layers.
    """
    psi = statevec(params, n, layers)
    return float(np.real(np.vdot(psi, dense_H @ psi)))


# ---------------------------------------------------------------------------
# Parameter-shift gradient (exact for single-qubit Pauli rotations)
# ---------------------------------------------------------------------------

def param_shift_grad_fast(
    params: np.ndarray,
    dense_H: np.ndarray,
    n: int,
    layers: int,
) -> np.ndarray:
    """Analytical parameter-shift gradient of the energy w.r.t. each parameter.

    For RY and RZ gates (generators with eigenvalues ±1/2), the exact shift rule is:
        dE/dtheta_k = (E(theta + pi/2 e_k) - E(theta - pi/2 e_k)) / 2.

    Cost: 2 * d statevector evolutions (d = 2*n*layers).
    """
    p = np.asarray(params, dtype=float).copy()
    d = len(p)
    grad = np.zeros(d, dtype=np.float64)
    shift = np.pi / 2.0

    for k in range(d):
        p_plus = p.copy()
        p_plus[k] += shift
        p_minus = p.copy()
        p_minus[k] -= shift
        grad[k] = (energy_fast(p_plus, dense_H, n, layers) -
                   energy_fast(p_minus, dense_H, n, layers)) / 2.0

    return grad


# ---------------------------------------------------------------------------
# Full Fubini-Study metric tensor
# ---------------------------------------------------------------------------

def metric_tensor_fast(
    params: np.ndarray,
    n: int,
    layers: int,
) -> np.ndarray:
    """Compute the full Fubini-Study metric tensor g(theta) via pi-shifted state derivatives.

    The exact relationship for single-qubit Pauli rotations is:
        d|psi>/d(theta_k) = 0.5 * |psi(theta_k + pi)>

    The metric tensor element is:
        g_{ij} = Re(<d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi>)

    Cost: d + 1 statevector evolutions (1 for |psi>, d for the pi-shifted states).

    Parameters
    ----------
    params : 1-D real array, length 2*n*layers.
    n      : Number of qubits.
    layers : Number of ansatz layers.

    Returns
    -------
    g : (d, d) float64 array — the full Fubini-Study metric tensor.
    """
    p = np.asarray(params, dtype=float).copy()
    d = len(p)

    psi = statevec(p, n, layers)

    # Compute derivative vectors: d_k|psi> = 0.5 * |psi(theta_k + pi)>
    dpsi = []
    for k in range(d):
        p_shift = p.copy()
        p_shift[k] += np.pi
        dpsi.append(0.5 * statevec(p_shift, n, layers))

    g = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i, d):
            t1 = np.vdot(dpsi[i], dpsi[j])
            t2 = np.vdot(dpsi[i], psi) * np.vdot(psi, dpsi[j])
            val = float(np.real(t1 - t2))
            g[i, j] = val
            g[j, i] = val

    return g
