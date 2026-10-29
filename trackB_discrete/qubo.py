"""trackB_discrete/qubo.py
-------------------------
Quadratic Unconstrained Binary Optimization (QUBO) and Ising model utilities.

Mathematical Conventions:
- QUBO matrix Q is upper-triangular (Q_{ij} = 0 for i > j).
  Energy: E_QUBO(x) = x^T Q x + offset = sum_i Q_{ii} x_i + sum_{i < j} Q_{ij} x_i x_j + offset.
- Ising spins: s_i in {-1, +1}, related to binary x_i in {0, 1} via s_i = 2 x_i - 1.
  Energy: E_Ising(s) = sum_{i < j} J_{ij} s_i s_j + sum_i h_i s_i + offset.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def evaluate_qubo(Q: np.ndarray, x: np.ndarray, offset: float = 0.0) -> float:
    """Evaluate E(x) = x^T Q x + offset for a binary vector x in {0, 1}^n."""
    x_vec = np.asarray(x, dtype=float)
    Q_mat = np.asarray(Q, dtype=float)
    return float(x_vec.T @ Q_mat @ x_vec + offset)


def maxcut_to_qubo(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert a Max-Cut graph problem to an upper-triangular QUBO minimization problem.

    For an undirected graph with symmetric edge weights w_{ij}:
        Cut(x) = sum_{i < j} w_{ij} (x_i + x_j - 2 x_i x_j)

    To maximize Cut(x), we minimize -Cut(x):
        -Cut(x) = - sum_{i=1}^n (sum_{j != i} w_{ij}) x_i + sum_{i < j} (2 w_{ij}) x_i x_j

    Therefore, upper-triangular Q has:
        Q_{ii} = - sum_{j != i} w_{ij}
        Q_{ij} = 2 w_{ij}  (for i < j)
        Q_{ij} = 0         (for i > j)
        offset = 0.0

    Relationship:
        Cut(x) = - (x^T Q x) - offset = - evaluate_qubo(Q, x, offset)
    """
    W = np.asarray(weights, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights must be a 2D square matrix.")

    n = W.shape[0]
    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        # Diagonal: negative weighted degree
        deg_i = float(np.sum(W[i, :]) - W[i, i])
        Q[i, i] = -deg_i
        for j in range(i + 1, n):
            # Upper-triangular off-diagonal: 2 * w_{ij}
            w_ij = float(W[i, j])
            Q[i, j] = 2.0 * w_ij

    offset = 0.0
    return Q, offset


def qubo_to_ising(
    Q: np.ndarray,
    qubo_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert an upper-triangular QUBO matrix Q to Ising parameters (J, h, offset).

    Using the bijection x_i = (s_i + 1) / 2 where s_i in {-1, +1}:
        x_i x_j = (s_i s_j + s_i + s_j + 1) / 4
        x_i = (s_i + 1) / 2

    Ising Energy:
        E_Ising(s) = sum_{i < j} J_{ij} s_i s_j + sum_i h_i s_i + offset

    Returns:
        J: Upper-triangular matrix of pairwise couplings (J_{ij} = 0 for i >= j).
        h: 1D array of on-site external fields.
        offset: Scalar energy constant such that E_Ising(s) == E_QUBO(x).
    """
    Q_mat = np.asarray(Q, dtype=float)
    if Q_mat.ndim != 2 or Q_mat.shape[0] != Q_mat.shape[1]:
        raise ValueError("Q must be a square matrix.")

    n = Q_mat.shape[0]
    J = np.zeros((n, n), dtype=float)
    h = np.zeros(n, dtype=float)
    offset = float(qubo_offset)

    for i in range(n):
        q_ii = Q_mat[i, i]
        h[i] += 0.5 * q_ii
        offset += 0.5 * q_ii

        for j in range(i + 1, n):
            q_ij = Q_mat[i, j]
            J[i, j] = 0.25 * q_ij
            h[i] += 0.25 * q_ij
            h[j] += 0.25 * q_ij
            offset += 0.25 * q_ij

    return J, h, offset


def ising_energy(
    s: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    offset: float = 0.0,
) -> float:
    """Compute the Ising energy E(s) = sum_{i < j} J_{ij} s_i s_j + sum_i h_i s_i + offset."""
    s_vec = np.asarray(s, dtype=float)
    J_mat = np.asarray(J, dtype=float)
    h_vec = np.asarray(h, dtype=float)

    n = len(s_vec)
    # Extract upper-triangular portion of J
    J_upper = np.triu(J_mat, k=1)
    coupling_energy = float(s_vec.T @ J_upper @ s_vec)
    field_energy = float(np.dot(h_vec, s_vec))
    return coupling_energy + field_energy + offset


def cardinality_qubo(
    values: Sequence[float],
    k: int,
    penalty: float,
) -> Tuple[np.ndarray, float]:
    """Formulate cardinality-constrained selection as a QUBO minimization problem.

    Problem:
        Maximize sum_{i=1}^n v_i x_i  subject to  sum_{i=1}^n x_i = k

    QUBO penalty objective to MINIMIZE:
        E(x) = - sum_{i=1}^n v_i x_i + P (sum_{i=1}^n x_i - k)^2

    Expansion:
        (sum x_i - k)^2 = (1 - 2k) sum_i x_i + 2 sum_{i < j} x_i x_j + k^2

    QUBO terms:
        Q_{ii} = - v_i + P (1 - 2k)
        Q_{ij} = 2 P  (for i < j)
        offset = P * k^2
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if k < 0 or k > n:
        raise ValueError(f"k must be in range [0, {n}], got {k}.")
    if penalty < 0:
        raise ValueError(f"penalty P must be non-negative, got {penalty}.")

    Q = np.zeros((n, n), dtype=float)
    P = float(penalty)

    for i in range(n):
        Q[i, i] = -v[i] + P * (1.0 - 2.0 * k)
        for j in range(i + 1, n):
            Q[i, j] = 2.0 * P

    offset = P * (k ** 2)
    return Q, offset


def brute_force_qubo(
    Q: np.ndarray,
    offset: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Find the exact global minimum of a QUBO by enumerating all 2^n configurations (n <= 20)."""
    Q_mat = np.asarray(Q, dtype=float)
    n = Q_mat.shape[0]
    if n > 20:
        raise ValueError(f"brute_force_qubo is limited to n <= 20, got n={n}.")

    best_e = float("inf")
    best_x = np.zeros(n, dtype=int)

    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        e = float(x.T @ Q_mat @ x + offset)
        if e < best_e:
            best_e = e
            best_x = x.astype(int)

    return best_e, best_x


def simulated_annealing_qubo(
    Q: np.ndarray,
    offset: float = 0.0,
    seed: int = 0,
    steps: int = 2000,
    t_init: float = 10.0,
    t_final: float = 0.01,
    restarts: int = 1,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Simulated annealing solver for QUBO minimization with single-bit flips and geometric cooling.

    Args:
        Q: Upper-triangular QUBO matrix.
        offset: Scalar energy offset.
        seed: Random seed for reproducibility.
        steps: Annealing steps per restart.
        t_init: Starting temperature.
        t_final: Ending temperature.
        restarts: Number of independent random restarts.

    Returns:
        (best_energy, best_x, info_dict)
    """
    t0 = time.perf_counter()
    Q_mat = np.asarray(Q, dtype=float)
    n = Q_mat.shape[0]

    rng = np.random.default_rng(seed)

    global_best_e = float("inf")
    global_best_x = np.zeros(n, dtype=int)

    # Pre-symmetrize for fast O(n) bit-flip delta computation:
    # Q_sym = Q + Q.T with diagonal divided by 2
    # E(x) = x^T Q_sym x + offset
    Q_sym = np.triu(Q_mat, k=1) + np.triu(Q_mat, k=1).T + np.diag(np.diag(Q_mat))

    for _ in range(restarts):
        # Random initial configuration
        x = rng.integers(0, 2, size=n).astype(float)
        current_e = float(x.T @ Q_mat @ x + offset)

        if current_e < global_best_e:
            global_best_e = current_e
            global_best_x = x.astype(int).copy()

        for step in range(steps):
            # Geometric temperature schedule
            T = t_init * ((t_final / t_init) ** (step / max(1, steps - 1)))

            # Pick a random bit to flip
            k = rng.integers(0, n)
            delta_x = 1.0 - 2.0 * x[k]  # if 0 -> +1, if 1 -> -1

            # Delta E when flipping bit k:
            # E(x + delta_x e_k) - E(x) = delta_x * [ Q_sym[k, k] * (2*x[k] + delta_x) / 2 + sum_{j != k} Q_sym[k, j] * x_j ]
            # Since delta_x = 1 - 2*x[k], (2*x[k] + delta_x) = 1.
            delta_e = delta_x * (Q_mat[k, k] + np.sum(Q_sym[k, :] * x) - Q_sym[k, k] * x[k])

            # Metropolis acceptance criterion
            if delta_e < 0.0 or rng.random() < np.exp(-delta_e / max(1e-12, T)):
                x[k] += delta_x
                current_e += delta_e

                if current_e < global_best_e:
                    global_best_e = current_e
                    global_best_x = x.astype(int).copy()

    runtime = time.perf_counter() - t0
    return global_best_e, global_best_x, {"runtime": runtime}
