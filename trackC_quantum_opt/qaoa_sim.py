"""trackC_quantum_opt/qaoa_sim.py
---------------------------------
Independent pure-NumPy statevector QAOA simulator for Max-Cut.

Bit-Order & State Convention:
-----------------------------
Computational basis states |b_0 b_1 ... b_{n-1}> are indexed by integer:
    idx = sum_{k=0}^{n-1} b_k * 2**(n - 1 - k)
where qubit / node 0 corresponds to the most significant bit (MSB, bit n-1),
and node n-1 corresponds to the least significant bit (LSB, bit 0).
The assignment vector x in {0, 1}^n has x[k] = (idx >> (n - 1 - k)) & 1.
An edge (u, v) with weight w is cut when x[u] != x[v].
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np


def parse_params(params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]], p: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Parse parameters into (gammas, betas) 1D numpy arrays."""
    if isinstance(params, (tuple, list)) and len(params) == 2 and isinstance(params[0], (list, tuple, np.ndarray)) and isinstance(params[1], (list, tuple, np.ndarray)):
        gammas = np.asarray(params[0], dtype=float)
        betas = np.asarray(params[1], dtype=float)
    else:
        arr = np.asarray(params, dtype=float)
        if arr.ndim != 1 or len(arr) % 2 != 0:
            raise ValueError(f"params must be a 1D array of even length (2*p), got shape {arr.shape}.")
        p_in = len(arr) // 2
        gammas = arr[:p_in]
        betas = arr[p_in:]

    if p is not None and len(gammas) != p:
        raise ValueError(f"Expected {p} layers, got {len(gammas)}.")
    if len(gammas) != len(betas):
        raise ValueError(f"Mismatched gammas length ({len(gammas)}) and betas length ({len(betas)}).")
    return gammas, betas


def maxcut_diagonal(
    n: int,
    edges: Sequence[Union[Tuple[int, int], Tuple[int, int, float], List[Any]]],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute the vector of cut values for all 2^n computational basis bitstrings.

    Parameters:
        n: Number of vertices (qubits).
        edges: List of edges. Each edge can be (u, v) or (u, v, weight).
        weights: Optional list of edge weights. If provided, overrides weights in edges.

    Returns:
        diag: 1D float64 array of shape (2^n,) with diag[idx] = Cut(x).
    """
    if n <= 0 or n > 24:
        raise ValueError(f"Number of qubits n must be in [1, 24], got {n}.")

    dim = 1 << n
    indices = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=np.float64)

    for i, e in enumerate(edges):
        u = int(e[0])
        v = int(e[1])
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Edge nodes ({u}, {v}) out of range [0, {n-1}].")
        if u == v:
            continue

        if weights is not None:
            w = float(weights[i])
        elif len(e) >= 3:
            w = float(e[2])
        else:
            w = 1.0

        # Big-endian: node k is bit (n - 1 - k)
        bu = (indices >> (n - 1 - u)) & 1
        bv = (indices >> (n - 1 - v)) & 1
        cut_mask = (bu != bv)
        diag += w * cut_mask

    return diag


def qaoa_state(
    gammas: Sequence[float],
    betas: Sequence[float],
    diag: np.ndarray,
    n: int,
) -> np.ndarray:
    """Simulate the statevector of a p-layer QAOA circuit on n qubits.

    Circuit structure:
        1. Uniform superposition |+>^{tensor n}
        2. For l = 0 ... p-1:
           a. Phase layer: exp(-i * gammas[l] * diag)
           b. Mixer layer: exp(-i * betas[l] * X_j) on each wire j = 0 ... n-1

    Returns:
        psi: Complex128 statevector of shape (2^n,).
    """
    g_arr = np.asarray(gammas, dtype=float)
    b_arr = np.asarray(betas, dtype=float)
    if len(g_arr) != len(b_arr):
        raise ValueError(f"Length of gammas ({len(g_arr)}) must equal betas ({len(b_arr)}).")
    p = len(g_arr)

    dim = 1 << n
    if diag.shape != (dim,):
        raise ValueError(f"diag shape must be ({dim},), got {diag.shape}.")

    # Initial uniform superposition |+>^{\otimes n}
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)

    for l in range(p):
        g = g_arr[l]
        b = b_arr[l]

        # 1. Phase layer: exp(-i * gamma * cut(x))
        if g != 0.0:
            psi = psi * np.exp(-1j * g * diag)

        # 2. Mixer layer: exp(-i * beta * X_j) on every qubit
        if b != 0.0:
            c = np.cos(b)
            s = -1j * np.sin(b)
            for j in range(n):
                sh = (1 << j, 2, 1 << (n - 1 - j))
                p_sh = psi.reshape(sh)
                p0 = p_sh[:, 0, :]
                p1 = p_sh[:, 1, :]
                psi = np.stack([c * p0 + s * p1, s * p0 + c * p1], axis=1).reshape(-1)

    return psi


def probabilities(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
) -> np.ndarray:
    """Compute the probability distribution over all 2^n computational basis states."""
    gammas, betas = parse_params(params)
    psi = qaoa_state(gammas, betas, diag, n)
    probs = np.abs(psi) ** 2
    # Ensure numerical normalization
    s = np.sum(probs)
    if s > 0:
        probs /= s
    return probs


def expected_cut(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
) -> float:
    """Calculate the expected cut value <C> = sum_x Cut(x) * |psi(x)|^2."""
    probs = probabilities(params, diag, n)
    return float(np.dot(probs, diag))


def sample(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
    shots: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample 'shots' measurement outcomes from the final state probability distribution.

    Returns:
        cut_samples: 1D array of length 'shots' containing the cut values of sampled bitstrings.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}.")
    if rng is None:
        rng = np.random.default_rng()

    probs = probabilities(params, diag, n)
    dim = len(probs)
    sampled_indices = rng.choice(dim, size=shots, p=probs)
    return diag[sampled_indices]


def approximation_ratio(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
    opt_cut: float,
) -> float:
    """Compute the approximation ratio <C> / C_opt."""
    if opt_cut <= 0:
        raise ValueError(f"opt_cut must be positive, got {opt_cut}.")
    return expected_cut(params, diag, n) / opt_cut


def probability_of_optimal(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
    opt_cut: Optional[float] = None,
) -> float:
    """Compute the probability of measuring an optimal cut configuration."""
    if opt_cut is None:
        opt_cut = float(np.max(diag))
    probs = probabilities(params, diag, n)
    opt_mask = np.isclose(diag, opt_cut, atol=1e-8)
    return float(np.sum(probs[opt_mask]))


def _parse_edges(
    edges: Sequence[Union[Tuple[int, int], Tuple[int, int, float], List[Any]]],
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[int, int, float]]:
    """Normalize edges into list of (u, v, weight)."""
    norm = []
    for i, e in enumerate(edges):
        u, v = int(e[0]), int(e[1])
        if weights is not None:
            w = float(weights[i])
        elif len(e) >= 3:
            w = float(e[2])
        else:
            w = 1.0
        norm.append((u, v, w))
    return norm


def param_shift_grad(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
    edges: Sequence[Union[Tuple[int, int], Tuple[int, int, float], List[Any]]],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute the exact analytical gradient of <C> using the per-gate parameter-shift rule.

    For each layer l:
        - For gamma_l: For each edge e=(u, v, w), shift the edge angle theta_e = gamma_l * w
          by +pi/2 and -pi/2, evaluate (f_+ - f_-) / 2, multiply by w, and sum over all edges.
        - For beta_l: For each qubit j, shift rotation angle phi_j = 2 * beta_l by +pi/2 and -pi/2
          (i.e. shift beta_l on qubit j by +pi/4 and -pi/4), evaluate f_+ - f_-, and sum over qubits.

    Returns:
        grad: 1D array of shape (2*p,) with [d<C>/dgamma_0 ... d<C>/dgamma_{p-1}, d<C>/dbeta_0 ... d<C>/dbeta_{p-1}].
    """
    gammas, betas = parse_params(params)
    p = len(gammas)
    edge_list = _parse_edges(edges, weights)

    dim = 1 << n
    indices = np.arange(dim, dtype=np.int64)

    # Precompute edge cut masks (eigenvalue 1 when cut, 0 otherwise)
    edge_masks = []
    for u, v, w in edge_list:
        bu = (indices >> (n - 1 - u)) & 1
        bv = (indices >> (n - 1 - v)) & 1
        edge_masks.append((bu != bv).astype(np.float64))

    # Precompute states before each layer to avoid re-evaluating preceding layers
    prefix_states = [np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)]
    for l in range(p - 1):
        psi_next = prefix_states[-1].copy()
        g = gammas[l]
        b = betas[l]
        if g != 0.0:
            psi_next *= np.exp(-1j * g * diag)
        if b != 0.0:
            c = np.cos(b)
            s = -1j * np.sin(b)
            for j in range(n):
                sh = (1 << j, 2, 1 << (n - 1 - j))
                p_sh = psi_next.reshape(sh)
                psi_next = np.stack([c * p_sh[:, 0, :] + s * p_sh[:, 1, :], s * p_sh[:, 0, :] + c * p_sh[:, 1, :]], axis=1).reshape(-1)
        prefix_states.append(psi_next)

    def evolve_from_layer(psi_start: np.ndarray, start_layer: int, layer_modified_phase: Optional[np.ndarray], modified_qubit: Optional[Tuple[int, float]]) -> float:
        psi = psi_start.copy()
        for l in range(start_layer, p):
            if l == start_layer and layer_modified_phase is not None:
                psi *= np.exp(-1j * layer_modified_phase)
            else:
                g = gammas[l]
                if g != 0.0:
                    psi *= np.exp(-1j * g * diag)

            b = betas[l]
            for j in range(n):
                bj = b
                if l == start_layer and modified_qubit is not None and modified_qubit[0] == j:
                    bj += modified_qubit[1]
                if bj != 0.0:
                    c = np.cos(bj)
                    s = -1j * np.sin(bj)
                    sh = (1 << j, 2, 1 << (n - 1 - j))
                    p_sh = psi.reshape(sh)
                    psi = np.stack([c * p_sh[:, 0, :] + s * p_sh[:, 1, :], s * p_sh[:, 0, :] + c * p_sh[:, 1, :]], axis=1).reshape(-1)

        probs = np.abs(psi) ** 2
        return float(np.dot(probs, diag))

    grad_gamma = np.zeros(p, dtype=np.float64)
    grad_beta = np.zeros(p, dtype=np.float64)

    for l in range(p):
        psi_pre = prefix_states[l]
        base_phase = gammas[l] * diag

        # 1. Gradient with respect to gamma_l
        for e_idx, (u, v, w) in enumerate(edge_list):
            mask = edge_masks[e_idx]
            # Shift edge angle by +pi/2 and -pi/2
            f_plus = evolve_from_layer(psi_pre, l, base_phase + (np.pi / 2.0) * mask, None)
            f_minus = evolve_from_layer(psi_pre, l, base_phase - (np.pi / 2.0) * mask, None)
            df_dtheta = (f_plus - f_minus) / 2.0
            grad_gamma[l] += w * df_dtheta

        # 2. Gradient with respect to beta_l
        for j in range(n):
            # Shift rotation angle on wire j by +pi/2 (i.e. beta_l shifted by +pi/4)
            f_plus = evolve_from_layer(psi_pre, l, base_phase, (j, np.pi / 4.0))
            f_minus = evolve_from_layer(psi_pre, l, base_phase, (j, -np.pi / 4.0))
            grad_beta[l] += (f_plus - f_minus)

    return np.concatenate([grad_gamma, grad_beta])


def finite_diff_grad(
    params: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    diag: np.ndarray,
    n: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the gradient of <C> using central finite differences."""
    gammas, betas = parse_params(params)
    theta = np.concatenate([gammas, betas])
    d = len(theta)
    grad = np.zeros(d, dtype=np.float64)

    for i in range(d):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        f_plus = expected_cut(theta_plus, diag, n)
        f_minus = expected_cut(theta_minus, diag, n)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)

    return grad
