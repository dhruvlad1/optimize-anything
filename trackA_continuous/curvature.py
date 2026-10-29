"""trackA_continuous/curvature.py
--------------------------------
Second-order curvature tools, Hessian approximations, and backpropagation adapters.

Mathematical components:
1. flatten_params / unflatten_params: bijective mapping between dict and 1D vector.
2. hvp: Hessian-vector product via central finite differences of the gradient.
3. power_iteration_top_eig: spectral analysis using only hvp or matrix-vector products.
4. per_sample_grads: exact sample-wise gradients via batched backprop on single samples.
5. empirical_fisher, diag_fisher, low_rank_fisher: Fisher information approximations.
6. gauss_newton: Non-linear least-squares with optional Levenberg-Marquardt damping.
7. preconditioned_gradient_step: natural-gradient style curvature preconditioning.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from trackA_continuous.compare.backprop_antigravity import loss_and_grads

# Canonical order of parameter keys for consistent vectorization
PARAM_KEYS = ("W1", "b1", "W2", "b2")


# ---------------------------------------------------------------------------
# 1. Parameter flattening and unflattening
# ---------------------------------------------------------------------------

def flatten_params(
    params: Dict[str, np.ndarray],
    keys: Tuple[str, ...] = PARAM_KEYS,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, ...]]]:
    """Flatten a parameter dictionary into a single 1D numpy vector.

    Args:
        params: Dictionary containing array parameters.
        keys: Ordered sequence of keys to extract.

    Returns:
        (flat_theta, shapes) where flat_theta is a 1D float vector and
        shapes maps parameter name to original array shape.
    """
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}.")

    shapes = {}
    flat_pieces = []
    for k in keys:
        if k not in params:
            raise KeyError(f"Missing parameter key '{k}' in params dict.")
        arr = np.asarray(params[k], dtype=float)
        shapes[k] = arr.shape
        flat_pieces.append(arr.ravel())

    flat_theta = np.concatenate(flat_pieces, axis=0) if flat_pieces else np.empty(0, dtype=float)
    return flat_theta, shapes


def unflatten_params(
    flat_theta: np.ndarray,
    shapes: Dict[str, Tuple[int, ...]],
    keys: Tuple[str, ...] = PARAM_KEYS,
) -> Dict[str, np.ndarray]:
    """Reconstruct a parameter dictionary from a 1D vector and shape metadata.

    Args:
        flat_theta: 1D numpy array of concatenated parameters.
        shapes: Dict mapping parameter name to its original tensor shape.
        keys: Ordered sequence of keys to restore.

    Returns:
        Dictionary of reshaped parameter arrays.
    """
    flat = np.asarray(flat_theta, dtype=float)
    if flat.ndim != 1:
        raise ValueError(f"flat_theta must be 1D, got ndim={flat.ndim} with shape {flat.shape}.")

    expected_size = sum(int(np.prod(s)) for s in shapes.values())
    if flat.size != expected_size:
        raise ValueError(
            f"Size mismatch: flat_theta has size {flat.size}, expected {expected_size} from shapes."
        )

    params = {}
    offset = 0
    for k in keys:
        if k not in shapes:
            raise KeyError(f"Missing shape for key '{k}'.")
        shape = shapes[k]
        size = int(np.prod(shape))
        params[k] = flat[offset : offset + size].reshape(shape).copy()
        offset += size

    return params


# ---------------------------------------------------------------------------
# 2. Hessian-Vector Product (HVP)
# ---------------------------------------------------------------------------

def hvp(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute the Hessian-vector product H(theta) @ v via central finite differences.

    Mathematical formulation:
        H(theta) @ v = lim_{eps -> 0} [grad(theta + eps * v) - grad(theta - eps * v)] / (2 * eps)

    Args:
        grad_fn: Function mapping 1D parameter vector theta to 1D gradient vector g.
        theta: Parameter vector at which the Hessian is evaluated, shape (D,).
        v: Direction vector to multiply with the Hessian, shape (D,).
        eps: Small perturbation step size (default: 1e-5).

    Returns:
        1D numpy array of shape (D,) representing H(theta) @ v.
    """
    th = np.asarray(theta, dtype=float)
    vec = np.asarray(v, dtype=float)

    if th.ndim != 1:
        raise ValueError(f"theta must be a 1D array, got shape {th.shape}.")
    if vec.ndim != 1:
        raise ValueError(f"v must be a 1D array, got shape {vec.shape}.")
    if th.shape != vec.shape:
        raise ValueError(f"Shape mismatch: theta has shape {th.shape}, v has shape {vec.shape}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be strictly positive, got {eps}.")

    g_plus = np.asarray(grad_fn(th + eps * vec), dtype=float)
    g_minus = np.asarray(grad_fn(th - eps * vec), dtype=float)

    if g_plus.shape != th.shape or g_minus.shape != th.shape:
        raise ValueError("grad_fn returned gradient with shape not matching theta.")

    return (g_plus - g_minus) / (2.0 * eps)


# ---------------------------------------------------------------------------
# 3. Power Iteration for Top Eigenvalue
# ---------------------------------------------------------------------------

def power_iteration_top_eig(
    hvp_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    iters: int = 100,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Estimate the leading eigenvalue (largest magnitude) of a linear operator H.

    Algorithm:
        v_0 ~ Uniform unit sphere
        v_{k+1} = H v_k / ||H v_k||
        lambda = v_{iters}^T (H v_{iters})

    Args:
        hvp_fn: Callable mapping vector v -> H @ v.
        dim: Dimension D of the parameter space.
        iters: Number of power iterations.
        seed: Random seed for initial random vector.

    Returns:
        (top_eigenvalue, eigenvector) where eigenvector is unit norm.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}.")

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    norm_v = float(np.linalg.norm(v))
    if norm_v == 0.0:
        v[0] = 1.0
        norm_v = 1.0
    v = v / norm_v

    for _ in range(iters):
        w = np.asarray(hvp_fn(v), dtype=float)
        norm_w = float(np.linalg.norm(w))
        if norm_w < 1e-15:
            # Operator vanishes on current subspace
            return 0.0, v
        v = w / norm_w

    # Final Rayleigh quotient estimate: v^T (H v)
    w_final = np.asarray(hvp_fn(v), dtype=float)
    top_eig = float(np.dot(v, w_final))
    return top_eig, v


# ---------------------------------------------------------------------------
# 4. Per-Sample Gradients and Empirical Fisher Matrix
# ---------------------------------------------------------------------------

def per_sample_grads(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
) -> np.ndarray:
    """Compute per-sample gradient vectors by evaluating backprop on single samples.

    Args:
        params: Network parameter dict.
        X: Input data matrix of shape (n, d).
        y: Target label vector of shape (n,).
        l2: Regularization strength (typically 0.0 for pure empirical Fisher).

    Returns:
        G: 2D array of shape (n, D), where row i is g_i = grad_theta loss(x_i, y_i).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n = len(X_arr)
    if n == 0:
        raise ValueError("X must contain at least one sample.")
    if len(y_arr) != n:
        raise ValueError(f"Length mismatch: X has {n} samples, y has {len(y_arr)}.")

    sample_grads_list = []
    for i in range(n):
        Xi = X_arr[i : i + 1]
        yi = y_arr[i : i + 1]
        _, g_dict = loss_and_grads(params, Xi, yi, l2=l2)
        g_flat, _ = flatten_params(g_dict)
        sample_grads_list.append(g_flat)

    return np.vstack(sample_grads_list)


def empirical_fisher(
    params: Optional[Dict[str, np.ndarray]] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    l2: float = 0.0,
    per_sample_g: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the Empirical Fisher Information Matrix:

        F = (1 / n) sum_{i=1}^n g_i g_i^T = (1 / n) G^T G

    where G is the (n, D) matrix of per-sample gradient vectors.

    Args:
        params, X, y, l2: Passed to per_sample_grads if per_sample_g is None.
        per_sample_g: Precomputed (n, D) matrix of per-sample gradients.

    Returns:
        F: Symmetric positive semi-definite matrix of shape (D, D).
    """
    if per_sample_g is None:
        if params is None or X is None or y is None:
            raise ValueError("Must provide either per_sample_g or (params, X, y).")
        G = per_sample_grads(params, X, y, l2=l2)
    else:
        G = np.asarray(per_sample_g, dtype=float)

    if G.ndim != 2:
        raise ValueError(f"Per-sample gradient matrix must be 2D, got shape {G.shape}.")

    n, d = G.shape
    if n == 0:
        raise ValueError("Per-sample gradient matrix must have n > 0 rows.")

    # (1 / n) G^T G
    F = (G.T @ G) / float(n)
    # Enforce exact floating-point numerical symmetry
    F = 0.5 * (F + F.T)
    return F


def diag_fisher(
    params: Optional[Dict[str, np.ndarray]] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    l2: float = 0.0,
    per_sample_g: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the diagonal of the empirical Fisher:

        diag(F)_j = (1 / n) sum_{i=1}^n (g_{i, j})^2

    Args:
        params, X, y, l2: Passed to compute gradients if per_sample_g and F are None.
        per_sample_g: Optional precomputed (n, D) matrix of per-sample gradients.
        F: Optional precomputed full (D, D) Fisher matrix.

    Returns:
        1D array of shape (D,) representing the diagonal entries of F.
    """
    if F is not None:
        F_mat = np.asarray(F, dtype=float)
        if F_mat.ndim != 2 or F_mat.shape[0] != F_mat.shape[1]:
            raise ValueError("F must be a 2D square matrix.")
        return np.diag(F_mat).copy()

    if per_sample_g is None:
        if params is None or X is None or y is None:
            raise ValueError("Must provide F, per_sample_g, or (params, X, y).")
        G = per_sample_grads(params, X, y, l2=l2)
    else:
        G = np.asarray(per_sample_g, dtype=float)

    if G.ndim != 2:
        raise ValueError(f"G must be 2D, got shape {G.shape}.")

    return np.mean(G ** 2, axis=0)


def low_rank_fisher(F: np.ndarray, r: int) -> np.ndarray:
    """Compute the optimal rank-r approximation of the Fisher matrix F via eigendecomposition.

    By the Eckart-Young-Mirsky theorem, the closest rank-r matrix in Frobenius norm is:
        F_r = sum_{k=1}^r lambda_k u_k u_k^T
    where lambda_1 >= lambda_2 >= ... are the top eigenvalues of the symmetric PSD matrix F.

    Args:
        F: Symmetric (D, D) Fisher matrix.
        r: Target approximation rank, 1 <= r <= D.

    Returns:
        F_r: Rank-r PSD matrix of shape (D, D).
    """
    F_mat = np.asarray(F, dtype=float)
    if F_mat.ndim != 2 or F_mat.shape[0] != F_mat.shape[1]:
        raise ValueError(f"F must be a 2D square matrix, got shape {F_mat.shape}.")

    D = F_mat.shape[0]
    if r < 1 or r > D:
        raise ValueError(f"Rank r must be between 1 and D={D}, got r={r}.")

    # np.linalg.eigh returns eigenvalues in ascending order
    eigvals, eigvecs = np.linalg.eigh(F_mat)

    # Pick the top r eigenvalues & eigenvectors (descending)
    top_indices = np.argsort(eigvals)[::-1][:r]
    top_eigvals = np.maximum(eigvals[top_indices], 0.0)  # clamp numerical negatives to 0
    top_eigvecs = eigvecs[:, top_indices]

    # Reconstruct rank-r matrix: V_r @ diag(w_r) @ V_r^T
    F_r = top_eigvecs @ np.diag(top_eigvals) @ top_eigvecs.T
    return 0.5 * (F_r + F_r.T)


# ---------------------------------------------------------------------------
# 5. Preconditioned Gradient Step
# ---------------------------------------------------------------------------

def preconditioned_gradient_step(
    theta: np.ndarray,
    g: np.ndarray,
    F_approx: Union[np.ndarray, float],
    lr: float,
    damping: float = 1e-4,
) -> np.ndarray:
    """Take a preconditioned gradient descent step:

        theta_{t+1} = theta_t - lr * (F_approx + damping * I)^{-1} g

    Supports:
        - Diagonal preconditioner: F_approx is a 1D vector of shape (D,)
        - Full or low-rank preconditioner: F_approx is a 2D matrix of shape (D, D)

    Args:
        theta: Current 1D parameter vector of shape (D,).
        g: Current 1D gradient vector of shape (D,).
        F_approx: 1D diagonal vector or 2D matrix curvature approximation.
        lr: Positive scalar learning rate.
        damping: Damping scalar (Tikhonov / Levenberg parameter) >= 0.

    Returns:
        theta_next: Updated parameter vector of shape (D,).
    """
    th = np.asarray(theta, dtype=float)
    grad = np.asarray(g, dtype=float)

    if th.ndim != 1 or grad.ndim != 1:
        raise ValueError("theta and g must be 1D arrays.")
    if th.shape != grad.shape:
        raise ValueError(f"Shape mismatch: theta {th.shape} vs g {grad.shape}.")
    if lr < 0.0:
        raise ValueError(f"lr must be non-negative, got {lr}.")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}.")

    F_arr = np.asarray(F_approx, dtype=float)

    if F_arr.ndim == 1:
        # Diagonal preconditioner: elementwise division
        if F_arr.shape != th.shape:
            raise ValueError(f"Diagonal F_approx shape {F_arr.shape} != theta shape {th.shape}.")
        denom = F_arr + damping
        if np.any(denom <= 0.0):
            # Guard against non-positive denominator
            denom = np.maximum(denom, 1e-12)
        step = grad / denom
    elif F_arr.ndim == 2:
        # Matrix preconditioner: solve linear system (F + damping * I) step = g
        D = th.shape[0]
        if F_arr.shape != (D, D):
            raise ValueError(f"F_approx matrix shape {F_arr.shape} != ({D}, {D}).")
        A = F_arr + damping * np.eye(D)
        # Solve A @ step = grad using standard symmetric / robust solver
        try:
            step = np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            step, _, _, _ = np.linalg.lstsq(A, grad, rcond=1e-12)
    else:
        raise ValueError(f"F_approx must be 1D or 2D, got ndim={F_arr.ndim}.")

    return th - lr * step


# ---------------------------------------------------------------------------
# 6. Gauss-Newton and Levenberg-Marquardt
# ---------------------------------------------------------------------------

def gauss_newton(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jac_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    lam: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
    lm: bool = False,
) -> Dict[str, Any]:
    """Solve non-linear least squares min_theta 0.5 * ||r(theta)||^2 via Gauss-Newton or LM.

    Objective:
        f(theta) = 0.5 * ||r(theta)||_2^2
        grad(theta) = J(theta)^T r(theta)
        H_GN(theta) = J(theta)^T J(theta)

    Update equation:
        (J^T J + lambda * I) delta_theta = - J^T r

    Args:
        residual_fn: Callable theta -> r(theta) of shape (m,).
        jac_fn: Callable theta -> J(theta) = dr/dtheta of shape (m, D).
        theta0: Initial parameter vector of shape (D,).
        lam: Damping parameter lambda >= 0. If lm=True, acts as initial lambda.
        max_iter: Maximum number of iterations.
        tol: Stopping threshold on the Euclidean norm of the gradient ||J^T r||.
        lm: If True, uses adaptive Levenberg-Marquardt trust-region / damping adjustments.

    Returns:
        Dict with keys:
            'theta': final parameter estimate
            'cost': final 0.5 * ||r||^2
            'grad_norm': final ||J^T r||
            'iters': number of iterations performed
            'converged': bool indicating whether ||g|| < tol
            'history': list of cost values per iteration
    """
    th = np.asarray(theta0, dtype=float).copy()
    if th.ndim != 1:
        raise ValueError(f"theta0 must be a 1D vector, got shape {th.shape}.")
    D = th.shape[0]

    current_lam = float(lam) if (lam > 0.0 or not lm) else 1e-2
    history = []

    for it in range(max_iter):
        r = np.asarray(residual_fn(th), dtype=float)
        cost = 0.5 * float(np.sum(r ** 2))
        history.append(cost)

        J = np.asarray(jac_fn(th), dtype=float)
        if J.ndim != 2 or J.shape[1] != D:
            raise ValueError(f"Jacobian must have shape (m, {D}), got {J.shape}.")

        g = J.T @ r  # gradient: J^T r, shape (D,)
        g_norm = float(np.linalg.norm(g))

        if g_norm < tol or not np.isfinite(g_norm):
            return {
                "theta": th,
                "cost": cost,
                "grad_norm": g_norm,
                "iters": it,
                "converged": bool(g_norm < tol),
                "history": history,
            }

        # Form normal equations: (J^T J + lambda * I) delta = -g
        JTJ = J.T @ J
        A = JTJ + current_lam * np.eye(D)

        try:
            delta = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(A, -g, rcond=1e-12)

        if not lm:
            # Standard Gauss-Newton (fixed lambda)
            th = th + delta
        else:
            # Levenberg-Marquardt with adaptive trust damping
            th_candidate = th + delta
            r_candidate = np.asarray(residual_fn(th_candidate), dtype=float)
            cost_candidate = 0.5 * float(np.sum(r_candidate ** 2))

            if np.isfinite(cost_candidate) and cost_candidate < cost:
                # Step accepted: decrease damping toward Gauss-Newton
                th = th_candidate
                current_lam = max(current_lam / 3.0, 1e-10)
            else:
                # Step rejected: increase damping toward gradient descent
                current_lam = min(current_lam * 10.0, 1e10)

    # Final evaluation after loop
    r_final = np.asarray(residual_fn(th), dtype=float)
    final_cost = 0.5 * float(np.sum(r_final ** 2))
    history.append(final_cost)
    J_final = np.asarray(jac_fn(th), dtype=float)
    final_g_norm = float(np.linalg.norm(J_final.T @ r_final))

    return {
        "theta": th,
        "cost": final_cost,
        "grad_norm": final_g_norm,
        "iters": max_iter,
        "converged": bool(final_g_norm < tol),
        "history": history,
    }
