"""trackD_parallel/multitask.py
------------------------------
Multi-task learning with shared representations:
  - Equal weighting (vanilla gradient descent on sum of task losses)
  - MGDA  (Multiple Gradient Descent Algorithm, closed-form for 2 tasks)
  - GradNorm (task-weight adaptation to equalise gradient magnitudes)

Architecture: shared layer W (d_in=10 → hidden=8, tanh) + two linear heads.
All gradients are hand-written NumPy; no torch.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Parameter utilities
# ---------------------------------------------------------------------------

Params = Dict[str, np.ndarray]


def init_params(d_in: int = 10, hidden: int = 8, seed: int = 0) -> Params:
    """Initialise all network parameters with small random weights."""
    rng = np.random.default_rng(seed)
    scale = 0.1
    return {
        "W": rng.standard_normal((d_in, hidden)) * scale,   # shared layer weights
        "b": np.zeros(hidden),                                # shared layer bias
        "h1": rng.standard_normal(hidden) * scale,           # head 1 (task 1)
        "b1": np.zeros(1),
        "h2": rng.standard_normal(hidden) * scale,           # head 2 (task 2)
        "b2": np.zeros(1),
    }


def _copy_params(p: Params) -> Params:
    return {k: v.copy() for k, v in p.items()}


# ---------------------------------------------------------------------------
# 2. Data generation
# ---------------------------------------------------------------------------

def make_data(
    n: int = 500,
    d_in: int = 10,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data for two tasks with deliberately conflicting gradients.

    Task 1: smooth regression, scale ~1.
    Task 2: larger-scale regression, structure partially opposite to task 1.

    Returns X (n, d_in), y1 (n,), y2 (n,).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))

    w1 = rng.standard_normal(d_in)
    w1 /= np.linalg.norm(w1)
    y1 = np.sin(X @ w1) + rng.standard_normal(n) * 0.1

    # Task 2 uses a partially negated direction to create conflict
    w2 = -w1 + rng.standard_normal(d_in) * 0.3
    w2 /= np.linalg.norm(w2)
    y2 = (X @ w2) * 20.0 + rng.standard_normal(n) * 0.5

    return X.astype(np.float64), y1.astype(np.float64), y2.astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------

def forward(params: Params, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (h, y1_pred, y2_pred).  h = tanh(X @ W + b)."""
    h = np.tanh(X @ params["W"] + params["b"])          # (n, hidden)
    y1_pred = h @ params["h1"] + params["b1"]            # (n,)
    y2_pred = h @ params["h2"] + params["b2"]            # (n,)
    return h, y1_pred.ravel(), y2_pred.ravel()


def losses(
    params: Params, X: np.ndarray, y1: np.ndarray, y2: np.ndarray
) -> Tuple[float, float]:
    """Returns (L1, L2) MSE losses."""
    h, y1p, y2p = forward(params, X)
    L1 = float(np.mean((y1p - y1) ** 2))
    L2 = float(np.mean((y2p - y2) ** 2))
    return L1, L2


# ---------------------------------------------------------------------------
# 4. Backpropagation (hand-written)
# ---------------------------------------------------------------------------

def backprop(
    params: Params,
    X: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    task_weights: Tuple[float, float] = (1.0, 1.0),
) -> Params:
    """Hand-written gradients for the weighted sum loss w1*L1 + w2*L2.

    Returns a gradient dict with the same keys as params.
    """
    n = len(y1)
    h, y1p, y2p = forward(params, X)
    w1, w2 = task_weights

    # Head gradients (each task separately)
    # dL1/dy1p = 2*(y1p - y1)/n
    dy1p = 2.0 * (y1p - y1) / n       # (n,)
    dy2p = 2.0 * (y2p - y2) / n       # (n,)

    # Gradient w.r.t. head weights
    grad_h1 = w1 * (h.T @ dy1p)        # (hidden,)
    grad_b1 = w1 * np.array([dy1p.sum()])
    grad_h2 = w2 * (h.T @ dy2p)        # (hidden,)
    grad_b2 = w2 * np.array([dy2p.sum()])

    # Backprop through tanh shared layer
    # dh/d(pre) = 1 - h^2
    dh = (w1 * dy1p[:, None] * params["h1"][None, :] +
          w2 * dy2p[:, None] * params["h2"][None, :])   # (n, hidden)
    dpre = dh * (1.0 - h ** 2)                          # (n, hidden)

    grad_W = X.T @ dpre                                  # (d_in, hidden)
    grad_b = dpre.sum(axis=0)                            # (hidden,)

    return {"W": grad_W, "b": grad_b, "h1": grad_h1, "b1": grad_b1,
            "h2": grad_h2, "b2": grad_b2}


def _task_grads_shared(
    params: Params, X: np.ndarray, y1: np.ndarray, y2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-task gradient of the shared layer W (flattened) for MGDA."""
    n = len(y1)
    h, y1p, y2p = forward(params, X)

    dy1p = 2.0 * (y1p - y1) / n
    dh1 = dy1p[:, None] * params["h1"][None, :]
    dpre1 = dh1 * (1.0 - h ** 2)
    gW1 = (X.T @ dpre1).ravel()

    dy2p = 2.0 * (y2p - y2) / n
    dh2 = dy2p[:, None] * params["h2"][None, :]
    dpre2 = dh2 * (1.0 - h ** 2)
    gW2 = (X.T @ dpre2).ravel()

    return gW1, gW2


# ---------------------------------------------------------------------------
# 5. MGDA update step (closed-form for 2 tasks)
# ---------------------------------------------------------------------------

def mgda_alpha(g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute the MGDA min-norm point coefficient alpha*.

    alpha* = clip( ((g2 - g1) . g2) / ||g1 - g2||^2, 0, 1 )
    Direction: d = alpha* * g1 + (1 - alpha*) * g2
    """
    diff = g1 - g2
    denom = np.dot(diff, diff)
    if denom < 1e-30:
        return 0.5  # degenerate: g1 ≈ g2, any convex combo works
    alpha = np.dot(g2 - g1, g2) / denom
    return float(np.clip(alpha, 0.0, 1.0))


def mgda_step(
    params: Params,
    X: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    lr: float,
) -> Tuple[Params, float, float, float, float]:
    """One MGDA gradient step.

    Shared layer updated with the min-norm direction; heads updated with their
    own task gradients.

    Returns (new_params, alpha, g1_norm, g2_norm, cosine_sim).
    """
    # Task gradients on the shared layer (flattened)
    g1_flat, g2_flat = _task_grads_shared(params, X, y1, y2)

    alpha = mgda_alpha(g1_flat, g2_flat)
    d_shared = alpha * g1_flat + (1.0 - alpha) * g2_flat  # (d_in * hidden,)

    new_p = _copy_params(params)
    new_p["W"] -= lr * d_shared.reshape(params["W"].shape)

    # Shared bias: use same alpha combination
    n = len(y1)
    h, y1p, y2p = forward(params, X)

    dy1p = 2.0 * (y1p - y1) / n
    dy2p = 2.0 * (y2p - y2) / n
    dpre1 = (dy1p[:, None] * params["h1"][None, :]) * (1.0 - h ** 2)
    dpre2 = (dy2p[:, None] * params["h2"][None, :]) * (1.0 - h ** 2)
    gb1 = dpre1.sum(axis=0)
    gb2 = dpre2.sum(axis=0)
    new_p["b"] -= lr * (alpha * gb1 + (1.0 - alpha) * gb2)

    # Head updates (each with its own task gradient only)
    new_p["h1"] -= lr * (h.T @ dy1p)
    new_p["b1"] -= lr * np.array([dy1p.sum()])
    new_p["h2"] -= lr * (h.T @ dy2p)
    new_p["b2"] -= lr * np.array([dy2p.sum()])

    n1 = float(np.linalg.norm(g1_flat))
    n2 = float(np.linalg.norm(g2_flat))
    cos = float(np.dot(g1_flat, g2_flat) / (n1 * n2 + 1e-30))
    return new_p, alpha, n1, n2, cos


# ---------------------------------------------------------------------------
# 6. GradNorm update step
# ---------------------------------------------------------------------------

def gradnorm_step(
    params: Params,
    task_weights: np.ndarray,
    X: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    lr: float,
    lr_weights: float,
    L1_0: float,
    L2_0: float,
    alpha_gn: float = 1.5,
) -> Tuple[Params, np.ndarray, float, float]:
    """One GradNorm step.

    task_weights : (2,) current task weights (must sum to 2 after renorm)
    alpha_gn     : hyperparameter controlling target uniformity (1.5 is common)

    Returns (new_params, new_weights, gn_loss, mean_weight).
    """
    w1, w2 = task_weights
    L1, L2 = losses(params, X, y1, y2)

    # Gradient step on network parameters with current task weights
    grad = backprop(params, X, y1, y2, task_weights=(float(w1), float(w2)))
    new_p = {k: params[k] - lr * g for k, g in grad.items()}

    # --- GradNorm weight update (treat task_weights as a mini-param) ---
    # G_i = ||w_i * grad_W L_i||  (gradient norm of the weighted task loss w.r.t. shared W)
    g1_flat, g2_flat = _task_grads_shared(params, X, y1, y2)
    G1 = w1 * float(np.linalg.norm(g1_flat))
    G2 = w2 * float(np.linalg.norm(g2_flat))
    Gbar = 0.5 * (G1 + G2)

    # Relative inverse training rates (safe against zero init)
    r1 = (L1 / (L1_0 + 1e-30)) / (0.5 * (L1 / (L1_0 + 1e-30) + L2 / (L2_0 + 1e-30)) + 1e-30)
    r2 = (L2 / (L2_0 + 1e-30)) / (0.5 * (L1 / (L1_0 + 1e-30) + L2 / (L2_0 + 1e-30)) + 1e-30)

    target1 = Gbar * (r1 ** alpha_gn)
    target2 = Gbar * (r2 ** alpha_gn)

    # GradNorm loss = sum |G_i - target_i|  (targets treated as constants)
    gn_loss = abs(G1 - target1) + abs(G2 - target2)

    # Gradient of GradNorm loss w.r.t. w_i
    # d/dw_i |w_i*||g_i|| - target_i| = ||g_i|| * sign(w_i*||g_i|| - target_i)
    n1 = float(np.linalg.norm(g1_flat))
    n2 = float(np.linalg.norm(g2_flat))
    dw1 = n1 * np.sign(G1 - target1)
    dw2 = n2 * np.sign(G2 - target2)

    new_w1 = max(1e-6, w1 - lr_weights * dw1)
    new_w2 = max(1e-6, w2 - lr_weights * dw2)

    # Renormalise so sum = num_tasks = 2
    total = new_w1 + new_w2
    new_w1 = 2.0 * new_w1 / total
    new_w2 = 2.0 * new_w2 / total

    new_weights = np.array([new_w1, new_w2])
    return new_p, new_weights, gn_loss, 0.5 * (new_w1 + new_w2)


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------

def train(
    method: str,
    params: Params,
    X: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    steps: int = 500,
    lr: float = 0.01,
    lr_weights: float = 0.01,
    alpha_gn: float = 1.5,
) -> Dict:
    """Train with 'equal', 'mgda', or 'gradnorm' method.

    Returns history dict with per-step lists:
      L1, L2, total_loss, alpha (mgda), weights (gradnorm), cosine_sim (mgda)
    """
    assert method in ("equal", "mgda", "gradnorm"), f"Unknown method: {method}"
    p = _copy_params(params)

    L1_0, L2_0 = losses(p, X, y1, y2)
    task_weights = np.array([1.0, 1.0])

    history: Dict[str, List] = {
        "L1": [], "L2": [], "total_loss": [],
        "alpha": [], "w1": [], "w2": [], "cosine_sim": [],
    }

    for step in range(steps):
        L1, L2 = losses(p, X, y1, y2)
        history["L1"].append(L1)
        history["L2"].append(L2)
        history["total_loss"].append(L1 + L2)

        if method == "equal":
            g = backprop(p, X, y1, y2, task_weights=(1.0, 1.0))
            p = {k: p[k] - lr * g[k] for k in p}
            history["alpha"].append(0.5)
            history["w1"].append(1.0)
            history["w2"].append(1.0)
            history["cosine_sim"].append(np.nan)

        elif method == "mgda":
            p, alpha, _, _, cos = mgda_step(p, X, y1, y2, lr)
            history["alpha"].append(alpha)
            history["w1"].append(np.nan)
            history["w2"].append(np.nan)
            history["cosine_sim"].append(cos)

        elif method == "gradnorm":
            p, task_weights, gn_loss, _ = gradnorm_step(
                p, task_weights, X, y1, y2, lr, lr_weights, L1_0, L2_0, alpha_gn
            )
            history["alpha"].append(np.nan)
            history["w1"].append(float(task_weights[0]))
            history["w2"].append(float(task_weights[1]))
            history["cosine_sim"].append(np.nan)

    return history
