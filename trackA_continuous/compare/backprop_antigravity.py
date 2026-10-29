"""Hand-written backpropagation for a 2-layer neural network with tanh and softmax cross-entropy.

Defines:
  - init_params(d, h, k, seed) -> dict
  - loss_and_grads(params, X, y, l2=0.0) -> (loss, grads)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def init_params(d: int, h: int, k: int, seed: int | None = None) -> Dict[str, np.ndarray]:
    """Initialize parameters for a 2-layer network.

    Args:
        d: Input feature dimension.
        h: Hidden layer dimension.
        k: Number of output classes.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys 'W1' (d, h), 'b1' (h,), 'W2' (h, k), 'b2' (k,).
        Weights are 0.5 * standard normal from np.random.default_rng(seed);
        biases are zeros.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": 0.5 * rng.standard_normal((d, h)),
        "b1": np.zeros(h, dtype=float),
        "W2": 0.5 * rng.standard_normal((h, k)),
        "b2": np.zeros(k, dtype=float),
    }


def loss_and_grads(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Compute loss and exact analytical gradients via hand-written backpropagation.

    Architecture:
        hidden = tanh(X @ W1 + b1)
        logits = hidden @ W2 + b2
        loss   = mean softmax cross-entropy over n samples + 0.5 * l2 * (|W1|^2 + |W2|^2)

    Args:
        params: Dict containing 'W1', 'b1', 'W2', 'b2'.
        X: Input features of shape (n, d).
        y: Target integer class labels in 0..k-1 of shape (n,).
        l2: L2 weight regularization penalty coefficient.

    Returns:
        (loss, grads) where loss is a float scalar and grads has the same keys
        and shapes as params.
    """
    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}.")

    required_keys = ("W1", "b1", "W2", "b2")
    for key in required_keys:
        if key not in params:
            raise KeyError(f"params missing required key '{key}'.")

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"X must be a 2D numpy array of shape (n, d), got shape {getattr(X, 'shape', None)}.")

    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(f"y must be a 1D numpy array of shape (n,), got shape {getattr(y, 'shape', None)}.")

    n, d = X.shape
    if n == 0:
        raise ValueError("X must contain at least one sample (n > 0).")

    if y.shape[0] != n:
        raise ValueError(f"Batch size mismatch: X has {n} samples, but y has {y.shape[0]} labels.")

    if W1.ndim != 2 or W1.shape[0] != d:
        raise ValueError(f"W1 shape mismatch: expected ({d}, h), got {W1.shape}.")
    h = W1.shape[1]

    if b1.ndim != 1 or b1.shape[0] != h:
        raise ValueError(f"b1 shape mismatch: expected ({h},), got {b1.shape}.")

    if W2.ndim != 2 or W2.shape[0] != h:
        raise ValueError(f"W2 shape mismatch: expected ({h}, k), got {W2.shape}.")
    k = W2.shape[1]

    if b2.ndim != 1 or b2.shape[0] != k:
        raise ValueError(f"b2 shape mismatch: expected ({k},), got {b2.shape}.")

    if np.any(y < 0) or np.any(y >= k):
        raise ValueError(f"Target labels y must contain integers in range [0, {k - 1}].")

    if l2 < 0.0:
        raise ValueError(f"l2 regularization strength must be non-negative, got {l2}.")

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    # 1. First layer pre-activations: a1 = X @ W1 + b1  (shape: n, h)
    a1 = X @ W1 + b1

    # 2. Hidden layer activation: hidden = tanh(a1)     (shape: n, h)
    hidden = np.tanh(a1)

    # 3. Output layer pre-activations: logits = hidden @ W2 + b2  (shape: n, k)
    logits = hidden @ W2 + b2

    # 4. Numerically stable softmax:
    #    Subtract max along class axis for numerical stability (prevents overflow for large logits).
    shift_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    probs = exp_logits / sum_exp  # P_{i, j} (shape: n, k)

    # 5. Stable cross-entropy loss:
    #    log(P_{i, y_i}) = (z_{i, y_i} - max_z_i) - log(sum_j exp(z_{i, j} - max_z_i))
    log_sum_exp = np.log(sum_exp)
    log_probs = shift_logits - log_sum_exp
    ce_loss = -np.mean(log_probs[np.arange(n), y])

    # 6. L2 weight regularization: 0.5 * l2 * (||W1||_F^2 + ||W2||_F^2)
    l2_reg = 0.5 * l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    total_loss = float(ce_loss + l2_reg)

    # -------------------------------------------------------------------------
    # Backward Pass (Hand-written Backpropagation)
    # -------------------------------------------------------------------------
    # Step 1: Chain rule for loss w.r.t. output logits z:
    #   d(ce_loss)/d(z_{i, j}) = (P_{i, j} - 1{y_i == j}) / n
    #   Since probs is bounded in [0, 1], dlogits stays strictly finite for any logit magnitude.
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n  # shape: (n, k)

    # Step 2: Chain rule for layer 2 weights and bias:
    #   logits = hidden @ W2 + b2
    #   dW2 = hidden.T @ dlogits + d(l2_reg)/dW2 = hidden.T @ dlogits + l2 * W2
    #   db2 = sum over samples of dlogits (bias is broadcast across samples)
    dW2 = hidden.T @ dlogits + l2 * W2      # shape: (h, k)
    db2 = np.sum(dlogits, axis=0)           # shape: (k,)

    # Step 3: Chain rule for hidden activation:
    #   dhidden = dlogits @ W2.T
    dhidden = dlogits @ W2.T                # shape: (n, h)

    # Step 4: Chain rule through elementwise tanh activation:
    #   hidden = tanh(a1)
    #   d(tanh(a1))/d(a1) = 1 - tanh^2(a1) = 1 - hidden^2
    #   da1 = dhidden * (1 - hidden^2)
    da1 = dhidden * (1.0 - hidden ** 2)     # shape: (n, h)

    # Step 5: Chain rule for layer 1 weights and bias:
    #   a1 = X @ W1 + b1
    #   dW1 = X.T @ da1 + d(l2_reg)/dW1 = X.T @ da1 + l2 * W1
    #   db1 = sum over samples of da1 (bias is broadcast across samples)
    dW1 = X.T @ da1 + l2 * W1               # shape: (d, h)
    db1 = np.sum(da1, axis=0)               # shape: (h,)

    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

    return total_loss, grads
