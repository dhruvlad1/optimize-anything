"""Two-layer tanh network with softmax cross-entropy and hand-written backprop (NumPy only)."""

import numpy as np

_PARAM_KEYS = ("W1", "b1", "W2", "b2")


def _check_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def init_params(d, h, k, seed):
    """Weights: 0.5 * standard normal from default_rng(seed); biases: zeros."""
    _check_positive_int("d", d)
    _check_positive_int("h", h)
    _check_positive_int("k", k)
    rng = np.random.default_rng(seed)
    return {
        "W1": 0.5 * rng.standard_normal((d, h)),
        "b1": np.zeros(h),
        "W2": 0.5 * rng.standard_normal((h, k)),
        "b2": np.zeros(k),
    }


def _validate(params, X, y, l2):
    """Validate all inputs; return (W1, b1, W2, b2, X, y) as arrays (no copies made or inputs mutated)."""
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")
    missing = [key for key in _PARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"params is missing keys: {missing}")

    W1, b1, W2, b2 = (np.asarray(params[key]) for key in _PARAM_KEYS)
    for name, arr, ndim in (("W1", W1, 2), ("b1", b1, 1), ("W2", W2, 2), ("b2", b2, 1)):
        if arr.ndim != ndim:
            raise ValueError(f"params['{name}'] must be {ndim}-D, got shape {arr.shape}")
        if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
            raise ValueError(f"params['{name}'] must be real numeric, got dtype {arr.dtype}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"params['{name}'] contains NaN or inf")

    d, h = W1.shape
    k = W2.shape[1]
    if b1.shape != (h,):
        raise ValueError(f"params['b1'] must have shape ({h},) to match W1 {W1.shape}, got {b1.shape}")
    if W2.shape[0] != h:
        raise ValueError(f"params['W2'] must have shape ({h}, k) to match W1 {W1.shape}, got {W2.shape}")
    if b2.shape != (k,):
        raise ValueError(f"params['b2'] must have shape ({k},) to match W2 {W2.shape}, got {b2.shape}")

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D with shape (n, {d}), got shape {X.shape}")
    if X.shape[1] != d:
        raise ValueError(f"X must have {d} columns to match W1 {W1.shape}, got shape {X.shape}")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one sample")
    if not np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.complexfloating):
        raise ValueError(f"X must be real numeric, got dtype {X.dtype}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or inf")
    n = X.shape[0]

    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"y must be 1-D with shape ({n},) to match X, got shape {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"y must hold integer class labels, got dtype {y.dtype}")
    if y.min() < 0 or y.max() >= k:
        raise ValueError(
            f"y labels must be in [0, {k - 1}], got range [{y.min()}, {y.max()}]"
        )

    if isinstance(l2, bool) or not isinstance(l2, (int, float, np.integer, np.floating)):
        raise TypeError(f"l2 must be a real number, got {type(l2).__name__}")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError(f"l2 must be finite and >= 0, got {l2!r}")

    return W1, b1, W2, b2, X, y


def loss_and_grads(params, X, y, l2=0.0):
    """Return (loss, grads) for the tanh network with mean softmax CE + 0.5*l2*(|W1|^2 + |W2|^2)."""
    W1, b1, W2, b2, X, y = _validate(params, X, y, l2)
    n = X.shape[0]
    rows = np.arange(n)

    # ---- Forward pass (all intermediates are new arrays; inputs are never written to) ----
    pre = X @ W1 + b1                      # (n, h)  pre-activation
    hidden = np.tanh(pre)                  # (n, h)
    logits = hidden @ W2 + b2              # (n, k)

    # Stable log-softmax: subtracting the row max leaves softmax unchanged
    # and makes exp() arguments <= 0, so nothing overflows even for |logits| ~ 1000.
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # sum >= 1, so log is finite
    log_probs = shifted - log_sum_exp      # (n, k)

    data_loss = -np.mean(log_probs[rows, y])
    reg_loss = 0.5 * l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    loss = float(data_loss + reg_loss)

    # ---- Backward pass ----
    # Step 1: loss -> logits.
    #   L_i = -log p_{i,y_i},  p = softmax(z)  =>  dL_i/dz_ij = p_ij - 1[j == y_i].
    #   The loss is the mean over n samples, so multiply by 1/n.
    dlogits = np.exp(log_probs)            # p (new array)
    dlogits[rows, y] -= 1.0
    dlogits /= n                           # (n, k)

    # Step 2: logits = hidden @ W2 + b2.
    #   dL/dW2 = hidden^T @ dlogits   (since dz_ij/dW2_cj = hidden_ic)
    #   dL/db2 = sum_i dlogits_i      (b2 is added to every row)
    #   plus the L2 term: d/dW2 [0.5*l2*|W2|^2] = l2 * W2
    dW2 = hidden.T @ dlogits + l2 * W2     # (h, k)
    db2 = dlogits.sum(axis=0)              # (k,)

    # Step 3: propagate to hidden activations.
    #   dL/dhidden = dlogits @ W2^T   (since dz_ij/dhidden_ic = W2_cj)
    dhidden = dlogits @ W2.T               # (n, h)

    # Step 4: through tanh.
    #   hidden = tanh(pre)  =>  d hidden / d pre = 1 - tanh(pre)^2 = 1 - hidden^2
    #   dL/dpre = dL/dhidden * (1 - hidden^2)   (elementwise)
    dpre = dhidden * (1.0 - hidden ** 2)   # (n, h)

    # Step 5: pre = X @ W1 + b1.
    #   dL/dW1 = X^T @ dpre   (since dpre_ic/dW1_dc = X_id)
    #   dL/db1 = sum_i dpre_i
    #   plus the L2 term: l2 * W1
    dW1 = X.T @ dpre + l2 * W1             # (d, h)
    db1 = dpre.sum(axis=0)                 # (h,)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return loss, grads