"""trackD_parallel/federated.py
------------------------------
Federated learning simulation: FedAvg vs centralised SGD for binary
logistic regression on a synthetic dataset.

All gradients are hand-written NumPy; no torch.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def make_clients(
    K: int = 10,
    d: int = 20,
    n_per_client: int = 200,
    heterogeneity: float = 0.0,
    l2: float = 1e-3,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generate K federated clients with controlled heterogeneity.

    Parameters
    ----------
    heterogeneity : float in [0, 1]
        0 → i.i.d.  data from the same distribution.
        1 → maximum label skew + per-client feature shift.

    Returns
    -------
    clients : list of (X, y) arrays per client
    w_true  : the shared ground-truth weight vector (d,)
    """
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
    if not isinstance(n_per_client, (int, np.integer)) or n_per_client < 1:
        raise ValueError("n_per_client must be a positive integer")
    if not np.isfinite(heterogeneity) or not 0 <= heterogeneity <= 1:
        raise ValueError("heterogeneity must be in [0, 1]")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and non-negative")
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    w_true /= np.linalg.norm(w_true)

    clients: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(K):
        # Feature shift: each client gets a slight mean shift scaled by het
        mu = rng.standard_normal(d) * heterogeneity * 0.5
        X = rng.standard_normal((n_per_client, d)) + mu

        # Ground-truth probabilities
        prob = _sigmoid(X @ w_true)

        # Label noise
        y_clean = (prob > 0.5).astype(float)
        noise_mask = rng.random(n_per_client) < 0.05  # 5% flip
        y = np.where(noise_mask, 1.0 - y_clean, y_clean)

        # Label skew: for heterogeneity > 0, bias client k towards class (k % 2)
        if heterogeneity > 0.0:
            preferred = k % 2
            skew_mask = (y != preferred) & (rng.random(n_per_client) < heterogeneity * 0.6)
            y[skew_mask] = preferred

        clients.append((X.astype(np.float64), y.astype(np.float64)))

    return clients, w_true


# ---------------------------------------------------------------------------
# 2. Loss and gradient
# ---------------------------------------------------------------------------

def logistic_loss_and_grad(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
) -> Tuple[float, np.ndarray]:
    """Binary cross-entropy + L2 regularisation.

    Returns (loss, grad) for weight vector w of shape (d,).
    """
    w, X, y = np.asarray(w, dtype=float), np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or w.ndim != 1 or X.shape[1] != w.size or y.shape != (X.shape[0],) or X.shape[0] == 0:
        raise ValueError("Expected non-empty X (n,d), w (d,), and y (n,)")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("Inputs must be finite and labels in [0, 1]")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and non-negative")
    n = len(y)
    prob = _sigmoid(X @ w)
    prob = np.clip(prob, 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)) + 0.5 * l2 * np.dot(w, w)
    grad = X.T @ (prob - y) / n + l2 * w
    return float(loss), grad


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    pred = (_sigmoid(X @ w) >= 0.5).astype(float)
    return float(np.mean(pred == y))


# ---------------------------------------------------------------------------
# 3. Local SGD
# ---------------------------------------------------------------------------

def local_sgd(
    w: np.ndarray,
    data: Tuple[np.ndarray, np.ndarray],
    E: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
    l2: float = 1e-3,
) -> np.ndarray:
    """Run exactly E sampled mini-batch gradient steps on local data."""
    if not isinstance(E, (int, np.integer)) or E < 1 or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("E and batch_size must be positive integers")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    X, y = data
    if X.ndim != 2 or y.shape != (X.shape[0],) or np.asarray(w).shape != (X.shape[1],) or X.shape[0] == 0:
        raise ValueError("Invalid local data or weight shape")
    n = len(y)
    w = w.copy()
    for _ in range(E):
        batch = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb, yb = X[batch], y[batch]
        prob = _sigmoid(Xb @ w)
        grad = Xb.T @ (prob - yb) / len(batch) + l2 * w
        w = w - lr * grad
    return w


# ---------------------------------------------------------------------------
# 4. FedAvg
# ---------------------------------------------------------------------------

def fedavg(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    rounds: int = 100,
    E: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    participation: float = 1.0,
    seed: int = 7,
    l2: float = 1e-3,
) -> List[Dict]:
    """FedAvg algorithm.

    Parameters
    ----------
    participation : float in (0, 1]
        Fraction of clients selected each round (at least 1).

    Returns
    -------
    history : list of dicts with keys {round, loss, accuracy, comm_volume}
              comm_volume = number of weight-floats transmitted in total up to this round.
    """
    if not clients or any(len(c[1]) == 0 or c[0].ndim != 2 or c[1].shape != (c[0].shape[0],) for c in clients):
        raise ValueError("clients must contain non-empty (X, y) pairs")
    if not isinstance(rounds, (int, np.integer)) or rounds < 1 or not isinstance(E, (int, np.integer)) or E < 1:
        raise ValueError("rounds and E must be positive integers")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("batch_size and lr must be positive")
    if not 0 < participation <= 1 or not np.isfinite(participation):
        raise ValueError("participation must be in (0, 1]")
    if any(c[0].shape[1] != clients[0][0].shape[1] for c in clients):
        raise ValueError("all clients must have the same feature dimension")
    rng = np.random.default_rng(seed)
    K = len(clients)
    d = clients[0][0].shape[1]

    # Global model
    w_global = np.zeros(d)

    # Total dataset for evaluation
    X_all = np.concatenate([c[0] for c in clients], axis=0)
    y_all = np.concatenate([c[1] for c in clients], axis=0)

    # Client data sizes (for weighted average)
    client_sizes = np.array([len(c[1]) for c in clients], dtype=float)

    n_selected = max(1, int(round(participation * K)))

    history: List[Dict] = []
    total_comm = 0  # floats transmitted

    for r in range(rounds):
        # Select clients
        selected = rng.choice(K, size=n_selected, replace=False)

        # Each client trains locally and returns updated weights
        local_weights = []
        local_ns = []
        for k in selected:
            w_local = local_sgd(w_global, clients[k], E, batch_size, lr, rng, l2)
            local_weights.append(w_local)
            local_ns.append(client_sizes[k])

        # Weighted average (weighted by number of samples)
        total_n = sum(local_ns)
        w_global = sum(w * (n / total_n) for w, n in zip(local_weights, local_ns))

        # Communication: server → clients (broadcast) + clients → server (upload)
        # Count both directions: n_selected * d (download) + n_selected * d (upload)
        total_comm += 2 * n_selected * d

        loss, _ = logistic_loss_and_grad(w_global, X_all, y_all, l2)
        acc = accuracy(w_global, X_all, y_all)
        history.append({"round": r + 1, "loss": loss, "accuracy": acc, "comm_volume": total_comm,
                        "weights": w_global.copy()})

    return history


# ---------------------------------------------------------------------------
# 5. Centralised baselines
# ---------------------------------------------------------------------------

def centralized_sgd(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    total_steps: int,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 7,
    l2: float = 1e-3,
) -> List[Dict]:
    """Centralised SGD on the pooled dataset with the same number of gradient steps
    as FedAvg would use in total (rounds * selected_clients * E_per_epoch_steps).
    Returns per-step loss and accuracy history."""
    rng = np.random.default_rng(seed)
    X_all = np.concatenate([c[0] for c in clients], axis=0)
    y_all = np.concatenate([c[1] for c in clients], axis=0)
    n = len(y_all)
    d = X_all.shape[1]
    w = np.zeros(d)
    history: List[Dict] = []
    for step in range(total_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb, yb = X_all[idx], y_all[idx]
        prob = _sigmoid(Xb @ w)
        grad = Xb.T @ (prob - yb) / len(idx) + l2 * w
        w = w - lr * grad
        if (step + 1) % max(1, total_steps // 100) == 0 or step == total_steps - 1:
            loss, _ = logistic_loss_and_grad(w, X_all, y_all, l2)
            acc = accuracy(w, X_all, y_all)
            history.append({"step": step + 1, "loss": loss, "accuracy": acc})
    return history


def centralized_gd(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    l2: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """Centralised gradient descent on pooled data via scipy BFGS.
    Returns (w_opt, loss_opt).
    """
    from scipy.optimize import minimize

    X_all = np.concatenate([c[0] for c in clients], axis=0)
    y_all = np.concatenate([c[1] for c in clients], axis=0)
    d = X_all.shape[1]

    def obj_and_grad(w):
        loss, grad = logistic_loss_and_grad(w, X_all, y_all, l2)
        return loss, grad

    res = minimize(obj_and_grad, x0=np.zeros(d), jac=True, method="BFGS",
                   options={"maxiter": 2000, "gtol": 1e-8})
    return res.x, float(res.fun)


def fedavg_full_batch(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    rounds: int = 100,
    E: int = 1,
    lr: float = 0.01,
    l2: float = 1e-3,
) -> np.ndarray:
    """Special-case: E=1, full batch, full participation.
    Should be mathematically identical to centralised GD (one step per round on pooled data).
    Returns final weight vector.
    """
    K = len(clients)
    d = clients[0][0].shape[1]
    client_sizes = np.array([len(c[1]) for c in clients], dtype=float)
    total_n = client_sizes.sum()
    w_global = np.zeros(d)

    for _ in range(rounds):
        local_ws = []
        for k in range(K):
            X, y = clients[k]
            prob = _sigmoid(X @ w_global)
            grad = X.T @ (prob - y) / len(y) + l2 * w_global
            w_local = w_global - lr * grad
            local_ws.append(w_local)
        w_global = sum(w * (n / total_n) for w, n in zip(local_ws, client_sizes))

    return w_global


def centralized_gd_steps(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    rounds: int = 100,
    lr: float = 0.01,
    l2: float = 1e-3,
) -> np.ndarray:
    """Centralised GD: one full-batch gradient step per round on pooled data.
    Returns final weight vector.
    """
    X_all = np.concatenate([c[0] for c in clients], axis=0)
    y_all = np.concatenate([c[1] for c in clients], axis=0)
    n, d = X_all.shape
    w = np.zeros(d)
    for _ in range(rounds):
        prob = _sigmoid(X_all @ w)
        grad = X_all.T @ (prob - y_all) / n + l2 * w
        w = w - lr * grad
    return w
