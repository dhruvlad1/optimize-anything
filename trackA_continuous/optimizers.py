"""
trackA_continuous/optimizers.py
--------------------------------
Pure-NumPy implementations of:
  gd, sgd, minibatch_sgd, momentum, nag, adagrad, rmsprop, adam, newton

Common return dict
------------------
{
  "path"       : np.ndarray (n_iters+1, dim)  -- iterates including x0
  "fvals"      : np.ndarray (n_iters+1,)      -- f(x) at each iterate
  "gnorms"     : np.ndarray (n_iters+1,)      -- ||grad f(x)|| at each iterate
  "n_iters"    : int                           -- iterations actually taken
  "n_grad_evals": int                          -- gradient evaluations
  "runtime"    : float                         -- wall-clock seconds
}
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(x0: np.ndarray, lr: float, max_iter: int, gtol: float) -> None:
    if not isinstance(x0, np.ndarray):
        raise TypeError("x0 must be a numpy array")
    if lr <= 0:
        raise ValueError(f"learning rate must be positive, got {lr}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if gtol < 0:
        raise ValueError(f"gtol must be >= 0, got {gtol}")


def _init_records(x0: np.ndarray, f: Callable, grad: Callable):
    """Pre-allocate result containers and record step 0."""
    fval0 = f(x0)
    gnorm0 = float(np.linalg.norm(grad(x0)))
    return fval0, gnorm0


def _pack(path: list, fvals: list, gnorms: list,
          n_iters: int, n_ge: int, t0: float) -> dict:
    return {
        "path": np.array(path),
        "fvals": np.array(fvals),
        "gnorms": np.array(gnorms),
        "n_iters": n_iters,
        "n_grad_evals": n_ge,
        "runtime": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# 1. Gradient Descent
#    x_{k+1} = x_k - lr * grad(x_k)
# ---------------------------------------------------------------------------

def gd(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 2000,
    gtol: float = 1e-8,
) -> dict:
    """Full-batch gradient descent with fixed step size."""
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        g = grad(x)
        n_ge += 1
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        x = x - lr * g           # GD update
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 2. SGD  (stochastic gradient oracle takes (x, rng))
#    x_{k+1} = x_k - lr_k * g~(x_k, rng)
#    decay: lr_k = lr / (1 + decay * k)  if decay > 0
# ---------------------------------------------------------------------------

def sgd(
    f: Callable,
    stoch_grad: Callable,   # stoch_grad(x, rng) -> gradient estimate
    x0: np.ndarray,
    lr: float = 0.01,
    decay: float = 0.0,     # lr_k = lr / (1 + decay*k); 0 = constant
    max_iter: int = 2000,
    gtol: float = 1e-8,
    seed: int = 0,
    full_grad: Optional[Callable] = None,  # for gnorm tracking
) -> dict:
    """SGD with optional learning-rate decay."""
    _validate(x0, lr, max_iter, gtol)
    rng = np.random.default_rng(seed)
    x = x0.copy().astype(float)
    _grad = full_grad if full_grad is not None else stoch_grad
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(_grad(x, rng))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        lr_k = lr / (1.0 + decay * k)
        g = stoch_grad(x, rng)   # stochastic gradient
        n_ge += 1
        full_g = full_grad(x, rng) if full_grad is not None else g
        gnorm = float(np.linalg.norm(full_g))
        if gnorm <= gtol:
            break
        x = x - lr_k * g         # SGD update
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(float(np.linalg.norm(g)))

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 3. Mini-Batch SGD
#    Same as SGD but stoch_grad receives (x, rng, batch_size)
# ---------------------------------------------------------------------------

def minibatch_sgd(
    f: Callable,
    stoch_grad: Callable,   # stoch_grad(x, rng, batch_size) -> gradient estimate
    x0: np.ndarray,
    lr: float = 0.01,
    batch_size: int = 8,
    decay: float = 0.0,
    max_iter: int = 2000,
    gtol: float = 1e-8,
    seed: int = 0,
    full_grad: Optional[Callable] = None,
) -> dict:
    """Mini-batch SGD."""
    _validate(x0, lr, max_iter, gtol)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rng = np.random.default_rng(seed)
    x = x0.copy().astype(float)
    _grad = full_grad if full_grad is not None else lambda xx, r: stoch_grad(xx, r, batch_size)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(_grad(x, rng))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        lr_k = lr / (1.0 + decay * k)
        g = stoch_grad(x, rng, batch_size)   # mini-batch gradient
        n_ge += 1                             # count as 1 eval regardless of batch size
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        x = x - lr_k * g
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 4. Momentum (Heavy Ball)
#    m_{k+1} = beta * m_k + grad(x_k)
#    x_{k+1} = x_k - lr * m_{k+1}
# ---------------------------------------------------------------------------

def momentum(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    max_iter: int = 2000,
    gtol: float = 1e-8,
) -> dict:
    """Heavy-ball / momentum gradient descent."""
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    m = np.zeros_like(x)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        g = grad(x)
        n_ge += 1
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        m = beta * m + g           # accumulate momentum
        x = x - lr * m             # update
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 5. Nesterov Accelerated Gradient (NAG)
#    y_k   = x_k + beta*(x_k - x_{k-1})
#    x_{k+1} = y_k - lr * grad(y_k)
# ---------------------------------------------------------------------------

def nag(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    max_iter: int = 2000,
    gtol: float = 1e-8,
) -> dict:
    """Nesterov accelerated gradient."""
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    x_prev = x.copy()
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        y = x + beta * (x - x_prev)   # look-ahead point
        g = grad(y)
        n_ge += 1
        gnorm = float(np.linalg.norm(grad(x)))  # track norm at current x
        if gnorm <= gtol:
            break
        x_prev = x.copy()
        x = y - lr * g               # update from look-ahead
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 6. AdaGrad
#    G_{k+1} = G_k + g_k^2   (element-wise)
#    x_{k+1} = x_k - lr / (sqrt(G_{k+1}) + eps) * g_k
# ---------------------------------------------------------------------------

def adagrad(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    lr: float = 0.1,
    eps: float = 1e-8,
    max_iter: int = 2000,
    gtol: float = 1e-8,
) -> dict:
    """AdaGrad adaptive gradient method."""
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        g = grad(x)
        n_ge += 1
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        G = G + g ** 2                            # accumulate squared grads
        x = x - lr / (np.sqrt(G) + eps) * g      # adaptive update
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 7. RMSProp
#    v_{k+1} = rho * v_k + (1-rho) * g_k^2
#    x_{k+1} = x_k - lr / (sqrt(v_{k+1}) + eps) * g_k
# ---------------------------------------------------------------------------

def rmsprop(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    lr: float = 0.01,
    rho: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = 2000,
    gtol: float = 1e-8,
) -> dict:
    """RMSProp."""
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        g = grad(x)
        n_ge += 1
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        v = rho * v + (1.0 - rho) * g ** 2       # EMA of squared grads
        x = x - lr / (np.sqrt(v) + eps) * g      # adaptive update
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 8. Adam
#    m_{k+1} = b1*m_k + (1-b1)*g_k
#    v_{k+1} = b2*v_k + (1-b2)*g_k^2
#    m_hat   = m_{k+1} / (1 - b1^{k+1})
#    v_hat   = v_{k+1} / (1 - b2^{k+1})
#    x_{k+1} = x_k - lr / (sqrt(v_hat) + eps) * m_hat
# ---------------------------------------------------------------------------

def adam(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 2000,
    gtol: float = 1e-8,
) -> dict:
    """Adam optimizer."""
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        g = grad(x)
        n_ge += 1
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_hat = m / (1.0 - beta1 ** (k + 1))     # bias correction
        v_hat = v / (1.0 - beta2 ** (k + 1))
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)


# ---------------------------------------------------------------------------
# 9. Newton's Method
#    Solve  H p = g,  then  x_{k+1} = x_k - lr * p
#    mode="pure"  : use H as-is
#    mode="damped": add lambda*I until H+lI is positive definite
# ---------------------------------------------------------------------------

def newton(
    f: Callable,
    grad: Callable,
    hess: Callable,
    x0: np.ndarray,
    lr: float = 1.0,
    mode: str = "pure",       # "pure" or "damped"
    lam_init: float = 1e-3,   # initial lambda for damped mode
    lam_max: float = 1e10,
    max_iter: int = 200,
    gtol: float = 1e-8,
) -> dict:
    """Newton's method with pure or damped Hessian."""
    if mode not in ("pure", "damped"):
        raise ValueError("mode must be 'pure' or 'damped'")
    _validate(x0, lr, max_iter, gtol)
    x = x0.copy().astype(float)
    path, fvals, gnorms = [x.copy()], [f(x)], [np.linalg.norm(grad(x))]
    n_ge = 1
    t0 = time.perf_counter()

    for k in range(max_iter):
        g = grad(x)
        H = hess(x)
        n_ge += 1
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break

        if mode == "pure":
            try:
                p = np.linalg.solve(H, g)   # H p = g
            except np.linalg.LinAlgError:
                p = g                        # fallback to GD step
        else:  # damped
            lam = lam_init
            while lam < lam_max:
                H_reg = H + lam * np.eye(len(g))
                # Check positive definite via Cholesky
                try:
                    np.linalg.cholesky(H_reg)
                    p = np.linalg.solve(H_reg, g)
                    break
                except np.linalg.LinAlgError:
                    lam *= 2.0
            else:
                p = g   # fallback

        x = x - lr * p
        path.append(x.copy())
        fvals.append(f(x))
        gnorms.append(gnorm)

    return _pack(path, fvals, gnorms, k + 1, n_ge, t0)
