"""trackC_quantum_opt/optimizers_q.py
-----------------------------------
Classical optimizers for variational quantum algorithms:
  - SPSA (Simultaneous Perturbation Stochastic Approximation)
  - Adam with Parameter-Shift Gradients (accounting 2 evals per parameter)
  - Scipy COBYLA wrapper with strict evaluation counting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizerResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    history: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


def spsa(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    iters: Optional[int] = None,
    budget: Optional[int] = 300,
    a: float = 0.2,
    c: float = 0.1,
    seed: int = 0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    A: Optional[float] = None,
) -> OptimizerResult:
    """SPSA optimizer (Spall 1992) for minimizing loss function f.

    At each iteration k, evaluates f at two perturbed points:
        x_plus  = x + c_k * delta
        x_minus = x - c_k * delta
    where delta in {-1, +1}^d is a Bernoulli / Rademacher random vector.
    Each iteration consumes exactly 2 function evaluations.
    """
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    rng = np.random.default_rng(seed)

    if iters is None:
        if budget is None:
            raise ValueError("Either iters or budget must be specified.")
        iters = budget // 2
    else:
        budget = iters * 2

    if A is None:
        A = 0.1 * iters

    best_x = x.copy()
    best_f = float("inf")
    history = []
    nfev = 0

    for k in range(iters):
        if nfev + 2 > budget:
            break

        # Gain sequences
        a_k = a / ((k + 1 + A) ** alpha)
        c_k = c / ((k + 1) ** gamma)

        # Rademacher perturbation vector delta in {-1, +1}^d
        delta = rng.choice([-1.0, 1.0], size=d)

        # Two function evaluations
        f_plus = float(f(x + c_k * delta))
        f_minus = float(f(x - c_k * delta))
        nfev += 2

        # Stochastic gradient estimate
        g_k = (f_plus - f_minus) / (2.0 * c_k) * delta

        # Update
        x -= a_k * g_k

        # Track objective (using midpoint / minimum of perturbed points)
        cur_min = min(f_plus, f_minus)
        if cur_min < best_f:
            best_f = cur_min
            best_x = (x + c_k * delta).copy() if f_plus < f_minus else (x - c_k * delta).copy()
        history.append(best_f)

    # Evaluate final point if budget permits, or update best_f
    return OptimizerResult(
        x=best_x,
        fun=best_f,
        nfev=nfev,
        nit=k + 1 if iters > 0 else 0,
        history=history,
    )


def adam_ps(
    loss_and_grad_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    iters: Optional[int] = None,
    budget: Optional[int] = 300,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> OptimizerResult:
    """Adam optimizer using a supplied exact parameter-shift gradient function.

    Accounting rule:
        Each gradient evaluation costs 2 circuit evaluations per parameter (2 * d).
        Therefore, each Adam step consumes 2 * d circuit evaluations.
    """
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    evals_per_step = 2 * d

    if iters is None:
        if budget is None:
            raise ValueError("Either iters or budget must be specified.")
        iters = max(1, budget // evals_per_step)
    else:
        budget = iters * evals_per_step

    m = np.zeros(d, dtype=float)
    v = np.zeros(d, dtype=float)

    best_x = x.copy()
    best_f = float("inf")
    history = []
    nfev = 0

    for t in range(1, iters + 1):
        if nfev + evals_per_step > budget:
            break

        f_val, grad = loss_and_grad_fn(x)
        nfev += evals_per_step

        if f_val < best_f:
            best_f = f_val
            best_x = x.copy()
        history.append(best_f)

        # Adam momentum and velocity updates
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)

        # Bias corrections
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        x -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return OptimizerResult(
        x=best_x,
        fun=best_f,
        nfev=nfev,
        nit=t if iters > 0 else 0,
        history=history,
    )


def cobyla_wrapper(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    budget: int = 300,
    rhobeg: float = 0.5,
) -> OptimizerResult:
    """Wrapper around scipy.optimize.minimize(method='COBYLA') with exact evaluation counting."""
    x_init = np.asarray(x0, dtype=float).copy()
    eval_count = 0
    best_f = float("inf")
    best_x = x_init.copy()
    history = []

    def wrapped_f(x_cand: np.ndarray) -> float:
        nonlocal eval_count, best_f, best_x
        if eval_count >= budget:
            return best_f
        val = float(f(x_cand))
        eval_count += 1
        if val < best_f:
            best_f = val
            best_x = x_cand.copy()
        history.append(best_f)
        return val

    res = minimize(
        wrapped_f,
        x_init,
        method="COBYLA",
        options={"maxiter": budget, "rhobeg": rhobeg},
    )

    return OptimizerResult(
        x=best_x,
        fun=best_f,
        nfev=eval_count,
        nit=int(res.nfev) if hasattr(res, "nfev") else eval_count,
        history=history,
    )
