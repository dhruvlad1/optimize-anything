"""
trackA_continuous/objectives.py
--------------------------------
Analytic objective functions for Track A (Continuous Optimization).

Every gradient and Hessian is written by hand – no autodiff.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import numpy as np


@dataclasses.dataclass
class Objective:
    name: str
    dim: int
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Optional[Callable[[np.ndarray], np.ndarray]] = None
    x_star: Optional[np.ndarray] = None   # known minimizer (or one of them)
    f_star: Optional[float] = None         # known minimum value
    L: Optional[float] = None              # smoothness constant
    mu: Optional[float] = None             # strong-convexity constant


# ---------------------------------------------------------------------------
# 1. Quadratic  f(x) = 0.5 x^T A x,  A = diag(lambdas)
#    eigenvalues log-spaced from 1 to kappa  =>  mu=1, L=kappa
# ---------------------------------------------------------------------------

def quadratic(kappa: float = 10.0, dim: int = 10) -> Objective:
    """Strongly convex diagonal quadratic with condition number kappa."""
    if kappa < 1:
        raise ValueError("kappa must be >= 1")
    lambdas = np.logspace(0, math.log10(kappa), dim)   # 1 … kappa

    def f(x: np.ndarray) -> float:
        return float(0.5 * np.dot(x, lambdas * x))

    def grad(x: np.ndarray) -> np.ndarray:
        return lambdas * x                              # A x

    def hess(x: np.ndarray) -> np.ndarray:             # noqa: ARG001
        return np.diag(lambdas)                        # constant A

    return Objective(
        name=f"quadratic(kappa={kappa},d={dim})",
        dim=dim,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.zeros(dim),
        f_star=0.0,
        L=float(kappa),
        mu=1.0,
    )


# ---------------------------------------------------------------------------
# 2. Quartic  f(x) = sum x_i^4   (convex, NOT strongly convex, mu=0)
# ---------------------------------------------------------------------------

def quartic(dim: int = 10) -> Objective:
    """Convex but not strongly convex: f(x)=sum(x_i^4)."""

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 4))

    def grad(x: np.ndarray) -> np.ndarray:
        return 4.0 * x ** 3

    def hess(x: np.ndarray) -> np.ndarray:
        return np.diag(12.0 * x ** 2)

    return Objective(
        name="quartic",
        dim=dim,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.zeros(dim),
        f_star=0.0,
        L=None,    # L depends on domain; not bounded globally
        mu=0.0,
    )


# ---------------------------------------------------------------------------
# 3. Saddle  f(x,y) = x^2 - y^2 + 0.25 y^4
#    Saddle at (0,0); minima at (0, ±sqrt(2)), f*=-1
# ---------------------------------------------------------------------------

def saddle() -> Objective:
    """2-D function with a saddle at the origin."""
    x_star = np.array([0.0, math.sqrt(2.0)])
    f_star = -1.0

    def f(x: np.ndarray) -> float:
        return float(x[0] ** 2 - x[1] ** 2 + 0.25 * x[1] ** 4)

    def grad(x: np.ndarray) -> np.ndarray:
        gx = 2.0 * x[0]
        gy = -2.0 * x[1] + x[1] ** 3
        return np.array([gx, gy])

    def hess(x: np.ndarray) -> np.ndarray:
        h00 = 2.0
        h11 = -2.0 + 3.0 * x[1] ** 2
        return np.array([[h00, 0.0], [0.0, h11]])

    return Objective(
        name="saddle",
        dim=2,
        f=f,
        grad=grad,
        hess=hess,
        x_star=x_star,
        f_star=f_star,
    )


# ---------------------------------------------------------------------------
# 4. Himmelblau  f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2  four minima with f=0
# ---------------------------------------------------------------------------

_HIMMELBLAU_MINIMA = [
    np.array([3.0, 2.0]),
    np.array([-2.805118, 3.131312]),
    np.array([-3.779310, -3.283186]),
    np.array([3.584428, -1.848126]),
]


def himmelblau() -> Objective:
    """Himmelblau's function: four local minima, all with f=0."""

    def f(x: np.ndarray) -> float:
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        return float(a ** 2 + b ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        gx = 4.0 * a * x[0] + 2.0 * b
        gy = 2.0 * a + 4.0 * b * x[1]
        return np.array([gx, gy])

    def hess(x: np.ndarray) -> np.ndarray:
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        h00 = 12.0 * x[0] ** 2 + 4.0 * x[1] - 44.0 + 2.0      # d²f/dx²
        h11 = 2.0 + 12.0 * x[1] ** 2 + 4.0 * x[0] - 28.0      # d²f/dy²
        h01 = 4.0 * x[0] + 4.0 * x[1]                           # d²f/dxdy
        return np.array([[h00, h01], [h01, h11]])

    return Objective(
        name="himmelblau",
        dim=2,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.array([3.0, 2.0]),
        f_star=0.0,
    )


def himmelblau_minima() -> list[np.ndarray]:
    """Return the list of four known Himmelblau minima."""
    return [m.copy() for m in _HIMMELBLAU_MINIMA]


# ---------------------------------------------------------------------------
# 5. Rosenbrock  f(x,y)=(1-x)^2+100(y-x^2)^2  min at (1,1), f*=0
# ---------------------------------------------------------------------------

def rosenbrock() -> Objective:
    """Rosenbrock banana function (2-D)."""

    def f(x: np.ndarray) -> float:
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        gx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
        gy = 200.0 * (x[1] - x[0] ** 2)
        return np.array([gx, gy])

    def hess(x: np.ndarray) -> np.ndarray:
        h00 = 2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2
        h11 = 200.0
        h01 = -400.0 * x[0]
        return np.array([[h00, h01], [h01, h11]])

    return Objective(
        name="rosenbrock",
        dim=2,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.array([1.0, 1.0]),
        f_star=0.0,
    )


# ---------------------------------------------------------------------------
# 6. Rastrigin (2-D)  f(x) = 10d + sum(x_i^2 - 10 cos(2 pi x_i))
#    global min at 0 with f*=0; many local minima
# ---------------------------------------------------------------------------

def rastrigin(dim: int = 2) -> Objective:
    """Rastrigin function: highly multimodal, global min at 0."""
    two_pi = 2.0 * math.pi

    def f(x: np.ndarray) -> float:
        return float(10.0 * dim + np.sum(x ** 2 - 10.0 * np.cos(two_pi * x)))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 20.0 * math.pi * np.sin(two_pi * x)

    def hess(x: np.ndarray) -> np.ndarray:
        diag = 2.0 + 40.0 * math.pi ** 2 * np.cos(two_pi * x)
        return np.diag(diag)

    return Objective(
        name=f"rastrigin(d={dim})",
        dim=dim,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.zeros(dim),
        f_star=0.0,
    )
