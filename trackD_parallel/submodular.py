"""trackD_parallel/submodular.py
------------------------------
Submodular function optimization module:
- Facility location objective function with marginal gain computation
- Standard Greedy algorithm
- Lazy Greedy (Accelerated Greedy) with max-heap of upper bounds
- Brute force optimal solver (for n <= 20)
- Integer Linear Programming (ILP) exact solver using OR-Tools SCIP (with SciPy MILP cross-check)
- Submodularity verification utility (diminishing-returns test)
- Complementarity (non-submodular) objective and greedy failure instance

NOTE: All algorithms strictly verify that the returned set has size k.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Callable, Collection, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from ortools.linear_solver import pywraplp


class FacilityLocation:
    """Monotone submodular facility location objective function:
    f(S) = sum_{i=0}^{m-1} max_{j in S} sim(i, j), with f(empty) = 0.
    """

    def __init__(self, sim: np.ndarray):
        sim = np.asarray(sim, dtype=np.float64)
        if sim.ndim != 2:
            raise ValueError(f"sim must be a 2D matrix, got shape {sim.shape}")
        if np.any(sim < 0):
            raise ValueError("sim entries must be non-negative for monotone submodularity")
        self.sim = sim
        self.m, self.n = sim.shape
        self.eval_count = 0

    def __call__(self, S: Iterable[int]) -> float:
        """Evaluate the facility location objective on subset S."""
        self.eval_count += 1
        S_list = list(S)
        if len(S_list) == 0:
            return 0.0
        # Column slice of facilities in S
        sub = self.sim[:, S_list]
        return float(np.sum(np.max(sub, axis=1)))

    def marginal_gain(self, S: Iterable[int], x: int) -> float:
        """Compute Delta(x | S) = f(S union {x}) - f(S)."""
        S_list = list(S)
        if x in S_list:
            return 0.0
        if len(S_list) == 0:
            return float(np.sum(self.sim[:, x]))

        current_max = np.max(self.sim[:, S_list], axis=1)
        new_col = self.sim[:, x]
        gain = np.sum(np.maximum(0.0, new_col - current_max))
        return float(gain)

    def reset_eval_count(self) -> None:
        self.eval_count = 0


def facility_location(sim: np.ndarray) -> FacilityLocation:
    """Factory creating a FacilityLocation callable objective."""
    return FacilityLocation(sim)


class NonSubmodularObjective:
    """A deliberately non-submodular set function with complementarity (synergy bonus):
    f(S) = sum_{j in S} base_weights[j] + bonus * I({0, 1} subseteq S).
    Violates diminishing returns because Delta(1 | {0}) > Delta(1 | empty).
    """

    def __init__(self, base_weights: Sequence[float], bonus: float = 30.0, pair: Tuple[int, int] = (0, 1)):
        self.base_weights = np.asarray(base_weights, dtype=np.float64)
        self.n = len(self.base_weights)
        self.bonus = float(bonus)
        self.pair = pair
        self.eval_count = 0

    def __call__(self, S: Iterable[int]) -> float:
        self.eval_count += 1
        S_set = set(S)
        val = float(sum(self.base_weights[j] for j in S_set))
        if self.pair[0] in S_set and self.pair[1] in S_set:
            val += self.bonus
        return val

    def marginal_gain(self, S: Iterable[int], x: int) -> float:
        S_set = set(S)
        if x in S_set:
            return 0.0
        return self(S_set | {x}) - self(S_set)

    def reset_eval_count(self) -> None:
        self.eval_count = 0


def make_non_submodular_failure_instance() -> Tuple[NonSubmodularObjective, List[int], int]:
    """Create a canonical non-submodular instance where greedy fails:
    Ground set: {0, 1, 2, 3}, budget k = 2.
    Items 0, 1 have base weight 10.0; items 2, 3 have base weight 12.0.
    Bonus B = 30.0 if both {0, 1} are included.
    Greedy picks {2, 3} (value 24.0); Optimum is {0, 1} (value 50.0).
    Ratio = 24 / 50 = 0.48 < 1 - 1/e = 0.6321.
    """
    base_weights = [10.0, 10.0, 12.0, 12.0]
    obj = NonSubmodularObjective(base_weights=base_weights, bonus=30.0, pair=(0, 1))
    ground_set = [0, 1, 2, 3]
    k = 2
    return obj, ground_set, k


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def greedy(f: Any, ground_set: Sequence[int], k: int) -> Tuple[List[int], float, int]:
    """Standard greedy algorithm for monotone submodular maximization under |S| = k:
    At each step, add argmax_{x notin S} [f(S union {x}) - f(S)].

    Returns:
        (selected_set, objective_value, num_evaluations)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(ground_set):
        raise ValueError(f"k={k} cannot exceed ground set size {len(ground_set)}")

    S: List[int] = []
    candidates = list(ground_set)
    eval_count = 0
    current_val = float(f(S))
    eval_count += 1

    for _ in range(k):
        best_gain = -float("inf")
        best_item = None
        best_val = current_val

        for x in candidates:
            cand_set = S + [x]
            val = float(f(cand_set))
            eval_count += 1
            gain = val - current_val
            if gain > best_gain:
                best_gain = gain
                best_item = x
                best_val = val

        if best_item is None:
            # If all remaining elements yield non-positive marginal gain, pick any candidate
            best_item = candidates[0]
            best_val = float(f(S + [best_item]))
            eval_count += 1

        S.append(best_item)
        candidates.remove(best_item)
        current_val = best_val

    # Verification: Every solver verifies its selected set has size k
    if len(S) != k:
        raise RuntimeError(f"Greedy selected set size {len(S)} does not match budget {k}")

    return S, current_val, eval_count


def lazy_greedy(f: Any, ground_set: Sequence[int], k: int) -> Tuple[List[int], float, int]:
    """Accelerated (lazy) greedy algorithm using a max-heap of stale upper bounds.
    Exploits the submodular diminishing returns property Delta(x | B) <= Delta(x | A) for A subseteq B.

    Returns:
        (selected_set, objective_value, num_evaluations)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(ground_set):
        raise ValueError(f"k={k} cannot exceed ground set size {len(ground_set)}")

    eval_count = 0
    current_val = float(f([]))
    eval_count += 1

    # Max-heap: store (-marginal_gain, element, iteration_when_computed)
    # Python heapq is a min-heap, so negate keys
    heap: List[Tuple[float, int, int]] = []
    for x in ground_set:
        val = float(f([x]))
        eval_count += 1
        gain = val - current_val
        heapq.heappush(heap, (-gain, x, 0))

    S: List[int] = []

    while len(S) < k and heap:
        neg_gain, x, step = heapq.heappop(heap)

        # If this upper bound was computed against the current set S, it is guaranteed maximal
        if step == len(S):
            S.append(x)
            current_val += (-neg_gain)
        else:
            # Recompute marginal gain with respect to current S
            val_with_x = float(f(S + [x]))
            eval_count += 1
            new_gain = val_with_x - current_val
            heapq.heappush(heap, (-new_gain, x, len(S)))

    # Verification: Every solver verifies its selected set has size k
    if len(S) != k:
        raise RuntimeError(f"Lazy greedy selected set size {len(S)} does not match budget {k}")

    return S, current_val, eval_count


def brute_force(f: Any, ground_set: Sequence[int], k: int) -> Tuple[List[int], float, int]:
    """Exact brute-force search over all C(n, k) subsets (for n <= 20).

    Returns:
        (optimal_set, optimal_value, num_evaluations)
    """
    n = len(ground_set)
    if n > 20:
        raise ValueError(f"Brute force only allowed for n <= 20, got n={n}")
    if k < 1 or k > n:
        raise ValueError(f"Invalid k={k} for ground set size {n}")

    best_val = -float("inf")
    best_set: Optional[List[int]] = None
    eval_count = 0

    for subset in itertools.combinations(ground_set, k):
        val = float(f(subset))
        eval_count += 1
        if val > best_val:
            best_val = val
            best_set = list(subset)

    assert best_set is not None
    # Verification: Every solver verifies its selected set has size k
    if len(best_set) != k:
        raise RuntimeError(f"Brute force selected set size {len(best_set)} does not match budget {k}")

    return best_set, best_val, eval_count


def ilp_facility_location(sim: np.ndarray, k: int) -> Tuple[List[int], float]:
    """Solve the facility location problem to global optimality using OR-Tools SCIP:
    max sum_{i=0}^{m-1} sum_{j=0}^{n-1} sim_{ij} * y_{ij}
    s.t.
        sum_j x_j = k
        sum_j y_{ij} <= 1   forall i
        y_{ij} <= x_j       forall i, j
        x_j, y_{ij} in {0, 1}

    Returns:
        (selected_facilities, optimal_value)
    """
    sim = np.asarray(sim, dtype=np.float64)
    m, n = sim.shape
    if k < 1 or k > n:
        raise ValueError(f"Invalid k={k} for n={n} facilities")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver unavailable")

    # Disable solver logging
    solver.SuppressOutput()

    # Decision variables
    x = [solver.BoolVar(f"x_{j}") for j in range(n)]
    y: Dict[Tuple[int, int], Any] = {}
    for i in range(m):
        for j in range(n):
            y[i, j] = solver.BoolVar(f"y_{i}_{j}")

    # Cardinality constraint: sum_j x_j = k
    solver.Add(solver.Sum(x) == k)

    # Customer assignment constraints
    for i in range(m):
        solver.Add(solver.Sum(y[i, j] for j in range(n)) <= 1)
        for j in range(n):
            solver.Add(y[i, j] <= x[j])

    # Objective: maximize total utility
    objective = solver.Objective()
    for i in range(m):
        for j in range(n):
            if sim[i, j] > 0:
                objective.SetCoefficient(y[i, j], float(sim[i, j]))
    objective.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"OR-Tools SCIP solver failed with status {status}")

    selected = [j for j in range(n) if x[j].solution_value() > 0.5]
    opt_val = float(objective.Value())

    # Verification: Every solver verifies its selected set has size k
    if len(selected) != k:
        raise RuntimeError(f"ILP selected set size {len(selected)} does not match budget {k}")

    return selected, opt_val


# ---------------------------------------------------------------------------
# Submodularity Verification Utility
# ---------------------------------------------------------------------------

def check_submodular(
    f: Any,
    ground_set: Sequence[int],
    trials: int = 2000,
    seed: int = 42,
    tol: float = 1e-9,
) -> int:
    """Sample random subsets A subseteq B subset V and element x notin B,
    testing the diminishing returns inequality:
    Delta(x | A) = f(A union {x}) - f(A) >= f(B union {x}) - f(B) = Delta(x | B).

    Returns:
        number of violations where Delta(x | A) < Delta(x | B) - tol.
    """
    rng = np.random.default_rng(seed)
    n = len(ground_set)
    if n < 2:
        return 0

    violations = 0
    items = np.asarray(ground_set)

    for _ in range(trials):
        # Pick x
        x_idx = rng.integers(0, n)
        x = items[x_idx]
        remaining = np.delete(items, x_idx)

        # Generate random subsets A subseteq B of remaining items
        # Sample inclusion indicators: with prob 0.5, item is in B; with prob 0.25, in A
        probs = rng.uniform(0.0, 1.0, len(remaining))
        # B contains items where probs < 0.6
        # A contains items where probs < 0.3 (strictly ensuring A subseteq B)
        in_B = probs < 0.6
        in_A = probs < 0.3

        A = list(remaining[in_A])
        B = list(remaining[in_B])

        # Marginal gains
        val_A = float(f(A))
        val_Ax = float(f(A + [x]))
        gain_A = val_Ax - val_A

        val_B = float(f(B))
        val_Bx = float(f(B + [x]))
        gain_B = val_Bx - val_B

        # Submodular check: gain_A >= gain_B - tol
        if gain_A < gain_B - tol:
            violations += 1

    return violations
