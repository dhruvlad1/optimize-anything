"""trackB_discrete/knapsack.py
----------------------------
0/1 Knapsack problem solvers with correctness and feasibility self-verification.

Solvers implemented:
  1. brute_force (n <= 20)
  2. greedy_ratio
  3. greedy_half_approx (1/2-approximation guarantee)
  4. dp_knapsack (exact dynamic programming, integer weights)
  5. ilp_knapsack (Google OR-Tools pywraplp with SCIP backend)
  6. milp_knapsack_scipy (scipy.optimize.milp independent cross-check)
  7. branch_and_bound (DFS with fractional Dantzig relaxation bound)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import Bounds, LinearConstraint, milp


def _validate_inputs(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate knapsack inputs and return numpy arrays."""
    v = np.asarray(values, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)

    if v.ndim != 1 or w.ndim != 1:
        raise ValueError("values and weights must be 1-dimensional.")
    if len(v) != len(w):
        raise ValueError(f"Length mismatch: {len(v)} values vs {len(w)} weights.")
    if len(v) == 0:
        raise ValueError("Item list must not be empty.")
    if np.any(v < 0):
        raise ValueError("All item values must be non-negative.")
    if np.any(w < 0):
        raise ValueError("All item weights must be non-negative.")
    if capacity < 0:
        raise ValueError(f"Knapsack capacity must be non-negative, got {capacity}.")

    return v, w, int(capacity)


def _verify_solution(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: int,
    reported_value: int,
    selected: List[int],
) -> None:
    """Verify that selected items are feasible and sum to reported_value."""
    if len(selected) > 0:
        sel_idx = np.asarray(selected, dtype=int)
        if np.any(sel_idx < 0) or np.any(sel_idx >= len(values)):
            raise ValueError(f"Selected item index out of range [0, {len(values)-1}].")
        total_w = int(np.sum(weights[sel_idx]))
        total_v = int(np.sum(values[sel_idx]))
    else:
        total_w = 0
        total_v = 0

    if total_w > capacity:
        raise ValueError(
            f"Infeasible selection: total weight {total_w} exceeds capacity {capacity}."
        )
    if total_v != reported_value:
        raise ValueError(
            f"Reported value {reported_value} does not match sum of selected items {total_v}."
        )


# ---------------------------------------------------------------------------
# 1. Brute Force (n <= 20)
# ---------------------------------------------------------------------------

def brute_force(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Solve 0/1 knapsack by exhaustive enumeration of all 2^n subsets."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)
    n = len(v)
    if n > 20:
        raise ValueError(f"brute_force is restricted to n <= 20, got n={n}.")

    best_val = 0
    best_mask = 0

    for mask in range(1 << n):
        cur_w = 0
        cur_v = 0
        for i in range(n):
            if (mask >> i) & 1:
                cur_w += w[i]
                if cur_w > W:
                    break
                cur_v += v[i]
        if cur_w <= W and cur_v > best_val:
            best_val = cur_v
            best_mask = mask

    selected = [i for i in range(n) if (best_mask >> i) & 1]
    runtime = time.perf_counter() - t0
    _verify_solution(v, w, W, best_val, selected)

    return best_val, selected, {"runtime": runtime, "subsets_evaluated": 1 << n}


# ---------------------------------------------------------------------------
# 2. Greedy by Value/Weight Ratio
# ---------------------------------------------------------------------------

def greedy_ratio(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Greedy heuristic: sort items by descending value/weight ratio."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)
    n = len(v)

    # Calculate ratios (safe division for 0 weight)
    ratios = np.where(w > 0, v / w, np.inf)
    # Sort indices by ratio descending, breaking ties by value descending
    sorted_idx = sorted(range(n), key=lambda i: (ratios[i], v[i]), reverse=True)

    rem_cap = W
    total_val = 0
    selected = []

    for i in sorted_idx:
        if w[i] <= rem_cap:
            rem_cap -= w[i]
            total_val += int(v[i])
            selected.append(i)

    selected.sort()
    runtime = time.perf_counter() - t0
    _verify_solution(v, w, W, total_val, selected)

    return total_val, selected, {"runtime": runtime}


# ---------------------------------------------------------------------------
# 3. Greedy 1/2-Approximation
# ---------------------------------------------------------------------------

def greedy_half_approx(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Returns the best of ratio-greedy and the single best feasible item (1/2-approx)."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)

    # 1. Ratio greedy solution
    val_ratio, sel_ratio, _ = greedy_ratio(v, w, W)

    # 2. Best single feasible item
    best_single_val = 0
    best_single_idx = -1
    for i in range(len(v)):
        if w[i] <= W and v[i] > best_single_val:
            best_single_val = int(v[i])
            best_single_idx = i

    runtime = time.perf_counter() - t0

    if best_single_idx >= 0 and best_single_val > val_ratio:
        selected = [best_single_idx]
        best_val = best_single_val
        winner = "single_item"
    else:
        selected = sel_ratio
        best_val = val_ratio
        winner = "ratio_greedy"

    _verify_solution(v, w, W, best_val, selected)
    return best_val, selected, {"runtime": runtime, "winner": winner}


# ---------------------------------------------------------------------------
# 4. Dynamic Programming (Integer Weights)
# ---------------------------------------------------------------------------

def dp_knapsack(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Exact pseudo-polynomial dynamic programming knapsack solver."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)
    n = len(v)

    # 1D DP value array and 2D boolean decision table for backtracking
    dp = np.zeros(W + 1, dtype=np.int64)
    take = np.zeros((n, W + 1), dtype=bool)

    for i in range(n):
        wi, vi = int(w[i]), int(v[i])
        if wi <= W:
            cand = dp[:-wi] + vi
            better = cand > dp[wi:]
            # Mark updates
            take[i, wi:][better] = True
            dp[wi:][better] = cand[better]

    best_val = int(dp[W])

    # Backtracking to recover selected items
    selected = []
    curr_w = W
    for i in range(n - 1, -1, -1):
        if take[i, curr_w]:
            selected.append(i)
            curr_w -= int(w[i])

    selected.reverse()
    runtime = time.perf_counter() - t0
    _verify_solution(v, w, W, best_val, selected)

    return best_val, selected, {"runtime": runtime}


# ---------------------------------------------------------------------------
# 5. Integer Linear Programming (OR-Tools SCIP)
# ---------------------------------------------------------------------------

def ilp_knapsack(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Exact ILP solver using Google OR-Tools with SCIP backend."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)
    n = len(v)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver backend is not available in ortools.")

    # Variables: x_i in {0, 1}
    x = [solver.BoolVar(f"x_{i}") for i in range(n)]

    # Capacity constraint: sum(w_i * x_i) <= W
    solver.Add(solver.Sum(int(w[i]) * x[i] for i in range(n)) <= W)

    # Objective: maximize sum(v_i * x_i)
    solver.Maximize(solver.Sum(int(v[i]) * x[i] for i in range(n)))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"ILP did not reach optimality, status code: {status}")

    best_val = int(round(solver.Objective().Value()))
    selected = [i for i in range(n) if x[i].solution_value() > 0.5]
    selected.sort()

    runtime = time.perf_counter() - t0
    _verify_solution(v, w, W, best_val, selected)

    return best_val, selected, {"runtime": runtime, "status": "OPTIMAL"}


# ---------------------------------------------------------------------------
# 6. SciPy MILP (Independent Cross-Check)
# ---------------------------------------------------------------------------

def milp_knapsack_scipy(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Exact MILP solver using scipy.optimize.milp for independent verification."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)
    n = len(v)

    # Maximize sum(v_i x_i) <=> Minimize -v^T x
    c = -v.astype(float)
    integrality = np.ones(n, dtype=int)
    bounds = Bounds(0.0, 1.0)
    constraints = LinearConstraint(w.reshape(1, -1), -np.inf, float(W))

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"scipy.optimize.milp failed: {res.status}")

    sol = np.round(res.x).astype(int)
    selected = [i for i in range(n) if sol[i] == 1]
    selected.sort()
    best_val = int(round(-res.fun))

    runtime = time.perf_counter() - t0
    _verify_solution(v, w, W, best_val, selected)

    return best_val, selected, {"runtime": runtime, "status": res.status}


# ---------------------------------------------------------------------------
# 7. Branch and Bound (DFS with Dantzig Fractional Bound)
# ---------------------------------------------------------------------------

def branch_and_bound(
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """Exact Branch and Bound solver using DFS and the Dantzig fractional relaxation bound."""
    t0 = time.perf_counter()
    v, w, W = _validate_inputs(values, weights, capacity)
    n = len(v)

    # Sort items by value/weight ratio descending
    ratios = np.where(w > 0, v / w, np.inf)
    order = sorted(range(n), key=lambda i: (ratios[i], v[i]), reverse=True)

    sorted_v = v[order]
    sorted_w = w[order]

    # Initialize incumbent with ratio-greedy
    init_val, init_sel, _ = greedy_half_approx(v, w, W)
    best_val = init_val
    best_selection = set(init_sel)

    nodes_explored = 0
    nodes_pruned = 0

    def dantzig_bound(level: int, cur_w: int, cur_v: int) -> float:
        """Compute the upper bound using fractional relaxation."""
        rem_cap = W - cur_w
        bound = float(cur_v)
        for i in range(level, n):
            wi = int(sorted_w[i])
            vi = int(sorted_v[i])
            if wi <= rem_cap:
                rem_cap -= wi
                bound += vi
            else:
                bound += vi * (rem_cap / wi)
                break
        return bound

    # Stack holds: (level, cur_w, cur_v, taken_mask)
    # We use iterative DFS to prevent recursion limit issues
    stack = [(0, 0, 0, [])]

    while stack:
        level, cur_w, cur_v, taken = stack.pop()
        nodes_explored += 1

        if level == n:
            if cur_v > best_val:
                best_val = cur_v
                best_selection = {order[i] for i in taken}
            continue

        # Check bound
        upper = dantzig_bound(level, cur_w, cur_v)
        if upper <= best_val:
            nodes_pruned += 1
            continue

        # Branching order: Try WITHOUT item first on stack (so WITH item is popped first)
        # 1. Branch: Do not take item at 'level'
        stack.append((level + 1, cur_w, cur_v, taken))

        # 2. Branch: Take item at 'level' (if feasible)
        wi = int(sorted_w[level])
        vi = int(sorted_v[level])
        if cur_w + wi <= W:
            stack.append((level + 1, cur_w + wi, cur_v + vi, taken + [level]))

    selected = sorted(best_selection)
    runtime = time.perf_counter() - t0
    _verify_solution(v, w, W, best_val, selected)

    return best_val, selected, {
        "runtime": runtime,
        "nodes_explored": nodes_explored,
        "nodes_pruned": nodes_pruned,
    }
