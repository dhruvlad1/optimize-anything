"""trackH_dbms/index_solvers.py
----------------------------
Pure Python optimization solvers for index selection under a storage budget.
No database connection is required for any solver in this module.

Solvers implemented:
  - greedy_additive: Greedy selection by benefit per MB under an additive model
  - dp_additive: 0/1 Knapsack Dynamic Programming on the additive model with sizes
                 rounded UP to whole MB (guaranteeing budget feasibility)
  - ilp_assignment: Integer Linear Program with query-to-index assignment using
                    Google OR-Tools (SCIP backend), capturing substitution
  - ilp_assignment_scipy: Independent ILP cross-check using scipy.optimize.milp
  - brute_force: Exhaustive power-set evaluation (for n <= 16) under assignment model

Also provides:
  - model_cost: evaluates assignment-model workload cost for any index subset
  - SolverResult: namedtuple (selected, total_size, model_cost)
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple, Sequence

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import Bounds, LinearConstraint, milp
import scipy.sparse as sp


class SolverResult(NamedTuple):
    selected: set[str]
    total_size: float
    model_cost: float


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    candidates: Sequence[str],
    sizes: dict[str, float],
    budget: float,
    cost_matrix: dict[str, dict[str, float]],
    cost_none: dict[str, float],
    weights: dict[str, float],
) -> list[str]:
    """Validate all solver inputs; return canonical candidate list."""
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got {budget}")

    cand_list = list(candidates)
    if len(cand_list) != len(set(cand_list)):
        raise ValueError("Candidates list contains duplicates")

    for c in cand_list:
        if c not in sizes:
            raise ValueError(f"Candidate {c!r} missing from sizes dict")
        if sizes[c] < 0:
            raise ValueError(f"Size of {c!r} must be non-negative, got {sizes[c]}")

    queries = list(cost_none.keys())
    if not queries:
        raise ValueError("Queries / cost_none cannot be empty")

    for q in queries:
        if q not in weights:
            raise ValueError(f"Query {q!r} missing from weights dict")
        if weights[q] < 0:
            raise ValueError(f"Weight for query {q!r} must be non-negative, got {weights[q]}")
        if cost_none[q] < 0:
            raise ValueError(f"Cost none for query {q!r} must be non-negative, got {cost_none[q]}")
        if q not in cost_matrix:
            raise ValueError(f"Query {q!r} missing from cost_matrix")

    return cand_list


# ---------------------------------------------------------------------------
# Workload cost evaluation under assignment model
# ---------------------------------------------------------------------------

def model_cost(
    selected: set[str] | Sequence[str],
    cost_matrix: dict[str, dict[str, float]],
    weights: dict[str, float],
    cost_none: dict[str, float] | None = None,
) -> float:
    """Compute assignment-model workload cost of a selection of indexes.

    For each query q in the workload:
      effective_cost(q) = min(cost_none[q], min_{c in selected} cost_matrix[q][c])
    workload_cost = sum_q weights[q] * effective_cost(q)

    Each query is assigned to the single cheapest option (either the baseline plan
    or the best single active index from `selected`).
    """
    if cost_none is None:
        # Check if "none" key exists in cost_matrix
        first_q = next(iter(cost_matrix.keys()))
        if "none" not in cost_matrix[first_q]:
            raise ValueError("cost_none must be provided when 'none' is not a candidate in cost_matrix")
        cost_none = {q: cost_matrix[q]["none"] for q in cost_matrix}

    selected_set = set(selected)
    total_cost = 0.0

    for q, f_q in weights.items():
        eff = cost_none[q]
        q_costs = cost_matrix.get(q, {})
        for c in selected_set:
            if c in q_costs:
                eff = min(eff, q_costs[c])
        total_cost += f_q * eff

    return round(float(total_cost), 6)


# ---------------------------------------------------------------------------
# (a) Greedy by benefit per MB (additive model)
# ---------------------------------------------------------------------------

def greedy_additive(
    candidates: Sequence[str],
    sizes: dict[str, float],
    budget: float,
    cost_matrix: dict[str, dict[str, float]],
    cost_none: dict[str, float],
    weights: dict[str, float],
) -> SolverResult:
    """Select indexes greedily by benefit per MB using an additive model.

    benefit_c = sum_q f_q * (cost_q(none) - cost_q({c}))
    """
    cand_list = _validate_inputs(candidates, sizes, budget, cost_matrix, cost_none, weights)

    # Compute additive benefit for each candidate
    benefits: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for c in cand_list:
        b = 0.0
        for q, f_q in weights.items():
            cost_with = cost_matrix[q].get(c, cost_none[q])
            b += f_q * (cost_none[q] - cost_with)
        benefits[c] = b
        s = sizes[c]
        ratios[c] = (b / s) if (s > 0 and b > 0) else 0.0

    # Sort descending by ratio; break ties by candidate name for determinism
    sorted_cands = sorted(
        [c for c in cand_list if benefits[c] > 0 and sizes[c] <= budget],
        key=lambda c: (-ratios[c], c),
    )

    selected: set[str] = set()
    current_size = 0.0

    for c in sorted_cands:
        s = sizes[c]
        if current_size + s <= budget + 1e-9:
            selected.add(c)
            current_size += s

    # Storage constraint verification
    if current_size > budget + 1e-9:
        raise RuntimeError(f"Greedy storage constraint violated: {current_size} > {budget}")

    cost = model_cost(selected, cost_matrix, weights, cost_none)
    return SolverResult(selected=selected, total_size=round(current_size, 4), model_cost=cost)


# ---------------------------------------------------------------------------
# (b) Dynamic Programming knapsack (additive model)
# ---------------------------------------------------------------------------

def dp_additive(
    candidates: Sequence[str],
    sizes: dict[str, float],
    budget: float,
    cost_matrix: dict[str, dict[str, float]],
    cost_none: dict[str, float],
    weights: dict[str, float],
    size_unit: float | None = None,
) -> SolverResult:
    """0/1 Knapsack DP on additive benefits with sizes rounded UP to whole MB.

    Rounding scheme:
      w_c = ceil(sizes[c] / unit) >= 1
      W   = floor(budget / unit)
    This guarantees that sum_{c in S} sizes[c] <= budget is never violated.
    """
    cand_list = _validate_inputs(candidates, sizes, budget, cost_matrix, cost_none, weights)

    # Determine unit for rounding: if byte sizes (> 10000), use 1 MiB (1024*1024); else 1.0
    if size_unit is None:
        max_s = max((sizes[c] for c in cand_list), default=0.0)
        size_unit = (1024.0 * 1024.0) if (max_s >= 10_000.0 or budget >= 10_000.0) else 1.0

    # Compute additive benefits
    items: list[tuple[str, int, float, float]] = []  # (cand, rounded_w, benefit, original_size)
    for c in cand_list:
        b = 0.0
        for q, f_q in weights.items():
            cost_with = cost_matrix[q].get(c, cost_none[q])
            b += f_q * (cost_none[q] - cost_with)
        if b > 0:
            rounded_w = max(1, int(math.ceil(sizes[c] / size_unit)))
            items.append((c, rounded_w, b, sizes[c]))

    W = int(math.floor(budget / size_unit))
    if W <= 0 or not items:
        cost = model_cost(set(), cost_matrix, weights, cost_none)
        return SolverResult(selected=set(), total_size=0.0, model_cost=cost)

    # Sort items deterministically
    items.sort(key=lambda it: it[0])

    n_items = len(items)
    # 2D DP table for exact backtracking
    dp = [[0.0] * (W + 1) for _ in range(n_items + 1)]

    for i in range(1, n_items + 1):
        c, w, val, _ = items[i - 1]
        for cap in range(W + 1):
            if cap < w:
                dp[i][cap] = dp[i - 1][cap]
            else:
                v_without = dp[i - 1][cap]
                v_with = dp[i - 1][cap - w] + val
                dp[i][cap] = v_with if v_with > v_without else v_without

    # Backtrack
    selected: set[str] = set()
    cur_cap = W
    for i in range(n_items, 0, -1):
        if dp[i][cur_cap] != dp[i - 1][cur_cap]:
            c, w, _, _ = items[i - 1]
            selected.add(c)
            cur_cap -= w

    total_size = sum(sizes[c] for c in selected)
    if total_size > budget + 1e-9:
        raise RuntimeError(f"DP storage constraint violated: {total_size} > {budget}")

    cost = model_cost(selected, cost_matrix, weights, cost_none)
    return SolverResult(selected=selected, total_size=round(total_size, 4), model_cost=cost)


# ---------------------------------------------------------------------------
# (c) Integer Linear Programming (assignment model via OR-Tools SCIP)
# ---------------------------------------------------------------------------

def ilp_assignment(
    candidates: Sequence[str],
    sizes: dict[str, float],
    budget: float,
    cost_matrix: dict[str, dict[str, float]],
    cost_none: dict[str, float],
    weights: dict[str, float],
) -> SolverResult:
    """Solve the index selection problem using OR-Tools (SCIP backend).

    Formulation:
      Variables:
        x_c in {0, 1} for each candidate c
        y_{q,c} in {0, 1} for query q using index c
        y_{q,0} in {0, 1} for query q using no index
      Constraints:
        y_{q,0} + sum_c y_{q,c} == 1  for all q
        y_{q,c} <= x_c                for all q, c
        sum_c size_c * x_c <= B
      Objective:
        min sum_q f_q * [ cost_none[q]*y_{q,0} + sum_c cost_matrix[q][c]*y_{q,c} ]
    """
    cand_list = _validate_inputs(candidates, sizes, budget, cost_matrix, cost_none, weights)
    queries = list(cost_none.keys())

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver could not be initialized")

    # Variables
    x: dict[str, pywraplp.Variable] = {
        c: solver.BoolVar(f"x_{c}") for c in cand_list
    }
    y_none: dict[str, pywraplp.Variable] = {
        q: solver.BoolVar(f"y_{q}_0") for q in queries
    }
    y: dict[tuple[str, str], pywraplp.Variable] = {
        (q, c): solver.BoolVar(f"y_{q}_{c}") for q in queries for c in cand_list
    }

    # Constraint 1: Single assignment per query
    for q in queries:
        solver.Add(y_none[q] + solver.Sum(y[(q, c)] for c in cand_list) == 1)

    # Constraint 2: Linking query assignment to index selection
    for q in queries:
        for c in cand_list:
            solver.Add(y[(q, c)] <= x[c])

    # Constraint 3: Storage budget
    solver.Add(solver.Sum(sizes[c] * x[c] for c in cand_list) <= budget)

    # Objective
    objective = solver.Objective()
    for q in queries:
        f_q = weights[q]
        objective.SetCoefficient(y_none[q], f_q * cost_none[q])
        for c in cand_list:
            c_cost = cost_matrix[q].get(c, cost_none[q])
            objective.SetCoefficient(y[(q, c)], f_q * c_cost)
    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"OR-Tools SCIP failed to find a solution (status={status})")

    selected = {c for c in cand_list if x[c].solution_value() > 0.5}
    total_size = sum(sizes[c] for c in selected)

    if total_size > budget + 1e-9:
        raise RuntimeError(f"ILP storage constraint violated: {total_size} > {budget}")

    cost = model_cost(selected, cost_matrix, weights, cost_none)
    return SolverResult(selected=selected, total_size=round(total_size, 4), model_cost=cost)


# ---------------------------------------------------------------------------
# (d) ILP Assignment cross-check via scipy.optimize.milp
# ---------------------------------------------------------------------------

def ilp_assignment_scipy(
    candidates: Sequence[str],
    sizes: dict[str, float],
    budget: float,
    cost_matrix: dict[str, dict[str, float]],
    cost_none: dict[str, float],
    weights: dict[str, float],
) -> SolverResult:
    """Independent cross-check for the assignment ILP using scipy.optimize.milp."""
    cand_list = _validate_inputs(candidates, sizes, budget, cost_matrix, cost_none, weights)
    queries = list(cost_none.keys())

    n_c = len(cand_list)
    n_q = len(queries)

    # Variable layout in flat vector z of length n_c + n_q + n_q * n_c:
    #   [0 .. n_c - 1]                 : x_c
    #   [n_c .. n_c + n_q - 1]         : y_{q,0}
    #   [n_c + n_q .. n_c + n_q + n_q*n_c - 1] : y_{q,c} indexed by (qi * n_c + ci)
    total_vars = n_c + n_q + n_q * n_c

    c_obj = np.zeros(total_vars, dtype=float)
    # x_c has cost 0
    # y_{q,0}
    for qi, q in enumerate(queries):
        c_obj[n_c + qi] = weights[q] * cost_none[q]
    # y_{q,c}
    for qi, q in enumerate(queries):
        for ci, c in enumerate(cand_list):
            c_cost = cost_matrix[q].get(c, cost_none[q])
            idx = n_c + n_q + qi * n_c + ci
            c_obj[idx] = weights[q] * c_cost

    rows = []
    lhs_list = []
    rhs_list = []

    # 1. Storage budget: sum_c size_c * x_c <= B
    storage_row = np.zeros(total_vars, dtype=float)
    for ci, c in enumerate(cand_list):
        storage_row[ci] = sizes[c]
    rows.append(storage_row)
    lhs_list.append(-np.inf)
    rhs_list.append(budget)

    # 2. Assignment: y_{q,0} + sum_c y_{q,c} == 1  for each q
    for qi, q in enumerate(queries):
        row = np.zeros(total_vars, dtype=float)
        row[n_c + qi] = 1.0
        for ci in range(n_c):
            row[n_c + n_q + qi * n_c + ci] = 1.0
        rows.append(row)
        lhs_list.append(1.0)
        rhs_list.append(1.0)

    # 3. Linking: y_{q,c} - x_c <= 0  for all q, c
    for qi in range(n_q):
        for ci in range(n_c):
            row = np.zeros(total_vars, dtype=float)
            row[n_c + n_q + qi * n_c + ci] = 1.0
            row[ci] = -1.0
            rows.append(row)
            lhs_list.append(-np.inf)
            rhs_list.append(0.0)

    A = sp.csr_matrix(np.array(rows, dtype=float))
    constraints = LinearConstraint(A, lhs_list, rhs_list)
    integrality = np.ones(total_vars, dtype=int)
    bounds = Bounds(0.0, 1.0)

    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"scipy.optimize.milp failed: {res.status} ({res.message})")

    x_sol = res.x[:n_c]
    selected = {cand_list[i] for i in range(n_c) if x_sol[i] > 0.5}
    total_size = sum(sizes[c] for c in selected)

    if total_size > budget + 1e-9:
        raise RuntimeError(f"SciPy ILP storage constraint violated: {total_size} > {budget}")

    cost = model_cost(selected, cost_matrix, weights, cost_none)
    return SolverResult(selected=selected, total_size=round(total_size, 4), model_cost=cost)


# ---------------------------------------------------------------------------
# (e) Brute Force exact reference (for n <= 16)
# ---------------------------------------------------------------------------

def brute_force(
    candidates: Sequence[str],
    sizes: dict[str, float],
    budget: float,
    cost_matrix: dict[str, dict[str, float]],
    cost_none: dict[str, float],
    weights: dict[str, float],
) -> SolverResult:
    """Exact baseline solver enumerating all 2^n subsets under the assignment model."""
    cand_list = _validate_inputs(candidates, sizes, budget, cost_matrix, cost_none, weights)
    n = len(cand_list)
    if n > 16:
        raise ValueError(f"brute_force only supports up to 16 candidates (got {n})")

    best_selected: set[str] = set()
    best_size = 0.0
    best_cost = model_cost(set(), cost_matrix, weights, cost_none)

    # Enumerate all combinations of all sizes
    for k in range(n + 1):
        for combo in itertools.combinations(cand_list, k):
            tot_s = sum(sizes[c] for c in combo)
            if tot_s <= budget + 1e-9:
                s_set = set(combo)
                c_cost = model_cost(s_set, cost_matrix, weights, cost_none)
                if c_cost < best_cost - 1e-9:
                    best_cost = c_cost
                    best_size = tot_s
                    best_selected = s_set
                elif abs(c_cost - best_cost) <= 1e-9:
                    # Deterministic tie break: prefer smaller total size, then sorted tuple
                    if tot_s < best_size - 1e-9:
                        best_cost = c_cost
                        best_size = tot_s
                        best_selected = s_set

    return SolverResult(
        selected=best_selected,
        total_size=round(best_size, 4),
        model_cost=round(best_cost, 6),
    )
