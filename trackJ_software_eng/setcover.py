"""Small weighted set-cover solvers used by Track J."""
from __future__ import annotations

import time

import numpy as np


def universe_from_sets(sets):
    return set().union(*(set(s) for s in sets)) if sets else set()


def greedy_cover(sets, costs):
    """Weighted greedy: repeatedly maximize newly-covered elements / cost."""
    sets = [set(s) for s in sets]
    costs = np.asarray(costs, dtype=float)
    if len(sets) != len(costs) or np.any(costs <= 0) or not np.all(np.isfinite(costs)):
        raise ValueError("one positive finite cost is required per set")
    uncovered = universe_from_sets(sets)
    selected = []
    while uncovered:
        choices = [(len(s & uncovered) / costs[i], -i, i) for i, s in enumerate(sets) if i not in selected and s & uncovered]
        if not choices:
            raise ValueError(f"uncoverable elements: {sorted(uncovered, key=str)[:10]}")
        i = max(choices)[2]
        selected.append(i)
        uncovered -= sets[i]
    return selected, float(costs[selected].sum())


def ilp_cover(sets, costs, time_limit=300):
    """Solve weighted set cover with OR-Tools' SCIP backend."""
    from ortools.linear_solver import pywraplp

    sets = [set(s) for s in sets]
    costs = np.asarray(costs, dtype=float)
    universe = universe_from_sets(sets)
    if len(sets) != len(costs) or np.any(costs < 0):
        raise ValueError("one non-negative cost is required per set")
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP backend is unavailable")
    solver.SetTimeLimit(int(time_limit * 1000))
    x = [solver.BoolVar(f"x_{i}") for i in range(len(sets))]
    for e in universe:
        solver.Add(sum(x[i] for i, s in enumerate(sets) if e in s) >= 1)
    solver.Minimize(sum(float(costs[i]) * x[i] for i in range(len(sets))))
    start = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - start
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"SCIP set cover failed with status {status}")
    selected = [i for i, v in enumerate(x) if v.solution_value() > .5]
    return selected, float(costs[selected].sum()), status == pywraplp.Solver.OPTIMAL, elapsed


def scipy_cover(sets, costs):
    """SciPy MILP cross-check; this is never the primary optimizer."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    sets = [set(s) for s in sets]
    costs = np.asarray(costs, dtype=float)
    universe = sorted(universe_from_sets(sets), key=str)
    A = lil_matrix((len(universe), len(sets)), dtype=float)
    for r, e in enumerate(universe):
        for c, s in enumerate(sets):
            A[r, c] = float(e in s)
    result = milp(costs, integrality=np.ones(len(sets)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(A.tocsr(), np.ones(len(universe),), np.inf))
    if not result.success or result.x is None:
        raise RuntimeError(f"SciPy MILP set cover failed: {result.message}")
    selected = [i for i, value in enumerate(result.x) if value > .5]
    return selected, float(costs[selected].sum())


def greedy_fastest_cover(sets, costs):
    """Take tests from fastest upward until all coverage elements are covered."""
    sets = [set(s) for s in sets]
    order = np.argsort(np.asarray(costs, dtype=float), kind="stable")
    uncovered = universe_from_sets(sets)
    selected = []
    for i in order:
        if sets[int(i)] & uncovered:
            selected.append(int(i))
            uncovered -= sets[int(i)]
            if not uncovered:
                return selected, float(np.asarray(costs)[selected].sum())
    raise ValueError("uncoverable elements")


def summarize_subset(sets, costs, selected, universe=None):
    universe = universe_from_sets(sets) if universe is None else set(universe)
    covered = set().union(*(sets[i] for i in selected)) if selected else set()
    return {"tests": len(selected), "cost": float(np.asarray(costs)[selected].sum()),
            "coverage": len(covered & universe) / len(universe) if universe else 1.0,
            "covered": covered}


def mutation_score(full_killed, subset_killed):
    denominator = sum(bool(x) for x in full_killed)
    if not denominator:
        return 1.0, 0, 0
    killed = sum(bool(a) and bool(b) for a, b in zip(full_killed, subset_killed))
    return killed / denominator, killed, denominator
