"""test_solvers.py - Unit tests for Track E CVRP optimization solvers.

Verifies:
1. All four solvers (Clarke-Wright, Local Search, ILP exact, OR-Tools routing)
   produce valid solutions on n = 10 and n = 50.
2. Clarke-Wright is deterministic across repeated runs.
3. Local search never increases the route distance and maintains solution validity.
4. On a hand-built two-cluster instance with capacity forcing two routes, Clarke-Wright
   returns the optimal two-route solution with the hand-computed analytical cost.
5. Exact ILP matches brute-force permutation search on a 5-customer instance.
6. OR-Tools routing solver returns a valid solution.
"""

from __future__ import annotations

import itertools
import math
import numpy as np
import pytest

from trackE_mumbai_routing.baselines import nearest_neighbour
from trackE_mumbai_routing.routing_common import (
    CVRPInstance,
    generate_instance,
    solution_cost,
    validate_solution,
)
from trackE_mumbai_routing.solvers import (
    clarke_wright,
    ilp_cvrp,
    local_search,
    ortools_routing,
)


@pytest.mark.parametrize("n", [10, 50])
def test_all_four_solvers_valid_on_n10_and_n50(n: int):
    """Clarke-Wright, Local Search, ILP, and OR-Tools produce valid solutions."""
    inst = generate_instance(n=n, seed=0)

    # 1. Clarke-Wright
    routes_cw = clarke_wright(inst)
    valid_cw, errs_cw = validate_solution(inst, routes_cw)
    assert valid_cw, f"CW failed validation on n={n}: {errs_cw}"

    # 2. Local Search (improving Clarke-Wright)
    routes_ls, _ = local_search(inst, routes_cw)
    valid_ls, errs_ls = validate_solution(inst, routes_ls)
    assert valid_ls, f"Local Search failed validation on n={n}: {errs_ls}"

    # 3. OR-Tools Routing
    routes_or, _ = ortools_routing(inst, time_limit_s=5.0)
    valid_or, errs_or = validate_solution(inst, routes_or)
    assert valid_or, f"OR-Tools failed validation on n={n}: {errs_or}"

    # 4. ILP CVRP (short time limit: optimal on n=10, feasible on n=50)
    routes_ilp, _, _, status = ilp_cvrp(inst, time_limit_s=10.0)
    assert status in ("OPTIMAL", "FEASIBLE"), f"ILP status unexpected: {status}"
    valid_ilp, errs_ilp = validate_solution(inst, routes_ilp)
    assert valid_ilp, f"ILP failed validation on n={n}: {errs_ilp}"


def test_clarke_wright_deterministic():
    """Clarke-Wright produces identical routes across multiple runs."""
    inst = generate_instance(n=25, seed=42)
    routes_1 = clarke_wright(inst)
    routes_2 = clarke_wright(inst)
    assert routes_1 == routes_2


def test_local_search_never_increases_distance_and_keeps_validity():
    """Local search strictly improves or preserves initial route distance."""
    inst = generate_instance(n=30, seed=7)
    # Start with Nearest Neighbour
    init_routes = nearest_neighbour(inst)
    init_cost = solution_cost(init_routes, inst.dist_matrix)

    opt_routes, moves = local_search(inst, init_routes)
    opt_cost = solution_cost(opt_routes, inst.dist_matrix)

    valid, errs = validate_solution(inst, opt_routes)
    assert valid, f"Local search produced invalid solution: {errs}"
    assert opt_cost <= init_cost + 1e-9
    assert moves >= 0


def test_clarke_wright_hand_built_two_clusters():
    """On two diametrically opposed clusters with capacity forcing two vehicles,

    Clarke-Wright finds the optimal two routes with exact analytical cost.

    Depot at (0, 0), Capacity Q = 20.
    Cluster 1: C1 at (10, 0), C2 at (10, 1), demands 10 each (total 20).
    Cluster 2: C3 at (-10, 0), C4 at (-10, 1), demands 10 each (total 20).
    Optimal routes: [0, 1, 2, 0] and [0, 3, 4, 0].
    Expected cost: 2 * (10 + 1 + sqrt(101)) = 22 + 2*sqrt(101) ~= 42.099751.
    """
    coords = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 1.0],
        [-10.0, 0.0],
        [-10.0, 1.0],
    ])
    demands = np.array([0, 10, 10, 10, 10], dtype=int)
    dist = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])

    inst = CVRPInstance(
        n=4,
        seed=1,
        coords=coords,
        demands=demands,
        capacity=20,
        depot_idx=0,
        dist_matrix=dist,
    )

    routes = clarke_wright(inst)
    assert len(routes) == 2
    # Verify customer groupings
    route_sets = [set(r[1:-1]) for r in routes]
    assert {1, 2} in route_sets
    assert {3, 4} in route_sets

    cost = solution_cost(routes, dist)
    expected_cost = 22.0 + 2.0 * math.sqrt(101.0)
    assert math.isclose(cost, expected_cost, abs_tol=1e-5)


def test_ilp_matches_brute_force_on_5_customers():
    """Exact ILP solution matches full brute-force permutation & partition search."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [3.0, 2.0],
        [1.0, 4.0],
        [4.0, 1.0],
    ], dtype=float)
    demands = np.array([0, 10, 15, 20, 12, 18], dtype=int)
    dist = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])

    inst = CVRPInstance(
        n=5,
        seed=42,
        coords=coords,
        demands=demands,
        capacity=50,
        depot_idx=0,
        dist_matrix=dist,
    )

    # Brute-force: enumerate all 5! permutations and all 2^4 split points
    min_bf_cost = float("inf")
    for perm in itertools.permutations(range(1, 6)):
        for cuts in range(1 << 4):
            routes = []
            cur = [0, perm[0]]
            for idx in range(4):
                if (cuts >> idx) & 1:
                    cur.append(0)
                    routes.append(cur)
                    cur = [0, perm[idx + 1]]
                else:
                    cur.append(perm[idx + 1])
            cur.append(0)
            routes.append(cur)
            if all(sum(demands[c] for c in r[1:-1]) <= 50 for r in routes):
                c = solution_cost(routes, dist)
                if c < min_bf_cost:
                    min_bf_cost = c

    routes_ilp, cost_ilp, proved_optimal, status = ilp_cvrp(inst, time_limit_s=10.0)
    assert proved_optimal is True
    assert status == "OPTIMAL"
    assert math.isclose(cost_ilp, min_bf_cost, abs_tol=1e-5)


def test_ortools_routing_returns_valid_solution():
    """OR-Tools guided local search solver returns a valid CVRP solution."""
    inst = generate_instance(n=15, seed=123)
    routes, cost = ortools_routing(inst, time_limit_s=3.0)
    valid, errs = validate_solution(inst, routes)
    assert valid, f"OR-Tools solution invalid: {errs}"
    assert len(routes) >= 1
    assert math.isclose(cost, solution_cost(routes, inst.dist_matrix), abs_tol=1e-5)
