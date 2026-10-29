"""solvers.py - Optimization solvers for Track E: DabbaDash (Mumbai CVRP).

Implements:
1. clarke_wright: Parallel Clarke-Wright savings heuristic with capacity checks.
2. local_search: Deterministic first-improvement intra-route 2-opt and inter-route
                 relocate / swap operators.
3. ilp_cvrp: Exact MTZ-formulation CVRP integer linear program via OR-Tools SCIP.
4. ortools_routing: Library reference using OR-Tools constraint-programming routing
                    solver with guided local search.

Every solver validates its output via routing_common.validate_solution and raises
an exception if invalid.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.linear_solver import pywraplp

from trackE_mumbai_routing.routing_common import (
    CVRPInstance,
    route_cost,
    solution_cost,
    validate_solution,
)


class ILPResult(tuple):
    """4-tuple (routes, cost, proved_optimal, status) with .best_bound attribute."""

    def __new__(cls, routes, cost, proved_optimal, status, best_bound=None):
        obj = super().__new__(cls, (routes, cost, proved_optimal, status))
        obj.routes = routes
        obj.cost = cost
        obj.proved_optimal = proved_optimal
        obj.status = status
        obj.best_bound = best_bound
        return obj


def _ensure_valid(instance: CVRPInstance | dict[str, Any], routes: list[list[int]], solver_name: str) -> None:
    """Validate routes and raise ValueError if invalid."""
    is_valid, errors = validate_solution(instance, routes)
    if not is_valid:
        raise ValueError(f"{solver_name} returned an invalid CVRP solution: {errors}")


# ---------------------------------------------------------------------------
# 1. Clarke-Wright Parallel Savings Algorithm
# ---------------------------------------------------------------------------

def clarke_wright(instance: CVRPInstance | dict[str, Any]) -> list[list[int]]:
    """Parallel Clarke-Wright Savings Algorithm for CVRP.

    Savings for customer pair (i, j):
        s_ij = d(0, i) + d(0, j) - d(i, j)

    Iteratively merges routes at their ends when the combined customer demand
    does not exceed vehicle capacity Q.

    Args:
        instance: CVRPInstance with 'n', 'demands', 'capacity', 'dist_matrix', 'depot_idx'.

    Returns:
        List of routes, each starting and ending at depot 0.
    """
    n = instance["n"] if isinstance(instance, dict) else instance.n
    demands = instance["demands"] if isinstance(instance, dict) else instance.demands
    capacity = instance["capacity"] if isinstance(instance, dict) else instance.capacity
    dist = instance["dist_matrix"] if isinstance(instance, dict) else instance.dist_matrix
    depot = instance["depot_idx"] if isinstance(instance, dict) else instance.depot_idx

    # Start with n individual routes: [depot, i, depot]
    # Keep track of routes and a mapping from customer to route index
    routes: list[list[int]] = [[depot, i, depot] for i in range(1, n + 1)]
    route_demand: list[int] = [int(demands[i]) for i in range(1, n + 1)]
    cust_to_route: dict[int, int] = {i: i - 1 for i in range(1, n + 1)}

    # Calculate savings s_ij for 1 <= i < j <= n
    savings: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist[depot, i] + dist[depot, j] - dist[i, j]
            savings.append((s, i, j))

    # Sort descending by savings; break ties deterministically by customer indices
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    for s_val, i, j in savings:
        r_i = cust_to_route[i]
        r_j = cust_to_route[j]

        # Cannot merge a route with itself
        if r_i == r_j:
            continue

        route_i = routes[r_i]
        route_j = routes[r_j]

        # Check capacity
        if route_demand[r_i] + route_demand[r_j] > capacity:
            continue

        # Check if i and j are at the ends of their respective routes (adjacent to depot)
        i_is_start = (route_i[1] == i)
        i_is_end = (route_i[-2] == i)
        j_is_start = (route_j[1] == j)
        j_is_end = (route_j[-2] == j)

        if not (i_is_start or i_is_end) or not (j_is_start or j_is_end):
            continue

        # Merge routes depending on which ends meet
        merged_customers: list[int]
        custs_i = route_i[1:-1]
        custs_j = route_j[1:-1]

        if i_is_end and j_is_start:
            merged_customers = custs_i + custs_j
        elif i_is_start and j_is_end:
            merged_customers = custs_j + custs_i
        elif i_is_end and j_is_end:
            merged_customers = custs_i + custs_j[::-1]
        elif i_is_start and j_is_start:
            merged_customers = custs_i[::-1] + custs_j
        else:
            continue

        new_route = [depot] + merged_customers + [depot]
        new_demand = route_demand[r_i] + route_demand[r_j]

        # Update routes: put merged in r_i, empty r_j
        routes[r_i] = new_route
        route_demand[r_i] = new_demand
        routes[r_j] = []
        route_demand[r_j] = 0

        # Update customer route mappings
        for c in merged_customers:
            cust_to_route[c] = r_i

    # Filter out empty routes
    final_routes = [r for r in routes if len(r) > 2]

    _ensure_valid(instance, final_routes, "clarke_wright")
    return final_routes


# ---------------------------------------------------------------------------
# 2. Local Search (Intra 2-opt, Inter Relocate, Inter Swap)
# ---------------------------------------------------------------------------

def local_search(
    instance: CVRPInstance | dict[str, Any],
    routes: list[list[int]],
) -> tuple[list[list[int]], int]:
    """Improve CVRP routes using deterministic first-improvement local search.

    Neighborhoods applied in order:
      1. Intra-route 2-opt: reverses subsegments within each route.
      2. Inter-route relocate: moves one customer from one route into another.
      3. Inter-route swap: swaps one customer from route A with one from route B.

    Accepts only moves that strictly reduce total distance and maintain capacity.
    Repeats until no improvement can be found.

    Args:
        instance: CVRPInstance with 'demands', 'capacity', 'dist_matrix', 'depot_idx'.
        routes: Initial valid routes.

    Returns:
        (improved_routes, num_improving_moves_made)
    """
    demands = instance["demands"] if isinstance(instance, dict) else instance.demands
    capacity = instance["capacity"] if isinstance(instance, dict) else instance.capacity
    dist = instance["dist_matrix"] if isinstance(instance, dict) else instance.dist_matrix
    depot = instance["depot_idx"] if isinstance(instance, dict) else instance.depot_idx

    current_routes = [list(r) for r in routes if len(r) > 2]
    num_moves = 0
    EPSILON = 1e-9

    improved = True
    while improved:
        improved = False

        # --- 1. Intra-route 2-opt ---
        for r_idx in range(len(current_routes)):
            r = current_routes[r_idx]
            n_nodes = len(r)
            if n_nodes < 5:  # Need at least [0, c1, c2, 0] to have meaningful 2-opt
                continue

            move_found = False
            for i in range(1, n_nodes - 2):
                for j in range(i + 1, n_nodes - 1):
                    # Segment [i .. j] to be reversed
                    # Change in cost:
                    # removes edges (r[i-1], r[i]) and (r[j], r[j+1])
                    # adds edges (r[i-1], r[j]) and (r[i], r[j+1])
                    old_cost = dist[r[i - 1], r[i]] + dist[r[j], r[j + 1]]
                    new_cost = dist[r[i - 1], r[j]] + dist[r[i], r[j + 1]]
                    if new_cost < old_cost - EPSILON:
                        current_routes[r_idx] = r[:i] + r[i : j + 1][::-1] + r[j + 1 :]
                        num_moves += 1
                        improved = True
                        move_found = True
                        break
                if move_found:
                    break
            if improved:
                break
        if improved:
            continue

        # --- 2. Inter-route Relocate ---
        for r_a_idx in range(len(current_routes)):
            r_a = current_routes[r_a_idx]
            for r_b_idx in range(len(current_routes)):
                if r_a_idx == r_b_idx:
                    continue
                r_b = current_routes[r_b_idx]

                b_load = sum(demands[c] for c in r_b[1:-1])

                move_found = False
                for i in range(1, len(r_a) - 1):
                    u = r_a[i]
                    u_demand = demands[u]

                    if b_load + u_demand > capacity:
                        continue

                    # Cost delta for removing u from r_a
                    cost_a_remove = (
                        dist[r_a[i - 1], r_a[i + 1]]
                        - dist[r_a[i - 1], u]
                        - dist[u, r_a[i + 1]]
                    )

                    # Try inserting u into every position j in r_b (1 <= j <= len(r_b) - 1)
                    for j in range(1, len(r_b)):
                        cost_b_insert = (
                            dist[r_b[j - 1], u]
                            + dist[u, r_b[j]]
                            - dist[r_b[j - 1], r_b[j]]
                        )
                        delta = cost_a_remove + cost_b_insert

                        if delta < -EPSILON:
                            # Apply relocate
                            new_r_a = r_a[:i] + r_a[i + 1 :]
                            new_r_b = r_b[:j] + [u] + r_b[j:]
                            current_routes[r_a_idx] = new_r_a
                            current_routes[r_b_idx] = new_r_b
                            # Remove r_a if it has no customers left
                            current_routes = [r for r in current_routes if len(r) > 2]
                            num_moves += 1
                            improved = True
                            move_found = True
                            break
                    if move_found:
                        break
                if move_found:
                    break
            if improved:
                break
        if improved:
            continue

        # --- 3. Inter-route Swap ---
        for r_a_idx in range(len(current_routes)):
            r_a = current_routes[r_a_idx]
            a_load = sum(demands[c] for c in r_a[1:-1])

            for r_b_idx in range(r_a_idx + 1, len(current_routes)):
                r_b = current_routes[r_b_idx]
                b_load = sum(demands[c] for c in r_b[1:-1])

                move_found = False
                for i in range(1, len(r_a) - 1):
                    u = r_a[i]
                    u_dem = demands[u]

                    for j in range(1, len(r_b) - 1):
                        v = r_b[j]
                        v_dem = demands[v]

                        # Check capacity after swap
                        if a_load - u_dem + v_dem > capacity:
                            continue
                        if b_load - v_dem + u_dem > capacity:
                            continue

                        # Delta cost
                        cost_a_old = dist[r_a[i - 1], u] + dist[u, r_a[i + 1]]
                        cost_a_new = dist[r_a[i - 1], v] + dist[v, r_a[i + 1]]

                        cost_b_old = dist[r_b[j - 1], v] + dist[v, r_b[j + 1]]
                        cost_b_new = dist[r_b[j - 1], u] + dist[u, r_b[j + 1]]

                        delta = (cost_a_new - cost_a_old) + (cost_b_new - cost_b_old)

                        if delta < -EPSILON:
                            r_a[i] = v
                            r_b[j] = u
                            num_moves += 1
                            improved = True
                            move_found = True
                            break
                    if move_found:
                        break
                if move_found:
                    break
            if improved:
                break

    _ensure_valid(instance, current_routes, "local_search")
    return current_routes, num_moves


# ---------------------------------------------------------------------------
# 3. Exact MTZ CVRP via OR-Tools SCIP
# ---------------------------------------------------------------------------

def ilp_cvrp(
    instance: CVRPInstance | dict[str, Any],
    time_limit_s: float = 60.0,
) -> tuple[list[list[int]], float, bool, str]:
    """Exact Miller-Tucker-Zemlin (MTZ) formulation for CVRP using OR-Tools SCIP.

    Formulation:
      Variables:
        x_{ij} in {0, 1}: 1 if a vehicle travels from node i to node j (i != j)
        u_i in [d_i, Q]: cumulative demand delivered upon visiting customer i (1 <= i <= n)
      Constraints:
        sum_{j != i} x_{ij} = 1  for all customers i in {1..n}
        sum_{i != j} x_{ij} = 1  for all customers j in {1..n}
        sum_{j=1}^n x_{0j} <= K_max  (upper bound from Clarke-Wright)
        sum_{i=1}^n x_{i0} = sum_{j=1}^n x_{0j}
        u_j >= u_i + d_j - Q * (1 - x_{ij})  for all customers i != j
      Objective:
        min sum_{i, j} dist[i, j] * x_{ij}

    Returns:
        (routes, objective_value, proved_optimal, status_str)
    """
    n = instance["n"] if isinstance(instance, dict) else instance.n
    demands = instance["demands"] if isinstance(instance, dict) else instance.demands
    capacity = instance["capacity"] if isinstance(instance, dict) else instance.capacity
    dist = instance["dist_matrix"] if isinstance(instance, dict) else instance.dist_matrix
    depot = instance["depot_idx"] if isinstance(instance, dict) else instance.depot_idx

    # Vehicle upper bound from Clarke-Wright
    cw_routes = clarke_wright(instance)
    k_max = len(cw_routes)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver could not be initialized")

    solver.SetTimeLimit(int(time_limit_s * 1000))

    # Arc variables x_{ij} for i != j in 0..n
    x: dict[tuple[int, int], pywraplp.Variable] = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                x[(i, j)] = solver.BoolVar(f"x_{i}_{j}")

    # MTZ load variables u_i for i in 1..n: d_i <= u_i <= Q
    u: dict[int, pywraplp.Variable] = {}
    for i in range(1, n + 1):
        u[i] = solver.NumVar(float(demands[i]), float(capacity), f"u_{i}")

    # Degree constraints for each customer i in 1..n
    for i in range(1, n + 1):
        solver.Add(solver.Sum(x[(i, j)] for j in range(n + 1) if j != i) == 1)
        solver.Add(solver.Sum(x[(j, i)] for j in range(n + 1) if j != i) == 1)

    # Depot outflow <= K_max
    solver.Add(solver.Sum(x[(0, j)] for j in range(1, n + 1)) <= k_max)

    # Flow conservation at depot: outflow == inflow
    solver.Add(
        solver.Sum(x[(0, j)] for j in range(1, n + 1))
        == solver.Sum(x[(i, 0)] for i in range(1, n + 1))
    )

    # MTZ subtour elimination and capacity constraints
    # u_j >= u_i + d_j - Q * (1 - x_{ij})
    # <=> u_i - u_j + Q * x_{ij} <= Q - d_j
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                solver.Add(
                    u[i] - u[j] + float(capacity) * x[(i, j)]
                    <= float(capacity) - float(demands[j])
                )

    # Objective
    objective = solver.Objective()
    for (i, j), var in x.items():
        objective.SetCoefficient(var, float(dist[i, j]))
    objective.SetMinimization()

    status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status_str = status_map.get(status, f"STATUS_{status}")
    proved_optimal = (status == pywraplp.Solver.OPTIMAL)

    best_bound = float(solver.Objective().BestBound()) if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else 0.0

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        # Return empty / fallback if no feasible solution within time limit
        return ILPResult([], float("inf"), False, status_str, best_bound)

    # Reconstruct routes by following active arcs from depot
    routes: list[list[int]] = []
    # Find all first steps from depot: x[(0, j)] == 1
    depot_departures = [j for j in range(1, n + 1) if x[(0, j)].solution_value() > 0.5]

    for first_node in sorted(depot_departures):
        route = [0, first_node]
        curr = first_node
        while curr != 0:
            next_node = None
            for j in range(n + 1):
                if j != curr and x[(curr, j)].solution_value() > 0.5:
                    next_node = j
                    break
            if next_node is None:
                raise RuntimeError(f"ILP solution has a broken route at node {curr}")
            route.append(next_node)
            curr = next_node
        routes.append(route)

    _ensure_valid(instance, routes, "ilp_cvrp")
    true_cost = solution_cost(routes, dist)
    return ILPResult(routes, true_cost, proved_optimal, status_str, best_bound)


# ---------------------------------------------------------------------------
# 4. OR-Tools Routing Library Reference
# ---------------------------------------------------------------------------

def ortools_routing(
    instance: CVRPInstance | dict[str, Any],
    time_limit_s: float = 10.0,
) -> tuple[list[list[int]], float]:
    """Library reference CVRP solver using Google OR-Tools constraint programming.

    Uses Guided Local Search with a vehicle capacity dimension. Distances are
    scaled by 1000 and rounded to integers for the constraint solver, and the
    true floating-point distance is recomputed from the unscaled distance matrix.

    Args:
        instance: CVRPInstance with 'n', 'demands', 'capacity', 'dist_matrix', 'depot_idx'.
        time_limit_s: Maximum search time in seconds.

    Returns:
        (routes, true_distance_km)
    """
    n = instance["n"] if isinstance(instance, dict) else instance.n
    demands = instance["demands"] if isinstance(instance, dict) else instance.demands
    capacity = instance["capacity"] if isinstance(instance, dict) else instance.capacity
    dist = instance["dist_matrix"] if isinstance(instance, dict) else instance.dist_matrix
    depot = instance["depot_idx"] if isinstance(instance, dict) else instance.depot_idx

    num_nodes = n + 1
    # Allow up to n vehicles (guaranteeing feasible capacity)
    num_vehicles = n

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback: integer-scaled distances (multiply by 1000)
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(dist[from_node, to_node] * 1000.0))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Capacity callback & dimension
    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [int(capacity)] * num_vehicles,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Search parameters: Guided Local Search
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("OR-Tools routing solver failed to find a feasible solution.")

    # Extract routes
    routes: list[list[int]] = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

        # Only retain routes that visit at least one customer
        if len(route) > 2:
            routes.append(route)

    _ensure_valid(instance, routes, "ortools_routing")
    true_distance = solution_cost(routes, dist)
    return routes, true_distance
