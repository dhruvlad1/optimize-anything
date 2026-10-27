"""baselines.py - Baseline heuristics for Track E: DabbaDash (Mumbai CVRP).

Implements:
1. driver_order: Sequential customer routing in index order with capacity cutoffs.
2. nearest_neighbour: Greedy nearest-neighbour search constrained by vehicle capacity.
"""

from __future__ import annotations

from typing import Any

from trackE_mumbai_routing.routing_common import CVRPInstance


def driver_order(instance: CVRPInstance | dict[str, Any]) -> list[list[int]]:
    """Driver Order Baseline Heuristic.

    Simulates a naive delivery dispatch strategy: customers are visited in
    strictly ascending index order (1, 2, ..., n). A vehicle keeps adding
    the next customer until the addition would exceed capacity Q = 50.
    At that point, the vehicle returns to the depot (0), and a new vehicle
    takes the next customer.

    Args:
        instance: CVRPInstance with 'n', 'demands', 'capacity', 'depot_idx'.

    Returns:
        List of routes, where each route is a list of node indices starting
        and ending with 0 (e.g. [[0, 1, 2, 0], [0, 3, 0]]).
    """
    n = instance["n"] if isinstance(instance, dict) else instance.n
    demands = (
        instance["demands"] if isinstance(instance, dict) else instance.demands
    )
    capacity = (
        instance["capacity"]
        if isinstance(instance, dict)
        else instance.capacity
    )
    depot = (
        instance["depot_idx"]
        if isinstance(instance, dict)
        else instance.depot_idx
    )

    routes: list[list[int]] = []
    current_route: list[int] = [depot]
    current_load: int = 0

    for customer in range(1, n + 1):
        demand = int(demands[customer])
        # If adding customer exceeds vehicle capacity, close current route
        if current_load + demand > capacity:
            current_route.append(depot)
            routes.append(current_route)
            # Start new route with this customer
            current_route = [depot, customer]
            current_load = demand
        else:
            current_route.append(customer)
            current_load += demand

    # Close and record final route
    if len(current_route) > 1:
        current_route.append(depot)
        routes.append(current_route)

    return routes


def nearest_neighbour(
    instance: CVRPInstance | dict[str, Any],
) -> list[list[int]]:
    """Greedy Nearest-Neighbour Baseline Heuristic.

    Dispatches delivery vehicles one by one from depot 0. From the vehicle's
    current node, it selects the closest unvisited customer whose demand fits
    into the remaining vehicle capacity. When no remaining unvisited customer
    can fit, the vehicle returns to the depot, and a fresh vehicle is dispatched.

    Args:
        instance: CVRPInstance with 'n', 'demands', 'capacity', 'depot_idx', 'dist_matrix'.

    Returns:
        List of routes, where each route is a list of node indices starting
        and ending with 0.
    """
    n = instance["n"] if isinstance(instance, dict) else instance.n
    demands = (
        instance["demands"] if isinstance(instance, dict) else instance.demands
    )
    capacity = (
        instance["capacity"]
        if isinstance(instance, dict)
        else instance.capacity
    )
    depot = (
        instance["depot_idx"]
        if isinstance(instance, dict)
        else instance.depot_idx
    )
    dist = (
        instance["dist_matrix"]
        if isinstance(instance, dict)
        else instance.dist_matrix
    )
    assert dist is not None, "Distance matrix must be present in instance."

    unvisited: set[int] = set(range(1, n + 1))
    routes: list[list[int]] = []

    while unvisited:
        current_route: list[int] = [depot]
        current_node: int = depot
        current_load: int = 0

        while True:
            # Identify unvisited customers that fit into remaining vehicle capacity
            eligible = [
                c for c in unvisited if current_load + demands[c] <= capacity
            ]
            if not eligible:
                # No unvisited customer fits; return to depot and end tour
                break

            # Choose eligible customer closest to current_node; break ties by index
            next_node = min(eligible, key=lambda c: (dist[current_node, c], c))

            current_route.append(next_node)
            current_load += int(demands[next_node])
            current_node = next_node
            unvisited.remove(next_node)

        # Complete tour back to depot
        current_route.append(depot)
        routes.append(current_route)

    return routes
