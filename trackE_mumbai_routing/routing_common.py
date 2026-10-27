"""routing_common.py - Common utilities for Track E: DabbaDash (Mumbai CVRP).

Provides distance calculation, synthetic instance generation, route cost
evaluation, and independent solution validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

# Road circuity factor representing Mumbai urban street grid vs. Haversine distance
CIRCUITY: float = 1.3

# Central Distribution Depot in Andheri East
DEPOT_COORDS: tuple[float, float] = (19.1136, 72.8697)

# Cluster centers across Mumbai for synthetic customer generation
MUMBAI_CLUSTERS: list[tuple[float, float]] = [
    (19.1136, 72.8697),  # Andheri
    (19.0596, 72.8295),  # Bandra
    (19.1176, 72.9060),  # Powai
    (19.2307, 72.8567),  # Borivali
    (19.0178, 72.8478),  # Dadar
    (19.0860, 72.9090),  # Ghatkopar
]

# Coordinate bounding box for Greater Mumbai
LAT_MIN, LAT_MAX = 18.95, 19.25
LON_MIN, LON_MAX = 72.80, 72.98

# Vehicle capacity
DEFAULT_CAPACITY: int = 50


@dataclass
class CVRPInstance:
    """Represents a Capacitated Vehicle Routing Problem instance."""

    n: int
    seed: int
    coords: np.ndarray  # Shape (n + 1, 2): [lat, lon], index 0 is depot
    demands: np.ndarray  # Shape (n + 1,): demand of each node (0 for depot)
    capacity: int = DEFAULT_CAPACITY
    depot_idx: int = 0
    dist_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.dist_matrix is None:
            self.dist_matrix = build_distance_matrix(self.coords)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access for convenience."""
        return getattr(self, key)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in km.

    Uses spherical Earth radius R = 6371.0 km.
    """
    r_lat1 = math.radians(lat1)
    r_lon1 = math.radians(lon1)
    r_lat2 = math.radians(lat2)
    r_lon2 = math.radians(lon2)

    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
    return 6371.0 * c


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Build the symmetric distance matrix for all coordinate pairs.

    Applies haversine_km * CIRCUITY, enforcing zero on the diagonal.
    """
    num_nodes = len(coords)
    dist = np.zeros((num_nodes, num_nodes), dtype=float)

    for i in range(num_nodes):
        lat_i, lon_i = coords[i, 0], coords[i, 1]
        for j in range(i + 1, num_nodes):
            lat_j, lon_j = coords[j, 0], coords[j, 1]
            d = haversine_km(lat_i, lon_i, lat_j, lon_j) * CIRCUITY
            dist[i, j] = d
            dist[j, i] = d

    np.fill_diagonal(dist, 0.0)
    return dist


def generate_instance(n: int, seed: int) -> CVRPInstance:
    """Generate a synthetic CVRP instance in Mumbai.

    Args:
        n: Number of customer delivery points (n >= 1).
        seed: Random seed for reproducibility.

    Returns:
        CVRPInstance containing coordinates, demands, capacity, and distance matrix.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"Customer count n must be an integer >= 1, got {n}")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    rng = np.random.default_rng(seed)

    coords = np.zeros((n + 1, 2), dtype=float)
    # Depot at index 0 (Andheri East)
    coords[0] = DEPOT_COORDS

    # Sample clusters for n customers
    cluster_indices = rng.integers(0, len(MUMBAI_CLUSTERS), size=n)
    centers = np.array(MUMBAI_CLUSTERS, dtype=float)[cluster_indices]
    offsets = rng.normal(loc=0.0, scale=0.012, size=(n, 2))
    raw_coords = centers + offsets

    # Clip to bounding box
    raw_coords[:, 0] = np.clip(raw_coords[:, 0], LAT_MIN, LAT_MAX)
    raw_coords[:, 1] = np.clip(raw_coords[:, 1], LON_MIN, LON_MAX)
    coords[1:] = raw_coords

    # Demands: 0 for depot, integers 1-9 for customers
    demands = np.zeros(n + 1, dtype=int)
    demands[1:] = rng.integers(1, 10, size=n)

    dist_matrix = build_distance_matrix(coords)

    return CVRPInstance(
        n=int(n),
        seed=int(seed),
        coords=coords,
        demands=demands,
        capacity=DEFAULT_CAPACITY,
        depot_idx=0,
        dist_matrix=dist_matrix,
    )


def route_cost(route: list[int], dist: np.ndarray) -> float:
    """Compute total distance of a single route."""
    if len(route) < 2:
        return 0.0
    cost = 0.0
    for idx in range(len(route) - 1):
        cost += dist[route[idx], route[idx + 1]]
    return float(cost)


def solution_cost(routes: list[list[int]], dist: np.ndarray) -> float:
    """Compute total distance across all routes in a solution."""
    return float(sum(route_cost(route, dist) for route in routes))


def validate_solution(
    instance: CVRPInstance | dict[str, Any],
    routes: list[list[int]],
) -> tuple[bool, list[str]]:
    """Independently validate a CVRP solution.

    Checks:
    1. Every customer in {1, ..., n} appears exactly once across all routes.
    2. Every route starts and ends at the depot (index 0).
    3. No route contains consecutive duplicate depot visits or internal depot visits.
    4. Total demand of customers in any route does not exceed vehicle capacity Q.

    Args:
        instance: CVRPInstance or dictionary with 'n', 'demands', 'capacity', 'depot_idx'.
        routes: List of routes, each route being a list of node indices.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    n = instance["n"] if isinstance(instance, dict) else instance.n
    demands = (
        instance["demands"] if isinstance(instance, dict) else instance.demands
    )
    capacity = (
        instance["capacity"]
        if isinstance(instance, dict)
        else instance.capacity
    )
    depot_idx = (
        instance["depot_idx"]
        if isinstance(instance, dict)
        else instance.depot_idx
    )

    if not isinstance(routes, list):
        return False, ["Routes must be provided as a list of lists."]

    visited_customers: list[int] = []

    for r_idx, route in enumerate(routes):
        if not isinstance(route, list) or len(route) < 3:
            # A valid route must visit at least one customer: [0, customer, 0]
            errors.append(
                f"Route {r_idx} is malformed or empty (length < 3): {route}"
            )
            continue

        if route[0] != depot_idx:
            errors.append(
                f"Route {r_idx} does not start at depot ({depot_idx}): starts with {route[0]}"
            )
        if route[-1] != depot_idx:
            errors.append(
                f"Route {r_idx} does not end at depot ({depot_idx}): ends with {route[-1]}"
            )

        # Internal nodes must be customers, not depot
        customers_in_route = route[1:-1]
        for node in customers_in_route:
            if node == depot_idx:
                errors.append(
                    f"Route {r_idx} contains an internal visit to depot {depot_idx}: {route}"
                )
            elif node < 1 or node > n:
                errors.append(
                    f"Route {r_idx} contains invalid customer index {node} (valid range 1..{n})"
                )
            else:
                visited_customers.append(node)

        # Capacity check
        route_demand = sum(demands[node] for node in customers_in_route if 0 <= node <= n)
        if route_demand > capacity:
            errors.append(
                f"Route {r_idx} exceeds capacity: demand {route_demand} > {capacity}"
            )

    # Check that every customer appears exactly once
    customer_counts: dict[int, int] = {}
    for c in visited_customers:
        customer_counts[c] = customer_counts.get(c, 0) + 1

    missing = [c for c in range(1, n + 1) if c not in customer_counts]
    if missing:
        errors.append(
            f"Missing {len(missing)} customer(s): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    duplicated = [c for c, count in customer_counts.items() if count > 1]
    if duplicated:
        errors.append(
            f"Duplicated {len(duplicated)} customer(s): {duplicated[:10]}{'...' if len(duplicated) > 10 else ''}"
        )

    return len(errors) == 0, errors
