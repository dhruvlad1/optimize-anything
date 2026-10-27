"""test_routing.py - Pytest test suite for Track E: DabbaDash (Mumbai CVRP)."""

import math
import numpy as np
import pytest

from trackE_mumbai_routing.baselines import driver_order, nearest_neighbour
from trackE_mumbai_routing.routing_common import (
    CIRCUITY,
    CVRPInstance,
    build_distance_matrix,
    generate_instance,
    haversine_km,
    route_cost,
    solution_cost,
    validate_solution,
)


def test_haversine_known_points() -> None:
    """Test haversine formula against known analytical and geographical distances."""
    # 1. Zero distance between identical coordinates
    assert haversine_km(19.0, 72.8, 19.0, 72.8) == pytest.approx(0.0, abs=1e-7)

    # 2. 1 degree of latitude along meridian: 2 * pi * 6371 / 360 ~ 111.1949 km
    expected_1deg_lat = (2.0 * math.pi * 6371.0) / 360.0
    computed_1deg_lat = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert computed_1deg_lat == pytest.approx(expected_1deg_lat, rel=1e-4)

    # 3. Andheri East (19.1136, 72.8697) to Bandra (19.0596, 72.8295) is ~7.3 km
    d_andheri_bandra = haversine_km(19.1136, 72.8697, 19.0596, 72.8295)
    assert 7.0 < d_andheri_bandra < 8.0


def test_distance_matrix_properties() -> None:
    """Test that distance matrix is symmetric with zero diagonal."""
    coords = np.array([
        [19.1136, 72.8697],
        [19.0596, 72.8295],
        [19.1176, 72.9060],
        [19.2307, 72.8567],
    ])
    dist = build_distance_matrix(coords)

    # Check shape
    assert dist.shape == (4, 4)

    # Check diagonal is strictly zero
    for i in range(4):
        assert dist[i, i] == 0.0

    # Check exact symmetry
    assert np.allclose(dist, dist.T, atol=1e-10)

    # Check positive off-diagonal distances
    for i in range(4):
        for j in range(4):
            if i != j:
                assert dist[i, j] > 0.0


def test_generate_instance_determinism() -> None:
    """Test that generate_instance is deterministic for same seed and varies across seeds."""
    inst_a = generate_instance(n=15, seed=123)
    inst_b = generate_instance(n=15, seed=123)
    inst_c = generate_instance(n=15, seed=456)

    # Identical for same seed
    assert np.array_equal(inst_a.coords, inst_b.coords)
    assert np.array_equal(inst_a.demands, inst_b.demands)
    assert np.allclose(inst_a.dist_matrix, inst_b.dist_matrix)

    # Different for different seeds
    assert not np.array_equal(inst_a.coords, inst_c.coords)
    assert not np.array_equal(inst_a.demands, inst_c.demands)

    # Validate bounds
    assert inst_a.coords[0, 0] == pytest.approx(19.1136)
    assert inst_a.coords[0, 1] == pytest.approx(72.8697)
    assert np.all(inst_a.demands[1:] >= 1)
    assert np.all(inst_a.demands[1:] <= 9)
    assert inst_a.demands[0] == 0


def test_generate_instance_validation() -> None:
    """Test input validation for generate_instance."""
    with pytest.raises(ValueError, match="n must be an integer >= 1"):
        generate_instance(n=0, seed=1)

    with pytest.raises(ValueError, match="n must be an integer >= 1"):
        generate_instance(n=-5, seed=1)

    with pytest.raises(TypeError, match="Seed must be an integer"):
        generate_instance(n=10, seed="not_an_int")  # type: ignore


def test_validate_solution() -> None:
    """Test validate_solution accepting valid tours and rejecting specific errors."""
    mock_instance = {
        "n": 3,
        "demands": np.array([0, 15, 20, 25]),
        "capacity": 50,
        "depot_idx": 0,
    }

    # 1. Valid solution: two routes [0, 1, 2, 0] (demand 35 <= 50) and [0, 3, 0] (demand 25 <= 50)
    valid_routes = [[0, 1, 2, 0], [0, 3, 0]]
    is_valid, errors = validate_solution(mock_instance, valid_routes)
    assert is_valid is True
    assert len(errors) == 0

    # 2. Reject missing customer (customer 3 omitted)
    missing_routes = [[0, 1, 2, 0]]
    is_valid_m, errors_m = validate_solution(mock_instance, missing_routes)
    assert is_valid_m is False
    assert any("Missing" in err for err in errors_m)

    # 3. Reject duplicated customer (customer 2 in both routes)
    dup_routes = [[0, 1, 2, 0], [0, 2, 3, 0]]
    is_valid_d, errors_d = validate_solution(mock_instance, dup_routes)
    assert is_valid_d is False
    assert any("Duplicated" in err for err in errors_d)

    # 4. Reject capacity violation: [0, 1, 2, 3, 0] has demand 15 + 20 + 25 = 60 > 50
    cap_routes = [[0, 1, 2, 3, 0]]
    is_valid_c, errors_c = validate_solution(mock_instance, cap_routes)
    assert is_valid_c is False
    assert any("capacity" in err.lower() for err in errors_c)

    # 5. Reject route not starting/ending at depot
    bad_depot_routes = [[1, 2, 0], [0, 3, 0]]
    is_valid_b, errors_b = validate_solution(mock_instance, bad_depot_routes)
    assert is_valid_b is False


def test_baselines_validity() -> None:
    """Test that both baseline heuristics produce valid solutions for n=10 and n=50."""
    for n in [10, 50]:
        instance = generate_instance(n=n, seed=42)

        # Driver order
        routes_do = driver_order(instance)
        valid_do, errors_do = validate_solution(instance, routes_do)
        assert valid_do is True, f"driver_order produced invalid solution for n={n}: {errors_do}"

        # Nearest neighbour
        routes_nn = nearest_neighbour(instance)
        valid_nn, errors_nn = validate_solution(instance, routes_nn)
        assert valid_nn is True, f"nearest_neighbour produced invalid solution for n={n}: {errors_nn}"

        # Nearest neighbour should achieve significantly lower distance than driver order
        cost_do = solution_cost(routes_do, instance.dist_matrix)
        cost_nn = solution_cost(routes_nn, instance.dist_matrix)
        assert cost_nn < cost_do


def test_hand_checkable_example() -> None:
    """Verify route and solution cost against independently calculated manual numbers.

    Setup: 3 customers along the equator:
    Depot: (0.0, 0.0)
    Cust 1: (0.0, 0.01)
    Cust 2: (0.0, 0.02)
    Cust 3: (0.0, 0.03)

    Demands: [0, 30, 30, 20], Capacity = 50.
    Route 1: [0, 1, 0]        (load: 30 <= 50)
    Route 2: [0, 2, 3, 0]     (load: 30 + 20 = 50 <= 50)
    """
    coords = np.array([
        [0.0, 0.0],    # Node 0 (Depot)
        [0.0, 0.01],   # Node 1
        [0.0, 0.02],   # Node 2
        [0.0, 0.03],   # Node 3
    ])
    demands = np.array([0, 30, 30, 20])
    dist = build_distance_matrix(coords)

    inst = CVRPInstance(
        n=3,
        seed=1,
        coords=coords,
        demands=demands,
        capacity=50,
        depot_idx=0,
        dist_matrix=dist,
    )

    routes = [[0, 1, 0], [0, 2, 3, 0]]
    is_valid, errors = validate_solution(inst, routes)
    assert is_valid is True, f"Hand-checkable solution should be valid: {errors}"

    # Independent distance calculations:
    # d(i, j) = haversine_km * CIRCUITY
    d01 = haversine_km(0.0, 0.0, 0.0, 0.01) * CIRCUITY
    d02 = haversine_km(0.0, 0.0, 0.0, 0.02) * CIRCUITY
    d23 = haversine_km(0.0, 0.02, 0.0, 0.03) * CIRCUITY
    d30 = haversine_km(0.0, 0.03, 0.0, 0.0) * CIRCUITY

    expected_r1 = 2.0 * d01
    expected_r2 = d02 + d23 + d30
    expected_total = expected_r1 + expected_r2

    calc_r1 = route_cost(routes[0], dist)
    calc_r2 = route_cost(routes[1], dist)
    calc_total = solution_cost(routes, dist)

    assert calc_r1 == pytest.approx(expected_r1, rel=1e-9)
    assert calc_r2 == pytest.approx(expected_r2, rel=1e-9)
    assert calc_total == pytest.approx(expected_total, rel=1e-9)
