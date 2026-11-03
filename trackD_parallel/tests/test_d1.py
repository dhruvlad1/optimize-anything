"""trackD_parallel/tests/test_d1.py
---------------------------------
Unit and regression tests for Track D part 1:
1. greedy on a hand-built coverage instance with a hand-computed answer
2. lazy greedy equals greedy on 10 random instances
3. ILP equals brute force on 5 random n = 10 instances
4. check_submodular finds 0 violations for facility location and > 0 for the non-submodular objective
5. the Amdahl formula at s = 0.1, p = 4 gives 1 / (0.1 + 0.9 / 4)
6. the Karp-Flatt formula recovers s from a synthetic Amdahl speedup
7. LPT never has a larger makespan than static equal split on a hand-built example
8. the work-stealing simulator conserves total work and never beats the lower bound max(sum / workers, longest task)
9. the sharded gradient sum equals the full gradient to 1e-10
"""

from __future__ import annotations

import numpy as np
import pytest

from trackD_parallel.parallel_ops import (
    amdahl_speedup,
    karp_flatt_serial_fraction,
    lpt_schedule,
    simulate_static_schedule,
    simulate_work_stealing,
)
from trackD_parallel.submodular import (
    brute_force,
    check_submodular,
    facility_location,
    greedy,
    ilp_facility_location,
    lazy_greedy,
    make_non_submodular_failure_instance,
)


def test_greedy_hand_built_instance():
    """1. Test greedy on a hand-built facility location instance with hand-computed answer."""
    # 3 customers, 3 facilities
    # Customer 0 values: f0=11, f1=0,  f2=5
    # Customer 1 values: f0=0,  f1=10, f2=5
    # Customer 2 values: f0=0,  f1=0,  f2=8
    W = np.array([
        [11.0, 0.0, 5.0],
        [0.0, 10.0, 5.0],
        [0.0, 0.0, 8.0],
    ], dtype=np.float64)
    f = facility_location(W)

    # Step 1:
    # f({0}) = 11 + 0 + 0 = 11
    # f({1}) = 0 + 10 + 0 = 10
    # f({2}) = 5 + 5 + 8 = 18
    # Greedy picks {2} with value 18
    sol1, val1, _ = greedy(f, [0, 1, 2], k=1)
    assert sol1 == [2]
    assert abs(val1 - 18.0) < 1e-12

    # Step 2:
    # Adding 0: max([11,5], [0,5], [0,8]) -> 11 + 5 + 8 = 24 (gain = 6)
    # Adding 1: max([0,5], [10,5], [0,8]) -> 5 + 10 + 8 = 23 (gain = 5)
    # Greedy picks 0 as second element -> total set [2, 0] with value 24
    sol2, val2, _ = greedy(f, [0, 1, 2], k=2)
    assert sol2 == [2, 0]
    assert abs(val2 - 24.0) < 1e-12


def test_lazy_greedy_equals_greedy_10_instances():
    """2. Test that lazy greedy produces identical selected sets and values as greedy across 10 random instances."""
    rng = np.random.default_rng(1234)
    for i in range(10):
        m, n = 15, 25
        k = 4
        W = rng.uniform(0.1, 5.0, (m, n))
        f = facility_location(W)
        ground_set = list(range(n))

        g_set, g_val, _ = greedy(f, ground_set, k)
        lg_set, lg_val, _ = lazy_greedy(f, ground_set, k)

        assert g_set == lg_set, f"Instance {i}: Greedy {g_set} != Lazy {lg_set}"
        assert abs(g_val - lg_val) < 1e-9, f"Instance {i}: value mismatch {g_val} vs {lg_val}"
        assert len(g_set) == k
        assert len(lg_set) == k


def test_ilp_equals_brute_force_5_instances():
    """3. Test that OR-Tools SCIP ILP and Brute Force agree exactly on 5 random n=10 instances."""
    rng = np.random.default_rng(5678)
    for i in range(5):
        m, n = 10, 10
        k = 3
        W = rng.uniform(0.1, 10.0, (m, n))
        f = facility_location(W)
        ground_set = list(range(n))

        bf_set, bf_val, _ = brute_force(f, ground_set, k)
        ilp_set, ilp_val = ilp_facility_location(W, k)

        assert abs(bf_val - ilp_val) < 1e-5, f"Instance {i}: BF {bf_val} != ILP {ilp_val}"
        assert len(bf_set) == k
        assert len(ilp_set) == k


def test_check_submodular_violations():
    """4. check_submodular finds 0 violations for facility location and > 0 for non-submodular objective."""
    rng = np.random.default_rng(42)
    W = rng.uniform(0.1, 5.0, (12, 12))
    f = facility_location(W)
    v_fl = check_submodular(f, list(range(12)), trials=500, seed=42)
    assert v_fl == 0, f"Facility location unexpectedly had {v_fl} submodularity violations"

    non_sub_obj, ground_set, _ = make_non_submodular_failure_instance()
    v_ns = check_submodular(non_sub_obj, ground_set, trials=500, seed=42)
    assert v_ns > 0, "Non-submodular objective unexpectedly had 0 violations"


def test_amdahl_formula():
    """5. The Amdahl formula at s = 0.1, p = 4 gives 1 / (0.1 + 0.9 / 4)."""
    s = 0.1
    p = 4
    val = amdahl_speedup(s, p)
    expected = 1.0 / (0.1 + 0.9 / 4.0)
    assert abs(val - expected) < 1e-14


def test_karp_flatt_recovers_serial_fraction():
    """6. The Karp-Flatt formula recovers s from a synthetic Amdahl speedup."""
    for s_true in [0.05, 0.15, 0.30]:
        for p in [2, 4, 8, 16]:
            speedup = amdahl_speedup(s_true, p)
            s_rec = karp_flatt_serial_fraction(p, speedup)
            assert abs(s_rec - s_true) < 1e-12, f"Failed at s={s_true}, p={p}: recovered {s_rec}"


def test_lpt_never_larger_makespan_than_static():
    """7. LPT never has a larger makespan than static equal split on a hand-built example."""
    # Hand-built skewed task distribution: heavy tasks concentrated at the beginning
    durations = [10.0, 9.0, 8.0, 1.0, 1.0, 1.0]
    p = 2

    # Static: worker 0 gets [10, 9, 8] (sum 27), worker 1 gets [1, 1, 1] (sum 3) -> makespan = 27
    static_m = simulate_static_schedule(durations, p)
    assert static_m == 27.0

    # LPT: sorts [10, 9, 8, 1, 1, 1] and greedily balances
    # w0: 10, w1: 9; then w1 gets 8 (17); w0 gets 1 (11); w0 gets 1 (12); w0 gets 1 (13)
    # makespan = max(13, 17) = 17 <= 27
    lpt_parts = lpt_schedule(durations, p)
    lpt_m = max(sum(durations[i] for i in part) for part in lpt_parts)
    assert lpt_m <= static_m
    assert lpt_m == 17.0


def test_work_stealing_simulation_properties():
    """8. The work-stealing simulator conserves total work and never beats the lower bound."""
    rng = np.random.default_rng(777)
    durations = list(rng.uniform(0.1, 5.0, 40))

    for p in [2, 4, 8]:
        ws = simulate_work_stealing(durations, p)
        total_orig = sum(durations)
        # Work conservation
        assert abs(ws["total_work_executed"] - total_orig) < 1e-9
        # Lower bound: max(sum / p, max_task)
        assert ws["simulated_makespan"] >= ws["theoretical_lower_bound"] - 1e-9


def test_sharded_gradient_sum_equals_full_gradient():
    """9. The sharded gradient sum equals the full gradient to 1e-10."""
    rng = np.random.default_rng(999)
    n, d = 1000, 20
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, n).astype(float)
    w = rng.standard_normal(d)

    # Full sequential gradient
    z = X @ w
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    full_grad = (X.T @ (p_hat - y)) / n

    # Sharded gradient over 4 shards
    p_procs = 4
    sharded_grads = []
    shard_size = n // p_procs
    for i in range(p_procs):
        X_shard = X[i * shard_size : (i + 1) * shard_size]
        y_shard = y[i * shard_size : (i + 1) * shard_size]
        z_s = X_shard @ w
        p_s = 1.0 / (1.0 + np.exp(-np.clip(z_s, -30.0, 30.0)))
        g_s = X_shard.T @ (p_s - y_shard)
        sharded_grads.append(g_s)

    summed_grad = sum(sharded_grads) / n
    max_diff = np.max(np.abs(full_grad - summed_grad))
    assert max_diff < 1e-10, f"Max diff {max_diff:.2e} >= 1e-10"
