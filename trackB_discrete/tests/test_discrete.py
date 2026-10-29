"""Unit tests for Track B discrete optimization solvers, QUBO formulations, and Ising equivalence."""

import networkx as nx
import numpy as np
import pytest

from trackB_discrete.knapsack import (
    branch_and_bound,
    brute_force,
    dp_knapsack,
    greedy_half_approx,
    greedy_ratio,
    ilp_knapsack,
    milp_knapsack_scipy,
)
from trackB_discrete.qubo import (
    brute_force_qubo,
    cardinality_qubo,
    evaluate_qubo,
    ising_energy,
    maxcut_to_qubo,
    qubo_to_ising,
    simulated_annealing_qubo,
)


def test_exact_knapsack_solvers_agree_on_n12():
    """All exact knapsack solvers must return the exact same optimum on random n=12 instances."""
    rng = np.random.default_rng(123)
    for _ in range(5):
        w = rng.integers(1, 30, size=12)
        v = rng.integers(1, 40, size=12)
        W = int(0.5 * np.sum(w))

        val_bf, _, _ = brute_force(v, w, W)
        val_dp, _, _ = dp_knapsack(v, w, W)
        val_ilp, _, _ = ilp_knapsack(v, w, W)
        val_scipy, _, _ = milp_knapsack_scipy(v, w, W)
        val_bb, _, _ = branch_and_bound(v, w, W)

        assert val_bf == val_dp == val_ilp == val_scipy == val_bb, (
            f"Exact solvers disagreement: BF={val_bf}, DP={val_dp}, ILP={val_ilp}, SciPy={val_scipy}, B&B={val_bb}"
        )


def test_greedy_bounds_and_half_approx_guarantee():
    """Greedy never exceeds optimum and greedy_half_approx reaches at least 0.5 * OPT on 20 instances."""
    rng = np.random.default_rng(456)
    for _ in range(20):
        w = rng.integers(1, 50, size=15)
        v = rng.integers(1, 50, size=15)
        W = int(0.4 * np.sum(w))

        val_opt, _, _ = dp_knapsack(v, w, W)
        val_ratio, _, _ = greedy_ratio(v, w, W)
        val_half, _, _ = greedy_half_approx(v, w, W)

        # Greedy never exceeds optimal
        assert val_ratio <= val_opt
        assert val_half <= val_opt

        # 1/2 approximation guarantee
        assert val_half >= 0.5 * val_opt, f"Greedy half approx {val_half} < 0.5 * {val_opt}"


def test_hand_built_greedy_failure_case():
    """Hand-built failure case where ratio greedy fails and greedy_half_approx fixes it."""
    v = [2, 10]
    w = [1, 10]
    W = 10

    val_ratio, sel_ratio, _ = greedy_ratio(v, w, W)
    val_half, sel_half, info_half = greedy_half_approx(v, w, W)
    val_opt, sel_opt, _ = dp_knapsack(v, w, W)

    # Ratio greedy takes item 0 (v=2, w=1, ratio 2.0) and blocks item 1 (v=10, w=10)
    assert val_ratio == 2
    assert sel_ratio == [0]

    # greedy_half_approx takes item 1 (v=10)
    assert val_half == 10
    assert sel_half == [1]
    assert info_half["winner"] == "single_item"
    assert val_half == val_opt


def test_maxcut_qubo_equals_minus_cut_value():
    """The QUBO energy of the Max-Cut encoding must equal minus the cut value for all 2^6 assignments."""
    G = nx.erdos_renyi_graph(n=6, p=0.6, seed=789)
    rng = np.random.default_rng(789)
    W = np.zeros((6, 6))
    for u, v in G.edges():
        w = rng.integers(1, 10)
        W[u, v] = w
        W[v, u] = w

    Q, offset = maxcut_to_qubo(W)

    for mask in range(1 << 6):
        x = np.array([(mask >> i) & 1 for i in range(6)], dtype=float)

        # Compute ground truth cut value
        cut_val = 0.0
        for i in range(6):
            for j in range(i + 1, 6):
                if x[i] != x[j]:
                    cut_val += W[i, j]

        # QUBO energy
        qubo_energy = evaluate_qubo(Q, x, offset)

        # Cut(x) == - (x^T Q x + offset)
        assert abs(cut_val - (-qubo_energy)) < 1e-12, (
            f"Cut value mismatch: Cut={cut_val}, -QUBO={-qubo_energy}"
        )


def test_ising_and_qubo_energies_agree_on_all_assignments():
    """Ising and QUBO energies must agree on all 2^7 assignments."""
    n = 7
    G = nx.erdos_renyi_graph(n=n, p=0.5, seed=321)
    rng = np.random.default_rng(321)
    W = np.zeros((n, n))
    for u, v in G.edges():
        w = rng.integers(1, 5)
        W[u, v] = w
        W[v, u] = w

    Q, qubo_off = maxcut_to_qubo(W)
    J, h, ising_off = qubo_to_ising(Q, qubo_off)

    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        s = 2.0 * x - 1.0

        e_qubo = evaluate_qubo(Q, x, qubo_off)
        e_ising = ising_energy(s, J, h, ising_off)

        assert abs(e_qubo - e_ising) < 1e-12, (
            f"Discrepancy at mask {mask}: E_QUBO={e_qubo}, E_Ising={e_ising}"
        )


def test_cardinality_qubo_large_penalty():
    """Cardinality QUBO with a sufficiently large penalty must return exactly k items."""
    rng = np.random.default_rng(999)
    n, k = 10, 4
    values = rng.integers(5, 50, size=n).astype(float)
    large_penalty = 10.0 * np.max(values)

    Q, off = cardinality_qubo(values, k, penalty=large_penalty)
    best_e, best_x = brute_force_qubo(Q, off)

    # Exactly k items selected
    assert int(np.sum(best_x)) == k

    # Must match the top k largest values
    expected_opt_indices = np.argsort(values)[::-1][:k]
    expected_opt_val = np.sum(values[expected_opt_indices])
    actual_val = np.sum(values * best_x)
    assert abs(actual_val - expected_opt_val) < 1e-9


def test_simulated_annealing_deterministic_with_fixed_seed():
    """Simulated annealing on QUBO must produce identical outputs when given the same seed."""
    rng = np.random.default_rng(111)
    n = 8
    Q = np.triu(rng.standard_normal((n, n)), k=0)

    e1, x1, _ = simulated_annealing_qubo(Q, seed=42, steps=500, restarts=2)
    e2, x2, _ = simulated_annealing_qubo(Q, seed=42, steps=500, restarts=2)

    assert abs(e1 - e2) < 1e-12
    assert np.array_equal(x1, x2)
