"""trackH_dbms/tests/test_index_solvers.py
-----------------------------------------
Unit tests for pure Python index selection solvers.
Requires NO database connection (runs entirely in-memory).

Verifies:
1. On 20 random synthetic instances (8 to 12 candidates, 6 queries, seeds 0-19):
   - ILP (SCIP) matches brute force and SciPy MILP.
   - All solvers respect the storage budget.
2. DP matches brute force on a purely additive instance.
3. Greedy benefit never exceeds brute force optimum and never violates budget.
4. Hand-built instance where greedy-by-ratio is worse than DP.
5. Hand-built instance where additive greedy and DP choose redundant indexes
   (both helping the same query), whereas the assignment ILP chooses a better pair.
6. Validation checks for malformed inputs.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from trackH_dbms.index_solvers import (
    SolverResult,
    brute_force,
    dp_additive,
    greedy_additive,
    ilp_assignment,
    ilp_assignment_scipy,
    model_cost,
)


# ---------------------------------------------------------------------------
# Synthetic instance generator
# ---------------------------------------------------------------------------

def _generate_synthetic_instance(seed: int) -> tuple[
    list[str], dict[str, float], float, dict[str, dict[str, float]], dict[str, float], dict[str, float]
]:
    """Generate a random instance with 8-12 candidates and 6 queries."""
    rng = np.random.default_rng(seed)
    n_cands = int(rng.integers(8, 13))  # 8 to 12 candidates
    n_queries = 6

    cands = [f"cand_{i}" for i in range(n_cands)]
    queries = [f"q_{j}" for j in range(n_queries)]

    # Positive integer query weights
    weights = {q: float(rng.integers(1, 6)) for q in queries}

    # Baseline query costs
    cost_none = {q: round(float(rng.uniform(50.0, 300.0)), 2) for q in queries}

    # Candidate sizes in MB (1.0 to 15.0 MB)
    sizes = {c: round(float(rng.uniform(1.0, 15.0)), 2) for c in cands}
    total_size = sum(sizes.values())

    # Budget between 20% and 55% of total size
    budget = round(float(total_size * rng.uniform(0.20, 0.55)), 2)

    # Cost matrix: candidate improves some queries, leaves others unaffected
    cost_matrix: dict[str, dict[str, float]] = {q: {} for q in queries}
    for q in queries:
        c_base = cost_none[q]
        for c in cands:
            # 50% chance candidate improves query by 10% to 80%
            if rng.random() < 0.50:
                discount = rng.uniform(0.10, 0.80)
                cost_matrix[q][c] = round(c_base * (1.0 - discount), 2)
            else:
                cost_matrix[q][c] = c_base

    return cands, sizes, budget, cost_matrix, cost_none, weights


# ---------------------------------------------------------------------------
# Test 1: Random synthetic instances: ILP (SCIP) vs SciPy MILP vs Brute Force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(20)))
def test_synthetic_ilp_matches_brute_force_and_scipy(seed: int):
    """Across seeds 0-19, ILP (SCIP) matches Brute Force and SciPy MILP."""
    cands, sizes, budget, cost_matrix, cost_none, weights = _generate_synthetic_instance(seed)

    res_bf = brute_force(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_scip = ilp_assignment(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_scipy = ilp_assignment_scipy(cands, sizes, budget, cost_matrix, cost_none, weights)

    # All solvers must strictly respect the budget
    assert res_bf.total_size <= budget + 1e-6
    assert res_scip.total_size <= budget + 1e-6
    assert res_scipy.total_size <= budget + 1e-6

    # Optimal model costs must match within numerical tolerance
    assert math.isclose(res_scip.model_cost, res_bf.model_cost, abs_tol=1e-3), (
        f"Seed {seed}: SCIP cost {res_scip.model_cost} != BF cost {res_bf.model_cost}"
    )
    assert math.isclose(res_scipy.model_cost, res_bf.model_cost, abs_tol=1e-3), (
        f"Seed {seed}: SciPy cost {res_scipy.model_cost} != BF cost {res_bf.model_cost}"
    )


# ---------------------------------------------------------------------------
# Test 2: Greedy bounds and budget compliance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(20)))
def test_greedy_never_exceeds_optimum_benefit_and_obeys_budget(seed: int):
    """Greedy benefit is <= exact optimum benefit and total size <= budget."""
    cands, sizes, budget, cost_matrix, cost_none, weights = _generate_synthetic_instance(seed)

    res_bf = brute_force(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_greedy = greedy_additive(cands, sizes, budget, cost_matrix, cost_none, weights)

    assert res_greedy.total_size <= budget + 1e-6

    cost_baseline = model_cost(set(), cost_matrix, weights, cost_none)
    benefit_bf = cost_baseline - res_bf.model_cost
    benefit_greedy = cost_baseline - res_greedy.model_cost

    # Greedy can never achieve strictly higher benefit than the exact optimum
    assert benefit_greedy <= benefit_bf + 1e-4


# ---------------------------------------------------------------------------
# Test 3: Purely additive instance: DP matches Brute Force
# ---------------------------------------------------------------------------

def test_dp_matches_brute_force_on_purely_additive_instance():
    """When queries are disjoint and sizes are integer, DP matches Brute Force exactly."""
    n = 8
    cands = [f"c_{i}" for i in range(n)]
    queries = [f"q_{i}" for i in range(n)]

    # Disjoint benefits: candidate i only helps query i
    cost_none = {f"q_{i}": 100.0 for i in range(n)}
    weights = {f"q_{i}": 1.0 for i in range(n)}
    # Sizes and benefits
    sizes = {
        "c_0": 3.0, "c_1": 4.0, "c_2": 2.0, "c_3": 5.0,
        "c_4": 1.0, "c_5": 6.0, "c_6": 3.0, "c_7": 2.0,
    }
    benefits = [20.0, 35.0, 18.0, 42.0, 8.0, 50.0, 25.0, 16.0]
    cost_matrix: dict[str, dict[str, float]] = {q: {} for q in queries}
    for i in range(n):
        for j in range(n):
            if i == j:
                cost_matrix[f"q_{i}"][f"c_{j}"] = cost_none[f"q_{i}"] - benefits[i]
            else:
                cost_matrix[f"q_{i}"][f"c_{j}"] = cost_none[f"q_{i}"]

    budget = 11.0  # whole integer capacity

    res_bf = brute_force(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_dp = dp_additive(cands, sizes, budget, cost_matrix, cost_none, weights, size_unit=1.0)

    assert res_dp.total_size <= budget + 1e-6
    assert math.isclose(res_dp.model_cost, res_bf.model_cost, abs_tol=1e-4)
    assert res_dp.selected == res_bf.selected


# ---------------------------------------------------------------------------
# Test 4: Hand-built instance where greedy-by-ratio is worse than DP
# ---------------------------------------------------------------------------

def test_hand_built_greedy_ratio_worse_than_dp():
    """Classic knapsack counterexample where greedy ratio choice blocks better items.

    Budget B = 10.
    Cand A: size = 6, benefit = 12. Ratio = 2.0.
    Cand B: size = 5, benefit = 9.  Ratio = 1.8.
    Cand C: size = 5, benefit = 9.  Ratio = 1.8.

    Greedy picks: {A} -> size 6, benefit 12.
    DP picks: {B, C} -> size 10, benefit 18.
    """
    cands = ["cand_A", "cand_B", "cand_C"]
    queries = ["q_A", "q_B", "q_C"]
    weights = {"q_A": 1.0, "q_B": 1.0, "q_C": 1.0}
    cost_none = {"q_A": 20.0, "q_B": 20.0, "q_C": 20.0}
    sizes = {"cand_A": 6.0, "cand_B": 5.0, "cand_C": 5.0}
    budget = 10.0

    cost_matrix = {
        "q_A": {"cand_A": 8.0, "cand_B": 20.0, "cand_C": 20.0},   # benefit 12
        "q_B": {"cand_A": 20.0, "cand_B": 11.0, "cand_C": 20.0},  # benefit 9
        "q_C": {"cand_A": 20.0, "cand_B": 20.0, "cand_C": 11.0},  # benefit 9
    }

    res_greedy = greedy_additive(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_dp = dp_additive(cands, sizes, budget, cost_matrix, cost_none, weights, size_unit=1.0)

    assert res_greedy.selected == {"cand_A"}
    assert res_dp.selected == {"cand_B", "cand_C"}
    # Model cost: lower is better
    # Baseline total cost = 60.
    # Greedy cost = 8 + 20 + 20 = 48 (benefit 12).
    # DP cost = 20 + 11 + 11 = 42 (benefit 18).
    assert res_dp.model_cost < res_greedy.model_cost
    assert res_greedy.model_cost == 48.0
    assert res_dp.model_cost == 42.0


# ---------------------------------------------------------------------------
# Test 5: Redundant indexes instance: Greedy & DP pick redundant, ILP picks better
# ---------------------------------------------------------------------------

def test_hand_built_redundant_indexes_greedy_dp_fail_ilp_wins():
    """Two candidates are useful for Query 1, while a third candidate is useful for Query 2.

    Query 1 (weight 1.0), baseline cost = 100.
    Query 2 (weight 1.0), baseline cost = 100.
    Budget = 2.0 (each candidate has size 1.0, so at most 2 indexes allowed).

    Candidate c1: reduces Q1 from 100 to 10 (benefit 90). No effect on Q2.
    Candidate c2: reduces Q1 from 100 to 12 (benefit 88). No effect on Q2.
    Candidate c3: reduces Q2 from 100 to 20 (benefit 80). No effect on Q1.
    Candidate c4: useless (benefit 0).

    Additive model:
      c1 benefit = 90 (ratio 90/1)
      c2 benefit = 88 (ratio 88/1)
      c3 benefit = 80 (ratio 80/1)
      Greedy and DP pick {c1, c2} assuming combined benefit 90+88 = 178.
      Actual assignment cost for {c1, c2}:
        Q1 uses c1 (cost 10); Q2 has no index (cost 100).
        Total workload cost = 110.0 (real benefit = 90).

    Assignment ILP:
      Knows each query can only use ONE index (substitution).
      Picks {c1, c3}:
        Q1 uses c1 (cost 10); Q2 uses c3 (cost 20).
        Total workload cost = 30.0 (real benefit = 170).

    Result: ILP achieves cost 30.0 vs Greedy/DP cost 110.0!
    """
    cands = ["c1", "c2", "c3", "c4"]
    queries = ["q1", "q2"]
    weights = {"q1": 1.0, "q2": 1.0}
    cost_none = {"q1": 100.0, "q2": 100.0}
    sizes = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    budget = 2.0

    cost_matrix = {
        "q1": {"c1": 10.0, "c2": 12.0, "c3": 100.0, "c4": 100.0},
        "q2": {"c1": 100.0, "c2": 100.0, "c3": 20.0, "c4": 100.0},
    }

    res_greedy = greedy_additive(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_dp = dp_additive(cands, sizes, budget, cost_matrix, cost_none, weights, size_unit=1.0)
    res_ilp = ilp_assignment(cands, sizes, budget, cost_matrix, cost_none, weights)
    res_scipy = ilp_assignment_scipy(cands, sizes, budget, cost_matrix, cost_none, weights)

    # Greedy and DP make the additive error of picking both redundant Q1 indexes
    assert res_greedy.selected == {"c1", "c2"}
    assert res_dp.selected == {"c1", "c2"}
    assert res_greedy.model_cost == 110.0
    assert res_dp.model_cost == 110.0

    # ILP correctly identifies substitution and picks {c1, c3}
    assert res_ilp.selected == {"c1", "c3"}
    assert res_scipy.selected == {"c1", "c3"}
    assert res_ilp.model_cost == 30.0
    assert res_scipy.model_cost == 30.0

    # ILP is strictly superior
    assert res_ilp.model_cost < res_greedy.model_cost


# ---------------------------------------------------------------------------
# Test 6: Input validation checks
# ---------------------------------------------------------------------------

def test_input_validation():
    """Solvers validate budget, duplicates, missing entries, and bounds."""
    cands = ["c1", "c2"]
    sizes = {"c1": 5.0, "c2": 3.0}
    budget = 10.0
    cost_none = {"q1": 100.0}
    weights = {"q1": 1.0}
    cost_matrix = {"q1": {"c1": 50.0, "c2": 80.0}}

    # Negative budget
    with pytest.raises(ValueError, match="Budget must be non-negative"):
        greedy_additive(cands, sizes, -1.0, cost_matrix, cost_none, weights)

    # Duplicate candidates
    with pytest.raises(ValueError, match="duplicates"):
        greedy_additive(["c1", "c1"], sizes, budget, cost_matrix, cost_none, weights)

    # Missing sizes
    with pytest.raises(ValueError, match="missing from sizes"):
        greedy_additive(["c1", "c3"], sizes, budget, cost_matrix, cost_none, weights)

    # Brute force candidate count guard (> 16)
    large_cands = [f"c_{i}" for i in range(17)]
    large_sizes = {c: 1.0 for c in large_cands}
    with pytest.raises(ValueError, match="only supports up to 16"):
        brute_force(large_cands, large_sizes, budget, cost_matrix, cost_none, weights)
