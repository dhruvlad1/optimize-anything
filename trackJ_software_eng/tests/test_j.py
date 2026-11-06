"""Fast tests for Track J solver and AST mutation primitives."""
import ast

import numpy as np
import pytest

from trackJ_software_eng.mutator import mutation_candidates, mutate_source
from trackJ_software_eng.setcover import greedy_cover, ilp_cover, mutation_score, scipy_cover, summarize_subset


def test_greedy_can_be_suboptimal_and_ilp_is_optimal():
    sets = [{1, 2, 4, 5}, {1, 2, 3}, {4, 5, 6}, {1, 2}]
    costs = [1, 1, 1, 1]
    greedy, gcost = greedy_cover(sets, costs)
    optimum, cost, proven, _ = ilp_cover(sets, costs)
    assert len(greedy) == 3 and gcost == 3
    assert len(optimum) == 2 and cost == 2 and proven
    assert summarize_subset(sets, costs, optimum)["coverage"] == 1
    assert sum(costs[i] for i in optimum) == cost
    assert scipy_cover(sets, costs)[1] == cost


def test_greedy_and_ilp_crosscheck_on_ten_random_instances():
    rng = np.random.default_rng(219)
    for _ in range(10):
        universe = set(range(10))
        sets = [set(np.flatnonzero(rng.random(10) < .35).tolist()) for _ in range(12)]
        sets.append(universe.copy())  # guaranteed feasibility
        costs = rng.uniform(.1, 4, len(sets))
        greedy, gcost = greedy_cover(sets, costs)
        optimum, cost, proven, _ = ilp_cover(sets, costs)
        scipy, scipy_cost = scipy_cover(sets, costs)
        assert summarize_subset(sets, costs, greedy)["coverage"] == 1
        assert cost <= gcost + 1e-9
        assert abs(cost - scipy_cost) < 1e-6
        assert abs(costs[optimum].sum() - cost) < 1e-9
        assert proven


def test_ast_mutator_changes_one_operation_and_compiles():
    source = "def f(a, b):\n    if a < b:\n        return a + 1\n    return 0\n"
    candidates = mutation_candidates(source)
    candidate = next(x["candidate"] for x in candidates if x["operator"] == "Add")
    changed = mutate_source(source, candidate)
    compile(changed, "mutant.py", "exec")
    assert ast.dump(ast.parse(source)) != ast.dump(ast.parse(changed))
    assert "a - 1" in changed


def test_full_suite_score_on_mutants_it_killed_is_one():
    score, killed, denominator = mutation_score([True, False, True, True], [True, False, True, True])
    assert (score, killed, denominator) == (1.0, 3, 3)


@pytest.mark.skipif(not __import__("pathlib").Path("trackJ_software_eng/target").exists(),
                    reason="target checkout is absent; run setup_target when Git/network use is authorized")
def test_target_checkout_is_available_for_integration():
    assert __import__("pathlib").Path("trackJ_software_eng/target/src/sortedcontainers").is_dir()
