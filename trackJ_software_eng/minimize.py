"""Solve the recorded Track J set-cover instance and validate real subsets."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trackJ_software_eng.profile_tests import _env, _tests_args
from trackJ_software_eng.setcover import greedy_cover, greedy_fastest_cover, ilp_cover, scipy_cover, summarize_subset
from trackJ_software_eng.setup_target import ROOT, WORK

RESULTS = ROOT / "results"
# Pytest reports durations at 10 ms resolution. A zero cost would make every
# zero-cost test free to add and would make the greedy coverage-per-cost ratio undefined.
COST_FLOOR_S = 0.01
REPETITIONS = 3


def _run(args, timeout=600):
    return subprocess.run(args, cwd=WORK, env=_env(WORK / "src"), text=True, capture_output=True, timeout=timeout)


def _pytest_base():
    return [sys.executable, "-m", "pytest", "-q"] + _tests_args()[:-1]


def _coverage_run(name, ids=None):
    data = WORK / f".coverage_{name}"
    data.unlink(missing_ok=True)
    args = [sys.executable, "-m", "coverage", "run", "--branch", "--data-file", str(data),
            "--source", "sortedcontainers", "-m", "pytest", "-q"] + _tests_args()[:-1]
    args += ["-p", "trackJ_software_eng.pytest_contexts"] + (["tests"] if ids is None else ids)
    start = time.perf_counter()
    p = _run(args, 600)
    elapsed = time.perf_counter() - start
    if p.returncode:
        raise RuntimeError(f"coverage pytest failed ({p.returncode})\n{p.stdout}\n{p.stderr}")
    from coverage import Coverage
    cov = Coverage(data_file=str(data), source=["sortedcontainers"])
    cov.load()
    report_path = WORK / f".coverage_{name}.json"
    cov.json_report(outfile=str(report_path), pretty_print=False)
    report = json.loads(report_path.read_text(encoding="utf-8")); report_path.unlink(missing_ok=True)
    lines, branches = set(), set()
    for filename, entry in report.get("files", {}).items():
        normalized = str(filename).replace("\\", "/")
        marker = "src/sortedcontainers/"; offset = normalized.find(marker)
        if offset < 0: continue
        rel = normalized[offset:]
        lines.update(f"{rel}:{line}" for line in entry.get("executed_lines", []))
        branches.update(f"{rel}:{arc[0]}->{arc[1]}" for arc in entry.get("executed_branches", []))
    return lines, branches, elapsed, p.stdout


def _facts(raw):
    print("Runtime facts from results/trackJ_profile.csv", flush=True)
    print(f"tests={len(raw)}; median_runtime_s == 0: {int((raw == 0).sum())}; smallest nonzero: {raw[raw > 0].min():.2f}s", flush=True)
    print("runtime bin (s) | tests", flush=True)
    rounded = np.round(raw, 2)
    for i in range(11):
        v = i / 100
        print(f"{v:.2f} | {int((rounded == v).sum())}", flush=True)
    print(f">0.10 | {int((raw > .10).sum())}", flush=True)


def _selected_summary(name, indices, elements, raw, costs, universe):
    summary = summarize_subset(elements, costs, indices, universe)
    raw_total = float(raw[indices].sum())
    covered_lines = set().union(*(set(elements[i]) for i in indices))
    line_count = sum(str(e).count(":") == 1 and "->" not in str(e) for e in covered_lines)
    branch_count = sum("->" in str(e) for e in covered_lines)
    total_lines = sum(str(e).count(":") == 1 and "->" not in str(e) for e in universe)
    total_branches = sum("->" in str(e) for e in universe)
    return {"method": name, "tests": len(indices), "floored_cost_s": summary["cost"], "raw_runtime_s": raw_total,
            "line_coverage": f"{line_count}/{total_lines}", "branch_coverage": f"{branch_count}/{total_branches}",
            "coverage_fraction": summary["coverage"]}


def main():
    profile_path, sets_path = RESULTS / "trackJ_profile.csv", RESULTS / "trackJ_coverage_sets.json"
    if not profile_path.exists() or not sets_path.exists():
        raise FileNotFoundError("run profile_tests after setup_target first")
    profile = pd.read_csv(profile_path)
    tests = profile.test_id.tolist()
    raw = profile.median_runtime_s.to_numpy(float)
    _facts(raw)
    costs = np.maximum(raw, COST_FLOOR_S)
    sets_json = json.loads(sets_path.read_text(encoding="utf-8"))
    elements = [set(sets_json[t]["lines"]) | set(sets_json[t]["branches"]) for t in tests]
    universe = set().union(*elements)
    full_lines = set().union(*(set(sets_json[t]["lines"]) for t in tests))
    full_branches = set().union(*(set(sets_json[t]["branches"]) for t in tests))
    print(f"Full coverage universe: {len(full_lines)} lines and {len(full_branches)} branches across {len(tests)} tests.", flush=True)
    weighted, _ = greedy_cover(elements, costs)
    try:
        ilp, ilp_cost, optimal, solve_s = ilp_cover(elements, costs)
        ilp_error = None
    except (ImportError, RuntimeError) as exc:
        ilp, ilp_cost, optimal, solve_s = None, None, False, None
        ilp_error = str(exc)
    sci, sci_cost = scipy_cover(elements, costs)
    agrees = None if ilp is None else abs(ilp_cost - sci_cost) <= 1e-6
    try:
        unit_ilp, unit_cost, unit_opt, unit_solve = ilp_cover(elements, np.ones(len(tests)))
        unit_error = None
    except (ImportError, RuntimeError) as exc:
        unit_ilp, unit_cost, unit_opt, unit_solve = None, None, False, None
        unit_error = str(exc)
    unit_sci, unit_sci_cost = scipy_cover(elements, np.ones(len(tests)))
    unit_agrees = None if unit_ilp is None else abs(unit_cost - unit_sci_cost) <= 1e-6
    if ilp is None:
        ilp, ilp_cost = sci, sci_cost
    if unit_ilp is None:
        unit_ilp, unit_cost = unit_sci, unit_sci_cost
    k = len(ilp)
    rng = np.random.default_rng(713)
    random_draws = [list(map(int, rng.choice(len(tests), size=k, replace=False))) for _ in range(20)]
    fastest, _ = greedy_fastest_cover(elements, costs)
    unit_greedy, _ = greedy_cover(elements, np.ones(len(tests)))
    subsets = {"full_suite": list(range(len(tests))), "ilp": ilp, "greedy": weighted,
               "fastest_until_covered": fastest, "first_k": list(range(k)),
               **{f"random_{i+1:02d}": draw for i, draw in enumerate(random_draws)},
               "unit_cost_ilp": unit_ilp, "unit_cost_greedy": unit_greedy}
    rows = [_selected_summary(name, idxs, elements, raw, costs, universe) for name, idxs in subsets.items()]
    table = pd.DataFrame(rows)
    if ilp_error:
        print(f"SCIP runtime ILP: unavailable ({ilp_error}); no optimality proof or SCIP solve time. SciPy MILP result: tests={len(sci)}, floored objective={sci_cost:.2f}s.", flush=True)
    else:
        print(f"SCIP runtime ILP: tests={len(ilp)}, floored objective={ilp_cost:.2f}s, optimal={optimal}, solve time={solve_s:.4f}s; SciPy cross-check objective={sci_cost:.2f}s, agrees={agrees}.", flush=True)
    if unit_error:
        print(f"Unit-cost SCIP ILP: unavailable ({unit_error}); SciPy MILP result: tests={len(unit_sci)}, objective={unit_sci_cost:.0f}.", flush=True)
    else:
        print(f"Unit-cost ILP: tests={len(unit_ilp)}, optimal={unit_opt}, solve time={unit_solve:.4f}s; SciPy objective={unit_sci_cost:.0f}, agrees={unit_agrees}.", flush=True)
    print("Solver and baseline results (line/branch coverage is against the recorded full-suite union)", flush=True)
    print(table.drop(columns="coverage_fraction").to_string(index=False), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "trackJ_subsets.json").write_text(json.dumps({n: [tests[i] for i in ix] for n, ix in subsets.items()}, indent=2), encoding="utf-8")
    table.to_csv(RESULTS / "trackJ_minimization.csv", index=False)

    # Validate every selected subset whose recorded union claims full coverage.
    cover_names = [r["method"] for r in rows if r["coverage_fraction"] >= 1.0]
    print("\nActual documented-suite execution and coverage validation", flush=True)
    actual_rows, timing_rows = [], []
    for name in cover_names:
        ids = None if name == "full_suite" else [tests[i] for i in subsets[name]]
        lines, branches, elapsed, out = _coverage_run("verify_" + name, ids)
        missing_l, missing_b = full_lines - lines, full_branches - branches
        print(f"{name}: coverage run wall={elapsed:.3f}s; actual coverage={len(lines)}/{len(full_lines)} lines, {len(branches)}/{len(full_branches)} branches; missing lines={len(missing_l)}, branches={len(missing_b)}", flush=True)
        print(out, end="", flush=True)
        if missing_l or missing_b:
            print(f"WARNING: {name} failed full coverage; missing lines={sorted(missing_l)[:20]}, branches={sorted(missing_b)[:20]}", flush=True)
        actual_rows.append({"method": name, "actual_lines": len(lines), "total_lines": len(full_lines), "actual_branches": len(branches), "total_branches": len(full_branches), "missing_lines": len(missing_l), "missing_branches": len(missing_b)})
        for with_cov in (False, True):
            samples = []
            for rep in range(REPETITIONS):
                if with_cov:
                    _, _, wall, output = _coverage_run(f"timing_{name}_{rep+1}", ids)
                else:
                    args = _pytest_base() + (["tests"] if ids is None else ids)
                    start = time.perf_counter(); p = _run(args, 600); wall = time.perf_counter() - start
                    if p.returncode: raise RuntimeError(f"pytest timing run failed for {name}:\n{p.stdout}\n{p.stderr}")
                    output = p.stdout
                samples.append(wall)
                print(f"timing {name} coverage={with_cov} repetition={rep+1}: {wall:.3f}s", flush=True)
            passed = next((line.strip() for line in output.splitlines() if " passed" in line), "pass count unavailable")
            timing_rows.append({"method": name, "tests": len(tests) if ids is None else len(ids), "coverage": with_cov,
                                "median_wall_s": statistics.median(samples), "spread_s": max(samples)-min(samples),
                                "samples_s": json.dumps(samples), "pytest_result": passed})
    timings = pd.DataFrame(timing_rows)
    for with_cov in (False, True):
        full = next((r for r in timing_rows if r["method"] == "full_suite" and r["coverage"] == with_cov), None)
        if full:
            for row in timing_rows:
                if row["coverage"] == with_cov:
                    row["speedup_vs_full"] = full["median_wall_s"] / row["median_wall_s"]
    timings = pd.DataFrame(timing_rows)
    print("\nThree repetitions; wall time includes pytest startup/collection. Optimised objective is sum of per-test floored costs, not wall time.", flush=True)
    print(timings.to_string(index=False), flush=True)
    print("\nObjective vs wall time (no coverage):", flush=True)
    no_cov = timings[timings.coverage == False].set_index("method")
    agree_rows = []
    for name in cover_names:
        obj = float(table.loc[table.method == name, "floored_cost_s"].iloc[0])
        wall = float(no_cov.loc[name, "median_wall_s"])
        spread = float(no_cov.loc[name, "spread_s"])
        agree_rows.append({"method": name, "objective_s": obj, "wall_median_s": wall, "startup_collection_overhead_s": wall-obj,
                           "objective_as_pct_wall": 100*obj/wall if wall else float("nan"), "wall_spread_s": spread,
                           "within_timing_spread": abs(wall-obj) <= spread})
    agree = pd.DataFrame(agree_rows)
    print(agree.to_string(index=False), flush=True)
    actual_df = pd.DataFrame(actual_rows)
    actual_df.to_csv(RESULTS / "trackJ_actual_coverage.csv", index=False)
    timings.to_csv(RESULTS / "trackJ_timings.csv", index=False)
    print(f"\nFiles written: {RESULTS / 'trackJ_subsets.json'}, {RESULTS / 'trackJ_minimization.csv'}, {RESULTS / 'trackJ_actual_coverage.csv'}, {RESULTS / 'trackJ_timings.csv'}", flush=True)


if __name__ == "__main__":
    main()

