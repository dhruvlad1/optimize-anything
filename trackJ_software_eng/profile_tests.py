"""Collect per-item run times and per-item line/branch coverage contexts."""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from trackJ_software_eng.setup_target import ROOT, WORK, prepare_work_copy

RESULTS = ROOT / "results"
REPS = 3
TEST_IGNORES = ["tests/benchmark.py", "tests/benchmark_plot.py", "tests/benchmark_sorteddict.py",
                "tests/benchmark_sortedlist.py", "tests/benchmark_sortedset.py",
                "tests/plot_lengths_histogram_add.py", "tests/plot_lengths_histogram_delitem.py"]


def _tests_args():
    args = ["-o", "addopts=", "--doctest-modules"]
    for ignored in TEST_IGNORES:
        args.extend(["--ignore", ignored])
    args.append("tests")
    return args


def _env(src):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(args, cwd, src, timeout=300, extra_env=None):
    env = _env(src)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)


def _collect(root, src):
    p = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"] + _tests_args(), root, src, 180)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return [line.strip() for line in p.stdout.splitlines() if "::" in line and line.strip().startswith("tests/")]


def _durations(output):
    found, fallback = {}, {}
    for line in output.splitlines():
        m = re.match(r"\s*([0-9.eE+-]+)s\s+(setup|call|teardown)\s+(.+?)\s*$", line)
        if m:
            nodeid = m.group(3).replace("\\", "/")
            found[nodeid] = found.get(nodeid, 0.0) + float(m.group(1))
            continue
        if line.startswith("TRACKJ_DURATION "):
            try:
                item = json.loads(line[len("TRACKJ_DURATION "):])
                nodeid = item["nodeid"].replace("\\", "/")
                fallback[nodeid] = fallback.get(nodeid, 0.0) + float(item["seconds"])
            except (ValueError, KeyError, TypeError):
                continue
    for nodeid, elapsed in fallback.items():
        found.setdefault(nodeid, elapsed)
    return found


def _print_pytest_config(root):
    for name in ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"):
        path = root / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        section = None
        print(f"pytest config source: {path}", flush=True)
        for i, line in enumerate(lines):
            if line.strip().lower() in ("[pytest]", "[tool:pytest]", "[tool.pytest.ini_options]"):
                section = i
                break
        if section is not None:
            relevant = [lines[section]]
            for line in lines[section+1:]:
                if line.startswith("["):
                    break
                if line.strip():
                    relevant.append(line)
            print("\n".join(relevant), flush=True)
            return
    print("No pytest configuration file with a pytest section was found.", flush=True)


def _item_coverage(cov_file, root, src, nodeids, files):
    from coverage import Coverage
    cov = Coverage(data_file=str(cov_file), source=["sortedcontainers"])
    cov.load()
    json_path = root / ".coverage_context.json"

    def context_elements(context=None):
        options = {} if context is None else {"contexts": [context]}
        cov.json_report(outfile=str(json_path), pretty_print=False, **options)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        line_ids, branch_ids = set(), set()
        for filename, entry in report.get("files", {}).items():
            normalized = str(filename).replace("\\", "/")
            marker = "src/sortedcontainers/"
            offset = normalized.find(marker)
            if offset < 0:
                continue
            rel = normalized[offset:]
            line_ids.update(f"{rel}:{line}" for line in entry.get("executed_lines", []))
            branch_ids.update(f"{rel}:{arc[0]}->{arc[1]}" for arc in entry.get("executed_branches", []))
        return line_ids, branch_ids

    result = {}
    for nodeid in nodeids:
        line_ids, branch_ids = context_elements(re.escape(nodeid))
        result[nodeid] = {"lines": sorted(set(line_ids)), "branches": sorted(set(branch_ids))}
    json_path.unlink(missing_ok=True)
    return result


def main():
    if not (WORK / "src" / "sortedcontainers").exists():
        prepare_work_copy()
    root, src = WORK, WORK / "src"
    _print_pytest_config(root)
    nodeids = _collect(root, src)
    if not nodeids:
        raise RuntimeError("pytest collection returned no test ids")
    print("Suite under study: only tests/ (documented project suite; doctest modules outside tests/ are excluded).", flush=True)
    expected = 299
    if len(nodeids) != expected:
        delta = len(nodeids) - expected
        print(f"Expected {expected} test ids, collected {len(nodeids)} (difference {delta:+d}). First collected ids: {nodeids[:5]}", flush=True)
        raise RuntimeError(f"expected exactly {expected} test ids under tests/, found {len(nodeids)} (difference {delta:+d})")
    help_result = _run([sys.executable, "-m", "pytest", "--help", "-o", "addopts="], root, src, 60)
    if help_result.returncode or "--durations-min" not in help_result.stdout:
        raise RuntimeError("this pytest version does not provide --durations-min")
    duration_help = next((line.strip() for line in help_result.stdout.splitlines() if "--durations-min" in line), "--durations-min exists")
    print(f"Verified pytest option: {duration_help}", flush=True)
    file_count = len({nodeid.split("::", 1)[0] for nodeid in nodeids})
    print(f"Estimated runtime: approximately {4 + 3*4 + len(nodeids)*.12 + file_count*2:.0f}s for one full per-test-context coverage run, {file_count} short per-file import-context passes, JSON context extraction, and three duration passes; isolated profiling would be about {len(nodeids)*3:.0f}s at 1s process startup per test. Chose item contexts plus per-file passes to attribute import-time execution, then three --durations=0 --durations-min=0 passes.", flush=True)
    print(f"Collected exactly {len(nodeids)} test ids from tests/.", flush=True)
    # Coverage is attributed per pytest item by an explicit switch_context plugin.
    (root / ".coveragerc").write_text("[run]\nbranch = True\ndynamic_context =\nsource = src/sortedcontainers\n", encoding="utf-8")
    datafile = root / ".coverage"
    erase = _run([sys.executable, "-m", "coverage", "erase", "--rcfile", str(root / ".coveragerc"),
                  "--data-file", str(datafile)], root, src, 60)
    if erase.returncode:
        raise RuntimeError(erase.stdout + erase.stderr)
    cov = _run([sys.executable, "-m", "coverage", "run", "--rcfile", str(root / ".coveragerc"),
                "--data-file", str(datafile), "-m", "pytest", "-q"] + _tests_args() +
               ["-p", "trackJ_software_eng.pytest_contexts"], root, src, 600)
    print(cov.stdout, end="", flush=True)
    if cov.returncode:
        raise RuntimeError(cov.stderr)
    durations = {n: [] for n in nodeids}
    overhead = []
    for rep in range(REPS):
        start = time.perf_counter()
        p = _run([sys.executable, "-m", "pytest", "-q", "--durations=0", "--durations-min=0"] +
                 _tests_args() + ["-p", "trackJ_software_eng.pytest_contexts"],
                 root, src, 300, {"TRACKJ_REPORT_DURATIONS": "1"})
        wall = time.perf_counter() - start
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
        for n, duration in _durations(p.stdout + p.stderr).items():
            n = n.replace("\\", "/")
            if n in durations:
                durations[n].append(duration)
        overhead.append(wall - sum(_durations(p.stdout + p.stderr).values()))
        print(f"duration pass {rep + 1}/{REPS}: process wall={wall:.3f}s", flush=True)
    missing = [n for n, vals in durations.items() if not vals]
    if missing:
        print(f"Duration report missing {len(missing)} test ids after --durations=0 --durations-min=0 and plugin fallback. First missing ids and diagnosis:", flush=True)
        for n in missing[:30]:
            print(f"  {n}: pytest emitted no setup/call/teardown duration report in any repetition; recording 0.0s (unmeasured).", flush=True)
        if len(missing) > 30:
            print(f"  ... {len(missing)-30} additional missing ids.", flush=True)
    sets = _item_coverage(datafile, root, src, nodeids, [])
    # Imports happen during pytest collection, before an individual item's
    # context starts. Re-run each test file in isolation and attribute its
    # empty-context import coverage to the tests in that file (each selected
    # item would import its containing test module when run alone).
    files = sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})
    for ix, test_file in enumerate(files):
        file_data = root / f".coverage_import_{ix:02d}"
        file_data.unlink(missing_ok=True)
        args = [sys.executable, "-m", "coverage", "run", "--rcfile", str(root / ".coveragerc"),
                "--data-file", str(file_data), "-m", "pytest", "-q"]
        args += _tests_args()[:-1] + ["-p", "trackJ_software_eng.pytest_contexts", test_file]
        p = _run(args, root, src, 180)
        if p.returncode:
            raise RuntimeError(f"per-file coverage run failed for {test_file}:\n{p.stdout}\n{p.stderr}")
        # Read empty context via coverage's documented JSON API.
        from coverage import Coverage
        cov_file = Coverage(data_file=str(file_data), source=["sortedcontainers"])
        cov_file.load()
        json_path = root / ".coverage_context.json"
        cov_file.json_report(contexts=[r"^$"], outfile=str(json_path), pretty_print=False)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        prefix = "src/sortedcontainers/"
        import_lines, import_branches = set(), set()
        for filename, entry in report.get("files", {}).items():
            normalized = str(filename).replace("\\", "/")
            offset = normalized.find(prefix)
            if offset >= 0:
                rel = normalized[offset:]
                import_lines.update(f"{rel}:{line}" for line in entry.get("executed_lines", []))
                import_branches.update(f"{rel}:{arc[0]}->{arc[1]}" for arc in entry.get("executed_branches", []))
        json_path.unlink(missing_ok=True)
        file_ids = [nodeid for nodeid in nodeids if nodeid.split("::", 1)[0] == test_file]
        for nodeid in file_ids:
            sets[nodeid]["lines"] = sorted(set(sets[nodeid]["lines"]) | import_lines)
            sets[nodeid]["branches"] = sorted(set(sets[nodeid]["branches"]) | import_branches)
        file_data.unlink(missing_ok=True)
    # Confirm that the union of the per-test sets recovers the full-suite
    # executed source elements, including import-time lines and branches.
    all_lines = set().union(*(set(v["lines"]) for v in sets.values()))
    all_branches = set().union(*(set(v["branches"]) for v in sets.values()))
    from coverage import Coverage
    full_cov = Coverage(data_file=str(datafile), source=["sortedcontainers"]); full_cov.load()
    full_json = root / ".coverage_full.json"
    full_cov.json_report(outfile=str(full_json), pretty_print=False)
    full_report = json.loads(full_json.read_text(encoding="utf-8")); full_json.unlink(missing_ok=True)
    expected_lines, expected_branches = set(), set()
    for filename, entry in full_report.get("files", {}).items():
        normalized = str(filename).replace("\\", "/"); offset = normalized.find("src/sortedcontainers/")
        if offset >= 0:
            rel = normalized[offset:]
            expected_lines.update(f"{rel}:{line}" for line in entry.get("executed_lines", []))
            expected_branches.update(f"{rel}:{arc[0]}->{arc[1]}" for arc in entry.get("executed_branches", []))
    print(f"Per-test set union vs full coverage JSON: lines {len(all_lines)}/{len(expected_lines)}, branches {len(all_branches)}/{len(expected_branches)}; missing lines={len(expected_lines-all_lines)}, branches={len(expected_branches-all_branches)}.", flush=True)
    rows = []
    for nodeid in nodeids:
        file = nodeid.split("::", 1)[0]
        lines, branches = sets[nodeid]["lines"], sets[nodeid]["branches"]
        rows.append({"test_id": nodeid, "file": file, "median_runtime_s": statistics.median(durations[nodeid]) if durations[nodeid] else 0.0,
                     "covered_lines": len(lines), "covered_branches": len(branches)})
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "trackJ_profile.csv", index=False)
    (RESULTS / "trackJ_coverage_sets.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    unique = set().union(*(set(v["lines"]) | set(v["branches"]) for v in sets.values()))
    frequency = {}
    for v in sets.values():
        for e in set(v["lines"]) | set(v["branches"]):
            frequency[e] = frequency.get(e, 0) + 1
    no_unique = sum(not any(frequency[e] == 1 for e in set(v["lines"]) | set(v["branches"])) for v in sets.values())
    with_lines = sum(bool(v["lines"]) for v in sets.values())
    print(f"Coverage attribution: {with_lines} tests have at least one covered source line; {len(sets)-with_lines} cover nothing.", flush=True)
    print("10 slowest tests", flush=True); print(frame.nlargest(10, "median_runtime_s").to_string(index=False), flush=True)
    shares = frame.groupby("file").median_runtime_s.sum().to_frame("runtime_s")
    shares["share"] = shares.runtime_s / shares.runtime_s.sum()
    print("Runtime share by test file", flush=True); print(shares.to_string(), flush=True)
    print(f"Tests covering nothing unique: {no_unique}; full-suite coverage elements: lines={len({e for v in sets.values() for e in v['lines']})}, branches={len({e for v in sets.values() for e in v['branches']})}; median pytest process overhead (not subtracted)={statistics.median(overhead):.3f}s", flush=True)
    print(f"Files written: {RESULTS / 'trackJ_profile.csv'}; {RESULTS / 'trackJ_coverage_sets.json'}", flush=True)


if __name__ == "__main__":
    main()
