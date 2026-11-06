"""Generate and evaluate single-change mutants on isolated target copies."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from trackJ_software_eng.mutator import mutation_candidates, mutate_source
from trackJ_software_eng.profile_tests import _env
from trackJ_software_eng.setup_target import ROOT, TARGET, WORK
from trackJ_software_eng.setcover import mutation_score

RESULTS = ROOT / "results"
FILES = ["src/sortedcontainers/sortedlist.py", "src/sortedcontainers/sortedset.py", "src/sortedcontainers/sorteddict.py"]


def _run(ids=None, timeout=120):
    cmd = [sys.executable, "-m", "pytest", "-q", "-x"] + (ids or [])
    try:
        p = subprocess.run(cmd, cwd=WORK, env=_env(WORK / "src"), text=True, capture_output=True, timeout=timeout)
        failed = re.findall(r"FAILED\s+([^\s]+)", p.stdout + p.stderr)
        return p.returncode == 0, "", failed[0] if failed else "", p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return False, "timeout", "", (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else "timeout"


def _mutate_copy(relative, candidate, text):
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(TARGET, WORK)
    file = WORK / relative
    file.write_text(mutate_source(text, candidate), encoding="utf-8")


def main():
    subsets_file = RESULTS / "trackJ_subsets.json"
    profile_file = RESULTS / "trackJ_profile.csv"
    if not subsets_file.exists() or not profile_file.exists():
        raise FileNotFoundError("run profile_tests and minimize before mutants")
    subsets = json.loads(subsets_file.read_text(encoding="utf-8"))
    profile = pd.read_csv(profile_file)
    # Full and subset test process launches dominate. Estimate conservatively and
    # honor the requested 40-mutant fallback when 60 would exceed 25 minutes.
    candidate_count = 60
    per_suite_est = 4.0 + 1.0 * len(subsets)  # 4s full suite plus ~1s process startup for each selected subset
    estimated = candidate_count * per_suite_est
    print(f"Estimated runtime: about {estimated:.0f}s for 60 mutants x one full suite and {len(subsets)} subset runs at {per_suite_est:.1f}s each.", flush=True)
    if estimated > 25 * 60:
        candidate_count = 40
        print(f"Reduced mutant sample from 60 to 40 because the estimated sweep exceeds 25 minutes; adjusted estimate is about {candidate_count * per_suite_est:.0f}s.", flush=True)
    rng = random.Random(671)
    pools = []
    for relative in FILES:
        source = (TARGET / relative).read_text(encoding="utf-8")
        pools.append((relative, source, mutation_candidates(source)))
    selected = []
    # Round-robin sampling guarantees a balanced spread across the three modules.
    by_file = [rng.sample(candidates, min(len(candidates), candidate_count // 3 + (1 if i < candidate_count % 3 else 0)))
               for i, (_, _, candidates) in enumerate(pools)]
    for turn in range(max(map(len, by_file))):
        for ix, pool in enumerate(by_file):
            if turn < len(pool):
                selected.append((pools[ix][0], pools[ix][1], pool[turn]))
    selected = selected[:candidate_count]
    catalog = []
    for mid, (relative, source, candidate) in enumerate(selected, 1):
        info = candidate.copy()
        info.update({"mutant_id": f"M{mid:03d}", "file": relative})
        catalog.append(info)
    pd.DataFrame([{k: v for k, v in item.items() if k != "candidate"} for item in catalog]).to_csv(RESULTS / "trackJ_mutants.csv", index=False)

    full_killed, records = [], []
    def execute(mutant, test_ids=None):
        relative, source, candidate = selected[mutant]
        _mutate_copy(relative, candidate["candidate"], source)
        return _run(test_ids)

    for i, item in enumerate(catalog):
        ok, status, first, output = execute(i)
        killed = not ok
        full_killed.append(killed)
        records.append({"mutant_id": item["mutant_id"], "file": item["file"], "line": item["line"],
                        "operator": item["operator"], "full_suite_killed": killed,
                        "full_suite_status": status or ("killed" if killed else "survived"),
                        "first_failing_test": first})
        print(f"progress full mutant {i+1}/{len(catalog)}: {item['mutant_id']} {'killed' if killed else 'survived'} {status}", flush=True)
    killed_idxs = [i for i, killed in enumerate(full_killed) if killed]
    print(f"Full suite killed {len(killed_idxs)}/{len(catalog)} mutants; {len(catalog)-len(killed_idxs)} survived (equivalent or undetected) and are excluded from subset denominators.", flush=True)
    score_rows = []
    tested_subsets = {name: ids for name, ids in subsets.items() if name != "full_suite"}
    tested_subsets["full_suite"] = []
    for name, ids in tested_subsets.items():
        killed_flags = []
        for j, mutant_idx in enumerate(killed_idxs, 1):
            ok, status, first, output = execute(mutant_idx, ids or None)
            killed_flags.append(not ok)
            records[mutant_idx][f"{name}_killed"] = not ok
            if not ok:
                records[mutant_idx][f"{name}_failure"] = status or first
        score, killed, denominator = mutation_score([True] * len(killed_idxs), killed_flags)
        score_rows.append({"subset": name, "mutants_killed": killed, "full_suite_killed_mutants": denominator,
                           "mutation_score": score})
        print(f"subset {name}: mutation score {killed}/{denominator} = {score:.3f}", flush=True)
    pd.DataFrame(records).to_csv(RESULTS / "trackJ_mutation.csv", index=False)
    print("Mutation scores", flush=True); print(pd.DataFrame(score_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
