"""run_baseline.py - Baseline evaluation runner for Track F: Compiler Optimization.

Compiles and benchmarks all programs in trackF_compiler/programs/ under opt_level=0,
validating outputs against the AST oracle and recording metrics to results/trackF_baseline.csv.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.compiler import compile_program
from trackF_compiler.minilang.vm import VirtualMachine


def main() -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    programs_dir = REPO_ROOT / "trackF_compiler" / "programs"

    program_files = sorted(programs_dir.glob("*.ml"))
    if not program_files:
        raise FileNotFoundError(f"No .ml programs found in {programs_dir}")

    records: list[dict] = []
    print("Running Track F (MiniLang) Baseline Benchmarks (opt_level=0)...\n")

    for p_path in program_files:
        p_name = p_path.stem
        source = p_path.read_text(encoding="utf-8")

        # Compile time measurement: median of 5 compilations
        compile_times: list[float] = []
        ast_prog = None
        vm_prog = None
        static_instr = 0

        for _ in range(5):
            t0 = time.perf_counter()
            ast_p, _, vm_p, stats = compile_program(source, opt_level=0)
            t_comp = time.perf_counter() - t0
            compile_times.append(t_comp)
            ast_prog = ast_p
            vm_prog = vm_p
            static_instr = stats["static_instr"]

        assert ast_prog is not None and vm_prog is not None
        med_compile_time = statistics.median(compile_times)

        # Oracle execution
        oracle = ASTInterpreter()
        oracle_out = oracle.run(ast_prog)

        # VM runtime measurement: median of 5 executions
        vm_runtimes: list[float] = []
        dyn_instr = 0
        vm_out = None

        for _ in range(5):
            vm = VirtualMachine()
            t0 = time.perf_counter()
            out, dyn = vm.run(vm_prog)
            t_run = time.perf_counter() - t0
            vm_runtimes.append(t_run)
            dyn_instr = dyn
            vm_out = out

        assert vm_out is not None
        med_vm_runtime = statistics.median(vm_runtimes)

        output_match = (oracle_out == vm_out)
        if not output_match:
            raise RuntimeError(
                f"Oracle and VM output mismatch on '{p_name}':\n"
                f"  Oracle : {oracle_out}\n"
                f"  VM     : {vm_out}"
            )

        records.append({
            "program": p_name,
            "static_instr": static_instr,
            "dynamic_instr": dyn_instr,
            "vm_runtime_s": round(med_vm_runtime, 6),
            "compile_time_s": round(med_compile_time, 6),
            "output_match": output_match,
        })

    # Write results/trackF_baseline.csv
    csv_path = results_dir / "trackF_baseline.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["program", "static_instr", "dynamic_instr", "vm_runtime_s", "compile_time_s", "output_match"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Benchmark results written to: {csv_path}\n")

    # Write results/trackF_environment.json
    env_info = {
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count(),
    }
    json_path = results_dir / "trackF_environment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    print(f"Environment metadata written to: {json_path}\n")

    # Summary table
    print("=" * 90)
    print("SUMMARY TABLE: MiniLang Compiler Baseline Performance (opt_level=0)")
    print("=" * 90)
    header = f"{'Program':<20} | {'Static Instr':<14} | {'Dynamic Instr':<16} | {'VM Runtime (s)':<16} | {'Compile Time (s)':<18} | {'Match':<6}"
    print(header)
    print("-" * len(header))

    for rec in records:
        print(
            f"{rec['program']:<20} | "
            f"{rec['static_instr']:<14} | "
            f"{rec['dynamic_instr']:<16} | "
            f"{rec['vm_runtime_s']:<16.6f} | "
            f"{rec['compile_time_s']:<18.6f} | "
            f"{str(rec['output_match']):<6}"
        )

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
