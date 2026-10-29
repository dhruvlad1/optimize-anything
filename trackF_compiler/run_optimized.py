"""run_optimized.py - Ablation benchmark: per-program table of dynamic instruction
count reductions under different pass configurations.

For each of the 6 benchmarks:
  - Baseline (opt_level=0)
  - Each single pass in isolation
  - Fixed order (constant_fold -> cse -> licm -> strength_reduce -> dead_code_elim)
  - Best order from exp_pass_order (licm -> constant_fold -> cse -> strength_reduce -> dead_code_elim)

Records: static instruction count, dynamic instruction count, VM runtime
(median of 5 runs), compile time (median of 5 runs), whether VM output
matches the AST Oracle.

Saves:
  results/trackF_optimized.csv   -- full data table
  results/trackF_optimized_bar.png -- bar chart of dynamic instruction % reduction vs baseline
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.codegen import CodeGenerator
from trackF_compiler.minilang.compiler import compile_program
from trackF_compiler.minilang.passes import FIXED_ORDER_PASSES, run_passes
from trackF_compiler.minilang.vm import VirtualMachine

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PROGRAMS_DIR = REPO_ROOT / "trackF_compiler" / "programs"

# Optimal permutation from exp_pass_order.py
BEST_ORDER = ["licm", "constant_fold", "cse", "strength_reduce", "dead_code_elim"]

# Single-pass names (same as PASS_REGISTRY keys)
SINGLE_PASSES = ["constant_fold", "cse", "licm", "strength_reduce", "dead_code_elim"]

N_TIMING_RUNS = 1  # Single run: large benchmarks (100k+ VM steps) make 5x timing impractical


def median_time(fn, n: int = N_TIMING_RUNS) -> float:
    """Return median wall time (seconds) of calling fn() n times."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_config(
    ir_raw,
    pass_names: list[str],
    oracle_output: list[int],
    codegen: CodeGenerator,
    vm: VirtualMachine,
) -> dict:
    """Apply passes, run VM, time compile+vm, return metrics dict."""
    # Compile timing (pass application + codegen)
    def do_compile():
        ir_opt = run_passes(ir_raw, pass_names)
        return codegen.generate(ir_opt)

    compile_time = median_time(do_compile)

    # Apply once to get the VM code
    ir_opt = run_passes(ir_raw, pass_names)
    vm_code = codegen.generate(ir_opt)
    static_instr = len(vm_code)

    # VM timing
    vm_out_ref = None

    def do_vm():
        nonlocal vm_out_ref
        out, dyn = vm.run(vm_code)
        vm_out_ref = (out, dyn)

    vm_time = median_time(do_vm)
    vm_output, dyn_instr = vm_out_ref

    correct = vm_output == oracle_output

    return {
        "static_instr": static_instr,
        "dyn_instr": dyn_instr,
        "vm_time_ms": round(vm_time * 1000, 3),
        "compile_time_ms": round(compile_time * 1000, 3),
        "correct": correct,
    }


def main() -> None:
    cg = CodeGenerator()
    vm = VirtualMachine()

    bench_paths = sorted(PROGRAMS_DIR.glob("*.ml"))
    if not bench_paths:
        print(f"ERROR: No .ml files found in {PROGRAMS_DIR}")
        sys.exit(1)

    all_rows = []

    for bench_path in bench_paths:
        bench_name = bench_path.stem
        src = bench_path.read_text(encoding="utf-8")

        # Compile baseline to get raw IR and oracle
        ast_p, ir_raw, _, _ = compile_program(src, opt_level=0)
        oracle_output = ASTInterpreter().run(ast_p)

        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"  Oracle output: {oracle_output}")

        # --- Baseline ---
        m = run_config(ir_raw, [], oracle_output, cg, vm)
        baseline_dyn = m["dyn_instr"]
        print(f"  Baseline: static={m['static_instr']} dyn={m['dyn_instr']} "
              f"vm={m['vm_time_ms']}ms correct={m['correct']}")
        all_rows.append({
            "benchmark": bench_name,
            "config": "baseline",
            **m,
            "dyn_pct_reduction": 0.0,
        })

        # --- Single passes ---
        for pass_name in SINGLE_PASSES:
            m = run_config(ir_raw, [pass_name], oracle_output, cg, vm)
            pct = 100.0 * (baseline_dyn - m["dyn_instr"]) / baseline_dyn if baseline_dyn > 0 else 0.0
            print(f"  [{pass_name}]: static={m['static_instr']} dyn={m['dyn_instr']} "
                  f"({pct:+.1f}%) vm={m['vm_time_ms']}ms correct={m['correct']}")
            all_rows.append({
                "benchmark": bench_name,
                "config": pass_name,
                **m,
                "dyn_pct_reduction": round(pct, 2),
            })

        # --- Fixed order ---
        m = run_config(ir_raw, FIXED_ORDER_PASSES, oracle_output, cg, vm)
        pct = 100.0 * (baseline_dyn - m["dyn_instr"]) / baseline_dyn if baseline_dyn > 0 else 0.0
        print(f"  [fixed_order]: static={m['static_instr']} dyn={m['dyn_instr']} "
              f"({pct:+.1f}%) vm={m['vm_time_ms']}ms correct={m['correct']}")
        all_rows.append({
            "benchmark": bench_name,
            "config": "fixed_order",
            **m,
            "dyn_pct_reduction": round(pct, 2),
        })

        # --- Best order (from exp_pass_order) ---
        m = run_config(ir_raw, BEST_ORDER, oracle_output, cg, vm)
        pct = 100.0 * (baseline_dyn - m["dyn_instr"]) / baseline_dyn if baseline_dyn > 0 else 0.0
        print(f"  [best_order]:  static={m['static_instr']} dyn={m['dyn_instr']} "
              f"({pct:+.1f}%) vm={m['vm_time_ms']}ms correct={m['correct']}")
        all_rows.append({
            "benchmark": bench_name,
            "config": "best_order",
            **m,
            "dyn_pct_reduction": round(pct, 2),
        })

    # -----------------------------------------------------------------------
    # Summary table: pivot % reduction by benchmark x config
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "trackF_optimized.csv", index=False)
    print(f"\nSaved full data to results/trackF_optimized.csv")

    configs_ordered = SINGLE_PASSES + ["fixed_order", "best_order"]
    bench_names = [p.stem for p in bench_paths]

    print("\n=== Dynamic Instruction Reduction (%) vs Baseline ===")
    print(f"{'Benchmark':<20}", end="")
    for c in configs_ordered:
        label = c[:14]
        print(f"  {label:>16}", end="")
    print()
    print("-" * (20 + 18 * len(configs_ordered)))

    zero_benefit = []  # (config, benchmark) pairs with 0% reduction

    for bname in bench_names:
        print(f"{bname:<20}", end="")
        for c in configs_ordered:
            row = df[(df["benchmark"] == bname) & (df["config"] == c)]
            if row.empty:
                print(f"  {'N/A':>16}", end="")
            else:
                pct = row["dyn_pct_reduction"].values[0]
                if pct == 0.0:
                    zero_benefit.append((c, bname))
                print(f"  {pct:>+15.1f}%", end="")
        print()

    if zero_benefit:
        print("\n*** Passes with ZERO benefit on at least one benchmark: ***")
        for cfg, bname in zero_benefit:
            print(f"  config='{cfg}' gave 0% reduction on '{bname}'")
    else:
        print("\nNo pass gave zero benefit on any benchmark.")

    # -----------------------------------------------------------------------
    # Bar chart: grouped by benchmark, bars per config
    # -----------------------------------------------------------------------
    pivot = df[df["config"] != "baseline"].pivot(
        index="benchmark", columns="config", values="dyn_pct_reduction"
    )
    # Reorder columns
    col_order = [c for c in configs_ordered if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot))
    width = 0.12
    n_configs = len(col_order)
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))
    for i, (col, offset, color) in enumerate(zip(col_order, offsets, colors)):
        vals = pivot[col].fillna(0).values
        ax.bar(x + offset, vals, width, label=col, color=color, edgecolor="white")

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Dynamic instruction reduction (%)")
    ax.set_title("Compiler Pass Ablation: Dynamic Instruction Reduction vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(RESULTS / "trackF_optimized_bar.png", dpi=120)
    plt.close()
    print("\nSaved bar chart to results/trackF_optimized_bar.png")


if __name__ == "__main__":
    main()
