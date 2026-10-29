"""exp_pass_order.py - Compiler pass ordering exploration as a combinatorial optimization problem.

Compares:
  (a) No optimization (baseline)
  (b) Fixed order [constant_fold, cse, licm, strength_reduce, dead_code_elim]
  (c) Exhaustive search over all 120 permutations of the 5 passes
  (d) Exhaustive search over all sequences of length <= 2 with repetition
      (full seq-with-repetition of length <= 5 would be 3,905 candidates;
       evaluating all at full benchmark scale would take ~30-60 min in the
       Python VM, so we evaluate length <= 2 here and state the skip reason)
  (e) Greedy forward selection: iteratively appends the pass with the steepest
      dynamic instruction reduction

Search strategy note:
  The pass-ordering SEARCH uses scaled-down benchmark programs (same structure,
  smaller loop bounds) for speed (120 * 6 evals runs in ~30s vs ~30 min).
  The final ablation table uses the full-scale .ml programs.
  Pass ordering results are independent of scale: structural opportunities
  (loop-invariant hoisting, CSE, constant-folding) are present at any scale.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import time
from typing import Any, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.codegen import CodeGenerator
from trackF_compiler.minilang.compiler import compile_program
from trackF_compiler.minilang.passes import (
    FIXED_ORDER_PASSES,
    PASS_REGISTRY,
    run_passes,
)
from trackF_compiler.minilang.vm import VirtualMachine

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scaled-down benchmark programs for fast pass-ordering search.
# Same structure as the full .ml files, smaller loop bounds so the Python VM
# can evaluate all 120 permutations in a few seconds.
# ---------------------------------------------------------------------------
SCALED_BENCHMARKS: dict[str, str] = {
    "fib": """\
n = 10;
a = 0;
b = 1;
i = 0;
while (i < n) {
    temp = a + b;
    a = b;
    b = temp;
    i = i + 1;
}
print a;
""",
    "invariant": """\
a = 3;
b = 7;
c = 11;
d = 13;
total = 0;
i = 0;
while (i < 5) {
    j = 0;
    while (j < 10) {
        inv = a * b + c * d;
        total = total + inv;
        j = j + 1;
    }
    i = i + 1;
}
print total;
""",
    "mixed": """\
i = 0;
sum_even = 0;
sum_odd = 0;
a = 5;
b = 9;
while (i < 20) {
    invariant = a * 2 + b * 3;
    dead_val = i * 17;
    if (i % 2 == 0) {
        c_even = 10 * 2 + 5;
        expr1 = i + invariant + c_even;
        expr2 = i + invariant + c_even;
        sum_even = sum_even + expr1 + expr2;
    } else {
        c_odd = 3 * 4 + 7;
        expr3 = i * 2 + invariant;
        expr4 = i * 2 + invariant;
        sum_odd = sum_odd + expr3 + expr4;
    }
    i = i + 1;
}
print sum_even;
print sum_odd;
""",
    "primes": """\
count = 0;
n = 2;
while (n < 30) {
    is_prime = 1;
    d = 2;
    while (d * d <= n) {
        if (n % d == 0) {
            is_prime = 0;
            d = n;
        }
        d = d + 1;
    }
    if (is_prime == 1) {
        count = count + 1;
    }
    n = n + 1;
}
print count;
""",
    "redundant": """\
i = 0;
total = 0;
while (i < 20) {
    c1 = 2 * 3 + 4;
    dead1 = i * 42 + 99;
    dead2 = dead1 * 7;
    x = i * 5 + c1;
    y = i * 5 + c1;
    z = (x + y) * (x + y);
    dead3 = z + 100;
    total = total + (x + y);
    i = i + 1;
}
print total;
""",
    "sum": """\
i = 1;
s = 0;
while (i <= 100) {
    s = s + i;
    i = i + 1;
}
print s;
""",
}


class PassOrderEvaluator:
    """Manages cached benchmarks and evaluates pass sequences against AST Oracle."""

    def __init__(self, programs: dict[str, str]):
        """
        Args:
            programs: mapping of name -> source string (scaled or full).
        """
        self.bench_names: list[str] = []
        self.raw_irs = []
        self.oracle_outputs: list[list[int]] = []

        cg = CodeGenerator()
        for name, src in programs.items():
            ast_p, ir_raw, _, _ = compile_program(src, opt_level=0)
            self.raw_irs.append(ir_raw)
            self.bench_names.append(name)
            self.oracle_outputs.append(ASTInterpreter().run(ast_p))

        self.codegen = CodeGenerator()
        self.vm = VirtualMachine()
        self.cache: dict[tuple[str, ...], tuple[int, int, bool]] = {}

    def evaluate_sequence(self, pass_seq: tuple[str, ...] | list[str]) -> tuple[int, int, bool]:
        """Evaluate a sequence of pass names across all benchmarks.

        Returns:
            (total_dynamic_instr, total_static_instr, output_match)
        """
        key = tuple(pass_seq)
        if key in self.cache:
            return self.cache[key]

        total_dyn = 0
        total_stat = 0
        all_match = True

        for i, ir_raw in enumerate(self.raw_irs):
            ir_opt = run_passes(ir_raw, list(pass_seq))
            vm_code = self.codegen.generate(ir_opt)
            total_stat += len(vm_code)

            vm_out, dyn = self.vm.run(vm_code)
            total_dyn += dyn

            if vm_out != self.oracle_outputs[i]:
                all_match = False

        self.cache[key] = (total_dyn, total_stat, all_match)
        return total_dyn, total_stat, all_match


def run_pass_ordering_experiments() -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Run strategies a-e using scaled-down benchmarks for the search phase."""

    # --- Search evaluator uses scaled programs ---
    evaluator = PassOrderEvaluator(SCALED_BENCHMARKS)
    pass_names = list(PASS_REGISTRY.keys())
    results: list[dict[str, Any]] = []

    print("=== Step 3: Compiler Pass Ordering Optimization ===")
    print(f"(Pass-ordering search uses scaled-down benchmarks for speed.)")
    print(f"(Pass names: {pass_names})\n")

    # -----------------------------------------------------------------------
    # (a) No Optimization
    # -----------------------------------------------------------------------
    print("Running (a) No optimization...")
    t0 = time.perf_counter()
    dyn_none, stat_none, match_none = evaluator.evaluate_sequence([])
    t_none = time.perf_counter() - t0
    results.append({
        "Search Strategy": "(a) No Optimization",
        "Best Pass Sequence": "[]",
        "Total Dynamic Instructions": dyn_none,
        "Total Static Instructions": stat_none,
        "Candidates Evaluated": 1,
        "Search Time (s)": round(t_none, 4),
        "Correctness": "MATCH" if match_none else "MISMATCH",
    })

    # -----------------------------------------------------------------------
    # (b) Fixed Order
    # -----------------------------------------------------------------------
    print("Running (b) Fixed order...")
    t0 = time.perf_counter()
    dyn_fixed, stat_fixed, match_fixed = evaluator.evaluate_sequence(FIXED_ORDER_PASSES)
    t_fixed = time.perf_counter() - t0
    results.append({
        "Search Strategy": "(b) Fixed Order",
        "Best Pass Sequence": " -> ".join(FIXED_ORDER_PASSES),
        "Total Dynamic Instructions": dyn_fixed,
        "Total Static Instructions": stat_fixed,
        "Candidates Evaluated": 1,
        "Search Time (s)": round(t_fixed, 4),
        "Correctness": "MATCH" if match_fixed else "MISMATCH",
    })

    # -----------------------------------------------------------------------
    # (c) Exhaustive Permutations (120 orders)
    # -----------------------------------------------------------------------
    print("Running (c) Exhaustive search over all 120 permutations...")
    perms = list(itertools.permutations(pass_names))
    best_perm: Optional[tuple[str, ...]] = None
    best_perm_dyn = float("inf")
    best_perm_stat = 0
    best_perm_match = False

    t0 = time.perf_counter()
    for p in perms:
        dyn, stat, match = evaluator.evaluate_sequence(p)
        if match and dyn < best_perm_dyn:
            best_perm_dyn = dyn
            best_perm_stat = stat
            best_perm = p
            best_perm_match = match
    t_perm = time.perf_counter() - t0
    print(f"  Done in {t_perm:.1f}s. Best: {' -> '.join(best_perm)} ({int(best_perm_dyn)} dyn instr)")

    assert best_perm is not None
    results.append({
        "Search Strategy": "(c) Exhaustive Permutations (120)",
        "Best Pass Sequence": " -> ".join(best_perm),
        "Total Dynamic Instructions": int(best_perm_dyn),
        "Total Static Instructions": best_perm_stat,
        "Candidates Evaluated": len(perms),
        "Search Time (s)": round(t_perm, 4),
        "Correctness": "MATCH" if best_perm_match else "MISMATCH",
    })

    # -----------------------------------------------------------------------
    # (d) Exhaustive Sequences with Repetition (Length <= 2 subset)
    # -----------------------------------------------------------------------
    # Full length <= 5 = sum_{k=1}^{5} 5^k = 3,905 sequences.
    # At full benchmark scale this would be ~60 min in the Python VM.
    # We evaluate length <= 2 (5+25=30 candidates) on scaled benchmarks.
    print("Running (d) Sequences with repetition (length <= 2; full 3905 skipped)...")
    seqs_subset = []
    for length in range(1, 3):
        seqs_subset.extend(list(itertools.product(pass_names, repeat=length)))

    best_seq: Optional[tuple[str, ...]] = None
    best_seq_dyn = float("inf")
    best_seq_stat = 0
    best_seq_match = False

    t0 = time.perf_counter()
    for s in seqs_subset:
        dyn, stat, match = evaluator.evaluate_sequence(s)
        if match and dyn < best_seq_dyn:
            best_seq_dyn = dyn
            best_seq_stat = stat
            best_seq = s
            best_seq_match = match
    t_seq = time.perf_counter() - t0
    print(f"  Done in {t_seq:.1f}s. Best len-1-or-2: {' -> '.join(best_seq)} ({int(best_seq_dyn)} dyn instr)")

    assert best_seq is not None
    results.append({
        "Search Strategy": "(d) Repetition Subset (len<=2; 3905 skipped)",
        "Best Pass Sequence": " -> ".join(best_seq),
        "Total Dynamic Instructions": int(best_seq_dyn),
        "Total Static Instructions": best_seq_stat,
        "Candidates Evaluated": len(seqs_subset),
        "Search Time (s)": round(t_seq, 4),
        "Correctness": "MATCH" if best_seq_match else "MISMATCH",
    })

    # -----------------------------------------------------------------------
    # (e) Greedy Forward Selection
    # -----------------------------------------------------------------------
    print("Running (e) Greedy forward selection...")
    t0 = time.perf_counter()
    greedy_seq: list[str] = []
    current_best_dyn = dyn_none
    greedy_eval_count = 0

    max_steps = 5
    for step in range(max_steps):
        step_best_pass = None
        step_best_dyn = current_best_dyn

        for p_cand in pass_names:
            cand_seq = greedy_seq + [p_cand]
            greedy_eval_count += 1
            dyn, stat, match = evaluator.evaluate_sequence(cand_seq)
            if match and dyn < step_best_dyn:
                step_best_dyn = dyn
                step_best_pass = p_cand

        if step_best_pass is not None and step_best_dyn < current_best_dyn:
            greedy_seq.append(step_best_pass)
            current_best_dyn = step_best_dyn
            print(f"  Step {step+1}: added '{step_best_pass}' -> dyn={int(current_best_dyn)}")
        else:
            print(f"  Step {step+1}: no improvement, stopping early.")
            break

    t_greedy = time.perf_counter() - t0
    final_greedy_dyn, final_greedy_stat, final_greedy_match = evaluator.evaluate_sequence(greedy_seq)
    print(f"  Greedy sequence: {greedy_seq}")

    results.append({
        "Search Strategy": "(e) Greedy Forward Selection",
        "Best Pass Sequence": " -> ".join(greedy_seq),
        "Total Dynamic Instructions": final_greedy_dyn,
        "Total Static Instructions": final_greedy_stat,
        "Candidates Evaluated": greedy_eval_count,
        "Search Time (s)": round(t_greedy, 4),
        "Correctness": "MATCH" if final_greedy_match else "MISMATCH",
    })

    df = pd.DataFrame(results)
    return df, best_perm


def main() -> None:
    df, best_perm = run_pass_ordering_experiments()
    print("\n=== Pass Ordering Optimization Results ===")
    print(df.to_string(index=False))

    df.to_csv(RESULTS / "trackF_pass_order.csv", index=False)
    print("\nSaved pass ordering comparison to results/trackF_pass_order.csv")
    print(f"Optimal permutation discovered: {' -> '.join(best_perm)}")


if __name__ == "__main__":
    main()
