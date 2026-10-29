"""fuzz.py - Randomized differential testing fuzzer for MiniLang compiler optimizations.

Generates complex syntactically valid MiniLang programs containing nested conditionals,
bounded while-loops, arithmetic expressions, constant and variable stores/loads,
and deliberate division-by-zero edge cases.

Compares execution outputs and runtime error behavior between the AST Oracle
and the Virtual Machine under:
  - Each individual pass in isolation (ablation)
  - The standard fixed optimization order [constant_fold, cse, licm, strength_reduce, dead_code_elim]
"""

from __future__ import annotations

import random
import sys
import time
from typing import Any, Optional

from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.compiler import compile_program
from trackF_compiler.minilang.passes import FIXED_ORDER_PASSES, PASS_REGISTRY
from trackF_compiler.minilang.vm import VirtualMachine


class RandomProgramGenerator:
    """Generates random terminating MiniLang programs with bounded loops and arithmetic."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.var_names = ["a", "b", "c", "d", "x", "y"]
        self.loop_counter = 0

    def gen_expr(self, depth: int = 0, allow_zero_div: bool = False) -> str:
        """Generate a random arithmetic expression."""
        if depth >= 3 or self.rng.random() < 0.4:
            # Leaf: integer constant or variable
            if self.rng.random() < 0.6:
                return str(self.rng.randint(-50, 100))
            else:
                return self.rng.choice(self.var_names)

        op = self.rng.choice(["+", "-", "*", "/", "%", "<", "==", "!="])

        if op in ("/", "%"):
            left = self.gen_expr(depth + 1, allow_zero_div)
            if allow_zero_div and self.rng.random() < 0.15:
                right = "0"
            else:
                # Guaranteed nonzero divisor by construction
                divisor = self.rng.randint(1, 15)
                right = str(divisor)
            return f"({left} {op} {right})"
        else:
            left = self.gen_expr(depth + 1, allow_zero_div)
            right = self.gen_expr(depth + 1, allow_zero_div)
            return f"({left} {op} {right})"

    def gen_statement(self, depth: int = 0) -> list[str]:
        """Generate a random MiniLang statement or structured control block."""
        lines = []
        action = self.rng.random()

        if action < 0.45 or depth >= 2:
            # Assignment
            var = self.rng.choice(self.var_names)
            expr = self.gen_expr(depth=0, allow_zero_div=(self.rng.random() < 0.08))
            lines.append(f"{var} = {expr};")

        elif action < 0.65:
            # Print
            expr = self.gen_expr(depth=0, allow_zero_div=False)
            lines.append(f"print({expr});")

        elif action < 0.85:
            # If / Else block
            cond = self.gen_expr(depth=1, allow_zero_div=False)
            lines.append(f"if ({cond}) {{")
            for _ in range(self.rng.randint(1, 2)):
                for s in self.gen_statement(depth + 1):
                    lines.append(f"    {s}")
            if self.rng.random() < 0.5:
                lines.append("} else {")
                for _ in range(self.rng.randint(1, 2)):
                    for s in self.gen_statement(depth + 1):
                        lines.append(f"    {s}")
            lines.append("}")

        else:
            # Strictly bounded while-loop (guaranteed to terminate)
            loop_var = f"iter_{self.loop_counter}"
            self.loop_counter += 1
            bound = self.rng.randint(2, 4)
            lines.append(f"{loop_var} = 0;")
            lines.append(f"while ({loop_var} < {bound}) {{")
            for _ in range(self.rng.randint(1, 2)):
                for s in self.gen_statement(depth + 1):
                    lines.append(f"    {s}")
            lines.append(f"    {loop_var} = {loop_var} + 1;")
            lines.append("}")

        return lines

    def generate(self) -> str:
        """Generate a complete MiniLang source program."""
        lines = []
        # Initialize all variables to avoid uninitialized variable reads
        for v in self.var_names:
            init_val = self.rng.randint(1, 10)
            lines.append(f"{v} = {init_val};")

        num_stmts = self.rng.randint(4, 9)
        for _ in range(num_stmts):
            lines.extend(self.gen_statement(depth=0))

        # Always end with printing state variables
        for v in self.var_names:
            lines.append(f"print({v});")

        return "\n".join(lines)


def run_fuzz_test(n_programs: int = 300, start_seed: int = 0) -> dict[str, Any]:
    """Execute differential fuzz testing over n_programs random MiniLang programs."""
    print(f"Starting MiniLang differential fuzzing ({n_programs} programs, seed {start_seed})...")
    t0 = time.perf_counter()

    passes_to_test = [[p] for p in PASS_REGISTRY.keys()] + [FIXED_ORDER_PASSES]

    tested_count = 0
    zero_div_count = 0
    clean_runs_count = 0

    for i in range(n_programs):
        seed = start_seed + i
        gen = RandomProgramGenerator(seed)
        source = gen.generate()

        # 1. Run on AST Oracle
        oracle = ASTInterpreter()
        oracle_error: Optional[str] = None
        oracle_out: list[int] = []

        try:
            ast_prog, _, _, _ = compile_program(source, opt_level=0)
            oracle_out = oracle.run(ast_prog)
            clean_runs_count += 1
        except ZeroDivisionError as e:
            oracle_error = "ZeroDivisionError"
            zero_div_count += 1
        except Exception as e:
            oracle_error = type(e).__name__

        # 2. Test each optimization configuration on VM
        for p_config in passes_to_test:
            config_name = "+".join(p_config)
            try:
                _, _, vm_prog, _ = compile_program(source, passes=p_config)
                vm = VirtualMachine()
                vm_out, _ = vm.run(vm_prog)
                vm_error = None
            except ZeroDivisionError:
                vm_error = "ZeroDivisionError"
                vm_out = []
            except Exception as e:
                vm_error = type(e).__name__
                vm_out = []

            # Differential assertion
            if oracle_error is not None:
                if vm_error != oracle_error:
                    print(f"\n[FAIL] Fuzzer detected error discrepancy at seed {seed}!", file=sys.stderr)
                    print(f"Passes: {config_name}", file=sys.stderr)
                    print(f"Oracle Error: {oracle_error}", file=sys.stderr)
                    print(f"VM Error:     {vm_error}", file=sys.stderr)
                    print(f"VM Output:    {vm_out}", file=sys.stderr)
                    print("\nSource code:\n" + source, file=sys.stderr)
                    raise AssertionError(f"Fuzz test failed at seed {seed} with config {config_name}")
            else:
                if vm_error is not None:
                    print(f"\n[FAIL] VM raised unexpected error at seed {seed}!", file=sys.stderr)
                    print(f"Passes: {config_name}", file=sys.stderr)
                    print(f"Oracle succeeded with output: {oracle_out}", file=sys.stderr)
                    print(f"VM Error: {vm_error}", file=sys.stderr)
                    print("\nSource code:\n" + source, file=sys.stderr)
                    raise AssertionError(f"VM crashed with {vm_error} at seed {seed}")

                if vm_out != oracle_out:
                    print(f"\n[FAIL] Output mismatch at seed {seed}!", file=sys.stderr)
                    print(f"Passes: {config_name}", file=sys.stderr)
                    print(f"Oracle Output ({len(oracle_out)} items): {oracle_out}", file=sys.stderr)
                    print(f"VM Output     ({len(vm_out)} items): {vm_out}", file=sys.stderr)
                    print("\nSource code:\n" + source, file=sys.stderr)
                    raise AssertionError(f"Output mismatch at seed {seed} under passes {config_name}")

        tested_count += 1
        if tested_count % 50 == 0:
            print(f"  Verified {tested_count}/{n_programs} programs... ({time.perf_counter() - t0:.2f}s elapsed)")

    total_time = time.perf_counter() - t0
    report = {
        "programs_tested": tested_count,
        "clean_terminations": clean_runs_count,
        "division_by_zero_traps": zero_div_count,
        "passes_configurations_evaluated": len(passes_to_test),
        "total_evaluations": tested_count * len(passes_to_test),
        "fuzz_duration_s": round(total_time, 3),
        "status": "ALL_PASSED",
    }
    return report


def main() -> None:
    report = run_fuzz_test(n_programs=300, start_seed=42)
    print("\n=== Differential Fuzzing Summary Report ===")
    for k, v in report.items():
        print(f"  {k:<35}: {v}")
    print("\nResult: 100% of fuzzed programs verified identical to AST Oracle under all optimization passes.")


if __name__ == "__main__":
    main()
