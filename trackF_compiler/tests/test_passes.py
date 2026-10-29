"""test_passes.py - Unit tests for MiniLang IR optimization passes.

Validates:
  - Hand-written before/after IR transformations for each of the 5 passes.
  - Overflow folding (2**62 * 4 wraps to 0 in 64-bit two's complement).
  - Truncated division and modulo folding (-7 / 2 == -3, -7 % 2 == -1).
  - Safety: dead division by a variable that is zero at runtime still raises ZeroDivisionError.
  - Safety: print instructions are never eliminated by dead_code_elim.
  - Safety: LICM does not hoist expressions whose operands are modified inside the loop.
  - Benchmark agreement: Oracle and VM match for all 6 benchmarks under every pass and fixed order.
  - Embedded differential fuzz test (50 random programs).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from trackF_compiler.fuzz import run_fuzz_test
from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.compiler import compile_program
from trackF_compiler.minilang.ir import (
    Binary,
    Const,
    Jump,
    JumpIfZero,
    Label,
    Load,
    PrintIR,
    Store,
    Unary,
)
from trackF_compiler.minilang.passes import (
    FIXED_ORDER_PASSES,
    PASS_REGISTRY,
    constant_fold,
    cse,
    dead_code_elim,
    licm,
    strength_reduce,
)
from trackF_compiler.minilang.vm import VirtualMachine

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_constant_fold_arithmetic_and_propagation():
    """Test constant folding and propagation within a basic block."""
    ir_before = [
        Const("t0", 10),
        Const("t1", 25),
        Binary("t2", "+", "t0", "t1"),
        Store("x", "t2"),
        Load("t3", "x"),
        Const("t4", 2),
        Binary("t5", "*", "t3", "t4"),
    ]
    ir_after = constant_fold(ir_before)

    # t2 should be folded to Const("t2", 35)
    assert Const("t2", 35) in ir_after
    # t3 should propagate constant 35: Const("t3", 35)
    assert Const("t3", 35) in ir_after
    # t5 should be folded to Const("t5", 70)
    assert Const("t5", 70) in ir_after


def test_constant_fold_overflow_wraparound():
    """Verify 64-bit wraparound on overflow: 2**62 * 4 wraps to 0."""
    val_62 = 1 << 62
    ir_before = [
        Const("t0", val_62),
        Const("t1", 4),
        Binary("t2", "*", "t0", "t1"),
    ]
    ir_after = constant_fold(ir_before)
    assert Const("t2", 0) in ir_after


def test_constant_fold_truncated_division_and_modulo():
    """Folding -7 / 2 must give -3 and -7 % 2 must give -1 (truncated C99 semantics)."""
    ir_before = [
        Const("t0", -7),
        Const("t1", 2),
        Binary("t2", "/", "t0", "t1"),
        Binary("t3", "%", "t0", "t1"),
    ]
    ir_after = constant_fold(ir_before)
    assert Const("t2", -3) in ir_after
    assert Const("t3", -1) in ir_after


def test_constant_fold_does_not_fold_division_by_zero():
    """Constant folding must NEVER fold division or modulo by zero."""
    ir_before = [
        Const("t0", 42),
        Const("t1", 0),
        Binary("t2", "/", "t0", "t1"),
        Binary("t3", "%", "t0", "t1"),
    ]
    ir_after = constant_fold(ir_before)
    # Binary instructions must remain so runtime error is preserved
    assert Binary("t2", "/", "t0", "t1") in ir_after
    assert Binary("t3", "%", "t0", "t1") in ir_after


def test_cse_and_store_to_load_forwarding():
    """CSE eliminates identical subexpressions and forwards stored values to subsequent loads."""
    ir_before = [
        Load("t0", "a"),
        Load("t1", "b"),
        Binary("t2", "+", "t0", "t1"),
        Binary("t3", "+", "t1", "t0"),  # Commutative duplicate of t2
        Store("c", "t3"),
        Load("t4", "c"),                # Redundant load immediately after store
        Binary("t5", "*", "t4", "t2"),
    ]
    ir_after = cse(ir_before)

    # Duplicate binary instruction t3 should be eliminated
    bin_ops = [inst for inst in ir_after if isinstance(inst, Binary)]
    assert len(bin_ops) == 2  # Only t2 and t5
    # Load t4 should be eliminated via store-to-load forwarding
    load_ops = [inst for inst in ir_after if isinstance(inst, Load)]
    assert len(load_ops) == 2  # Only loads for a and b


def test_strength_reduce_algebraic_and_shift():
    """Strength reduction simplifies x*0, x*1, x+0, x-0, and transforms x * 8 to x << 3."""
    ir_before = [
        Load("t0", "x"),
        Const("t_zero", 0),
        Binary("t1", "*", "t0", "t_zero"),
        Binary("t2", "+", "t0", "t_zero"),
        Binary("t3", "-", "t0", "t_zero"),
        Const("t_one", 1),
        Binary("t4", "*", "t0", "t_one"),
        Const("t_eight", 8),
        Binary("t5", "*", "t0", "t_eight"),
        PrintIR("t1"),
        PrintIR("t2"),
        PrintIR("t3"),
        PrintIR("t4"),
        PrintIR("t5"),
    ]
    ir_after = strength_reduce(ir_before)

    # t1 should be folded to Const("t1", 0)
    assert Const("t1", 0) in ir_after
    # t5 = t0 * 8 should be transformed into shift << 3
    shl_ops = [inst for inst in ir_after if isinstance(inst, Binary) and inst.op == "<<"]
    assert len(shl_ops) == 1
    assert shl_ops[0].dest == "t5"


def test_dead_code_elim_preserves_print_and_dead_division_by_zero():
    """Dead code elimination removes dead computations but preserves print and potential division traps."""
    # Source with dead division by zero: z is 0 at runtime, result of div is never used
    source = """
    z = 0;
    unused = 100 / z;
    print(999);
    """
    # Oracle raises ZeroDivisionError
    ast_p, _, _, _ = compile_program(source, opt_level=0)
    with pytest.raises(ZeroDivisionError):
        ASTInterpreter().run(ast_p)

    # Under dead_code_elim and under fixed order, VM MUST still raise ZeroDivisionError
    for passes in [["dead_code_elim"], FIXED_ORDER_PASSES]:
        _, _, vm_prog, _ = compile_program(source, passes=passes)
        with pytest.raises(ZeroDivisionError):
            VirtualMachine().run(vm_prog)


def test_dead_code_elim_removes_dead_store_and_unused_inst():
    """Dead stores and unused arithmetic must be eliminated."""
    ir_before = [
        Const("t0", 10),
        Store("x", "t0"),     # Dead store: x is overwritten before being read
        Const("t1", 20),
        Store("x", "t1"),     # Live store: read later
        Const("t2", 999),     # Dead computation: never used
        Load("t3", "x"),
        PrintIR("t3"),
    ]
    ir_after = dead_code_elim(ir_before)

    # First Store("x", t0) must be eliminated
    stores = [inst for inst in ir_after if isinstance(inst, Store)]
    assert len(stores) == 1
    assert stores[0].src == "t1"
    # Const t2 must be eliminated
    assert not any(isinstance(inst, Const) and inst.dest == "t2" for inst in ir_after)
    # PrintIR must remain
    assert any(isinstance(inst, PrintIR) for inst in ir_after)


def test_licm_does_not_hoist_loop_variant_computations():
    """LICM hoists pure invariant computations but does NOT hoist expressions whose operands change in the loop."""
    source = """
    n = 10;
    i = 0;
    k = 5;
    while (i < n) {
        inv = k * 7;
        variant = i * 2;
        print(inv);
        print(variant);
        i = i + 1;
    }
    """
    _, ir_raw, _, _ = compile_program(source, opt_level=0)
    ir_opt = licm(ir_raw)

    # Find loop label
    loop_label_idx = -1
    for idx, inst in enumerate(ir_opt):
        if isinstance(inst, Label) and inst.name.startswith("L_loop"):
            loop_label_idx = idx
            break

    assert loop_label_idx != -1

    preheader_insts = ir_opt[:loop_label_idx]
    loop_body_insts = ir_opt[loop_label_idx:]

    # The invariant computation (k * 7) and load of k should be hoisted into preheader
    assert any(isinstance(inst, Binary) and inst.op == "*" for inst in preheader_insts)

    # The variant computation (i * 2) must remain INSIDE the loop body
    variant_in_body = any(
        isinstance(inst, Binary) and inst.op == "*" for inst in loop_body_insts
    )
    assert variant_in_body, "Loop-variant computation i * 2 was incorrectly hoisted!"


def test_all_benchmarks_match_oracle_under_all_passes():
    """Verify that Oracle and VM outputs match across all 6 benchmarks for every single pass and fixed order."""
    programs_dir = REPO_ROOT / "trackF_compiler" / "programs"
    benchmarks = sorted(programs_dir.glob("*.ml"))
    assert len(benchmarks) == 6, f"Expected 6 benchmark files, found {len(benchmarks)}"

    test_configs = [[p] for p in PASS_REGISTRY.keys()] + [FIXED_ORDER_PASSES]

    for b_path in benchmarks:
        source = b_path.read_text(encoding="utf-8")
        ast_prog, _, _, _ = compile_program(source, opt_level=0)
        oracle_out = ASTInterpreter().run(ast_prog)

        for p_config in test_configs:
            _, _, vm_prog, _ = compile_program(source, passes=p_config)
            vm_out, _ = VirtualMachine().run(vm_prog)
            assert vm_out == oracle_out, (
                f"Mismatch on benchmark {b_path.name} with passes {p_config}!\n"
                f"Oracle: {oracle_out}\n"
                f"VM:     {vm_out}"
            )


def test_embedded_fuzz_50_programs():
    """Run an automated differential fuzz test on 50 programs as a pytest assertion."""
    report = run_fuzz_test(n_programs=50, start_seed=100)
    assert report["status"] == "ALL_PASSED"
    assert report["programs_tested"] == 50
