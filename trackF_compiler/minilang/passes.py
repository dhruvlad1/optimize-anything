"""passes.py - Intermediate Representation (IR) optimization passes for MiniLang.

Implements five semantics-preserving optimization passes:
  1. constant_fold: Intra-block constant evaluation and propagation using 64-bit integer semantics.
  2. cse: Common subexpression elimination and store-to-load forwarding.
  3. licm: Loop-invariant code motion for while loops hoisting pure operations to a preheader.
  4. strength_reduce: Algebraic simplifications and shift equivalents for x * 2^k.
  5. dead_code_elim: Global CFG liveness analysis eliminating unused values and dead stores.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple
from trackF_compiler.minilang.ir import (
    Binary,
    Const,
    IRInst,
    Jump,
    JumpIfNonZero,
    JumpIfZero,
    Label,
    Load,
    PrintIR,
    Store,
    Unary,
)
from trackF_compiler.minilang.semantics import (
    apply_binary_op,
    apply_unary_op,
    wrap64,
)


# ---------------------------------------------------------------------------
# CFG and Basic Block Utilities
# ---------------------------------------------------------------------------

class BasicBlock:
    def __init__(self, block_id: int):
        self.block_id: int = block_id
        self.instructions: list[IRInst] = []
        self.predecessors: set[int] = set()
        self.successors: set[int] = set()

    def __repr__(self) -> str:
        return f"Block_{self.block_id}(len={len(self.instructions)})"


def split_into_basic_blocks(instructions: list[IRInst]) -> list[BasicBlock]:
    """Partition a linear sequence of TAC instructions into basic blocks."""
    if not instructions:
        return []

    # Find leader indices
    leaders = {0}
    for i, inst in enumerate(instructions):
        if isinstance(inst, Label):
            leaders.add(i)
        elif isinstance(inst, (Jump, JumpIfZero, JumpIfNonZero)):
            if i + 1 < len(instructions):
                leaders.add(i + 1)

    sorted_leaders = sorted(leaders)
    blocks: list[BasicBlock] = []
    leader_to_block: dict[int, int] = {}
    label_to_block: dict[str, int] = {}

    for b_idx, start in enumerate(sorted_leaders):
        end = sorted_leaders[b_idx + 1] if b_idx + 1 < len(sorted_leaders) else len(instructions)
        block = BasicBlock(b_idx)
        block.instructions = instructions[start:end]
        blocks.append(block)
        leader_to_block[start] = b_idx
        for inst in block.instructions:
            if isinstance(inst, Label):
                label_to_block[inst.name] = b_idx

    # Build CFG edges (predecessors and successors)
    for b_idx, block in enumerate(blocks):
        if not block.instructions:
            continue
        last = block.instructions[-1]

        if isinstance(last, Jump):
            target_b = label_to_block.get(last.target)
            if target_b is not None:
                block.successors.add(target_b)
                blocks[target_b].predecessors.add(b_idx)
        elif isinstance(last, (JumpIfZero, JumpIfNonZero)):
            target_b = label_to_block.get(last.target)
            if target_b is not None:
                block.successors.add(target_b)
                blocks[target_b].predecessors.add(b_idx)
            # Fall-through to next block
            if b_idx + 1 < len(blocks):
                block.successors.add(b_idx + 1)
                blocks[b_idx + 1].predecessors.add(b_idx)
        else:
            # Normal fall-through
            if b_idx + 1 < len(blocks):
                block.successors.add(b_idx + 1)
                blocks[b_idx + 1].predecessors.add(b_idx)

    return blocks


def apply_reg_aliases(insts: list[IRInst], reg_alias: dict[str, str]) -> list[IRInst]:
    """Substitute all register references according to reg_alias mapping."""
    if not reg_alias:
        return insts

    def resolve(r: str) -> str:
        curr = r
        visited = set()
        while curr in reg_alias and curr not in visited:
            visited.add(curr)
            curr = reg_alias[curr]
        return curr

    res: list[IRInst] = []
    for inst in insts:
        if isinstance(inst, Store):
            res.append(Store(inst.var_name, resolve(inst.src)))
        elif isinstance(inst, PrintIR):
            res.append(PrintIR(resolve(inst.src)))
        elif isinstance(inst, JumpIfZero):
            res.append(JumpIfZero(resolve(inst.cond), inst.target))
        elif isinstance(inst, JumpIfNonZero):
            res.append(JumpIfNonZero(resolve(inst.cond), inst.target))
        elif isinstance(inst, Unary):
            res.append(Unary(inst.dest, inst.op, resolve(inst.src)))
        elif isinstance(inst, Binary):
            res.append(Binary(inst.dest, inst.op, resolve(inst.src1), resolve(inst.src2)))
        else:
            res.append(inst)
    return res


# ---------------------------------------------------------------------------
# 1. Constant Folding & Propagation
# ---------------------------------------------------------------------------

def constant_fold(instructions: list[IRInst]) -> list[IRInst]:
    """Fold constant arithmetic and propagate constants within basic blocks.

    Enforces:
      - Uses wrap64, div_trunc, mod_trunc (via apply_binary_op/apply_unary_op).
      - Never folds a division or modulo whose divisor is zero or unknown.
    """
    blocks = split_into_basic_blocks(instructions)
    optimized_insts: list[IRInst] = []

    for block in blocks:
        const_regs: dict[str, int] = {}
        const_vars: dict[str, int] = {}

        for inst in block.instructions:
            if isinstance(inst, Const):
                val = wrap64(inst.value)
                const_regs[inst.dest] = val
                optimized_insts.append(Const(inst.dest, val))

            elif isinstance(inst, Load):
                if inst.var_name in const_vars:
                    val = const_vars[inst.var_name]
                    const_regs[inst.dest] = val
                    optimized_insts.append(Const(inst.dest, val))
                else:
                    const_regs.pop(inst.dest, None)
                    optimized_insts.append(inst)

            elif isinstance(inst, Store):
                if inst.src in const_regs:
                    const_vars[inst.var_name] = const_regs[inst.src]
                else:
                    const_vars.pop(inst.var_name, None)
                optimized_insts.append(inst)

            elif isinstance(inst, Unary):
                if inst.src in const_regs:
                    val = apply_unary_op(inst.op, const_regs[inst.src])
                    const_regs[inst.dest] = val
                    optimized_insts.append(Const(inst.dest, val))
                else:
                    const_regs.pop(inst.dest, None)
                    optimized_insts.append(inst)

            elif isinstance(inst, Binary):
                c1 = const_regs.get(inst.src1)
                c2 = const_regs.get(inst.src2)

                if c1 is not None and c2 is not None:
                    # Do NOT fold division or modulo with zero divisor
                    if inst.op == "<<":
                        val = wrap64(c1 << c2) if 0 <= c2 < 64 else 0
                        const_regs[inst.dest] = val
                        optimized_insts.append(Const(inst.dest, val))
                    elif inst.op in ("/", "%") and c2 == 0:
                        const_regs.pop(inst.dest, None)
                        optimized_insts.append(inst)
                    else:
                        val = apply_binary_op(inst.op, c1, c2)
                        const_regs[inst.dest] = val
                        optimized_insts.append(Const(inst.dest, val))
                else:
                    const_regs.pop(inst.dest, None)
                    optimized_insts.append(inst)

            else:
                optimized_insts.append(inst)

    return optimized_insts


# ---------------------------------------------------------------------------
# 2. Common Subexpression Elimination (CSE) & Store-to-Load Forwarding
# ---------------------------------------------------------------------------

def cse(instructions: list[IRInst]) -> list[IRInst]:
    """Common subexpression elimination and store-to-load forwarding within basic blocks."""
    blocks = split_into_basic_blocks(instructions)
    optimized_insts: list[IRInst] = []
    global_reg_alias: dict[str, str] = {}

    def resolve(r: str) -> str:
        curr = r
        visited = set()
        while curr in global_reg_alias and curr not in visited:
            visited.add(curr)
            curr = global_reg_alias[curr]
        return curr

    for block in blocks:
        # Map: expr_key -> destination register that already computed it (block-local)
        expr_table: dict[tuple, str] = {}
        # Map: variable_name -> register holding its most recently stored value (block-local)
        last_stored: dict[str, str] = {}

        for inst in block.instructions:
            if isinstance(inst, Const):
                expr_key = ("CONST", wrap64(inst.value))
                if expr_key in expr_table:
                    global_reg_alias[inst.dest] = expr_table[expr_key]
                else:
                    expr_table[expr_key] = inst.dest
                    optimized_insts.append(Const(inst.dest, wrap64(inst.value)))

            elif isinstance(inst, Store):
                src = resolve(inst.src)
                last_stored[inst.var_name] = src
                # Invalidate any cached LOAD for this variable since it was updated
                expr_table.pop(("LOAD", inst.var_name), None)
                optimized_insts.append(Store(inst.var_name, src))

            elif isinstance(inst, Load):
                if inst.var_name in last_stored:
                    # Store-to-load forwarding!
                    global_reg_alias[inst.dest] = last_stored[inst.var_name]
                else:
                    expr_key = ("LOAD", inst.var_name)
                    if expr_key in expr_table:
                        global_reg_alias[inst.dest] = expr_table[expr_key]
                    else:
                        expr_table[expr_key] = inst.dest
                        optimized_insts.append(Load(inst.dest, inst.var_name))

            elif isinstance(inst, Binary):
                s1 = resolve(inst.src1)
                s2 = resolve(inst.src2)

                # Canonicalize commutative operations (+, *, ==, !=)
                if inst.op in ("+", "*", "==", "!="):
                    operands = tuple(sorted([s1, s2]))
                else:
                    operands = (s1, s2)

                expr_key = ("BINARY", inst.op, operands)
                if expr_key in expr_table:
                    global_reg_alias[inst.dest] = expr_table[expr_key]
                else:
                    expr_table[expr_key] = inst.dest
                    optimized_insts.append(Binary(inst.dest, inst.op, s1, s2))

            elif isinstance(inst, Unary):
                s = resolve(inst.src)
                expr_key = ("UNARY", inst.op, s)
                if expr_key in expr_table:
                    global_reg_alias[inst.dest] = expr_table[expr_key]
                else:
                    expr_table[expr_key] = inst.dest
                    optimized_insts.append(Unary(inst.dest, inst.op, s))

            elif isinstance(inst, PrintIR):
                optimized_insts.append(PrintIR(resolve(inst.src)))

            elif isinstance(inst, JumpIfZero):
                optimized_insts.append(JumpIfZero(resolve(inst.cond), inst.target))

            elif isinstance(inst, JumpIfNonZero):
                optimized_insts.append(JumpIfNonZero(resolve(inst.cond), inst.target))

            else:
                optimized_insts.append(inst)

    return apply_reg_aliases(optimized_insts, global_reg_alias)


# ---------------------------------------------------------------------------
# 3. Loop-Invariant Code Motion (LICM)
# ---------------------------------------------------------------------------

def licm(instructions: list[IRInst]) -> list[IRInst]:
    """Hoist pure computations whose operands are not modified inside while loops into a preheader.

    Never hoists division or modulo unless the divisor is a nonzero constant.
    """
    # Identify loops: In MiniLang, while loops are delimited by L_loop_X and L_end_X
    # Look for matching pairs of Label(L_loop_*) and Jump(L_loop_*)
    n = len(instructions)
    loop_regions: list[tuple[int, int, str]] = []

    for i, inst in enumerate(instructions):
        if isinstance(inst, Label) and inst.name.startswith("L_loop"):
            loop_lbl = inst.name
            # Find the back-edge Jump(loop_lbl)
            for j in range(i + 1, n):
                if isinstance(instructions[j], Jump) and instructions[j].target == loop_lbl:
                    loop_regions.append((i, j, loop_lbl))
                    break

    if not loop_regions:
        return list(instructions)

    result = list(instructions)

    # Process innermost / smallest loops first (sorted by length ascending)
    loop_regions.sort(key=lambda t: t[1] - t[0])

    for start_idx, end_idx, loop_lbl in loop_regions:
        loop_sub = result[start_idx : end_idx + 1]

        # Find variables assigned inside loop
        assigned_vars = {inst.var_name for inst in loop_sub if isinstance(inst, Store)}

        # Find registers defined outside the loop
        outside_regs = set()
        for idx in range(start_idx):
            inst = result[idx]
            if hasattr(inst, "dest"):
                outside_regs.add(getattr(inst, "dest"))

        # Find known constant values available before the loop
        outside_consts = {}
        for idx in range(start_idx):
            inst = result[idx]
            if isinstance(inst, Const):
                outside_consts[inst.dest] = inst.value

        # Iteratively identify hoistable instructions within the loop body
        hoisted_indices: set[int] = set()
        hoisted_regs: set[str] = set()

        changed = True
        while changed:
            changed = False
            for k, inst in enumerate(loop_sub):
                if k in hoisted_indices:
                    continue
                # Cannot hoist branches, stores, prints, labels
                if isinstance(inst, (Label, Jump, JumpIfZero, JumpIfNonZero, Store, PrintIR)):
                    continue

                if isinstance(inst, Const):
                    hoisted_indices.add(k)
                    hoisted_regs.add(inst.dest)
                    outside_consts[inst.dest] = inst.value
                    changed = True

                elif isinstance(inst, Load):
                    if inst.var_name not in assigned_vars:
                        hoisted_indices.add(k)
                        hoisted_regs.add(inst.dest)
                        changed = True

                elif isinstance(inst, Unary):
                    if inst.src in outside_regs or inst.src in hoisted_regs:
                        hoisted_indices.add(k)
                        hoisted_regs.add(inst.dest)
                        changed = True

                elif isinstance(inst, Binary):
                    s1_ok = inst.src1 in outside_regs or inst.src1 in hoisted_regs
                    s2_ok = inst.src2 in outside_regs or inst.src2 in hoisted_regs

                    if s1_ok and s2_ok:
                        # Safety check: division and modulo must have nonzero constant divisor
                        if inst.op in ("/", "%"):
                            c2 = outside_consts.get(inst.src2)
                            if c2 is not None and c2 != 0:
                                hoisted_indices.add(k)
                                hoisted_regs.add(inst.dest)
                                changed = True
                        else:
                            hoisted_indices.add(k)
                            hoisted_regs.add(inst.dest)
                            changed = True

        if hoisted_indices:
            # Extract hoisted instructions in their original relative order
            hoisted_insts = [loop_sub[k] for k in sorted(hoisted_indices)]
            remaining_loop = [loop_sub[k] for k in range(len(loop_sub)) if k not in hoisted_indices]

            # Reconstruct result: place hoisted_insts immediately before start_idx (preheader)
            result = result[:start_idx] + hoisted_insts + remaining_loop + result[end_idx + 1 :]
            # Update subsequent loop regions offsets if needed
            return licm(result)

    return result


# ---------------------------------------------------------------------------
# 4. Strength Reduction
# ---------------------------------------------------------------------------

def strength_reduce(instructions: list[IRInst]) -> list[IRInst]:
    """Replace x * 2^k by shift-equivalent <<, and simplify x*0, x*1, x+0, x-0."""
    blocks = split_into_basic_blocks(instructions)
    optimized_insts: list[IRInst] = []
    global_reg_alias: dict[str, str] = {}

    # Helper: is positive power of 2?
    def is_power_of_two(val: int) -> Optional[int]:
        if val > 0 and (val & (val - 1)) == 0 and 0 < val < (1 << 63):
            return val.bit_length() - 1
        return None

    temp_shift_counter = 0

    def resolve(r: str) -> str:
        curr = r
        visited = set()
        while curr in global_reg_alias and curr not in visited:
            visited.add(curr)
            curr = global_reg_alias[curr]
        return curr

    for block in blocks:
        const_regs: dict[str, int] = {}

        for inst in block.instructions:
            if isinstance(inst, Const):
                const_regs[inst.dest] = wrap64(inst.value)
                optimized_insts.append(Const(inst.dest, wrap64(inst.value)))

            elif isinstance(inst, Binary):
                s1 = resolve(inst.src1)
                s2 = resolve(inst.src2)
                c1 = const_regs.get(s1)
                c2 = const_regs.get(s2)

                if inst.op == "*":
                    # x * 0 or 0 * x -> const 0
                    if c2 == 0 or c1 == 0:
                        const_regs[inst.dest] = 0
                        optimized_insts.append(Const(inst.dest, 0))
                    # x * 1 -> x
                    elif c2 == 1:
                        global_reg_alias[inst.dest] = s1
                        if s1 in const_regs:
                            const_regs[inst.dest] = const_regs[s1]
                    # 1 * x -> x
                    elif c1 == 1:
                        global_reg_alias[inst.dest] = s2
                        if s2 in const_regs:
                            const_regs[inst.dest] = const_regs[s2]
                    # x * 2^k -> x << k
                    elif c2 is not None and is_power_of_two(c2) is not None:
                        k = is_power_of_two(c2)
                        shift_reg = f"_shk_{temp_shift_counter}"
                        temp_shift_counter += 1
                        optimized_insts.append(Const(shift_reg, k))
                        optimized_insts.append(Binary(inst.dest, "<<", s1, shift_reg))
                        const_regs.pop(inst.dest, None)
                    # 2^k * x -> x << k
                    elif c1 is not None and is_power_of_two(c1) is not None:
                        k = is_power_of_two(c1)
                        shift_reg = f"_shk_{temp_shift_counter}"
                        temp_shift_counter += 1
                        optimized_insts.append(Const(shift_reg, k))
                        optimized_insts.append(Binary(inst.dest, "<<", s2, shift_reg))
                        const_regs.pop(inst.dest, None)
                    else:
                        optimized_insts.append(Binary(inst.dest, "*", s1, s2))

                elif inst.op == "+":
                    # x + 0 -> x
                    if c2 == 0:
                        global_reg_alias[inst.dest] = s1
                        if s1 in const_regs:
                            const_regs[inst.dest] = const_regs[s1]
                    # 0 + x -> x
                    elif c1 == 0:
                        global_reg_alias[inst.dest] = s2
                        if s2 in const_regs:
                            const_regs[inst.dest] = const_regs[s2]
                    else:
                        optimized_insts.append(Binary(inst.dest, "+", s1, s2))

                elif inst.op == "-":
                    # x - 0 -> x
                    if c2 == 0:
                        global_reg_alias[inst.dest] = s1
                        if s1 in const_regs:
                            const_regs[inst.dest] = const_regs[s1]
                    else:
                        optimized_insts.append(Binary(inst.dest, "-", s1, s2))

                else:
                    optimized_insts.append(Binary(inst.dest, inst.op, s1, s2))

            elif isinstance(inst, Unary):
                s = resolve(inst.src)
                optimized_insts.append(Unary(inst.dest, inst.op, s))

            elif isinstance(inst, Store):
                optimized_insts.append(Store(inst.var_name, resolve(inst.src)))

            elif isinstance(inst, PrintIR):
                optimized_insts.append(PrintIR(resolve(inst.src)))

            elif isinstance(inst, JumpIfZero):
                optimized_insts.append(JumpIfZero(resolve(inst.cond), inst.target))

            elif isinstance(inst, JumpIfNonZero):
                optimized_insts.append(JumpIfNonZero(resolve(inst.cond), inst.target))

            else:
                optimized_insts.append(inst)

    return apply_reg_aliases(optimized_insts, global_reg_alias)


# ---------------------------------------------------------------------------
# 5. Dead Code Elimination (DCE) via Global CFG Liveness Analysis
# ---------------------------------------------------------------------------

def dead_code_elim(instructions: list[IRInst]) -> list[IRInst]:
    """Eliminate dead instructions and stores using global CFG liveness analysis.

    Enforces:
      - Never removes PrintIR.
      - Never removes a division or modulo unless the divisor is a nonzero constant.
    """
    # Track known constant registers across instructions for division safety check
    known_consts: dict[str, int] = {}
    for inst in instructions:
        if isinstance(inst, Const):
            known_consts[inst.dest] = wrap64(inst.value)

    blocks = split_into_basic_blocks(instructions)
    if not blocks:
        return []

    num_blocks = len(blocks)

    # Compute use and def sets for each basic block
    use_sets: list[set[str]] = [set() for _ in range(num_blocks)]
    def_sets: list[set[str]] = [set() for _ in range(num_blocks)]

    for b_idx, block in enumerate(blocks):
        b_use = use_sets[b_idx]
        b_def = def_sets[b_idx]

        for inst in block.instructions:
            # Reads
            reads: list[str] = []
            if isinstance(inst, Load):
                reads.append(inst.var_name)
            elif isinstance(inst, Store):
                reads.append(inst.src)
            elif isinstance(inst, Binary):
                reads.extend([inst.src1, inst.src2])
            elif isinstance(inst, Unary):
                reads.append(inst.src)
            elif isinstance(inst, PrintIR):
                reads.append(inst.src)
            elif isinstance(inst, (JumpIfZero, JumpIfNonZero)):
                reads.append(inst.cond)

            for r in reads:
                if r not in b_def:
                    b_use.add(r)

            # Writes
            if isinstance(inst, (Const, Load, Binary, Unary)):
                b_def.add(inst.dest)
            elif isinstance(inst, Store):
                b_def.add(inst.var_name)

    # Iterative backward dataflow liveness analysis
    live_in: list[set[str]] = [set() for _ in range(num_blocks)]
    live_out: list[set[str]] = [set() for _ in range(num_blocks)]

    changed = True
    while changed:
        changed = False
        for b_idx in reversed(range(num_blocks)):
            # live_out[B] = union_{S in succ[B]} live_in[S]
            new_out: set[str] = set()
            for s_idx in blocks[b_idx].successors:
                new_out.update(live_in[s_idx])
            live_out[b_idx] = new_out

            # live_in[B] = use[B] union (live_out[B] - def[B])
            new_in = use_sets[b_idx].union(live_out[b_idx] - def_sets[b_idx])
            if new_in != live_in[b_idx]:
                live_in[b_idx] = new_in
                changed = True

    # Sweep each block backwards, eliminating dead instructions
    optimized_insts: list[IRInst] = []

    for b_idx, block in enumerate(blocks):
        current_live = set(live_out[b_idx])
        surviving_block_insts: list[IRInst] = []

        for inst in reversed(block.instructions):
            keep = True

            if isinstance(inst, Store):
                # Dead store if variable is never read afterwards
                if inst.var_name not in current_live:
                    keep = False
                else:
                    current_live.discard(inst.var_name)
                    current_live.add(inst.src)

            elif isinstance(inst, Load):
                if inst.dest not in current_live:
                    keep = False
                else:
                    current_live.discard(inst.dest)
                    current_live.add(inst.var_name)

            elif isinstance(inst, Binary):
                if inst.dest not in current_live:
                    # Division / modulo safety: never remove unless divisor is nonzero constant
                    if inst.op in ("/", "%"):
                        divisor = known_consts.get(inst.src2)
                        if divisor is not None and divisor != 0:
                            keep = False
                        else:
                            keep = True
                            current_live.add(inst.src1)
                            current_live.add(inst.src2)
                    else:
                        keep = False
                else:
                    current_live.discard(inst.dest)
                    current_live.add(inst.src1)
                    current_live.add(inst.src2)

            elif isinstance(inst, Unary):
                if inst.dest not in current_live:
                    keep = False
                else:
                    current_live.discard(inst.dest)
                    current_live.add(inst.src)

            elif isinstance(inst, Const):
                if inst.dest not in current_live:
                    keep = False
                else:
                    current_live.discard(inst.dest)

            elif isinstance(inst, PrintIR):
                # Never remove print
                keep = True
                current_live.add(inst.src)

            elif isinstance(inst, (JumpIfZero, JumpIfNonZero)):
                keep = True
                current_live.add(inst.cond)

            elif isinstance(inst, (Jump, Label)):
                keep = True

            if keep:
                surviving_block_insts.append(inst)

        surviving_block_insts.reverse()
        optimized_insts.extend(surviving_block_insts)

    return optimized_insts


# ---------------------------------------------------------------------------
# Pass Registry & Pipeline Runner
# ---------------------------------------------------------------------------

PASS_REGISTRY: dict[str, Callable[[list[IRInst]], list[IRInst]]] = {
    "constant_fold": constant_fold,
    "cse": cse,
    "licm": licm,
    "strength_reduce": strength_reduce,
    "dead_code_elim": dead_code_elim,
}

FIXED_ORDER_PASSES: list[str] = [
    "constant_fold",
    "cse",
    "licm",
    "strength_reduce",
    "dead_code_elim",
]


def run_passes(
    instructions: list[IRInst],
    passes: Optional[list[str]] = None,
) -> list[IRInst]:
    """Execute a list of optimization passes in sequential order on TAC IR."""
    if passes is None:
        return list(instructions)

    curr = list(instructions)
    for p_name in passes:
        if p_name not in PASS_REGISTRY:
            raise ValueError(f"Unknown IR pass: '{p_name}'. Available: {list(PASS_REGISTRY.keys())}")
        curr = PASS_REGISTRY[p_name](curr)

    return curr
