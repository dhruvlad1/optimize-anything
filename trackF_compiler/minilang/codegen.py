"""codegen.py - VM code generator and bytecode assembler for MiniLang.

Converts linear Three-Address Code into concrete VM instruction tuples,
resolving symbolic labels into absolute program counter (PC) jump targets.
"""

from __future__ import annotations

from typing import Any
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

# VM instruction format: tuple of (OPCODE, *args)
VMInstruction = tuple[Any, ...]


class CodeGenerator:
    """Assembles linear TAC IR into an array of executable VM instruction tuples."""

    def generate(self, ir_instructions: list[IRInst]) -> list[VMInstruction]:
        # Pass 1: Record label targets and strip Label pseudo-instructions
        label_map: dict[str, int] = {}
        stripped: list[IRInst] = []

        for inst in ir_instructions:
            if isinstance(inst, Label):
                label_map[inst.name] = len(stripped)
            else:
                stripped.append(inst)

        # Pass 2: Emit tuples with resolved integer jump targets
        vm_code: list[VMInstruction] = []
        for pc, inst in enumerate(stripped):
            if isinstance(inst, Const):
                vm_code.append(("CONST", inst.dest, inst.value))
            elif isinstance(inst, Load):
                vm_code.append(("LOAD", inst.dest, inst.var_name))
            elif isinstance(inst, Store):
                vm_code.append(("STORE", inst.var_name, inst.src))
            elif isinstance(inst, Binary):
                vm_code.append(("BINARY", inst.dest, inst.op, inst.src1, inst.src2))
            elif isinstance(inst, Unary):
                vm_code.append(("UNARY", inst.dest, inst.op, inst.src))
            elif isinstance(inst, PrintIR):
                vm_code.append(("PRINT", inst.src))
            elif isinstance(inst, Jump):
                target_pc = label_map[inst.target]
                vm_code.append(("JUMP", target_pc))
            elif isinstance(inst, JumpIfZero):
                target_pc = label_map[inst.target]
                vm_code.append(("JUMP_IF_ZERO", inst.cond, target_pc))
            elif isinstance(inst, JumpIfNonZero):
                target_pc = label_map[inst.target]
                vm_code.append(("JUMP_IF_NONZERO", inst.cond, target_pc))
            else:
                raise NotImplementedError(f"Unsupported IR instruction: {type(inst).__name__}")

        return vm_code


def dump_asm(vm_program: list[VMInstruction]) -> str:
    """Render the assembled VM instructions as indexed assembly text."""
    lines: list[str] = []
    for pc, inst in enumerate(vm_program):
        op = inst[0]
        args_str = ", ".join(str(a) for a in inst[1:])
        lines.append(f"{pc:04d}: {op:<16} {args_str}")
    return "\n".join(lines)
