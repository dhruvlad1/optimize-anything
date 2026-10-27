"""vm.py - Virtual Machine execution engine for MiniLang bytecode.

Executes linear VM instruction tuples with an isolated register and variable store,
dynamically counting every executed instruction and enforcing 64-bit integer semantics.
"""

from __future__ import annotations

from typing import Any
from trackF_compiler.minilang.codegen import VMInstruction
from trackF_compiler.minilang.semantics import (
    apply_binary_op,
    apply_unary_op,
    wrap64,
)


class VirtualMachine:
    """Stackless register/variable-store execution engine for MiniLang programs."""

    def __init__(self) -> None:
        self.registers: dict[str, int] = {}
        self.variables: dict[str, int] = {}
        self.output: list[int] = []
        self.dynamic_instr_count: int = 0

    def run(self, program: list[VMInstruction]) -> tuple[list[int], int]:
        """Execute a VM bytecode program.

        Returns:
            (output_list, dynamic_instruction_count)
        """
        self.registers.clear()
        self.variables.clear()
        self.output.clear()
        self.dynamic_instr_count = 0

        pc = 0
        prog_len = len(program)

        while pc < prog_len:
            inst = program[pc]
            self.dynamic_instr_count += 1
            op = inst[0]

            if op == "CONST":
                # ("CONST", dest, value)
                _, dest, value = inst
                self.registers[dest] = wrap64(value)
                pc += 1

            elif op == "LOAD":
                # ("LOAD", dest, var_name)
                _, dest, var_name = inst
                if var_name not in self.variables:
                    raise RuntimeError(f"VM error: uninitialised variable '{var_name}' at pc {pc}")
                self.registers[dest] = self.variables[var_name]
                pc += 1

            elif op == "STORE":
                # ("STORE", var_name, src)
                _, var_name, src = inst
                self.variables[var_name] = self.registers[src]
                pc += 1

            elif op == "BINARY":
                # ("BINARY", dest, op, src1, src2)
                _, dest, bin_op, src1, src2 = inst
                val1 = self.registers[src1]
                val2 = self.registers[src2]
                try:
                    self.registers[dest] = apply_binary_op(bin_op, val1, val2)
                except ZeroDivisionError as e:
                    raise ZeroDivisionError(f"VM runtime error at pc {pc}: {e}") from None
                pc += 1

            elif op == "UNARY":
                # ("UNARY", dest, op, src)
                _, dest, un_op, src = inst
                val = self.registers[src]
                self.registers[dest] = apply_unary_op(un_op, val)
                pc += 1

            elif op == "PRINT":
                # ("PRINT", src)
                _, src = inst
                val = self.registers[src]
                self.output.append(val)
                pc += 1

            elif op == "JUMP":
                # ("JUMP", target_pc)
                _, target_pc = inst
                pc = target_pc

            elif op == "JUMP_IF_ZERO":
                # ("JUMP_IF_ZERO", cond, target_pc)
                _, cond, target_pc = inst
                if self.registers[cond] == 0:
                    pc = target_pc
                else:
                    pc += 1

            elif op == "JUMP_IF_NONZERO":
                # ("JUMP_IF_NONZERO", cond, target_pc)
                _, cond, target_pc = inst
                if self.registers[cond] != 0:
                    pc = target_pc
                else:
                    pc += 1

            else:
                raise NotImplementedError(f"Unknown VM opcode '{op}' at pc {pc}")

        return list(self.output), self.dynamic_instr_count
