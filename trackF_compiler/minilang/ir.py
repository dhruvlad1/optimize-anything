"""ir.py - Intermediate Representation (linear Three-Address Code) for MiniLang.

Defines instructions over virtual registers (t0, t1, ...), named user variables,
symbolic labels, and conditional/unconditional branches, along with a textual dump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class IRInst:
    """Base class for Three-Address Code (TAC) intermediate instructions."""
    pass


@dataclass
class Label(IRInst):
    name: str

    def __str__(self) -> str:
        return f"{self.name}:"


@dataclass
class Jump(IRInst):
    target: str

    def __str__(self) -> str:
        return f"    jump {self.target}"


@dataclass
class JumpIfZero(IRInst):
    cond: str
    target: str

    def __str__(self) -> str:
        return f"    jump_if_zero {self.cond}, {self.target}"


@dataclass
class JumpIfNonZero(IRInst):
    cond: str
    target: str

    def __str__(self) -> str:
        return f"    jump_if_nonzero {self.cond}, {self.target}"


@dataclass
class Const(IRInst):
    dest: str
    value: int

    def __str__(self) -> str:
        return f"    {self.dest} = const {self.value}"


@dataclass
class Load(IRInst):
    dest: str
    var_name: str

    def __str__(self) -> str:
        return f"    {self.dest} = load {self.var_name}"


@dataclass
class Store(IRInst):
    var_name: str
    src: str

    def __str__(self) -> str:
        return f"    store {self.var_name}, {self.src}"


@dataclass
class Binary(IRInst):
    dest: str
    op: str
    src1: str
    src2: str

    def __str__(self) -> str:
        return f"    {self.dest} = {self.src1} {self.op} {self.src2}"


@dataclass
class Unary(IRInst):
    dest: str
    op: str
    src: str

    def __str__(self) -> str:
        return f"    {self.dest} = {self.op}{self.src}"


@dataclass
class PrintIR(IRInst):
    src: str

    def __str__(self) -> str:
        return f"    print {self.src}"


def dump_ir(instructions: list[IRInst]) -> str:
    """Render a sequence of IR instructions as human-readable assembly text."""
    lines: list[str] = []
    for inst in instructions:
        lines.append(str(inst))
    return "\n".join(lines)
