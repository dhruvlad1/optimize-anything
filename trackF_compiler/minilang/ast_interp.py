"""ast_interp.py - Tree-walking AST reference interpreter for MiniLang.

Directly interprets MiniLang AST nodes without generating IR or bytecode,
serving as the functional correctness oracle.
"""

from __future__ import annotations

from typing import Any
from trackF_compiler.minilang.ast_nodes import (
    Assign,
    BinaryOp,
    Block,
    Expr,
    If,
    IntLiteral,
    Print,
    Program,
    Stmt,
    UnaryOp,
    Variable,
    While,
)
from trackF_compiler.minilang.semantics import (
    apply_binary_op,
    apply_unary_op,
    wrap64,
)


class ASTInterpreter:
    """Reference interpreter executing MiniLang programs directly over the AST."""

    def __init__(self) -> None:
        self.env: dict[str, int] = {}
        self.output: list[int] = []

    def run(self, program: Program) -> list[int]:
        self.env.clear()
        self.output.clear()
        for stmt in program.statements:
            self._exec_stmt(stmt)
        return list(self.output)

    def _exec_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, Assign):
            val = self._eval_expr(stmt.value)
            self.env[stmt.name] = wrap64(val)
        elif isinstance(stmt, Print):
            val = self._eval_expr(stmt.value)
            self.output.append(wrap64(val))
        elif isinstance(stmt, Block):
            for s in stmt.statements:
                self._exec_stmt(s)
        elif isinstance(stmt, If):
            cond = self._eval_expr(stmt.condition)
            if cond != 0:
                self._exec_stmt(stmt.then_branch)
            elif stmt.else_branch is not None:
                self._exec_stmt(stmt.else_branch)
        elif isinstance(stmt, While):
            while self._eval_expr(stmt.condition) != 0:
                self._exec_stmt(stmt.body)
        else:
            raise NotImplementedError(f"Unsupported statement type: {type(stmt).__name__}")

    def _eval_expr(self, expr: Expr) -> int:
        if isinstance(expr, IntLiteral):
            return wrap64(expr.value)
        elif isinstance(expr, Variable):
            if expr.name not in self.env:
                raise RuntimeError(f"Uninitialised variable '{expr.name}' at line {expr.line}, col {expr.col}")
            return self.env[expr.name]
        elif isinstance(expr, UnaryOp):
            val = self._eval_expr(expr.operand)
            return apply_unary_op(expr.op, val)
        elif isinstance(expr, BinaryOp):
            left_val = self._eval_expr(expr.left)
            right_val = self._eval_expr(expr.right)
            return apply_binary_op(expr.op, left_val, right_val)
        else:
            raise NotImplementedError(f"Unsupported expression type: {type(expr).__name__}")
