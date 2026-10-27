"""irgen.py - Intermediate Representation code generator for MiniLang.

Lowers MiniLang AST programs into linear Three-Address Code (TAC), converting
high-level structured control flow (if/else, while) into explicit labels and branches.
"""

from __future__ import annotations

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
from trackF_compiler.minilang.ir import (
    Binary,
    Const,
    IRInst,
    Jump,
    JumpIfZero,
    Label,
    Load,
    PrintIR,
    Store,
    Unary,
)


class IRGenerator:
    """Translates MiniLang AST to linear Three-Address Code."""

    def __init__(self) -> None:
        self.instructions: list[IRInst] = []
        self._reg_counter = 0
        self._label_counter = 0

    def _new_reg(self) -> str:
        r = f"t{self._reg_counter}"
        self._reg_counter += 1
        return r

    def _new_label(self, prefix: str = "L") -> str:
        lbl = f"{prefix}_{self._label_counter}"
        self._label_counter += 1
        return lbl

    def _emit(self, inst: IRInst) -> None:
        self.instructions.append(inst)

    def generate(self, program: Program) -> list[IRInst]:
        self.instructions = []
        self._reg_counter = 0
        self._label_counter = 0

        for stmt in program.statements:
            self._gen_stmt(stmt)

        return list(self.instructions)

    def _gen_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, Assign):
            res_reg = self._gen_expr(stmt.value)
            self._emit(Store(var_name=stmt.name, src=res_reg))
        elif isinstance(stmt, Print):
            res_reg = self._gen_expr(stmt.value)
            self._emit(PrintIR(src=res_reg))
        elif isinstance(stmt, Block):
            for s in stmt.statements:
                self._gen_stmt(s)
        elif isinstance(stmt, If):
            cond_reg = self._gen_expr(stmt.condition)
            if stmt.else_branch is not None:
                lbl_else = self._new_label("L_else")
                lbl_end = self._new_label("L_end")
                self._emit(JumpIfZero(cond=cond_reg, target=lbl_else))
                self._gen_stmt(stmt.then_branch)
                self._emit(Jump(target=lbl_end))
                self._emit(Label(name=lbl_else))
                self._gen_stmt(stmt.else_branch)
                self._emit(Label(name=lbl_end))
            else:
                lbl_end = self._new_label("L_end")
                self._emit(JumpIfZero(cond=cond_reg, target=lbl_end))
                self._gen_stmt(stmt.then_branch)
                self._emit(Label(name=lbl_end))
        elif isinstance(stmt, While):
            lbl_loop = self._new_label("L_loop")
            lbl_end = self._new_label("L_end")
            self._emit(Label(name=lbl_loop))
            cond_reg = self._gen_expr(stmt.condition)
            self._emit(JumpIfZero(cond=cond_reg, target=lbl_end))
            self._gen_stmt(stmt.body)
            self._emit(Jump(target=lbl_loop))
            self._emit(Label(name=lbl_end))
        else:
            raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")

    def _gen_expr(self, expr: Expr) -> str:
        if isinstance(expr, IntLiteral):
            reg = self._new_reg()
            self._emit(Const(dest=reg, value=expr.value))
            return reg
        elif isinstance(expr, Variable):
            reg = self._new_reg()
            self._emit(Load(dest=reg, var_name=expr.name))
            return reg
        elif isinstance(expr, UnaryOp):
            src_reg = self._gen_expr(expr.operand)
            dest_reg = self._new_reg()
            self._emit(Unary(dest=dest_reg, op=expr.op, src=src_reg))
            return dest_reg
        elif isinstance(expr, BinaryOp):
            src1_reg = self._gen_expr(expr.left)
            src2_reg = self._gen_expr(expr.right)
            dest_reg = self._new_reg()
            self._emit(Binary(dest=dest_reg, op=expr.op, src1=src1_reg, src2=src2_reg))
            return dest_reg
        else:
            raise NotImplementedError(f"Unsupported expression: {type(expr).__name__}")
