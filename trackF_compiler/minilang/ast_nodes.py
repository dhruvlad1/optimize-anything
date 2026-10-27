"""ast_nodes.py - Abstract Syntax Tree dataclasses for MiniLang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ASTNode:
    """Base class for all AST nodes with source position."""
    line: int
    col: int


# --- Expressions ---

@dataclass
class Expr(ASTNode):
    """Base class for expressions."""
    pass


@dataclass
class IntLiteral(Expr):
    value: int


@dataclass
class Variable(Expr):
    name: str


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class BinaryOp(Expr):
    left: Expr
    op: str
    right: Expr


# --- Statements ---

@dataclass
class Stmt(ASTNode):
    """Base class for statements."""
    pass


@dataclass
class Assign(Stmt):
    name: str
    value: Expr


@dataclass
class Print(Stmt):
    value: Expr


@dataclass
class If(Stmt):
    condition: Expr
    then_branch: Stmt
    else_branch: Optional[Stmt] = None


@dataclass
class While(Stmt):
    condition: Expr
    body: Stmt


@dataclass
class Block(Stmt):
    statements: list[Stmt]


@dataclass
class Program(ASTNode):
    statements: list[Stmt]
