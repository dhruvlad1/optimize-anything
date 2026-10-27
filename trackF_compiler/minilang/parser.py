"""parser.py - Recursive descent parser and static semantic validator for MiniLang.

Constructs an Abstract Syntax Tree (AST) from a stream of tokens, enforcing
the grammar, operator precedence, and compile-time uninitialized variable checks.
"""

from __future__ import annotations

from typing import Set
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
from trackF_compiler.minilang.lexer import Token, TokenType


class ParseError(Exception):
    """Raised for syntax errors with line and column metadata."""
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"Parse error at line {line}, col {col}: {message}")
        self.line = line
        self.col = col


class SemanticError(Exception):
    """Raised for compile-time semantic errors such as uninitialized variables."""
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"Semantic error at line {line}, col {col}: {message}")
        self.line = line
        self.col = col


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _previous(self) -> Token:
        return self.tokens[self.pos - 1]

    def _check(self, token_type: TokenType) -> bool:
        return self._peek().type == token_type

    def _match(self, *token_types: TokenType) -> bool:
        for t in token_types:
            if self._check(t):
                self._advance()
                return True
        return False

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def _consume(self, token_type: TokenType, err_msg: str) -> Token:
        if self._check(token_type):
            return self._advance()
        tok = self._peek()
        raise ParseError(f"{err_msg} (got {tok.type.name})", tok.line, tok.col)

    def parse(self) -> Program:
        statements: list[Stmt] = []
        start_line = self._peek().line
        start_col = self._peek().col

        while not self._check(TokenType.EOF):
            statements.append(self._statement())

        prog = Program(statements=statements, line=start_line, col=start_col)
        self._validate_variables(prog)
        return prog

    # --- Statements ---

    def _statement(self) -> Stmt:
        if self._check(TokenType.PRINT):
            return self._print_statement()
        elif self._check(TokenType.IF):
            return self._if_statement()
        elif self._check(TokenType.WHILE):
            return self._while_statement()
        elif self._check(TokenType.LBRACE):
            return self._block_statement()
        elif self._check(TokenType.IDENT):
            return self._assign_statement()
        else:
            tok = self._peek()
            raise ParseError(f"Unexpected token in statement: {tok.type.name}", tok.line, tok.col)

    def _assign_statement(self) -> Assign:
        ident_tok = self._consume(TokenType.IDENT, "Expected identifier")
        name = str(ident_tok.value)
        self._consume(TokenType.ASSIGN, "Expected '=' after identifier in assignment")
        expr = self._expression()
        self._consume(TokenType.SEMI, "Expected ';' after assignment")
        return Assign(name=name, value=expr, line=ident_tok.line, col=ident_tok.col)

    def _print_statement(self) -> Print:
        print_tok = self._consume(TokenType.PRINT, "Expected 'print'")
        expr = self._expression()
        self._consume(TokenType.SEMI, "Expected ';' after print expression")
        return Print(value=expr, line=print_tok.line, col=print_tok.col)

    def _if_statement(self) -> If:
        if_tok = self._consume(TokenType.IF, "Expected 'if'")
        self._consume(TokenType.LPAREN, "Expected '(' after 'if'")
        cond = self._expression()
        self._consume(TokenType.RPAREN, "Expected ')' after if condition")
        then_branch = self._statement()
        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self._statement()
        return If(condition=cond, then_branch=then_branch, else_branch=else_branch, line=if_tok.line, col=if_tok.col)

    def _while_statement(self) -> While:
        while_tok = self._consume(TokenType.WHILE, "Expected 'while'")
        self._consume(TokenType.LPAREN, "Expected '(' after 'while'")
        cond = self._expression()
        self._consume(TokenType.RPAREN, "Expected ')' after while condition")
        body = self._statement()
        return While(condition=cond, body=body, line=while_tok.line, col=while_tok.col)

    def _block_statement(self) -> Block:
        brace_tok = self._consume(TokenType.LBRACE, "Expected '{'")
        stmts: list[Stmt] = []
        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            stmts.append(self._statement())
        self._consume(TokenType.RBRACE, "Expected '}' closing block")
        return Block(statements=stmts, line=brace_tok.line, col=brace_tok.col)

    # --- Expressions ---

    def _expression(self) -> Expr:
        return self._comparison()

    def _comparison(self) -> Expr:
        expr = self._addition()
        comp_types = (TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE, TokenType.EQ, TokenType.NE)
        while any(self._check(t) for t in comp_types):
            op_tok = self._advance()
            right = self._addition()
            expr = BinaryOp(left=expr, op=str(op_tok.value), right=right, line=op_tok.line, col=op_tok.col)
        return expr

    def _addition(self) -> Expr:
        expr = self._multiplication()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op_tok = self._previous()
            right = self._multiplication()
            expr = BinaryOp(left=expr, op=str(op_tok.value), right=right, line=op_tok.line, col=op_tok.col)
        return expr

    def _multiplication(self) -> Expr:
        expr = self._unary()
        while self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op_tok = self._previous()
            right = self._unary()
            expr = BinaryOp(left=expr, op=str(op_tok.value), right=right, line=op_tok.line, col=op_tok.col)
        return expr

    def _unary(self) -> Expr:
        if self._match(TokenType.MINUS):
            op_tok = self._previous()
            operand = self._unary()
            return UnaryOp(op="-", operand=operand, line=op_tok.line, col=op_tok.col)
        return self._primary()

    def _primary(self) -> Expr:
        if self._match(TokenType.INT):
            tok = self._previous()
            return IntLiteral(value=int(tok.value), line=tok.line, col=tok.col)
        if self._match(TokenType.IDENT):
            tok = self._previous()
            return Variable(name=str(tok.value), line=tok.line, col=tok.col)
        if self._match(TokenType.LPAREN):
            lparen_tok = self._previous()
            expr = self._expression()
            self._consume(TokenType.RPAREN, "Expected ')' after parenthesized expression")
            return expr
        tok = self._peek()
        raise ParseError(f"Expected expression, got {tok.type.name}", tok.line, tok.col)

    # --- Definite Assignment / Uninitialized Variable Check ---

    def _validate_variables(self, prog: Program) -> None:
        """Statically verify that every variable read is defined/assigned beforehand."""
        def check_expr(expr: Expr, initialized: Set[str]) -> None:
            if isinstance(expr, Variable):
                if expr.name not in initialized:
                    raise SemanticError(f"Use of uninitialised variable '{expr.name}'", expr.line, expr.col)
            elif isinstance(expr, UnaryOp):
                check_expr(expr.operand, initialized)
            elif isinstance(expr, BinaryOp):
                check_expr(expr.left, initialized)
                check_expr(expr.right, initialized)

        def check_stmt(stmt: Stmt, initialized: Set[str]) -> Set[str]:
            if isinstance(stmt, Assign):
                check_expr(stmt.value, initialized)
                return initialized | {stmt.name}
            elif isinstance(stmt, Print):
                check_expr(stmt.value, initialized)
                return initialized
            elif isinstance(stmt, Block):
                cur = initialized
                for s in stmt.statements:
                    cur = check_stmt(s, cur)
                return cur
            elif isinstance(stmt, If):
                check_expr(stmt.condition, initialized)
                init_then = check_stmt(stmt.then_branch, initialized)
                if stmt.else_branch:
                    init_else = check_stmt(stmt.else_branch, initialized)
                    return init_then & init_else
                return initialized
            elif isinstance(stmt, While):
                check_expr(stmt.condition, initialized)
                check_stmt(stmt.body, initialized)
                return initialized
            return initialized

        init_vars: Set[str] = set()
        for s in prog.statements:
            init_vars = check_stmt(s, init_vars)
