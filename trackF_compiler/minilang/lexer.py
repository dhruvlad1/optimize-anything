"""lexer.py - Lexical analyzer for MiniLang.

Converts MiniLang source code into a stream of tokens, tracking 1-indexed
line and column positions for every token.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Literals & Identifiers
    INT = auto()
    IDENT = auto()

    # Keywords
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    PRINT = auto()

    # Operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    PERCENT = auto()    # %

    # Comparisons
    LT = auto()         # <
    LE = auto()         # <=
    GT = auto()         # >
    GE = auto()         # >=
    EQ = auto()         # ==
    NE = auto()         # !=

    # Assignment & Punctuation
    ASSIGN = auto()     # =
    SEMI = auto()       # ;
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    LPAREN = auto()     # (
    RPAREN = auto()     # )

    EOF = auto()


KEYWORDS: dict[str, TokenType] = {
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "print": TokenType.PRINT,
}


@dataclass
class Token:
    type: TokenType
    value: int | str | None
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.col})"


class LexerError(Exception):
    """Raised for lexical errors with line and column location."""
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"Lexer error at line {line}, col {col}: {message}")
        self.line = line
        self.col = col


class Lexer:
    """Tokenizer for MiniLang source programs."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.length = len(source)

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < self.length:
            return self.source[idx]
        return ""

    def _advance(self) -> str:
        if self.pos >= self.length:
            return ""
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # Skip whitespace
            if ch in " \t\r\n":
                self._advance()
                continue

            # Skip line comments //
            if ch == "/" and self._peek(1) == "/":
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue

            start_line = self.line
            start_col = self.col

            # Integer literals
            if ch.isdigit():
                num_str = ""
                while self._peek().isdigit():
                    num_str += self._advance()
                tokens.append(Token(TokenType.INT, int(num_str), start_line, start_col))
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                ident_str = ""
                while self._peek().isalnum() or self._peek() == "_":
                    ident_str += self._advance()
                token_type = KEYWORDS.get(ident_str, TokenType.IDENT)
                tokens.append(Token(token_type, ident_str, start_line, start_col))
                continue

            # Two-character operators
            two_ch = ch + self._peek(1)
            if two_ch == "<=":
                self._advance(); self._advance()
                tokens.append(Token(TokenType.LE, "<=", start_line, start_col))
            elif two_ch == ">=":
                self._advance(); self._advance()
                tokens.append(Token(TokenType.GE, ">=", start_line, start_col))
            elif two_ch == "==":
                self._advance(); self._advance()
                tokens.append(Token(TokenType.EQ, "==", start_line, start_col))
            elif two_ch == "!=":
                self._advance(); self._advance()
                tokens.append(Token(TokenType.NE, "!=", start_line, start_col))
            # Single-character tokens
            elif ch == "+":
                self._advance()
                tokens.append(Token(TokenType.PLUS, "+", start_line, start_col))
            elif ch == "-":
                self._advance()
                tokens.append(Token(TokenType.MINUS, "-", start_line, start_col))
            elif ch == "*":
                self._advance()
                tokens.append(Token(TokenType.STAR, "*", start_line, start_col))
            elif ch == "/":
                self._advance()
                tokens.append(Token(TokenType.SLASH, "/", start_line, start_col))
            elif ch == "%":
                self._advance()
                tokens.append(Token(TokenType.PERCENT, "%", start_line, start_col))
            elif ch == "<":
                self._advance()
                tokens.append(Token(TokenType.LT, "<", start_line, start_col))
            elif ch == ">":
                self._advance()
                tokens.append(Token(TokenType.GT, ">", start_line, start_col))
            elif ch == "=":
                self._advance()
                tokens.append(Token(TokenType.ASSIGN, "=", start_line, start_col))
            elif ch == ";":
                self._advance()
                tokens.append(Token(TokenType.SEMI, ";", start_line, start_col))
            elif ch == "{":
                self._advance()
                tokens.append(Token(TokenType.LBRACE, "{", start_line, start_col))
            elif ch == "}":
                self._advance()
                tokens.append(Token(TokenType.RBRACE, "}", start_line, start_col))
            elif ch == "(":
                self._advance()
                tokens.append(Token(TokenType.LPAREN, "(", start_line, start_col))
            elif ch == ")":
                self._advance()
                tokens.append(Token(TokenType.RPAREN, ")", start_line, start_col))
            else:
                self._advance()
                raise LexerError(f"Unexpected character {ch!r}", start_line, start_col)

        tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return tokens
