"""semantics.py - Shared 64-bit integer semantics for MiniLang.

Defines exact signed 64-bit two's-complement arithmetic, truncated C-style
division and modulo, and division-by-zero checks shared across all pipeline components.
"""

from __future__ import annotations

INT64_MIN: int = -(1 << 63)
INT64_MAX: int = (1 << 63) - 1
INT64_MASK: int = (1 << 64) - 1


def wrap64(val: int) -> int:
    """Wrap an integer into a signed 64-bit range [-2^63, 2^63 - 1]."""
    # Two's complement wraparound modulo 2^64
    val_mod = (val + (1 << 63)) % (1 << 64)
    return int(val_mod - (1 << 63))


def div_trunc(a: int, b: int) -> int:
    """Truncated integer division toward zero (C semantics).

    Raises:
        ZeroDivisionError: If divisor b is 0.
    """
    a_w = wrap64(a)
    b_w = wrap64(b)
    if b_w == 0:
        raise ZeroDivisionError("MiniLang runtime error: division by zero")

    # In C99, integer division truncates toward zero
    quotient = int(a_w / b_w)
    return wrap64(quotient)


def mod_trunc(a: int, b: int) -> int:
    """Truncated integer remainder (C semantics: a - (a / b) * b).

    Raises:
        ZeroDivisionError: If divisor b is 0.
    """
    a_w = wrap64(a)
    b_w = wrap64(b)
    if b_w == 0:
        raise ZeroDivisionError("MiniLang runtime error: modulo by zero")

    # Matches C semantics: sign follows dividend a
    remainder = a_w - int(a_w / b_w) * b_w
    return wrap64(remainder)


def apply_binary_op(op: str, left: int, right: int) -> int:
    """Apply a binary operator under MiniLang 64-bit semantics."""
    l = wrap64(left)
    r = wrap64(right)
    if op == "+":
        return wrap64(l + r)
    elif op == "-":
        return wrap64(l - r)
    elif op == "*":
        return wrap64(l * r)
    elif op == "/":
        return div_trunc(l, r)
    elif op == "%":
        return mod_trunc(l, r)
    elif op == "<":
        return 1 if l < r else 0
    elif op == "<=":
        return 1 if l <= r else 0
    elif op == ">":
        return 1 if l > r else 0
    elif op == ">=":
        return 1 if l >= r else 0
    elif op == "==":
        return 1 if l == r else 0
    elif op == "!=":
        return 1 if l != r else 0
    else:
        raise ValueError(f"Unknown binary operator: {op}")


def apply_unary_op(op: str, val: int) -> int:
    """Apply a unary operator under MiniLang 64-bit semantics."""
    v = wrap64(val)
    if op == "-":
        return wrap64(-v)
    else:
        raise ValueError(f"Unknown unary operator: {op}")
