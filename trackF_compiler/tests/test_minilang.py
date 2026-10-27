"""test_minilang.py - Pytest test suite for MiniLang compiler pipeline."""

from pathlib import Path
import pytest

from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.compiler import compile_program
from trackF_compiler.minilang.lexer import Lexer, LexerError, TokenType
from trackF_compiler.minilang.parser import ParseError, Parser, SemanticError
from trackF_compiler.minilang.semantics import div_trunc, mod_trunc, wrap64
from trackF_compiler.minilang.vm import VirtualMachine


def test_lexer_tokens_and_errors() -> None:
    """Test tokenization of basic statements and error reporting."""
    source = "x = 42 + y * (z - 5);"
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    expected_types = [
        TokenType.IDENT, TokenType.ASSIGN, TokenType.INT, TokenType.PLUS,
        TokenType.IDENT, TokenType.STAR, TokenType.LPAREN, TokenType.IDENT,
        TokenType.MINUS, TokenType.INT, TokenType.RPAREN, TokenType.SEMI,
        TokenType.EOF
    ]
    assert [t.type for t in tokens] == expected_types
    assert tokens[0].value == "x"
    assert tokens[0].line == 1 and tokens[0].col == 1
    assert tokens[2].value == 42

    # Unexpected character test
    with pytest.raises(LexerError) as exc_info:
        Lexer("x = @;").tokenize()
    assert exc_info.value.line == 1 and exc_info.value.col == 5


def test_operator_precedence() -> None:
    """Verify operator precedence rules across additive, multiplicative, unary, and comparisons."""
    # Multiplicative binds tighter than additive: 2 + 3 * 4 == 14 (not 20)
    src1 = "x = 2 + 3 * 4; print x;"
    ast1, _, vm1, _ = compile_program(src1)
    assert ASTInterpreter().run(ast1) == [14]
    assert VirtualMachine().run(vm1)[0] == [14]

    # Additive left-associativity: 10 - 4 - 2 == 4
    src2 = "x = 10 - 4 - 2; print x;"
    ast2, _, vm2, _ = compile_program(src2)
    assert VirtualMachine().run(vm2)[0] == [4]

    # Unary minus binds tighter than binary ops: -5 + 3 == -2
    src3 = "x = -5 + 3; print x;"
    ast3, _, vm3, _ = compile_program(src3)
    assert VirtualMachine().run(vm3)[0] == [-2]

    # Comparison binds looser than additive: 1 + 2 < 2 * 2 => 3 < 4 => 1
    src4 = "x = 1 + 2 < 2 * 2; print x;"
    ast4, _, vm4, _ = compile_program(src4)
    assert VirtualMachine().run(vm4)[0] == [1]

    # Equality comparison: 5 == 2 + 3 => 1
    src5 = "x = 5 == 2 + 3; print x;"
    ast5, _, vm5, _ = compile_program(src5)
    assert VirtualMachine().run(vm5)[0] == [1]


def test_semantics_functions() -> None:
    """Test 64-bit integer wraparound and C-style truncated division and modulo."""
    # Truncated division toward zero
    assert div_trunc(-7, 2) == -3
    assert div_trunc(7, -2) == -3
    assert div_trunc(-7, -2) == 3
    assert div_trunc(7, 2) == 3

    # Truncated modulo (remainder follows dividend sign)
    assert mod_trunc(-7, 2) == -1
    assert mod_trunc(7, -2) == 1
    assert mod_trunc(-7, -2) == -1
    assert mod_trunc(7, 2) == 1

    # Two's-complement 64-bit wraparound
    max_int64 = (1 << 63) - 1
    min_int64 = -(1 << 63)
    assert wrap64(max_int64) == max_int64
    assert wrap64(min_int64) == min_int64
    assert wrap64(1 << 63) == min_int64
    assert wrap64((1 << 63) + 1) == min_int64 + 1
    assert wrap64(min_int64 - 1) == max_int64


def test_division_by_zero_error() -> None:
    """Verify runtime division or modulo by zero raises clear error in semantics, oracle, and VM."""
    with pytest.raises(ZeroDivisionError):
        div_trunc(10, 0)

    with pytest.raises(ZeroDivisionError):
        mod_trunc(10, 0)

    prog_div = "x = 10 / 0; print x;"
    ast_p, _, vm_p, _ = compile_program(prog_div)

    with pytest.raises(ZeroDivisionError):
        ASTInterpreter().run(ast_p)

    with pytest.raises(ZeroDivisionError):
        VirtualMachine().run(vm_p)


def test_uninitialised_variable_error() -> None:
    """Verify compile-time detection of uninitialized variable usage."""
    # Immediate read before assignment
    with pytest.raises(SemanticError) as exc_info1:
        compile_program("x = y + 1;")
    assert "uninitialised variable 'y'" in str(exc_info1.value)

    # Incomplete assignment across branches (only in then branch)
    src_branch = "if (1) { y = 10; } x = y;"
    with pytest.raises(SemanticError) as exc_info2:
        compile_program(src_branch)
    assert "uninitialised variable 'y'" in str(exc_info2.value)

    # Valid initialization
    ast_valid, _, vm_valid, _ = compile_program("y = 10; x = y + 1; print x;")
    assert VirtualMachine().run(vm_valid)[0] == [11]


def test_control_flow_tiny_programs() -> None:
    """Verify branch and loop execution on tiny hand-computed programs."""
    src_if = """
    x = 10;
    if (x > 5) {
        res = 1;
    } else {
        res = 2;
    }
    print res;
    """
    ast_if, _, vm_if, _ = compile_program(src_if)
    assert ASTInterpreter().run(ast_if) == [1]
    assert VirtualMachine().run(vm_if)[0] == [1]

    src_while = """
    i = 0;
    acc = 0;
    while (i < 5) {
        acc = acc + i;
        i = i + 1;
    }
    print acc;
    """
    ast_while, _, vm_while, _ = compile_program(src_while)
    assert ASTInterpreter().run(ast_while) == [10]
    assert VirtualMachine().run(vm_while)[0] == [10]


def test_oracle_and_vm_match_all_benchmarks() -> None:
    """Verify that AST oracle and VM produce identical outputs on all programs in trackF_compiler/programs/."""
    programs_dir = Path(__file__).resolve().parent.parent / "programs"
    program_files = sorted(programs_dir.glob("*.ml"))
    assert len(program_files) >= 6

    known_expectations = {
        "bench_sum.ml": [5000050000],
        "bench_fib.ml": [832040],
        "bench_primes.ml": [303],
    }

    for p in program_files:
        source = p.read_text(encoding="utf-8")
        ast_prog, _, vm_prog, _ = compile_program(source, opt_level=0)

        oracle_out = ASTInterpreter().run(ast_prog)
        vm_out, _ = VirtualMachine().run(vm_prog)

        assert oracle_out == vm_out, f"Output mismatch in {p.name}: {oracle_out} != {vm_out}"

        if p.name in known_expectations:
            assert vm_out == known_expectations[p.name], f"Expected {known_expectations[p.name]} for {p.name}, got {vm_out}"
