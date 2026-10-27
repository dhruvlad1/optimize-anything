"""compiler.py - End-to-end MiniLang compiler pipeline and CLI driver.

Coordinates lexical analysis, parsing, AST validation, IR lowering, and VM code generation.
Includes CLI interface for dumping intermediate representations and running programs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from trackF_compiler.minilang.ast_interp import ASTInterpreter
from trackF_compiler.minilang.ast_nodes import Program
from trackF_compiler.minilang.codegen import CodeGenerator, VMInstruction, dump_asm
from trackF_compiler.minilang.ir import IRInst, dump_ir
from trackF_compiler.minilang.irgen import IRGenerator
from trackF_compiler.minilang.lexer import Lexer
from trackF_compiler.minilang.parser import Parser
from trackF_compiler.minilang.vm import VirtualMachine


def compile_program(
    source: str, opt_level: int = 0
) -> tuple[Program, list[IRInst], list[VMInstruction], dict[str, Any]]:
    """Compile MiniLang source code into IR and VM bytecode.

    Args:
        source: MiniLang source code as string.
        opt_level: Optimization level (only 0 is supported currently).

    Returns:
        (ast_program, ir_instructions, vm_instructions, stats_dict)
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be a string, got {type(source).__name__}")
    if opt_level != 0:
        raise NotImplementedError(
            f"Optimization level {opt_level} is not implemented. Only opt_level=0 is supported."
        )

    t0 = time.perf_counter()

    # 1. Lexing
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    # 2. Parsing & Static Semantic Checks
    parser = Parser(tokens)
    ast_prog = parser.parse()

    # 3. IR Generation (Three-Address Code)
    ir_gen = IRGenerator()
    ir_prog = ir_gen.generate(ast_prog)

    # 4. VM Code Generation (Label Resolution)
    code_gen = CodeGenerator()
    vm_prog = code_gen.generate(ir_prog)

    compile_time_s = time.perf_counter() - t0

    stats = {
        "static_instr": len(vm_prog),
        "compile_time_s": compile_time_s,
    }

    return ast_prog, ir_prog, vm_prog, stats


def compile(
    source: str, opt_level: int = 0
) -> tuple[list[IRInst], list[VMInstruction], dict[str, Any]]:
    """Pipeline convenience function returning IR, VM bytecode, and stats."""
    _, ir_prog, vm_prog, stats = compile_program(source, opt_level=opt_level)
    return ir_prog, vm_prog, stats


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="MiniLang compiler and execution CLI."
    )
    arg_parser.add_argument("file", help="Path to MiniLang source file (.ml)")
    arg_parser.add_argument("--dump-ir", action="store_true", help="Dump linear TAC IR")
    arg_parser.add_argument("--dump-asm", action="store_true", help="Dump assembled VM instructions")
    arg_parser.add_argument("--run", action="store_true", help="Execute program on MiniLang VM")
    arg_parser.add_argument("--oracle", action="store_true", help="Execute program on AST oracle interpreter")

    args = arg_parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        sys.stderr.write(f"Error: file not found '{args.file}'\n")
        sys.exit(1)

    source = file_path.read_text(encoding="utf-8")
    ast_prog, ir_prog, vm_prog, stats = compile_program(source, opt_level=0)

    if args.dump_ir:
        print("=== Intermediate Representation (TAC) ===")
        print(dump_ir(ir_prog))
        print()

    if args.dump_asm:
        print("=== Assembled VM Bytecode ===")
        print(dump_asm(vm_prog))
        print()

    if args.oracle:
        print("=== Running on AST Oracle Interpreter ===")
        oracle = ASTInterpreter()
        output = oracle.run(ast_prog)
        for line in output:
            print(line)

    if args.run or (not args.dump_ir and not args.dump_asm and not args.oracle):
        if args.oracle:
            print("=== Running on MiniLang Virtual Machine ===")
        vm = VirtualMachine()
        output, dyn_count = vm.run(vm_prog)
        for line in output:
            print(line)


if __name__ == "__main__":
    main()
