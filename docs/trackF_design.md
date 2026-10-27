# Track F: MiniLang Compiler Optimization & Design Specification

## Overview

MiniLang is an imperative, statically checked expression-and-statement language designed to explore compiler optimization passes, intermediate representation (IR) lowering, and virtual machine execution metrics. It supports structured control flow, basic arithmetic, 64-bit integer semantics, and standard I/O via print statements.

---

## 1. Concrete Syntax & Grammar (EBNF)

MiniLang programs consist of a sequence of statements. Statements are terminated by semicolons `;` (except compound blocks), and braces `{ ... }` delimit statement blocks.

### Lexical Conventions
- **Whitespace & Comments:** Whitespace (spaces, tabs, newlines) is ignored. Comments begin with `//` and extend to the end of the line.
- **Identifiers:** `[a-zA-Z_][a-zA-Z0-9_]*`, excluding reserved keywords (`if`, `else`, `while`, `print`).
- **Integer Literals:** Sequences of decimal digits `[0-9]+`.

### EBNF Grammar

```ebnf
program         = { statement } ;

statement       = assign_stmt
                | print_stmt
                | if_stmt
                | while_stmt
                | block_stmt ;

assign_stmt     = IDENT "=" expr ";" ;
print_stmt      = "print" expr ";" ;
if_stmt         = "if" "(" expr ")" statement [ "else" statement ] ;
while_stmt      = "while" "(" expr ")" statement ;
block_stmt      = "{" { statement } "}" ;

expr            = comparison ;

comparison      = addition { ( "<" | "<=" | ">" | ">=" | "==" | "!=" ) addition } ;

addition        = multiplication { ( "+" | "-" ) multiplication } ;

multiplication  = unary { ( "*" | "/" | "%" ) unary } ;

unary           = ( "-" ) unary
                | primary ;

primary         = INT
                | IDENT
                | "(" expr ")" ;
```

### Operator Precedence & Associativity
From lowest to highest binding precedence:
1. **Comparison** (`<`, `<=`, `>`, `>=`, `==`, `!=`): Non-associative / left-associative.
2. **Additive** (`+`, `-`): Left-associative.
3. **Multiplicative** (`*`, `/`, `%`): Left-associative.
4. **Unary** (`-`): Prefix, right-associative.
5. **Primary** (literals, variables, parenthesized sub-expressions): Highest.

---

## 2. Formal Semantics

### Data Representation & Arithmetic
- **64-bit Signed Integers:** All variables, literals, and intermediate values are represented as signed 64-bit integers in two's-complement form ($[-2^{63}, 2^{63} - 1]$).
- **Two's-Complement Wraparound:** Any arithmetic operation (`+`, `-`, `*`) that exceeds 64-bit boundaries wraps around symmetrically modulo $2^{64}$:
  $$\text{wrap64}(x) = ((x + 2^{63}) \bmod 2^{64}) - 2^{63}$$
- **Truncated Division and Modulo (C99 / C++ Semantics):**
  - Integer division `/` and remainder `%` truncate toward zero:
    $$\text{div\_trunc}(a, b) = \text{sgn}(a \cdot b) \cdot \lfloor |a| / |b| \rfloor$$
    $$\text{mod\_trunc}(a, b) = a - \text{div\_trunc}(a, b) \cdot b$$
  - Examples:
    - `-7 / 2` evaluates to `-3`
    - `-7 % 2` evaluates to `-1`
    - `7 / -2` evaluates to `-3`
    - `7 % -2` evaluates to `1`
- **Division and Modulo by Zero:** Dividing or computing remainder with a divisor of zero raises an immediate, non-ignorable runtime error.
- **Relational Operations:** Comparison operators yield boolean values encoded as integers: `1` for true and `0` for false.
- **Truthiness:** In `if` and `while` conditions, `0` is interpreted as false; any non-zero integer is interpreted as true.
- **Variable Initialization:** Reading or referencing an uninitialized variable triggers a compile-time (static validation) error.
- **Standard Output:** The `print` statement outputs the evaluated integer value followed by a newline.

---

## 3. Compiler Pipeline Architecture

The MiniLang compilation pipeline follows a classical multi-tier architecture:

```
[Source .ml]
     │
     ▼ (lexer.py)
[Token Stream]
     │
     ▼ (parser.py)
[Abstract Syntax Tree (AST)] ───► [ast_interp.py (AST Reference Oracle)]
     │
     ▼ (irgen.py)
[Linear Three-Address Code (IR)]
     │
     ▼ (codegen.py)
[VM Bytecode Instructions]
     │
     ▼ (vm.py)
[Virtual Machine Execution Engine]
```

1. **Lexical Analysis (`lexer.py`):** Converts raw UTF-8 source text into a stream of typed tokens with accurate 1-indexed line and column metadata for error reporting.
2. **Parsing & AST Construction (`parser.py`, `ast_nodes.py`):** Recursive descent parser enforcing the EBNF grammar and building strongly-typed dataclass nodes. Performs uninitialized variable verification.
3. **AST Reference Interpreter (`ast_interp.py`):** Tree-walking reference engine executing directly over AST nodes, serving as an independent oracle for functional correctness.
4. **Intermediate Representation Generation (`irgen.py`, `ir.py`):** Lowers structured AST control flow into linear three-address code (TAC) with explicit virtual registers (`t0, t1, ...`), variable storage operations (`LOAD`, `STORE`), labels, and conditional/unconditional branches.
5. **Code Generation (`codegen.py`):** Assembles linear IR into an executable tuple-based VM instruction array, resolving symbolic labels into concrete program counter jump offsets.
6. **Virtual Machine (`vm.py`):** Stackless register/variable-store execution engine tracking exact instruction counts and enforcing runtime safety (e.g. division by zero).
7. **Pipeline Driver & CLI (`compiler.py`):** Coordinates pipeline execution, measures compile times, and provides CLI flags (`--dump-ir`, `--dump-asm`, `--run`, `--oracle`).

---

## 4. Measurement & Optimization Metric Plan

Every compilation and execution run is benchmarked across four fundamental metrics:

- **Static Instruction Count:** The total number of instructions in the emitted VM program before execution.
- **Dynamic Instruction Count:** The total count of VM instructions executed at runtime by the virtual machine interpreter loop.
- **Runtime (s):** Wall-clock execution time of the virtual machine running the compiled bytecode, measured using high-precision performance timers (`time.perf_counter`) over multiple runs (median of 5).
- **Compile Time (s):** Wall-clock duration required to lex, parse, validate, lower to IR, and emit VM bytecode (median of 5).
- **Functional Equivalence:** Strict equality check verifying that the VM output matches the AST reference oracle output identically on all test programs.
