"""Hand-written, single-change AST mutant generation primitives."""
import ast


class SingleChange(ast.NodeTransformer):
    def __init__(self, candidate):
        self.candidate = candidate
        self.changed = False

    def _hit(self, node, kind):
        return (not self.changed and (type(node).__name__, getattr(node, "lineno", -1),
                getattr(node, "col_offset", -1), kind) == self.candidate)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and self._hit(node, "compare"):
            replacements = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
                            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot,
                            ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
            op = node.ops[0]
            if type(op) in replacements:
                node.ops[0] = replacements[type(op)]()
                self.changed = True
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        replacements = {ast.Add: ast.Sub, ast.Sub: ast.Add}
        if self._hit(node, "binop") and type(node.op) in replacements:
            node.op = replacements[type(node.op)]()
            self.changed = True
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        replacements = {ast.And: ast.Or, ast.Or: ast.And}
        if self._hit(node, "boolop") and type(node.op) in replacements:
            node.op = replacements[type(node.op)]()
            self.changed = True
        return node

    def visit_Constant(self, node):
        if type(node.value) is int and self._hit(node, "constant"):
            node.value += 1
            self.changed = True
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self._hit(node, "if"):
            node.test = ast.copy_location(ast.Constant(value=False), node.test)
            self.changed = True
        return node


def mutation_candidates(source):
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn):
            kind = "compare"
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub):
            kind = "binop"
        elif isinstance(node, ast.BoolOp) and type(node.op) in (ast.And, ast.Or):
            kind = "boolop"
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            kind = "constant"
        elif isinstance(node, ast.If):
            kind = "if"
        else:
            continue
        opname = type(node.ops[0]).__name__ if isinstance(node, ast.Compare) else (type(node.op).__name__ if hasattr(node, "op") else kind)
        out.append({"candidate": (type(node).__name__, node.lineno, node.col_offset, kind), "line": node.lineno,
                    "operator": opname, "description": f"{kind}: {opname} at line {node.lineno}"})
    return out


def mutate_source(source, candidate):
    tree = ast.parse(source)
    transformer = SingleChange(candidate)
    changed = transformer.visit(tree)
    if not transformer.changed:
        raise ValueError("candidate did not produce a mutation")
    ast.fix_missing_locations(changed)
    result = ast.unparse(changed) + "\n"
    compile(result, "<mutant>", "exec")
    return result
