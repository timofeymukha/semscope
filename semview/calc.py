"""Field calculator: arithmetic expressions on nodal data with spectrally exact
derivatives of arbitrary order.

Expressions use a small, safe subset of Python syntax::

    sqrt(u^2 + v^2)              # arithmetic: + - * / ^ (or **), parentheses
    dx(v) - dy(u)                # first derivatives d/dx, d/dy
    dx(u, 2) + dy(u, 2)          # second derivatives
    dx(dy(p))                    # mixed derivatives by nesting
    0.5 * (u^2 + v^2)            # numbers and constants (pi, e)

Names refer to the fields stored in the file, to fields defined earlier with
the calculator, and to the coordinates ``x`` and ``y``.  Derivatives are
formed element by element from the Lagrange basis and the geometry map, so
they are exact for the polynomial the data represents (and, as usual for CG
data, may jump between elements).
"""

from __future__ import annotations

import ast
import re

import numpy as np

__all__ = [
    "ExpressionError",
    "FUNCTIONS",
    "BINARY_FUNCTIONS",
    "DERIVATIVES",
    "CONSTANTS",
    "COORDINATES",
    "RESERVED",
    "syntax_help",
    "is_identifier",
    "parse",
    "free_names",
    "evaluate",
    "Expressions",
]


class ExpressionError(ValueError):
    """Malformed expression, unknown name, or circular definition."""


FUNCTIONS = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10, "abs": np.abs, "sign": np.sign,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
}
BINARY_FUNCTIONS = {"atan2": np.arctan2, "min": np.minimum, "max": np.maximum, "pow": np.power}
DERIVATIVES = {"dx": 0, "dy": 1}   # name -> physical direction
CONSTANTS = {"pi": np.pi, "e": np.e}
COORDINATES = ("x", "y")
RESERVED = set(FUNCTIONS) | set(BINARY_FUNCTIONS) | set(DERIVATIVES) | set(CONSTANTS) | set(COORDINATES)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod}
_UNARY = {ast.USub: np.negative, ast.UAdd: lambda a: a}
_OP_NAMES = {ast.BitXor: "^", ast.BitAnd: "&", ast.BitOr: "|", ast.FloorDiv: "//", ast.MatMult: "@", ast.LShift: "<<", ast.RShift: ">>"}


def syntax_help() -> dict:
    """Machine-readable description of the expression language (for the GUI)."""
    return {
        "operators": ["+", "-", "*", "/", "^"],
        "derivatives": [f"{d}(f)" for d in DERIVATIVES] + [f"{d}(f, order)" for d in DERIVATIVES],
        "functions": sorted(FUNCTIONS),
        "binary_functions": sorted(BINARY_FUNCTIONS),
        "constants": sorted(CONSTANTS),
        "coordinates": list(COORDINATES),
    }


def is_identifier(name: str) -> bool:
    return isinstance(name, str) and bool(_IDENT.match(name))


def check_name(name: str, stored: list[str] | tuple[str, ...] | set[str] = ()) -> str:
    """Validate the name of a calculator field; returns the stripped name."""
    name = (name or "").strip()
    if not is_identifier(name):
        raise ExpressionError(f"{name!r} is not a valid field name (letters, digits and _ only, not starting with a digit)")
    if name in RESERVED:
        raise ExpressionError(f"{name!r} is reserved (function, constant or coordinate)")
    if name in stored:
        raise ExpressionError(f"{name!r} is a field of the file and cannot be redefined")
    return name


# --------------------------------------------------------------------------- parsing
def parse(expr: str) -> ast.expr:
    """Parse and validate an expression; returns the AST of its body."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("empty expression")
    src = expr.replace("^", "**")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as exc:
        col = exc.offset or 0
        raise ExpressionError(f"syntax error near column {col}" + (f": {exc.text.strip()[max(0, col - 6):col + 4]!r}" if exc.text else "")) from None
    _validate(tree.body)
    return tree.body


def _validate(node: ast.AST) -> None:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _OPS:
            raise ExpressionError(f"operator {_OP_NAMES.get(type(node.op), type(node.op).__name__)!r} is not supported")
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY:
            raise ExpressionError(f"operator {type(node.op).__name__} is not supported")
        _validate(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only plain function calls such as sqrt(u) are allowed")
        fn = node.func.id
        if node.keywords:
            raise ExpressionError(f"{fn}(): keyword arguments are not supported")
        n = len(node.args)
        if fn in DERIVATIVES:
            if n not in (1, 2):
                raise ExpressionError(f"{fn}(f) or {fn}(f, order) expected")
            if n == 2:
                o = node.args[1]
                if not (isinstance(o, ast.Constant) and isinstance(o.value, int) and not isinstance(o.value, bool) and o.value >= 1):
                    raise ExpressionError(f"the order in {fn}(f, order) must be a positive integer")
            _validate(node.args[0])
            return
        if fn in FUNCTIONS:
            if n != 1:
                raise ExpressionError(f"{fn}() takes exactly one argument")
        elif fn in BINARY_FUNCTIONS:
            if n != 2:
                raise ExpressionError(f"{fn}() takes exactly two arguments")
        else:
            raise ExpressionError(f"unknown function {fn}()")
        for a in node.args:
            _validate(a)
    elif isinstance(node, ast.Name):
        return
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("only numbers are allowed as literals")
    else:
        raise ExpressionError(f"{type(node).__name__} is not allowed in a field expression")


def free_names(node: ast.AST | str) -> set[str]:
    """Names an expression refers to (fields and coordinates; constants excluded)."""
    if isinstance(node, str):
        node = parse(node)
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id not in CONSTANTS and n.id not in FUNCTIONS and n.id not in BINARY_FUNCTIONS and n.id not in DERIVATIVES:
            out.add(n.id)
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.discard(n.func.id)
    return out


# --------------------------------------------------------------------------- evaluation
def evaluate(node: ast.AST | str, resolve, derivative):
    """Evaluate an expression.

    ``resolve(name)`` returns the nodal array of a field (or coordinate);
    ``derivative(values, direction, order)`` returns the ``order``-th physical
    derivative (direction 0 = x, 1 = y).  Scalars stay scalars until they
    meet an array.
    """
    if isinstance(node, str):
        node = parse(node)

    def ev(n):
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in CONSTANTS:
                return CONSTANTS[n.id]
            return resolve(n.id)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            with np.errstate(all="ignore"):
                return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp):
            return _UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call):
            fn = n.func.id
            if fn in DERIVATIVES:
                order = int(n.args[1].value) if len(n.args) == 2 else 1
                return derivative(ev(n.args[0]), DERIVATIVES[fn], order)
            with np.errstate(all="ignore"):
                if fn in FUNCTIONS:
                    return FUNCTIONS[fn](ev(n.args[0]))
                return BINARY_FUNCTIONS[fn](ev(n.args[0]), ev(n.args[1]))
        raise ExpressionError(f"cannot evaluate {type(n).__name__}")

    return ev(node)


# --------------------------------------------------------------------------- definitions
class Expressions(dict):
    """Ordered ``name -> expression`` definitions shared by the steps of a dataset.

    ``version`` increases on every change so cached evaluations can be dropped.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = 0

    def define(self, name: str, expr: str, stored=()) -> str:
        """Add or replace a definition after checking syntax, names and cycles."""
        name = check_name(name, stored)
        node = parse(expr)
        known = set(stored) | set(COORDINATES) | set(self) | {name}
        unknown = sorted(free_names(node) - known)
        if unknown:
            raise ExpressionError(f"unknown field{'s' if len(unknown) > 1 else ''} {', '.join(repr(u) for u in unknown)}")
        trial = dict(self)
        trial[name] = expr
        _check_cycles(trial, name)
        self[name] = expr.strip()
        self.version += 1
        return name

    def remove(self, name: str) -> None:
        if name not in self:
            raise KeyError(name)
        deps = self.dependents(name)
        if deps:
            raise ExpressionError(f"{name!r} is used by {', '.join(repr(d) for d in deps)}")
        del self[name]
        self.version += 1

    def dependents(self, name: str) -> list[str]:
        """Definitions that refer to ``name`` directly."""
        return [k for k, e in self.items() if k != name and name in free_names(e)]

    def to_list(self) -> list[dict]:
        return [{"name": k, "expr": v} for k, v in self.items()]


def _check_cycles(defs: dict, start: str) -> None:
    stack: list[str] = []

    def visit(name):
        if name in stack:
            cyc = stack[stack.index(name):] + [name]
            raise ExpressionError("circular definition: " + " -> ".join(cyc))
        if name not in defs:
            return
        stack.append(name)
        for dep in free_names(defs[name]):
            visit(dep)
        stack.pop()

    visit(start)
