#!/usr/bin/env python3
"""
Powerful but simple command-line calculator
Features:
- Safe expression evaluation using AST (supports + - * / ** % //, parentheses)
- Math functions: sin, cos, tan, asin, acos, atan, log, log10, exp, sqrt, pow
- Constants: pi, e
- Complex number support using 'j' and complex() if needed
- Variables: let x = 3*sqrt(2)
- History and variables persistence (save/load)
- A few unit conversions (length, temperature)
- Commands: :help, :vars, :history, :let, :save, :load, :quit

Usage: run `python3 power_calculator.py` then type expressions or commands.
"""

import ast
import math
import cmath
import readline
import json
import sys
from typing import Any, Dict, List


class SafeEvaluator(ast.NodeVisitor):
    """Evaluate a mathematical expression parsed to AST safely."""

    ALLOWED_NODE_TYPES = (
        ast.Expression,
        ast.UnaryOp,
        ast.BinOp,
        ast.Num,
        ast.Constant,  # for Python 3.8+
        ast.Name,
        ast.Call,
        ast.Load,
        ast.Expr,
        ast.Pow,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Tuple,
    )

    def __init__(self, variables: Dict[str, Any], functions: Dict[str, Any]):
        self.vars = variables
        self.funcs = functions

    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODE_TYPES):
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Num(self, node: ast.Num):
        return node.n

    def visit_Name(self, node: ast.Name):
        id_ = node.id
        if id_ in self.vars:
            return self.vars[id_]
        if id_ in self.funcs:
            return self.funcs[id_]
        raise NameError(f"Unknown identifier: {id_}")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Pow):
            return left ** right
        raise ValueError(f"Unsupported binary operator: {type(op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call):
        # Support only simple calls: func(arg, ...)
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls allowed")
        func_name = node.func.id
        if func_name not in self.funcs:
            raise NameError(f"Unknown function: {func_name}")
        func = self.funcs[func_name]
        args = [self.visit(a) for a in node.args]
        # disallow keywords for simplicity
        return func(*args)


# Build environment
DEFAULT_FUNCS = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
# add cmath functions for complex support
CMATH_FUNCS = {k: getattr(cmath, k) for k in [
    'phase', 'polar', 'rect', 'isfinite', 'isnan'
] if hasattr(cmath, k)}

# add safe wrappers
DEFAULT_FUNCS.update({
    'pow': pow,
    'abs': abs,
    'round': round,
    'sqrt': math.sqrt,
})
DEFAULT_FUNCS.update(CMATH_FUNCS)

CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau if hasattr(math, 'tau') else 2 * math.pi,
    'i': 1j,
    'j': 1j,
}


class Calculator:
    def __init__(self):
        self.vars: Dict[str, Any] = {}
        self.vars.update(CONSTANTS)
        self.funcs = DEFAULT_FUNCS
        self.history: List[str] = []

    def evaluate(self, expression: str):
        expression = expression.strip()
        # Allow entering complex numbers like 3+4j
        try:
            tree = ast.parse(expression, mode='eval')
            evaluator = SafeEvaluator(self.vars, self.funcs)
            result = evaluator.visit(tree)
            return result
        except Exception as e:
            # Fallback: try eval in a controlled namespace for simple cases
            # (this should rarely be needed)
            raise

    def set_var(self, name: str, expr: str):
        if not name.isidentifier():
            raise ValueError('Invalid variable name')
        val = self.evaluate(expr)
        self.vars[name] = val
        return val

    def add_history(self, entry: str):
        self.history.append(entry)
        if len(self.history) > 200:
            self.history.pop(0)

    # small unit converters
    def convert(self, amount: float, from_unit: str, to_unit: str):
        # length: m, cm, mm, km, in, ft
        length_factors = {
            'm': 1.0,
            'cm': 0.01,
            'mm': 0.001,
            'km': 1000.0,
            'in': 0.0254,
            'ft': 0.3048,
        }
        if from_unit in length_factors and to_unit in length_factors:
            meters = amount * length_factors[from_unit]
            return meters / length_factors[to_unit]
        # temperature
        if from_unit == 'C' and to_unit == 'F':
            return amount * 9 / 5 + 32
        if from_unit == 'F' and to_unit == 'C':
            return (amount - 32) * 5 / 9
        raise ValueError('Unsupported conversion')

    def save_state(self, filename: str):
        payload = {
            'vars': {k: (v.real if isinstance(v, complex) and v.imag == 0 else v) for k, v in self.vars.items() if k not in CONSTANTS},
            'history': self.history,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_state(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        for k, v in payload.get('vars', {}).items():
            self.vars[k] = v
        self.history = payload.get('history', [])


HELP_TEXT = '''
Simple powerful calculator commands:
- Type any math expression, e.g. 2+2, sin(pi/4), 3+4j, (2+3)*4
- let <name> = <expression>  -> create variable
- :vars        -> list user variables
- :history     -> show recent expressions
- :save <file> -> save variables & history to a file
- :load <file> -> load variables & history from a file
- :convert <amount> <from> <to> -> convert units (m, cm, mm, km, in, ft, C, F)
- :help        -> show this help
- :quit or :exit -> exit

Examples:
let r = 5
pi * r**2
:convert 10 ft m
'''


def repl():
    calc = Calculator()
    print("Powerful Python Calculator — type :help for commands")
    while True:
        try:
            line = input('calc> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break
        if not line:
            continue
        if line.startswith(':'):
            parts = line[1:].split()
            cmd = parts[0] if parts else ''
            args = parts[1:]
            if cmd in ('quit', 'exit'):
                print('Bye!')
                break
            if cmd == 'help':
                print(HELP_TEXT)
                continue
            if cmd == 'vars':
                user_vars = {k: v for k, v in calc.vars.items() if k not in CONSTANTS}
                if not user_vars:
                    print('(no user variables)')
                else:
                    for k, v in user_vars.items():
                        print(f'{k} = {v}')
                continue
            if cmd == 'history':
                for i, h in enumerate(calc.history[-50:], start=1):
                    print(f'{i}: {h}')
                continue
            if cmd == 'save' and args:
                try:
                    calc.save_state(args[0])
                    print('Saved to', args[0])
                except Exception as e:
                    print('Save failed:', e)
                continue
            if cmd == 'load' and args:
                try:
                    calc.load_state(args[0])
                    print('Loaded from', args[0])
                except Exception as e:
                    print('Load failed:', e)
                continue
            if cmd == 'convert' and len(args) == 3:
                try:
                    amt = float(args[0])
                    out = calc.convert(amt, args[1], args[2])
                    print(out)
                except Exception as e:
                    print('Conversion error:', e)
                continue
            print('Unknown command. Type :help')
            continue

        # assignment: let x = expr  OR x = expr
        if line.startswith('let '):
            try:
                rest = line[4:]
                if '=' not in rest:
                    print('Use: let name = expression')
                    continue
                name, expr = rest.split('=', 1)
                name = name.strip()
                expr = expr.strip()
                val = calc.set_var(name, expr)
                calc.add_history(f'{name} = {expr} -> {val}')
                print(f'{name} = {val}')
            except Exception as e:
                print('Error:', e)
            continue
        if '=' in line and not line.startswith('=='):
            try:
                name, expr = line.split('=', 1)
                name = name.strip()
                expr = expr.strip()
                val = calc.set_var(name, expr)
                calc.add_history(f'{name} = {expr} -> {val}')
                print(f'{name} = {val}')
            except Exception as e:
                print('Error:', e)
            continue

        # evaluate expression
        try:
            result = calc.evaluate(line)
            calc.add_history(f'{line} = {result}')
            print(result)
        except Exception as e:
            print('Evaluation error:', e)


if __name__ == '__main__':
    repl()
