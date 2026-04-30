from __future__ import annotations

import ast
import copy
from typing import Any

_VOID_METHODS: frozenset[str] = frozenset({"die", "kill", "divide"})
_RAND_METHODS: frozenset[str] = frozenset({"random", "random_dir"})


class RandExtractor(ast.NodeTransformer):
    def __init__(self, start_id: int):
        self.counter = start_id
        self.assignments: list[ast.Assign] = []

    def visit_Call(self, node: ast.Call) -> Any:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _RAND_METHODS:
            var_name = f"_rand_{self.counter}"
            self.counter += 1
            assign = ast.Assign(targets=[ast.Name(id=var_name, ctx=ast.Store())], value=node)
            self.assignments.append(assign)
            return ast.Name(id=var_name, ctx=ast.Load())
        return node


class FlattenRandPass(ast.NodeTransformer):
    def __init__(self) -> None:
        self.counter = 0

    def _flatten_stmts(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        new_stmts = []
        for stmt in stmts:
            res = self.visit(stmt)
            if isinstance(res, list): new_stmts.extend(res)
            else: new_stmts.append(res)
        return new_stmts

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.body = self._flatten_stmts(node.body)
        return node

    def visit_If(self, node: ast.If) -> Any:
        node.body = self._flatten_stmts(node.body)
        node.orelse = self._flatten_stmts(node.orelse)
        extractor = RandExtractor(self.counter)
        node.test = extractor.visit(copy.deepcopy(node.test))
        if extractor.assignments:
            self.counter = extractor.counter
            return extractor.assignments + [node]
        return node

    def visit_While(self, node: ast.While) -> Any:
        node.body = self._flatten_stmts(node.body)
        extractor = RandExtractor(self.counter)
        new_test = extractor.visit(copy.deepcopy(node.test))
        if not extractor.assignments: return node
        self.counter = extractor.counter
        if_break = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=new_test), body=[ast.Break()], orelse=[])
        node.test = ast.Constant(value=True)
        node.body = extractor.assignments + [if_break] + node.body
        return node

    def _hoist_simple_stmt(self, node: ast.stmt) -> Any:
        extractor = RandExtractor(self.counter)
        new_node = extractor.visit(copy.deepcopy(node))
        if extractor.assignments:
            self.counter = extractor.counter
            return extractor.assignments + [new_node]
        return node

    def visit_Assign(self, node: ast.Assign) -> Any: return self._hoist_simple_stmt(node)
    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any: return self._hoist_simple_stmt(node)
    def visit_Expr(self, node: ast.Expr) -> Any: return self._hoist_simple_stmt(node)
    def visit_Return(self, node: ast.Return) -> Any: return self._hoist_simple_stmt(node)


def apply_passes(func_def: ast.FunctionDef) -> ast.FunctionDef:
    func_def = copy.deepcopy(func_def)
    func_def = FlattenRandPass().visit(func_def)
    ast.fix_missing_locations(func_def)
    return func_def
