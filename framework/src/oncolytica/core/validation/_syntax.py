from __future__ import annotations

import ast
import re
import inspect
from typing import Any, Dict, List, Optional, FrozenSet

from oncolytica.core.utils._types import (  # noqa: E402
    Tissue, Chemistry, Cell, Metrics,
    PRIMITIVE_TYPES, BASE_CLASSES,
)

from oncolytica.core.utils._errors import CompilationError
from ._base import BaseValidator
from ._context import ValidationContext, RULE_DECORATOR_NAMES, TupleType, resolve_tuple_annotation


# ============================================================================
# PHASE 1-A – SYNTAX VALIDATOR
# ============================================================================

class SyntaxValidator(BaseValidator):
    """Phase 1b: DSL structure and WGSL-compatibility enforcement."""

    _MACRO_ITER_ATTRS: FrozenSet[str] = frozenset({"cells", "tissues", "neighbors"})

    def __init__(self) -> None:
        super().__init__()
        self._classdef_depth: int = 0
        self._in_annotation: bool = False

    def visit_Import(self, node: ast.Import) -> None:
        raise CompilationError(f"'import' statements are forbidden inside a Simulation class.", node=node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise CompilationError(
            f"'from … import' statements are forbidden inside a Simulation class.", node=node)

    def visit_Try(self, node: ast.Try) -> None:
        raise CompilationError(f"try/except blocks are forbidden.", node=node)

    def visit_With(self, node: ast.With) -> None:
        raise CompilationError(f"'with' statements are forbidden.", node=node)

    def visit_For(self, node: ast.For) -> None:
        if not self._is_allowed_for(node):
            raise CompilationError(
                f"Forbidden for-loop form. "
                f"Allowed: 'for i in range(n)' or "
                f"'for x in obj.{{cells,tissues,neighbors}}'."
            )
        self.generic_visit(node)

    def _is_allowed_for(self, node: ast.For) -> bool:
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and isinstance(node.target, ast.Name)):
            return True
        if (isinstance(node.iter, ast.Attribute)
                and node.iter.attr in self._MACRO_ITER_ATTRS
                and isinstance(node.target, ast.Name)):
            return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._classdef_depth += 1
        if self._classdef_depth > 1:
            raise CompilationError(f"Nested class definitions are forbidden.", node=node)
        for stmt in node.body:
            self.visit(stmt)
        self._classdef_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Visit the return annotation (node.returns) in annotation context.
        if node.returns is not None:
            self._in_annotation = True
            self.visit(node.returns)
            self._in_annotation = False
        # Visit args (annotations handled in visit_arg below).
        for arg in node.args.args:
            self.visit(arg)
        # Visit decorators and body normally.
        for dec in node.decorator_list:
            self.visit(dec)
        for stmt in node.body:
            self.visit(stmt)

    def visit_arg(self, node: ast.arg) -> None:
        # Visit the parameter annotation in annotation context so that
        # tuple[f32, i32] slices (ast.Tuple inside ast.Subscript) are allowed.
        if node.annotation is not None:
            self._in_annotation = True
            self.visit(node.annotation)
            self._in_annotation = False

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise CompilationError(f"Lambda expressions are forbidden.", node=node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        raise CompilationError(f"f-strings are forbidden.", node=node)

    def visit_Dict(self, node: ast.Dict) -> None:
        raise CompilationError(f"Dictionary literals are forbidden.", node=node)

    def visit_Set(self, node: ast.Set) -> None:
        raise CompilationError(f"Set literals are forbidden.", node=node)

    def visit_List(self, node: ast.List) -> None:
        raise CompilationError(f"List literals are forbidden.", node=node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        # Allowed in annotation context: tuple[f32, i32] slice.
        if self._in_annotation:
            return
        # Tuple nodes on the LHS of an unpacking assignment are allowed:
        #   a, b = func()
        # All other tuple literals (RHS, nested, standalone) remain forbidden.
        # LHS tuples are identified by their Store context.
        if isinstance(node.ctx, ast.Store):
            # Validate that every element is a plain Name (no nested tuples).
            for elt in node.elts:
                if not isinstance(elt, ast.Name):
                    raise CompilationError(
                        f"Only simple names are allowed in tuple unpacking targets "
                        f"(e.g. 'a, b = func()'). Nested unpacking is forbidden.",
                        node=node,
                    )
            return  # valid — let generic_visit recurse into children
        raise CompilationError(
            f"Tuple literals are forbidden. "
            f"To unpack a function return value use: a, b = func(). "
            f"For vectors use vec3 or ivec3.", node=node
        )

    def visit_Return(self, node: ast.Return) -> None:
        # return x, y  produces ast.Return(value=ast.Tuple(ctx=Load)).
        # This is the only place where a Load-context Tuple is allowed —
        # it represents a multi-value return, not a stored tuple literal.
        # We validate the elements individually instead of letting
        # generic_visit fall into visit_Tuple which would reject them.
        if isinstance(node.value, ast.Tuple):
            if not isinstance(node.value.ctx, ast.Load):
                raise CompilationError(
                    "Only 'return x, y' form is allowed for multi-value returns.",
                    node=node,
                )
            for elt in node.value.elts:
                if isinstance(elt, ast.Tuple):
                    raise CompilationError(
                        "Nested tuples in return values are forbidden.",
                        node=node,
                    )
                self.visit(elt)
            return  # do NOT call generic_visit — that would hit visit_Tuple
        # Single-value or bare return — visit normally.
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise CompilationError(f"List comprehensions are forbidden.", node=node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise CompilationError(f"Dict comprehensions are forbidden.", node=node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise CompilationError(f"Set comprehensions are forbidden.", node=node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise CompilationError(f"Generator expressions are forbidden.", node=node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) > 1:
            raise CompilationError(f"Cascading assignment (a = b = c) is forbidden.", node=node)
        if isinstance(node.targets[0], ast.List):
            raise CompilationError(f"List-unpacking assignment is forbidden.", node=node)
        # ast.Tuple on the LHS is tuple unpacking — validated in visit_Tuple.
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.comparators) > 1:
            raise CompilationError(f"Chained comparisons (e.g. '0 < x < 1') are forbidden.", node=node)
        self.generic_visit(node)


# ============================================================================
# PHASE 1-B – NAMING VALIDATOR
# ============================================================================

class NamingValidator(BaseValidator):
    _SNAKE_RE: re.Pattern[str] = re.compile(r'^[a-z][a-z0-9_]*$')

    def _is_snake(self, name: str) -> bool:
        return bool(self._SNAKE_RE.match(name)) and name[0] != '_'

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._is_snake(node.name):
            raise CompilationError(f"Method name '{node.name}' must be snake_case.", node=node)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg != "self" and not self._is_snake(node.arg):
            raise CompilationError(f"Parameter '{node.arg}' must be snake_case.", node=node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name.isupper():
                raise CompilationError(f"Assignment to UPPER_CASE name '{name}' is forbidden.", node=node)
            if not self._is_snake(name):
                raise CompilationError(f"Variable '{name}' must be snake_case.", node=node)
        elif isinstance(node.ctx, ast.Load):
            if name.isupper() and self.ctx and name not in self.ctx.constants:
                raise CompilationError(f"Unknown constant '{name}'.", node=node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if not self._is_snake(node.attr):
                raise CompilationError(f"Attribute 'self.{node.attr}' must be strictly snake_case.", node=node)
        self.generic_visit(node)


# ============================================================================
# PHASE 1-С – SIGNATURE VALIDATOR
# ============================================================================

class SignatureValidator(BaseValidator):
    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        for name, node in ctx.method_nodes.items():
            hints = ctx.method_type_hints.get(name, {})
            self._validate_signature(name, node, hints)

    def _validate_signature(
            self,
            name: str,
            node: ast.FunctionDef,
            hints: Dict[str, Any],
    ) -> None:

        # 1. READ ARGUMENTS DIRECTLY FROM AST (Bypasses Python decorators entirely)
        params = [arg.arg for arg in node.args.args]

        if not params or params[0] != "self":
            raise CompilationError(
                f"Method '{name}' must declare 'self' as its first argument.", node=node
            )

        rule_decs: List[str] = [
            self._extract_dec_name(d)
            for d in node.decorator_list
        ]
        rule_decs = [r for r in rule_decs if r and r in RULE_DECORATOR_NAMES]

        if len(node.decorator_list) > 1:
            raise CompilationError(f"Method '{name}' uses more than 1 decorator.", node=node)

        if node.decorator_list and not rule_decs:
            raise CompilationError(
                f"Method '{name}' has an unrecognized decorator "
                f"'{self._extract_dec_name(node.decorator_list[0])}'. "
                f"Only @ol.*_rule decorators are allowed.", node=node
            )

        for param in params[1:]:
            if param not in hints:
                raise CompilationError(
                    f"Parameter '{param}' in '{name}' lacks a type annotation.", node=node
                )
            self._assert_valid_annotation(hints[param], node, param, name)

        if len(rule_decs) == 1:
            self._check_rule_signature(name, node, rule_decs[0], params, hints)

    def _assert_valid_annotation(
            self,
            t: Any,
            node: ast.AST,
            param: str,
            method: str,
    ) -> None:
        if t in PRIMITIVE_TYPES:
            return
        if isinstance(t, TupleType):
            # Each element must itself be a valid primitive or domain type.
            for elem in t.elements:
                self._assert_valid_annotation(elem, node, param, method)
            return
        if isinstance(t, type) and any(issubclass(t, b) for b in BASE_CLASSES):
            return
        raise CompilationError(
            f"Parameter '{param}' in '{method}' has unsupported type '{t!r}'.", node=node
        )

    def _check_rule_signature(
            self,
            name: str,
            node: ast.FunctionDef,
            rule_name: str,
            params: List[str],
            hints: Dict[str, Any],
    ) -> None:
        def need(n: int, spec: str) -> None:
            if len(params) != n:
                raise CompilationError(
                    f"@{rule_name} '{name}' needs {n} parameter(s) {spec}; got {len(params)}.", node=node
                )

        def expect_base(p: str, base: type) -> None:
            t = hints.get(p)
            if t and (not isinstance(t, type) or not issubclass(t, base)):
                raise CompilationError(
                    f"Parameter '{p}' in '{name}' must be a "
                    f"subclass of {base.__name__}, got '{getattr(t, '__name__', t)!r}'.", node=node
                )

        if rule_name == "cell_rule":
            need(2, "(self, cell)")
            expect_base(params[1], Cell)
        elif rule_name == "tissue_rule":
            need(2, "(self, voxel)")
            expect_base(params[1], Tissue)
        elif rule_name == "chemistry_rule":
            need(2, "(self, chem)")
            expect_base(params[1], Chemistry)
        elif rule_name == "metric_rule":
            need(3, "(self, item, metrics)")
            self._check_metric_item(hints.get(params[1]), node, name)
            expect_base(params[2], Metrics)

    def _check_metric_item(self, t: Any, node: ast.AST, method: str) -> None:
        ITEM_BASES = (Cell, Tissue, Chemistry)
        if not isinstance(t, type) or not any(issubclass(t, b) for b in ITEM_BASES):
            raise CompilationError(
                f"The first parameter in '{method}' must be a "
                f"Cell, Tissue, or Chemistry subclass; got '{getattr(t, '__name__', t)!r}'.", node=node
            )

    @staticmethod
    def _extract_dec_name(dec: ast.expr) -> Optional[str]:
        node = dec.func if isinstance(dec, ast.Call) else dec
        return getattr(node, "attr", getattr(node, "id", None))
