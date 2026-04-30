"""
oncolytica._validator
~~~~~~~~~~~~~~~~~~~~~
Static-analysis pipeline (compiler frontend) for user Simulation classes.

Pipeline  (5 Phases, Dragon-Book style)
----------------------------------------
  Phase 0   ContextBuilder
              Parse source → AST; extract signatures, fields, constants.
              Zero validation here – pure data extraction.

  Phase 1   Syntactic & Lexical  (fail fast on surface errors)
              NamingValidator  – snake_case / UPPER_SNAKE_CASE enforcement
              SyntaxValidator  – WGSL-incompatible construct rejection

  Phase 2   Symbol Definition & Scope  (know every name before using it)
              ScopeBuilder     – validates signatures & decorators,
                                 builds SymbolTable (ctx.symbol_table),
                                 detects constant shadowing

  Phase 3   Type Inference & Type Checking  (all expressions typed)
              TypeChecker      – bottom-up type inference, writes TypeMap
                                 (ctx.type_map), enforces strict WGSL
                                 type compatibility and return-type contracts

  Phase 4   Control Flow & Call-Graph  (structural correctness)
              CallGraphValidator – cycle / recursion detection,
                                   topological sort for WGSL emission

  Phase 5   Domain-Specific Semantics  (Oncolytica business rules)
              DomainValidator  – mutability constraints, valid field access,
                                 action-call ownership rules

Blackboard
----------
All validators communicate exclusively through ``ValidationContext``.
Phases may *add* derived data to it but must never mutate data that an
earlier phase already produced.  On the first error the pipeline raises
``CompilationError`` (Fail-Fast).
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
import typing
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from oncolytica.core._types import (  # noqa: E402
    i32, i64, u32, u64, f32, f64,
    bool as ol_bool,
    vec3,
    TissueData, ChemistryData, CellData, MetricsData,
)
from oncolytica.core._errors import CompilationError

# ============================================================================
# GLOBAL TYPE REGISTRY
# ============================================================================

ALLOWED_PRIMITIVE_TYPES: FrozenSet[Any] = frozenset({
    i32, i64, u32, u64, f32, f64, ol_bool, vec3,
    int, float, bool,
})

BASE_CLASSES: FrozenSet[type] = frozenset({
    CellData, TissueData, ChemistryData, MetricsData,
})

RULE_DECORATOR_NAMES: FrozenSet[str] = frozenset({
    "cell_rule", "tissue_rule", "chemistry_rule", "metric_rule",
})

# Valid built-in method names per framework base class.
AGENT_BUILTINS: Dict[type, FrozenSet[str]] = {
    CellData:      frozenset({"die", "kill", "divide", "neighbors"}),
    TissueData:    frozenset({"cells", "neighbors"}),
    ChemistryData: frozenset({"cells", "tissues", "neighbors"}),
    MetricsData:   frozenset(),
}

# Fields injected by the framework (not declared by the user).
SYSTEM_FIELDS: Dict[type, FrozenSet[str]] = {
    CellData: frozenset({"pos"}),
}

# Scalar type groups – types within the same group are mutually compatible;
# types from different groups require an explicit cast in WGSL.
_SCALAR_GROUPS: Tuple[FrozenSet[Any], ...] = (
    frozenset({i32, i64, u32, u64, int}),   # integer group
    frozenset({f32, f64, float}),            # floating-point group
    frozenset({ol_bool, bool}),              # boolean group
)

# Human-readable annotation name → resolved type (used by TypeChecker /
# ScopeBuilder when the annotation is a bare string or ast.Name node).
_ANNOTATION_NAMES: Dict[str, Any] = {
    "i32": i32, "i64": i64, "u32": u32, "u64": u64,
    "f32": f32, "f64": f64, "bool": ol_bool, "vec3": vec3,
    "int": int, "float": float,
}


# ============================================================================
# PHASE 0-A – VALIDATION CONTEXT  (Blackboard)
# ============================================================================

class ValidationContext:
    """
    Shared state object passed through every phase of the pipeline.

    Layers are added by each phase; earlier layers are never mutated.

        Phase 0  populates: source_code, tree, method_* dicts, class_fields,
                            class_builtins, constants, rule/helper names.
        Phase 2  adds:      symbol_table
        Phase 3  adds:      type_map
        Phase 4  adds:      call_graph, ordered_methods
    """

    def __init__(
            self,
            sim_instance: Any,
            memory_base_map: Dict[type, type],
    ) -> None:
        # ── Input ────────────────────────────────────────────────────────────
        self.sim_instance: Any = sim_instance
        self.memory_base_map: Dict[type, type] = memory_base_map

        # ── Phase 0 outputs ──────────────────────────────────────────────────
        self.source_code: str = ""
        self.tree: Optional[ast.Module] = None

        self.method_signatures:   Dict[str, inspect.Signature] = {}
        self.method_type_hints:   Dict[str, Dict[str, Any]] = {}
        self.method_return_hints: Dict[str, Optional[Any]] = {}

        self.class_fields:   Dict[type, Set[str]] = {}
        self.class_builtins: Dict[type, Set[str]] = {}
        self.constants:      Dict[str, Any] = {}

        self.method_nodes:        Dict[str, ast.FunctionDef] = {}
        self.rule_method_names:   Set[str] = set()
        self.helper_method_names: Set[str] = set()

        # ── Phase 2 output (ScopeBuilder) ────────────────────────────────────
        # method_name → {var_name: declared_type_or_None}
        self.symbol_table: Dict[str, Dict[str, Optional[Any]]] = {}

        # ── Phase 3 output (TypeChecker) ─────────────────────────────────────
        # id(ast_node) → resolved type  (populated for every typed expression)
        self.type_map: Dict[int, Any] = {}

        # ── Phase 4 output (CallGraphValidator) ──────────────────────────────
        self.call_graph:      Dict[str, Set[str]] = {}
        self.ordered_methods: List[str] = []

    # ── Convenience helpers ──────────────────────────────────────────────────

    def all_valid_attrs(self, cls: type) -> Set[str]:
        """All attribute names that are syntactically legal for ``cls``."""
        return self.class_fields.get(cls, set()) | self.class_builtins.get(cls, set())

    def lookup_memory_class_by_ast_name(self, name: str) -> Optional[type]:
        for cls in self.memory_base_map:
            if cls.__name__ == name:
                return cls
        return None

    def is_memory_class(self, cls: type) -> bool:
        return cls in self.memory_base_map

    def get_base_class(self, memory_cls: type) -> Optional[type]:
        return self.memory_base_map.get(memory_cls)

    def base_class_of(self, cls: Any) -> Optional[type]:
        """Return the direct Oncolytica base class (CellData etc.) or None."""
        if not isinstance(cls, type):
            return None
        for klass in cls.__mro__:
            if klass in BASE_CLASSES:
                return klass
        return None


# ============================================================================
# PHASE 0-B – CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """
    Entry point for Phase 0.

    Parses the user Simulation class and fills a ``ValidationContext``.
    No validation logic lives here – only data extraction.
    """

    @staticmethod
    def build(
            sim_instance: Any,
            memory_base_map: Dict[type, type],
    ) -> ValidationContext:
        ctx = ValidationContext(sim_instance, memory_base_map)
        sim_class = sim_instance.__class__
        ContextBuilder._validate_memory_base_map(ctx.memory_base_map)
        ContextBuilder._build_ast(ctx, sim_class)
        ContextBuilder._build_signatures(ctx, sim_class)
        ContextBuilder._build_class_metadata(ctx)
        ContextBuilder._build_constants(ctx)
        ContextBuilder._classify_methods(ctx)
        return ctx

    # ── Memory base-map validation ─────────────────────────────────────────

    @staticmethod
    def _validate_memory_base_map(memory_base_map: Dict[type, type]) -> None:
        for cls, expected_base in memory_base_map.items():
            actual_bases = [b for b in cls.__bases__ if b in BASE_CLASSES]
            if len(actual_bases) == 0:
                raise CompilationError(
                    f"Memory class '{cls.__name__}' must inherit directly from exactly one of: "
                    f"{', '.join(b.__name__ for b in BASE_CLASSES)}"
                )
            if len(actual_bases) > 1:
                raise CompilationError(
                    f"Memory class '{cls.__name__}' inherits from multiple framework base classes: "
                    f"{', '.join(b.__name__ for b in actual_bases)}. Multiple inheritance is not supported."
                )
            if actual_bases[0] is not expected_base:
                raise CompilationError(
                    f"Memory class '{cls.__name__}' inherits from '{actual_bases[0].__name__}' "
                    f"but map expects '{expected_base.__name__}'"
                )

    # ── Step 0-1: Source & AST ─────────────────────────────────────────────

    @staticmethod
    def _build_ast(ctx: ValidationContext, sim_class: type) -> None:
        try:
            source = inspect.getsource(sim_class)
            ctx.source_code = textwrap.dedent(source)
        except (OSError, TypeError) as exc:
            raise CompilationError(
                f"Cannot retrieve source for '{sim_class.__name__}': {exc}.\n"
                "Ensure the class is defined in a file (not an interactive shell)."
            ) from exc

        try:
            ctx.tree = ast.parse(ctx.source_code)
        except SyntaxError as exc:
            raise CompilationError(
                f"Syntax error in '{sim_class.__name__}': {exc}"
            ) from exc

        # Annotate every node with its parent for upward-walk checks.
        ctx.tree.parent = None  # type: ignore[attr-defined]
        for node in ast.walk(ctx.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore[attr-defined]

        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ctx.method_nodes:
                    raise CompilationError(
                        f"Line {node.lineno}: function '{node.name}' is already defined."
                    )
                ctx.method_nodes[node.name] = node

    # ── Step 0-2: Signatures & resolved type hints ─────────────────────────

    @staticmethod
    def _build_signatures(ctx: ValidationContext, sim_class: type) -> None:
        for name, method in inspect.getmembers(sim_class, predicate=inspect.isfunction):
            ctx.method_signatures[name] = inspect.signature(method)
            param_hints, return_hint = ContextBuilder._resolve_hints(method)
            ctx.method_type_hints[name] = param_hints
            ctx.method_return_hints[name] = return_hint

    @staticmethod
    def _resolve_hints(func: Any) -> Tuple[Dict[str, Any], Optional[Any]]:
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = getattr(func, "__annotations__", {}).copy()
        return_type = hints.pop("return", None)
        return hints, return_type

    # ── Step 0-3: Memory Layout metadata ──────────────────────────────────

    @staticmethod
    def _build_class_metadata(ctx: ValidationContext) -> None:
        for cls in ctx.memory_base_map.keys():
            ctx.class_fields[cls] = ContextBuilder._collect_fields(cls)
            ctx.class_builtins[cls] = ContextBuilder._collect_builtins(cls)

    @staticmethod
    def _collect_fields(cls: type) -> Set[str]:
        fields: Set[str] = set()
        for klass in cls.__mro__:
            if klass is object:
                break
            for fname in klass.__dict__.get("__annotations__", {}):
                if not fname.startswith("_"):
                    fields.add(fname)
        for base_cls, sys_fields in SYSTEM_FIELDS.items():
            if isinstance(cls, type) and issubclass(cls, base_cls):
                fields.update(sys_fields)
        return fields

    @staticmethod
    def _collect_builtins(cls: type) -> Set[str]:
        if not isinstance(cls, type):
            return set()
        for base_cls, methods in AGENT_BUILTINS.items():
            if issubclass(cls, base_cls):
                return set(methods)
        return set()

    # ── Step 0-4: Constants ────────────────────────────────────────────────

    @staticmethod
    def _build_constants(ctx: ValidationContext) -> None:
        valid_value_types = (int, float, bool, vec3)
        sim_class = ctx.sim_instance.__class__
        user_module = inspect.getmodule(sim_class)
        if user_module is not None:
            for k, v in vars(user_module).items():
                if k.isupper() and isinstance(v, valid_value_types):
                    ctx.constants[k] = v
        for k, v in ctx.sim_instance.__dict__.items():
            if k.isupper() and isinstance(v, valid_value_types):
                ctx.constants[k] = v

    # ── Step 0-5: Classify methods ─────────────────────────────────────────

    @staticmethod
    def _classify_methods(ctx: ValidationContext) -> None:
        if ctx.tree is None:
            return
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if ContextBuilder._has_rule_decorator(node):
                ctx.rule_method_names.add(node.name)
            else:
                ctx.helper_method_names.add(node.name)

    @staticmethod
    def _has_rule_decorator(node: ast.FunctionDef) -> bool:
        for dec in node.decorator_list:
            attr_node: Optional[ast.Attribute] = None
            if isinstance(dec, ast.Attribute):
                attr_node = dec
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                attr_node = dec.func
            if attr_node is not None and attr_node.attr in RULE_DECORATOR_NAMES:
                return True
        return False


# ============================================================================
# BASE VALIDATOR
# ============================================================================

class BaseValidator(ast.NodeVisitor):
    """
    Common base for all pipeline validators.

    Each concrete validator is an ``ast.NodeVisitor`` whose ``visit_*``
    methods raise ``CompilationError`` on violation.  After the AST walk,
    ``post_validate()`` is called for checks that require the full tree
    to have been traversed first (e.g., graph cycle detection).
    """

    def __init__(self) -> None:
        self.ctx: Optional[ValidationContext] = None

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        if ctx.tree:
            self.visit(ctx.tree)
        self.post_validate()

    def post_validate(self) -> None:
        """Override for checks that need the complete traversal result."""

    @staticmethod
    def _is_dunder(name: str) -> bool:
        return name.startswith("__") and name.endswith("__")


# ============================================================================
# PHASE 1-A – NAMING VALIDATOR
# ============================================================================

class NamingValidator(BaseValidator):
    """
    Phase 1a: Naming convention enforcement.

    Invariants
    ----------
    * Method names            → strict snake_case (no leading _, not dunder)
    * Parameter names         → strict snake_case ('self' exempt)
    * Local variable stores   → strict snake_case
    * UPPER_CASE stores       → forbidden inside rules (constants are read-only)
    * Unknown UPPER_CASE loads → forbidden (undefined constant reference)
    """

    _SNAKE_RE: re.Pattern[str] = re.compile(r'^[a-z][a-z0-9_]*$')

    def _is_snake(self, name: str) -> bool:
        return bool(self._SNAKE_RE.match(name))

    # ── Method names ───────────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._is_dunder(node.name) and not self._is_snake(node.name):
            raise CompilationError(
                f"Line {node.lineno}: Method name '{node.name}' must be "
                f"snake_case and cannot start with an underscore."
            )
        self.generic_visit(node)

    # ── Parameter names ────────────────────────────────────────────────────

    def visit_arg(self, node: ast.arg) -> None:
        name = node.arg
        if name != "self" and not self._is_snake(name):
            raise CompilationError(
                f"Line {node.lineno}: Parameter '{name}' must be "
                f"snake_case and cannot start with an underscore."
            )
        self.generic_visit(node)

    # ── Variable name stores and loads ────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id

        if isinstance(node.ctx, ast.Store):
            if name.isupper():
                raise CompilationError(
                    f"Line {node.lineno}: Assignment to UPPER_CASE name "
                    f"'{name}' is forbidden. Constants are read-only inside rules."
                )
            if not self._is_dunder(name) and name != "self" and not self._is_snake(name):
                raise CompilationError(
                    f"Line {node.lineno}: Variable '{name}' must be "
                    f"snake_case and cannot start with an underscore."
                )

        elif isinstance(node.ctx, ast.Load):
            if name.isupper() and self.ctx and name not in self.ctx.constants:
                raise CompilationError(
                    f"Line {node.lineno}: Unknown constant '{name}'. "
                    f"Constants must be defined at the module level or in __init__."
                )

        self.generic_visit(node)

    # ── Attribute access (self.CONSTANT or self.variable) ─────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            name = node.attr
            if not (name.isupper() or self._is_snake(name) or self._is_dunder(name)):
                raise CompilationError(
                    f"Line {node.lineno}: Attribute 'self.{name}' must be "
                    f"either strictly snake_case or UPPER_CASE."
                )
        self.generic_visit(node)


# ============================================================================
# PHASE 1-B – SYNTAX VALIDATOR
# ============================================================================

class SyntaxValidator(BaseValidator):
    """
    Phase 1b: DSL structure and WGSL-compatibility enforcement.

    Forbidden constructs
    --------------------
    * import / from … import
    * try/except, with, lambda, f-strings
    * list / dict / set literals and comprehensions, generator expressions
    * tuple literals and unpacking assignments
    * nested class definitions
    * while loops
    * for loops other than: ``for i in range(…)`` or
      ``for x in <obj>.{cells,tissues,neighbors}``
    * cascading assignment  ``a = b = c``
    * chained comparisons   ``0 < x < 1``
    """

    _MACRO_ITER_ATTRS: FrozenSet[str] = frozenset({"cells", "tissues", "neighbors"})

    def __init__(self) -> None:
        super().__init__()
        self._classdef_depth: int = 0

    # ── Forbidden imports ──────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        raise CompilationError(
            f"Line {node.lineno}: 'import' statements are forbidden inside a Simulation class."
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise CompilationError(
            f"Line {node.lineno}: 'from … import' statements are forbidden inside a Simulation class."
        )

    # ── Forbidden control flow ─────────────────────────────────────────────

    def visit_Try(self, node: ast.Try) -> None:
        raise CompilationError(f"Line {node.lineno}: try/except blocks are forbidden.")

    def visit_With(self, node: ast.With) -> None:
        raise CompilationError(f"Line {node.lineno}: 'with' statements are forbidden.")

    def visit_While(self, node: ast.While) -> None:
        raise CompilationError(f"Line {node.lineno}: 'while' loops are forbidden.")

    def visit_For(self, node: ast.For) -> None:
        if not self._is_allowed_for(node):
            raise CompilationError(
                f"Line {node.lineno}: Forbidden for-loop form. "
                f"Allowed: 'for i in range(n)' or "
                f"'for x in obj.{{cells,tissues,neighbors}}'."
            )
        self.generic_visit(node)  # visit body exactly once

    def _is_allowed_for(self, node: ast.For) -> bool:
        # for i in range(n)
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and isinstance(node.target, ast.Name)):
            return True
        # for x in obj.{cells,tissues,neighbors}
        if (isinstance(node.iter, ast.Attribute)
                and node.iter.attr in self._MACRO_ITER_ATTRS
                and isinstance(node.target, ast.Name)):
            return True
        return False

    # ── Nested class definitions ───────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._classdef_depth += 1
        if self._classdef_depth > 1:
            raise CompilationError(
                f"Line {node.lineno}: Nested class definitions are forbidden."
            )
        # Skip base-class list (contains the Generic[…] tuple) to avoid a
        # spurious "Tuples are forbidden" error on line 1.
        for stmt in node.body:
            self.visit(stmt)
        self._classdef_depth -= 1

    # ── Forbidden expressions ──────────────────────────────────────────────

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise CompilationError(f"Line {node.lineno}: Lambda expressions are forbidden.")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        raise CompilationError(f"Line {node.lineno}: f-strings are forbidden.")

    def visit_Dict(self, node: ast.Dict) -> None:
        raise CompilationError(f"Line {node.lineno}: Dictionary literals are forbidden.")

    def visit_Set(self, node: ast.Set) -> None:
        raise CompilationError(f"Line {node.lineno}: Set literals are forbidden.")

    def visit_List(self, node: ast.List) -> None:
        raise CompilationError(f"Line {node.lineno}: List literals are forbidden.")

    def visit_Tuple(self, node: ast.Tuple) -> None:
        raise CompilationError(
            f"Line {node.lineno}: Tuples are forbidden. "
            f"Tuple unpacking is not supported – use individual variables."
        )

    # ── Comprehensions ─────────────────────────────────────────────────────

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise CompilationError(f"Line {node.lineno}: List comprehensions are forbidden.")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise CompilationError(f"Line {node.lineno}: Dict comprehensions are forbidden.")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise CompilationError(f"Line {node.lineno}: Set comprehensions are forbidden.")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise CompilationError(f"Line {node.lineno}: Generator expressions are forbidden.")

    # ── Assignment restrictions ────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) > 1:
            raise CompilationError(
                f"Line {node.lineno}: Cascading assignment (a = b = c) is forbidden. "
                f"Use separate assignment statements."
            )
        if isinstance(node.targets[0], (ast.Tuple, ast.List)):
            raise CompilationError(
                f"Line {node.lineno}: Tuple/List-unpacking assignment is forbidden. "
                f"Declare variables individually."
            )
        self.generic_visit(node)

    # ── Chained comparisons ────────────────────────────────────────────────

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.comparators) > 1:
            raise CompilationError(
                f"Line {node.lineno}: Chained comparisons (e.g. '0 < x < 1') are forbidden. "
                f"Use two separate conditions joined with 'and'."
            )
        self.generic_visit(node)


# ============================================================================
# PHASE 2 – SCOPE BUILDER
# ============================================================================

class ScopeBuilder(BaseValidator):
    """
    Phase 2: Signature validation and Symbol Table construction.

    Responsibilities
    ----------------
    1. **Signature validation** (absorbs former SignatureValidator):
       - Every non-self parameter must have a type annotation.
       - Rule methods must carry exactly one recognised @ol.*_rule decorator.
       - Decorator counts and parameter shapes are enforced per rule kind.

    2. **Symbol Table** (``ctx.symbol_table``):
       Maps  ``method_name → {var_name: declared_type_or_None}``.
       Populated from parameter annotations and explicit ``ast.AnnAssign``
       local declarations.  TypeChecker enriches types bottom-up in Phase 3.

    3. **Constant shadowing check**:
       A local variable name that matches an UPPER_CASE constant is rejected
       here (complementing the NamingValidator store-context check).
    """

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        ctx.symbol_table = {}
        for name, node in ctx.method_nodes.items():
            if self._is_dunder(name):
                ctx.symbol_table[name] = {}
                continue
            hints = ctx.method_type_hints.get(name, {})
            sig   = ctx.method_signatures.get(name)
            self._validate_signature(name, node, sig, hints)
            ctx.symbol_table[name] = self._build_scope(node, sig, hints)

    # ── Signature validation ───────────────────────────────────────────────

    def _validate_signature(
            self,
            name: str,
            node: ast.FunctionDef,
            sig: Optional[inspect.Signature],
            hints: Dict[str, Any],
    ) -> None:
        params = list(sig.parameters.keys()) if sig else []
        if not params or params[0] != "self":
            raise CompilationError(
                f"Line {node.lineno}: Method '{name}' must declare 'self' as its first argument."
            )

        rule_decs = [d for d in node.decorator_list if self._is_ol_rule_dec(d)]

        # No unrecognised decorators allowed
        if len(rule_decs) != len(node.decorator_list):
            raise CompilationError(
                f"Line {node.lineno}: Method '{name}' uses an unrecognised decorator. "
                f"Only @ol.*_rule decorators are allowed (0 or 1)."
            )

        if len(rule_decs) > 1:
            raise CompilationError(
                f"Line {node.lineno}: Method '{name}' has multiple @ol.*_rule decorators; "
                f"at most one is permitted."
            )

        # All non-self params must be annotated
        for param in params[1:]:
            if param not in hints:
                raise CompilationError(
                    f"Line {node.lineno}: Parameter '{param}' in '{name}' lacks a type annotation."
                )
            ptype = hints[param]
            self._assert_valid_annotation(ptype, node, param, name)

        rule_name = self._get_rule_name(node)
        if rule_name:
            self._check_rule_signature(name, node, rule_name, params, hints)

    def _assert_valid_annotation(
            self,
            t: Any,
            node: ast.AST,
            param: str,
            method: str,
    ) -> None:
        if t in ALLOWED_PRIMITIVE_TYPES:
            return
        if isinstance(t, type) and any(issubclass(t, b) for b in BASE_CLASSES):
            return
        raise CompilationError(
            f"Line {node.lineno}: Parameter '{param}' in '{method}' "
            f"has unsupported type '{t!r}'."
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
                    f"Line {node.lineno}: @{rule_name} '{name}' "
                    f"needs {n} parameter(s) {spec}; got {len(params)}."
                )

        def expect_base(p: str, base: type) -> None:
            t = hints.get(p)
            if t and (not isinstance(t, type) or not issubclass(t, base)):
                raise CompilationError(
                    f"Line {node.lineno}: Parameter '{p}' in '{name}' must be a "
                    f"subclass of {base.__name__}, got '{getattr(t, '__name__', t)!r}'."
                )

        if rule_name == "cell_rule":
            need(2, "(self, cell)")
            expect_base(params[1], CellData)
        elif rule_name == "tissue_rule":
            need(2, "(self, voxel)")
            expect_base(params[1], TissueData)
        elif rule_name == "chemistry_rule":
            need(2, "(self, chem)")
            expect_base(params[1], ChemistryData)
        elif rule_name == "metric_rule":
            need(3, "(self, item, metrics)")
            self._check_metric_item(hints.get(params[1]), node, name)
            expect_base(params[2], MetricsData)

    def _check_metric_item(self, t: Any, node: ast.AST, method: str) -> None:
        ITEM_BASES = (CellData, TissueData, ChemistryData)
        if not isinstance(t, type) or not any(issubclass(t, b) for b in ITEM_BASES):
            raise CompilationError(
                f"Line {node.lineno}: The first parameter in '{method}' must be a "
                f"CellData, TissueData, or ChemistryData subclass; "
                f"got '{getattr(t, '__name__', t)!r}'."
            )

    # ── Symbol Table construction ──────────────────────────────────────────

    def _build_scope(
            self,
            node: ast.FunctionDef,
            sig: Optional[inspect.Signature],
            hints: Dict[str, Any],
    ) -> Dict[str, Optional[Any]]:
        scope: Dict[str, Optional[Any]] = {}
        if sig:
            for pname in sig.parameters:
                if pname == "self":
                    continue
                scope[pname] = hints.get(pname)
        self._scan_body(node.body, scope)
        return scope

    def _scan_body(self, stmts: List[ast.stmt], scope: Dict[str, Optional[Any]]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                var = stmt.targets[0].id
                scope.setdefault(var, None)

            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                var = stmt.target.id
                scope[var] = self._resolve_annotation_node(stmt.annotation)

            elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                scope.setdefault(stmt.target.id, None)
                self._scan_body(stmt.body, scope)

            elif isinstance(stmt, ast.If):
                self._scan_body(stmt.body, scope)
                self._scan_body(stmt.orelse, scope)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_annotation_node(ann: ast.expr) -> Optional[Any]:
        if isinstance(ann, ast.Name):
            return _ANNOTATION_NAMES.get(ann.id)
        if isinstance(ann, ast.Attribute):
            return _ANNOTATION_NAMES.get(ann.attr)
        return None

    @staticmethod
    def _get_rule_name(node: ast.FunctionDef) -> Optional[str]:
        for dec in node.decorator_list:
            node_to_check = dec.func if isinstance(dec, ast.Call) else dec
            name = getattr(node_to_check, "attr", getattr(node_to_check, "id", None))
            if name in RULE_DECORATOR_NAMES:
                return name
        return None

    @staticmethod
    def _is_ol_rule_dec(dec: ast.expr) -> bool:
        node_to_check = dec.func if isinstance(dec, ast.Call) else dec
        name = getattr(node_to_check, "attr", getattr(node_to_check, "id", None))
        return name in RULE_DECORATOR_NAMES


# ============================================================================
# PHASE 3 – TYPE CHECKER
# ============================================================================

class TypeChecker(BaseValidator):
    """
    Phase 3: Type inference and strict type-compatibility enforcement.

    Algorithm
    ---------
    For each method, a per-method *type environment* is initialised from
    ``ctx.symbol_table`` (Phase 2 output), then enriched by forward data-
    flow as the body is walked statement by statement.

    Every typed expression node is written into ``ctx.type_map`` so that
    Phase 5 (DomainValidator) can look up types without re-inferring them.

    WGSL strict-typing rules enforced
    ----------------------------------
    * Integer and floating-point types may not be mixed without an explicit
      cast  (e.g. ``f32 + int`` is an error; write ``f32 + f32(5)``).
    * Return values must match the declared return type.
    * ``vec3`` may be scaled by float scalars ( * / ) but not added to one.
    """

    _VEC3_ATTRS: FrozenSet[str] = frozenset({"x", "y", "z"})

    def __init__(self) -> None:
        super().__init__()
        self._env:            Dict[str, Any] = {}
        self._current_method: Optional[str] = None
        self._return_type:    Optional[Any] = None

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        ctx.type_map = {}
        for method_name, func_node in ctx.method_nodes.items():
            self._check_function(func_node)

    # ── Function-level entry ───────────────────────────────────────────────

    def _check_function(self, node: ast.FunctionDef) -> None:
        saved = (self._env, self._current_method, self._return_type)
        method_name = node.name

        self._current_method = method_name
        self._return_type = self.ctx.method_return_hints.get(method_name)

        # Seed env from parameter type hints.
        self._env = {
            pname: ptype
            for pname, ptype in self.ctx.method_type_hints.get(method_name, {}).items()
            if pname != "self"
        }

        for stmt in node.body:
            self._check_stmt(stmt)

        (self._env, self._current_method, self._return_type) = saved

    # ── Statement dispatcher ───────────────────────────────────────────────

    def _check_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            self._on_assign(node)
        elif isinstance(node, ast.AnnAssign):
            self._on_ann_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._on_aug_assign(node)
        elif isinstance(node, ast.Return):
            self._on_return(node)
        elif isinstance(node, ast.If):
            self._infer(node.test)
            for s in node.body:
                self._check_stmt(s)
            for s in node.orelse:
                self._check_stmt(s)
        elif isinstance(node, ast.For):
            self._on_for(node)
        elif isinstance(node, ast.Expr):
            self._infer(node.value)

    def _on_assign(self, node: ast.Assign) -> None:
        rhs_type = self._infer(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            var = target.id
            existing = self._env.get(var)
            if existing is not None and rhs_type is not None:
                if not _types_compatible(existing, rhs_type):
                    raise CompilationError(
                        f"Line {node.lineno}: Type mismatch: cannot assign "
                        f"'{_type_name(rhs_type)}' to '{var}' "
                        f"(declared type '{_type_name(existing)}')."
                    )
            if rhs_type is not None:
                self._env[var] = rhs_type
                self.ctx.type_map[id(target)] = rhs_type
        # For attribute targets (cell.health = x) just infer for side-effects.
        elif isinstance(target, ast.Attribute):
            self._infer(target)

    def _on_ann_assign(self, node: ast.AnnAssign) -> None:
        ann_type = _resolve_annotation_node(node.annotation)
        val_type = self._infer(node.value) if node.value else None

        if ann_type is not None and val_type is not None:
            if not _types_compatible(ann_type, val_type):
                raise CompilationError(
                    f"Line {node.lineno}: Type mismatch: declared '{_type_name(ann_type)}' "
                    f"but value has type '{_type_name(val_type)}'."
                )

        if isinstance(node.target, ast.Name):
            var = node.target.id
            resolved = ann_type if ann_type is not None else val_type
            if resolved is not None:
                self._env[var] = resolved
                self.ctx.type_map[id(node.target)] = resolved

    def _on_aug_assign(self, node: ast.AugAssign) -> None:
        target_type = self._infer(node.target)
        value_type  = self._infer(node.value)
        if target_type is not None and value_type is not None:
            if not _types_compatible(target_type, value_type):
                raise CompilationError(
                    f"Line {node.lineno}: Type mismatch: cannot apply "
                    f"'{type(node.op).__name__}=' to '{_type_name(target_type)}' "
                    f"and '{_type_name(value_type)}'."
                )

    def _on_return(self, node: ast.Return) -> None:
        expected = self._return_type
        is_void  = expected is None or expected is type(None)

        if node.value is None:
            if not is_void:
                raise CompilationError(
                    f"Line {node.lineno}: Method '{self._current_method}' "
                    f"must return '{_type_name(expected)}'."
                )
            return

        actual = self._infer(node.value)
        if not is_void and actual is not None:
            if not _types_compatible(actual, expected):
                raise CompilationError(
                    f"Line {node.lineno}: Return type mismatch in "
                    f"'{self._current_method}': expected '{_type_name(expected)}', "
                    f"got '{_type_name(actual)}'."
                )

    def _on_for(self, node: ast.For) -> None:
        it = node.iter
        loop_var = node.target.id if isinstance(node.target, ast.Name) else None

        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            if loop_var:
                self._env[loop_var] = i32
                self.ctx.type_map[id(node.target)] = i32
        elif isinstance(it, ast.Attribute):
            container_type = self._infer(it.value)
            elem_type = self._resolve_iterator_elem_type(container_type, it.attr, node.lineno)
            if loop_var and elem_type is not None:
                self._env[loop_var] = elem_type
                self.ctx.type_map[id(node.target)] = elem_type

        # Visit body ONCE (fix for double-generic_visit bug in original).
        for s in node.body:
            self._check_stmt(s)

    def _resolve_iterator_elem_type(
            self,
            container_type: Any,
            attr: str,
            lineno: int,
    ) -> Optional[Any]:
        base = self.ctx.base_class_of(container_type)
        if base is None:
            return None
        mapping: Dict[type, Dict[str, type]] = {
            CellData:      {"neighbors": CellData},
            TissueData:    {"cells": CellData,    "neighbors": TissueData},
            ChemistryData: {"cells": CellData,    "tissues": TissueData,
                            "neighbors": ChemistryData},
        }
        elem_base = mapping.get(base, {}).get(attr)
        if elem_base is None:
            raise CompilationError(
                f"Line {lineno}: '{_type_name(base)}' does not support "
                f"'.{attr}' iteration."
            )
        # Return the user subclass if available.
        for u_cls, b_cls in self.ctx.memory_base_map.items():
            if b_cls is elem_base:
                return u_cls
        return elem_base

    # ── Expression type inference ──────────────────────────────────────────

    def _infer(self, node: Optional[ast.expr]) -> Optional[Any]:
        """
        Infer the type of an expression, write it to ``ctx.type_map``,
        and return it.  Returns ``None`` when the type cannot be determined.
        """
        if node is None:
            return None

        t = self._infer_impl(node)
        if t is not None:
            self.ctx.type_map[id(node)] = t
        return t

    def _infer_impl(self, node: ast.expr) -> Optional[Any]:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool): return bool
            if isinstance(v, int):  return int
            if isinstance(v, float): return float
            return None

        if isinstance(node, ast.Name):
            t = self._env.get(node.id)
            if t is None and self.ctx:
                cv = self.ctx.constants.get(node.id)
                if cv is not None:
                    return type(cv)
            return t

        if isinstance(node, ast.Attribute):
            parent_t = self._infer(node.value)
            return self._get_attr_type(parent_t, node.attr)

        if isinstance(node, ast.BinOp):
            lt = self._infer(node.left)
            rt = self._infer(node.right)
            return self._check_binop(node, lt, rt)

        if isinstance(node, ast.UnaryOp):
            return self._infer(node.operand)

        if isinstance(node, ast.IfExp):
            self._infer(node.test)
            t = self._infer(node.body)
            self._infer(node.orelse)
            return t

        if isinstance(node, ast.Call):
            return self._infer_call(node)

        if isinstance(node, ast.Compare):
            self._infer(node.left)
            for c in node.comparators:
                self._infer(c)
            return ol_bool

        return None

    def _check_binop(
            self,
            node: ast.BinOp,
            lt: Optional[Any],
            rt: Optional[Any],
    ) -> Optional[Any]:
        if lt is None or rt is None:
            return lt or rt

        l_is_vec    = lt is vec3
        r_is_vec    = rt is vec3
        l_is_scalar = _is_scalar(lt)
        r_is_scalar = _is_scalar(rt)

        if l_is_vec and r_is_vec:
            return vec3

        if l_is_vec and r_is_scalar:
            if not isinstance(node.op, (ast.Mult, ast.Div)):
                raise CompilationError(
                    f"Line {node.lineno}: vec3 {type(node.op).__name__} scalar is not "
                    f"allowed. Only '*' and '/' are supported between vec3 and scalar."
                )
            return vec3

        if l_is_scalar and r_is_vec:
            if not isinstance(node.op, ast.Mult):
                raise CompilationError(
                    f"Line {node.lineno}: scalar {type(node.op).__name__} vec3 is not "
                    f"allowed. Only 'scalar * vec3' is supported."
                )
            return vec3

        if l_is_scalar and r_is_scalar:
            if not _types_compatible(lt, rt):
                raise CompilationError(
                    f"Line {node.lineno}: Type mismatch in binary operation: "
                    f"'{_type_name(lt)}' and '{_type_name(rt)}' belong to different "
                    f"type groups. WGSL requires an explicit cast (e.g., f32(...))."
                )
            return _wider_type(lt, rt)

        raise CompilationError(
            f"Line {node.lineno}: Unsupported operand types: "
            f"'{_type_name(lt)}' and '{_type_name(rt)}'."
        )

    def _infer_call(self, node: ast.Call) -> Optional[Any]:
        self._infer(node.func)

        fn = node.func

        # ol.random(…) → f32 ;  ol.random_dir(…) → vec3
        if (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "ol"):
            if fn.attr == "random":     return f32
            if fn.attr == "random_dir": return vec3

        # self.helper(…) → look up declared return type
        if (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "self"):
            ret = self.ctx.method_return_hints.get(fn.attr)
            # Infer argument types for side-effects
            for arg in node.args:
                self._infer(arg)
            return ret

        # Explicit type casts: f32(x), i32(x), …
        if isinstance(fn, ast.Name):
            cast_t = _ANNOTATION_NAMES.get(fn.id)
            if cast_t is not None:
                for arg in node.args:
                    self._infer(arg)
                return cast_t

        # Fallback: infer arguments for side-effects
        for arg in node.args:
            self._infer(arg)
        return None

    def _get_attr_type(self, parent_type: Any, attr: str) -> Optional[Any]:
        if parent_type is None:
            return None
        if parent_type is vec3 and attr in self._VEC3_ATTRS:
            return f32
        if isinstance(parent_type, type):
            is_mem = parent_type in BASE_CLASSES or any(
                issubclass(parent_type, b) for b in BASE_CLASSES
            )
            if is_mem:
                for klass in parent_type.__mro__:
                    if klass is object:
                        break
                    ann = klass.__dict__.get("__annotations__", {})
                    if attr in ann:
                        raw = ann[attr]
                        if isinstance(raw, str):
                            return _ANNOTATION_NAMES.get(raw)
                        return raw
                # Framework-injected fields
                if attr == "pos":
                    return vec3
        return None


# ============================================================================
# PHASE 4 – CALL GRAPH VALIDATOR
# ============================================================================

class CallGraphValidator(BaseValidator):
    """
    Phase 4: Call-graph analysis.

    Invariants
    ----------
    * Direct and indirect recursion is forbidden (no stack in WGSL).
    * Calls to undefined ``self.*`` methods are forbidden.

    Produces
    --------
    * ``ctx.call_graph``     – directed dependency graph of internal methods.
    * ``ctx.ordered_methods``– topologically sorted list for WGSL emission
                               (callees always precede callers).
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_caller: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._current_caller is not None:
            return  # Don't recurse into nested function defs
        self._current_caller = node.name
        if self.ctx and self._current_caller not in self.ctx.call_graph:
            self.ctx.call_graph[self._current_caller] = set()
        self.generic_visit(node)
        self._current_caller = None

    def visit_Call(self, node: ast.Call) -> None:
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):
            callee = node.func.attr
            if self.ctx and self._current_caller:
                self.ctx.call_graph[self._current_caller].add(callee)
        self.generic_visit(node)

    def post_validate(self) -> None:
        """DFS cycle detection and topological sort."""
        if not self.ctx:
            return
        graph = self.ctx.call_graph

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {name: WHITE for name in graph}
        ordered: List[str] = []
        path: List[str] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            path.append(node)
            for callee in sorted(graph.get(node, set())):
                if callee not in graph:
                    src_node = self.ctx.method_nodes.get(node) if self.ctx else None
                    lineno = src_node.lineno if src_node else "?"
                    raise CompilationError(
                        f"Line {lineno}: Method 'self.{callee}()' is not defined "
                        f"in the Simulation class."
                    )
                if color[callee] == GRAY:
                    cycle_start = path.index(callee)
                    cycle_repr  = " → ".join(path[cycle_start:] + [callee])
                    src_node = self.ctx.method_nodes.get(node) if self.ctx else None
                    lineno = src_node.lineno if src_node else "?"
                    raise CompilationError(
                        f"Line {lineno}: Recursive call detected: {cycle_repr}. "
                        f"Recursion is forbidden (WGSL has no call stack)."
                    )
                if color[callee] == WHITE:
                    dfs(callee)
            path.pop()
            color[node] = BLACK
            ordered.append(node)

        for method_name in sorted(graph.keys()):
            if color[method_name] == WHITE:
                dfs(method_name)

        self.ctx.ordered_methods = ordered


# ============================================================================
# PHASE 5 – DOMAIN VALIDATOR
# ============================================================================

class DomainValidator(BaseValidator):
    """
    Phase 5: Oncolytica domain-specific semantic rules.

    At this phase every expression already has a type in ``ctx.type_map``
    (Phase 3 output), so no type inference is repeated here.

    Invariants
    ----------
    * ``self.<attr>`` mutation is only permitted inside ``__init__``.
    * Pointer arguments (memory-class parameters) cannot be rebound.
    * Attribute access on a memory-class instance must reference a
      declared field or a framework built-in.
    * ``cell.die()`` / ``cell.divide()`` may only be called on the agent
      argument passed to the enclosing rule (not on neighbors).
    * vec3 attribute access must be ``.x``, ``.y``, or ``.z``.
    """

    _VEC3_ATTRS: FrozenSet[str] = frozenset({"x", "y", "z"})

    def __init__(self) -> None:
        super().__init__()
        self._current_method: Optional[str] = None
        self._in_init:        bool = False
        self._pointer_args:   Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self.ctx:
            return
        saved = (self._current_method, self._in_init, self._pointer_args.copy())

        self._current_method = node.name
        self._in_init        = (node.name == "__init__")
        self._pointer_args   = {
            pname
            for pname, ptype in self.ctx.method_type_hints.get(node.name, {}).items()
            if _is_memory_type(ptype)
        }

        self.generic_visit(node)
        (self._current_method, self._in_init, self._pointer_args) = saved

    # ── Mutation checks ────────────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._current_method:
            self._check_lhs(node.targets[0], node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._current_method:
            self._check_lhs(node.target, node.lineno)
        self.generic_visit(node)

    def _check_lhs(self, target: ast.expr, lineno: int) -> None:
        if isinstance(target, ast.Attribute):
            # Walk down to the root object.
            root = target.value
            while isinstance(root, ast.Attribute):
                root = root.value

            if isinstance(root, ast.Name):
                obj_name = root.id
                if obj_name == "self":
                    if not self._in_init:
                        raise CompilationError(
                            f"Line {lineno}: Assignment to 'self' is forbidden outside __init__."
                        )
                    return
                # Check ownership: only the rule's own pointer arg may be mutated.
                obj_type = self.ctx.type_map.get(id(root)) if self.ctx else None
                if _is_memory_type(obj_type) and obj_name not in self._pointer_args:
                    raise CompilationError(
                        f"Line {lineno}: Cannot modify attribute of '{obj_name}'. "
                        f"You can ONLY modify the state of the current agent/voxel "
                        f"passed to the rule."
                    )
            return

        if isinstance(target, ast.Name):
            var = target.id
            if var in self._pointer_args:
                raise CompilationError(
                    f"Line {lineno}: Cannot rebind pointer argument '{var}'."
                )

    # ── Field access validation ────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not self._current_method:
            self.generic_visit(node)
            return
        # Skip self.xxx – handled by _check_lhs for write side.
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            self.generic_visit(node)
            return

        parent_type = self.ctx.type_map.get(id(node.value)) if self.ctx else None
        if parent_type is not None:
            if _is_memory_type(parent_type) and isinstance(parent_type, type):
                valid = self.ctx.all_valid_attrs(parent_type) if self.ctx else set()
                if valid and node.attr not in valid:
                    raise CompilationError(
                        f"Line {node.lineno}: '{node.attr}' is not a valid field "
                        f"on '{_type_name(parent_type)}'."
                    )
            elif parent_type is vec3 and node.attr not in self._VEC3_ATTRS:
                raise CompilationError(
                    f"Line {node.lineno}: '{node.attr}' is not a valid vec3 "
                    f"component (use .x, .y, or .z)."
                )
        self.generic_visit(node)

    # ── Action-call ownership ──────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        # ── Existing: Action-call ownership (die/divide) ──
        if (self._current_method
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)):
            obj_name = node.func.value.id
            method_name = node.func.attr
            if obj_name != "self":
                obj_type = self.ctx.type_map.get(id(node.func.value)) if self.ctx else None
                if _is_cell_type(obj_type) and method_name in ("die", "divide"):
                    if obj_name not in self._pointer_args:
                        raise CompilationError(
                            f"Line {node.lineno}: Cannot call '{method_name}()' on "
                            f"'{obj_name}'. Only the current rule argument "
                            f"may be terminated or divided."
                        )

        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):

            # 1. Detect if any argument is a pointer/memory type
            has_pointer_arg = False
            for arg in node.args:
                if _is_memory_type(self.ctx.type_map.get(id(arg)) if self.ctx else None):
                    has_pointer_arg = True
                    break
            for kw in getattr(node, 'keywords', []):
                if _is_memory_type(self.ctx.type_map.get(id(kw.value)) if self.ctx else None):
                    has_pointer_arg = True
                    break

            # 2. If pointer args are present, enforce top-level context only
            if has_pointer_arg:
                parent = getattr(node, 'parent', None)
                is_standalone = isinstance(parent, ast.Expr)
                is_direct_assign = isinstance(parent, (ast.Assign, ast.AnnAssign)) and getattr(parent, 'value',
                                                                                               None) is node
                is_direct_return = isinstance(parent, ast.Return) and getattr(parent, 'value', None) is node

                if not (is_standalone or is_direct_assign or is_direct_return):
                    raise CompilationError(
                        f"Line {node.lineno}: Helper calls with pointer arguments "
                        f"cannot be nested in complex expressions."
                    )

        self.generic_visit(node)


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _is_scalar(t: Any) -> bool:
    """True iff ``t`` belongs to one of the three WGSL scalar groups."""
    for group in _SCALAR_GROUPS:
        if t in group:
            return True
    return False


def _types_compatible(a: Any, b: Any) -> bool:
    """
    True iff ``a`` and ``b`` may appear on both sides of an assignment
    or binary operation without an explicit cast.

    Same group → compatible.  Cross-group → incompatible (WGSL strict-typing).
    """
    if a is b:
        return True
    for group in _SCALAR_GROUPS:
        if a in group and b in group:
            return True
    return False


def _wider_type(a: Any, b: Any) -> Any:
    """Return the 'wider' type within a compatible scalar group."""
    # Lower index = narrower (bool < int subtypes < float subtypes).
    priority = {
        ol_bool: 0, bool: 1,
        i32: 2, u32: 3, i64: 4, u64: 5, int: 6,
        f32: 7, f64: 8, float: 9,
    }
    if a in priority and b in priority:
        return a if priority[a] < priority[b] else b
    return a


def _resolve_annotation_node(ann: ast.expr) -> Optional[Any]:
    if isinstance(ann, ast.Name):
        return _ANNOTATION_NAMES.get(ann.id)
    if isinstance(ann, ast.Attribute):
        return _ANNOTATION_NAMES.get(ann.attr)
    return None


def _is_memory_type(t: type) -> bool:
    return isinstance(t, type) and (
        t in BASE_CLASSES or any(issubclass(t, b) for b in BASE_CLASSES)
    )


def _is_cell_type(t: type) -> bool:
    return isinstance(t, type) and issubclass(t, CellData)


def _type_name(t: Any) -> str:
    """Human-readable type name for error messages."""
    return getattr(t, "__name__", repr(t))


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ValidatorEngine:
    """
    Runs all five phases in order against a ``ValidationContext``.

    Each phase validator is a ``BaseValidator`` sub-class; they communicate
    exclusively through the ``ValidationContext`` blackboard.
    On the first ``CompilationError`` the pipeline stops (Fail-Fast).
    """

    def __init__(self) -> None:
        self.pipeline: List[BaseValidator] = [
            # Phase 1 – Syntactic & Lexical
            NamingValidator(),
            SyntaxValidator(),
            # Phase 2 – Symbol Definition & Scope
            ScopeBuilder(),
            # Phase 3 – Type Inference & Checking
            TypeChecker(),
            # Phase 4 – Control Flow & Call Graph
            CallGraphValidator(),
            # Phase 5 – Domain-Specific Semantics
            DomainValidator(),
        ]

    def run(
            self,
            sim_instance: Any,
            memory_base_map: Dict[type, type],
    ) -> ValidationContext:
        ctx = ContextBuilder.build(sim_instance, memory_base_map)
        for validator in self.pipeline:
            validator.validate(ctx)
        return ctx
