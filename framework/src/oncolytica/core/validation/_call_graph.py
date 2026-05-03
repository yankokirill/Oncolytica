from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set

from oncolytica.core.utils._types import (
    Cell, Tissue, Chemistry, Metrics,
    BASE_CLASSES,
)
from oncolytica.core.utils._errors import CompilationError
from ._base import BaseValidator
from ._context import ValidationContext

# ── Name mangling ──────────────────────────────────────────────────────────────

# Maps framework base classes to their name prefix.
_BASE_PREFIX: Dict[type, str] = {
    Cell:      "cell",
    Tissue:    "tissue",
    Chemistry: "chemistry",
    Metrics:   "metrics",
}

# Built-in self.X() calls that are NOT user-defined methods and must be ignored.
_BUILTIN_SELF_CALLS: frozenset[str] = frozenset({"tissue_at", "chemistry_at"})


def _mangle_sim_method(method_name: str) -> str:
    """sim-method: 'spawn' → 'sim_spawn'."""
    return f"sim_{method_name}"


def _mangle_domain_method(base_cls: type, method_name: str) -> str:
    """domain-method: Cell, 'update' → 'cell_update'."""
    prefix = _BASE_PREFIX.get(base_cls, base_cls.__name__.lower())
    return f"{prefix}_{method_name}"


def _base_class_of(cls: type, memory_base_map: Dict[type, type]) -> Optional[type]:
    """Return the framework base class (Cell/Tissue/…) for a user domain class."""
    return memory_base_map.get(cls)


# ============================================================================
# PHASE 3 – CALL GRAPH VALIDATOR
# ============================================================================

class CallGraphValidator(BaseValidator):
    """
    Phase 3: Call-graph analysis.

    Runs **after** TypeChecker so that ctx.type_map is fully populated and
    obj.method() calls on domain objects can be resolved to their mangled names.

    Invariants
    ----------
    * Direct and indirect recursion is forbidden (no stack in WGSL).
    * Calls to undefined methods are forbidden.

    Produces
    --------
    * ``ctx.call_graph``      – directed dependency graph; keys and values are
                                mangled method names (e.g. "sim_spawn",
                                "cell_update").
    * ``ctx.ordered_methods`` – topologically sorted list for WGSL emission
                                (callees always precede callers).
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_caller: Optional[str] = None

    # ── Public entry point ────────────────────────────────────────────────────

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        self._build_graph()
        self._topo_sort()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> None:
        ctx = self.ctx

        # Pre-populate a node for every known method so isolated methods
        # (those that call nothing) still appear in the graph.
        for raw_name in ctx.method_nodes:
            mangled = self._mangle_current(raw_name)
            ctx.call_graph.setdefault(mangled, set())

        # Also register every domain-class method as a potential callee node.
        for cls, methods in ctx.class_methods.items():
            base = _base_class_of(cls, ctx.memory_base_map)
            if base is None:
                continue
            for m in methods:
                node_name = _mangle_domain_method(base, m)
                ctx.call_graph.setdefault(node_name, set())

        for raw_name, func_node in ctx.method_nodes.items():
            caller_mangled = self._mangle_current(raw_name)
            self._current_caller = caller_mangled
            self._visit_function(func_node)
            self._current_caller = None

    def _visit_function(self, node: ast.FunctionDef) -> None:
        """Walk the function body collecting outgoing call edges."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._process_call(child)

    def _process_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        func = node.func
        if not isinstance(func.value, ast.Name):
            return

        callee_mangled: Optional[str] = None

        if func.value.id == "self":
            # self.method() — simulation method
            method_name = func.attr
            if method_name in _BUILTIN_SELF_CALLS:
                return
            callee_mangled = _mangle_sim_method(method_name)

        else:
            # obj.method() — domain object method; requires type_map
            obj_name = func.value.id
            method_name = func.attr

            # Resolve the AST node for the Name to look up its type.
            obj_type = self.ctx.type_map.get(id(func.value))
            if obj_type is None:
                return  # Unknown type — not a domain object, skip.

            base = _base_class_of(obj_type, self.ctx.memory_base_map)
            if base is None:
                return  # Not a domain class, skip.

            callee_mangled = _mangle_domain_method(base, method_name)

        if callee_mangled is not None and self._current_caller is not None:
            self.ctx.call_graph[self._current_caller].add(callee_mangled)

    # ── Topological sort + cycle detection ───────────────────────────────────

    def _topo_sort(self) -> None:
        ctx = self.ctx
        graph = ctx.call_graph

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in graph}
        ordered: List[str] = []
        path: List[str] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            path.append(node)

            for callee in sorted(graph.get(node, set())):
                if callee not in graph:
                    src_node = ctx.method_nodes.get(_unmangled(node, ctx))
                    callee = "_".join(callee.split("_")[1:])
                    raise CompilationError(
                        f"Method '{callee}' is called but not defined.", node=src_node
                    )
                if color[callee] == GRAY:
                    cycle_start = path.index(callee)
                    cycle_repr = " → ".join(path[cycle_start:] + [callee])
                    src_node = ctx.method_nodes.get(_unmangled(node, ctx))
                    raise CompilationError(
                        f"Recursive call detected: {cycle_repr}. "
                        f"Recursion is forbidden (WGSL has no call stack).", node=src_node
                    )
                if color[callee] == WHITE:
                    dfs(callee)

            path.pop()
            color[node] = BLACK
            ordered.append(node)

        for method_name in sorted(graph.keys()):
            if color[method_name] == WHITE:
                dfs(method_name)

        ctx.ordered_methods = ordered

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _mangle_current(self, raw_name: str) -> str:
        """Return the canonical graph node name for a sim-class method.

        ctx.method_nodes keys are **already mangled** (e.g. "sim_rule",
        "sim_clamp_health") by ContextBuilder, so no further mangling is needed.
        """
        return raw_name


def _unmangled(mangled: str, ctx: ValidationContext) -> Optional[str]:
    """
    Try to recover the original method name from a mangled name so that
    error messages can reference the correct source location.
    Only covers sim_ methods (the ones with AST nodes in ctx.method_nodes).
    """
    if mangled.startswith("sim_"):
        raw = mangled[len("sim_"):]
        if raw in ctx.method_nodes:
            return raw
    return None
