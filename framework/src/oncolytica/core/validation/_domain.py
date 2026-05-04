from __future__ import annotations

import ast
from typing import Any, Dict, FrozenSet, Optional, Set

from oncolytica.core.utils._types import (
    vec3, ivec3,
    Cell,
)

_VECTOR_TYPES: frozenset = frozenset({vec3, ivec3})
from oncolytica.core.utils._errors import CompilationError
from ._base import BaseValidator
from ._context import ValidationContext, type_name
import textwrap as _textwrap
import inspect as _inspect


# ── RHS source classification ──────────────────────────────────────────────────

# Built-in grid getters that return read-only snapshots (Rule 2.4).
_GRID_GETTERS: frozenset[str] = frozenset({"chemistry_at", "tissue_at"})

# Built-in methods that are always safe to call regardless of mutability.
# copy()      — Rule 1.3 / Rule 2.2: produces an independent mutable copy.
# copy_from() — in-place field copy, but requires no ownership of the *source*;
#               ownership of the *receiver* is checked separately via pointer-arg.
_MUTABILITY_EXEMPT_METHODS: frozenset[str] = frozenset({"copy", "copy_from"})


_VECTOR_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"vec3", "ivec3"})


def _is_constructor_call(node: ast.expr, ctx: ValidationContext) -> bool:
    """Rule 2.2 — True when *node* is a memory-type or vector constructor call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # Domain constructors: DivCell(), MyTissue(), …
    if isinstance(func, ast.Name):
        for user_cls in ctx.memory_base_map:
            if getattr(user_cls, "__name__", None) == func.id:
                return True
        # Bare vector constructors: vec3(...), ivec3(...)
        if func.id in _VECTOR_CONSTRUCTOR_NAMES:
            return True
    # Qualified vector constructors: ol.vec3(...), ol.ivec3(...)
    if (isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "ol"
            and func.attr in _VECTOR_CONSTRUCTOR_NAMES):
        return True
    return False


def _is_copy_call(node: ast.expr) -> bool:
    """Rule 1.3 — True when *node* is obj.copy()."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "copy" and not node.args


def _is_grid_getter_call(node: ast.expr) -> bool:
    """Rule 2.4 — True when *node* is self.chemistry_at(...) or self.tissue_at(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _GRID_GETTERS
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    )


def _is_self_call(node: ast.expr) -> bool:
    """Rule 1.2 — True when *node* is self.some_method(...) that is NOT a grid getter.

    Covers user-defined helpers and constructors invoked through self, regardless
    of whether the callee mutates anything.  Any self.X() that is not a built-in
    grid getter is treated as potentially mutable (conservative).
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and func.attr not in _GRID_GETTERS
    )


def _rhs_source(node: ast.expr, ctx: ValidationContext) -> str:
    """Classify the RHS of an assignment into one of five source kinds:

    ``"constructor"``  — memory-type constructor call  (→ mutable, Rule 2.2)
    ``"copy"``         — obj.copy()                    (→ mutable, Rule 1.3)
    ``"self_call"``    — self.my_helper(...)            (→ mutable, Rule 1.2)
    ``"grid_getter"``  — self.chemistry_at / tissue_at (→ read-only, Rule 2.4)
    ``"alias"``        — any other memory-type expr    (→ forbidden, Rule 1.1)
    ``"other"``        — non-memory-type expr          (→ irrelevant)
    """
    rhs_type = ctx.type_map.get(id(node))
    is_domain_obj = (
        ctx.get_base_class(rhs_type) is not None
        or rhs_type in _VECTOR_TYPES
    )
    if not is_domain_obj:
        return "other"
    if rhs_type in _VECTOR_TYPES and not isinstance(node, ast.Name) and not isinstance(node, ast.Attribute):
        return "constructor"
    if _is_constructor_call(node, ctx):
        return "constructor"
    if _is_copy_call(node):
        return "copy"
    if _is_grid_getter_call(node):
        return "grid_getter"
    if _is_self_call(node):
        return "self_call"
    return "alias"


# ============================================================================
# PHASE 5 – DOMAIN VALIDATOR
# ============================================================================

class DomainValidator(BaseValidator, ast.NodeVisitor):
    """
    Phase 5: Oncolytica domain-specific semantic rules.
    Enforces Mutability and Ownership constraints for GPU generation.
    """

    _VEC3_ATTRS: FrozenSet[str] = frozenset({"x", "y", "z"})

    def __init__(self) -> None:
        super().__init__()
        self._current_method: Optional[str] = None
        self._in_init:        bool           = False
        self._pointer_args:   Set[str]       = set()
        self._all_params:     Set[str]       = set()
        self._mutable_locals: Set[str]       = set()
        self._readonly_locals: Set[str]      = set()

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        self._analyze_method_mutations()
        for func_node in ctx.method_nodes.values():
            self.visit(func_node)

    # ── Mutation analysis (pre-pass) ──────────────────────────────────────────

    def _analyze_method_mutations(self) -> None:
        """Fill ctx.method_mutating_params for every method in ctx.ordered_methods.

        Processes methods in topological order so that when we analyse a caller
        the callee's mutating positions are already known (transitive propagation).

        Built-in / framework methods that mutate their first argument (self/pos 0)
        by convention are seeded before the user-method scan.
        """
        ctx = self.ctx

        # Seed: known framework methods that mutate self (position 0).
        # Only explicitly listed built-in methods are marked mutating.
        # copy() and copy_from() are intentionally excluded — they are
        # read-safe utilities that require no mutability on the receiver.
        _TRULY_MUTATING_METHODS: FrozenSet[str] = frozenset({"die"})

        _FRAMEWORK_MUTATING: Dict[str, Set[int]] = {}
        for cls, methods in ctx.class_methods.items():
            for raw_name in methods:
                mangled = ctx.mangle_name(cls, raw_name)
                if raw_name in _TRULY_MUTATING_METHODS:
                    _FRAMEWORK_MUTATING.setdefault(mangled, {0})

        # Initialise mutating-params from framework seeds.
        for mangled, positions in _FRAMEWORK_MUTATING.items():
            ctx.method_mutating_params.setdefault(mangled, set(positions))

        # Build a local lookup for mutation analysis — does NOT touch ctx.method_nodes.
        _local_nodes: Dict[str, ast.FunctionDef] = dict(ctx.method_nodes)
        for user_cls, base_cls in ctx.memory_base_map.items():
            for raw_method_name in ctx.class_methods.get(user_cls, set()):
                mangled = ctx.mangle_name(user_cls, raw_method_name)
                if mangled in _local_nodes:
                    continue
                domain_node = ctx.domain_method_nodes.get((user_cls, raw_method_name))
                if domain_node is not None:
                    _local_nodes[mangled] = domain_node
                    continue
                method = getattr(user_cls, raw_method_name, None)
                if method is None:
                    continue
                try:
                    src = _textwrap.dedent(_inspect.getsource(method))
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            _local_nodes[mangled] = node
                            break
                except Exception:
                    pass

        # Now analyse every method AST node in topological order.
        for mangled_name in ctx.ordered_methods:
            func_node = _local_nodes.get(mangled_name)
            if func_node is None:
                continue
            hints = ctx.method_type_hints.get(mangled_name, {})
            params = list(hints.keys())  # ordered parameter names, excl. "self"

            mutating: Set[int] = set()

            for stmt in ast.walk(func_node):
                # Rule (1): direct field mutation on a parameter → param is mutating.
                if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    target = (
                        stmt.targets[0]
                        if isinstance(stmt, ast.Assign)
                        else stmt.target
                    )
                    root = target
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name):
                        if root.id == "self":
                            if not mangled_name.startswith("sim_"):
                                mutating.add(0)
                        elif root.id in params:
                            idx = params.index(root.id)
                            mutating.add(idx + 1)  # +1 because self=0 is absent from hints

                # Rule (2): transitive — argument passed to a mutating position.
                if isinstance(stmt, ast.Call):
                    func_attr = stmt.func
                    if isinstance(func_attr, ast.Attribute):
                        obj = func_attr.value
                        method_attr = func_attr.attr
                        if isinstance(obj, ast.Name):
                            obj_type = ctx.type_map.get(id(obj))
                            if obj_type is not None and ctx.get_base_class(obj_type) is not None:
                                callee_mangled = ctx.mangle_name(obj_type, method_attr)
                                callee_mutating = ctx.method_mutating_params.get(callee_mangled, set())
                                # Position 0 of callee == the receiver object.
                                if 0 in callee_mutating:
                                    if obj.id == "self":
                                        if not mangled_name.startswith("sim_"):
                                            mutating.add(0)
                                    else:
                                        try:
                                            idx = params.index(obj.id)
                                            mutating.add(idx + 1)
                                        except ValueError:
                                            pass
                                # Positional args: callee position j+1 == arg j.
                                for j, arg in enumerate(stmt.args):
                                    if (j + 1) in callee_mutating:
                                        if isinstance(arg, ast.Name) and arg.id in params:
                                            try:
                                                idx = params.index(arg.id)
                                                mutating.add(idx + 1)
                                            except ValueError:
                                                pass

                            # Rule (2b): transitive via self.helper(arg) calls.
                            elif obj.id == "self" and method_attr not in _GRID_GETTERS:
                                if not mangled_name.startswith("sim_"):
                                    # Domain-method: self.method() calls another method
                                    # on the same domain class. Resolve via the prefix
                                    # extracted from the current mangled name.
                                    prefix = mangled_name.split("_")[0]  # e.g. "cell"
                                    callee_mangled = f"{prefix}_{method_attr}"
                                else:
                                    callee_mangled = ctx.mangle_name(None, method_attr)
                                callee_mutating = ctx.method_mutating_params.get(callee_mangled, set())
                                # Position 0 of callee == self → mark self (pos 0) mutating.
                                if 0 in callee_mutating and not mangled_name.startswith("sim_"):
                                    mutating.add(0)
                                for j, arg in enumerate(stmt.args):
                                    if (j + 1) in callee_mutating:
                                        if isinstance(arg, ast.Name) and arg.id in params:
                                            try:
                                                idx = params.index(arg.id)
                                                mutating.add(idx + 1)
                                            except ValueError:
                                                pass

            ctx.method_mutating_params[mangled_name] = mutating

    # ── Visitor plumbing ──────────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        saved_method    = self._current_method
        saved_in_init   = self._in_init
        saved_pointers  = self._pointer_args.copy()
        saved_all_params = self._all_params.copy()
        saved_mutable   = self._mutable_locals.copy()
        saved_readonly  = self._readonly_locals.copy()

        self._current_method  = node.name
        self._in_init         = (node.name == "__init__")

        # Collect pointer-args: parameters whose types are domain memory types.
        # method_type_hints keys are mangled ("sim_rule"), but node.name is raw
        # ("rule"), so try both the raw name and the sim-mangled name.
        _hints = (
            self.ctx.method_type_hints.get(node.name)
            or self.ctx.method_type_hints.get(f"sim_{node.name}")
            or {}
        )
        self._pointer_args = {
            pname
            for pname, ptype in _hints.items()
            if pname != "self" and self.ctx.get_base_class(ptype) is not None
        }

        # Rule 3.4: track ALL parameters (excl. self) to forbid reassignment.
        self._all_params = {
            arg.arg
            for arg in node.args.args
            if arg.arg != "self"
        }

        # If this is a domain-class method registered in ctx.class_methods,
        # add "self" (position 0) as a pointer-arg so mutation checks cover it.
        for cls, methods in self.ctx.class_methods.items():
            if node.name in methods:
                self._pointer_args.add("self")
                break

        self._mutable_locals  = set(self._pointer_args)
        self._readonly_locals = set()

        self.generic_visit(node)

        self._current_method  = saved_method
        self._in_init         = saved_in_init
        self._pointer_args    = saved_pointers
        self._all_params      = saved_all_params
        self._mutable_locals  = saved_mutable
        self._readonly_locals = saved_readonly

    # ── Assignment visitors ───────────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._current_method:
            target = node.targets[0]
            rhs    = node.value

            self._check_lhs(target, node)

            if isinstance(target, ast.Name):
                self._classify_and_register(target.id, rhs, node)

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._current_method:
            self._check_lhs(node.target, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._current_method and node.value is not None:
            target = node.target
            rhs    = node.value

            if isinstance(target, ast.Name):
                self._classify_and_register(target.id, rhs, node)

            self._check_lhs(target, node)

        self.generic_visit(node)

    def _classify_and_register(
        self,
        var_name: str,
        rhs: ast.expr,
        node: ast.stmt,
    ) -> None:
        source = _rhs_source(rhs, self.ctx)

        if source == "other":
            return

        if source == "alias":
            raise CompilationError(
                f"Cannot assign '{self._rhs_name(rhs)}' to a new variable '{var_name}'.\n\n"
                f"How to fix this:\n"
                f"• If you want to make an independent snapshot to modify later, use copy:\n"
                f"    {var_name} = {self._rhs_name(rhs)}.copy()\n"
                f"• If you are trying to spawn a brand new agent, use its class name:\n"
                f"    {var_name} = {type_name(self.ctx.type_map.get(id(rhs)))}()",
                node=node,
            )

        if source in ("constructor", "copy", "self_call"):
            if source == "copy":
                self._check_copy_source_is_not_readonly(rhs, node)
            self._mutable_locals.add(var_name)
            self._readonly_locals.discard(var_name)
            return

        if source == "grid_getter":
            self._readonly_locals.add(var_name)
            self._mutable_locals.discard(var_name)
            return

    def _check_copy_source_is_not_readonly(
        self,
        copy_call: ast.expr,
        node: ast.stmt,
    ) -> None:
        pass

    # ── For-loop visitor ──────────────────────────────────────────────────────

    def visit_For(self, node: ast.For) -> None:
        if self._current_method and isinstance(node.target, ast.Name):
            # Use the loop variable's type from type_map (keyed on the target
            # Name node), not the iterator expression's type — generator
            # attributes return None from type_map because they have no scalar
            # WGSL type. TypeChecker writes the element type on id(node.target).
            loop_var  = node.target.id
            elem_type = self.ctx.type_map.get(id(node.target))
            is_domain_elem = (
                elem_type is not None
                and (
                    self.ctx.get_base_class(elem_type) is not None
                    or elem_type in _VECTOR_TYPES
                )
            )
            if is_domain_elem:
                self._readonly_locals.add(loop_var)
                self._mutable_locals.discard(loop_var)

        self.generic_visit(node)

    # ── Return visitor ────────────────────────────────────────────────────────

    def visit_Return(self, node: ast.Return) -> None:
        if self._current_method and node.value is not None:
            self._check_return_value(node)
        self.generic_visit(node)

    def _check_return_value(self, node: ast.Return) -> None:
        val = node.value
        if not isinstance(val, ast.Name):
            return

        var_name  = val.id
        var_type  = self.ctx.type_map.get(id(val))

        if var_type is None or self.ctx.get_base_class(var_type) is None:
            return

        if var_name in self._readonly_locals:
            raise CompilationError(
                f"Cannot return '{var_name}' directly.\n\n"
                f"'{var_name}' is a read-only object (like a neighbor cell or background tissue). "
                f"If you return it to modify it later, your changes will not affect the simulation.\n\n"
                f"How to fix this:\n"
                f"Return a copy to create an independent snapshot that you can safely modify:\n"
                f"    return {var_name}.copy()",
                node=node,
            )

    # ── LHS / attribute mutation checks ──────────────────────────────────────

    @staticmethod
    def _lhs_root_is_call(target: ast.Attribute) -> bool:
        """Return True when the deepest sub-expression of the LHS chain is a Call.

        Covers every pattern where a domain object is produced inline and
        never bound to a named variable, e.g.:
          self.tissue_at(pos).field = X   (grid-getter — Rule 2.4)
          self.helper().field = X         (user helper — Rule 2.5)
          MyCell().field = X              (constructor)
          some_obj.copy().field = X       (copy call)

        In all cases the object is a temporary: it has no name, cannot be
        tracked in _mutable_locals / _readonly_locals, and must be assigned
        to a variable before its fields can be written.
        """
        node = target.value
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Call)

    def _check_lhs(self, target: ast.expr, node) -> None:
        if isinstance(target, ast.Attribute):

            # ── Inline call write: any_call().field = X ───────────────────────
            if self._lhs_root_is_call(target):
                raise CompilationError(
                    f"Temper object cannot be modified'\n\n"
                    f"The object whose field '{target.attr}' you are trying to set "
                    f"was never assigned to a variable — it is a temporary.\n"
                    f"Temporary objects (returned from constructors, getters, helpers, "
                    f"or .copy() calls) are read-only until explicitly named.\n\n"
                    f"How to fix:\n"
                    f"Assign the object to a variable first, then modify it:\n"
                    f"    temp = <expression>\n"
                    f"    temp.{target.attr} = ...",
                    node=node,
                )

            root = target.value
            while isinstance(root, ast.Attribute):
                root = root.value

            if isinstance(root, ast.Name):
                obj_name = root.id

                if obj_name == "self":
                    if "self" in self._mutable_locals:
                        return
                    if not self._in_init:
                        raise CompilationError(
                            f" Cannot modify 'self' properties here.\n\n"
                            f"Simulation parameters and global variables can only be set inside the '__init__' method.",
                            node=node
                        )
                    return

                obj_type = self.ctx.type_map.get(id(root))
                if obj_type is None or self.ctx.get_base_class(obj_type) is None:
                    return

                if obj_name in self._readonly_locals:
                    raise CompilationError(
                        f"Cannot modify '{obj_name}'.\n\n"
                        f"'{obj_name}' is a read-only snapshot. "
                        f"In this framework, agents can only read their environment, they cannot directly alter other cells or tissues.\n\n"
                        f"How to fix:\n"
                        f"If you just need a temporary variable for calculations, make a copy first:\n"
                        f"    temp = {obj_name}.copy()\n"
                        f"    temp.some_field = ...", node=node
                    )

                if obj_name not in self._mutable_locals:
                    raise CompilationError(
                        f"Cannot modify '{obj_name}'.\n\n"
                        f"You are only allowed to change the properties of:\n"
                        f"1. The main argument of the rule (e.g., the current 'cell' or 'voxel').\n"
                        f"2. A brand new object you just created (e.g., 'new_obj = MyCell()').\n"
                        f"3. An explicit copy of an object (e.g., 'clone = obj.copy()').", node=node
                    )
            return

        if isinstance(target, ast.Name):
            var = target.id
            if var in self._pointer_args:
                raise CompilationError(
                    f"Forbidden reassignment of the main argument '{var}'.\n"
                    f"How to fix:\n"
                    f"Use {var}.copy_from(other_object)",
                    node=node
                )
            # Rule 3.4: reassignment to any function parameter is forbidden.
            if var in self._all_params and var not in self._pointer_args:
                raise CompilationError(
                    f"Forbidden reassignment of parameter '{var}'.\n\n"
                    f"Function parameters cannot be reassigned inside a function body.\n\n"
                    f"How to fix:\n"
                    f"• To copy the value of another object into '{var}', use:\n"
                    f"    {var}.copy_from(other_object)\n"
                    f"• Or introduce a new local variable for the new value:\n"
                    f"    local_{var} = <expression>",
                    node=node,
                )

    # ── Attribute access check ────────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._current_method:
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                self._check_attr_access(node)
        self.generic_visit(node)

    def _check_attr_access(self, node: ast.Attribute) -> None:
        parent_type = self.ctx.type_map.get(id(node.value))
        if parent_type is None:
            return

        if self.ctx.get_base_class(parent_type) is not None:
            valid = self.ctx.all_valid_attrs(parent_type)
            if valid and node.attr not in valid:
                raise CompilationError(
                    f"'{node.attr}' does not exist in '{type_name(parent_type)}'.\n\n"
                    f"Check your class definition to make sure this field is declared.", node=node
                )
            return

        if parent_type is vec3 and node.attr not in self._VEC3_ATTRS:
            if node.attr in _MUTABILITY_EXEMPT_METHODS:
                return   # .copy() / .copy_from() are valid on vectors
            raise CompilationError(
                f"'{node.attr}' is not a valid vec3 component.\n\n"
                f"Use .x, .y, or .z to access vector components.", node=node
            )

    # ── Call checks ───────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_method:
            if isinstance(node.func, ast.Attribute):
                self._check_call_ownership(node)
                self._check_helper_arg_mutability(node)
        self.generic_visit(node)

    def _check_call_ownership(self, node: ast.Call) -> None:
        """Forbid calling a mutating method on an object that is not a pointer-arg.

        For any ``obj.method()`` where ``obj`` is a domain-class instance and
        ``obj`` is NOT in ``_pointer_args``:
          - compute the mangled name via ctx.mangle_name
          - look up ctx.method_mutating_params
          - if position 0 (self) is marked mutating → CompilationError
        """
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            return

        obj_name    = func.value.id
        method_name = func.attr

        # Ignore self.X() — handled elsewhere; also skip if already a pointer-arg.
        # copy() and copy_from() are always safe regardless of mutability status.
        if obj_name == "self" or obj_name in self._pointer_args:
            return
        if method_name in _MUTABILITY_EXEMPT_METHODS:
            return

        obj_type = self.ctx.type_map.get(id(func.value))
        if obj_type is None or self.ctx.get_base_class(obj_type) is None:
            return

        mangled = self.ctx.mangle_name(obj_type, method_name)
        mutating = self.ctx.method_mutating_params.get(mangled, set())

        if 0 in mutating:
            if obj_name in self._mutable_locals:
                return
            if obj_name in self._readonly_locals:
                raise CompilationError(
                    f"Cannot call '{method_name}()' on '{obj_name}'.\n\n"
                    f"'{method_name}' mutates its receiver, but '{obj_name}' is "
                    f"read-only (e.g. returned from tissue_at / chemistry_at or a loop iterator).\n\n"
                    f"How to fix:\n"
                    f"• If you need a mutable snapshot, copy it first:\n"
                    f"    tmp = {obj_name}.copy()\n"
                    f"    tmp.{method_name}()",
                    node=node,
                )
            raise CompilationError(
                f"Cannot call '{method_name}()' on '{obj_name}'.\n\n"
                f"'{method_name}' mutates its receiver, but '{obj_name}' is not a "
                f"mutable object. Only [Mutable] objects (rule arguments, constructor "
                f"results, or explicit copies) may have mutating methods called on them.",
                node=node,
            )

    def _check_helper_arg_mutability(self, node: ast.Call) -> None:
        """Rule 3.1 — Forbid passing read-only or temporary domain objects to mutating parameters.

        For ``self.helper(arg0, arg1, ...)`` calls:
          - resolve the mangled name of the callee
          - for each positional argument at index j, callee parameter position = j + 1
          - if that position is in method_mutating_params:
              * if the arg is a named read-only local  → CompilationError
              * if the arg is an inline call (temporary) and NOT a mutable copy → CompilationError

        Rule 3.2 (read-only params) and Rule 3.3 (primitives) are implicitly satisfied
        because we only check domain-type arguments that land in mutating positions.
        """
        func = node.func
        # Only handle self.helper(...) calls.
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            return

        method_name = func.attr
        # Resolve mangled name — sim-class helpers use mangle_sim.
        mangled = self.ctx.mangle_name(None, method_name)
        mutating_positions: Set[int] = self.ctx.method_mutating_params.get(mangled, set())

        if not mutating_positions:
            return

        for j, arg in enumerate(node.args):
            param_pos = j + 1  # position 1-based (self=0 absent from args)
            if param_pos not in mutating_positions:
                continue

            arg_type = self.ctx.type_map.get(id(arg))
            if arg_type is None or self.ctx.get_base_class(arg_type) is None:
                continue

            if isinstance(arg, ast.Name):
                arg_name = arg.id
                if arg_name in self._readonly_locals:
                    raise CompilationError(
                        f"Cannot pass '{arg_name}' to '{method_name}()' at position {param_pos}.\n\n"
                        f"'{arg_name}' is a read-only object (e.g. returned from tissue_at / "
                        f"chemistry_at, a loop iterator, or a grid getter), but '{method_name}' "
                        f"mutates that parameter.\n\n"
                        f"How to fix:\n"
                        f"• If you need a mutable snapshot, copy it first:\n"
                        f"    tmp = {arg_name}.copy()\n"
                        f"    self.{method_name}(..., tmp, ...)\n"
                        f"• Or restructure so only the rule's primary agent is mutated.",
                        node=node,
                    )

            elif isinstance(arg, ast.Call):
                if _is_grid_getter_call(arg):
                    source_desc = f"the result of a grid getter (e.g. tissue_at / chemistry_at)"
                elif _is_copy_call(arg):
                    source_desc = (
                        f"an inline .copy() call — which is a read-only temporary.\n"
                        f"Assign the copy to a variable first:\n"
                        f"    tmp = <obj>.copy()\n"
                        f"    self.{method_name}(..., tmp, ...)"
                    )
                elif _is_self_call(arg):
                    source_desc = f"the inline return value of another helper"
                else:
                    source_desc = f"an inline temporary expression"

                raise CompilationError(
                    f"Cannot pass an inline temporary to '{method_name}()' at position {param_pos}.\n\n"
                    f"The argument is {source_desc}. "
                    f"Temporary domain objects are read-only and cannot be passed to a mutating parameter.\n\n"
                    f"How to fix:\n"
                    f"• Assign the expression to a named variable first, then pass the variable:\n"
                    f"    tmp = <expression>.copy()\n"
                    f"    self.{method_name}(..., tmp, ...)",
                    node=node,
                )



    @staticmethod
    def _rhs_name(rhs: ast.expr) -> str:
        if isinstance(rhs, ast.Name):
            return rhs.id
        if isinstance(rhs, ast.Attribute):
            return rhs.attr
        return "<expression>"
