from __future__ import annotations

import ast
from typing import Any, Dict, FrozenSet, List, Optional, TypeVar, get_type_hints

from oncolytica.core.utils._math import _INTRINSIC_FUNCTIONS, _INTRINSIC_ARG_TYPES
from oncolytica.core.utils._types import (
    i32, u32, f32,
    bool as ol_bool,
    vec3, ivec3,
    Tissue, Chemistry, Cell,
    PRIMITIVE_TYPES, BASE_CLASSES,
)
from oncolytica.core.utils._errors import CompilationError
from ._base import BaseValidator
from ._context import ValidationContext, type_name, mangle_sim, mangle_domain, TupleType, resolve_tuple_annotation

# ── Type-group helpers ────────────────────────────────────────────────────────

_FLOAT_TYPES: frozenset = frozenset({float, f32})
_INT_TYPES:   frozenset = frozenset({int, i32, u32})
_BOOL_TYPES:  frozenset = frozenset({ol_bool})
_VEC_TYPES:   frozenset = frozenset({vec3, ivec3})

_I32_BOOL_COMPAT: frozenset = frozenset({int, i32, ol_bool})
_U32_COMPAT:      frozenset = frozenset({int, u32})

_TYPE_RANK: Dict[Any, int] = {
    ol_bool: 0,
    int: 1, i32: 1, u32: 1,
    f32: 2, float: 2,
}

_ANNOTATION_NAMES: Dict[str, Any] = {
    "i32": i32, "u32": u32,
    "f32": f32, "bool": ol_bool,
    "vec3": vec3, "ivec3": ivec3,
    "int": int, "float": float,
}
_BUILTIN_NAMES: frozenset = frozenset({"self", "ol", "range", "True", "False", "None"})

_VEC3_COMPONENT_TYPE: Dict[Any, Any] = {vec3: f32, ivec3: i32}
_VEC_ATTRS: FrozenSet[str] = frozenset({"x", "y", "z"})

# Built-in self.X() grid-getter calls — resolved by special-case logic,
# not via method_return_hints.
_GRID_GETTERS: frozenset = frozenset({"tissue_at", "chemistry_at"})
_FRAMEWORK_SKIP_METHODS = frozenset({"copy", "copy_from"})

def _types_compatible(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a in _I32_BOOL_COMPAT and b in _I32_BOOL_COMPAT:
        return True
    if a in _FLOAT_TYPES and b in _FLOAT_TYPES:
        return True
    if a in _U32_COMPAT and b in _U32_COMPAT:
        return True
    return False


def _wider_type(a: Any, b: Any) -> Any:
    return a if _TYPE_RANK.get(a, -1) >= _TYPE_RANK.get(b, -1) else b


def _is_scalar(t: Any) -> bool:
    return t in _FLOAT_TYPES or t in _INT_TYPES or t in _BOOL_TYPES


def _resolve_annotation_node(ann: ast.expr) -> Optional[Any]:
    if isinstance(ann, ast.Name):
        return _ANNOTATION_NAMES.get(ann.id)
    if isinstance(ann, ast.Attribute):
        return _ANNOTATION_NAMES.get(ann.attr)
    return None


# =============================================================================
# PHASE 2 – TYPE CHECKER
# =============================================================================

class TypeChecker(BaseValidator):
    """
    Phase 2: Bottom-up type inference and type checking.

    Iterates sim-class method nodes in the order stored in ctx.method_nodes
    (which is insertion order = AST order).  All lookups into method_* dicts
    use **mangled names** (see _context.mangle_sim / mangle_domain).

    Writes
    ------
    ctx.type_map      – maps id(ast_node) → inferred type for every expression.
    ctx.symbol_table  – maps mangled_method_name → final local env after check.
    """

    _GENERATOR_ELEM_BASE: Dict[type, Dict[str, type]] = {
        Cell:      {"neighbors": Cell},
        Tissue:    {"cells": Cell, "neighbors": Tissue},
        Chemistry: {"cells": Cell, "tissues": Tissue, "neighbors": Chemistry},
    }

    def __init__(self) -> None:
        super().__init__()
        self._env: Dict[str, Any] = {}
        self._current_method: Optional[str] = None   # mangled name
        self._return_type: Optional[Any] = None

    # ── Entry point ───────────────────────────────────────────────────────────

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        ctx.type_map = {}
        for mangled_name, func_node in ctx.method_nodes.items():
            self._check_function(mangled_name, func_node)

    # ── Per-method check ──────────────────────────────────────────────────────

    def _check_function(self, mangled_name: str, node: ast.FunctionDef) -> None:
        saved = (self._env, self._current_method, self._return_type)

        self._current_method = mangled_name
        self._return_type    = self.ctx.method_return_hints.get(mangled_name)

        # Build local env from parameter hints.
        # For sim-methods "self" is absent from method_type_hints (excluded in
        # ContextBuilder); for domain-methods it is present at index 0 but
        # those nodes are not in ctx.method_nodes, so this branch only ever
        # sees sim-method hints where self is correctly absent.
        self._env = dict(self.ctx.method_type_hints.get(mangled_name, {}))

        for stmt in node.body:
            self._check_stmt(stmt)

        self.ctx.symbol_table[mangled_name] = {
            k: v for k, v in self._env.items() if k != "self"
        }
        self._env, self._current_method, self._return_type = saved

    # ── Statement dispatch ────────────────────────────────────────────────────

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
            for s in node.body:   self._check_stmt(s)
            for s in node.orelse: self._check_stmt(s)
        elif isinstance(node, ast.For):
            self._on_for(node)
        elif isinstance(node, ast.Expr):
            self._infer(node.value)

    # ── Assignment handlers ───────────────────────────────────────────────────

    def _on_assign(self, node: ast.Assign) -> None:
        rhs_type = self._infer(node.value)
        target   = node.targets[0]

        # ── Tuple unpacking:  a, b = func() ──────────────────────────────────
        if isinstance(target, ast.Tuple):
            if not isinstance(rhs_type, TupleType):
                raise CompilationError(
                    f"Cannot unpack: the right-hand side does not return a tuple type. "
                    f"Got '{type_name(rhs_type)}'.",
                    node=node,
                )
            if len(target.elts) != len(rhs_type.elements):
                raise CompilationError(
                    f"Cannot unpack {len(rhs_type.elements)}-element tuple "
                    f"into {len(target.elts)} variables.",
                    node=node,
                )
            self.ctx.collected_tuple_types.add(rhs_type)
            for elt, elem_type in zip(target.elts, rhs_type.elements):
                if not isinstance(elt, ast.Name):
                    continue
                var = elt.id
                resolved = i32 if elem_type in _I32_BOOL_COMPAT else elem_type
                existing = self._env.get(var)
                if existing is not None and not _types_compatible(existing, resolved):
                    raise CompilationError(
                        f"Type mismatch: cannot unpack '{type_name(resolved)}' "
                        f"into '{var}' (declared type '{type_name(existing)}').",
                        node=node,
                    )
                self._env[var] = resolved
                self.ctx.type_map[id(elt)] = resolved
            return

        target_name = (
            target.id if isinstance(target, ast.Name)
            else getattr(target, "attr", "<attribute>")
        )

        if isinstance(rhs_type, TupleType):
            raise CompilationError(
                f"Cannot assign a tuple to '{target_name}'. "
                f"Tuples must be unpacked immediately (e.g. 'a, b = func()') "
                f"or indexed directly (e.g. 'func()[0]').",
                node=node,
            )

        if rhs_type is None:
            raise CompilationError(
                f"Cannot infer type for assignment to '{target_name}'. "
                f"The right-hand side has an unknown or unresolvable type. "
                f"Hint: use an explicit type annotation "
                f"(e.g. '{target_name}: Type = ...').",
                node=node,
            )

        if isinstance(target, ast.Name):
            var = target.id
            if rhs_type in _I32_BOOL_COMPAT:
                rhs_type = i32

            existing = self._env.get(var)
            if existing is not None:
                if not _types_compatible(existing, rhs_type):
                    raise CompilationError(
                        f"Type mismatch: cannot assign '{type_name(rhs_type)}' "
                        f"to '{var}' (declared type '{type_name(existing)}').",
                        node=node,
                    )
                self.ctx.type_map[id(target)] = existing
            else:
                self._env[var] = rhs_type
                self.ctx.type_map[id(target)] = rhs_type

        elif isinstance(target, ast.Attribute):
            target_type = self._infer(target)
            if target_type is not None and rhs_type is not None:
                if not _types_compatible(target_type, rhs_type):
                    raise CompilationError(
                        f"Type mismatch: cannot assign '{type_name(rhs_type)}' "
                        f"to attribute '{target.attr}' "
                        f"(declared type '{type_name(target_type)}').",
                        node=node,
                    )

    def _on_ann_assign(self, node: ast.AnnAssign) -> None:
        ann_type = _resolve_annotation_node(node.annotation)
        # ol.bool / bool annotations declare an i32 variable (no bool in WGSL structs).
        if ann_type in _BOOL_TYPES:
            ann_type = i32
        val_type = self._infer(node.value) if node.value else None

        if isinstance(val_type, TupleType) or isinstance(ann_type, TupleType):
            target_name = getattr(node.target, "id", "<attribute>")
            raise CompilationError(
                f"Cannot assign a tuple to '{target_name}'. "
                f"Tuples must be unpacked immediately.",
                node=node,
            )

        if ann_type is not None and val_type is not None:
            if not _types_compatible(ann_type, val_type):
                raise CompilationError(
                    f"Type mismatch: declared '{type_name(ann_type)}' "
                    f"but value has type '{type_name(val_type)}'.",
                    node=node,
                )

        if isinstance(node.target, ast.Name):
            var      = node.target.id
            resolved = ann_type if ann_type is not None else val_type

            if resolved is None:
                raise CompilationError(
                    f"Cannot resolve type annotation for '{var}'.", node=node
                )

            existing = self._env.get(var)
            if existing is not None and existing is not resolved:
                if not _types_compatible(existing, resolved):
                    raise CompilationError(
                        f"Re-declaration of '{var}' with incompatible type "
                        f"'{type_name(resolved)}' "
                        f"(previously '{type_name(existing)}').",
                        node=node,
                    )
                self.ctx.type_map[id(node.target)] = existing
            else:
                self._env[var] = resolved
                self.ctx.type_map[id(node.target)] = resolved

    def _on_aug_assign(self, node: ast.AugAssign) -> None:
        target_type = self._infer(node.target)
        value_type  = self._infer(node.value)

        if value_type is None:
            raise CompilationError(
                f"Cannot infer type for the right-hand side of "
                f"'{type(node.op).__name__}='.",
                node=node,
            )

        if target_type is not None and value_type is not None:
            if not _types_compatible(target_type, value_type):
                raise CompilationError(
                    f"Type mismatch: cannot apply '{type(node.op).__name__}=' to "
                    f"'{type_name(target_type)}' and '{type_name(value_type)}'.",
                    node=node,
                )

    def _on_return(self, node: ast.Return) -> None:
        expected = self._return_type
        is_void  = expected is None or expected is type(None)

        if node.value is None:
            if not is_void:
                raise CompilationError(
                    f"Method '{self._current_method}' "
                    f"must return '{type_name(expected)}'.",
                    node=node,
                )
            return

        # ── TupleType return: return (a, b) ──────────────────────────────────
        if isinstance(expected, TupleType):
            self.ctx.collected_tuple_types.add(expected)
            if not isinstance(node.value, ast.Tuple):
                raise CompilationError(
                    f"Return type mismatch in '{self._current_method}': "
                    f"expected a tuple expression like '(a, b)', got a single value.",
                    node=node,
                )
            if len(node.value.elts) != len(expected.elements):
                raise CompilationError(
                    f"Return type mismatch in '{self._current_method}': "
                    f"expected {len(expected.elements)}-element tuple, "
                    f"got {len(node.value.elts)}.",
                    node=node,
                )
            for elt, exp_elem in zip(node.value.elts, expected.elements):
                act_elem = self._infer(elt)
                if act_elem is not None and not _types_compatible(act_elem, exp_elem):
                    raise CompilationError(
                        f"Return type mismatch in '{self._current_method}': "
                        f"tuple element expected '{type_name(exp_elem)}', "
                        f"got '{type_name(act_elem)}'.",
                        node=node,
                    )
            # Record the TupleType on the Tuple node itself so the statement
            # translator can look it up via type_map when emitting the return.
            self.ctx.type_map[id(node.value)] = expected
            return

        actual = self._infer(node.value)
        if not is_void and actual is not None:
            if not _types_compatible(actual, expected):
                raise CompilationError(
                    f"Return type mismatch in '{self._current_method}': "
                    f"expected '{type_name(expected)}', got '{type_name(actual)}'.",
                    node=node,
                )

    def _on_for(self, node: ast.For) -> None:
        it       = node.iter
        loop_var = node.target.id if isinstance(node.target, ast.Name) else None

        if (isinstance(it, ast.Call)
                and isinstance(it.func, ast.Name)
                and it.func.id == "range"):
            for arg in it.args:
                t = self._infer(arg)
                if t is not None and t not in _INT_TYPES | _BOOL_TYPES:
                    raise CompilationError(
                        f"range() argument must be an integer type, "
                        f"got '{type_name(t)}'.",
                        node=node,
                    )
            if loop_var:
                self._env[loop_var] = i32
                self.ctx.type_map[id(node.target)] = i32

        elif isinstance(it, ast.Attribute):
            container_type = self._infer(it.value)
            elem_type      = self._resolve_generator_elem_type(
                container_type, it.attr, node
            )
            if loop_var and elem_type is not None:
                self._env[loop_var] = elem_type
                self.ctx.type_map[id(node.target)] = elem_type

        for s in node.body:
            self._check_stmt(s)

    # ── Type inference ────────────────────────────────────────────────────────

    def _infer(self, node: Optional[ast.expr]) -> Optional[Any]:
        if node is None:
            return None
        t = self._infer_impl(node)
        if t is not None:
            self.ctx.type_map[id(node)] = t
        return t

    def _infer_impl(self, node: ast.expr) -> Optional[Any]:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):  return ol_bool
            if isinstance(v, int):   return int
            if isinstance(v, float): return float
            return None

        if isinstance(node, ast.Name):
            if node.id in ("True", "False"):
                return ol_bool
            if node.id in _BUILTIN_NAMES:
                return None
            if node.id in _ANNOTATION_NAMES:
                return _ANNOTATION_NAMES[node.id]
            cv = self.ctx.constants.get(node.id) if self.ctx else None
            if cv is not None:
                return type(cv)
            t = self._env.get(node.id)
            if t is None:
                raise CompilationError(
                    f"Variable '{node.id}' has no known type at this point.",
                    node=node,
                )
            return t

        if isinstance(node, ast.Attribute):
            # self.params.attr  →  Uniforms field lookup
            if (isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"
                    and node.value.attr == "params"):
                params_class = type(self.ctx.sim_instance._params)
                hints        = get_type_hints(params_class)
                hint         = hints.get(node.attr)
                if hint in _BOOL_TYPES:
                    return i32
                return hint

            parent_t = self._infer(node.value)
            return self._get_attr_type(parent_t, node.attr, node)

        if isinstance(node, ast.BinOp):
            lt = self._infer(node.left)
            rt = self._infer(node.right)
            return self._check_binop(node, lt, rt)

        if isinstance(node, ast.BoolOp):
            for v in node.values:
                self._infer(v)
            return ol_bool

        if isinstance(node, ast.UnaryOp):
            operand_type = self._infer(node.operand)
            if isinstance(node.op, ast.Not):
                return ol_bool
            return operand_type

        if isinstance(node, ast.IfExp):
            self._infer(node.test)
            t = self._infer(node.body)
            self._infer(node.orelse)
            return t

        if isinstance(node, ast.Subscript):
            container_type = self._infer(node.value)
            if isinstance(container_type, TupleType):
                slice_val = node.slice
                if isinstance(slice_val, ast.Constant) and isinstance(slice_val.value, int):
                    idx = slice_val.value
                    if 0 <= idx < len(container_type.elements):
                        return container_type.elements[idx]
                    raise CompilationError(
                        f"Tuple index {idx} out of range for tuple of length {len(container_type.elements)}.",
                        node=node
                    )
                raise CompilationError(
                    "Tuples can only be indexed with integer constants.",
                    node=node
                )
            return None

        if isinstance(node, ast.Call):
            return self._infer_call(node)

        if isinstance(node, ast.Compare):
            self._infer(node.left)
            for c in node.comparators:
                self._infer(c)
            return ol_bool

        return None

    # ── Call inference ────────────────────────────────────────────────────────

    def _infer_call(self, node: ast.Call) -> Optional[Any]:
        # Infer all argument types first so type_map is populated for callees.
        for arg in node.args:
            self._infer(arg)
        for kw in node.keywords:
            self._infer(kw.value)

        func = node.func

        # ── 1. self.method() — sim-class method or built-in grid getter ───────
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"):

            method_name = func.attr

            # Built-in grid getters: return the matching user domain class.
            if method_name == "tissue_at":
                for user_cls, base_cls in self.ctx.memory_base_map.items():
                    if base_cls is Tissue:
                        return user_cls

            if method_name == "chemistry_at":
                for user_cls, base_cls in self.ctx.memory_base_map.items():
                    if base_cls is Chemistry:
                        return user_cls

            # Regular sim-method: look up via mangled name.
            ret = self.ctx.method_return_hints.get(mangle_sim(method_name))
            if isinstance(ret, TupleType):
                self.ctx.collected_tuple_types.add(ret)
            return ret

        # ── 2. obj.copy() — returns same type as obj ──────────────────────────
        # Applies to domain objects (Cell, Tissue, …) AND vectors (vec3, ivec3).
        # vec3 is classified as "primitive" by _is_primitive_type, but .copy()
        # on a vector is explicitly permitted by Rule 1.3 and must preserve type.
        if (isinstance(func, ast.Attribute)
                and func.attr == "copy"
                and not node.args):
            obj_type = self._infer(func.value)
            if obj_type is not None and (
                not self._is_primitive_type(obj_type)   # domain objects
                or obj_type in _VEC_TYPES               # vec3 / ivec3
            ):
                return obj_type

        # ── 3. obj.method() — method on a domain object ───────────────────────
        if isinstance(func, ast.Attribute):
            obj_type = self._infer(func.value)
            if obj_type is not None and not self._is_primitive_type(obj_type):
                base_cls = self.ctx.get_base_class(obj_type)
                if base_cls is not None:
                    mangled = mangle_domain(base_cls, func.attr)
                    return self.ctx.method_return_hints.get(mangled)

        # ── 4. Intrinsic (ol.math) functions ──────────────────────────────────
        func_name: Optional[str] = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if func_name and func_name in _INTRINSIC_FUNCTIONS:
            self._check_intrinsic_args(func_name, node)
            if func_name in {"clamp", "abs", "sign", "min", "max"}:
                if node.args:
                    arg_type = self._infer(node.args[0])
                    if arg_type is not None:
                        return arg_type
            return _INTRINSIC_FUNCTIONS[func_name]

        # ── 5. Constructor calls ───────────────────────────────────────────────
        if func_name:
            if func_name in _ANNOTATION_NAMES:
                return _ANNOTATION_NAMES[func_name]

            for registered_type in self.ctx.class_field_types.keys():
                if getattr(registered_type, "__name__", None) == func_name:
                    return registered_type

            for user_cls in self.ctx.memory_base_map.keys():
                if getattr(user_cls, "__name__", None) == func_name:
                    return user_cls

        return None

    # ── Attribute type lookup ─────────────────────────────────────────────────

    def _get_attr_type(
        self,
        parent_type: Any,
        attr: str,
        node: ast.Attribute,
    ) -> Optional[Any]:
        if parent_type is None:
            return None

        if parent_type in _VEC_TYPES:
            if attr in _VEC_ATTRS:
                return _VEC3_COMPONENT_TYPE[parent_type]
            raise CompilationError(
                f"'{attr}' is not a valid vec3 component (use .x, .y, or .z).",
                node=node,
            )

        field_types = self.ctx.class_field_types.get(parent_type)
        if field_types is not None:
            t = field_types.get(attr)
            if t is not None:
                # Bool fields are physically i32 in WGSL structs.
                if t in _BOOL_TYPES:
                    return i32
                return t

            # Method access as attribute (e.g. used in call expression):
            # not an error, just no type to return.
            if attr in self.ctx.class_methods.get(parent_type, set()):
                return None

            # Generator attribute (iterable property): valid, no scalar type.
            if attr in self.ctx.class_generators.get(parent_type, set()):
                return None

            raise CompilationError(
                f"'{type_name(parent_type)}' has no field or property '{attr}'.",
                node=node,
            )

        return None

    # ── Generator element type resolution ────────────────────────────────────

    def _resolve_generator_elem_type(
        self,
        container_type: Any,
        attr: str,
        node: ast.For,
    ) -> Optional[Any]:
        if container_type is None:
            return None

        base      = self.ctx.get_base_class(container_type)
        elem_base = self._GENERATOR_ELEM_BASE.get(base, {}).get(attr)

        if elem_base is None:
            field_types = self.ctx.class_field_types.get(container_type, {})
            if attr in field_types:
                raise CompilationError(
                    f"'{type_name(container_type)}.{attr}' is not iterable.",
                    node=node,
                )
            raise CompilationError(
                f"'{type_name(base)}' has no generator attribute '{attr}'.",
                node=node,
            )

        for user_cls, base_cls in self.ctx.memory_base_map.items():
            if base_cls is elem_base:
                return user_cls
        return elem_base

    # ── Binary operation type checking ────────────────────────────────────────

    def _is_primitive_type(self, t: Any) -> bool:
        return _is_scalar(t) or t in _VEC_TYPES

    def _check_binop(
        self,
        node: ast.BinOp,
        lt: Optional[Any],
        rt: Optional[Any],
    ) -> Optional[Any]:
        if lt is None or rt is None:
            raise CompilationError(
                f"Cannot determine type of operands in binary expression.",
                node=node,
            )

        l_is_vec    = lt in _VEC_TYPES
        r_is_vec    = rt in _VEC_TYPES
        l_is_scalar = _is_scalar(lt)
        r_is_scalar = _is_scalar(rt)

        if l_is_vec and r_is_vec:
            if lt is not rt:
                raise CompilationError(
                    f"Cannot mix '{type_name(lt)}' and '{type_name(rt)}'.",
                    node=node,
                )
            return lt

        if l_is_vec and r_is_scalar:
            if not isinstance(node.op, (ast.Mult, ast.Div)):
                raise CompilationError(
                    f"Unsupported vector-scalar operation.", node=node
                )
            return lt

        if l_is_scalar and r_is_vec:
            if not isinstance(node.op, ast.Mult):
                raise CompilationError(
                    f"Unsupported scalar-vector operation.", node=node
                )
            return rt

        if l_is_scalar and r_is_scalar:
            if not _types_compatible(lt, rt):
                raise CompilationError(
                    f"Type mismatch in binary operation.", node=node
                )
            if lt in _BOOL_TYPES and rt in _BOOL_TYPES:
                return i32
            return _wider_type(lt, rt)

        raise CompilationError(
            f"Unsupported operand types: '{type_name(lt)}' and '{type_name(rt)}'.",
            node=node,
        )

    # ── Intrinsic argument checking ───────────────────────────────────────────

    def _check_intrinsic_args(self, func_name: str, node: ast.Call) -> None:
        given_args = node.args
        n_given    = len(given_args)

        if func_name == "clamp":
            if n_given != 3:
                raise CompilationError(
                    f"'{func_name}' expects 3 argument(s), got {n_given}.",
                    node=node,
                )
            self._check_clamp_args(node)
            return

        constraints = _INTRINSIC_ARG_TYPES.get(func_name)
        if constraints is None:
            return  # No constraints registered → accept anything.

        n_expected = len(constraints)
        if n_given != n_expected:
            raise CompilationError(
                f"'{func_name}' expects {n_expected} argument(s), got {n_given}.",
                node=node,
            )

        if func_name in {"min", "max"}:
            t1 = self._infer(given_args[0])
            t2 = self._infer(given_args[1])
            for idx, t in enumerate((t1, t2)):
                if t is not None and t not in _FLOAT_TYPES and t not in _INT_TYPES:
                    raise CompilationError(
                        f"Argument {idx + 1} of '{func_name}' must be f32 or i32, "
                        f"got '{type_name(t)}'.",
                        node=node,
                    )
            if t1 is not None and t2 is not None:
                if (t1 in _FLOAT_TYPES) != (t2 in _FLOAT_TYPES):
                    raise CompilationError(
                        f"Type mismatch: '{func_name}' arguments must have the same "
                        f"type, got '{type_name(t1)}' and '{type_name(t2)}'.",
                        node=node,
                    )

        for idx, (arg_node, allowed_types) in enumerate(zip(given_args, constraints)):
            arg_type = self._infer(arg_node)
            if arg_type is None:
                continue
            if arg_type not in allowed_types:
                nice_allowed = " | ".join(
                    type_name(t) for t in sorted(allowed_types, key=type_name)
                )
                raise CompilationError(
                    f"Argument {idx + 1} of '{func_name}' must be {nice_allowed}, "
                    f"got '{type_name(arg_type)}'. "
                    f"Hint: use a float literal (e.g. {func_name}(5.0)) "
                    f"or an explicit cast (e.g. {func_name}(f32(x))).",
                    node=arg_node,
                )

    def _check_clamp_args(self, node: ast.Call) -> None:
        """clamp is overloaded: (f32,f32,f32) or (vec3,vec3,vec3)."""
        arg_types = [self._infer(a) for a in node.args]
        known     = [t for t in arg_types if t is not None]
        if not known:
            return

        first    = known[0]
        is_vec   = first in _VEC_TYPES
        is_float = first in _FLOAT_TYPES

        if not (is_vec or is_float):
            raise CompilationError(
                f"'clamp' requires f32 or vec3 arguments, got '{type_name(first)}'. "
                f"Hint: use a float literal or an explicit f32(...) cast.",
                node=node,
            )

        for idx, t in enumerate(arg_types):
            if t is None:
                continue
            if is_vec and t not in _VEC_TYPES:
                raise CompilationError(
                    f"Argument {idx + 1} of 'clamp' must be vec3 "
                    f"(to match argument 1), got '{type_name(t)}'.",
                    node=node,
                )
            if is_float and t not in _FLOAT_TYPES:
                raise CompilationError(
                    f"Argument {idx + 1} of 'clamp' must be f32 "
                    f"(to match argument 1), got '{type_name(t)}'.",
                    node=node,
                )
