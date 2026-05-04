from __future__ import annotations

import ast
import inspect
import textwrap
from typing import get_origin, get_type_hints, get_args
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Generator, Annotated, Iterator

from oncolytica.core.utils._types import (
    BASE_CLASSES, ANNOTATION_NAMES, ivec3,
    i32, u32, f32,
    bool as ol_bool,
    vec3,
    Tissue, Chemistry, Cell, Metrics,
    PRIMITIVE_TYPES,
)
from oncolytica.core.utils._errors import CompilationError


# ── TupleType ─────────────────────────────────────────────────────────────────

_WGSL_SHORT: Dict[Any, str] = {}


def _wgsl_elem_name(t: Any) -> str:
    from oncolytica.gpu.compiler._type_system import py_type_to_wgsl
    try:
        return py_type_to_wgsl(t)
    except TypeError:
        return getattr(t, "__name__", repr(t))


class TupleType:
    __slots__ = ("elements",)

    def __init__(self, *elements: Any) -> None:
        self.elements: tuple = tuple(elements)

    def wgsl_name(self) -> str:
        return "Tuple_" + "_".join(_wgsl_elem_name(t) for t in self.elements)

    def __repr__(self) -> str:
        names = ", ".join(getattr(t, "__name__", repr(t)) for t in self.elements)
        return f"TupleType({names})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleType) and self.elements == other.elements

    def __hash__(self) -> int:
        return hash(("TupleType", self.elements))


def resolve_tuple_annotation(hint: Any) -> Optional["TupleType"]:
    import typing
    origin = get_origin(hint)
    if origin not in (tuple, typing.Tuple):
        return None
    args = get_args(hint)
    if not args:
        return None
    if len(args) == 2 and args[1] is Ellipsis:
        return None
    return TupleType(*args)


RULE_DECORATOR_NAMES: FrozenSet[str] = frozenset({
    "cell_rule", "tissue_rule", "chemistry_rule", "metric_rule",
})

SYSTEM_FIELDS: Dict[type, FrozenSet[str]] = {
    Cell: frozenset({"pos"}),
}

_BASE_PREFIX: Dict[type, str] = {
    Cell:      "cell",
    Tissue:    "tissue",
    Chemistry: "chemistry",
    Metrics:   "metrics",
}


def _is_gpu_compatible_type(t: Any) -> bool:
    if t in PRIMITIVE_TYPES:
        return True
    if not isinstance(t, type):
        return False
    return any(issubclass(t, base) for base in BASE_CLASSES if isinstance(base, type))


def mangle_sim(method_name: str) -> str:
    return f"sim_{method_name}"


def mangle_domain(base_cls: type, method_name: str) -> str:
    prefix = _BASE_PREFIX.get(base_cls, base_cls.__name__.lower())
    return f"{prefix}_{method_name}"


def type_name(t: Any) -> str:
    return getattr(t, "__name__", repr(t))


# =============================================================================
# VALIDATION CONTEXT
# =============================================================================

class ValidationContext:
    def __init__(self, sim_instance: Any) -> None:
        self.sim_instance: Any = sim_instance

        self.memory_base_map: Dict[type, type] = {
            sim_instance._spec.tissue_class:    Tissue,
            sim_instance._spec.chemistry_class: Chemistry,
            sim_instance._spec.cell_class:      Cell,
            sim_instance._spec.metrics_class:   Metrics,
        }

        self.source_code: str = ""
        self.tree: Optional[ast.Module] = None

        self.method_signatures:   Dict[str, inspect.Signature] = {}
        self.method_type_hints:   Dict[str, Dict[str, Any]] = {}
        self.method_return_hints: Dict[str, Optional[Any]] = {}

        self.class_fields:        Dict[type, Set[str]] = {}
        self.class_properties:    Dict[type, Set[str]] = {}
        self.class_methods:       Dict[type, Set[str]] = {}
        self.class_generators:    Dict[type, Set[str]] = {}
        self.class_field_types:   Dict[type, Dict[str, Any]] = {}
        self.constants:           Dict[str, Any] = {}

        # Sim-method AST nodes — keyed by mangled sim name ("sim_spawn").
        self.method_nodes:        Dict[str, ast.FunctionDef] = {}

        # Domain-method AST nodes — keyed by (user_cls, raw_method_name).
        # Populated by ContextBuilder._build_domain_method_signatures.
        self.domain_method_nodes: Dict[Tuple[type, str], ast.FunctionDef] = {}

        self.rule_method_names:   Set[str] = set()
        self.helper_method_names: Set[str] = set()

        self.symbol_table:    Dict[str, Dict[str, Optional[Any]]] = {}
        self.type_map:        Dict[int, Any] = {}
        self.call_graph:      Dict[str, Set[str]] = {}
        self.ordered_methods: List[str] = []

        self.method_locations: Dict[str, Tuple[str, int]] = {}

        self.method_mutating_params: Dict[str, Set[int]] = {}

        self.collected_tuple_types: Set[TupleType] = set()

    def mangle_name(self, cls: Optional[type], method_name: str) -> str:
        if cls is None:
            return mangle_sim(method_name)
        base = self.memory_base_map.get(cls)
        if base is None:
            raise ValueError(f"'{type_name(cls)}' is not a registered domain class.")
        return mangle_domain(base, method_name)

    def all_valid_attrs(self, cls: type) -> Set[str]:
        return (
            self.class_fields.get(cls, set())
            | self.class_properties.get(cls, set())
            | self.class_methods.get(cls, set())
            | self.class_generators.get(cls, set())
        )

    def lookup_memory_class_by_ast_name(self, name: str) -> Optional[type]:
        for cls in self.memory_base_map:
            if cls.__name__ == name:
                return cls
        return None

    def is_memory_class(self, cls: type) -> bool:
        return cls in self.memory_base_map

    def get_base_class(self, memory_cls: type) -> Optional[type]:
        return self.memory_base_map.get(memory_cls)


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

class ContextBuilder:

    @staticmethod
    def build(sim_instance: Any) -> ValidationContext:
        ctx = ValidationContext(sim_instance)
        sim_class = sim_instance.__class__
        ContextBuilder._build_ast(ctx, sim_class)
        ContextBuilder._build_sim_signatures(ctx, sim_class)
        ContextBuilder._build_class_metadata(ctx)
        ContextBuilder._build_domain_method_signatures(ctx)
        ContextBuilder._build_constants(ctx)
        ContextBuilder._classify_methods(ctx)
        return ctx

    @staticmethod
    def _build_ast(ctx: ValidationContext, sim_class: type) -> None:
        try:
            source = inspect.getsource(sim_class)
            ctx.source_code = textwrap.dedent(source)
        except (OSError, TypeError) as exc:
            raise CompilationError(
                f"Cannot retrieve source for '{sim_class.__name__}': {exc}."
            ) from exc

        try:
            ctx.tree = ast.parse(ctx.source_code)
        except SyntaxError as exc:
            raise CompilationError(
                f"Syntax error in '{sim_class.__name__}': {exc}"
            ) from exc

        ctx.tree.parent = None  # type: ignore[attr-defined]
        for node in ast.walk(ctx.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore[attr-defined]

        seen_raw: Dict[str, int] = {}
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in seen_raw:
                    raise CompilationError(
                        f"Line {node.lineno}: function '{node.name}' is already defined."
                    )
                seen_raw[node.name] = node.lineno
                ctx.method_nodes[mangle_sim(node.name)] = node

    @staticmethod
    def _build_sim_signatures(ctx: ValidationContext, sim_class: type) -> None:
        for mangled, node in ctx.method_nodes.items():
            raw_name = node.name
            method = getattr(sim_class, raw_name, None)
            param_hints: Dict[str, Any] = {}
            return_hint: Optional[Any] = None

            if method is not None:
                try:
                    ctx.method_signatures[mangled] = inspect.signature(method)
                    param_hints, return_hint = ContextBuilder._resolve_hints(method)
                except (ValueError, TypeError):
                    pass
                try:
                    filename = inspect.getsourcefile(method) or "<unknown>"
                    _, start_line = inspect.getsourcelines(method)
                    ctx.method_locations[mangled] = (filename, start_line)
                except (OSError, TypeError):
                    ctx.method_locations[mangled] = ("<unknown>", 1)
            else:
                ctx.method_locations[mangled] = ("<unknown>", 1)

            clean_hints: Dict[str, Any] = {}
            for arg in node.args.args:
                arg_name = arg.arg
                if arg_name == "self":
                    continue
                if arg_name in param_hints:
                    clean_hints[arg_name] = param_hints[arg_name]
                elif arg.annotation:
                    resolved = ContextBuilder._resolve_ast_annotation(arg.annotation, ctx)
                    if resolved is not None:
                        clean_hints[arg_name] = resolved

            if return_hint is None and node.returns:
                resolved = ContextBuilder._resolve_ast_annotation(node.returns, ctx)
                if resolved is not None:
                    return_hint = resolved

            ctx.method_type_hints[mangled]   = clean_hints
            ctx.method_return_hints[mangled] = return_hint

    @staticmethod
    def _build_domain_method_signatures(ctx: ValidationContext) -> None:
        """Build signatures and AST nodes for all domain-class methods.

        AST nodes are stored in ctx.domain_method_nodes keyed by
        (user_cls, raw_method_name) so CallGraphValidator can walk their bodies.
        """
        for user_cls, base_cls in ctx.memory_base_map.items():
            for raw_method_name in ctx.class_methods.get(user_cls, set()):
                mangled = mangle_domain(base_cls, raw_method_name)

                method = getattr(user_cls, raw_method_name, None)
                if method is None:
                    continue

                param_hints: Dict[str, Any] = {}
                return_hint: Optional[Any] = None

                try:
                    ctx.method_signatures[mangled] = inspect.signature(method)
                    param_hints, return_hint = ContextBuilder._resolve_hints(method)
                    if raw_method_name == "copy" and return_hint is Any:
                        return_hint = user_cls
                except (ValueError, TypeError):
                    pass

                try:
                    filename = inspect.getsourcefile(method) or "<unknown>"
                    _, start_line = inspect.getsourcelines(method)
                    ctx.method_locations[mangled] = (filename, start_line)
                except (OSError, TypeError):
                    ctx.method_locations[mangled] = ("<unknown>", 1)

                # ── Parse AST for this domain method ──────────────────────────
                func_node: Optional[ast.FunctionDef] = None
                try:
                    raw_func = getattr(method, "__func__", method)
                    src = textwrap.dedent(inspect.getsource(raw_func))
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_node = node
                            break
                except (OSError, TypeError, SyntaxError):
                    pass

                if func_node is not None:
                    ctx.domain_method_nodes[(user_cls, raw_method_name)] = func_node

                # ── Build param hints ─────────────────────────────────────────
                clean_hints: Dict[str, Any] = {}
                clean_hints["self"] = user_cls

                try:
                    sig = inspect.signature(method)
                    for pname, param in sig.parameters.items():
                        if pname == "self":
                            continue
                        if pname in param_hints:
                            clean_hints[pname] = param_hints[pname]
                        elif param.annotation is not inspect.Parameter.empty:
                            ann = param.annotation
                            if isinstance(ann, str):
                                ann = ANNOTATION_NAMES.get(ann)
                            if ann is not None:
                                clean_hints[pname] = ann
                except (ValueError, TypeError):
                    pass

                ctx.method_type_hints[mangled]   = clean_hints
                ctx.method_return_hints[mangled] = return_hint

    @staticmethod
    def _resolve_ast_annotation(ann: ast.expr, ctx: ValidationContext) -> Any:
        if isinstance(ann, ast.Name):
            name = ann.id
        elif isinstance(ann, ast.Attribute):
            name = ann.attr
        else:
            return None

        if name in ANNOTATION_NAMES:
            return ANNOTATION_NAMES[name]
        for cls in ctx.memory_base_map:
            if getattr(cls, "__name__", "") == name:
                return cls
        if name == "Metrics":
            return Metrics
        return None

    @staticmethod
    def _build_class_metadata(ctx: ValidationContext) -> None:
        for cls in ctx.memory_base_map.keys():
            ContextBuilder._collect_fields(ctx, cls)
            ContextBuilder._collect_properties(ctx, cls)
            ContextBuilder._collect_methods(ctx, cls)
            ContextBuilder._collect_field_types(ctx, cls)

    @staticmethod
    def _collect_fields(ctx: ValidationContext, cls: type) -> None:
        fields: Set[str] = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            for fname in klass.__dict__.get("__annotations__", {}):
                if not fname.startswith("_"):
                    fields.add(fname)
        for base_cls, sys_fields in SYSTEM_FIELDS.items():
            if issubclass(cls, base_cls):
                fields.update(sys_fields)
        ctx.class_fields[cls] = fields

    @staticmethod
    def _collect_properties(ctx: ValidationContext, cls: type) -> None:
        properties: Set[str] = set()
        generators: Set[str] = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            for fname, val in klass.__dict__.items():
                if not fname.startswith("_") and isinstance(val, property):
                    if ContextBuilder._is_generator_property(val):
                        generators.add(fname)
                    else:
                        properties.add(fname)
        ctx.class_properties[cls] = properties
        ctx.class_generators[cls] = generators

    @staticmethod
    def _is_generator_property(prop: property) -> bool:
        getter = prop.fget
        if getter is None:
            return False
        try:
            hints = get_type_hints(getter)
            ret_type = hints.get("return")
            if ret_type is None:
                return False
            return get_origin(ret_type) is Generator
        except Exception:
            ann = getattr(getter, "__annotations__", {})
            ret_type = ann.get("return")
            if ret_type is None:
                return False
            if isinstance(ret_type, str):
                return ret_type.startswith(
                    ("Generator", "collections.abc.Generator", "typing.Generator")
                )
            origin = get_origin(ret_type)
            return origin is Generator or getattr(ret_type, "__name__", "") == "Generator"

    @staticmethod
    def _collect_methods(ctx: ValidationContext, cls: type) -> None:
        methods: Set[str] = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            for fname, val in klass.__dict__.items():
                if not fname.startswith("_") and inspect.isfunction(val):
                    methods.add(fname)
        ctx.class_methods[cls] = methods

    @staticmethod
    def _collect_field_types(ctx: ValidationContext, cls: type) -> None:
        field_types: Dict[str, Any] = {}
        hints = get_type_hints(cls, include_extras=True)

        for klass in cls.__mro__:
            if klass is object:
                continue

            ann = klass.__dict__.get("__annotations__", {})
            for fname, raw_type in ann.items():
                if fname.startswith("_") or fname in field_types:
                    continue
                t = hints.get(fname, raw_type)
                if get_origin(t) is Annotated:
                    args = get_args(t)
                    if "cpu_only" in args[1:]:
                        continue
                    t = args[0]
                if isinstance(t, str):
                    t = ANNOTATION_NAMES.get(t, t)
                if not _is_gpu_compatible_type(t):
                    continue
                field_types[fname] = t

            for fname, val in klass.__dict__.items():
                if fname.startswith("_") or not isinstance(val, property):
                    continue
                if fname in field_types or ContextBuilder._is_generator_property(val):
                    continue
                getter = val.fget
                if not getter:
                    continue
                try:
                    prop_hints = get_type_hints(getter, include_extras=True)
                    ret_type = prop_hints.get("return")
                except Exception:
                    raw_ann = getattr(getter, "__annotations__", {})
                    ret_type = raw_ann.get("return")
                if ret_type is None:
                    continue
                if get_origin(ret_type) is Annotated:
                    args = get_args(ret_type)
                    if "cpu_only" in args[1:]:
                        continue
                    ret_type = args[0]
                if isinstance(ret_type, str):
                    ret_type = ANNOTATION_NAMES.get(ret_type, ret_type)
                if not _is_gpu_compatible_type(ret_type):
                    continue
                field_types[fname] = ret_type

        for base_cls, sys_fields in SYSTEM_FIELDS.items():
            if issubclass(cls, base_cls):
                for fname in sys_fields:
                    if fname == "pos":
                        field_types[fname] = vec3
                    elif fname == "coord":
                        field_types[fname] = ivec3

        ctx.class_field_types[cls] = field_types

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

    @staticmethod
    def _classify_methods(ctx: ValidationContext) -> None:
        if ctx.tree is None:
            return
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            mangled = mangle_sim(node.name)
            if ContextBuilder._has_rule_decorator(node):
                ctx.rule_method_names.add(mangled)
            else:
                ctx.helper_method_names.add(mangled)

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

    @staticmethod
    def _resolve_hints(func: Any) -> Tuple[Dict[str, Any], Optional[Any]]:
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = getattr(func, "__annotations__", {}).copy()
        return_type = hints.pop("return", None)
        if return_type is type(None):
            return_type = None
        if return_type is not None:
            tt = resolve_tuple_annotation(return_type)
            if tt is not None:
                return_type = tt
        return hints, return_type
