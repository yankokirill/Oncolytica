from __future__ import annotations

import ast
import inspect
import textwrap
from typing import get_origin, get_type_hints, get_args
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Generator, Annotated

from oncolytica.core.utils._types import (
    BASE_CLASSES, ANNOTATION_NAMES, ivec3,
    i32, u32, f32,
    bool as ol_bool,
    vec3,
    Tissue, Chemistry, Cell, Metrics,
    PRIMITIVE_TYPES,
)
from oncolytica.core.utils._errors import CompilationError


RULE_DECORATOR_NAMES: FrozenSet[str] = frozenset({
    "cell_rule", "tissue_rule", "chemistry_rule", "metric_rule",
})

SYSTEM_FIELDS: Dict[type, FrozenSet[str]] = {
    Cell: frozenset({"pos"}),
}

# Maps framework base class → canonical name prefix used in mangling.
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


# ── Public mangling helpers ────────────────────────────────────────────────────

def mangle_sim(method_name: str) -> str:
    """Sim-class method: 'spawn' → 'sim_spawn'."""
    return f"sim_{method_name}"


def mangle_domain(base_cls: type, method_name: str) -> str:
    """Domain-class method: Cell, 'update' → 'cell_update'."""
    prefix = _BASE_PREFIX.get(base_cls, base_cls.__name__.lower())
    return f"{prefix}_{method_name}"


# ── Type-name utility (single authoritative definition) ───────────────────────

def type_name(t: Any) -> str:
    return getattr(t, "__name__", repr(t))


# =============================================================================
# VALIDATION CONTEXT
# =============================================================================

class ValidationContext:
    """
    Blackboard shared by all pipeline phases.

    Naming convention
    -----------------
    All keys in method_* dicts use **mangled names**:
      • sim-class methods  → "sim_{name}"   (e.g. "sim_spawn")
      • domain-class methods → "{prefix}_{name}"  (e.g. "cell_update")

    Use ``mangle_sim`` / ``mangle_domain`` helpers (module-level) or the
    instance method ``mangle_name`` to produce keys consistently.
    """

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

        # Keys are mangled names for sim-methods; raw names for domain-methods
        # are additionally mangled via mangle_domain before insertion.
        self.method_signatures:   Dict[str, inspect.Signature] = {}
        self.method_type_hints:   Dict[str, Dict[str, Any]] = {}   # includes "self" at index 0 for domain methods
        self.method_return_hints: Dict[str, Optional[Any]] = {}

        # Class-level metadata for domain classes (Cell, Tissue, …)
        self.class_fields:        Dict[type, Set[str]] = {}
        self.class_properties:    Dict[type, Set[str]] = {}
        self.class_methods:       Dict[type, Set[str]] = {}          # raw method names per class
        self.class_generators:    Dict[type, Set[str]] = {}
        self.class_field_types:   Dict[type, Dict[str, Any]] = {}
        self.constants:           Dict[str, Any] = {}

        # AST nodes — keyed by mangled name for sim-methods;
        # domain-method nodes are NOT stored here (no source available from
        # the sim class AST).
        self.method_nodes:        Dict[str, ast.FunctionDef] = {}

        # Mangled names
        self.rule_method_names:   Set[str] = set()
        self.helper_method_names: Set[str] = set()

        self.symbol_table:    Dict[str, Dict[str, Optional[Any]]] = {}
        self.type_map:        Dict[int, Any] = {}
        self.call_graph:      Dict[str, Set[str]] = {}
        self.ordered_methods: List[str] = []

        # Source location for error reporting — keyed by mangled name.
        self.method_locations: Dict[str, Tuple[str, int]] = {}

        # Mutating parameter positions — filled by DomainValidator.
        # Maps mangled method name → set of argument indices (0 = self / first ptr).
        self.method_mutating_params: Dict[str, Set[int]] = {}

    # ── Mangling ───────────────────────────────────────────────────────────────

    def mangle_name(self, cls: Optional[type], method_name: str) -> str:
        """
        Produce a mangled method name.

        ``cls=None``          → sim-method  ("sim_{method_name}")
        ``cls=<domain cls>``  → domain method ("{prefix}_{method_name}")
        """
        if cls is None:
            return mangle_sim(method_name)
        base = self.memory_base_map.get(cls)
        if base is None:
            raise ValueError(f"'{type_name(cls)}' is not a registered domain class.")
        return mangle_domain(base, method_name)

    # ── Attribute helpers ──────────────────────────────────────────────────────

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
        ContextBuilder._validate_memory_base_map(ctx.memory_base_map)
        ContextBuilder._build_ast(ctx, sim_class)
        ContextBuilder._build_sim_signatures(ctx, sim_class)
        ContextBuilder._build_class_metadata(ctx)
        ContextBuilder._build_domain_method_signatures(ctx)
        ContextBuilder._build_constants(ctx)
        ContextBuilder._classify_methods(ctx)
        return ctx

    # ── Memory map validation ──────────────────────────────────────────────────

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
                    f"Memory class '{cls.__name__}' inherits from multiple framework base classes. "
                    f"Multiple inheritance is not supported."
                )
            if actual_bases[0] is not expected_base:
                raise CompilationError(
                    f"Memory class '{cls.__name__}' inherits from '{actual_bases[0].__name__}' "
                    f"but map expects '{expected_base.__name__}'"
                )

    # ── AST construction ───────────────────────────────────────────────────────

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

        # Annotate every AST node with its parent for error reporting.
        ctx.tree.parent = None  # type: ignore[attr-defined]
        for node in ast.walk(ctx.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore[attr-defined]

        # Store FunctionDef nodes keyed by mangled sim name.
        seen_raw: Dict[str, int] = {}
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in seen_raw:
                    raise CompilationError(
                        f"Line {node.lineno}: function '{node.name}' is already defined."
                    )
                seen_raw[node.name] = node.lineno
                ctx.method_nodes[mangle_sim(node.name)] = node

    # ── Sim-class method signatures ────────────────────────────────────────────

    @staticmethod
    def _build_sim_signatures(ctx: ValidationContext, sim_class: type) -> None:
        """
        Build method_type_hints / method_return_hints for all sim-class methods.
        Keys are mangled sim names ("sim_spawn", …).
        The "self" parameter is intentionally excluded from type_hints for sim
        methods (it is the simulation object, not a domain pointer).
        """
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

            # AST fallback — guarantees we read exactly what the user wrote.
            clean_hints: Dict[str, Any] = {}
            for arg in node.args.args:
                arg_name = arg.arg
                if arg_name == "self":
                    continue  # exclude self for sim methods
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

    # ── Domain-class method signatures ────────────────────────────────────────

    @staticmethod
    def _build_domain_method_signatures(ctx: ValidationContext) -> None:
        """
        Build method_type_hints / method_return_hints for methods defined on
        domain classes (Cell, Tissue, Chemistry, Metrics).

        Keys are mangled domain names ("cell_update", …).
        "self" IS included at position 0 with the domain class as its type,
        mirroring the function-with-pointer convention.
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
                except (ValueError, TypeError):
                    pass

                try:
                    filename = inspect.getsourcefile(method) or "<unknown>"
                    _, start_line = inspect.getsourcelines(method)
                    ctx.method_locations[mangled] = (filename, start_line)
                except (OSError, TypeError):
                    ctx.method_locations[mangled] = ("<unknown>", 1)

                # Build ordered param dict: self first, then the rest.
                clean_hints: Dict[str, Any] = {}

                # Position 0: self — typed as the user domain class (pointer).
                clean_hints["self"] = user_cls

                try:
                    sig = inspect.signature(method)
                    for pname, param in sig.parameters.items():
                        if pname == "self":
                            continue
                        if pname in param_hints:
                            clean_hints[pname] = param_hints[pname]
                        elif param.annotation is not inspect.Parameter.empty:
                            # param.annotation may already be a resolved type
                            ann = param.annotation
                            if isinstance(ann, str):
                                ann = ANNOTATION_NAMES.get(ann)
                            if ann is not None:
                                clean_hints[pname] = ann
                except (ValueError, TypeError):
                    pass

                ctx.method_type_hints[mangled]   = clean_hints
                ctx.method_return_hints[mangled] = return_hint

    # ── Class metadata ─────────────────────────────────────────────────────────

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

    # ── Constants ──────────────────────────────────────────────────────────────

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

    # ── Method classification ──────────────────────────────────────────────────

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

    # ── Generic hint resolver ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_hints(func: Any) -> Tuple[Dict[str, Any], Optional[Any]]:
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = getattr(func, "__annotations__", {}).copy()
        return_type = hints.pop("return", None)
        return hints, return_type
