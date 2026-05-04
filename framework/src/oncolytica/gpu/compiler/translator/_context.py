import ast

from oncolytica.gpu.compiler._type_system import py_type_to_wgsl
from oncolytica.core.validation._context import ValidationContext
from oncolytica.core.utils._math import _INTRINSIC_FUNCTIONS

from typing import Any

# WGSL struct names for built-in DSL agent types.
_RULE_TYPE_TO_MOCK_STRUCT: dict[str, str] = {
    "cell": "MockCell",
    "tissue": "MockTissue",
    "chemistry": "MockChemistry",
}

# self.<method> calls that are DSL builtins bypassing the TypeChecker.
_SELF_BUILTIN_RETURN: dict[str, str] = {
    "tissue_at": "Tissue",
    "chemistry_at": "Chemistry",
}

# Index variable name used in the compute kernel for each rule type.
_RULE_TYPE_INDEX: dict[str, str] = {
    "cell": "cell_index",
    "tissue": "tissue_index",
    "chemistry": "chem_index",
}

# Out-buffer write statement template for each rule type that owns an agent.
_RULE_TYPE_OUT_WRITE: dict[str, str] = {
    "cell": "Cells_Out[{index}] = {param};",
    "tissue": "Tissue_Out[{index}] = {param};",
    "chemistry": "Chemistry_Out[{index}] = {param};",
}


class TranslationContext:
    def __init__(
            self,
            val_ctx: ValidationContext,
            rule_type: str,
            main_param: str,
            main_class: type,
            method_name: str,
            *,
            is_rule: bool = True,
            metrics_param: str | None = None,
            uniforms: dict[str, str] | None = None,
            self_is_ptr: bool = False,
    ) -> None:
        self.val_ctx = val_ctx
        self.rule_type = rule_type
        self.main_param = main_param
        self.metrics_param = metrics_param
        self.method_name = method_name
        self.uniforms = uniforms or {}
        self.main_class_type: type | None = main_class

        # True  → rule kernel body (has _rng local, must write Out-buffer + rng_state before every return).
        # False → helper function (receives rng_state as ptr param).
        self.is_rule: bool = is_rule

        # True  → domain-method helper where 'self' is passed as ptr<function, T>.
        # False → sim-method helper or value-receiver domain method.
        self.self_is_ptr: bool = self_is_ptr

        self.globals_dict: dict[str, Any] = dict(val_ctx.constants)
        self.extracted_constants: dict[str, tuple[str, str]] = {}

        # ── Process global constants ─────────────────────────────────────────
        for name, val in self.globals_dict.items():
            if isinstance(val, bool):
                self.extracted_constants[name] = ("i32", "1" if val else "0")
            elif isinstance(val, int):
                self.extracted_constants[name] = ("i32", str(val))
            elif isinstance(val, float):
                self.extracted_constants[name] = ("f32", str(val))

        if not self.is_rule and main_class is not None:
            from oncolytica.gpu.compiler._type_system import domain_base_of
            base = domain_base_of(main_class)
            prefix = base.__name__.lower() if base else main_class.__name__.lower()
            mangled_name = f"{prefix}_{method_name}"
        else:
            mangled_name = f"sim_{method_name}"

        self.method_params: dict[str, str] = {}
        for pname, ptype in val_ctx.method_type_hints.get(mangled_name, {}).items():
            if pname == "self":
                # 'self' in domain-method hints → exposed as '_self' in WGSL
                self.method_params["_self"] = py_type_to_wgsl(ptype)
            else:
                self.method_params[pname] = py_type_to_wgsl(ptype)

        self.local_vars: dict[str, str] = {}
        for var_name, py_type in val_ctx.symbol_table.get(mangled_name, {}).items():
            if var_name in self.method_params:
                continue
            self.local_vars[var_name] = py_type_to_wgsl(py_type)

        self.let_vars: set[str] = set()

        # ── Pointer parameters ───────────────────────────────────────────────
        mutating_positions: set[int] = val_ctx.method_mutating_params.get(mangled_name, set())
        # _param_names excludes 'self' (it's stored separately in method_type_hints for domain methods)
        _param_names = [
            pname for pname in val_ctx.method_type_hints.get(mangled_name, {}).keys()
            if pname != "self"
        ]
        self.ptr_params: set[str] = set()
        if not is_rule:
            for pos in mutating_positions:
                if pos == 0:
                    # Position 0 = self → _self in WGSL
                    if self_is_ptr:
                        self.ptr_params.add("_self")
                    continue
                # pos is 1-based (0 = self); _param_names is 0-indexed excluding self
                idx = pos - 1
                if 0 <= idx < len(_param_names):
                    self.ptr_params.add(_param_names[idx])

        self.main_class: str | None = main_class.__name__ if main_class is not None else None
        self.main_wgsl_struct: str | None = None
        if main_class is not None:
            from oncolytica.gpu.compiler._type_system import domain_base_of
            base = domain_base_of(main_class)
            if base is not None:
                self.main_wgsl_struct = base.__name__

        self.struct_hints: dict[str, dict[str, str]] = {}
        for mem_class, fields in val_ctx.class_field_types.items():
            wgsl_name = val_ctx.get_base_class(mem_class).__name__
            self.struct_hints[wgsl_name] = {}
            for fname, ftype in fields.items():
                wt = py_type_to_wgsl(ftype)
                if wt == "bool":
                    wt = "i32"
                self.struct_hints[wgsl_name][fname] = wt

        self.metrics_hints: dict[str, str] = self.struct_hints.get("Metrics", {})
        self.main_hints: dict[str, str] = {}
        if main_class is not None:
            from oncolytica.gpu.compiler._type_system import domain_base_of
            base = domain_base_of(main_class)
            if base is not None:
                self.main_hints = self.struct_hints.get(base.__name__, {})

        self._lines: list[str] = []
        self._indent: int = 0
        self.source_map: dict[int, int] = {}
        self._tmp_counter = 0
        self._source_map_shifted = False
        self.tuple_aliases: dict[str, str] = {}

    # ── RNG / epilogue helpers ────────────────────────────────────────────────

    @property
    def rng_index(self) -> str:
        return _RULE_TYPE_INDEX.get(self.rule_type, "cell_index")

    def rule_epilogue_lines(self) -> list[str]:
        param = self.main_param
        index = self.rng_index
        lines: list[str] = [
            f"{param}._rng_state = _rng;",
        ]
        out_tmpl = _RULE_TYPE_OUT_WRITE.get(self.rule_type)
        if out_tmpl:
            lines.append(out_tmpl.format(param=param, index=index))
        return lines

    # ── Primary type lookup ───────────────────────────────────────────────────

    def node_wgsl_type(self, node: ast.expr) -> str:
        wgsl = self.get_wgsl_type_of_node(node)
        if wgsl is not None:
            if isinstance(node, ast.Attribute) and wgsl == "bool":
                return "i32"
            return wgsl

        if isinstance(node, ast.Constant):
            from oncolytica.gpu.compiler._type_system import infer_literal_wgsl_type
            literal_type = infer_literal_wgsl_type(node)
            if literal_type is not None:
                return literal_type

        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "self"):
                builtin = _SELF_BUILTIN_RETURN.get(fn.attr)
                if builtin is not None:
                    return builtin

            func_name = None
            if isinstance(fn, ast.Attribute):
                func_name = fn.attr
            elif isinstance(fn, ast.Name):
                func_name = fn.id

            if func_name is not None:
                if func_name in _INTRINSIC_FUNCTIONS:
                    try:
                        wgsl_t = py_type_to_wgsl(_INTRINSIC_FUNCTIONS[func_name])
                        if wgsl_t == "bool":
                            return "i32"
                        return wgsl_t
                    except TypeError:
                        pass

        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and self.main_class is not None
                and node.func.id == self.main_class):
            if self.main_wgsl_struct:
                return self.main_wgsl_struct

        if isinstance(node, ast.BinOp):
            left_type = self.node_wgsl_type(node.left)
            right_type = self.node_wgsl_type(node.right)
            if "vec3" in left_type:
                return left_type
            if "vec3" in right_type:
                return right_type
            if left_type == "f32" or right_type == "f32":
                return "f32"

        if isinstance(node, ast.UnaryOp):
            return self.node_wgsl_type(node.operand)

        return "i32"

    def get_wgsl_type_of_node(self, node: ast.AST) -> str | None:
        py_type = self.val_ctx.type_map.get(id(node))
        if py_type is None:
            return None
        try:
            return py_type_to_wgsl(py_type)
        except TypeError:
            return None

    # ── Output helpers ────────────────────────────────────────────────────────

    def get_output(self) -> str:
        decls = [f"var {vname}: {vtype};" for vname, vtype in self.local_vars.items()
                 if vname not in self.tuple_aliases]

        if decls and not self._source_map_shifted:
            self.source_map = {k + len(decls): v for k, v in self.source_map.items()}
            self._source_map_shifted = True

        return "\n".join(decls + self._lines)

    def emit(self, code: str, py_line: int | None = None) -> None:
        line_no = len(self._lines) + 1
        if py_line is not None:
            self.source_map[line_no] = py_line
        indent = "    " * self._indent
        self._lines.append(indent + code)

    def indent(self) -> None:
        self._indent += 1

    def dedent(self) -> None:
        self._indent -= 1

    def fresh_tmp(self, prefix: str = "_t") -> str:
        name = f"{prefix}{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def is_metrics_attr(self, obj_name: str) -> bool:
        return self.metrics_param is not None and obj_name == self.metrics_param
