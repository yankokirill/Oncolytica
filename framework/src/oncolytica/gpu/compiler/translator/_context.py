import ast

from oncolytica.gpu.compiler._type_system import py_type_to_wgsl
from oncolytica.core.validation._context import ValidationContext

from typing import Any

# WGSL struct names for built-in DSL agent types.
# These are code-gen aliases that live outside py_type_to_wgsl.
_RULE_TYPE_TO_MOCK_STRUCT: dict[str, str] = {
    "cell":      "MockCell",
    "tissue":    "MockTissue",
    "chemistry": "MockChemistry",
}

# self.<method> calls that are DSL builtins bypassing the TypeChecker.
# TypeChecker doesn't know their return types, so we declare them here.
_SELF_BUILTIN_RETURN: dict[str, str] = {
    "tissue_at":    "Tissue",
    "chemistry_at": "Chemistry",
}

# Index variable name used in the compute kernel for each rule type.
_RULE_TYPE_INDEX: dict[str, str] = {
    "cell":      "cell_index",
    "tissue":    "tissue_index",
    "chemistry": "chem_index",
}

# Out-buffer write statement template for each rule type that owns an agent.
# {param} — the main parameter name; {index} — the kernel index variable.
_RULE_TYPE_OUT_WRITE: dict[str, str] = {
    "cell":      "Cells_Out[{index}] = {param};",
    "tissue":    "Tissue_Out[{index}] = {param};",
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
    ) -> None:
        self.val_ctx = val_ctx
        self.rule_type = rule_type
        self.main_param = main_param
        self.metrics_param = metrics_param
        self.method_name = method_name
        self.uniforms = uniforms or {}

        # True  → this context is translating a rule kernel body (has _rng local,
        #          must write Out-buffer + rng_state before every return).
        # False → translating a helper function (receives rng_state as ptr param).
        self.is_rule: bool = is_rule

        self.globals_dict: dict[str, Any] = dict(val_ctx.constants)
        self.extracted_constants: dict[str, tuple[str, str]] = {}

        # ── Process global constants ─────────────────────────────────────────
        # Booleans must be treated as i32 (0/1) for WGSL buffer compatibility.
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
            prefix = base.__name__ if base else main_class.__name__.lower()
            mangled_name = f"{prefix}_{method_name}"
        else:
            mangled_name = f"sim_{method_name}"

        self.method_params: dict[str, str] = {}
        for pname, ptype in val_ctx.method_type_hints.get(mangled_name, {}).items():
            self.method_params[pname] = py_type_to_wgsl(ptype)

        self.local_vars: dict[str, str] = {}
        for var_name, py_type in val_ctx.symbol_table.get(mangled_name, {}).items():
            if var_name in self.method_params:
                continue
            self.local_vars[var_name] = py_type_to_wgsl(py_type)

        self.let_vars: set[str] = set()

        # ── Pointer parameters ───────────────────────────────────────────────
        # Names of function parameters that are passed as ptr<function, T> in
        # WGSL (i.e. whose position appears in method_mutating_params for this
        # method).  These must be dereferenced as (*param) when their fields are
        # read/written, and passed bare (already a pointer) — not &param — when
        # forwarded to another mutating-parameter position.
        mutating_positions: set[int] = val_ctx.method_mutating_params.get(mangled_name, set())
        _param_names = list(val_ctx.method_type_hints.get(mangled_name, {}).keys())
        self.ptr_params: set[str] = set()
        # ptr_params only applies to helper functions (is_rule=False).
        # In rule kernels the main agent is a plain `var` local, not a pointer —
        # dereferencing it would be a WGSL type error.
        if not is_rule:
            for pos in mutating_positions:
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
            wgsl_name = val_ctx.get_base_class(mem_class).__name__  # e.g., "Cell"
            self.struct_hints[wgsl_name] = {}
            for fname, ftype in fields.items():
                wt = py_type_to_wgsl(ftype)
                # Rule: Boolean fields in structs are physically stored as i32 in WGSL.
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

    # ── RNG / epilogue helpers ────────────────────────────────────────────────

    @property
    def rng_index(self) -> str:
        """Kernel index variable for the current rule type (e.g. "cell_index")."""
        return _RULE_TYPE_INDEX.get(self.rule_type, "cell_index")

    def rule_epilogue_lines(self) -> list[str]:
        """Return the WGSL lines that must be emitted before every ``return``
        in a rule kernel: save rng_state into the agent, then write Out-buffer.

        For rule_type values that have no dedicated Out-buffer (e.g. metric rules)
        only the rng line is returned.
        """
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
        """Single source of truth for expression types during code generation.

        Resolution order:
        1. ``val_ctx.type_map`` — populated by TypeChecker.
        2. DSL builtins (``self.tissue_at`` etc.).
        3. Agent constructors (``MyCell()`` → main_wgsl_struct).
        4. Fallback: "i32".
        """
        # ── 1. Validated type map (primary) ──────────────────────────────────
        wgsl = self.get_wgsl_type_of_node(node)
        if wgsl is not None:
            # Enforce i32 for any attribute access inferred as bool by the TypeChecker.
            if isinstance(node, ast.Attribute) and wgsl == "bool":
                return "i32"
            return wgsl

        # ── 2. DSL builtin calls: self.tissue_at / self.chemistry_at ─────────
        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "self"):
                builtin = _SELF_BUILTIN_RETURN.get(fn.attr)
                if builtin is not None:
                    return builtin

        # ── 3. Agent constructor: MyCell() → main_wgsl_struct ────────────────
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and self.main_class is not None
                and node.func.id == self.main_class):
            if self.main_wgsl_struct:
                return self.main_wgsl_struct

        # ── 4. Fallback ───────────────────────────────────────────────────────
        return "i32"

    def get_wgsl_type_of_node(self, node: ast.AST) -> str | None:
        """Convert a type_map entry to a WGSL type string, or None if absent."""
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
                 if vname not in self.let_vars]

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
