# oncolytica/gpu/_compiler.py
from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
from typing import Any, Optional, get_type_hints

from ._shader_builder import ShaderBuilder
from ._type_system import py_type_to_wgsl

from oncolytica.gpu.compiler.translator._context import TranslationContext
from oncolytica.gpu.compiler.translator._stmt_translator import StmtTranslator
from oncolytica.core.utils._types import Cell, Tissue, Chemistry, Metrics, BASE_CLASSES, PRIMITIVE_TYPES
from ._type_system import domain_base_of

# ── AST / introspection helpers ───────────────────────────────────────────────

def _get_func_ast(func: Any) -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError(f"No FunctionDef found in source of {func!r}")


def _rule_param_info(bound_method: Any) -> tuple[str, Optional[type]]:
    try:
        sig = inspect.signature(bound_method)
        params = list(sig.parameters.values())
        if not params:
            return "item", None
        first = params[0]
        try:
            hints = get_type_hints(bound_method)
            ann = hints.get(first.name, first.annotation)
        except Exception:
            ann = first.annotation
        cls = ann if isinstance(ann, type) else None
        return first.name, cls
    except (ValueError, TypeError):
        return "item", None


def _metric_params_info(bound_method: Any) -> tuple[str, str, type, type]:
    sig = inspect.signature(bound_method)
    params = list(sig.parameters.values())
    item_name    = params[0].name
    metrics_name = params[1].name
    hints = get_type_hints(bound_method)
    item_cls    = hints.get(item_name,    params[0].annotation)
    metrics_cls = hints.get(metrics_name, params[1].annotation)
    return item_name, metrics_name, item_cls, metrics_cls


# ── Helper-function compilation ───────────────────────────────────────────────

def _compile_helper_fn(
        method: Any,
        uniforms_map: dict,
        val_ctx: Any,
        *,
        wgsl_fn_name: str,          # final WGSL function name, e.g. "_gamma_infected" or "Cell_update"
        main_param: str = "self",
        main_class: Optional[type] = None,
) -> tuple[str, dict[str, tuple[str, str]]]:
    """Compile a non-rule method into a standalone WGSL function."""
    hints = get_type_hints(method)
    sig   = inspect.signature(method)

    # Determine which parameter positions are mutated (from DomainValidator).
    mutating_positions: set[int] = val_ctx.method_mutating_params.get(wgsl_fn_name, set())

    # Enumerate explicit parameters (skip 'self'), tracking 1-based position.
    param_items = [
        (pos + 1, pname, hints.get(pname, param.annotation))
        for pos, (pname, param) in enumerate(sig.parameters.items())
    ]

    # WGSL parameter list:
    # - mutating domain params  → ptr<function, StructName>
    # - mutating primitive params → "_pname: T" in signature, "var pname = _pname" in body
    # - everything else          → plain "pname: T"
    wgsl_params: list[str] = []
    mutable_primitive_params: list[tuple[str, str]] = []   # (pname, wgsl_type)
    for pos, pname, ann in param_items:
        if ann is inspect.Parameter.empty:
            continue
        try:
            wgsl_type = py_type_to_wgsl(ann)
        except TypeError:
            continue
        is_domain   = domain_base_of(ann) is not None
        is_primitive = ann in PRIMITIVE_TYPES
        is_mutating  = pos in mutating_positions
        if is_domain and is_mutating:
            # Passed by pointer so the function can mutate the caller's copy.
            wgsl_params.append(f"{pname}: ptr<function, {wgsl_type}>")
        elif is_primitive and is_mutating:
            # Declare as private "_pname" in signature; body will shadow with var.
            wgsl_params.append(f"_{pname}: {wgsl_type}")
            mutable_primitive_params.append((pname, wgsl_type))
        else:
            wgsl_params.append(f"{pname}: {wgsl_type}")

    # Helpers receive rng state as a pointer instead of owning a local _rng.
    wgsl_params.append("_rng_state: ptr<function, u32>")

    # Return type
    ret_ann  = hints.get("return", inspect.Parameter.empty)
    ret_wgsl = ""
    if ret_ann is not inspect.Parameter.empty:
        try:
            ret_wgsl = f" -> {py_type_to_wgsl(ret_ann)}"
        except TypeError:
            pass

    func     = getattr(method, "__func__", method)
    func_def = _get_func_ast(func)

    ctx = TranslationContext(
        val_ctx=val_ctx,
        rule_type="helper",
        main_param=main_param,
        main_class=main_class,
        method_name=func_def.name,
        uniforms=uniforms_map,
        is_rule=False,
    )
    StmtTranslator(ctx).translate_body(_skip_docstring(func_def.body))

    # Build prologue: "var d = _d;" for each mutable primitive param.
    prologue_lines = [f"    var {pname}: {wtype} = _{pname};" for pname, wtype in mutable_primitive_params]
    prologue       = ("\n".join(prologue_lines) + "\n") if prologue_lines else ""

    compiled_body = textwrap.indent(ctx.get_output(), "    ")
    params_str    = ", ".join(wgsl_params)
    wgsl_fn_str   = (
        f"fn {wgsl_fn_name}({params_str}){ret_wgsl} {{\n"
        f"{prologue}"
        f"{compiled_body}\n"
        f"}}"
    )
    return wgsl_fn_str, ctx.extracted_constants


# ── Internal utils ────────────────────────────────────────────────────────────

def _skip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[1:]
    return body


# ── Compiler ──────────────────────────────────────────────────────────────────

class WGSLCompiler:

    def __init__(self, ctx: Any, spec: Any, engine: Any, user_params: dict, serializers: dict):
        self.engine      = engine
        self.user_params = user_params
        self.val_ctx     = ctx
        self._spec       = spec

        self.builder = ShaderBuilder(
            spec=spec,
            cell_packed=serializers["cell"].fields,
            tissue_packed=serializers["tissue"].fields,
            chem_packed=serializers["chem"].fields,
            metrics_packed=serializers["metric"].fields,
            user_params=user_params,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def compile(self) -> str:
        uniforms_map  = {name: wtype for name, (wtype, _) in self.user_params.items()}
        all_constants: dict[str, tuple[str, str]] = {}

        sim = self.engine._sim

        # ── Simulation-level helpers  (_method_name) ─────────────────────────
        for name in dir(sim):
            if name.startswith("_"):
                continue
            method = getattr(type(sim), name, None)
            if method is None or not callable(method) or not inspect.isfunction(method):
                continue
            if getattr(getattr(sim, name, None), "_rule_type", None) is not None:
                continue
            if name in ("chemistry_at", "tissue_at", "copy", "copy_from"):
                continue
            try:
                wgsl_fn, constants = _compile_helper_fn(
                    getattr(sim, name),
                    uniforms_map=uniforms_map,
                    val_ctx=self.val_ctx,
                    wgsl_fn_name=f"sim_{name}",
                )
                all_constants.update(constants)
                self.builder.add_helper_fn(wgsl_fn)
            except Exception as exc:
                warnings.warn(f"Could not compile Simulation helper '{name}': {exc}")

        # ── Domain-class helpers  (Cell_method, Tissue_method, …) ────────────
        domain_classes = [
            self._spec.cell_class,
            self._spec.tissue_class,
            self._spec.chemistry_class,
            self._spec.metrics_class,
        ]
        for cls in domain_classes:
            base = domain_base_of(cls)
            if base is None:
                continue
            prefix = base.__name__

            for name, val in inspect.getmembers(cls, predicate=inspect.isfunction):
                if name.startswith("_"):
                    continue
                if name in ("copy", "copy_from"):
                    continue
                # skip methods inherited from framework base classes
                if any(name in vars(b) for b in BASE_CLASSES if b is not object):
                    continue
                try:
                    wgsl_fn, constants = _compile_helper_fn(
                        val,
                        uniforms_map=uniforms_map,
                        val_ctx=self.val_ctx,
                        wgsl_fn_name=f"{prefix}_{name}",
                        main_param="self",
                        main_class=cls,
                    )
                    all_constants.update(constants)
                    self.builder.add_helper_fn(wgsl_fn)
                except Exception as exc:
                    warnings.warn(f"Could not compile {prefix} helper '{name}': {exc}")

        # ── Cell rules ────────────────────────────────────────────────────────
        for idx, rule in enumerate(self.engine._cell_rules):
            pname, pcls = _rule_param_info(rule)
            body, constants = self._compile_rule(rule, "cell", pname, pcls, uniforms_map)
            all_constants.update(constants)
            self.builder.add_cell_kernel(f"Kernel_CellRule_{idx}", pname, body)

        # ── Tissue rules ──────────────────────────────────────────────────────
        for idx, rule in enumerate(self.engine._tissue_rules):
            pname, pcls = _rule_param_info(rule)
            body, constants = self._compile_rule(rule, "tissue", pname, pcls, uniforms_map)
            all_constants.update(constants)
            self.builder.add_tissue_kernel(f"Kernel_TissueRule_{idx}", pname, body)

        # ── Chemistry rules ───────────────────────────────────────────────────
        for idx, rule in enumerate(self.engine._chemistry_rules):
            pname, pcls = _rule_param_info(rule)
            body, constants = self._compile_rule(rule, "chemistry", pname, pcls, uniforms_map)
            all_constants.update(constants)
            self.builder.add_chem_kernel(f"Kernel_ChemRule_{idx}", pname, body)

        # ── Metric rules ──────────────────────────────────────────────────────
        for idx, rule in enumerate(self.engine._metric_rules):
            body, constants, k_name, item_name = self._compile_metric_rule(
                rule, idx, uniforms_map
            )
            all_constants.update(constants)
            self.builder.add_metric_kernel(k_name, item_name, body)

        self.builder.add_all_constants(all_constants)
        return self.builder.build(ShaderBuilder.find_template())

    # ── Private compile helpers ───────────────────────────────────────────────

    def _compile_rule(
            self,
            rule_method: Any,
            rule_type: str,
            param_name: str,
            pcls: type,
            uniforms_map: dict,
    ) -> tuple[str, dict[str, tuple[str, str]]]:
        func     = getattr(rule_method, "__func__", rule_method)
        func_def = _get_func_ast(func)

        ctx = TranslationContext(
            val_ctx=self.val_ctx,
            rule_type=rule_type,
            main_param=param_name,
            main_class=pcls,
            method_name=func_def.name,
            uniforms=uniforms_map,
        )
        StmtTranslator(ctx).translate_body(_skip_docstring(func_def.body))
        return ctx.get_output(), ctx.extracted_constants

    def _compile_metric_rule(
            self,
            rule: Any,
            idx: int,
            uniforms_map: dict,
    ) -> tuple[str, dict[str, tuple[str, str]], str, str]:
        item_name, m_name, item_cls, _m_cls = _metric_params_info(rule)

        func     = getattr(rule, "__func__", rule)
        func_def = _get_func_ast(func)

        ctx = TranslationContext(
            val_ctx=self.val_ctx,
            rule_type="metric",
            main_param=item_name,
            main_class=item_cls,
            method_name=func_def.name,
            metrics_param=m_name,
            uniforms=uniforms_map,
        )
        StmtTranslator(ctx).translate_body(_skip_docstring(func_def.body))

        k_name = f"Kernel_MetricRule_{idx}"
        return ctx.get_output(), ctx.extracted_constants, k_name, item_name
