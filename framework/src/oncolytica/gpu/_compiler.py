# oncolytica/gpu/_compiler.py
from __future__ import annotations
import ast
import textwrap
import inspect
from typing import Any, Optional, get_type_hints

from ._shader_builder import ShaderBuilder
from ._codegen import TranslationContext, StmtTranslator
from ._ast_passes import apply_passes


def _rule_param_info(bound_method: Any) -> tuple[str, Optional[type]]:
    try:
        sig = inspect.signature(bound_method)
        params = list(sig.parameters.values())
        if not params: return "item", None
        first = params[0]
        try:
            import typing
            hints = typing.get_type_hints(bound_method)
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

    item_name = params[0].name
    metrics_name = params[1].name

    hints = get_type_hints(bound_method)
    item_cls = hints.get(item_name, params[0].annotation)
    metrics_cls = hints.get(metrics_name, params[1].annotation)

    return item_name, metrics_name, item_cls, metrics_cls

def _get_func_ast(func: Any) -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef): return node
    raise ValueError(f"No FunctionDef found in source of {func!r}")


class WGSLCompiler:

    def __init__(self, spec: Any, engine: Any, user_params: dict, serializers: dict):
        self.engine = engine
        self.user_params = user_params

        self.builder = ShaderBuilder(
            spec=spec,
            cell_packed=serializers["cell"].fields,
            tissue_packed=serializers["tissue"].fields,
            chem_packed=serializers["chem"].fields,
            metrics_packed=serializers["metric"].fields,
            user_params=user_params,
        )

    def compile(self) -> str:
        uniforms_map = {name: wtype for name, (wtype, _) in self.user_params.items()}

        def _compile_rule(rule_method: Any, rule_type: str, param_name: str, pcls: type):
            func = getattr(rule_method, "__func__", rule_method)
            func_def = apply_passes(_get_func_ast(func))
            ctx = TranslationContext(
                rule_type=rule_type, main_param=param_name, main_class=pcls,
                uniforms=uniforms_map, globals_dict=getattr(func, "__globals__", {})
            )
            translator = StmtTranslator(ctx)

            body = func_def.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)):
                body = body[1:]  # Пропуск docstring

            translator.translate_body(body)
            return ctx.get_output(), ctx.extracted_constants

        # Cells
        for idx, rule in enumerate(self.engine._cell_rules):
            pname, pcls = _rule_param_info(rule)
            body, constants = _compile_rule(rule, "cell", pname, pcls)
            self.builder.add_cell_kernel(f"Kernel_CellRule_{idx}", pname, body, constants)

        # Tissue
        for idx, rule in enumerate(self.engine._tissue_rules):
            pname, pcls = _rule_param_info(rule)
            body, constants = _compile_rule(rule, "tissue", pname, pcls)
            self.builder.add_tissue_kernel(f"Kernel_TissueRule_{idx}", pname, body, constants)

        # Chemistry
        for idx, rule in enumerate(self.engine._chemistry_rules):
            pname, pcls = _rule_param_info(rule)
            body, constants = _compile_rule(rule, "chemistry", pname, pcls)
            self.builder.add_chem_kernel(f"Kernel_ChemRule_{idx}", pname, body, constants)

        # Metrics
        for idx, rule in enumerate(self.engine._metric_rules):
            item_name, m_name, item_cls, m_cls = _metric_params_info(rule)

            ctx = TranslationContext(
                rule_type="metric",
                main_param=item_name,
                main_class=item_cls,
                metrics_param=m_name,
                metrics_class=m_cls,
                uniforms=uniforms_map,
                globals_dict=getattr(rule, "__globals__", {})
            )

            translator = StmtTranslator(ctx)
            func_def = apply_passes(_get_func_ast(rule))

            body = func_def.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]

            translator.translate_body(body)

            k_name = f"Kernel_MetricRule_{idx}"
            self.builder.add_metric_kernel(k_name, item_name, ctx.get_output(), ctx.extracted_constants)

        return self.builder.build(ShaderBuilder.find_template())
