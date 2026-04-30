from __future__ import annotations

import ast
from typing import Any

from oncolytica.core._errors import CompilationError
from ._type_system import (
    py_type_to_wgsl,
    infer_literal_wgsl_type,
    format_float_literal,
    format_int_literal,
)
from oncolytica.core._types import _resolve_own_hints

# (Далее весь код без изменений, начиная с _BINOP_MAP...)
_BINOP_MAP: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.Pow: "**",
}

_CMP_MAP: dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_AUGOP_MAP: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

_MATH_MAP: dict[str, str] = {
    "length": "length",
    "length_sq": None,
    "distance": "distance",
    "distance_sq": None,
    "normalize": "normalize",
    "dot": "dot",
    "cross": "cross",
    "reflect": "reflect",
    "lerp_vec": "mix",
    "lerp": "mix",
    "clamp": "clamp",
    "smoothstep": "smoothstep",
    "sign": "sign",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log2": "log2",
    "pow": "pow",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan2": "atan2",
    "floor": "floor",
    "ceil": "ceil",
    "fabs": "abs",
    "abs": "abs",
}


class TranslationContext:
    def __init__(
            self,
            rule_type: str,
            main_param: str,
            main_class: type | None,
            *,
            metrics_param: str | None = None,
            metrics_class: type | None = None,
            uniforms: dict[str, str] | None = None,
            globals_dict: dict[str, Any] | None = None,
    ) -> None:
        self.rule_type = rule_type
        self.main_param = main_param
        self.main_class = main_class
        self.metrics_param = metrics_param
        self.metrics_class = metrics_class
        self.uniforms: dict[str, str] = uniforms or {}
        self.globals_dict: dict[str, Any] = globals_dict or {}
        self.extracted_constants: dict[str, tuple[str, str]] = {}
        self.main_hints: dict[str, str] = {}
        self.metrics_hints: dict[str, str] = {}

        if main_class is not None:
            raw = _resolve_own_hints(main_class)
            for base in reversed(main_class.__mro__):
                if base is object: continue
                for fname, ftype in _resolve_own_hints(base).items():
                    if not fname.startswith("_"):
                        try:
                            self.main_hints[fname] = py_type_to_wgsl(ftype)
                        except TypeError:
                            pass

        if metrics_class is not None:
            for fname, ftype in _resolve_own_hints(metrics_class).items():
                if not fname.startswith("_"):
                    try:
                        self.metrics_hints[fname] = py_type_to_wgsl(ftype)
                    except TypeError:
                        pass

        self.local_vars: dict[str, str] = {}
        self.let_vars: set[str] = set()   # vars emitted as `let`; excluded from var-decl block
        self._lines: list[str] = []
        self._indent: int = 0
        self.source_map: dict[int, int] = {}
        self._tmp_counter = 0
        self._source_map_shifted = False

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

    def get_output(self) -> str:
        decls = [f"var {vname}: {vtype};" for vname, vtype in self.local_vars.items()
                 if vname not in self.let_vars]
        if decls and not self._source_map_shifted:
            self.source_map = {k + len(decls): v for k, v in self.source_map.items()}
            self._source_map_shifted = True
        return "\n".join(decls + self._lines)

    def field_type(self, obj_name: str, field_name: str) -> str | None:
        if obj_name == self.main_param: return self.main_hints.get(field_name)
        if obj_name == self.metrics_param: return self.metrics_hints.get(field_name)
        return None

    def is_metrics_attr(self, obj_name: str) -> bool:
        return self.metrics_param is not None and obj_name == self.metrics_param


class ExprTranslator(ast.NodeVisitor):
    def __init__(self, ctx: TranslationContext) -> None:
        self.ctx = ctx

    def translate(self, node: ast.expr) -> str:
        result = self.visit(node)
        if result is None:
            raise CompilationError(f"Cannot translate expression: {ast.dump(node)}",
                                   python_line=getattr(node, "lineno", None))
        return result

    def visit_Constant(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, bool): return "true" if v else "false"
        if isinstance(v, float): return format_float_literal(v)
        if isinstance(v, int): return str(v)
        raise CompilationError(f"Unsupported constant type {type(v).__name__!r}: {v!r}", python_line=node.lineno)

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name == "True": return "true"
        if name == "False": return "false"
        if name in self.ctx.local_vars: return name
        if name in self.ctx.globals_dict:
            val = self.ctx.globals_dict[name]
            if isinstance(val, int) and not isinstance(val, bool):
                self.ctx.extracted_constants[name] = ("i32", format_int_literal(val))
                return name
            elif isinstance(val, float):
                self.ctx.extracted_constants[name] = ("f32", format_float_literal(val))
                return name
        return name

    def visit_Attribute(self, node: ast.Attribute) -> str:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"U.{node.attr}"
        obj = self.visit(node.value)
        return f"{obj}.{node.attr}"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.translate(node.left)
        right = self.translate(node.right)
        if isinstance(node.op, ast.Pow): return f"pow({left}, {right})"
        op = _BINOP_MAP.get(type(node.op))
        if op is None: raise CompilationError(f"Unsupported binary operator {type(node.op).__name__}",
                                              python_line=getattr(node, "lineno", None))
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.translate(node.operand)
        if isinstance(node.op, ast.USub): return f"-{operand}"
        if isinstance(node.op, ast.UAdd): return operand
        if isinstance(node.op, ast.Not): return f"!({operand})"
        raise CompilationError(f"Unsupported unary op {type(node.op).__name__}")

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = "&&" if isinstance(node.op, ast.And) else "||"
        parts = [f"({self.translate(v)})" for v in node.values]
        return f" {op} ".join(parts)

    def visit_Compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise CompilationError("Chained comparisons are not supported in GPU rules",
                                   python_line=getattr(node, "lineno", None))
        left = self.translate(node.left)
        op = _CMP_MAP.get(type(node.ops[0]))
        right = self.translate(node.comparators[0])
        if op is None: raise CompilationError(f"Unsupported comparison {type(node.ops[0]).__name__}")
        return f"({left} {op} {right})"

    def visit_Call(self, node: ast.Call) -> str:
        fn = node.func

        # 1. ol.math.XXX
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) and
                isinstance(fn.value.value, ast.Name) and fn.value.value.id == "ol" and fn.value.attr == "math"):
            return self._translate_math_call(fn.attr, node)

        # 3. ol.random(agent) → _next_rand(&_rng)
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and
                fn.value.id == "ol" and fn.attr == "random"):
            agent = _get_name(node.args[0]) if node.args else self.ctx.main_param
            return f"_next_rand(&_rng)"

        # 4. ol.random_dir(agent) → _rand_dir(&_rng)
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and
                fn.value.id == "ol" and fn.attr == "random_dir"):
            agent = _get_name(node.args[0]) if node.args else self.ctx.main_param
            return f"_rand_dir(&_rng)"

        # 5. ol.vec3
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and
                fn.value.id == "ol" and fn.attr == "vec3"):
            args = [self.translate(a) for a in node.args]
            return f"vec3<f32>({', '.join(args)})"

        # 6. vec3 (direct import)
        if isinstance(fn, ast.Name) and fn.id == "vec3":
            args = [self.translate(a) for a in node.args]
            return f"vec3<f32>({', '.join(args)})"

        # 7. Constructor calls for the main Class (e.g. MyCell(pos=...))
        if isinstance(fn, ast.Name) and self.ctx.main_class and fn.id == self.ctx.main_class.__name__:
            struct_name = "Cell" if self.ctx.rule_type == "cell" else \
                "Tissue" if self.ctx.rule_type == "tissue" else \
                    "Chemistry" if self.ctx.rule_type == "chemistry" else fn.id

            if len(node.args) > 0 or len(node.keywords) > 0:
                raise CompilationError(
                    f"To create a new agent, use an empty constructor `{fn.id}()` and assign fields manually:\n"
                    f"  new_cell = {fn.id}()\n  new_cell.pos = ...",
                    lineno=getattr(node, "lineno", None)
                )

            return f"{struct_name}()"

        # 8. Helper methods
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
            method = fn.attr
            args = [self.translate(a) for a in node.args]
            kw_args = [f"{kw.arg}={self.translate(kw.value)}" for kw in node.keywords]
            all_args = ", ".join(args + kw_args)
            return f"_{method}({all_args})"

        # 9. neighbors error
        if isinstance(fn, ast.Attribute) and fn.attr == "neighbors":
            raise CompilationError(
                "ol.neighbors() must be used as the iterator of a for-loop: `for n in ol.neighbors(cell, radius=r):`",
                lineno=getattr(node, "lineno", None))

        # 9. Fallback (general function call)
        fn_str = self.translate(fn)
        args = [self.translate(a) for a in node.args]
        return f"{fn_str}({', '.join(args)})"

    def _translate_math_call(self, func_name: str, node: ast.Call) -> str:
        wgsl_fn = _MATH_MAP.get(func_name)
        args = [self.translate(a) for a in node.args]
        if func_name == "length_sq":
            v = args[0]
            return f"dot({v}, {v})"
        if func_name == "distance_sq":
            d = self.ctx.fresh_tmp("_dsq")
            return f"dot({args[0]} - {args[1]}, {args[0]} - {args[1]})"
        if wgsl_fn is None:
            raise CompilationError(
                f"ol.math.{func_name}() has no direct WGSL equivalent. Use a supported function instead.",
                python_line=getattr(node, "lineno", None))
        return f"{wgsl_fn}({', '.join(args)})"


class StmtTranslator(ast.NodeVisitor):
    def __init__(self, ctx: TranslationContext) -> None:
        self.ctx = ctx
        self._expr = ExprTranslator(ctx)

    def translate_body(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            result = self.visit(stmt)
            if isinstance(result, list):
                for s in result:
                    self.visit(s)

    def _expr_str(self, node: ast.expr) -> str:
        return self._expr.translate(node)

    def _infer_type(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant): return infer_literal_wgsl_type(node)
        if isinstance(node, ast.Name):
            if node.id in self.ctx.local_vars: return self.ctx.local_vars[node.id]
            if node.id in self.ctx.extracted_constants: return self.ctx.extracted_constants[node.id][0]
            return None
        if isinstance(node, ast.Attribute):
            obj = node.value
            if isinstance(obj, ast.Name):
                t = self.ctx.field_type(obj.id, node.attr)
                if t: return t
                if obj.id == "self": return self.ctx.uniforms.get(node.attr)
            return None
        if isinstance(node, ast.BinOp):
            lt = self._infer_type(node.left)
            rt = self._infer_type(node.right)
            if lt and rt:
                if lt == "f32" or rt == "f32": return "f32"
                return lt
            return lt or rt
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "vec3": return "vec3<f32>"
            if isinstance(fn, ast.Name) and fn.id == "vec3": return "vec3<f32>"

            if isinstance(fn, ast.Name) and self.ctx.main_class and fn.id == self.ctx.main_class.__name__:
                if self.ctx.rule_type == "cell": return "Cell"
                if self.ctx.rule_type == "tissue": return "Tissue"
                if self.ctx.rule_type == "chemistry": return "Chemistry"

            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) and fn.value.attr == "math":
                if fn.attr in (
                        "length", "distance", "dot", "length_sq", "distance_sq", "clamp", "lerp",
                        "smoothstep"): return "f32"
                if fn.attr in ("normalize", "reflect", "lerp_vec", "cross"): return "vec3<f32>"
            # ol.random(agent) → f32 ,  ol.random_dir(agent) → vec3<f32>
            if (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "ol"):
                if fn.attr == "random": return "f32"
                if fn.attr == "random_dir": return "vec3<f32>"
        return None

    def _is_rand_call(self, node: ast.expr) -> tuple[bool, str, str]:
        """Returns (is_rand, wgsl_call_str, wgsl_type) if node is ol.random/ol.random_dir."""
        if not isinstance(node, ast.Call):
            return False, "", ""
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "ol"):
            return False, "", ""
        agent = _get_name(node.args[0]) if node.args else self.ctx.main_param
        if fn.attr == "random":
            return True, f"_next_rand(&_rng)", "f32"
        if fn.attr == "random_dir":
            return True, f"_rand_dir(&_rng)", "vec3<f32>"
        return False, "", ""

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1: raise CompilationError("Multi-target assignments are not supported in GPU rules",
                                                          lineno=node.lineno)
        target = node.targets[0]
        value_node = node.value

        # Rand call → always emit as `let`, never register in local_vars (no hoisted var needed)
        if isinstance(target, ast.Name):
            is_rand, rand_str, rand_type = self._is_rand_call(value_node)
            if is_rand:
                var_name = target.id
                self.ctx.emit(f"let {var_name} = {rand_str};", node.lineno)
                # Register type for inference; mark as let so no var-decl is emitted
                self.ctx.local_vars[var_name] = rand_type
                self.ctx.let_vars.add(var_name)
                return

        value_str = self._expr_str(value_node)
        if isinstance(target, ast.Name):
            var_name = target.id
            if var_name not in self.ctx.local_vars:
                inferred = self._infer_type(value_node)
                if inferred:
                    self.ctx.local_vars[var_name] = inferred
            self.ctx.emit(f"{var_name} = {value_str};", node.lineno)
            return

        target_str = self._expr_str(target)
        self.ctx.emit(f"{target_str} = {value_str};", node.lineno)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name): raise CompilationError(
            "Only simple name targets are supported in annotated assignments", python_line=node.lineno)
        var_name = node.target.id
        ann = node.annotation
        wgsl_type = None
        if isinstance(ann, ast.Name):
            py_map = {"f32": "f32", "i32": "i32", "u32": "u32", "float": "f32", "int": "i32", "bool": "bool",
                      "vec3": "vec3<f32>"}
            wgsl_type = py_map.get(ann.id)
        if wgsl_type is None:
            wgsl_type = self._infer_type(node.value) if node.value else "f32"

        self.ctx.local_vars[var_name] = wgsl_type
        if node.value:
            val_str = self._expr_str(node.value)
            self.ctx.emit(f"{var_name} = {val_str};", node.lineno)
        else:
            from ._type_system import wgsl_zero
            self.ctx.emit(f"{var_name} = {wgsl_zero(wgsl_type)};", node.lineno)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        op = _AUGOP_MAP.get(type(node.op))
        if op is None: raise CompilationError(f"Unsupported augmented-assignment operator {type(node.op).__name__}",
                                              python_line=node.lineno)
        value_str = self._expr_str(node.value)

        if (isinstance(node.target, ast.Attribute) and
                isinstance(node.target.value, ast.Name) and
                self.ctx.is_metrics_attr(node.target.value.id)):

            field = node.target.attr
            field_wgsl_type = self.ctx.metrics_hints.get(field, "i32")

            if isinstance(node.op, ast.Add):
                if field_wgsl_type == "f32":
                    self.ctx.emit(f"_atomicAddF32(&MetricsBuffer.{field}_bits, {value_str});", node.lineno)
                else:
                    self.ctx.emit(f"atomicAdd(&MetricsBuffer.{field}, {value_str});", node.lineno)
            elif isinstance(node.op, ast.Sub):
                if field_wgsl_type == "f32":
                    self.ctx.emit(f"_atomicAddF32(&MetricsBuffer.{field}_bits, -({value_str}));", node.lineno)
                else:
                    self.ctx.emit(f"atomicSub(&MetricsBuffer.{field}, {value_str});", node.lineno)
            else:
                raise CompilationError(f"Only += and -= are supported for metric fields (field: {field!r})",
                                       python_line=node.lineno)
            return

        target_str = self._expr_str(node.target)
        self.ctx.emit(f"{target_str} {op}= {value_str};", node.lineno)

    def visit_If(self, node: ast.If) -> None:
        cond = self._expr_str(node.test)
        self.ctx.emit(f"if ({cond}) {{", node.lineno)
        self.ctx.indent()
        self.translate_body(node.body)
        self.ctx.dedent()
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                elif_node = node.orelse[0]
                elif_cond = self._expr_str(elif_node.test)
                self.ctx.emit(f"}} else if ({elif_cond}) {{", elif_node.lineno)
                self.ctx.indent()
                self.translate_body(elif_node.body)
                self.ctx.dedent()
                if elif_node.orelse:
                    self._emit_else_chain(elif_node.orelse)
                else:
                    self.ctx.emit("}")
            else:
                self.ctx.emit("} else {")
                self.ctx.indent()
                self.translate_body(node.orelse)
                self.ctx.dedent()
                self.ctx.emit("}")
        else:
            self.ctx.emit("}")

    def _emit_else_chain(self, orelse: list[ast.stmt]) -> None:
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            elif_node = orelse[0]
            elif_cond = self._expr_str(elif_node.test)
            self.ctx.emit(f"}} else if ({elif_cond}) {{", elif_node.lineno)
            self.ctx.indent()
            self.translate_body(elif_node.body)
            self.ctx.dedent()
            if elif_node.orelse:
                self._emit_else_chain(elif_node.orelse)
            else:
                self.ctx.emit("}")
        else:
            self.ctx.emit("} else {")
            self.ctx.indent()
            self.translate_body(orelse)
            self.ctx.dedent()
            self.ctx.emit("}")

    def visit_For(self, node: ast.For) -> None:
        iter_node = node.iter

        if _is_attr_iter(iter_node, "neighbors"):
            if self.ctx.rule_type == "cell":
                self._emit_cell_neighbors_macro(node)
            elif self.ctx.rule_type == "tissue":
                self._emit_tissue_neighbors_macro(node)
            elif self.ctx.rule_type == "chemistry":
                self._emit_chem_neighbors_macro(node)
            return

        if _is_attr_iter(iter_node, "cells"):
            if self.ctx.rule_type == "tissue":
                self._emit_tissue_cells_macro(node)
            elif self.ctx.rule_type == "chemistry":
                self._emit_chem_cells_macro(node)
            return

        if _is_attr_iter(iter_node, "tissues"):
            if self.ctx.rule_type == "chemistry": self._emit_chem_tissues_macro(node)
            return

        raise CompilationError(
            "Unsupported for-loop. Use attributes like 'cell.neighbors', 'tissue.cells'.",
            python_line=node.lineno)

    def _emit_cell_neighbors_macro(self, node: ast.For) -> None:
        agent_var = _get_name(node.iter.value)
        nb_var = _get_name(node.target)

        k = self.ctx.fresh_tmp("_k")
        nv = self.ctx.fresh_tmp("_nv")
        nkey = self.ctx.fresh_tmp("_nkey")
        ns = self.ctx.fresh_tmp("_ns")
        ne = self.ctx.fresh_tmp("_ne")
        j = self.ctx.fresh_tmp("_j")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"let _my_voxel = vec3<i32>({agent_var}.pos / U.TissueVoxelSize);")
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(
            f"let {nv} = clamp(_my_voxel + MooreOffsets[{k}], vec3<i32>(0), vec3<i32>(i32(U.TissueGridDim) - 1));")
        self.ctx.emit(f"let {nkey} = z_order_hash({nv});")
        self.ctx.emit(f"let {ns} = VoxelTable[{nkey}].startIndex;")
        self.ctx.emit(f"let {ne} = {ns} + atomicLoad(&VoxelTable[{nkey}].count);")
        self.ctx.emit(f"for (var {j} = {ns}; {j} < {ne}; {j} = {j} + 1u) {{")
        self.ctx.indent()

        if self.ctx.rule_type == "cell":
            self.ctx.emit(f"if ({j} == cell_index) {{ continue; }}")

        self.ctx.emit(f"var {nb_var} = Cells_In[{j}];")

        self.translate_body(node.body)

        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_tissue_neighbors_macro(self, node: ast.For) -> None:
        t_var = _get_name(node.target)
        k = self.ctx.fresh_tmp("_k")
        nc = self.ctx.fresh_tmp("_nc")
        nkey = self.ctx.fresh_tmp("_nkey")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"if ({k} == 13u) {{ continue; }}")
        self.ctx.emit(
            f"let {nc} = clamp(tissue_coord + MooreOffsets[{k}], vec3<i32>(0), vec3<i32>(i32(U.TissueGridDim) - 1));")
        self.ctx.emit(f"let {nkey} = z_order_hash({nc});")
        self.ctx.emit(f"var {t_var} = Tissue_In[{nkey}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_tissue_neighbors_macro(self, node: ast.For) -> None:
        t_var = _get_name(node.target)
        k = self.ctx.fresh_tmp("_k")
        nc = self.ctx.fresh_tmp("_nc")
        nkey = self.ctx.fresh_tmp("_nkey")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"if ({k} == 13u) {{ continue; }}")
        self.ctx.emit(
            f"let {nc} = clamp(tissue_coord + MooreOffsets[{k}], vec3<i32>(0), vec3<i32>(i32(U.TissueGridDim) - 1));")
        self.ctx.emit(f"let {nkey} = z_order_hash({nc});")
        self.ctx.emit(f"var {t_var} = Tissue_In[{nkey}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_tissue_cells_macro(self, node: ast.For) -> None:
        c_var = _get_name(node.target)
        j = self.ctx.fresh_tmp("_j")
        tkey = self.ctx.fresh_tmp("_tkey")
        ts = self.ctx.fresh_tmp("_ts")
        te = self.ctx.fresh_tmp("_te")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"let {tkey} = tissue_index;")
        self.ctx.emit(f"let {ts} = VoxelTable[{tkey}].startIndex;")
        self.ctx.emit(f"let {te} = {ts} + atomicLoad(&VoxelTable[{tkey}].count);")
        self.ctx.emit(f"for (var {j} = {ts}; {j} < {te}; {j} = {j} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"var {c_var} = Cells_In[{j}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_chem_neighbors_macro(self, node: ast.For) -> None:
        c_var = _get_name(node.target)
        k = self.ctx.fresh_tmp("_k")
        nc = self.ctx.fresh_tmp("_nc")
        nkey = self.ctx.fresh_tmp("_nkey")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"if ({k} == 13u) {{ continue; }}")
        self.ctx.emit(
            f"let {nc} = clamp(chem_coord + MooreOffsets[{k}], vec3<i32>(0), vec3<i32>(i32(U.ChemicalGridDim) - 1));")
        self.ctx.emit(f"let {nkey} = get_chemical_voxel_key({nc});")
        self.ctx.emit(f"var {c_var} = Chemistry_In[{nkey}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_chem_tissues_macro(self, node: ast.For) -> None:
        t_var = _get_name(node.target)
        i = self.ctx.fresh_tmp("_i")
        start = self.ctx.fresh_tmp("_tstart")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"let {start} = chem_index * 8u;")
        self.ctx.emit(f"for (var {i} = 0u; {i} < 8u; {i} = {i} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"var {t_var} = TissueBuffer[{start} + {i}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_chem_cells_macro(self, node: ast.For) -> None:
        c_var = _get_name(node.target)
        i = self.ctx.fresh_tmp("_i")
        start = self.ctx.fresh_tmp("_tstart")
        j = self.ctx.fresh_tmp("_j")
        tkey = self.ctx.fresh_tmp("_tkey")
        ts = self.ctx.fresh_tmp("_ts")
        te = self.ctx.fresh_tmp("_te")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"let {start} = chem_index * 8u;")
        self.ctx.emit(f"for (var {i} = 0u; {i} < 8u; {i} = {i} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"let {tkey} = {start} + {i};")
        self.ctx.emit(f"let {ts} = VoxelTable[{tkey}].startIndex;")
        self.ctx.emit(f"let {te} = {ts} + atomicLoad(&VoxelTable[{tkey}].count);")
        self.ctx.emit(f"for (var {j} = {ts}; {j} < {te}; {j} = {j} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"var {c_var} = Cells_In[{j}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            self.ctx.emit("return;", node.lineno)
        else:
            val = self._expr_str(node.value)
            self.ctx.emit(f"return {val};", node.lineno)

    def visit_Pass(self, node: ast.Pass) -> None:
        self.ctx.emit("// pass", node.lineno)

    def visit_Expr(self, node: ast.Expr) -> None:
        call = node.value
        if not isinstance(call, ast.Call):
            self.ctx.emit(f"let _ = {self._expr_str(call)};", node.lineno)
            return
        fn = call.func
        if isinstance(fn, ast.Attribute) and fn.attr == "die":
            obj = _get_name(fn.value)
            self.ctx.emit(
                f"{obj}.pos = vec3<f32>(f32(U.TissueGridDimX), f32(U.TissueGridDimY), f32(U.TissueGridDimZ)) "
                f"* U.TissueVoxelSize + vec3<f32>(U.TissueVoxelSize) / 2;",
                node.lineno
            )
            return
        if isinstance(fn, ast.Attribute) and fn.attr == "divide":
            obj = _get_name(fn.value)
            if call.args:
                new_cell = self._expr_str(call.args[0])
                idx = self.ctx.fresh_tmp("_spawn_idx")
                self.ctx.emit(f"let {idx} = atomicAdd(&State.NewTotalAgents, 1u);", node.lineno)
                self.ctx.emit(f"if ({idx} < arrayLength(&Cells_Out)) {{")
                self.ctx.indent()
                self.ctx.emit(f"Cells_Out[{idx}] = {new_cell};")
                self.ctx.dedent()
                self.ctx.emit("}")
            return
        call_str = self._expr_str(call)
        self.ctx.emit(f"{call_str};", node.lineno)


def _get_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return f"{_get_name(node.value)}.{node.attr}"
    return ast.unparse(node)


def _is_neighbors_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call): return False
    fn = node.func
    return (isinstance(fn, ast.Attribute) and fn.attr == "neighbors" and (
                (isinstance(fn.value, ast.Name) and fn.value.id == "ol") or isinstance(fn.value, ast.Attribute)))


def _is_attr_iter(node: ast.expr, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr
