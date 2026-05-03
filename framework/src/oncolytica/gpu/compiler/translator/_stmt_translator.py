from __future__ import annotations

import ast

from ._expr_translator import ExprTranslator
from ._context import TranslationContext
from oncolytica.gpu.compiler._type_system import wgsl_zero
from ._constants import _AUGOP_MAP


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
        """Return the WGSL type of *node* via ctx.node_wgsl_type."""
        return self.ctx.node_wgsl_type(node)

    # ── Rule epilogue ─────────────────────────────────────────────────────────

    def _emit_rule_epilogue(self, py_line: int | None = None) -> None:
        """Emit the two lines that must precede every ``return`` in a rule kernel:
            <param>._rng_state = _rng;
            Cells_Out[cell_index] = <param>;   (or Tissue_Out / Chemistry_Out)

        For non-rule (helper) contexts this is a no-op.
        """
        if not self.ctx.is_rule:
            return
        for line in self.ctx.rule_epilogue_lines():
            self.ctx.emit(line, py_line)

    # ── Visitors ──────────────────────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        target     = node.targets[0]
        value_node = node.value

        # ── Domain class constructor with keyword args: x = MyCell(pos=…) ────
        #    Matches any registered domain class, not just main_class, so that
        #    helpers can construct any domain type (e.g. MyTissue(oxygen=0.5)).
        if (isinstance(target, ast.Name)
                and isinstance(value_node, ast.Call)
                and isinstance(value_node.func, ast.Name)
                and value_node.keywords):
            matched_base: type | None = None
            for user_cls, base_cls in self.ctx.val_ctx.memory_base_map.items():
                if getattr(user_cls, "__name__", None) == value_node.func.id:
                    matched_base = base_cls
                    break
            if matched_base is not None:
                var_name    = target.id
                struct_name = matched_base.__name__
                hints       = self.ctx.struct_hints.get(struct_name, {})

                self.ctx.local_vars.setdefault(var_name, struct_name)
                self.ctx.emit(f"{var_name} = {struct_name}();", node.lineno)

                for kw in value_node.keywords:
                    field_name = kw.arg
                    if field_name is None:
                        continue
                    field_val_node  = kw.value
                    field_wgsl_type = hints.get(field_name)
                    field_val_str   = (
                        self._expr.translate_as(field_val_node, field_wgsl_type)
                        if field_wgsl_type
                        else self._expr_str(field_val_node)
                    )
                    self.ctx.emit(f"{var_name}.{field_name} = {field_val_str};")
                return

        # ── General assignment ─────────────────────────────────────────────────
        # ol.random / ol.random_dir are handled by ExprTranslator.visit_Call,
        # which inlines them directly as _next_rand(&_rng) / _rand_dir(&_rng).
        # No special path needed here.
        if isinstance(target, ast.Name):
            var_name = target.id
            is_param = var_name in self.ctx.method_params
            if not is_param and var_name not in self.ctx.local_vars:
                inferred = self._infer_type(value_node)
                if inferred:
                    self.ctx.local_vars[var_name] = inferred
            local_type = self.ctx.local_vars.get(var_name) or self.ctx.method_params.get(var_name)
            value_str  = (
                self._expr.translate_as(value_node, local_type)
                if local_type
                else self._expr_str(value_node)
            )
            self.ctx.emit(f"{var_name} = {value_str};", node.lineno)
            return

        target_str = self._expr_str(target)
        if isinstance(target, ast.Attribute):
            field_type = None
            for struct_fields in self.ctx.struct_hints.values():
                t = struct_fields.get(target.attr)
                if t is not None:
                    field_type = t
                    break
            value_str = (
                self._expr.translate_as(value_node, field_type)
                if field_type
                else self._expr_str(value_node)
            )
        else:
            value_str = self._expr_str(value_node)
        self.ctx.emit(f"{target_str} = {value_str};", node.lineno)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        var_name = node.target.id
        ann      = node.annotation

        # ``ol.bool`` / ``bool`` → WGSL ``i32`` for local variables.
        _ANN_MAP: dict[str, str] = {
            "f32": "f32", "i32": "i32", "u32": "u32",
            "float": "f32", "int": "i32", "bool": "i32",
            "vec3": "vec3<f32>",
        }

        wgsl_type: str | None = None
        if isinstance(ann, ast.Name):
            wgsl_type = _ANN_MAP.get(ann.id)
        elif (isinstance(ann, ast.Attribute)
              and isinstance(ann.value, ast.Name)
              and ann.value.id == "ol"):
            wgsl_type = _ANN_MAP.get(ann.attr)

        if wgsl_type is None:
            wgsl_type = self._infer_type(node.value) if node.value else "f32"

        if var_name not in self.ctx.method_params:
            self.ctx.local_vars[var_name] = wgsl_type
        else:
            pass

        if node.value:
            val_str = self._expr.translate_as(node.value, wgsl_type)
            self.ctx.emit(f"{var_name} = {val_str};", node.lineno)
        else:
            self.ctx.emit(f"{var_name} = {wgsl_zero(wgsl_type)};", node.lineno)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        op = _AUGOP_MAP.get(type(node.op))
        value_str = self._expr_str(node.value)

        # Metrics fields use atomic helpers.
        if (isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and self.ctx.is_metrics_attr(node.target.value.id)):

            field           = node.target.attr
            field_wgsl_type = self.ctx.metrics_hints.get(field, "i32")

            if isinstance(node.op, ast.Add):
                if field_wgsl_type == "f32":
                    self.ctx.emit(
                        f"_atomicAddF32(&MetricsBuffer.{field}_bits, {value_str});",
                        node.lineno,
                    )
                else:
                    self.ctx.emit(
                        f"atomicAdd(&MetricsBuffer.{field}, {value_str});",
                        node.lineno,
                    )
            elif isinstance(node.op, ast.Sub):
                if field_wgsl_type == "f32":
                    self.ctx.emit(
                        f"_atomicAddF32(&MetricsBuffer.{field}_bits, -({value_str}));",
                        node.lineno,
                    )
                else:
                    self.ctx.emit(
                        f"atomicSub(&MetricsBuffer.{field}, {value_str});",
                        node.lineno,
                    )
            return

        target_str = self._expr_str(node.target)
        self.ctx.emit(f"{target_str} {op}= {value_str};", node.lineno)

    def _cond_str(self, node: ast.expr) -> str:
        """Translate a condition, coercing i32 → bool via node_wgsl_type."""
        return self._expr.translate_as(node, "bool")

    def visit_If(self, node: ast.If) -> None:
        cond = self._cond_str(node.test)
        self.ctx.emit(f"if ({cond}) {{", node.lineno)
        self.ctx.indent()
        self.translate_body(node.body)
        self.ctx.dedent()
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                elif_node = node.orelse[0]
                elif_cond = self._cond_str(elif_node.test)
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
            elif_cond = self._cond_str(elif_node.test)
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
            if self.ctx.rule_type == "chemistry":
                self._emit_chem_tissues_macro(node)
            return

    def visit_While(self, node: ast.While) -> None:
        cond = self._cond_str(node.test)
        self.ctx.emit(f"loop {{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"if (!({cond})) {{ break; }}")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")

    def visit_Break(self, node: ast.Break) -> None:
        self.ctx.emit("break;", node.lineno)

    def visit_Continue(self, node: ast.Continue) -> None:
        self.ctx.emit("continue;", node.lineno)

    # ── Simple statements ─────────────────────────────────────────────────────

    def visit_Return(self, node: ast.Return) -> None:
        # In rule kernels: flush rng_state and write Out-buffer before leaving.
        # In helper functions: no epilogue (caller owns the Out-buffer write).
        self._emit_rule_epilogue(node.lineno)

        if node.value is None:
            self.ctx.emit("return;", node.lineno)
        else:
            self.ctx.emit(f"return {self._expr_str(node.value)};", node.lineno)

    def visit_Pass(self, node: ast.Pass) -> None:
        self.ctx.emit("// pass", node.lineno)

    def visit_Expr(self, node: ast.Expr) -> None:
        call = node.value
        if not isinstance(call, ast.Call):
            self.ctx.emit(f"let _ = {self._expr_str(call)};", node.lineno)
            return
        fn = call.func

        # a.copy_from(b)  →  a = b;   (or (*a) = b if a is a ptr-param)
        if (isinstance(fn, ast.Attribute)
                and fn.attr == "copy_from"
                and len(call.args) == 1):
            lhs_name = _get_name(fn.value)
            # If the receiver is a ptr-param, dereference it on the LHS.
            lhs = f"(*{lhs_name})" if lhs_name in self.ctx.ptr_params else lhs_name
            rhs = self._expr_str(call.args[0])
            self.ctx.emit(f"{lhs} = {rhs};", node.lineno)
            return

        if isinstance(fn, ast.Attribute) and fn.attr == "die":
            obj_name = _get_name(fn.value)
            # If the receiver is a ptr-param, dereference it.
            obj = f"(*{obj_name})" if obj_name in self.ctx.ptr_params else obj_name
            self.ctx.emit(
                f"{obj}.pos = vec3<f32>(f32(U.TissueGridDimX), f32(U.TissueGridDimY), f32(U.TissueGridDimZ)) "
                f"* U.TissueVoxelSize + vec3<f32>(U.TissueVoxelSize) / 2;",
                node.lineno,
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

        self.ctx.emit(f"{self._expr_str(call)};", node.lineno)

    # ── Neighbor / iteration macros ───────────────────────────────────────────

    def _emit_cell_neighbors_macro(self, node: ast.For) -> None:
        agent_var = _get_name(node.iter.value)
        nb_var    = _get_name(node.target)

        k    = self.ctx.fresh_tmp("_k")
        nv   = self.ctx.fresh_tmp("_nv")
        nkey = self.ctx.fresh_tmp("_nkey")
        ns   = self.ctx.fresh_tmp("_ns")
        ne   = self.ctx.fresh_tmp("_ne")
        j    = self.ctx.fresh_tmp("_j")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"let _my_voxel = vec3<i32>({agent_var}.pos / U.TissueVoxelSize);")
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u) {{")
        self.ctx.indent()
        self.ctx.emit(f"let {nv} = _my_voxel + MooreOffsets[{k}];")
        self.ctx.emit(f"if (!_is_tissue_coord_in_bounds({nv})) {{ continue; }}")
        self.ctx.emit(f"let {nkey} = _z_order_hash({nv});")
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
        t_var      = _get_name(node.target)
        source_obj = _get_name(node.iter.value)

        k    = self.ctx.fresh_tmp("_k")
        nc   = self.ctx.fresh_tmp("_nc")
        nkey = self.ctx.fresh_tmp("_nkey")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u + u32({k} == 13)) {{")
        self.ctx.indent()
        self.ctx.emit(f"let {nc} = {source_obj}._coord + MooreOffsets[{k}];")
        self.ctx.emit(f"if (!_is_tissue_coord_in_bounds({nc})) {{ continue; }}")
        self.ctx.emit(
            f"let {nc} = clamp(tissue_coord + MooreOffsets[{k}], "
            f"vec3<i32>(0), vec3<i32>(i32(U.TissueGridDim) - 1));"
        )
        self.ctx.emit(f"let {nkey} = z_order_hash({nc});")
        self.ctx.emit(f"var {t_var} = Tissue_In[{nkey}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_tissue_cells_macro(self, node: ast.For) -> None:
        c_var = _get_name(node.target)
        j     = self.ctx.fresh_tmp("_j")
        tkey  = self.ctx.fresh_tmp("_tkey")
        ts    = self.ctx.fresh_tmp("_ts")
        te    = self.ctx.fresh_tmp("_te")

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
        c_var      = _get_name(node.target)
        source_obj = _get_name(node.iter.value)

        k    = self.ctx.fresh_tmp("_k")
        nc   = self.ctx.fresh_tmp("_nc")
        nkey = self.ctx.fresh_tmp("_nkey")

        self.ctx.emit("{", node.lineno)
        self.ctx.indent()
        self.ctx.emit(f"for (var {k} = 0u; {k} < 27u; {k} = {k} + 1u + u32({k} == 13)) {{")
        self.ctx.indent()
        self.ctx.emit(f"let {nc} = {source_obj}._coord + MooreOffsets[{k}];")
        self.ctx.emit(f"if (!_is_chem_coord_in_bounds({nc})) {{ continue; }}")
        self.ctx.emit(f"let {nkey} = get_chemical_voxel_key({nc});")
        self.ctx.emit(f"var {c_var} = Chemistry_In[{nkey}];")
        self.translate_body(node.body)
        self.ctx.dedent()
        self.ctx.emit("}")
        self.ctx.dedent()
        self.ctx.emit("}")

    def _emit_chem_tissues_macro(self, node: ast.For) -> None:
        t_var = _get_name(node.target)
        i     = self.ctx.fresh_tmp("_i")
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
        c_var  = _get_name(node.target)
        i      = self.ctx.fresh_tmp("_i")
        start  = self.ctx.fresh_tmp("_tstart")
        j      = self.ctx.fresh_tmp("_j")
        tkey   = self.ctx.fresh_tmp("_tkey")
        ts     = self.ctx.fresh_tmp("_ts")
        te     = self.ctx.fresh_tmp("_te")

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


# ── Utilities ─────────────────────────────────────────────────────────────────

def _get_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):      return node.id
    if isinstance(node, ast.Attribute): return f"{_get_name(node.value)}.{node.attr}"
    return ast.unparse(node)


def _is_neighbors_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    return (isinstance(fn, ast.Attribute)
            and fn.attr == "neighbors"
            and (
                (isinstance(fn.value, ast.Name) and fn.value.id == "ol")
                or isinstance(fn.value, ast.Attribute)
            ))


def _is_attr_iter(node: ast.expr, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr
