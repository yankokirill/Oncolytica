from __future__ import annotations

import ast

from ._context import TranslationContext
from oncolytica.core.utils._errors import CompilationError
from oncolytica.gpu.compiler._type_system import (
    format_float_literal,
    format_int_literal,
)
from ._constants import (
    _BINOP_MAP,
    _CMP_MAP,
    _MATH_MAP,
)
from .._type_system import domain_base_of

# Grid-getter self.X() calls that are DSL builtins — they do NOT receive &_rng.
_GRID_GETTERS: frozenset[str] = frozenset({"tissue_at", "chemistry_at"})


class ExprTranslator(ast.NodeVisitor):
    def __init__(self, ctx: TranslationContext) -> None:
        self.ctx = ctx

    def translate(self, node: ast.expr) -> str:
        return self.visit(node)

    def get_expr_type(self, node: ast.expr) -> str:
        """Determine the WGSL type of *node*.

        Priority:
        1. Structural type from the AST node kind (Compare/BoolOp/Not → "bool",
           arithmetic → "i32"/"f32", bool constants → "bool").
        2. TypeChecker type_map via ``ctx.node_wgsl_type`` (variables, attrs,
           calls, …).
        3. Fallback: "i32".
        """
        # ── Constant literals ─────────────────────────────────────────────────
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):  return "bool"
            if isinstance(node.value, int):   return "i32"
            if isinstance(node.value, float): return "f32"
        # ── bool Name literals ────────────────────────────────────────────────
        if isinstance(node, ast.Name) and node.id in ("True", "False"):
            return "bool"
        # ── Logical producers → always bool ──────────────────────────────────
        if isinstance(node, ast.Compare):
            return "bool"
        if isinstance(node, ast.BoolOp):
            return "bool"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return "bool"
        # ── Delegate to TypeChecker for everything else ───────────────────────
        t = self.ctx.node_wgsl_type(node)
        return t if t else "i32"

    def translate_as(self, node: ast.expr, target_type: str) -> str:
        """Translate *node* and cast to *target_type* if needed.

        Supported coercions:
        * ``bool`` → ``i32`` :  ``i32(<expr>)``  (with constant folding)
        * ``i32``  → ``bool`` :  ``(<expr> != 0)``

        Forbidden (raises CompilationError):
        * ``bool`` ↔ ``u32``
        """
        actual_type = self.get_expr_type(node)

        if actual_type == target_type:
            return self.translate(node)

        # ── Forbidden cast ────────────────────────────────────────────────────
        if (actual_type == "u32" and target_type == "bool") or \
           (actual_type == "bool" and target_type == "u32"):
            raise CompilationError(
                f"Implicit cast between 'bool' and 'u32' is forbidden."
            )

        # ── bool → i32 constant folding ───────────────────────────────────────
        if target_type == "i32" and actual_type == "bool":
            if isinstance(node, ast.Constant) and isinstance(node.value, bool):
                return "1" if node.value else "0"
            if isinstance(node, ast.Name) and node.id in ("True", "False"):
                return "1" if node.id == "True" else "0"

        expr_str = self.translate(node)

        # ── Coercions ─────────────────────────────────────────────────────────
        if target_type == "i32" and actual_type == "bool":
            return f"i32({expr_str})"
        if target_type == "bool" and actual_type == "i32":
            return f"({expr_str} != 0)"

        return expr_str

    # ── Literals ──────────────────────────────────────────────────────────────

    def visit_Constant(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, bool):  return "true" if v else "false"
        if isinstance(v, float): return format_float_literal(v)
        if isinstance(v, int):   return str(v)

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name == "True":  return "true"
        if name == "False": return "false"
        # ptr-parameter used as a value expression (e.g. passed to a read-only
        # param or returned): dereference to obtain the struct value.
        if name in self.ctx.ptr_params:
            return f"(*{name})"
        if name in self.ctx.local_vars:
            return name
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
        # self.params.X  →  U.X
        if (isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "params"):
            return f"U.{node.attr}"

        # self.X  →  U.X  (sim uniform / global)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"U.{node.attr}"

        # ptr-parameter field access: (*param).field
        if (isinstance(node.value, ast.Name)
                and node.value.id in self.ctx.ptr_params):
            return f"(*{node.value.id}).{node.attr}"

        obj = self.visit(node.value)
        return f"{obj}.{node.attr}"

    def _as_numeric(self, expr_str: str, node: ast.expr) -> str:
        """Emit bool → i32 coercion for arithmetic contexts."""
        if self.ctx.node_wgsl_type(node) == "bool":
            return f"i32({expr_str})"
        return expr_str

    # ── Operators ─────────────────────────────────────────────────────────────

    def visit_BinOp(self, node: ast.BinOp) -> str:
        if isinstance(node.op, ast.Pow):
            left  = self.translate(node.left)
            right = self.translate(node.right)
            return f"pow({left}, {right})"
        op = _BINOP_MAP.get(type(node.op))
        lt = self.get_expr_type(node.left)
        rt = self.get_expr_type(node.right)
        target = "f32" if "f32" in (lt, rt) else ("u32" if lt == rt == "u32" else "i32")
        left  = self.translate_as(node.left,  target)
        right = self.translate_as(node.right, target)
        return f"({left} {op} {right})"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = "&&" if isinstance(node.op, ast.And) else "||"
        parts = [f"({self.translate_as(v, 'bool')})" for v in node.values]
        return f" {op} ".join(parts)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        if isinstance(node.op, ast.USub):
            t = self.get_expr_type(node.operand)
            target = "f32" if t == "f32" else "i32"
            return f"-{self.translate_as(node.operand, target)}"
        if isinstance(node.op, ast.UAdd):
            t = self.get_expr_type(node.operand)
            target = "f32" if t == "f32" else "i32"
            return self.translate_as(node.operand, target)
        if isinstance(node.op, ast.Not):
            return f"!({self.translate_as(node.operand, 'bool')})"
        raise CompilationError(f"Unsupported unary op {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> str:
        lt = self.get_expr_type(node.left)
        rt = self.get_expr_type(node.comparators[0])
        if "f32" in (lt, rt):
            operand_target = "f32"
        elif "u32" in (lt, rt):
            operand_target = "u32"
        else:
            operand_target = "i32"
        left  = self.translate_as(node.left, operand_target)
        right = self.translate_as(node.comparators[0], operand_target)
        op = _CMP_MAP.get(type(node.ops[0]))
        return f"({left} {op} {right})"

    # ── Calls ─────────────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> str:
        fn = node.func

        # 1. ol.math.XXX
        if (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Attribute)
                and isinstance(fn.value.value, ast.Name)
                and fn.value.value.id == "ol"
                and fn.value.attr == "math"):
            return self._translate_math_call(fn.attr, node)

        # 2. ol.random(agent)  →  _next_rand(&_rng)   [inlined everywhere]
        if (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "ol"
                and fn.attr == "random"):
            return "_next_rand(&_rng)"

        # 3. ol.random_dir(agent)  →  _rand_dir(&_rng)   [inlined everywhere]
        if (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "ol"
                and fn.attr == "random_dir"):
            return "_rand_dir(&_rng)"

        # 4–5. vec3 constructors
        if (isinstance(fn, ast.Attribute) and fn.attr == "vec3") or (
                isinstance(fn, ast.Name) and fn.id == "vec3"):
            args = [self.translate(a) for a in node.args]
            return f"vec3<f32>({', '.join(args)})"

        # 6–7. ivec3 constructors
        if (isinstance(fn, ast.Attribute) and fn.attr == "ivec3") or (
                isinstance(fn, ast.Name) and fn.id == "ivec3"):
            args = [self.translate(a) for a in node.args]
            return f"vec3<i32>({', '.join(args)})"

        # 8. Domain class constructor: MyCell() → Cell(), MyTissue() → Tissue(), …
        #    Matches any registered domain class by Python name, not just main_class.
        #    Required so that helper methods can construct any domain type regardless
        #    of the rule context they are called from (e.g. make_cell() -> Cell()).
        if isinstance(fn, ast.Name):
            for user_cls, base_cls in self.ctx.val_ctx.memory_base_map.items():
                if getattr(user_cls, "__name__", None) == fn.id:
                    return f"{base_cls.__name__}()"

        # 9. self.method() — DSL grid samplers and sim helpers
        if (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "self"):
            method = fn.attr

            # 9a. Spatial grid samplers — expanded inline, no &_rng needed.
            if method == "tissue_at":
                pos_str = self.translate(node.args[0])
                return (
                    f"Tissue_In[_z_order_hash(clamp(vec3<i32>({pos_str} / U.TissueVoxelSize), "
                    f"vec3<i32>(0, 0, 0), "
                    f"vec3<i32>(i32(U.TissueGridDimX) - 1, "
                    f"i32(U.TissueGridDimY) - 1, i32(U.TissueGridDimZ) - 1)))]"
                )
            if method == "chemistry_at":
                pos_str = self.translate(node.args[0])
                return (
                    f"Chemistry_In[_get_chemical_voxel_key(clamp("
                    f"vec3<i32>({pos_str} / (U.TissueVoxelSize * 2.0)), "
                    f"vec3<i32>(0, 0, 0), "
                    f"vec3<i32>(i32(U.TissueGridDimX / 2u) - 1, "
                    f"i32(U.TissueGridDimY / 2u) - 1, "
                    f"i32(U.TissueGridDimZ / 2u) - 1)))]"
                )

            # 9b. User-defined sim helper: self.helper(args)  →  _helper(args, &_rng)
            #     Parameters that are mutating (domain objects at mutating positions)
            #     are passed as pointers (&arg), except when the argument is itself
            #     already a ptr-parameter of the current function — it is forwarded
            #     bare (already &T in WGSL).
            mangled = f"sim_{method}"
            mutating_pos: set[int] = self.ctx.val_ctx.method_mutating_params.get(mangled, set())

            from .._type_system import domain_base_of as _dbo

            def _arg_is_domain(a: ast.expr) -> bool:
                """True when *a* resolves to a domain struct type."""
                # Primary: TypeChecker type_map (keyed on AST node id).
                arg_type = self.ctx.val_ctx.type_map.get(id(a))
                if arg_type is not None:
                    return _dbo(arg_type) is not None
                # Fallback: named variable — check local_vars / method_params
                # which store WGSL type strings like "Cell", "Tissue", etc.
                if isinstance(a, ast.Name):
                    wgsl_t = (self.ctx.local_vars.get(a.id)
                              or self.ctx.method_params.get(a.id))
                    if wgsl_t is not None:
                        # Domain structs have capitalised single-word WGSL names.
                        from oncolytica.core.utils._types import BASE_CLASSES
                        return any(wgsl_t == b.__name__ for b in BASE_CLASSES)
                return False

            translated_args: list[str] = []
            for j, a in enumerate(node.args):
                param_pos = j + 1  # 1-based; position 0 = self (absent)
                if param_pos in mutating_pos and _arg_is_domain(a):
                    raw_name = a.id if isinstance(a, ast.Name) else None
                    if raw_name is not None and raw_name in self.ctx.ptr_params:
                        # Already a ptr-param — forward the pointer directly.
                        translated_args.append(raw_name)
                    else:
                        # Regular local/param — take its address.
                        raw_str = (raw_name if raw_name is not None
                                   else self.translate(a))
                        translated_args.append(f"&{raw_str}")
                    continue
                translated_args.append(self.translate(a))

            kw_args  = [f"{kw.arg}={self.translate(kw.value)}" for kw in node.keywords]
            # In a rule kernel _rng is a plain local var → pass as &_rng.
            # In a helper _rng_state is already a ptr<function, u32> parameter
            # → forward bare (no &) to avoid "operand must be a reference" error.
            rng_arg  = "&_rng" if self.ctx.is_rule else "_rng_state"
            all_args = translated_args + kw_args + [rng_arg]
            return f"sim_{method}({', '.join(all_args)})"

        # 10. obj.method() where obj is a domain type (Cell, Tissue, …)
        if isinstance(fn, ast.Attribute) and not (
                isinstance(fn.value, ast.Name) and fn.value.id == "self"
        ):
            # obj.copy()  →  obj   (identity in WGSL; structs are value types)
            if fn.attr == "copy" and not node.args:
                return self.visit(fn.value)

            obj_type = self.ctx.val_ctx.type_map.get(id(fn.value))
            if obj_type is not None:
                base = domain_base_of(obj_type)
                if base is not None:
                    obj_str = self.visit(fn.value)
                    args = [f"&{obj_str}"] + [self.translate(a) for a in node.args]
                    return f"{base.__name__}_{fn.attr}({', '.join(args)})"

        # 11. Fallback — general function call
        fn_str = self.translate(fn)
        args   = [self.translate(a) for a in node.args]
        return f"{fn_str}({', '.join(args)})"

    def _translate_math_call(self, func_name: str, node: ast.Call) -> str:
        args = [self.translate(a) for a in node.args]

        if func_name == "length_sq":
            v = args[0]
            return f"dot({v}, {v})"
        if func_name == "distance_sq":
            diff = f"({args[0]} - {args[1]})"
            return f"dot({diff}, {diff})"

        wgsl_fn = _MATH_MAP.get(func_name)
        return f"{wgsl_fn}({', '.join(args)})"
