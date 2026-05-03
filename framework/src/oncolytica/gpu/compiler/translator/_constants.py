import ast

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
    "distance": "distance",
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

_INT_WGSL   = frozenset({"i32", "u32"})
_FLOAT_WGSL = frozenset({"f32"})
_BOOL_WGSL  = frozenset({"bool"})
_VEC_WGSL   = frozenset({"vec3<f32>", "vec3<i32>"})

# Return types of ol.math.* functions
_MATH_RETURN: dict[str, str] = {
    # → f32
    "length":      "f32",
    "distance":    "f32",
    "dot":         "f32",
    "length_sq":   "f32",
    "distance_sq": "f32",
    "clamp":       "f32",
    "lerp":        "f32",
    "smoothstep":  "f32",
    "sign":        "f32",
    "sqrt":        "f32",
    "exp":         "f32",
    "log":         "f32",
    "log2":        "f32",
    "pow":         "f32",
    "sin":         "f32",
    "cos":         "f32",
    "tan":         "f32",
    "asin":        "f32",
    "acos":        "f32",
    "atan2":       "f32",
    "floor":       "f32",
    "ceil":        "f32",
    "fabs":        "f32",
    "abs":         "f32",
    # → vec3<f32>
    "normalize":   "vec3<f32>",
    "reflect":     "vec3<f32>",
    "lerp_vec":    "vec3<f32>",
    "cross":       "vec3<f32>",
}

# Explicit cast map: function name (bare or attr) → target WGSL type
# Covers both bare names (i32, f32, …) and ol.i32 / any_prefix.i32 forms.
_EXPLICIT_CAST: dict[str, str] = {
    "i32":   "i32",
    "int":   "i32",
    "u32":   "u32",
    "f32":   "f32",
    "float": "f32",
    "bool":  "bool",
}

# Zero literal for != 0 checks when coercing int → bool
_INT_ZERO: dict[str, str] = {
    "i32": "0",
    "u32": "0u",
}