"""oncolytica._math — math namespace for simulation code.

All functions accept ``ol.vec3`` objects (and plain scalars where appropriate).
The API mirrors what will be available in the WGSL GPU backend so user
code does not need to change when switching backends.

Import via:  ``import oncolytica as ol; ol.math.length(v)``
"""

from __future__ import annotations

import builtins
import math as _math
from typing import overload

from oncolytica.core.utils._types import (
    i32, u32, f32, vec3, ivec3,
    bool as ol_bool
)

# ── Vector functions ──────────────────────────────────────────────────────────

def length(v: vec3) -> float:
    """Euclidean length (L2 norm) of *v*."""
    return _math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def length_sq(v: vec3) -> float:
    """Squared length of *v* (cheaper when only comparison is needed)."""
    return v.x * v.x + v.y * v.y + v.z * v.z


def distance(a: vec3, b: vec3) -> float:
    """Euclidean distance between *a* and *b*."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return _math.sqrt(dx * dx + dy * dy + dz * dz)


def distance_sq(a: vec3, b: vec3) -> float:
    """Squared distance (cheaper when only comparison is needed)."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def normalize(v: vec3) -> vec3:
    """Return a unit vector in the direction of *v*.

    Returns the zero vector when *v* has zero length (safe normalise).
    """
    n = length(v)
    if n == 0.0:
        return vec3(0.0, 0.0, 0.0)
    inv = 1.0 / n
    return vec3(v.x * inv, v.y * inv, v.z * inv)


def dot(a: vec3, b: vec3) -> float:
    """Dot product of *a* and *b*."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: vec3, b: vec3) -> vec3:
    """Cross product of *a* and *b*."""
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def reflect(v: vec3, normal: vec3) -> vec3:
    """Reflect *v* about *normal* (normal assumed to be unit-length)."""
    d = 2.0 * dot(v, normal)
    return vec3(v.x - d * normal.x, v.y - d * normal.y, v.z - d * normal.z)


def lerp_vec(a: vec3, b: vec3, t: float) -> vec3:
    """Component-wise linear interpolation between *a* and *b* by factor *t*."""
    it = 1.0 - t
    return vec3(a.x * it + b.x * t, a.y * it + b.y * t, a.z * it + b.z * t)


# ── Scalar functions ──────────────────────────────────────────────────────────

@overload
def clamp(value: float, min_val: float, max_val: float) -> float: ...

@overload
def clamp(value: vec3, min_val: vec3, max_val: vec3) -> vec3: ...

def clamp(value, min_val, max_val):
    if isinstance(value, vec3):
        v_min = vec3._coerce(min_val)
        v_max = vec3._coerce(max_val)
        return vec3(
            builtins.max(v_min.x, builtins.min(v_max.x, value.x)),
            builtins.max(v_min.y, builtins.min(v_max.y, value.y)),
            builtins.max(v_min.z, builtins.min(v_max.z, value.z))
        )
    return builtins.max(min_val, builtins.min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Scalar linear interpolation: ``a + (b – a) * t``."""
    return a + (b - a) * t


def sign(x: float) -> float:
    """Return +1, 0, or -1 matching the sign of *x*."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation between 0 and 1."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ── Re-exported standard scalar math ─────────────────────────────────────────
# Users can write  ol.math.sqrt(...)  etc. without importing math themselves.

sqrt  = _math.sqrt
exp   = _math.exp
log   = _math.log
log2  = _math.log2
pow   = _math.pow
sin   = _math.sin
cos   = _math.cos
tan   = _math.tan
asin  = _math.asin
acos  = _math.acos
atan2 = _math.atan2
floor = _math.floor
ceil  = _math.ceil
fabs  = _math.fabs
abs   = builtins.abs   # works on int, float, complex

pi    = _math.pi
tau   = _math.tau
inf   = _math.inf

_INTRINSIC_FUNCTIONS = {
    # ── Vector Functions (Return Scalars) ──
    "length":      f32,
    "length_sq":   f32,
    "distance":    f32,
    "distance_sq": f32,
    "dot":         f32,

    # ── Vector Functions (Return Vectors) ──
    "normalize":   vec3,
    "cross":       vec3,
    "reflect":     vec3,
    "lerp_vec":    vec3,
    "random_dir":  vec3,

    # ── Scalar Math (Return f32) ──
    "sin":         f32,
    "cos":         f32,
    "tan":         f32,
    "asin":        f32,
    "acos":        f32,
    "atan":        f32,
    "atan2":       f32,
    "sinh":        f32,
    "cosh":        f32,
    "tanh":        f32,
    "pow":         f32,
    "exp":         f32,
    "exp2":        f32,
    "log":         f32,
    "log2":        f32,
    "sqrt":        f32,
    "inverse_sqrt":f32,
    "abs":         f32,
    "fabs":        f32,
    "sign":        f32,
    "floor":       f32,
    "ceil":        f32,
    "round":       f32,
    "trunc":       f32,
    "fract":       f32,
    "min":         f32,
    "max":         f32,
    "clamp":       f32,
    "lerp":        f32,
    "step":        f32,
    "smoothstep":  f32,
    "random":      f32,

    # ── Explicit Type Constructors / Casts ──
    "i32":         i32,
    "u32":         u32,
    "f32":         f32,
    "bool":        ol_bool,
    "vec3":        vec3,
    "ivec3":       ivec3,
    "int":         i32,   # Alias
    "float":       f32,   # Alias
}

# ── Argument type constraints for WGSL strict-typing validation ───────────────
# Maps function name → tuple of accepted argument type-sets (one per positional arg).
# frozenset means "any of these types is accepted for this argument".

_FLOAT_ARG = frozenset({f32, float})
_INT_ARG   = frozenset({i32, u32, int})
_SCALAR_ARG = frozenset({f32, float, i32, u32, int})
_VEC_ARG   = frozenset({vec3})
_VEC_OR_SCALAR_ARG = frozenset({vec3, f32, float})

_INTRINSIC_ARG_TYPES: dict[str, tuple[frozenset, ...]] = {
    # ── Trig / float-only unary ──
    "sin":         (_FLOAT_ARG,),
    "cos":         (_FLOAT_ARG,),
    "tan":         (_FLOAT_ARG,),
    "asin":        (_FLOAT_ARG,),
    "acos":        (_FLOAT_ARG,),
    "atan":        (_FLOAT_ARG,),
    "sinh":        (_FLOAT_ARG,),
    "cosh":        (_FLOAT_ARG,),
    "tanh":        (_FLOAT_ARG,),
    "exp":         (_FLOAT_ARG,),
    "exp2":        (_FLOAT_ARG,),
    "log":         (_FLOAT_ARG,),
    "log2":        (_FLOAT_ARG,),
    "sqrt":        (_FLOAT_ARG,),
    "inverse_sqrt":(_FLOAT_ARG,),
    "fract":       (_FLOAT_ARG,),
    "floor":       (_FLOAT_ARG,),
    "ceil":        (_FLOAT_ARG,),
    "round":       (_FLOAT_ARG,),
    "trunc":       (_FLOAT_ARG,),

    # ── pow(f32, f32) ──
    "pow":         (_FLOAT_ARG, _FLOAT_ARG),

    # ── atan2(f32, f32) ──
    "atan2":       (_FLOAT_ARG, _FLOAT_ARG),

    # ── Scalar generic (int or float) ──
    "abs":         (_SCALAR_ARG,),
    "fabs":        (_FLOAT_ARG,),
    "sign":        (_SCALAR_ARG,),
    "min":         (_SCALAR_ARG, _SCALAR_ARG),
    "max":         (_SCALAR_ARG, _SCALAR_ARG),

    # ── lerp(f32, f32, f32) ──
    "lerp":        (_FLOAT_ARG, _FLOAT_ARG, _FLOAT_ARG),

    # ── step(f32, f32), smoothstep(f32, f32, f32) ──
    "step":        (_FLOAT_ARG, _FLOAT_ARG),
    "smoothstep":  (_FLOAT_ARG, _FLOAT_ARG, _FLOAT_ARG),

    # ── Vector functions ──
    "length":      (_VEC_ARG,),
    "length_sq":   (_VEC_ARG,),
    "normalize":   (_VEC_ARG,),
    "dot":         (_VEC_ARG, _VEC_ARG),
    "cross":       (_VEC_ARG, _VEC_ARG),
    "reflect":     (_VEC_ARG, _VEC_ARG),
    "distance":    (_VEC_ARG, _VEC_ARG),
    "distance_sq": (_VEC_ARG, _VEC_ARG),
    "lerp_vec":    (_VEC_ARG, _VEC_ARG, _FLOAT_ARG),

}
