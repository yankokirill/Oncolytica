"""oncolytica.gpu._type_system — Python/oncolytica → WGSL type mapping."""

from __future__ import annotations

import builtins
from typing import Any, Optional

from oncolytica.core.utils._types import f32, i32, u32, vec3, ivec3
from oncolytica.core.utils._types import BASE_CLASSES, Tissue, Chemistry, Cell, Metrics

# ── Primary type map: Python type → WGSL type string ──────────────────────────
# ``i32``, ``u32``, etc. are now distinct sentinel subclasses (not plain
# ``int`` aliases) so there are no duplicate dict keys.

_MAP: dict[Any, str] = {
    float:         "f32",
    f32:           "f32",
    int:           "i32",
    i32:           "i32",
    u32:           "u32",
    builtins.bool: "i32",
    vec3:          "vec3<f32>",
    ivec3:         "vec3<i32>",
    Tissue:        "Tissue",
    Chemistry:     "Chemistry",
    Cell:          "Cell",
    Metrics:       "Metrics",
}

# ── (size_bytes, alignment_bytes) for WGSL types ──────────────────────────────
# vec3<f32> has 12 bytes of data but 16-byte alignment in structs (leaves a
# 4-byte hole that we fill with the next scalar field or a hidden _pad).

_ALIGN: dict[str, tuple[int, int]] = {
    "f32":        (4, 4),
    "i32":        (4, 4),
    "u32":        (4, 4),
    "vec3<f32>":  (12, 16),
    "vec3<u32>":  (12, 16),
    "vec3<i32>":  (12, 16),
    "vec4<f32>":  (16, 16),
}

# ── Zero initialiser literals ─────────────────────────────────────────────────

_ZERO: dict[str, str] = {
    "f32":        "0.0",
    "i32":        "0",
    "u32":        "0u",
    "bool":       "false",
    "vec3<f32>":  "vec3<f32>(0.0, 0.0, 0.0)",
}

# ── WGSL literal suffix for integer types ─────────────────────────────────────

_SUFFIX: dict[str, str] = {
    "i32": "",
    "u32": "u",
    "f32": "",
}


def domain_base_of(cls: type) -> Optional[type]:
    """Return the framework base class (Cell/Tissue/Chemistry/Metrics) for a user class."""
    if not isinstance(cls, type):
        return None
    for base in BASE_CLASSES:
        if issubclass(cls, base):
            return base
    return None


def base_class_of(cls: type) -> type:
    if not isinstance(cls, type):
        return cls
    for base in BASE_CLASSES:
        if issubclass(cls, base):
            return base
    return cls


def py_type_to_wgsl(py_type: type) -> str:
    """Convert a Python/oncolytica type annotation to its WGSL type string.

    Raises ``TypeError`` if the type has no registered mapping.
    """
    wt = _MAP.get(base_class_of(py_type))
    if wt is None:
        raise TypeError(f"No WGSL mapping for Python type {py_type!r}")
    return wt


def wgsl_zero(wgsl_type: str) -> str:
    """Return a WGSL zero-value literal for *wgsl_type*."""
    return _ZERO.get(wgsl_type, "0")


def wgsl_alignment(wgsl_type: str) -> tuple[int, int]:
    """Return ``(size_bytes, align_bytes)`` for *wgsl_type*."""
    return _ALIGN.get(wgsl_type, (4, 4))


def is_float_type(wgsl_type: str) -> bool:
    return wgsl_type in ("f32", "vec3<f32>", "vec4<f32>")


def is_int_type(wgsl_type: str) -> bool:
    return wgsl_type in ("i32", "u32")


def wgsl_cast(value: str, from_type: str, to_type: str) -> str:
    """Emit an explicit cast expression if the types differ."""
    if from_type == to_type:
        return value
    if to_type == "f32" and from_type in ("i32", "u32"):
        return f"f32({value})"
    if to_type == "i32" and from_type == "f32":
        return f"i32({value})"
    if to_type == "u32" and from_type in ("i32", "f32"):
        return f"u32({value})"
    return value


def infer_literal_wgsl_type(node) -> str | None:
    """Infer the WGSL type of an ``ast.Constant`` node, or return *None*."""
    import ast
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "u32"
        if isinstance(node.value, int):
            return "i32"
        if isinstance(node.value, float):
            return "f32"
    return None


def format_float_literal(value: float) -> str:
    """Format a Python float as a valid WGSL f32 literal."""
    s = repr(value)
    if "." not in s and "e" not in s:
        s = s + ".0"
    return s


def format_int_literal(value: int, wgsl_type: str = "i32") -> str:
    """Format a Python int as a WGSL integer literal."""
    if wgsl_type == "u32":
        return f"{value}u"
    return str(value)
