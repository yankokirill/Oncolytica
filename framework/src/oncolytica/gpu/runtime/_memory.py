from __future__ import annotations
import struct
from typing import Any, Optional
from oncolytica.core.utils._types import _resolve_own_hints, vec3, ivec3
from oncolytica.gpu.compiler._type_system import py_type_to_wgsl, wgsl_alignment


_RNG_STATE_FIELD = ("_rng_state", "u32")
_COORD_FIELD = ("_coord", "vec3<i32>")


def optimize_struct_layout(fields: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Packs WGSL struct fields to minimize padding.
    vec3 occupies 12 bytes but aligns to 16, leaving a 4-byte "hole".
    This function greedily places 4-byte scalars immediately after vec3.
    """
    vec4s = []
    vec3s = []
    vec2s = []
    scalars = []

    for name, ftype in fields:
        # Strip whitespace if present for accurate comparison
        clean_type = ftype.strip()
        if clean_type.startswith("vec4"):
            vec4s.append((name, clean_type))
        elif clean_type.startswith("vec3"):
            vec3s.append((name, clean_type))
        elif clean_type.startswith("vec2"):
            vec2s.append((name, clean_type))
        else:
            # Treat everything else (f32, u32, i32) as 4-byte scalars
            scalars.append((name, clean_type))

    packed = []

    # 1. Add vec4 first (perfectly fits 16 bytes)
    packed.extend(vec4s)

    # 2. Add vec3 and immediately fill the 4-byte hole with a scalar
    for v3 in vec3s:
        packed.append(v3)
        if scalars:
            packed.append(scalars.pop(0)) # Take the first available scalar (often _rng_state)

    # 3. Add vec2 (8 bytes each)
    packed.extend(vec2s)

    # 4. Add remaining 4-byte scalars
    packed.extend(scalars)

    return packed


def wgsl_struct(name: str, packed: list[tuple[str, str]]) -> str:
    """Generates WGSL struct definition with optimized field layout."""
    lines = [f"struct {name} {{"]
    for fname, ftype in packed:
        lines.append(f"    {fname}: {ftype},")
    lines.append("};")
    return "\n".join(lines)


class StructSerializer:
    """Provides C-compatible memory layout for WebGPU."""

    def __init__(self, name: str, cls: Optional[type], is_atomic: bool = False):
        self.name = name
        self.fields: list[tuple[str, str]] = []
        self.offsets: dict[str, int] = {}
        self.stride: int = 0
        self.is_atomic = is_atomic

        if cls is not None:
            self._build_layout(cls)

    def _build_layout(self, cls: type) -> None:
        raw_fields = []
        seen = set()
        for base in reversed(cls.__mro__):
            if base is object: continue
            for fname, ftype in _resolve_own_hints(base).items():
                if getattr(ftype, "__metadata__", [None])[0] == "cpu_only": continue
                try:
                    wgsl_t = py_type_to_wgsl(ftype)
                    if self.is_atomic and wgsl_t in ("i32", "u32"):
                        wgsl_t = f"atomic<{wgsl_t}>"
                    raw_fields.append((fname, wgsl_t))
                    seen.add(fname)
                except TypeError: pass

        offset = 0
        max_align = 4
        for fname, wtype in raw_fields:
            sz, al = wgsl_alignment(wtype)
            max_align = max(max_align, al)
            # Align current field
            if offset % al != 0:
                offset += al - (offset % al)

            self.fields.append((fname, wtype))
            self.offsets[fname] = offset
            offset += sz

        # Align total structure size (stride)
        if offset % max_align != 0:
            offset += max_align - (offset % max_align)
        self.stride = max(offset, 16)

    def generate_wgsl_struct(self) -> str:
        """Generates WGSL struct definition."""
        if not self.fields: return ""
        lines = [f"struct {self.name} {{"]
        for fname, ftype in self.fields:
            lines.append(f"    {fname}: {ftype},")
        lines.append("};")
        return "\n".join(lines)

    def pack(self, obj: Any) -> bytes:
        """Packs a Python object into bytes according to the defined layout."""
        buf = bytearray(self.stride)
        for fname, ftype in self.fields:
            off = self.offsets[fname]
            if ftype == "vec3<f32>":
                v = getattr(obj, fname, None)
                if v is None:
                    struct.pack_into("3f", buf, off, 0.0, 0.0, 0.0)
                else:
                    struct.pack_into("3f", buf, off, v.x, v.y, v.z)
            elif ftype == "vec3<i32>":
                v = getattr(obj, fname, None)
                if v is None:
                    struct.pack_into("3i", buf, off, 0, 0, 0)
                else:
                    struct.pack_into("3i", buf, off, int(v.x), int(v.y), int(v.z))
            elif ftype == "f32":
                struct.pack_into("f", buf, off, float(getattr(obj, fname, 0.0)))
            elif ftype in ("i32", "u32", "bool", "atomic<i32>", "atomic<u32>"):
                fmt = "i" if ftype in ("i32", "atomic<i32>") else "I"
                struct.pack_into(fmt, buf, off, int(getattr(obj, fname, 0)))
        return bytes(buf)

    def unpack(self, raw: bytes, base: int, obj: Any) -> None:
        """Unpacks bytes into a Python object at the specified base offset."""
        for fname, ftype in self.fields:
            off = base + self.offsets[fname]
            if ftype == "vec3<f32>":
                x, y, z = struct.unpack_from("3f", raw, off)
                setattr(obj, fname, vec3(x, y, z))
            elif ftype == "vec3<i32>":
                x, y, z = struct.unpack_from("3i", raw, off)
                setattr(obj, fname, ivec3(x, y, z))
            elif ftype == "f32":
                (v,) = struct.unpack_from("f", raw, off)
                setattr(obj, fname, v)
            elif ftype in ("i32", "u32", "bool", "atomic<i32>", "atomic<u32>"):
                fmt = "i" if ftype in ("i32", "atomic<i32>") else "I"
                (v,) = struct.unpack_from(fmt, raw, off)
                setattr(obj, fname, bool(v) if ftype == "bool" else v)

    def update_layout(self, optimized_fields: list[tuple[str, str]]) -> None:
        """Rebuilds offsets and stride from an already-optimized field list (e.g. from ShaderBuilder).
        Uses the same wgsl_alignment logic as _build_layout — no manual type parsing."""
        self.fields = optimized_fields
        self.offsets = {}

        offset = 0
        max_align = 4
        for fname, wtype in self.fields:
            sz, al = wgsl_alignment(wtype)
            max_align = max(max_align, al)
            if offset % al != 0:
                offset += al - (offset % al)
            self.offsets[fname] = offset
            offset += sz

        if offset % max_align != 0:
            offset += max_align - (offset % max_align)
        self.stride = max(offset, 16)
