# oncolytica/gpu/_memory.py
from __future__ import annotations
import struct
from typing import Any, Optional
from oncolytica.core._types import _resolve_own_hints, vec3
from ._type_system import py_type_to_wgsl, wgsl_alignment


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
                if fname.startswith("_") or fname in seen: continue
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
