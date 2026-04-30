"""oncolytica._types — core data types."""

from __future__ import annotations

import random as _random
import math as _math

from typing import NewType, Any, Generator, TypeVar, get_type_hints
import builtins as _builtins
import dataclasses
import sys as _sys
from dataclasses import dataclass, field as _dc_field

# ── Primitive type aliases ────────────────────────────────────────────────────

i32 = NewType('i32', int)
i64 = NewType('i64', int)
u32 = NewType('u32', int)
u64 = NewType('u64', int)
f32 = NewType('f32', float)
f64 = NewType('f64', float)
bool = _builtins.bool

def _resolve_own_hints(cls: type) -> dict:
    own_ann = cls.__dict__.get("__annotations__", {})
    if not own_ann: return {}
    module  = _sys.modules.get(cls.__module__)
    globalns = vars(module) if module is not None else {}
    resolved = {}
    for fname, ftype in own_ann.items():
        if isinstance(ftype, str):
            try: resolved[fname] = eval(ftype, globalns)   # noqa: S307
            except Exception: resolved[fname] = ftype
        else:
            resolved[fname] = ftype
    return resolved

# ── vec3 ──────────────────────────────────────────────────────────────────────

class vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("vec3 is immutable. Create a new instance instead.")

    def __class_getitem__(cls, _dtype: Any) -> type:
        return cls

    @staticmethod
    def _coerce(other: Any) -> "vec3":
        if isinstance(other, vec3):
            return other
        return NotImplemented

    def __add__(self, other: Any) -> "vec3":
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "vec3":
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Any) -> "vec3":
        if isinstance(other, (float, int)):
            s = float(other)
            return vec3(self.x * s, self.y * s, self.z * s)

        if isinstance(other, vec3):
            return vec3(self.x * other.x, self.y * other.y, self.z * other.z)

        return NotImplemented

    def __rmul__(self, other: Any) -> "vec3":
        if isinstance(other, (float, int)):
            s = float(other)
            return vec3(self.x * s, self.y * s, self.z * s)
        return NotImplemented

    def __neg__(self) -> "vec3":
        return vec3(-self.x, -self.y, -self.z)

    def __iadd__(self, other: Any) -> "vec3":
        return self.__add__(other)

    def __isub__(self, other: Any) -> "vec3":
        return self.__sub__(other)

    def __imul__(self, other: Any) -> "vec3":
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, vec3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self) -> str:
        return f"vec3({self.x:.4f}f, {self.y:.4f}f, {self.z:.4f}f)"


_FLOAT_TYPES = {_builtins.float, f32, f64}
_VEC_TYPES = {vec3}
_INT_TYPES = {_builtins.int, i32, i64, u32, u64}
_BOOL_TYPES = {_builtins.bool}

# ── BaseData ────────────────────────────────────────────────────────────────────

class BaseData:
    _rng_state: u32

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        hints = get_type_hints(cls)
        defaults = {n: getattr(cls, n) for n in hints if hasattr(cls, n)}

        def __init__(self, **kwargs_init):
            for name, dtype in hints.items():
                if name in kwargs_init: val = kwargs_init[name]
                elif name in defaults: val = defaults[name]
                else:
                    if dtype in _FLOAT_TYPES: val = 0.0
                    elif dtype in _INT_TYPES: val = 0
                    elif dtype in _BOOL_TYPES: val = False
                    elif dtype in _VEC_TYPES: val = vec3()
                    else: val = None
                setattr(self, name, val)
        cls.__init__ = __init__

# ── Spatial Helpers ──────────────────────────────────────────────────────────

def _moore_neighbors(grid: Any, cx: int, cy: int, cz: int) -> Generator[Any, None, None]:
    if grid is None: return
    shape = grid.shape
    max_x = shape[0] - 1 if len(shape) > 0 else 0
    max_y = shape[1] - 1 if len(shape) > 1 else 0
    max_z = shape[2] - 1 if len(shape) > 2 else 0

    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0: continue
                nx = max(0, min(cx + dx, max_x))
                ny = max(0, min(cy + dy, max_y))
                nz = max(0, min(cz + dz, max_z))

                idx = []
                if len(shape) > 0: idx.append(nx)
                if len(shape) > 1: idx.append(ny)
                if len(shape) > 2: idx.append(nz)
                yield grid[tuple(idx)]

# ── Data Layout base classes ──────────────────────────────────────────────────

class TissueData(BaseData):
    @property
    def neighbors(self) -> Generator[Any, None, None]:
        grid = getattr(self, "_grid", None)
        yield from _moore_neighbors(grid, getattr(self, "_x", 0), getattr(self, "_y", 0), getattr(self, "_z", 0))

    @property
    def cells(self) -> Generator[Any, None, None]:
        from oncolytica.core._engine import _get_active
        engine = _get_active()
        if engine is not None:
            yield from engine._get_cells_in_voxel(getattr(self, "_x", 0), getattr(self, "_y", 0), getattr(self, "_z", 0))


class ChemistryData(BaseData):
    @property
    def neighbors(self) -> Generator[Any, None, None]:
        grid = getattr(self, "_grid", None)
        yield from _moore_neighbors(grid, getattr(self, "_x", 0), getattr(self, "_y", 0), getattr(self, "_z", 0))

    @property
    def tissues(self) -> Generator[Any, None, None]:
        from oncolytica.core._engine import _get_active
        engine = _get_active()
        if engine is None or engine.tissue is None: return
        t_grid = engine.tissue
        shape = t_grid.shape
        tx, ty, tz = getattr(self, "_x", 0) * 2, getattr(self, "_y", 0) * 2, getattr(self, "_z", 0) * 2
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    nx, ny, nz = tx + dx, ty + dy, tz + dz
                    if nx < shape[0] and ny < shape[1] and nz < shape[2]:
                        yield t_grid[nx, ny, nz]

    @property
    def cells(self) -> Generator[Any, None, None]:
        for t in self.tissues:
            yield from t.cells


class CellData(BaseData):
    _alive = False
    _agentList: Any = None

    @property
    def neighbors(self) -> Generator[Any, None, None]:
        pass

    def divide(self, cell: CellData) -> None:
        self._agentList.add(cell)

    def die(self):
        self._alive = False


# ── MetricsData ───────────────────────────────────────────────────────────────

class MetricsData:
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        hints = _resolve_own_hints(cls)
        for fname, ftype in hints.items():
            if fname.startswith("_") or fname in cls.__dict__: continue
            if ftype in _FLOAT_TYPES: setattr(cls, fname, 0.0)
            elif ftype in _VEC_TYPES: setattr(cls, fname, vec3(0.0, 0.0, 0.0))
            elif ftype in _INT_TYPES: setattr(cls, fname, 0)
            elif ftype in _BOOL_TYPES: setattr(cls, fname, False)
        dataclass(cls)

    def clear(self) -> None:
        try:
            for f in dataclasses.fields(self):  # type: ignore[arg-type]
                val = getattr(self, f.name)
                if isinstance(val, float): object.__setattr__(self, f.name, 0.0)
                elif isinstance(val, int): object.__setattr__(self, f.name, 0)
                elif isinstance(val, _builtins.bool): object.__setattr__(self, f.name, False)
                elif isinstance(val, vec3): val.x = val.y = val.z = 0.0
        except TypeError: pass

    def __repr__(self) -> str:
        try:
            fields_str = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in dataclasses.fields(self)) # type: ignore[arg-type]
            return f"{self.__class__.__name__}({fields_str})"
        except TypeError: return f"{self.__class__.__name__}()"

# ── Module-level RNG functions (ol.random / ol.random_dir) ───────────────────

def random() -> float:
    return _random.random()


def random_dir() -> "vec3":
    phi = random() * _math.tau
    costheta = random() * 2.0 - 1.0
    theta = _math.acos(costheta)
    sin_theta = _math.sin(theta)
    return vec3(sin_theta * _math.cos(phi), sin_theta * _math.sin(phi), costheta)


# ── Generic types ──────────────────────────────────────────────────
Tissue = TypeVar("Tissue", bound=TissueData)
Chemistry = TypeVar("Chemistry", bound=ChemistryData)
Cell = TypeVar("Cell", bound=CellData)
Metrics = TypeVar("Metrics", bound=MetricsData)
