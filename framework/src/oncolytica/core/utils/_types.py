import random as _random
import math as _math
import copy as _copy

from typing import get_origin, get_args, get_type_hints
from typing import NewType, Any, Generator, TypeVar, Annotated, Generic
from types import SimpleNamespace

import builtins as _builtins
import sys as _sys
import dataclasses
from dataclasses import dataclass

# ── Generic types ──────────────────────────────────────────────────

S = TypeVar("S", bound="Simulation")
T = TypeVar("T", bound="Tissue")
C = TypeVar("C", bound="Chemistry")
A = TypeVar("A", bound="Cell")
M = TypeVar("M", bound="Metrics")
P = TypeVar("P", bound="Params")

U = TypeVar("U")

# ── Primitive type aliases ────────────────────────────────────────────────────

i32 = NewType('i32', int)
u32 = NewType('u32', int)
f32 = NewType('f32', float)
bool = _builtins.bool


def _resolve_own_hints(cls: type) -> dict:
    own_ann = cls.__dict__.get("__annotations__", {})
    if not own_ann: return {}
    module = _sys.modules.get(cls.__module__)
    globalns = vars(module) if module is not None else {}
    resolved = {}
    for fname, ftype in own_ann.items():
        if isinstance(ftype, str):
            try:
                resolved[fname] = eval(ftype, globalns)  # noqa: S307
            except Exception:
                resolved[fname] = ftype
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
        if name in self.__slots__:
            object.__setattr__(self, name, float(value))
        else:
            raise AttributeError(f"'vec3' object has no attribute '{name}'")

    def __class_getitem__(cls, _dtype: Any) -> type:
        return cls

    def copy(self) -> "vec3":
        """Explicitly create a new independent vector."""
        return vec3(self.x, self.y, self.z)

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


# ── ivec3 ─────────────────────────────────────────────────────────────────────

class ivec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: int = 0, y: int = 0, z: int = 0) -> None:
        object.__setattr__(self, "x", int(x))
        object.__setattr__(self, "y", int(y))
        object.__setattr__(self, "z", int(z))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, int(value))
        else:
            raise AttributeError(f"'ivec3' object has no attribute '{name}'")

    def __class_getitem__(cls, _dtype: Any) -> type:
        return cls

    def copy(self) -> "ivec3":
        """Explicitly create a new independent integer vector."""
        return ivec3(self.x, self.y, self.z)

    @staticmethod
    def _coerce(other: Any) -> "ivec3":
        if isinstance(other, ivec3):
            return other
        return NotImplemented

    def __add__(self, other: Any) -> "ivec3":
        if not isinstance(other, ivec3):
            return NotImplemented
        return ivec3(self.x + other.x, self.y + other.y, self.z + other.z)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "ivec3":
        if not isinstance(other, ivec3):
            return NotImplemented
        return ivec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Any) -> "ivec3":
        if isinstance(other, (int, float)):
            s = int(other)
            return ivec3(self.x * s, self.y * s, self.z * s)

        if isinstance(other, ivec3):
            return ivec3(self.x * other.x, self.y * other.y, self.z * other.z)

        return NotImplemented

    def __rmul__(self, other: Any) -> "ivec3":
        if isinstance(other, (int, float)):
            s = int(other)
            return ivec3(self.x * s, self.y * s, self.z * s)
        return NotImplemented

    def __neg__(self) -> "ivec3":
        return ivec3(-self.x, -self.y, -self.z)

    def __iadd__(self, other: Any) -> "ivec3":
        return self.__add__(other)

    def __isub__(self, other: Any) -> "ivec3":
        return self.__sub__(other)

    def __imul__(self, other: Any) -> "ivec3":
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ivec3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self) -> str:
        return f"ivec3({self.x:d}, {self.y:d}, {self.z:d})"


_FLOAT_TYPES = {_builtins.float, f32}
_VEC_TYPES = {vec3, ivec3}
_INT_TYPES = {_builtins.int, i32, u32}
_BOOL_TYPES = {_builtins.bool}
_ZERO: dict[type, Any] = {float: 0.0, int: 0, bool: False, vec3: vec3(), ivec3: ivec3()}

PRIMITIVE_TYPES = _FLOAT_TYPES | _VEC_TYPES | _INT_TYPES | _BOOL_TYPES


# ── Simulation ────────────────────────────────────────────────────────────────────

def _validate_tissue_grid_size(size: tuple) -> None:
    if not isinstance(size, (list, tuple)) or not (3 <= len(size) <= 3):
        raise TypeError(
            f"TISSUE_GRID_SIZE must be a tuple of 3 ints, got {size!r}"
        )

    for i, dim in enumerate(size):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(
                f"TISSUE_GRID_SIZE component {i} must be int, "
                f"got {type(dim).__name__!r} ({dim!r})"
            )
        if dim <= 0:
            raise ValueError(
                f"TISSUE_GRID_SIZE component {i} must be strictly positive, got {dim}"
            )
        if dim > 240:
            raise ValueError(
                f"TISSUE_GRID_SIZE component {i} must be ≤ 240, got {dim}"
            )
        if dim % 2 != 0:
            raise ValueError(
                f"TISSUE_GRID_SIZE component {i} must be even, got {dim}"
            )


class Simulation(Generic[T, C, A, M, P]):
    """Base class for user-defined simulation models."""
    engine: Any
    _params: P

    def __init__(self):
        self._params: P = self._spec.params_class()
        self._metrics: M = self._spec.metrics_class(self)
        self._params._validate()

    def __init_subclass__(cls, **kwargs):
        generic_base = None

        for base in getattr(cls, "__orig_bases__", []):
            if get_origin(base) is Simulation:
                generic_base = base
                break

        if generic_base is None:
            raise RuntimeError(
                "Explicit type specialization is required for Simulation class.\n"
                "Usage: class MySimulation(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):"
            )

        actual_types = get_args(generic_base)

        cls._spec = SimpleNamespace(
            tissue_class=actual_types[0],
            chemistry_class=actual_types[1],
            cell_class=actual_types[2],
            metrics_class=actual_types[3],
            params_class=actual_types[4],
        )

        for i, actual_class in enumerate(actual_types):
            base_class = BASE_CLASSES[i]
            for i, actual_class in enumerate(actual_types):
                base_class = BASE_CLASSES[i]
                # ForwardRef возникает когда строковые аннотации переданы до объявления классов.
                if isinstance(actual_class, str) or not isinstance(actual_class, type):
                    raise TypeError(
                        f"Parameter {i + 1} of ol.Simulation is a forward reference "
                        f"(got string '{actual_class}' instead of a class). "
                        f"Define all data classes before declaring the Simulation base, "
                        f"or pass the classes directly: "
                        f"class MySim(ol.Simulation[MyTissue, MyChemistry, MyCell, MyMetrics, MyParams])."
                    )
                if not issubclass(actual_class, BASE_CLASSES[i]):
                    raise TypeError(
                        f"Validation failed for parameter {i + 1} in ol.Simulation.\n"
                        f"The type '{actual_class.__name__}' must be a subclass of 'ol.{base_class.__name__}'.\n"
                        f"Example: MySimulation(ol.Simulation[MyTissue, MyChemistry, MyCell, MyMetrics, MyParams]).\n"
                    )

            if base_class is Cell:
                hints = get_type_hints(actual_class)
                if 'pos' not in hints or hints['pos'] is not vec3:
                    raise TypeError(
                        f"Class '{actual_class.__name__}' must have attribute 'pos: ol.vec3'."
                    )

    @property
    def params(self) -> P:
        return self._params

    # -------------------------------------------------------------------------
    # Built-in Methods for Rules (CPU Implementation)
    # -------------------------------------------------------------------------

    def tissue_at(self, pos: vec3) -> T:
        """
        Sample tissue data using absolute spatial coordinates.
        Index = floor(pos / tissue_voxel_size)
        """
        if self._engine is None or self._engine.tissue is None:
            raise RuntimeError("Simulation.sample_tissue() called but engine/tissue is not initialized.")

        v_size = self._engine.tissue_voxel_size
        shape = self._engine.tissue_shape

        ix = max(0, min(int(pos.x / v_size), shape[0] - 1))
        iy = max(0, min(int(pos.y / v_size), shape[1] - 1))
        iz = max(0, min(int(pos.z / v_size), shape[2] - 1))

        return self._engine.tissue.sample(ix, iy, iz)

    def chemistry_at(self, pos: vec3) -> C:
        """
        Sample chemistry data using absolute spatial coordinates.
        Index = floor(pos / (2 * tissue_voxel_size))
        """
        if self._engine is None or self._engine.chemistry is None:
            raise RuntimeError("Simulation.sample_chemistry() called but engine/chemistry is not initialized.")

        v_size = self._engine.chemistry_voxel_size
        shape = self._engine.chemistry_shape

        ix = max(0, min(int(pos.x / v_size), shape[0] - 1))
        iy = max(0, min(int(pos.y / v_size), shape[1] - 1))
        iz = max(0, min(int(pos.z / v_size), shape[2] - 1))

        return self._engine.chemistry.sample(ix, iy, iz)

    # -------------------------------------------------------------------------
    # Core Overrides
    # -------------------------------------------------------------------------

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name == "params":
            if not isinstance(value, self._spec.params_class):
                raise TypeError(
                    f"Invalid type for 'params'.\n"
                    f"Expected an instance of '{self._spec.params_class.__name__}', "
                    f"but got '{type(value).__name__}'."
                )
            self.params._validate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── BaseData ────────────────────────────────────────────────────────────────────

class BaseData(Generic[U]):
    _rng_state: u32
    _engine: Annotated[Any, "cpu_only"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        hints = get_type_hints(cls)
        defaults = {n: getattr(cls, n) for n in hints if hasattr(cls, n)}

        def __init__(self, **kwargs_init):
            for name, dtype in hints.items():
                if name in kwargs_init:
                    val = kwargs_init[name]
                elif name in defaults:
                    val = defaults[name]
                else:
                    if dtype in _FLOAT_TYPES:
                        val = 0.0
                    elif dtype in _INT_TYPES:
                        val = 0
                    elif dtype in _BOOL_TYPES:
                        val = False
                    elif dtype is vec3:
                        val = vec3()
                    elif dtype is ivec3:
                        val = ivec3()
                    else:
                        val = None
                setattr(self, name, val)

        cls.__init__ = __init__

    def copy(self) -> Any:
        new_obj = _copy.copy(self)

        for name, val in new_obj.__dict__.items():
            if isinstance(val, (vec3, ivec3)):
                new_obj.__dict__[name] = val.copy()

        return new_obj

    def copy_from(self, other: U) -> None:
        hints = get_type_hints(type(self))

        for name in hints:
            if name.startswith("_"):
                continue

            value = getattr(other, name)

            if isinstance(value, (vec3, ivec3)):
                setattr(self, name, value.copy())
            else:
                setattr(self, name, value)

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

class Tissue(BaseData, Generic[S, T, C, A, M, P]):
    """
    All fields in a subclass of ol.Tissue must be explicitly annotated.
    """

    _coord: ivec3
    _engine: Annotated[Any, "cpu_only"]

    @property
    def sim(self) -> Simulation[T, C, A, M, P]:
        return self._engine._sim

    @property
    def coord(self) -> ivec3:
        return self._coord

    @property
    def pos(self) -> vec3:
        voxel_size = self._grid._engine.tissue_voxel_size
        return self._coord * voxel_size + voxel_size / 2

    @property
    def neighbors(self) -> Generator[T, None, None]:
        x, y, z = self._coord
        yield from _moore_neighbors(self._grid, x, y, z)

    @property
    def cells(self) -> Generator[A, None, None]:
        backend = self._grid._engine._backend_impl
        yield from backend.get_cells_in_voxel(self._coord.x, self._coord.y, self._coord.z)


class Chemistry(BaseData, Generic[S, T, C, A, M, P]):
    """
    All fields in a subclass of ol.Chemistry must be explicitly annotated.
    """

    _coord: ivec3
    _engine: Annotated[Any, "cpu_only"]

    @property
    def sim(self) -> Simulation[T, C, A, M, P]:
        return self._engine._sim

    @property
    def coord(self) -> ivec3:
        return self._coord

    @property
    def pos(self) -> vec3:
        voxel_size = self._grid._engine.chemistry_voxel_size
        return self._coord * voxel_size + voxel_size / 2

    @property
    def neighbors(self) -> Generator[C, None, None]:
        yield from _moore_neighbors(self._grid, self._coord.x, self._coord.y, self._coord.z)

    @property
    def tissues(self) -> Generator[T, None, None]:
        tissue_grid = self._grid._engine.tissue
        if tissue_grid is None:
            return

        shape = tissue_grid.shape
        max_x = shape[0] - 1
        max_y = shape[1] - 1
        max_z = shape[2] - 1

        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    tx = 2 * self._coord.x + dx
                    ty = 2 * self._coord.y + dy
                    tz = 2 * self._coord.z + dz
                    if tx <= max_x and ty <= max_y and tz <= max_z:
                        yield tissue_grid[tx, ty, tz]

    @property
    def cells(self) -> Generator[A, None, None]:
        for t in self.tissues:
            yield from t.cells


class Cell(BaseData, Generic[S, T, C, A, M, P]):
    """
    All fields in a subclass of ol.Cell must be explicitly annotated.
    """

    _alive: Annotated[bool, "cpu_only"] = False
    _agent_list: Annotated[Any, "cpu_only"] = None

    @property
    def sim(self) -> Simulation[T, C, A, M, P]:
        return self._agent_list._engine._sim

    @property
    def neighbors(self) -> Generator[A, None, None]:

        pos = getattr(self, "pos", None)
        if pos is None:
            return

        engine = self._agent_list._engine
        backend = engine._backend_impl
        v_size = engine.tissue_voxel_size
        shape = engine.tissue.shape

        cx = int(pos.x / v_size)
        cy = int(pos.y / v_size)
        cz = int(pos.z / v_size)

        max_x = shape[0] - 1 if len(shape) > 0 else 0
        max_y = shape[1] - 1 if len(shape) > 1 else 0
        max_z = shape[2] - 1 if len(shape) > 2 else 0

        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx = max(0, min(cx + dx, max_x))
                    ny = max(0, min(cy + dy, max_y))
                    nz = max(0, min(cz + dz, max_z))
                    for nb in backend.get_cells_in_voxel(nx, ny, nz):
                        if nb is not self:
                            yield nb

    def divide(self, cell: A) -> None:
        self._agent_list.add(cell)

    def die(self):
        self._alive = False

# ── Metrics ───────────────────────────────────────────────────────────────

class Metrics(Generic[S, T, C, A, M, P]):
    """
    All fields in a subclass of ol.Metrics must be explicitly annotated.
    """

    total_alive: int
    _sim: Annotated[Simulation[T, C, A, M, P], "cpu_only"]

    def __init__(self, sim: Simulation[T, C, A, M, P]):
        self._sim = sim

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        hints = _resolve_own_hints(cls)
        for fname, ftype in hints.items():
            if fname.startswith("_") or fname in cls.__dict__: continue
            if ftype in _FLOAT_TYPES:
                setattr(cls, fname, 0.0)
            elif ftype in _VEC_TYPES:
                setattr(cls, fname, vec3(0.0, 0.0, 0.0))
            elif ftype in _INT_TYPES:
                setattr(cls, fname, 0)
            elif ftype in _BOOL_TYPES:
                setattr(cls, fname, False)

    @property
    def sim(self) -> Simulation[T, C, A, M, P]:
        return self._sim

    def _clear(self) -> None:
        try:
            for f in dataclasses.fields(self):  # type: ignore[arg-type]
                val = getattr(self, f.name)
                if isinstance(val, float):
                    object.__setattr__(self, f.name, 0.0)
                elif isinstance(val, int):
                    object.__setattr__(self, f.name, 0)
                elif isinstance(val, _builtins.bool):
                    object.__setattr__(self, f.name, False)
                elif isinstance(val, vec3):
                    val.x = val.y = val.z = 0.0
        except TypeError:
            pass

    def __repr__(self) -> str:
        try:
            fields_str = ", ".join(
                f"{f.name}={getattr(self, f.name)!r}" for f in dataclasses.fields(self))  # type: ignore[arg-type]
            return f"{self.__class__.__name__}({fields_str})"
        except TypeError:
            return f"{self.__class__.__name__}()"


# ── Parameters ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, init=False)
class Params:
    """Base class for simulation parameters."""

    def __init__(self, **kwargs: Any) -> None:
        hints = get_type_hints(type(self))
        for arg in kwargs:
            if arg not in hints:
                raise TypeError(f"Missing field '{arg}' in {type(self).__name__}")

        for name, annotation in hints.items():
            if name in kwargs:
                val = kwargs[name]
            elif hasattr(type(self), name):
                val = getattr(type(self), name)
            elif annotation in _ZERO:
                val = _ZERO[annotation]
            else:
                raise TypeError(f"Missing value for field '{name}' in {type(self).__name__}")
            object.__setattr__(self, name, val)
        self._validate()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True, init=False)(cls)

        unannotated = {
            k for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
               and k not in getattr(cls, "__annotations__", {})
        }
        if unannotated:
            raise TypeError(f"All fields in {cls.__name__} must be annotated\n"
                            f"Unannotated: {", ".join(list(unannotated))}")

        cls()._validate()

    def _validate_field(self, name: str, val: Any) -> None:
        hints = get_type_hints(type(self))
        if name not in hints:
            raise TypeError(f"Missing field '{name}' in {type(self).__name__}")
        typ = hints[name]

        if typ in _FLOAT_TYPES and not isinstance(val, float) \
                or typ in _INT_TYPES and (not isinstance(val, int) or isinstance(val, bool)) \
                or typ in _BOOL_TYPES and not isinstance(val, bool) \
                or typ in _VEC_TYPES and not isinstance(val, typ):
            raise TypeError(
                f"Type mismatch for '{name}' in {type(self).__name__}: "
                f"expected {typ.__name__}, got {type(val).__name__} ({val!r})"
            )

    def _validate(self) -> None:
        hints = get_type_hints(type(self))
        for name, typ in hints.items():
            if typ not in (_FLOAT_TYPES | _INT_TYPES | _BOOL_TYPES | _VEC_TYPES):
                raise TypeError(f"Unsupported type '{typ}' for '{name}' in {type(self).__name__}")
            if not hasattr(self, name):
                continue
            val = getattr(self, name)
            self._validate_field(name, val)


# ── Module-level RNG functions (ol.random / ol.random_dir) ───────────────────

def random() -> float:
    return _random.random()


def random_dir() -> "vec3":
    phi = random() * _math.tau
    costheta = random() * 2.0 - 1.0
    theta = _math.acos(costheta)
    sin_theta = _math.sin(theta)
    return vec3(sin_theta * _math.cos(phi), sin_theta * _math.sin(phi), costheta)


BASE_CLASSES = [Tissue, Chemistry, Cell, Metrics, Params]
ANNOTATION_NAMES = {
    "i32": i32,
    "u32": u32,
    "f32": f32,
    "bool": bool,
    "vec3": vec3,
    "ivec3": ivec3,
    "int": int,
    "float": float,
}
