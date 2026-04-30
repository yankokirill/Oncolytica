"""oncolytica._simulation — Simulation base class."""

from __future__ import annotations
from typing import Generic
from typing import get_origin, get_args, get_type_hints
from types import SimpleNamespace
from oncolytica.core._types import Tissue, Chemistry, Cell, Metrics, vec3
from oncolytica.core._errors import CompilationError
from oncolytica.core._types import CellData


def _validate_tissue_grid_size(size: tuple) -> None:
    """Validate a TISSUE_GRID_SIZE value.

    Rules
    -----
    * Must be a tuple/list of 1, 2, or 3 elements (1D, 2D, or 3D).
    * Every component must be a plain ``int``.
    * Every component must be strictly positive (≥ 1).

    Note: The GPU backend will automatically pad these dimensions
    to the next power of two for optimal Z-order curve mapping.

    Raises
    ------
    TypeError
        If the value is not a sequence of length 1-3, or a component is not int.
    ValueError
        If a component is zero or negative.
    """
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

class Simulation(Generic[Tissue, Chemistry, Cell, Metrics]):
    """Base class for user-defined simulation models.

    Serves as a container for **hyperparameters** (called *uniforms* in
    shader language parlance) and **rule methods** decorated with
    ``@ol.tissue_rule``, ``@ol.chemistry_rule``, ``@ol.cell_rule``, or
    ``@ol.metric_rule``.

    The ``Engine`` discovers rules automatically when ``load_model()`` is
    called — no explicit registration is needed.

    Special parameter
    -----------------
    ``TISSUE_GRID_SIZE : tuple[int, ...]``
        Optional grid dimensions (X, Y, Z). Can be 2D, or 3D.
    """

    def __init__(self):
        instance_class = type(self)
        generic_base = None

        for base in getattr(instance_class, "__orig_bases__", []):
            if get_origin(base) is Simulation:
                generic_base = base
                break

        if generic_base is None:
            raise CompilationError(
                "Explicit type specialization is required for Simulation class.\n"
                "Usage: class MySimulation(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):"
            )

        actual_types = get_args(generic_base)
        expected_vars = Simulation.__parameters__

        self.spec = SimpleNamespace(
            tissue_class=actual_types[0],
            chemistry_class=actual_types[1],
            cell_class=actual_types[2],
            metrics_class=actual_types[3]
        )

        for i, (actual, expected_var) in enumerate(zip(actual_types, expected_vars)):
            bound = expected_var.__bound__

            if not issubclass(actual, bound):
                raise CompilationError(
                    f"Validation failed for parameter {i + 1} in ol.Simulation ({expected_var.__name__}).\n"
                    f"The type '{actual.__name__}' must be a subclass of '{bound.__name__}'.\n"
                    "Usage: class MySimulation(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):"
                )

            if bound is CellData:
                hints = get_type_hints(actual)
                if 'pos' not in hints or hints['pos'] is not vec3:
                    raise CompilationError(
                        f"Class '{actual.__name__}' must have attribute 'pos: ol.vec3'.\n"
                        f"Please add the type hint to your cell class definition."
                    )

    def __setattr__(self, name: str, value: object) -> None:
        if name == "TISSUE_GRID_SIZE":
            _validate_tissue_grid_size(value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"