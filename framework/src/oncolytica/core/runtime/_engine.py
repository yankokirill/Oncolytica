from __future__ import annotations

import inspect
from typing import Any, Optional, Type

from oncolytica.core.utils._types import (
    Tissue,
    Chemistry,
    Cell,
    P,
    Simulation,
)
from oncolytica.core.validation._validator import ValidatorEngine
from oncolytica.core.utils._errors import CompilationError
from oncolytica.core.runtime._backend import ISimulationBackend
from oncolytica.core.utils._containers import Grid, AgentList


class Engine:
    """
    Core orchestrator of the simulation.
    Holds state, handles user rules registration, and dispatches execution
    to the active backend (CPU or GPU).
    """

    def __init__(self, backend: str = "cpu", max_agents: int = 1024 * 256):
        if backend not in ("cpu", "gpu"):
            raise ValueError(f"Unknown backend {backend!r}. Choose 'cpu' (default) or 'gpu'.")

        self.backend: str = backend
        self.max_agents: int = max_agents

        # Geometry Defaults
        self.tissue_voxel_size: float = 1.4
        self.chemistry_voxel_size: float = 2.8
        self.cell_diameter: float = 1.0
        self.tissue_shape: Optional[tuple[int, int, int]] = (64, 64, 64)
        self.chemistry_shape: Optional[tuple[int, int, int]] = (32, 32, 32)

        # Inner Setup State
        self._step: int = 0
        self.cells: AgentList = AgentList(self)
        self.tissue: Optional[Grid] = None
        self.chemistry: Optional[Grid] = None
        self._metrics: Optional[P] = None

        # Simulation configuration
        self._sim: Optional[Simulation] = None
        self._tissue_rules: list = []
        self._chemistry_rules: list = []
        self._cell_rules: list = []
        self._metric_rules: list = []

        # Backend Implementation (CPU or GPU)
        self._backend_impl: Optional[ISimulationBackend] = None

    # -------------------------------------------------------------------------
    # State Initialization
    # -------------------------------------------------------------------------

    def setup_params(self, params: P) -> None:
        self._sim._params = params

    def update_params(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            self._sim._params._validate_field(name, value)
        for name, value in kwargs.items():
            object.__setattr__(self._sim._params, name, value)


    def setup_geometry(
            self,
            tissue_shape: tuple[int, int, int] = (120, 120, 120),
            tissue_voxel_size: float = 1.4,
            cell_diameter: float = 1.0
    ) -> None:
        self._validate_geometry(tissue_shape)
        self.tissue_shape = tissue_shape
        self.chemistry_shape = (tissue_shape[0] // 2, tissue_shape[1] // 2, tissue_shape[2] // 2)
        self.cell_diameter = cell_diameter
        self.tissue_voxel_size = tissue_voxel_size
        self.chemistry_voxel_size = 2 * tissue_voxel_size

    def _validate_geometry(self, tissue_shape: tuple[int, int, int]) -> None:
        if not isinstance(tissue_shape, (list, tuple)) or not (3 <= len(tissue_shape) <= 3):
            raise TypeError(f"tissue_shape must be a tuple of 3 ints, got {tissue_shape!r}")

        for i, dim in enumerate(tissue_shape):
            if not isinstance(dim, int):
                raise TypeError(f"tissue_shape component {i} must be int, got {type(dim).__name__!r}")
            if dim <= 0:
                raise ValueError(f"tissue_shape component {i} must be positive, got {dim}")
            if dim > 240:
                raise ValueError(f"tissue_shape component {i} must be ≤ 240, got {dim}")
            if dim % 2 != 0:
                raise ValueError(f"tissue_shape component {i} must be even, got {dim}")

    def setup_tissue(self, tissue_class: Type) -> None:
        if not issubclass(tissue_class, Tissue):
            raise ValueError(f"tissue_class must be a subclass of ol.TissueData, got {tissue_class}")
        self.tissue = Grid(tissue_class, self.tissue_shape, self)

    def setup_chemistry(self, chemistry_class: Type) -> None:
        if not issubclass(chemistry_class, Chemistry):
            raise ValueError(f"chemistry_class must be a subclass of ol.ChemistryData, got {chemistry_class}")
        self.chemistry = Grid(chemistry_class, self.chemistry_shape, self)

    @property
    def step(self) -> int:
        return self._step

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def load_model(self, sim_instance: Any) -> None:
        if not isinstance(sim_instance, Simulation):
            raise CompilationError(f"Expected an instance of 'ol.Simulation', got '{type(sim_instance).__name__}'.")

        self._sim = sim_instance
        self._metrics = sim_instance._spec.metrics_class(sim_instance)
        sim_instance._engine = self

        self._tissue_rules.clear()
        self._chemistry_rules.clear()
        self._cell_rules.clear()
        self._metric_rules.clear()

        # Parse user rules
        for name in dir(sim_instance):
            if name.startswith("_"): continue
            try:
                method = getattr(sim_instance, name)
            except Exception:
                continue

            if not callable(method): continue

            rule_type = getattr(method, "_rule_type", None)
            if rule_type == "tissue":
                self._tissue_rules.append(method)
            elif rule_type == "chemistry":
                self._chemistry_rules.append(method)
            elif rule_type == "cell":
                self._cell_rules.append(method)
            elif rule_type == "metric":
                self._metric_rules.append(method)

        # Validate memory structure
        ctx = ValidatorEngine().run(sim_instance)

        # Initialize requested backend
        if self.backend == "gpu":
            from oncolytica.gpu.runtime._gpu_backend import GpuBackend
            self._backend_impl = GpuBackend(self, ctx)
        else:
            # Импортируем наш новый отдельный CPUBackend
            from oncolytica.cpu._cpu_backend import CPUBackend
            self._backend_impl = CPUBackend(self)

        self._backend_impl.compile(sim_instance)

    # -------------------------------------------------------------------------
    # Execution Lifecycle
    # -------------------------------------------------------------------------

    def run_step(self, collect_metrics: bool = False) -> None:
        if self._sim is None or self._backend_impl is None:
            raise RuntimeError("Engine.run_step() called before load_model().")

        self._step += 1

        self._backend_impl.run_step(collect_metrics=collect_metrics)

    def get_metrics(self) -> Optional[P]:
        return self._metrics

    def sync_to_host(self):
        self._backend_impl.sync_to_host()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve_container_for(self, bound_method: Any) -> Any:
        try:
            sig = inspect.signature(bound_method)
            params = list(sig.parameters.values())
            if not params: return ()

            first_name = params[0].name
            annotation = params[0].annotation

            try:
                import typing
                hints = typing.get_type_hints(bound_method)
                annotation = hints.get(first_name, annotation)
            except Exception:
                pass

            if annotation is inspect.Parameter.empty:
                return ()

            if isinstance(annotation, type):
                if issubclass(annotation, Cell): return self.cells
                if issubclass(annotation, Chemistry): return self.chemistry if self.chemistry is not None else ()
                if issubclass(annotation, Tissue): return self.tissue if self.tissue is not None else ()
        except (ValueError, TypeError):
            pass
        return ()

    def __repr__(self) -> str:
        return f"Engine(backend={self.backend!r}, step={self._step}, cells={len(self.cells)})"