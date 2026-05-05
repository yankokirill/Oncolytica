from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any, Generator, Optional

# Context for the active backend in CPU rules (safe for async/threads)
_active_backend: ContextVar[Optional["CPUBackend"]] = ContextVar(
    "_active_backend", default=None
)


def get_active_backend() -> Optional["CPUBackend"]:
    return _active_backend.get()


class CPUBackend:
    """
    Implements ISimulationBackend for CPU.
    Stores a reference to the Engine only for accessing data (cells, tissue, etc.).
    All computational logic, loops, and spatial hashing are encapsulated here.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._spatial_grid: dict[tuple[int, int, int], list[int]] = {}

    # ------------------------------------------------------------------
    # ISimulationBackend
    # ------------------------------------------------------------------

    def compile(self, sim_instance: Any) -> None:
        """CPU backend requires no compilation."""
        pass

    def run_step(self, collect_metrics: bool = False) -> None:
        self._build_spatial_grid()
        engine = self._engine

        # Clean up dead agents before processing
        engine.cells._remove_dead()

        # Set this backend as active for ol.neighbors() queries
        token = _active_backend.set(self)
        try:
            if engine._cell_rules:
                for rule in engine._cell_rules:
                    for cell in engine.cells:
                        rule(cell)
        finally:
            _active_backend.reset(token)

        if engine._tissue_rules and engine.tissue is not None:
            for rule in engine._tissue_rules:
                for voxel in engine.tissue:
                    rule(voxel)

        if engine._chemistry_rules and engine.chemistry is not None:
            for rule in engine._chemistry_rules:
                iters = getattr(rule, "_iterations", 1)
                for _ in range(iters):
                    for voxel in engine.chemistry:
                        rule(voxel)

        if collect_metrics and engine._metric_rules and engine._metrics is not None:
            engine._metrics._clear()
            for rule in engine._metric_rules:
                container = engine._resolve_container_for(rule)
                for item in container:
                    rule(item, engine._metrics)

    def sync_to_host(self) -> None:
        pass  # Data is already on host memory

    def sync_to_device(self) -> None:
        pass  # No device exists

    def get_metrics(self) -> Any:
        return self._engine._metrics

    # ------------------------------------------------------------------
    # Spatial queries (called from CPU rules via ol.neighbors)
    # ------------------------------------------------------------------

    def query_neighbors(self, agent: Any, radius: float) -> Generator[Any, None, None]:
        ts = self._engine.tissue_voxel_size
        cx = int(math.floor(agent.pos.x / ts))
        cy = int(math.floor(agent.pos.y / ts))
        cz = int(math.floor(agent.pos.z / ts))
        r_sq = radius * radius

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = self._spatial_grid.get((cx + dx, cy + dy, cz + dz))
                    if bucket is None:
                        continue
                    for idx in bucket:
                        nb = self._engine.cells._data[idx]
                        if not getattr(nb, "_alive", True) or nb is agent:
                            continue
                        dist_sq = (
                                (nb.pos.x - agent.pos.x) ** 2
                                + (nb.pos.y - agent.pos.y) ** 2
                                + (nb.pos.z - agent.pos.z) ** 2
                        )
                        if dist_sq <= r_sq:
                            yield nb

    def get_cells_in_voxel(self, x: int, y: int, z: int) -> Generator[Any, None, None]:
        bucket = self._spatial_grid.get((x, y, z))
        if bucket is not None:
            for idx in bucket:
                nb = self._engine.cells._data[idx]
                if getattr(nb, "_alive", True):
                    yield nb

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_spatial_grid(self) -> None:
        ts = self._engine.tissue_voxel_size
        grid: dict[tuple[int, int, int], list[int]] = {}
        for i, agent in enumerate(self._engine.cells._data):
            if not getattr(agent, "_alive", True):
                continue
            key = (
                int(math.floor(agent.pos.x / ts)),
                int(math.floor(agent.pos.y / ts)),
                int(math.floor(agent.pos.z / ts)),
            )
            grid.setdefault(key, []).append(i)
        self._spatial_grid = grid


# TODO: double buffering
