"""oncolytica._containers — data containers.

* ``Field``     — N-dimensional grid of BaseData instances (tissue voxels,
                  chemical fields, environment).
* ``AgentList`` — dynamic list of cell agents.
"""

from typing import Any, Iterator, Type

import numpy as _np

from oncolytica.core.utils._types import BaseData, ivec3


# ── Field ─────────────────────────────────────────────────────────────────────

class Grid:
    """N-dimensional grid where every cell stores one ``BaseData`` instance.

    Backed by a NumPy ``object`` array for fast multi-index access.

    Parameters
    ----------
    struct_class:
        The ``ol.BaseData`` subclass that defines each voxel's data layout.
    shape:
        Integer tuple, e.g. ``(100, 100)`` for 2-D or ``(50, 50, 20)``
        for 3-D.
    """

    def __init__(self, struct_class: Type[BaseData], shape: tuple[int, ...], engine: Any) -> None:
        self._struct_class = struct_class
        self._engine = engine
        self._shape = tuple(shape)
        # Allocate and populate with fresh struct instances.
        self._data: _np.ndarray = _np.empty(self._shape, dtype=object)

        it = _np.nditer(self._data, flags=["multi_index", "refs_ok"])
        while not it.finished:
            inst = struct_class()

            # Inject grid reference and coordinates for neighbor queries
            inst._grid = self
            inst._engine = engine
            mi = it.multi_index
            inst._coord = ivec3(mi[0], mi[1], mi[2])

            self._data[it.multi_index] = inst
            it.iternext()

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple:
        return self._shape

    # ── element access ───────────────────────────────────────────────────────

    def __getitem__(self, index) -> BaseData:
        return self._data[index]

    def __setitem__(self, index, value: BaseData) -> None:
        self._data[index] = value

    def __iter__(self) -> Iterator[BaseData]:
        """Iterate over every voxel in row-major order."""
        for s in self._data.flat:
            yield s

    def __len__(self) -> int:
        return self._data.size

    # ── spatial sampling ─────────────────────────────────────────────────────

    def sample(self, x: int, y: int, z: int) -> BaseData:
        """Return the voxel"""
        return self._data[x, y, z]

    def __repr__(self) -> str:
        return (
            f"Field(struct={self._struct_class.__name__!r}, shape={self._shape})"
        )


# ── AgentList ─────────────────────────────────────────────────────────────────

class AgentList:
    """Dynamic array of cell agents.

    Agents are appended via ``add()`` (initialisation) or via the
    ``agent.divide(new_agent)`` helper (runtime).  Dead agents are excluded
    from iteration; call ``remove_dead()`` to compaction the backing store.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._data: list = []

    def __iter__(self) -> Iterator[BaseData]:
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # ── public write interface ────────────────────────────────────────────────

    def add(self, agent: Any) -> None:
        """Append *agent* to the list and inject lifecycle helpers."""
        self._inject_helpers(agent)
        self._data.append(agent)

    # ── compaction ────────────────────────────────────────────────────────────

    def _remove_dead(self) -> int:
        before = len(self._data)
        self._data = [a for a in self._data if getattr(a, "_alive", True)]
        return before - len(self._data)

    # ── lifecycle injection ───────────────────────────────────────────────────

    def _inject_helpers(self, agent: Any) -> None:
        agent._alive = True
        agent._agent_list = self
