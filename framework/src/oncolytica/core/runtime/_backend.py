from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ISimulationBackend(Protocol):
    """Contract that any backend (CPU / GPU) must implement."""

    def compile(self, sim_instance: Any) -> None:
        """
        Compiles/initializes the backend for a specific simulation instance.
        Called once after load_model().
        """
        ...

    def run_step(self, collect_metrics: bool = False) -> None:
        """Executes one simulation step."""
        ...

    def sync_to_host(self) -> None:
        """
        Synchronizes agent state from device (GPU) → host (CPU).
        For CPU backend — no-op.
        """
        ...

    def sync_to_device(self) -> None:
        """
        Synchronizes host changes → device.
        For CPU backend — no-op.
        """
        ...

    def get_metrics(self) -> Any:
        """Returns last updated metrics."""
