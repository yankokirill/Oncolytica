"""oncolytica._decorators — rule decorators for the v2 Engine pipeline.

Four decorators mark simulation methods for automatic discovery and
execution by ``ol.Engine``:

* ``@ol.tissue_rule()``        — runs per tissue voxel
* ``@ol.chemistry_rule()``     — runs per chemistry voxel
* ``@ol.cell_rule``            — runs per cell agent (spatial grid available)
* ``@ol.metric_rule(interval)``— runs per data item every N steps

On the CPU backend the decorators simply tag the function with metadata
attributes; no wrapping overhead is added.  The GPU backend will read
the same tags to decide which methods to compile as WGSL compute shaders.
"""

from __future__ import annotations

from typing import Callable, Optional


# ── tissue_rule ───────────────────────────────────────────────────────────────

def tissue_rule(_func: Optional[Callable] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._rule_type = "tissue"
        return func

    return decorator(_func) if _func is not None else decorator


# ── chemistry_rule ────────────────────────────────────────────────────────────

def chemistry_rule(iterations: int = 1) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._rule_type = "chemistry"
        func._iterations = iterations
        return func

    return decorator


# ── cell_rule ─────────────────────────────────────────────────────────────────

def cell_rule(_func: Optional[Callable] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._rule_type = "cell"
        return func

    return decorator(_func) if _func is not None else decorator


# ── metric_rule ───────────────────────────────────────────────────────────────

def metric_rule(_func: Optional[Callable] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._rule_type = "metric"
        return func

    return decorator(_func) if _func is not None else decorator
