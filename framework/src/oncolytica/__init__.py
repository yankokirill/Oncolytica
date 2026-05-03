"""oncolytica (ol) — Agent-based computational biology framework.

Unified CPU & GPU (WebGPU/WGSL) Backends · v2.0
"""

from __future__ import annotations

# ── Primitive type aliases ────────────────────────────────────────────────────
from oncolytica.core.utils._types import f32, i32, u32, bool, Simulation

# ── Core types & Data-layout base classes ─────────────────────────────────────
from oncolytica.core.utils._types import (
    vec3,
    ivec3,
    BaseData,
    Tissue,
    Chemistry,
    Cell,
    Metrics,
    Params,
)

from oncolytica.core.utils._types import random, random_dir

# ── Math namespace ────────────────────────────────────────────────────────────
import oncolytica.core.utils._math as math

# ── Data containers ───────────────────────────────────────────────────────────
from oncolytica.core.utils._containers import Grid, AgentList

# ── Rule decorators ───────────────────────────────────────────────────────────
from oncolytica.cpu._decorators import tissue_rule, chemistry_rule, cell_rule, metric_rule

# ── Simulation & Engine ───────────────────────────────────────────────────────
from oncolytica.core.runtime._engine import Engine

# ── GPU Compiler Exceptions ───────────────────────────────────────────────────
from oncolytica.core.utils._errors import CompilationError

__all__ = [
    # primitives
    "f32", "i32", "u32", "bool",
    # core types
    "vec3",
    # data layouts
    "BaseData", "Tissue", "Chemistry", "Cell", "Metrics",
    # math
    "math",
    # containers
    "Grid", "AgentList",
    # decorators
    "tissue_rule", "chemistry_rule", "cell_rule", "metric_rule",
    # simulation
    "Simulation",
    # engine
    "Engine",
    # compiler exceptions
    "CompilationError",
]

__version__ = "2.0.0"
