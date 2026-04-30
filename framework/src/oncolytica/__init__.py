"""oncolytica (ol) — Agent-based computational biology framework.

Unified CPU & GPU (WebGPU/WGSL) Backends · v2.0
"""

from __future__ import annotations

# ── Primitive type aliases ────────────────────────────────────────────────────
from oncolytica.core._types import f32, f64, i32, i64, u32, u64, bool

# ── Core types & Data-layout base classes ─────────────────────────────────────
from oncolytica.core._types import (
    vec3,
    BaseData,
    TissueData,
    ChemistryData,
    CellData,
    MetricsData,
)

from oncolytica.core._types import random, random_dir

# ── Math namespace ────────────────────────────────────────────────────────────
import oncolytica.core._math as math

# ── Data containers ───────────────────────────────────────────────────────────
from oncolytica.cpu._containers import Grid, AgentList

# ── Rule decorators ───────────────────────────────────────────────────────────
from oncolytica.cpu._decorators import tissue_rule, chemistry_rule, cell_rule, metric_rule

# ── Simulation & Engine ───────────────────────────────────────────────────────
from oncolytica.core._simulation import Simulation
from oncolytica.core._engine import Engine

# ── GPU Compiler Exceptions ───────────────────────────────────────────────────
from oncolytica.core._errors import CompilationError

__all__ = [
    # primitives
    "f32", "f64", "i32", "i64", "u32", "u64", "bool",
    # core types
    "vec3",
    # data layouts
    "BaseData", "TissueData", "ChemistryData", "CellData", "MetricsData",
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
