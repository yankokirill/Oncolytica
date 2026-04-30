"""
Builds the final WGSL shader from a template and generated code.

Responsibilities:
  - Generating WGSL structures (Cell, Tissue, Chemistry)
  - Extending uniforms with user-defined parameters
  - Wrapping compiled rules into kernel functions
  - Final shader assembly via marker replacement in the template
"""
from __future__ import annotations

import os
import textwrap
from typing import Any

_RNG_STATE_FIELD = ("_rng_state", "u32")

def optimize_struct_layout(fields: list[tuple[str, str]], inject_rng: bool = False) -> list[tuple[str, str]]:
    """
    Packs WGSL struct fields to minimize padding.
    vec3 occupies 12 bytes but aligns to 16, leaving a 4-byte "hole".
    This function greedily places 4-byte scalars immediately after vec3.
    """
    vec4s = []
    vec3s = []
    vec2s = []
    scalars = []

    if inject_rng:
        scalars.append(_RNG_STATE_FIELD)

    for name, ftype in fields:
        # Strip whitespace if present for accurate comparison
        clean_type = ftype.strip()
        if clean_type.startswith("vec4"):
            vec4s.append((name, clean_type))
        elif clean_type.startswith("vec3"):
            vec3s.append((name, clean_type))
        elif clean_type.startswith("vec2"):
            vec2s.append((name, clean_type))
        else:
            # Treat everything else (f32, u32, i32) as 4-byte scalars
            scalars.append((name, clean_type))

    packed = []

    # 1. Add vec4 first (perfectly fits 16 bytes)
    packed.extend(vec4s)

    # 2. Add vec3 and immediately fill the 4-byte hole with a scalar
    for v3 in vec3s:
        packed.append(v3)
        if scalars:
            packed.append(scalars.pop(0)) # Take the first available scalar (often _rng_state)

    # 3. Add vec2 (8 bytes each)
    packed.extend(vec2s)

    # 4. Add remaining 4-byte scalars
    packed.extend(scalars)

    return packed


def wgsl_struct(name: str, packed: list[tuple[str, str]]) -> str:
    """Generates WGSL struct definition with optimized field layout."""
    lines = [f"struct {name} {{"]
    for fname, ftype in packed:
        lines.append(f"    {fname}: {ftype},")
    lines.append("};")
    return "\n".join(lines)


class ShaderBuilder:
    """
    Assembles a WGSL shader from parts.
    """

    def __init__(
        self,
        spec: Any,
        cell_packed: list[tuple[str, str]],
        tissue_packed: list[tuple[str, str]],
        chem_packed: list[tuple[str, str]],
        metrics_packed: list[tuple[str, str]],
        user_params: dict[str, tuple[str, Any]],
    ) -> None:
        self._spec = spec

        # === APPLY FIELD ORDER OPTIMIZATION ===
        self.cell_packed = optimize_struct_layout(cell_packed, inject_rng=True)
        self.tissue_packed = optimize_struct_layout(tissue_packed, inject_rng=True)
        self.chem_packed = optimize_struct_layout(chem_packed, inject_rng=True)
        self.metrics_packed = optimize_struct_layout(metrics_packed, inject_rng=False)

        self._user_params = user_params

        self._methods: list[str] = []       # Stores constant declarations and helper functions
        self._kernels: list[str] = []       # Stores compute shader kernel functions

        self.cell_kernel_names:   list[str] = []
        self.tissue_kernel_names: list[str] = []
        self.chem_kernel_names:   list[str] = []
        self.metric_kernel_names: list[str] = []

    # ------------------------------------------------------------------
    # Adding kernel functions
    # ------------------------------------------------------------------

    def add_metric_kernel(self, kernel_name, param_name, body, constants=None):
        """Adds a metric computation kernel that reads cell data without modifying it."""
        self._add_constants(constants)
        self._kernels.append(self._wrap_metric_cell_kernel(kernel_name, param_name, body))
        self.metric_kernel_names.append(kernel_name)

    def add_cell_kernel(self, kernel_name: str, param_name: str, body: str, constants: dict[str, tuple[str, str]] | None = None) -> None:
        """Adds a kernel that processes and updates cell data."""
        self._add_constants(constants)
        self._kernels.append(self._wrap_cell_kernel(kernel_name, param_name, body))
        self.cell_kernel_names.append(kernel_name)

    def add_tissue_kernel(self, kernel_name: str, param_name: str, body: str, constants: dict[str, tuple[str, str]] | None = None) -> None:
        """Adds a kernel that processes and updates tissue data."""
        self._add_constants(constants)
        self._kernels.append(self._wrap_tissue_kernel(kernel_name, param_name, body))
        self.tissue_kernel_names.append(kernel_name)

    def add_chem_kernel(self, kernel_name: str, param_name: str, body: str, constants: dict[str, tuple[str, str]] | None = None) -> None:
        """Adds a kernel that processes and updates chemistry data."""
        self._add_constants(constants)
        self._kernels.append(self._wrap_chem_kernel(kernel_name, param_name, body))
        self.chem_kernel_names.append(kernel_name)

    # ------------------------------------------------------------------
    # Building the final shader
    # ------------------------------------------------------------------

    def build(self, template_path: str) -> str:
        """Builds the complete WGSL shader by injecting generated code into the template."""
        with open(template_path, "r", encoding="utf-8") as fh:
            template = fh.read()

        template = self._inject_user_uniforms(template)
        return (
            template
            .replace("// {{ USER_STRUCTS }}",  self._build_structs())
            .replace("// {{ USER_METHODS }}", "\n\n".join(self._methods))
            .replace("// {{ USER_LOGIC }}",   "\n\n".join(self._kernels))
            .replace("// {{ USER_CONSTANTS }}", "")
        )

    @staticmethod
    def find_template() -> str:
        """Locates the template.wgsl file by searching in common locations."""
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "template.wgsl"),
            os.path.join(here, "..", "gpu", "template.wgsl"),
            os.path.join(os.path.dirname(here), "gpu", "template.wgsl"),
        ]
        for path in candidates:
            normalised = os.path.normpath(path)
            if os.path.exists(normalised):
                return normalised
        raise FileNotFoundError(
            "template.wgsl not found. Searched:\n"
            + "\n".join(f"  {os.path.normpath(c)}" for c in candidates)
        )

    # ------------------------------------------------------------------
    # Internal: generate WGSL fragments
    # ------------------------------------------------------------------

    def _build_structs(self) -> str:
        """Generates all WGSL struct definitions with optimized layouts."""
        parts: list[str] = []
        # We've removed manual addition of [_RNG_STATE_FIELD],
        # because optimize_struct_layout already embeds it!
        parts.append(wgsl_struct("Cell",      self.cell_packed))
        parts.append(wgsl_struct("Tissue",    self.tissue_packed))
        parts.append(wgsl_struct("Chemistry", self.chem_packed))
        parts.append(wgsl_struct("Metrics",   self.metrics_packed))
        return "\n\n".join(parts)

    def _inject_user_uniforms(self, template_code: str) -> str:
        """Injects user-defined hyperparameters into the uniform buffer."""
        if not self._user_params:
            return template_code
        extra = "\n".join(
            f"    {name}: {wtype},"
            for name, (wtype, _) in sorted(self._user_params.items())
        )
        marker = "    BS_L2_offset: u32,\n};"
        replacement = f"    BS_L2_offset: u32,\n    // User hyperparameters\n{extra}\n}};"
        return template_code.replace(marker, replacement, 1)

    def _add_constants(self, constants: dict[str, tuple[str, str]] | None) -> None:
        """Adds constant declarations to the shader, avoiding duplicates."""
        if not constants: return
        for cname, (ctype, cval) in constants.items():
            decl = f"const {cname}: {ctype} = {cval};"
            if decl not in self._methods:
                self._methods.append(decl)

    # ------------------------------------------------------------------
    # Kernel wrappers - generate WGSL compute shader code
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_metric_cell_kernel(name: str, param: str, body: str) -> str:
        """Wraps metric calculation code into a compute shader that reads cells."""
        indented = textwrap.indent(body, "    ")
        return (
            f"@compute @workgroup_size(256, 1, 1)\n"
            f"fn {name}(@builtin(global_invocation_id) _id: vec3<u32>) {{\n"
            f"    let cell_index = _id.x;\n"
            f"    if (cell_index >= State.TotalAgents) {{ return; }}\n"
            f"    let {param} = Cells_In[cell_index];\n"
            f"{indented}\n"
            f"}}"
        )

    @staticmethod
    def _wrap_cell_kernel(name: str, param: str, body: str) -> str:
        """Wraps cell update code into a compute shader that modifies cell data."""
        indented = textwrap.indent(body, "    ")
        return (
            f"@compute @workgroup_size(256, 1, 1)\n"
            f"fn {name}(@builtin(global_invocation_id) _id: vec3<u32>) {{\n"
            f"    let cell_index = _id.x;\n"
            f"    if (cell_index >= State.TotalAgents) {{ return; }}\n"
            f"    var {param} = Cells_In[cell_index];\n"
            f"    var _rng = {param}._rng_state;\n"
            f"{indented}\n"
            f"    {param}._rng_state = _rng;\n"
            f"    Cells_Out[cell_index] = {param};\n"
            f"}}"
        )

    @staticmethod
    def _wrap_tissue_kernel(name: str, param: str, body: str) -> str:
        """Wraps tissue update code into a compute shader that modifies tissue data."""
        indented = textwrap.indent(body, "    ")
        return (
            f"@compute @workgroup_size(256, 1, 1)\n"
            f"fn {name}(@builtin(global_invocation_id) _id: vec3<u32>) {{\n"
            f"    let tissue_index = _id.x;\n"
            f"    if (tissue_index >= U.NumVoxelTableEntries) {{ return; }}\n"
            f"    var {param} = Tissue_In[tissue_index];\n"
            f"    var _rng = {param}._rng_state;\n"
            f"{indented}\n"
            f"    {param}._rng_state = _rng;\n"
            f"    Tissue_Out[tissue_index] = {param};\n"
            f"}}"
        )

    @staticmethod
    def _wrap_chem_kernel(name: str, param: str, body: str) -> str:
        """Wraps chemistry update code into a compute shader that modifies chemistry data."""
        indented = textwrap.indent(body, "    ")
        return (
            f"@compute @workgroup_size(256, 1, 1)\n"
            f"fn {name}(@builtin(global_invocation_id) _id: vec3<u32>) {{\n"
            f"    let chem_index = _id.x;\n"
            f"    if (chem_index >= U.NumVoxelTableEntries << 3) {{ return; }}\n"
            f"    var {param} = Chemistry_In[chem_index];\n"
            f"    var _rng = {param}._rng_state;\n"
            f"{indented}\n"
            f"    {param}._rng_state = _rng;\n"
            f"    Chemistry_Out[chem_index] = {param};\n"
            f"}}"
        )
