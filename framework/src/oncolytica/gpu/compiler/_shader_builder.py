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
from oncolytica.gpu.runtime._memory import wgsl_struct, optimize_struct_layout

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
        self.cell_packed = optimize_struct_layout(cell_packed)
        self.tissue_packed = optimize_struct_layout(tissue_packed)
        self.chem_packed = optimize_struct_layout(chem_packed)
        self.metrics_packed = optimize_struct_layout(metrics_packed)

        self._user_params = user_params

        self._constants: list[str] = []     # const X: T = V; declarations
        self._helpers: list[str] = []       # WGSL helper fn bodies (non-rule methods)
        self._kernels: list[str] = []       # Stores compute shader kernel functions

        self.cell_kernel_names:   list[str] = []
        self.tissue_kernel_names: list[str] = []
        self.chem_kernel_names:   list[str] = []
        self.metric_kernel_names: list[str] = []

    # ------------------------------------------------------------------
    # Adding kernel functions
    # ------------------------------------------------------------------

    def add_metric_kernel(self, kernel_name, param_name, body):
        """Adds a metric computation kernel that reads cell data without modifying it."""
        self._kernels.append(self._wrap_metric_cell_kernel(kernel_name, param_name, body))
        self.metric_kernel_names.append(kernel_name)

    def add_cell_kernel(self, kernel_name: str, param_name: str, body: str) -> None:
        """Adds a kernel that processes and updates cell data."""
        self._kernels.append(self._wrap_cell_kernel(kernel_name, param_name, body))
        self.cell_kernel_names.append(kernel_name)

    def add_tissue_kernel(self, kernel_name: str, param_name: str, body: str) -> None:
        """Adds a kernel that processes and updates tissue data."""
        self._kernels.append(self._wrap_tissue_kernel(kernel_name, param_name, body))
        self.tissue_kernel_names.append(kernel_name)

    def add_chem_kernel(self, kernel_name: str, param_name: str, body: str) -> None:
        """Adds a kernel that processes and updates chemistry data."""
        self._kernels.append(self._wrap_chem_kernel(kernel_name, param_name, body))
        self.chem_kernel_names.append(kernel_name)

    # ------------------------------------------------------------------
    # Adding constants and functions
    # ------------------------------------------------------------------

    def add_helper_fn(self, wgsl_fn: str) -> None:
        """Adds a compiled WGSL helper function (non-rule method) to USER_METHODS."""
        if wgsl_fn not in self._helpers:
            self._helpers.append(wgsl_fn)

    def add_all_constants(self, constants: dict[str, tuple[str, str]]) -> None:
        """Adds all discovered constants to the shader as 'const' declarations."""
        if not constants:
            return

        for name in sorted(constants.keys()):
            ctype, cval = constants[name]
            self._constants.append(f"const {name}: {ctype} = {cval};")

    # ------------------------------------------------------------------
    # Building the final shader
    # ------------------------------------------------------------------

    def build(self, template_path: str) -> str:
        """Builds the complete WGSL shader by injecting generated code into the template."""
        with open(template_path, "r", encoding="utf-8") as fh:
            template = fh.read()

        return (
            template
            .replace("// {{ USER_STRUCTS }}",  self._build_structs())
            .replace("// {{ USER_CONSTANTS }}", "\n".join(self._constants))
            .replace("// {{ USER_METHODS }}", "\n\n".join(self._helpers))
            .replace("// {{ USER_LOGIC }}",   "\n\n".join(self._kernels))
            .replace("// {{ USER_CONFIG }}",   self._build_user_config())
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
        parts.append(wgsl_struct("Cell",      self.cell_packed))
        parts.append(wgsl_struct("Tissue",    self.tissue_packed))
        parts.append(wgsl_struct("Chemistry", self.chem_packed))
        parts.append(wgsl_struct("Metrics",   self.metrics_packed))
        return "\n\n".join(parts)

    def _build_user_config(self) -> str:
        """Renders user hyperparameter fields for the // {{ USER_CONFIG }} marker."""
        if not self._user_params:
            return ""
        return "\n".join(
            f"    {name}: {wtype},"
            for name, (wtype, _) in sorted(self._user_params.items())
        )

    def _add_constants(self, constants: dict[str, tuple[str, str]] | None) -> None:
        """Adds constant declarations to the shader, avoiding duplicates."""
        if not constants: return
        for cname, (ctype, cval) in constants.items():
            decl = f"const {cname}: {ctype} = {cval};"
            if decl not in self._constants:
                self._constants.append(decl)

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
