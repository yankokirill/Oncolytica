"""
GPU backend implementation for WebGPU-based simulation acceleration.
"""
from __future__ import annotations
import math
import struct
import json
import os
from typing import Any

from oncolytica.core.runtime._backend import ISimulationBackend
from oncolytica.gpu.runtime._device_manager import DeviceManager, SORT_KERNELS
from oncolytica.gpu.runtime._memory import StructSerializer
from oncolytica.gpu.compiler._compiler import WGSLCompiler
from oncolytica.gpu.compiler._type_system import wgsl_alignment
from oncolytica.core.utils._config import settings
from oncolytica.core.utils._types import _resolve_own_hints, _FLOAT_TYPES, _INT_TYPES, _BOOL_TYPES


class GpuBackend(ISimulationBackend):
    """GPU backend that compiles Python rules to WGSL and executes on WebGPU."""

    def __init__(self, engine: Any, ctx: Any) -> None:
        self.engine = engine
        self._ctx = ctx
        self._spec: Any = None
        self._device_mgr = DeviceManager()

        self._ser: dict[str, StructSerializer] = {}
        self._user_params: dict[str, tuple[str, Any]] = {}
        self._all_kernel_names: list[str] = list(SORT_KERNELS)
        self._kernel_lists: dict[str, list[str]] = {}
        self._shader_code: str = ""

        # Ping-pong states.
        self._cell_flip: int = 0
        self._tissue_flip: int = 0
        self._chem_flip: int = 0

        self._steps_run: int = 0

    def compile(self, sim_instance: Any) -> None:
        """Compiles simulation rules and initializes GPU resources."""

        self._spec = sim_instance._spec

        # 1. Initialize memory layouts
        self._ser["tissue"] = StructSerializer("MockTissue", self._spec.tissue_class)
        self._ser["chem"] = StructSerializer("MockChemistry", self._spec.chemistry_class)
        self._ser["cell"] = StructSerializer("MockCell", self._spec.cell_class)
        self._ser["metric"] = StructSerializer("MockMetrics", self._spec.metrics_class, is_atomic=True)

        self._extract_user_params(sim_instance)

        # 2. Compile Python rules to WGSL
        compiler = WGSLCompiler(self._ctx, self._spec, self.engine, self._user_params, self._ser)

        self._ser["cell"].update_layout(compiler.builder.cell_packed)
        self._ser["tissue"].update_layout(compiler.builder.tissue_packed)
        self._ser["chem"].update_layout(compiler.builder.chem_packed)
        self._ser["metric"].update_layout(compiler.builder.metrics_packed)

        self._shader_code = compiler.compile()
        if settings.SAVE_WGSL_PATH != "":
            self.dump_wgsl(settings.SAVE_WGSL_PATH)

        # Register kernel names
        self._all_kernel_names.extend(compiler.builder.cell_kernel_names)
        self._all_kernel_names.extend(compiler.builder.tissue_kernel_names)
        self._all_kernel_names.extend(compiler.builder.chem_kernel_names)
        self._all_kernel_names.extend(compiler.builder.metric_kernel_names)
        self._kernel_lists = {
            "cell": compiler.builder.cell_kernel_names,
            "tissue": compiler.builder.tissue_kernel_names,
            "chem": compiler.builder.chem_kernel_names,
            "metric": compiler.builder.metric_kernel_names,
        }

        # 3. Set up WebGPU resources
        self._device_mgr.initialize()
        shader_mod = self._device_mgr.device.create_shader_module(code=self._shader_code)

        tgd = self.engine.tissue_shape
        cgd = self.engine.chemistry_shape

        uniform_data = self._device_mgr.pack_uniforms(
            tissue_grid_dim=tgd,
            tissue_voxel_size=self.engine.tissue_voxel_size,
            user_params=self._user_params,
        )

        self._device_mgr.create_buffers(
            total_agents=len(self.engine.cells),
            max_agents=self.engine.max_agents,
            tissue_grid_dim=tgd,
            cell_stride=self._ser["cell"].stride,
            tissue_stride=self._ser["tissue"].stride,
            chem_stride=self._ser["chem"].stride,
            metrics_stride=self._ser["metric"].stride,
            uniform_data=uniform_data
        )

        self._device_mgr.create_pipelines(shader_mod, self._all_kernel_names)
        self.sync_to_device()

    def _flush_and_debug(self, encoder: Any, message: str) -> Any:
        device = self._device_mgr.device
        device.queue.submit([encoder.finish()])
        self.sync_to_host()
        print(f"[DEBUG] {message}")
        return device.create_command_encoder()

    def run_step(self, collect_metrics: bool = False, debug: bool = False) -> None:
        mgr = self._device_mgr
        device = mgr.device
        encoder = device.create_command_encoder()

        if collect_metrics:
            metrics_size = mgr.buf["MetricsBuffer"].size
            encoder.clear_buffer(mgr.buf["MetricsBuffer"], 0, metrics_size)

        t_count = mgr.buf["Tissue_In"].size // self._ser["tissue"].stride
        c_count = mgr.buf["Chem_In"].size // self._ser["chem"].stride
        cell_count = mgr.buf["Cells_In"].size // self._ser["cell"].stride

        # 1. MockTissue Rules
        if self._kernel_lists["tissue"]:
            wgs = math.ceil(t_count / 256)
            for k in self._kernel_lists["tissue"]:
                self._dispatch(encoder, k, wgs)
                self._tissue_flip = 1 - self._tissue_flip

        # 2. MockChemistry Rules
        if self._kernel_lists["chem"]:
            wgs = math.ceil(c_count / 256)
            for i, k in enumerate(self._kernel_lists["chem"]):
                rule_func = self.engine._chemistry_rules[i]
                iters = getattr(rule_func, "_iterations", 1)

                for _ in range(iters):
                    self._dispatch(encoder, k, wgs)
                    self._chem_flip = 1 - self._chem_flip

        # 3. MockCell Rules
        if self._kernel_lists["cell"]:
            wgs = math.ceil(cell_count / 256)
            for k in self._kernel_lists["cell"]:
                self._dispatch(encoder, k, wgs)
                self._cell_flip = 1 - self._cell_flip

        # 4. MockCell Sort (Morton)
        self._dispatch_sort_pipeline(encoder, cell_count)
        self._cell_flip = 1 - self._cell_flip

        # 5. Update State
        self._dispatch(encoder, "Kernel_UpdateState", 1)

        # 6. MockMetrics Collection
        if collect_metrics and self._kernel_lists["metric"]:
            wgs = math.ceil(cell_count / 256)
            for k in self._kernel_lists["metric"]:
                self._dispatch(encoder, k, wgs)

        device.queue.submit([encoder.finish()])

        if collect_metrics:
            b = mgr.buf["MetricsBuffer"]
            raw_metrics = bytes(device.queue.read_buffer(b))
            self._ser["metric"].unpack(raw_metrics, 0, self.engine._metrics)

        self._steps_run += 1

    def dump_wgsl(self, path: str = "shader.wgsl") -> None:
        """
        Saves the generated WGSL shader code to a file for debugging and inspection.

        Parameters
        ----------
        path : str
            The destination file path. Defaults to 'shader.wgsl'.
        """
        # Проверяем, была ли запущена компиляция (load_model)
        if not hasattr(self, "_shader_code") or self._shader_code is None or self._shader_code == "":
            raise RuntimeError(
                "WGSL code has not been generated yet. "
                "You must call 'engine.load_model()' before dumping the shader."
            )

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._shader_code)
            print(f"[Oncolytica] WGSL shader dumped to: {os.path.abspath(path)}")
        except Exception as e:
            raise IOError(f"Failed to write WGSL file to {path}: {e}")

    def get_config(self) -> dict[str, Any]:
        """
        Generates the configuration dictionary.
        Note: The WGSL shader is no longer embedded here. It will be saved separately.
        """
        if not hasattr(self, "_shader_code"):
            raise RuntimeError("Simulation must be compiled before generating config.")

        # 1. Compute structs offsets
        structs = {}
        for name, ser in self._ser.items():
            structs[name] = {
                "stride": ser.stride,
                "fields": {
                    fname: {"type": ftype, "offset": ser.offsets[fname]}
                    for fname, ftype in ser.fields
                }
            }

        # 2. Compute uniforms padding and offsets
        uniforms = {}
        curr_offset = 0
        base_uniforms = [("tissue_grid_dim", "vec3<u32>"), ("tissue_voxel_size", "f32")]

        for pname in sorted(self._user_params.keys()):
            ptype, _ = self._user_params[pname]
            base_uniforms.append((pname, ptype))

        for name, wtype in base_uniforms:
            # Assuming wgsl_alignment returns (size, alignment)
            sz, al = wgsl_alignment(wtype)
            if curr_offset % al != 0:
                curr_offset += al - (curr_offset % al)  # WGSL padding
            uniforms[name] = {"type": wtype, "offset": curr_offset}
            curr_offset += sz

        if curr_offset % 16 != 0:
            curr_offset += 16 - (curr_offset % 16)
        uniform_stride = curr_offset

        # 3. Execution order
        execution_order = []
        if self._kernel_lists["tissue"]:
            execution_order.append({"pass": "tissue", "kernels": self._kernel_lists["tissue"], "iterations": 1})
        if self._kernel_lists["chem"]:
            for i, k in enumerate(self._kernel_lists["chem"]):
                iters = getattr(self.engine._chemistry_rules[i], "_iterations", 1)
                execution_order.append({"pass": "chem", "kernels": [k], "iterations": iters})
        if self._kernel_lists["cell"]:
            execution_order.append({"pass": "cell", "kernels": self._kernel_lists["cell"], "iterations": 1})

        execution_order.append({
            "pass": "sort",
            "kernels": ["Kernel_ClearVoxels", "Kernel_CountAndOffset", "Kernel_Scan_L0", "Kernel_Scan_L1",
                        "Kernel_Scan_L2", "Kernel_AddBack_L1", "Kernel_AddBack_L0", "Kernel_Scatter"],
            "iterations": 1
        })
        execution_order.append({"pass": "update", "kernels": ["Kernel_UpdateState"], "iterations": 1})
        if self._kernel_lists["metric"]:
            execution_order.append({"pass": "metric", "kernels": self._kernel_lists["metric"], "iterations": 1})

        # 4. Standardized Bind Groups mapping
        bind_groups = {
            "0": {"name": "Uniforms", "bindings": [{"binding": 0, "type": "uniform_buffer", "name": "UniformBuffer"}]},
            "1": {"name": "Cells_and_State", "bindings": [
                {"binding": 0, "type": "storage_buffer", "name": "Cells_In"},
                {"binding": 1, "type": "storage_buffer", "name": "Cells_Out"},
                {"binding": 2, "type": "storage_buffer", "name": "State"}
            ]},
            "2": {"name": "Spatial_Sorting", "bindings": [
                {"binding": 0, "type": "storage_buffer", "name": "Voxels"},
                {"binding": 1, "type": "storage_buffer", "name": "PrefixSum"}
            ]},
            "3": {"name": "Environment_and_Metrics", "bindings": [
                {"binding": 0, "type": "storage_buffer", "name": "Tissue_In"},
                {"binding": 1, "type": "storage_buffer", "name": "Tissue_Out"},
                {"binding": 2, "type": "storage_buffer", "name": "Chem_In"},
                {"binding": 3, "type": "storage_buffer", "name": "Chem_Out"},
                {"binding": 4, "type": "storage_buffer", "name": "MetricsBuffer"}
            ]}
        }

        params_dict = {name: val[1] for name, val in self._user_params.items()}

        return {
            "params": params_dict,
            "layout": {
                "tissue_shape": self.engine.tissue_shape,
                "chemistry_shape": self.engine.chemistry_shape,
                "max_agents": self.engine.max_agents,
                "structs": structs,
                "uniforms": {"size": uniform_stride, "fields": uniforms}
            },
            "bind_groups": bind_groups,
            "execution_order": execution_order,
            "shader_path": "shader.wgsl",  # Pointer to the separate WGSL file
            "buffers": {}
        }

    def export_bundle(self, export_dir: str, export_initial_state: bool = True) -> str:
        """
        Exports the entire simulation state (JSON, WGSL, and BIN files) into a specific directory.
        Returns the absolute path to the generated config.json.
        """
        config = self.get_config()

        # Save WGSL explicitly into its own file
        shader_path = os.path.join(export_dir, config["shader_path"])
        with open(shader_path, "w", encoding="utf-8") as f:
            f.write(self._shader_code)

        if export_initial_state:
            # Sync RAM to GPU to capture the absolute initial state
            self.sync_to_device()
            device = self._device_mgr.device
            b = self._device_mgr.buf

            def save_buffer(buf_name: str, filename: str) -> str | None:
                if buf_name not in b: return None
                raw_data = bytes(device.queue.read_buffer(b[buf_name]))
                full_path = os.path.join(export_dir, filename)
                with open(full_path, "wb") as f:
                    f.write(raw_data)
                return filename  # Return relative filename for the JSON config

            # Inject bin references into config
            config["buffers"] = {
                "UniformBuffer": save_buffer("UniformBuffer", "uniforms.bin"),
                "State": save_buffer("State", "state.bin"),
                "Cells_In": save_buffer("Cells_In", "cells_in.bin"),
                "Tissue_In": save_buffer("Tissue_In", "tissue_in.bin") if self.engine.tissue is not None else None,
                "Chem_In": save_buffer("Chem_In", "chem_in.bin") if self.engine.chemistry is not None else None,
            }

        config_path = os.path.join(export_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return config_path

    def _extract_user_params(self, sim_instance: Any) -> None:
        """Collects user-defined hyperparameters from sim_instance._params."""
        params = sim_instance._params
        params_class = type(params)

        from typing import get_type_hints
        hints = get_type_hints(params_class)

        for name, ftype in hints.items():
            val = getattr(params, name, None)
            if ftype in _BOOL_TYPES or isinstance(val, bool):
                self._user_params[name] = ("u32", int(bool(val)))
            elif ftype in _FLOAT_TYPES or isinstance(val, float):
                self._user_params[name] = ("f32", float(val))
            elif ftype in _INT_TYPES or isinstance(val, int):
                self._user_params[name] = ("i32", int(val))

    def sync_to_device(self) -> None:
        device = self._device_mgr.device
        b = self._device_mgr.buf

        c_buf = "Cells_In" if self._cell_flip == 0 else "Cells_Out"
        t_buf = "Tissue_In" if self._tissue_flip == 0 else "Tissue_Out"
        ch_buf = "Chem_In" if self._chem_flip == 0 else "Chem_Out"

        if cells := self.engine.cells._data:
            cell_bytes = b"".join(self._ser["cell"].pack(c) for c in cells)
            device.queue.write_buffer(b[c_buf], 0, cell_bytes)

        if self.engine.tissue is not None:
            tissue_bytes = b"".join(self._ser["tissue"].pack(v) for v in self.engine.tissue)
            device.queue.write_buffer(b[t_buf], 0, tissue_bytes)

        if self.engine.chemistry is not None:
            chem_bytes = b"".join(self._ser["chem"].pack(v) for v in self.engine.chemistry)
            device.queue.write_buffer(b[ch_buf], 0, chem_bytes)

    def sync_to_host(self) -> None:
        device = self._device_mgr.device
        b = self._device_mgr.buf

        c_buf = "Cells_In" if self._cell_flip == 0 else "Cells_Out"
        t_buf = "Tissue_In" if self._tissue_flip == 0 else "Tissue_Out"
        ch_buf = "Chem_In" if self._chem_flip == 0 else "Chem_Out"

        raw_cells = bytes(device.queue.read_buffer(b[c_buf]))
        cell_stride = self._ser["cell"].stride

        raw_state = bytes(device.queue.read_buffer(b["State"]))
        n = struct.unpack_from("I", raw_state, 0)[0]

        self.engine.cells._data = [self._spec.cell_class() for _ in range(n)]
        for i, cell in enumerate(self.engine.cells._data):
            self._ser["cell"].unpack(raw_cells, i * cell_stride, cell)

        if self.engine.tissue is not None:
            raw_tissue = bytes(device.queue.read_buffer(b[t_buf]))
            tissue_stride = self._ser["tissue"].stride
            for i, voxel in enumerate(self.engine.tissue):
                if (i + 1) * tissue_stride > len(raw_tissue):
                    break
                self._ser["tissue"].unpack(raw_tissue, i * tissue_stride, voxel)

        if self.engine.chemistry is not None:
            raw_chem = bytes(device.queue.read_buffer(b[ch_buf]))
            chem_stride = self._ser["chem"].stride
            for i, voxel in enumerate(self.engine.chemistry):
                if (i + 1) * chem_stride > len(raw_chem):
                    break
            self._ser["chem"].unpack(raw_chem, i * chem_stride, voxel)

    def _dispatch(self, encoder: Any, kernel: str, x_wgs: int) -> None:
        mgr = self._device_mgr
        pass_enc = encoder.begin_compute_pass()

        pass_enc.set_bind_group(0, mgr.bg0, [])
        pass_enc.set_bind_group(1, mgr.bg1_combos[self._cell_flip], [])
        pass_enc.set_bind_group(2, mgr.bg2, [])
        pass_enc.set_bind_group(3, mgr.bg3_combos[(self._tissue_flip, self._chem_flip)], [])

        pass_enc.set_pipeline(mgr.pipelines[kernel])
        pass_enc.dispatch_workgroups(max(x_wgs, 1), 1, 1)
        pass_enc.end()

    def _dispatch_sort_pipeline(self, encoder: Any, max_cells: int) -> None:
        from oncolytica.core.runtime._geometry import voxel_table_size
        num_vox = voxel_table_size(self.engine.tissue_shape)
        l0_wgs = math.ceil(num_vox / 256)

        self._dispatch(encoder, "Kernel_ClearVoxels", math.ceil(num_vox / 256))
        self._dispatch(encoder, "Kernel_CountAndOffset", math.ceil(max_cells / 256))
        self._dispatch(encoder, "Kernel_Scan_L0", l0_wgs)
        self._dispatch(encoder, "Kernel_Scan_L1", math.ceil(l0_wgs / 256))
        self._dispatch(encoder, "Kernel_Scan_L2", 1)
        self._dispatch(encoder, "Kernel_AddBack_L1", math.ceil(l0_wgs / 256))
        self._dispatch(encoder, "Kernel_AddBack_L0", l0_wgs)
        self._dispatch(encoder, "Kernel_Scatter", math.ceil(max_cells / 256))
