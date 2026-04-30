"""
Manages GPU device, buffers, and pipelines.
"""
from __future__ import annotations

import math
from typing import Any
import struct as _struct_mod
from oncolytica.core._geometry import voxel_table_size

SORT_KERNELS: list[str] = [
    "Kernel_ClearVoxels",
    "Kernel_CountAndOffset",
    "Kernel_Scan_L0",
    "Kernel_Scan_L1",
    "Kernel_Scan_L2",
    "Kernel_AddBack_L1",
    "Kernel_AddBack_L0",
    "Kernel_Scatter",
    "Kernel_UpdateState",
]

class DeviceManager:
    def __init__(self) -> None:
        self._wgpu: Any = None
        self.device: Any = None
        self.buf: dict[str, Any] = {}

        self.bg0: Any = None
        self.bg2: Any = None

        # Комбинации для Cells: индекс = cell_flip
        self.bg1_combos: list[Any] = [None, None]
        # Комбинации для Tissue и Chem: ключ = (tissue_flip, chem_flip)
        self.bg3_combos: dict[tuple[int, int], Any] = {}

        self.pipelines: dict[str, Any] = {}

    def initialize(self) -> None:
        try:
            import wgpu  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Requires wgpu-py.") from exc

        self._wgpu = wgpu
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance", force_fallback_adapter=False)
        self.device = adapter.request_device(required_features=["subgroups"])

    def create_buffers(
        self, total_agents: int, max_agents: int, tissue_grid_dim: tuple[int, int, int],
        chem_grid_dim: tuple[int, int, int], cell_stride: int, tissue_stride: int,
        chem_stride: int, metrics_stride: int, uniform_data: bytes
    ) -> None:
        wgpu = self._wgpu
        device = self.device
        BU = wgpu.BufferUsage
        b = self.buf
        FULL_USAGE = BU.STORAGE | BU.COPY_DST | BU.COPY_SRC

        b["U"] = device.create_buffer(size=len(uniform_data), usage=BU.UNIFORM | BU.COPY_DST)
        device.queue.write_buffer(b["U"], 0, uniform_data)

        b["State"] = device.create_buffer(size=16, usage=FULL_USAGE)
        state_data = _struct_mod.pack("2I", total_agents, total_agents) + b"\x00" * 8
        device.queue.write_buffer(b["State"], 0, state_data)

        b["MetricsBuffer"] = device.create_buffer(size=max(metrics_stride, 16), usage=FULL_USAGE)

        cells_size = max(max_agents * cell_stride, 16)
        b["Cells_In"]  = device.create_buffer(size=cells_size, usage=FULL_USAGE)
        b["Cells_Out"] = device.create_buffer(size=cells_size, usage=FULL_USAGE)
        b["SortData"]  = device.create_buffer(size=max(max_agents * 8, 16), usage=BU.STORAGE)

        num_voxel_table = voxel_table_size(tissue_grid_dim)
        l0_wgs       = math.ceil(num_voxel_table / 256)
        l1_wgs       = math.ceil(l0_wgs / 256)
        bs_l2_offset = l0_wgs + l1_wgs

        b["VoxelTable"] = device.create_buffer(size=num_voxel_table * 8, usage=BU.STORAGE | BU.COPY_SRC)
        b["BlockSums"] = device.create_buffer(size=max((bs_l2_offset + 1) * 4, 16), usage=BU.STORAGE)

        num_tissue = num_voxel_table
        tissue_size = max(num_tissue * tissue_stride, 16)
        b["Tissue_In"]  = device.create_buffer(size=tissue_size, usage=FULL_USAGE)
        b["Tissue_Out"] = device.create_buffer(size=tissue_size, usage=FULL_USAGE)

        num_chem = (num_voxel_table << 3) + 1
        chem_size = max(num_chem * chem_stride, 16)
        b["Chem_In"]  = device.create_buffer(size=chem_size, usage=FULL_USAGE)
        b["Chem_Out"] = device.create_buffer(size=chem_size, usage=FULL_USAGE)

    def create_pipelines(self, shader_module: Any, all_kernel_names: list[str]) -> None:
        wgpu = self._wgpu
        device = self.device
        b = self.buf
        SS = wgpu.ShaderStage.COMPUTE

        def _buf_entry(binding: int, read_only: bool) -> dict:
            return {"binding": binding, "visibility": SS,
                    "buffer": {"type": "read-only-storage" if read_only else "storage"}}

        bg0_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": SS, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": SS, "buffer": {"type": "storage"}},
            {"binding": 2, "visibility": SS, "buffer": {"type": "storage"}},
        ])
        bg1_layout = device.create_bind_group_layout(
            entries=[_buf_entry(0, True), _buf_entry(1, False), _buf_entry(2, False)])
        bg2_layout = device.create_bind_group_layout(entries=[_buf_entry(0, False), _buf_entry(1, False)])
        bg3_layout = device.create_bind_group_layout(entries=[
            _buf_entry(0, True), _buf_entry(1, False), _buf_entry(2, True), _buf_entry(3, False)
        ])

        def _bind(buf_name: str, binding: int) -> dict:
            return {"binding": binding, "resource": {"buffer": b[buf_name], "offset": 0, "size": b[buf_name].size}}

        self.bg0 = device.create_bind_group(layout=bg0_layout, entries=[
            _bind("U", 0), _bind("State", 1), _bind("MetricsBuffer", 2)
        ])
        self.bg2 = device.create_bind_group(layout=bg2_layout, entries=[_bind("VoxelTable", 0), _bind("BlockSums", 1)])

        self.bg1_combos[0] = device.create_bind_group(
            layout=bg1_layout, entries=[_bind("Cells_In", 0), _bind("Cells_Out", 1), _bind("SortData", 2)]
        )
        self.bg1_combos[1] = device.create_bind_group(
            layout=bg1_layout, entries=[_bind("Cells_Out", 0), _bind("Cells_In", 1), _bind("SortData", 2)]
        )

        for t_flip in (0, 1):
            for c_flip in (0, 1):
                t_in = "Tissue_In" if t_flip == 0 else "Tissue_Out"
                t_out = "Tissue_Out" if t_flip == 0 else "Tissue_In"
                c_in = "Chem_In" if c_flip == 0 else "Chem_Out"
                c_out = "Chem_Out" if c_flip == 0 else "Chem_In"
                
                self.bg3_combos[(t_flip, c_flip)] = device.create_bind_group(
                    layout=bg3_layout, entries=[
                        _bind(t_in, 0), _bind(t_out, 1),
                        _bind(c_in, 2), _bind(c_out, 3)
                    ]
                )

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bg0_layout, bg1_layout, bg2_layout, bg3_layout])
        
        for name in all_kernel_names:
            self.pipelines[name] = device.create_compute_pipeline(
                layout=pipeline_layout, compute={"module": shader_module, "entry_point": name}
            )

    def pack_uniforms(
        self, tissue_grid_dim: tuple[int, int, int], tissue_voxel_size: float,
        user_params: dict[str, tuple[str, Any]]
    ) -> bytes:
        num_voxel_table = voxel_table_size(tissue_grid_dim)
        l0_wgs       = math.ceil(num_voxel_table / 256)
        l1_wgs       = math.ceil(l0_wgs / 256)
        bs_l1_offset = l0_wgs
        bs_l2_offset = bs_l1_offset + l1_wgs

        infra = _struct_mod.pack(
            "4I1f2I", num_voxel_table, tissue_grid_dim[0], tissue_grid_dim[1], tissue_grid_dim[2],
            float(tissue_voxel_size), bs_l1_offset, bs_l2_offset,
        )

        user_bytes = b""
        for name in sorted(user_params):
            wtype, value = user_params[name]
            if wtype == "f32": user_bytes += _struct_mod.pack("f", float(value))
            else: user_bytes += _struct_mod.pack("I", int(value) & 0xFFFFFFFF)

        raw = infra + user_bytes
        rem = len(raw) % 16
        if rem: raw += b"\x00" * (16 - rem)
        return raw
