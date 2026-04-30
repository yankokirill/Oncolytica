"""GPU Morton-sort test using high-level Engine API with fake Simulation."""
import math, random, pytest
from oncolytica.core._simulation import Simulation
from oncolytica.core._types import TissueData, ChemistryData, CellData, MetricsData, vec3
from oncolytica.core._geometry import get_tissue_voxel_key, voxel_table_size
from oncolytica.core._engine import Engine

# ── Mock data classes (must inherit from base types for validation) ──
class Cell(CellData): pos: vec3; id: int
class Tissue(TissueData): pad: int = 0
class Chemistry(ChemistryData): pad: int = 0
class Metrics(MetricsData): pad: int = 0

class FakeSim(Simulation[Tissue, Chemistry, Cell, Metrics]):
    pass

def test_gpu_sort_highlevel():
    try: import wgpu  # noqa
    except ImportError: pytest.skip("wgpu required")

    N, GDIM, VS = 10_000, (50, 50, 50), 2.0

    # ── 1. Setup Engine + Simulation ───────────────────────────────
    sim = FakeSim()
    eng = Engine(backend='gpu', max_agents=N)
    eng.setup_geometry(GDIM, VS)

    # ── 2. Generate test data + CPU reference ──────────────────────
    random.seed(42)
    ref = {}
    for i in range(N):
        p = [random.uniform(0, GDIM[j]*VS*0.99) for j in range(3)]
        c = Cell()
        c.pos, c.id = vec3(*p), i
        eng.cells.add(c)
        coord = tuple(int(x/VS) for x in p)
        key = get_tissue_voxel_key(*coord)
        ref.setdefault(key, []).append(i)

    # ── 3. Run one step (executes full sort pipeline) ──────────────
    eng.load_model(sim)     # ← compiles shader, creates buffers, pipelines
    eng.run_step()          # ← dispatches all sort kernels + syncs internally
    eng.sync_to_host()      # ← reads back Cells_Out, updates eng.cells._data

    # ── 4. Validate results ────────────────────────────────────────
    nv = voxel_table_size(GDIM)
    buckets: dict[int, list[int]] = {vk: [] for vk in range(nv)}

    for c in eng.cells._data:  # ← already synced, contains sorted output
        coord = tuple(int(x/VS) for x in [c.pos.x, c.pos.y, c.pos.z])
        vk = get_tissue_voxel_key(*coord)
        buckets[vk].append(c.id)

    for vk in range(nv):
        gpu_ids, cpu_ids = buckets.get(vk, []), ref.get(vk, [])
        assert len(gpu_ids) == len(cpu_ids), f"Voxel {vk}: {len(gpu_ids)} vs {len(cpu_ids)}"
        assert set(gpu_ids) == set(cpu_ids), f"Voxel {vk} ID mismatch"

    # Final integrity: all IDs present exactly once
    all_ids = [c.id for c in eng.cells._data]
    assert sorted(all_ids) == list(range(N)), "Agent integrity check failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
