"""
test_oncolytica.py - Test suite for the Oncolytica framework.
Covers basic CPU/GPU execution and compilation edge-cases (WGSL translation pitfalls).
"""

import pytest
import math
import oncolytica as ol
from oncolytica import CompilationError


# ==============================================================================
# 1. FIXTURES & BASIC DATA STRUCTURES
# ==============================================================================

class MyCell(ol.Cell):
    pos: ol.vec3 = ol.vec3(50.0, 50.0, 50.0)
    health: ol.f32
    mutated: ol.bool


class MyTissue(ol.Tissue):
    oxygen: ol.f32 = 1.0


class MyChem(ol.Chemistry):
    drug_conc: ol.f32 = 0.0


class MyMetrics(ol.Metrics):
    alive_cells: ol.u32 = 0

class MyParams(ol.Params):
    pass

# ==============================================================================
# 2. BASIC TESTS (HAPPY PATH)
# ==============================================================================

def test_core_types_and_math():
    """Test core vector operations and math functions on the CPU."""
    v1 = ol.vec3(1.0, 2.0, 3.0)
    v2 = ol.vec3(0.0, 1.0, 0.0)

    # Vector arithmetic
    v3 = v1 + v2
    assert v3.x == 1.0 and v3.y == 3.0 and v3.z == 3.0

    v4 = v1 * 2.0
    assert v4.x == 2.0 and v4.y == 4.0 and v4.z == 6.0

    # Math functions
    assert math.isclose(ol.math.length(v2), 1.0)
    assert math.isclose(ol.math.distance(v1, v1), 0.0)
    assert ol.math.clamp(5.0, 0.0, 1.0) == 1.0


def test_basic_cpu_simulation():
    """Test the full lifecycle of a basic simulation on the CPU backend."""
    class BasicSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def cell_logic(self, cell: MyCell):
            cell.health -= 10.0
            if cell.health <= 0.0:
                cell.die()

        @ol.metric_rule
        def track_deaths(self, cell: MyCell, metrics: MyMetrics):
            metrics.alive_cells += 1

    engine = ol.Engine(backend="cpu")

    # Add cells
    for i in range(5):
        x = ol.f32(i)
        cell = MyCell()
        cell.pos = ol.vec3(50.0, 50.0, 50.0) + 5.0 * ol.vec3(x)
        engine.cells.add(cell)

    engine.load_model(BasicSim())

    # Run 10 steps (health starts at 100, decreases by 10 per step)
    for _ in range(9):
        engine.run_step()

    engine.run_step(collect_metrics=True)
    metrics = engine.get_metrics()
    assert metrics is not None
    # All 5 cells should be marked as dead
    assert metrics.alive_cells == 0
    # The engine should have compacted the dead cells away
    engine.run_step()
    assert len(engine.cells._data) == 0


def test_basic_gpu_compilation():
    """Test that a valid simulation compiles successfully to WGSL via wgpu."""
    wgpu = pytest.importorskip("wgpu")

    class GpuSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        o2_consumption: ol.f32 = 0.1  # Valid constant declaration

        @ol.cell_rule
        def cell_logic(self, cell: MyCell):
            if cell.health > 50.0:
                cell.pos.x += 1.0

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    # If the WGSL compiler finds an error, wgpu will raise an exception during pipeline creation
    try:
        engine.load_model(GpuSim())
    except Exception as e:
        pytest.fail(f"GPU Compilation failed on valid WGSL code: {e}")

def test_local_variables():
    """Test that a valid simulation compiles successfully to WGSL via wgpu."""
    wgpu = pytest.importorskip("wgpu")

    class GpuSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        o2_consumption: ol.f32 = 0.1  # Valid constant declaration

        @ol.cell_rule
        def cell_logic(self, cell: MyCell):
            if False:
                b = 5
            b = 4

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    # If the WGSL compiler finds an error, wgpu will raise an exception during pipeline creation
    try:
        engine.load_model(GpuSim())
    except Exception as e:
        pytest.fail(f"GPU Compilation failed on valid WGSL code: {e}")

# ==============================================================================
# 3. EDGE CASES & COMPILATION ERRORS
# ==============================================================================

def test_edge_case_constant_mutation():
    """Constants (UPPER_CASE variables) cannot be reassigned inside rules."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            O2_CONSUMPTION = 0.5  # Forbidden: UPPER_CASE assignment

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Assignment to UPPER_CASE name"):
        engine.load_model(BadSim())


def test_edge_case_forbidden_syntax():
    """WGSL does not support try/except blocks."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            try:
                cell.health = 100.0
            except:
                pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="try/except blocks are forbidden"):
        engine.load_model(BadSim())


def test_edge_case_wgsl_vector_scalar_broadcast():
    """
    Python allows `vec3 + scalar`. WGSL does not allow `vec3<f32> += f32`.
    This test ensures that wgpu rejects the generated WGSL.
    """
    wgpu = pytest.importorskip("wgpu")

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            # Python allows this, but StmtTranslator will emit `cell.pos += 1.0;`
            # WGSL strictly requires `cell.pos += vec3<f32>(1.0);`
            cell.pos += 1.0

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    # Catch the wgpu compilation error (usually raised by the driver backend)
    with pytest.raises(Exception) as excinfo:
        engine.load_model(BadSim())

    # The specific error depends on the wgpu backend (Vulkan/Metal/DX12),
    # but it will complain about type mismatch between vec3<f32> and f32.
    assert "type" in str(excinfo.value).lower() or "match" in str(excinfo.value).lower()


def test_edge_case_wgsl_strict_math_typing():
    """
    WGSL requires exact types for math functions (e.g., sin(f32)).
    Passing an integer literal creates an invalid WGSL abstract-int downcast.
    """
    wgpu = pytest.importorskip("wgpu")

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            # Transpiles to `sin(5)` in WGSL. `sin()` expects f32/f16.
            # Passing an int literal will crash the WGSL compiler.
            v = ol.math.sin(5)

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    with pytest.raises(Exception) as excinfo:
        engine.load_model(BadSim())

    # Error should mention overload resolution or type matching for 'sin'
    assert "sin" in str(excinfo.value)


def test_edge_case_scope_leakage():
    """
    Python variables leak out of `if` blocks. WGSL variables do not.
    This creates an 'unknown identifier' error in WGSL.
    """
    wgpu = pytest.importorskip("wgpu")

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            if cell.pos.x > 0.0:
                step = 5.0
            else:
                step = -5.0

            cell.pos.y += step

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    try:
        engine.load_model(GoodSim())
    except Exception as e:
        pytest.fail(f"GPU Compilation failed on valid WGSL code: {e}")


def test_nested_neighbors_supported_and_safe():
    """
    Test that nested neighbor iteration works, but strictly enforces
    immutability of the neighbors (Corrected version of the previous test).
    """
    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            for n1 in cell.neighbors:
                for n2 in n1.neighbors:
                    # We can read from n2 and n1, but we must ONLY modify 'cell'
                    cell.health -= n2.health * 0.1

    engine = ol.Engine(backend="cpu")
    try:
        engine.load_model(GoodSim())
    except CompilationError as e:
        pytest.fail(f"Valid nested neighbor read raised CompilationError: {e}")

def test_gpu_noncubic_cell_simulation_runs():
    """
    End-to-end: a non-cubic GPU simulation must execute run_step() without
    error and produce a coherent metrics result.
    """
    wgpu = pytest.importorskip("wgpu")

    class FullSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def cell_logic(self, cell: MyCell):
            if cell.health > 50.0:
                cell.pos.x += 0.5

        @ol.tissue_rule
        def tissue_logic(self, t: MyTissue):
            t.oxygen -= 0.01

    engine = ol.Engine(backend="gpu")
    engine.setup_geometry(tissue_shape=(120, 64, 32))
    for i in range(4):
        c = MyCell()
        c.pos = ol.vec3(float(i) * 5.0, 0.0, 0.0)
        engine.cells.add(c)


    try:
        engine.load_model(FullSim())
        engine.run_step()
        engine.run_step()
    except Exception as e:
        pytest.fail(f"GPU run_step() failed on non-cubic grid: {e}")


def test_gpu_teleport_die_and_atomic_metrics():
    """
    Tests death via teleportation, atomic metrics, and agent count reduction.
    """
    wgpu = pytest.importorskip("wgpu")

    # Redefine classes with type_id for testing
    class TestCell(ol.Cell):
        pos: ol.vec3
        type_id: ol.i32 = 0

    class TestMetrics(ol.Metrics):
        alive_count: ol.u32 = 0
    
    class TestParams(ol.Params):
        pass

    class DieSim(ol.Simulation[MyTissue, MyChem, TestCell, TestMetrics, TestParams]):
        @ol.cell_rule
        def rule(self, cell: TestCell):
            if cell.type_id == 1:  # Kill cell with type_id=1
                cell.die()

        @ol.metric_rule
        def calc_metrics(self, cell: TestCell, m: TestMetrics):
            m.alive_count += 1

    engine = ol.Engine(backend="gpu", max_agents=4)
    engine.setup_geometry(tissue_shape=(10, 10, 10), tissue_voxel_size=2.0)

    # Add two cells: one will survive, one will die
    engine.cells.add(TestCell(pos=ol.vec3(5.0, 5.0, 5.0), type_id=2))  # Alive
    engine.cells.add(TestCell(pos=ol.vec3(1.0, 1.0, 1.0), type_id=1))  # Will die

    engine.load_model(DieSim())

    engine.run_step(collect_metrics=True)

    metrics = engine.get_metrics()
    assert metrics.alive_count == 1
    engine._backend_impl.sync_to_host()

    # Find cells by their ID and check their state
    dead_cell_found = False
    alive_cell_found = False
    expected_pos = (10 * 2.0) + 1.0

    for cell in engine.cells._data:
        if cell.type_id == 1:
            dead_cell_found = True
            assert math.isclose(cell.pos.x, expected_pos), "MockCell did not teleport"
        elif cell.type_id == 2:
            alive_cell_found = True
            assert math.isclose(cell.pos.x, 5.0), "Alive cell position changed"

    assert not dead_cell_found and alive_cell_found, "One of the cells was lost after synchronization"

    # --- Step 2 ---
    # Run another step. Now TotalAgents on GPU should be 1.
    engine.run_step(collect_metrics=True)
    metrics = engine.get_metrics()

    # The metric should again count only one live cell.
    # If TotalAgents hadn't been updated, the metric would count the dead cell too.
    assert metrics.alive_count == 1, "TotalAgents counter on GPU did not update"


def test_gpu_cell_division():
    """
    Test that cell.divide() correctly spawns a new cell and updates the total agent count.
    """
    wgpu = pytest.importorskip("wgpu")

    class DivCell(ol.Cell):
        pos: ol.vec3
        generation: ol.i32 = 0

    class DivideSim(ol.Simulation[MyTissue, MyChem, DivCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: DivCell):
            if cell.generation == 0:
                # Create a new cell displaced by (1, 0, 0)
                new_cell = DivCell()
                new_cell.pos = cell.pos + ol.vec3(1.0, 0.0, 0.0)
                new_cell.generation = 1
                cell.divide(new_cell)
                # Mark parent so it doesn't divide again
                cell.generation = 1

    engine = ol.Engine(backend="gpu", max_agents=10)
    engine.setup_geometry(tissue_shape=(10, 10, 10))
    engine.cells.add(DivCell(pos=ol.vec3(5.0, 5.0, 5.0)))

    engine.load_model(DivideSim())

    # Run one step to trigger division
    engine.run_step()
    engine._backend_impl.sync_to_host()

    # The new architecture should pull the exact number of agents from the State buffer
    assert len(engine.cells._data) == 2, "MockCell division did not increase the agent count."

    # Verify properties of the new and old cells
    gen1_cells = [c for c in engine.cells._data if c.generation == 1]
    assert len(gen1_cells) == 2, "MockCell generation flag was not correctly assigned/copied."


def test_chemistry_rule_iterations():
    """
    Test that @ol.chemistry_rule(iterations=N) actually runs the inner logic N times per step.
    (Testing on CPU for predictable exact values, though compilation applies to GPU too).
    """

    class ChemSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.chemistry_rule(iterations=3)
        def diffuse(self, chem: MyChem):
            chem.drug_conc += 1.0  # Should be called 3 times per run_step

    engine = ol.Engine(backend="gpu")
    engine.setup_geometry(tissue_shape=(2, 2, 2))

    engine.load_model(ChemSim())

    if engine.chemistry is None:
        engine.chemistry = [MyChem()]

    for c in engine.chemistry:
        c.drug_conc = 0.0

    engine.run_step()
    engine.sync_to_host()

    # Since iterations=3, every chem voxel should now have drug_conc = 3.0
    for c in engine.chemistry:
        assert math.isclose(c.drug_conc, 3.0), f"Expected 3.0 after 3 iterations, got {c.drug_conc}"


def test_gpu_randomness_compilation():
    """
    Test that cell.random() and cell.random_dir() successfully compile to WGSL
    and execute without crashing the shader.
    """
    wgpu = pytest.importorskip("wgpu")

    class RandSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            dist = ol.random() * 2.0
            direction = ol.random_dir()
            cell.pos += direction * dist

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell(pos=ol.vec3(10.0, 10.0, 10.0)))

    try:
        engine.load_model(RandSim())
        engine.run_step()
    except Exception as e:
        pytest.fail(f"GPU execution failed for random functions: {e}")


def test_multiple_rules_execution():
    """
    Test that multiple rules of the same type are all discovered and executed.
    """

    class MultiRuleSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule_one(self, cell: MyCell):
            cell.health -= 5.0

        @ol.cell_rule
        def rule_two(self, cell: MyCell):
            cell.health -= 15.0

    engine = ol.Engine(backend="cpu")
    engine.cells.add(MyCell(health=100.0))
    engine.load_model(MultiRuleSim())

    engine.run_step()

    # 100 - 5 - 15 = 80
    assert math.isclose(engine.cells._data[0].health, 80.0)



def test_fields_in_sim_forbidden():
    """
    Test that multiple rules of the same type are all discovered and executed.
    """

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        local_field: int

    engine = ol.Engine(backend="cpu")
    engine.cells.add(MyCell(health=100.0))

    with pytest.raises(RuntimeError, match="Fields in Simulation class are forbidden."):
        engine.load_model(BadSim())
