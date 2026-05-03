"""
Tests that every `return` in a rule kernel is preceded by the full epilogue:
    cell._rng_state = _rng;
    Cells_Out[cell_index] = cell;

Currently FAILING — early returns lack the epilogue.
"""
import re
import pytest
import oncolytica as ol


# ── Common domain types ───────────────────────────────────────────────────────

class MyTissue(ol.Tissue):
    oxygen:    ol.f32 = 1.0
    stiffness: ol.f32 = 0.5

class MyChem(ol.Chemistry):
    drug:  ol.f32 = 0.0
    toxin: ol.f32 = 0.0

class MyCell(ol.Cell):
    pos:       ol.vec3
    health:    ol.f32
    energy:    ol.f32
    cell_type: ol.i32

class MyMetrics(ol.Metrics):
    total: ol.u32 = 0

class MyParams(ol.Params):
    rate: ol.f32 = 1.0


def _get_wgsl(sim_cls) -> str:
    engine = ol.Engine(backend="gpu")
    engine.load_model(sim_cls())
    return engine._backend_impl._shader_code


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_kernel(wgsl: str, kernel_name: str) -> str:
    """Extract the full WGSL function body for the given kernel name."""
    pattern = rf"fn {re.escape(kernel_name)}\b[^{{]*\{{(.*?)^\}}"
    m = re.search(pattern, wgsl, re.DOTALL | re.MULTILINE)
    assert m, f"Kernel '{kernel_name}' not found in WGSL:\n{wgsl}"
    return m.group(0)


def _find_all_return_blocks(kernel_body: str) -> list[str]:
    """Return the 3-line window ending at each `return;` statement.

    We collect the two lines immediately preceding each bare `return;`
    so we can assert the epilogue is emitted before every exit point.
    """
    lines = kernel_body.splitlines()
    windows = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "return;":
            preceding = [lines[j].strip() for j in range(max(0, i - 2), i)]
            windows.append(preceding)
    return windows


# =============================================================================
# Early-return epilogue tests
# =============================================================================

class TestRuleEarlyReturnEpilogue:

    def test_single_early_return_has_epilogue(self):
        """A rule with one early return must emit the full epilogue before it."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                if cell.health <= 0.0:
                    return          # ← early exit — epilogue required here
                cell.energy += 1.0  # normal path — epilogue at implicit end

        wgsl   = _get_wgsl(S)
        kernel = _extract_kernel(wgsl, "Kernel_CellRule_0")
        blocks = _find_all_return_blocks(kernel)

        assert blocks, "No explicit return; found in kernel — check extraction."

        for preceding in blocks:
            assert any("_rng_state = _rng" in ln for ln in preceding), (
                f"Missing rng epilogue before return.\nPreceding lines: {preceding}\n\nFull kernel:\n{kernel}"
            )
            assert any("Cells_Out[" in ln for ln in preceding), (
                f"Missing Cells_Out write before return.\nPreceding lines: {preceding}\n\nFull kernel:\n{kernel}"
            )

    def test_multiple_early_returns_all_have_epilogue(self):
        """Every branch exit in a multi-branch rule must carry the epilogue."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                if cell.health <= 0.0:
                    return                      # exit 1
                if cell.energy < 0.5:
                    cell.energy += 0.1
                    return                      # exit 2
                cell.health += self.params.rate  # exit 3 (fall-through)

        wgsl   = _get_wgsl(S)
        kernel = _extract_kernel(wgsl, "Kernel_CellRule_0")
        blocks = _find_all_return_blocks(kernel)

        # Expect at least the two explicit early returns (exit 1 and 2).
        assert len(blocks) >= 2, (
            f"Expected at least 2 explicit return; statements, found {len(blocks)}.\n{kernel}"
        )

        for i, preceding in enumerate(blocks):
            assert any("_rng_state = _rng" in ln for ln in preceding), (
                f"Return #{i+1}: missing rng epilogue.\nPreceding: {preceding}\n\nKernel:\n{kernel}"
            )
            assert any("Cells_Out[" in ln for ln in preceding), (
                f"Return #{i+1}: missing Cells_Out write.\nPreceding: {preceding}\n\nKernel:\n{kernel}"
            )

    def test_nested_if_early_return_has_epilogue(self):
        """Epilogue is required even for a return buried inside nested if blocks."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                if cell.health > 0.0:
                    if cell.energy > 0.5:
                        cell.energy -= 0.1
                        return          # ← deeply nested early exit

        wgsl   = _get_wgsl(S)
        kernel = _extract_kernel(wgsl, "Kernel_CellRule_0")
        blocks = _find_all_return_blocks(kernel)

        assert blocks, "No explicit return; found in kernel."

        for preceding in blocks:
            assert any("_rng_state = _rng" in ln for ln in preceding), (
                f"Nested return missing rng epilogue.\nPreceding: {preceding}\n\nKernel:\n{kernel}"
            )
            assert any("Cells_Out[" in ln for ln in preceding), (
                f"Nested return missing Cells_Out write.\nPreceding: {preceding}\n\nKernel:\n{kernel}"
            )
