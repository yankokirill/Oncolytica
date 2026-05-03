"""
test_simulation_base_params.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for error correctness when specifying Simulation type parameters.

All checks happen at class declaration time (class MySim(...)),
i.e., BEFORE calling engine.load_model() — these are TypeErrors from __init_subclass__,
not CompilationErrors from the validator pipeline.

Covered cases
------------------
  A  — ForwardRef: strings instead of classes
  B  — Wrong base class for each of the 5 slots
  C  — Slot swapping (Cell instead of Tissue and vice versa)
  D  — Correct declaration order passes without errors
"""

import pytest
import oncolytica as ol


# ── Reference classes (declared before any Simulation) ─────────────────────────

class GoodTissue(ol.Tissue):
    oxygen: ol.f32 = 1.0

class GoodChem(ol.Chemistry):
    drug: ol.f32 = 0.0

class GoodCell(ol.Cell):
    pos:    ol.vec3
    health: ol.f32

class GoodMetrics(ol.Metrics):
    count: ol.u32 = 0

class GoodParams(ol.Params):
    rate: ol.f32 = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A — ForwardRef: strings instead of classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestForwardRef:
    """
    ol.Simulation["MyTissue", ...] passes strings (ForwardRef) instead of types.
    __init_subclass__ should catch this immediately with a clear message
    — not AttributeError on __name__, not silent acceptance.
    """

    def test_string_tissue_param_raises(self):
        """String instead of Tissue class raises TypeError mentioning forward reference."""
        with pytest.raises(TypeError, match="forward reference|string|Define all data classes"):
            class BadSim(ol.Simulation["GoodTissue", GoodChem, GoodCell, GoodMetrics, GoodParams]):
                pass

    def test_string_cell_param_raises(self):
        with pytest.raises(TypeError, match="forward reference|string|Define all data classes"):
            class BadSim(ol.Simulation[GoodTissue, GoodChem, "GoodCell", GoodMetrics, GoodParams]):
                pass

    def test_all_string_params_raise(self):
        """All five parameters are strings."""
        with pytest.raises(TypeError, match="forward reference|string"):
            class BadSim(ol.Simulation[
                "GoodTissue", "GoodChem", "GoodCell", "GoodMetrics", "GoodParams"
            ]):
                pass

    def test_forward_ref_error_not_attribute_error(self):
        """
        Regression test: before the fix, this would raise AttributeError ('ForwardRef' object
        has no attribute '__name__'). Now it should raise a clean TypeError.
        """
        with pytest.raises(TypeError):
            # Reproduce the original pattern from gbm_model.py:
            # SimBase is created with strings, then used as a base class.
            _T = "GoodTissue"
            _C = "GoodChem"
            _A = "GoodCell"
            _M = "GoodMetrics"
            _P = "GoodParams"
            SimBase = ol.Simulation[_T, _C, _A, _M, _P]

            class BadSim(SimBase):
                pass

    def test_none_param_raises(self):
        """None instead of a class — also not a type."""
        with pytest.raises(TypeError):
            class BadSim(ol.Simulation[None, GoodChem, GoodCell, GoodMetrics, GoodParams]):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B — Wrong base class for each slot
# ═══════════════════════════════════════════════════════════════════════════════

class TestWrongBaseClass:
    """
    Each Simulation parameter expects a specific framework base class.
    Passing a class with the wrong hierarchy should raise TypeError indicating
    which base class was expected.
    """

    def test_wrong_tissue_slot_gives_clear_error(self):
        """Slot 1 (Tissue): passing Chemistry — should get a clear error."""
        with pytest.raises(TypeError, match="Tissue|parameter 1"):
            class BadSim(ol.Simulation[GoodChem, GoodChem, GoodCell, GoodMetrics, GoodParams]):
                pass

    def test_wrong_chem_slot_gives_clear_error(self):
        """Slot 2 (Chemistry): passing Tissue."""
        with pytest.raises(TypeError, match="Chemistry|parameter 2"):
            class BadSim(ol.Simulation[GoodTissue, GoodTissue, GoodCell, GoodMetrics, GoodParams]):
                pass

    def test_wrong_cell_slot_gives_clear_error(self):
        """Slot 3 (Cell): passing Tissue."""
        with pytest.raises(TypeError, match="Cell|parameter 3"):
            class BadSim(ol.Simulation[GoodTissue, GoodChem, GoodTissue, GoodMetrics, GoodParams]):
                pass

    def test_wrong_metrics_slot_gives_clear_error(self):
        """Slot 4 (Metrics): passing Cell."""
        with pytest.raises(TypeError, match="Metrics|parameter 4"):
            class BadSim(ol.Simulation[GoodTissue, GoodChem, GoodCell, GoodCell, GoodParams]):
                pass

    def test_wrong_params_slot_gives_clear_error(self):
        """Slot 5 (Params): passing Metrics."""
        with pytest.raises(TypeError, match="Params|parameter 5"):
            class BadSim(ol.Simulation[GoodTissue, GoodChem, GoodCell, GoodMetrics, GoodCell]):
                pass

    def test_plain_object_in_tissue_slot_raises(self):
        """Arbitrary Python class (not from framework) in Tissue slot."""
        class JustAPythonClass:
            pass

        with pytest.raises(TypeError, match="Tissue|subclass"):
            class BadSim(ol.Simulation[
                JustAPythonClass, GoodChem, GoodCell, GoodMetrics, GoodParams
            ]):
                pass

    def test_error_message_contains_example(self):
        """Error message should contain a hint/example."""
        with pytest.raises(TypeError, match=r"ol\.Simulation\["):
            class BadSim(ol.Simulation[GoodChem, GoodChem, GoodCell, GoodMetrics, GoodParams]):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C — Slot swaps
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlotSwaps:
    """
    Typical error: Cell and Tissue are swapped,
    or Chemistry is passed where Cell is expected.
    """

    def test_tissue_and_cell_swapped(self):
        """GoodCell in Tissue slot, GoodTissue in Cell slot."""
        with pytest.raises(TypeError):
            class BadSim(ol.Simulation[GoodCell, GoodChem, GoodTissue, GoodMetrics, GoodParams]):
                pass

    def test_metrics_and_params_swapped(self):
        """GoodParams in Metrics slot, GoodMetrics in Params slot."""
        with pytest.raises(TypeError):
            class BadSim(ol.Simulation[GoodTissue, GoodChem, GoodCell, GoodParams, GoodMetrics]):
                pass

    def test_all_slots_shifted_by_one(self):
        """All parameters shifted left by one position (rotated)."""
        with pytest.raises(TypeError):
            class BadSim(ol.Simulation[GoodChem, GoodCell, GoodMetrics, GoodParams, GoodTissue]):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D — Correct declaration order: positive tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectDeclaration:
    """
    Ensure that when the order is correct (classes declared before Simulation)
    no errors occur.
    """

    def test_correct_order_no_error(self):
        """Standard correct order causes no errors."""
        class GoodSim(ol.Simulation[GoodTissue, GoodChem, GoodCell, GoodMetrics, GoodParams]):
            @ol.cell_rule
            def rule(self, cell: GoodCell):
                cell.health = 1.0

        engine = ol.Engine(backend="cpu")
        engine.load_model(GoodSim())  # should not raise

    def test_subclasses_of_base_classes_are_accepted(self):
        """Subclasses of framework base classes should be accepted."""
        class ExtendedTissue(ol.Tissue):
            density: ol.f32 = 1.0

        class ExtendedCell(ol.Cell):
            pos:    ol.vec3
            energy: ol.f32 = 100.0

        class GoodSim(ol.Simulation[
            ExtendedTissue, GoodChem, ExtendedCell, GoodMetrics, GoodParams
        ]):
            @ol.cell_rule
            def rule(self, cell: ExtendedCell):
                cell.energy -= 1.0

        engine = ol.Engine(backend="cpu")
        engine.load_model(GoodSim())

    def test_declaring_simbase_alias_after_classes_works(self):
        """
        Pattern with alias is allowed if the alias is created AFTER all
        data classes are declared.
        """
        class LateCell(ol.Cell):
            pos:    ol.vec3
            health: ol.f32

        # Alias is created with real types — no ForwardRef.
        SimAlias = ol.Simulation[GoodTissue, GoodChem, LateCell, GoodMetrics, GoodParams]

        class GoodSim(SimAlias):
            @ol.cell_rule
            def rule(self, cell: LateCell):
                cell.health = 0.0

        engine = ol.Engine(backend="cpu")
        engine.load_model(GoodSim())
