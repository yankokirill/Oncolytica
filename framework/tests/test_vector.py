"""
Tests for vec3/ivec3 mutability validation (Rules 1.1, 2.2, 2.3).

Vectors share the same ownership rules as domain objects (Cell, Tissue, etc.):
  - Rule 1.1  alias assignment forbidden   (b = a  where a is an existing vec)
  - Rule 2.2  constructor/copy → mutable   (b = ivec3(); b = a.copy())
  - Rule 2.3  loop iterators → read-only   (for v in ...: v.x = 1  is an error)
"""
import pytest
import oncolytica as ol
from oncolytica.core.validation._validator import ValidatorEngine
from oncolytica.core.utils._errors import CompilationError


# ── Shared domain fixtures ────────────────────────────────────────────────────

class Tissue(ol.Tissue):
    oxygen: ol.f32 = 1.0

class Chem(ol.Chemistry):
    drug: ol.f32 = 0.0

class Cell(ol.Cell):
    pos:    ol.vec3
    health: ol.f32
    coord:  ol.ivec3

class Metrics(ol.Metrics):
    total: ol.u32 = 0

class Params(ol.Params):
    rate: ol.f32 = 1.0


def _validate(sim_cls):
    """Run the full validator pipeline and return the context (or raise)."""
    return ValidatorEngine().run(sim_cls())


# =============================================================================
# Rule 1.1 — alias assignment forbidden for vec3 / ivec3
# =============================================================================

class TestVectorAliasRule:
    """Direct assignment of one vector variable to another is forbidden (Rule 1.1)."""

    def test_vec3_alias_raises(self):
        """b = a  where a is a vec3 must raise CompilationError."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                a = cell.pos        # read-only from field
                b = a               # ← alias: forbidden

        with pytest.raises(CompilationError, match="alias|assign|copy"):
            _validate(Sim)

    def test_ivec3_alias_raises(self):
        """b = a  where a is an ivec3 must raise CompilationError."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                a = cell.coord

        with pytest.raises(CompilationError, match="assign|copy"):
            _validate(Sim)

    def test_vec3_copy_is_allowed(self):
        """b = a.copy()  must pass validation (explicit copy, Rule 1.3)."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                a = cell.pos.copy()
                b = a.copy()        # ← explicit copy: allowed
                b.x = 1.0

        _validate(Sim)  # must not raise

    def test_ivec3_constructor_is_allowed(self):
        """b = ivec3(1, 2, 3)  must pass validation (constructor, Rule 2.2)."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                v = ol.ivec3(1, 2, 3)   # ← constructor: allowed
                v.x = 0

        _validate(Sim)  # must not raise

    def test_vec3_constructor_is_allowed(self):
        """b = vec3(...)  must pass validation."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                v = ol.vec3(0.0, 1.0, 0.0)
                v.y = 2.0

        _validate(Sim)  # must not raise

    def test_vec3_returned_from_helper_is_allowed(self):
        """b = self.make_pos()  (helper return) must pass validation (Rule 1.2)."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def make_pos(self) -> ol.vec3:
                return ol.vec3(0.0, 0.0, 0.0)

            @ol.cell_rule
            def rule(self, cell: Cell):
                v = self.make_pos()   # ← self_call: allowed
                v.x = 1.0

        _validate(Sim)  # must not raise


# =============================================================================
# Rule 2.3 — for-loop iterator variables are read-only
# =============================================================================

class TestVectorLoopReadOnly:
    """Loop variables whose type is vec3/ivec3 must be treated as read-only."""

    def test_mutate_vec3_loop_var_raises(self):
        """Writing to a vec3 loop variable's field must raise CompilationError."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                for n in cell.neighbors:
                    n.pos.x = 0.0   # ← n is read-only

        with pytest.raises(CompilationError):
            _validate(Sim)

    def test_read_vec3_loop_var_is_allowed(self):
        """Reading a vec3 field of a loop variable must pass validation."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                for n in cell.neighbors:
                    x = n.pos.x     # ← read only: allowed
                    cell.health += x

        _validate(Sim)  # must not raise


# =============================================================================
# Rule 2.2 — copy() of a read-only vector produces a mutable result
# =============================================================================

class TestVectorCopyMutability:
    """A copy of a read-only vector is itself mutable (Rule 2.2)."""

    def test_copy_of_readonly_vec3_is_mutable(self):
        """Copying a read-only vec3 (e.g. cell.pos) and mutating the copy must pass."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                p = cell.pos.copy()
                p.x += 1.0          # ← copy is mutable: allowed

        _validate(Sim)  # must not raise

    def test_mutate_vec3_field_directly_on_main_arg_is_allowed(self):
        """The rule's main arg is always mutable — mutating its vec3 field is fine."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                cell.pos.x = 1.0    # ← cell is mutable (Rule 2.1): allowed

        _validate(Sim)  # must not raise
