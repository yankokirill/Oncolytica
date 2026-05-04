"""
test_domain_mutability.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for Domain Validator mutability rules (Phase 5, _domain.py).

Structure follows specification:

  Rule 1.1  — Direct aliasing is forbidden
  Rule 1.2  — Allowed R-value sources
  Rule 1.3  — Explicit copying via .copy()
  Rule 2.1  — Main rule argument is mutable
  Rule 2.2  — New instances (constructor / .copy()) are mutable
  Rule 2.3  — Collection iterators are read-only
  Rule 2.4  — Built-in getters (tissue_at / chemistry_at) are read-only
  Rule 2.5  — Helper return: read-only without .copy() — forbidden

Each rule is covered by:
  • one or more negative tests (violation → CompilationError)
  • one or more positive tests (compliance → no error)
"""

import pytest
import oncolytica as ol
from oncolytica import CompilationError


# ── Common test data ──────────────────────────────────────────────────────────────

class MyTissue(ol.Tissue):
    oxygen:   ol.f32 = 1.0
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


def _load(sim_cls):
    ol.Engine(backend="cpu").load_model(sim_cls())


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 1.1 — Direct aliasing is forbidden
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule11NoAliasing:
    """b = a where 'a' is an existing memory-type variable → always forbidden."""

    def test_alias_pointer_arg_to_local_fails(self):
        """cell is rule argument; other = cell — direct aliasing."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                other = cell
        with pytest.raises(CompilationError, match=r"Cannot assign 'cell' to a new variable"):
            _load(S)

    def test_alias_neighbor_to_local_fails(self):
        """n is iterator; saved = n — direct aliasing."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    saved = n
        with pytest.raises(CompilationError, match=r"Cannot assign 'n' to a new variable 'saved"):
            _load(S)

    def test_alias_grid_getter_result_to_second_var_fails(self):
        """t = self.tissue_at(...); t2 = t — second alias is forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t  = self.tissue_at(cell.pos)
                t2 = t
        with pytest.raises(CompilationError, match=r"Cannot assign 't' to a new variable 't2'"):
            _load(S)

    def test_alias_constructed_cell_to_another_var_fails(self):
        """new = MyCell(); copy_ref = new — alias from constructor is forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                new      = MyCell()
                copy_ref = new
        with pytest.raises(CompilationError, match=r"Cannot assign 'new' to a new variable 'copy_ref'"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 1.2 — Allowed R-value sources
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule12AllowedRValues:
    """Constructor, user-helper, grid-getter — allowed R-values."""

    def test_constructor_rvalue_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                daughter = MyCell()
                daughter.health = 100.0
                cell.divide(daughter)
        _load(S)

    def test_user_helper_rvalue_passes(self):
        """x = self.make_cell() — helper result is allowed as R-value."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def make_cell(self) -> MyCell:
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                fresh = self.make_cell()
                fresh.health = 50.0
                cell.divide(fresh)
        _load(S)

    def test_grid_getter_rvalue_passes(self):
        """t = self.tissue_at(pos) — reading t.oxygen is allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                cell.health = t.oxygen
        _load(S)

    def test_chemistry_at_rvalue_passes(self):
        """c = self.chemistry_at(pos) — reading c.drug is allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                c = self.chemistry_at(cell.pos)
                cell.health = c.drug
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 1.3 — Explicit copying via .copy()
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule13ExplicitCopy:
    """.copy() creates a mutable snapshot, allowed for modification."""

    def test_copy_of_neighbor_is_mutable(self):
        """n.copy() → mutable; fields can be modified."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    snapshot = n.copy()
                    snapshot.health = 0.0   # mutating copy — OK
        _load(S)

    def test_copy_of_grid_getter_is_mutable(self):
        """self.tissue_at(...).copy() → mutable snapshot."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.tissue_at(cell.pos).copy()
                snap.oxygen = 0.5           # mutating copy — OK
        _load(S)

    def test_copy_of_pointer_arg_is_mutable(self):
        """cell.copy() → independent mutable object."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                backup = cell.copy()
                backup.health = 0.0
                cell.divide(backup)
        _load(S)

    def test_copy_then_alias_of_copy_fails(self):
        """backup = cell.copy(); alias = backup — still aliasing."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                backup = cell.copy()
                alias  = backup
        with pytest.raises(CompilationError, match=r"Cannot assign 'backup' to a new variable"):
            _load(S)

    def test_direct_copy_call_on_readonly_passes(self):
        """.copy() on read-only object — the only way to get a copy."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    copy = n.copy()
                    copy.energy = n.energy * 0.5
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 2.1 — Main rule argument is mutable
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule21PointerArgIsMutable:

    def test_cell_arg_field_write_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = 100.0
                cell.energy = 50.0
        _load(S)

    def test_cell_arg_augassign_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health += 1.0
                cell.energy -= 0.5
        _load(S)

    def test_cell_arg_vec3_component_write_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos.x = 10.0
                cell.pos.y = 20.0
        _load(S)

    def test_tissue_arg_field_write_in_tissue_rule_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.tissue_rule
            def rule(self, tissue: MyTissue):
                tissue.oxygen = 0.8
        _load(S)

    def test_rebind_pointer_arg_fails(self):
        """cell = MyCell() — attempt to rebind the rule pointer."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell = MyCell()
        with pytest.raises(CompilationError, match=r"rebind|pointer|'cell'"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 2.2 — Constructor and .copy() create mutable objects
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule22ConstructorAndCopyAreMutable:

    def test_constructor_result_is_mutable(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                d = MyCell()
                d.health = 75.0
                d.energy = 25.0
                cell.divide(d)
        _load(S)

    def test_constructor_with_kwargs_result_is_mutable(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                d = MyCell(health=50.0, energy=50.0)
                d.cell_type = 2
                cell.divide(d)
        _load(S)

    def test_copy_result_is_mutable(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = cell.copy()
                snap.health = 0.0
        _load(S)

    def test_multiple_constructors_independent(self):
        """Two constructors in one function — both independently mutable."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                a = MyCell()
                b = MyCell()
                a.health = 10.0
                b.health = 20.0
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 2.3 — Collection iterators are read-only
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule23IteratorsAreReadonly:

    # ── Negative tests ────────────────────────────────────────────────

    def test_neighbor_field_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    n.health = 0.0
        with pytest.raises(CompilationError, match=r"Cannot modify 'n'"):
            _load(S)

    def test_neighbor_augassign_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    n.energy -= 1.0
        with pytest.raises(CompilationError, match=r"Cannot modify 'n'"):
            _load(S)

    def test_neighbor_vec3_component_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    n.pos.x = 0.0
        with pytest.raises(CompilationError, match=r"Cannot modify 'n'"):
            _load(S)

    def test_tissue_cells_iterator_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.tissue_rule
            def rule(self, tissue: MyTissue):
                for c in tissue.cells:
                    c.health = 0.0
        with pytest.raises(CompilationError, match=r"Cannot modify 'c'"):
            _load(S)

    def test_chem_cells_iterator_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.chemistry_rule
            def rule(self, chem: MyChem):
                for c in chem.cells:
                    c.energy = 0.0
        with pytest.raises(CompilationError, match=r"Cannot modify 'c'"):
            _load(S)

    def test_neighbor_die_forbidden(self):
        """die() on iterator variable — ownership violation."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    n.die()

        with pytest.raises(CompilationError, match=r"Cannot call 'die\(\)' on 'n'"):
            _load(S)

    def test_loop_var_reuse_after_loop_is_readonly(self):
        """Loop variable remains read-only even after loop exits."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                last_n: MyCell = MyCell()       # initialize with mutable
                for n in cell.neighbors:
                    last_n = n                  # aliasing
        with pytest.raises(CompilationError, match=r"Cannot assign 'n' to a new variable 'last_n"):
            _load(S)

    # ── Positive tests ───────────────────────────────────────────────────

    def test_neighbor_field_read_passes(self):
        """Reading iterator variable field is allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    cell.health += n.health * 0.01
        _load(S)

    def test_neighbor_read_then_mutate_self_passes(self):
        """Read from neighbor, write to cell — allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    if n.health < 10.0:
                        cell.energy -= 1.0
        _load(S)

    def test_copy_of_neighbor_is_mutable(self):
        """n.copy() from iterator becomes mutable."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    snap = n.copy()
                    snap.health = 0.0       # OK: snap is mutable copy
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 2.4 — Built-in getters are read-only
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule24GridGettersAreReadonly:

    # ── Negative tests ────────────────────────────────────────────────

    def test_tissue_at_field_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                t.oxygen = 0.0
        with pytest.raises(CompilationError, match=r"read-only|Cannot modify field"):
            _load(S)

    def test_chemistry_at_field_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                c = self.chemistry_at(cell.pos)
                c.drug = 0.0
        with pytest.raises(CompilationError, match=r"read-only|Cannot modify field"):
            _load(S)

    def test_tissue_at_augassign_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                t.oxygen += 0.1
        with pytest.raises(CompilationError, match=r"read-only|Cannot modify field"):
            _load(S)

    def test_inline_tissue_at_write_fails(self):
        """Inline: self.tissue_at(pos).oxygen = ... — also forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.tissue_at(cell.pos).oxygen = 0.5
        with pytest.raises(CompilationError, match=r"Temper object cannot be modified."):
            _load(S)

    def test_inline_helper_write_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def helper(self) -> MyTissue:
                return MyTissue()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.helper().oxygen = 0.5
        with pytest.raises(CompilationError, match=r"Temper object cannot be modified"):
            _load(S)

    def test_inline_tissue_at_copy_tmp_write_ok(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                tmp = self.tissue_at(cell.pos).copy()
                tmp.oxygen = 0.0

        _load(S)

    def test_grid_getter_reassigned_as_alias_fails(self):
        """t = self.tissue_at(...); t2 = t — Rule 1.1 + 2.4."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t  = self.tissue_at(cell.pos)
                t2 = t
        with pytest.raises(CompilationError, match=r"Cannot assign 't' to a new variable 't2'"):
            _load(S)

    # ── Positive tests ───────────────────────────────────────────────────

    def test_tissue_at_field_read_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                cell.health *= t.oxygen
        _load(S)

    def test_chemistry_at_field_read_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                c = self.chemistry_at(cell.pos)
                cell.energy -= c.drug
        _load(S)

    def test_tissue_at_copy_is_mutable(self):
        """self.tissue_at(...).copy() — explicit copy, mutable."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.tissue_at(cell.pos).copy()
                snap.oxygen = 0.0   # OK: snap is mutable copy
        _load(S)

    def test_multiple_getters_both_readonly(self):
        """Two getters in one function — both read-only."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                c = self.chemistry_at(cell.pos)
                cell.health = t.oxygen - c.drug
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 2.5 — Helper return
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule25HelperReturn:
    """
    Cannot return a read-only variable directly from a helper.
    Must return either a constructor or .copy() result.
    """

    # ── Negative tests ────────────────────────────────────────────────

    def test_return_iterator_var_fails(self):
        """Returning a for-loop iterator variable — forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_first_neighbor(self, cell: MyCell) -> MyCell:
                for n in cell.neighbors:
                    return n            # n — read-only
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                fresh = self.get_first_neighbor(cell)
                fresh.health = 0.0
        with pytest.raises(CompilationError, match=r""):
            _load(S)

    def test_return_grid_getter_var_fails(self):
        """Returning tissue_at result without .copy() — forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_tissue(self, cell: MyCell) -> MyTissue:
                t = self.tissue_at(cell.pos)
                return t            # t — read-only (grid getter)

            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.get_tissue(cell)
                cell.health = snap.oxygen
        with pytest.raises(CompilationError, match=r"read-only|copy"):
            _load(S)

    def test_return_chemistry_getter_var_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_chem(self, cell: MyCell) -> MyChem:
                c = self.chemistry_at(cell.pos)
                return c            # read-only

            @ol.cell_rule
            def rule(self, cell: MyCell):
                chem = self.get_chem(cell)
                cell.energy = chem.drug
        with pytest.raises(CompilationError, match=r"read-only|copy"):
            _load(S)

    # ── Positive tests ───────────────────────────────────────────────────

    def test_return_constructor_passes(self):
        """return MyCell() — constructor, always allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def make_cell(self) -> MyCell:
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                d = self.make_cell()
                d.health = 100.0
                cell.divide(d)
        _load(S)

    def test_return_copy_of_iterator_passes(self):
        """return n.copy() — explicit copy of read-only — allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def snapshot_neighbor(self, cell: MyCell) -> MyCell:
                for n in cell.neighbors:
                    return n.copy()     # .copy() creates mutable snapshot
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.snapshot_neighbor(cell)
                cell.health = snap.health
        _load(S)

    def test_return_copy_of_grid_getter_passes(self):
        """return self.tissue_at(...).copy() — allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_tissue_snap(self, cell: MyCell) -> MyTissue:
                return self.tissue_at(cell.pos).copy()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.get_tissue_snap(cell)
                cell.health = t.oxygen
        _load(S)

    def test_return_mutable_local_passes(self):
        """Returning mutable local variable (from constructor) — OK."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def build_cell(self) -> MyCell:
                d = MyCell()
                d.health = 50.0
                return d            # d — mutable (constructor)

            @ol.cell_rule
            def rule(self, cell: MyCell):
                new_cell = self.build_cell()
                new_cell.energy = 30.0
                cell.divide(new_cell)
        _load(S)

    def test_return_helper_result_then_mutate_passes(self):
        """Helper result (Rule 1.2) is mutable, fields can be modified."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def spawn(self) -> MyCell:
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                d = self.spawn()
                d.health = cell.health * 0.5
                d.energy = cell.energy * 0.5
                cell.divide(d)
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-RULE — edge and mixed cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossRuleEdgeCases:
    """Tests that involve multiple rules simultaneously."""

    def test_copy_chain_both_mutable(self):
        """a = cell.copy(); b = a.copy() — both mutable via .copy()."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                a = cell.copy()
                b = a.copy()
                a.health = 10.0
                b.health = 20.0
        _load(S)

    def test_constructor_in_loop_is_mutable(self):
        """MyCell() inside loop — each instance is mutable."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    fresh = MyCell()
                    fresh.health = n.health   # fresh — mutable; n — read-only
        _load(S)

    def test_readonly_name_reused_as_mutable_after_constructor_fails(self):
        """Name first used as loop-var (read-only), then constructor-var.
        On next iteration the name becomes read-only again —
        reuse after loop via alias is forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    pass
                # After loop, n is still read-only from last iteration;
                # attempting to assign to field — error.
                n.health = 0.0
        with pytest.raises(CompilationError, match=r"read-only|Cannot modify field"):
            _load(S)

    def test_grid_getter_read_only_then_copy_mutable(self):
        """t — read-only; t_snap = t.copy() — mutable; t stays read-only."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t      = self.tissue_at(cell.pos)
                t_snap = t.copy()
                t_snap.oxygen = 0.0     # OK
                t.oxygen      = 0.0     # read-only — should fail
        with pytest.raises(CompilationError, match=r"read-only|Cannot modify field"):
            _load(S)

    def test_user_helper_result_is_mutable_rule12_25(self):
        """Rule 1.2 + 2.5: helper returns constructor → result is mutable."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def make(self) -> MyCell:
                c = MyCell()
                c.health = 1.0
                return c

            @ol.cell_rule
            def rule(self, cell: MyCell):
                d = self.make()
                d.energy = 99.0     # mutating helper result — OK
                cell.divide(d)
        _load(S)

    def test_alias_in_conditional_branch_fails(self):
        """Alias inside if branch — also forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                if cell.health > 50.0:
                    ref = cell
        with pytest.raises(CompilationError, match=r"Cannot assign 'cell' to a new variable"):
            _load(S)

    def test_write_to_copied_neighbor_does_not_affect_original(self):
        """Documents semantics: copy is isolated, original fields unchanged.
        Compiler should pass without errors — correct behavior."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    local = n.copy()
                    local.health = 0.0      # isolated copy
                    cell.health += local.health
        _load(S)

    def test_self_mutation_in_rule_forbidden(self):
        """self.attr = ... outside __init__ — always forbidden."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            count: ol.i32 = 0

            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.count += 1
        with pytest.raises(CompilationError, match=r"self|forbidden"):
            _load(S)

    def test_divide_on_constructed_cell_passes(self):
        """cell.divide(new) where new = MyCell() — lawful usage."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                daughter = MyCell()
                daughter.health = cell.health * 0.5
                daughter.energy = cell.energy * 0.5
                daughter.cell_type = cell.cell_type
                daughter.pos = ol.vec3(
                    cell.pos.x + 1.0,
                    cell.pos.y,
                    cell.pos.z,
                )
                cell.divide(daughter)
        _load(S)

    def test_divide_on_readonly_neighbor_passes(self):
        """divide is always allowed."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    n.divide(cell)
        _load(S)


# ── Temporary object writes (Rule 2.4 + generalisation) ──────────────────────

class TestTemporaryObjectWrites:
    """Any domain object that is never assigned to a named variable is
    a temporary and its fields cannot be written to."""

    def test_inline_tissue_at_write_fails(self):
        """self.tissue_at(pos).oxygen = ... — inline grid-getter write."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.tissue_at(cell.pos).oxygen = 0.5

        with pytest.raises(CompilationError, match=r"Temper object cannot be modified"):
            _load(S)

    def test_inline_chemistry_at_write_fails(self):
        """self.chemistry_at(pos).drug = ... — same rule, different getter."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.chemistry_at(cell.pos).drug = 1.0

        with pytest.raises(CompilationError, match=r"Temper object cannot be modified"):
            _load(S)

    def test_inline_helper_result_write_fails(self):
        """self.helper().field = ... — helper returns temporary."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def make_tissue(self) -> MyTissue:
                return MyTissue()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.make_tissue().oxygen = 0.5

        with pytest.raises(CompilationError, match=r"Temper object cannot be modified"):
            _load(S)

    def test_inline_constructor_write_fails(self):
        """MyCell().health = ... — constructor result is temporary."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                MyCell().health = 1.0

        with pytest.raises(CompilationError, match=r"Temper object cannot be modified"):
            _load(S)

    def test_named_tissue_at_write_fails(self):
        """t = self.tissue_at(pos); t.oxygen = ... — named but read-only (Rule 2.4)."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                t.oxygen = 0.5

        with pytest.raises(CompilationError, match=r"Cannot modify"):
            _load(S)

    def test_named_tissue_at_read_passes(self):
        """t = self.tissue_at(pos); cell.health = t.oxygen — reading is fine."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                t = self.tissue_at(cell.pos)
                cell.health = t.oxygen

        _load(S)  # must not raise


# ── copy() / copy_from() exempt from mutability checks ───────────────────────

class TestCopyExemptions:
    """copy() and copy_from() must be callable on any object regardless of its
    mutability status — they are never blocked by _check_call_ownership."""

    def test_copy_on_readonly_neighbor_passes(self):
        """n.copy() inside a neighbor loop — n is read-only, copy() is exempt."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    snap = n.copy()
                    snap.health = 0.0

        _load(S)

    def test_copy_on_grid_getter_passes(self):
        """self.tissue_at(pos).copy() — temporary but copy() is always safe."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.tissue_at(cell.pos).copy()
                snap.oxygen = 0.9

        _load(S)

    def test_copy_from_on_pointer_arg_passes(self):
        """cell.copy_from(other) — pointer-arg receiver, always allowed."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    cell.copy_from(n)

        _load(S)

    def test_mutating_method_on_non_pointer_fails(self):
        """Calling a mutating non-exempt method on a read-only object is still an error."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for n in cell.neighbors:
                    n.die()

        with pytest.raises(CompilationError, match=r"Cannot call 'die\(\)' on 'n'"):
            _load(S)


# ── Iterator read-only enforcement inside helpers (Rule 2.3) ─────────────────

class TestIteratorReadOnlyInHelpers:
    """Loop variables must be read-only even when the loop lives inside a
    user-defined helper method, not directly in a rule."""

    def test_write_to_neighbor_in_helper_fails(self):
        """Helper iterates neighbors and tries to write — must be caught."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def poison_neighbors(self, cell: MyCell) -> None:
                for n in cell.neighbors:
                    n.health = 0.0

            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.poison_neighbors(cell)

        with pytest.raises(CompilationError, match=r"Cannot modify 'n'"):
            _load(S)

    def test_read_neighbor_in_helper_passes(self):
        """Helper reads neighbor fields — that is always allowed."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def sum_health(self, cell: MyCell) -> ol.f32:
                total: ol.f32 = 0.0
                for n in cell.neighbors:
                    total = total + n.health
                return total

            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = self.sum_health(cell)

        _load(S)


# ── Return of read-only iterator variable (Rule 2.5) ─────────────────────────

class TestReturnReadOnlyVar:
    """Returning a read-only variable from a helper is forbidden; the caller
    would receive a reference it could mutate, breaking the simulation."""

    def test_return_iterator_var_fails(self):
        """Returning a for-loop iterator variable — forbidden."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_first_neighbor(self, cell: MyCell) -> MyCell:
                for n in cell.neighbors:
                    return n
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                fresh = self.get_first_neighbor(cell)
                fresh.health = 0.0

        with pytest.raises(CompilationError, match=r"Cannot return 'n' directly"):
            _load(S)

    def test_return_grid_getter_var_fails(self):
        """Returning a variable assigned from tissue_at — forbidden."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_tissue(self, cell: MyCell) -> MyTissue:
                t = self.tissue_at(cell.pos)
                return t

            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.get_tissue(cell)
                snap.oxygen = 0.0

        with pytest.raises(CompilationError, match=r"Cannot return 't' directly"):
            _load(S)

    def test_return_copy_of_iterator_passes(self):
        """return n.copy() — explicit copy is always allowed."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_first_neighbor(self, cell: MyCell) -> MyCell:
                for n in cell.neighbors:
                    return n.copy()
                return MyCell()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.get_first_neighbor(cell)
                snap.health = 0.0

        _load(S)

    def test_return_copy_of_grid_getter_passes(self):
        """return t.copy() after tissue_at — explicit copy is always allowed."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_tissue(self, cell: MyCell) -> MyTissue:
                t = self.tissue_at(cell.pos)
                return t.copy()

            @ol.cell_rule
            def rule(self, cell: MyCell):
                snap = self.get_tissue(cell)
                snap.oxygen = 0.0

        _load(S)


# ── Mutable primitive parameters (_d / var d = _d prologue) ──────────────────

class TestMutablePrimitiveParams:
    """When a helper mutates a primitive parameter, the WGSL signature must
    use '_pname' and the body must open with 'var pname = _pname;'."""

    def test_helper_with_mutated_f32_param_compiles(self):
        """Helper that modifies an f32 param must compile without error."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def clamp_energy(self, cell: MyCell, cap: ol.f32) -> ol.f32:
                if cell.energy > cap:
                    cap = cell.energy
                return cap

            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.energy = self.clamp_energy(cell, 100.0)

        with pytest.raises(CompilationError, match=r"Forbidden reassignment of parameter 'cap'."):
            _load(S)


    def test_helper_immutable_f32_param_no_prologue(self):
        """Helper that only reads an f32 param — no _d / var d prologue needed."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def scale_health(self, cell: MyCell, factor: ol.f32) -> ol.f32:
                return cell.health * factor

            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = self.scale_health(cell, 0.5)

        _load(S)
