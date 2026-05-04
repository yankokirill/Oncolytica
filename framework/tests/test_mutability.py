"""
Tests for pointer/value codegen:
- mutating domain params → ptr<function, T> in helper signature
- mutating domain args   → &var at call site
- ptr-params inside body → (*param).field
- non-mutating and primitive params → plain value passing
"""
import re
import pytest
import oncolytica as ol
from oncolytica.core.validation._validator import ValidatorEngine
from oncolytica.gpu.compiler._compiler import _compile_helper_fn, WGSLCompiler


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_wgsl(sim_cls) -> str:
    """Compile sim_cls and return the full WGSL shader string."""
    engine = ol.Engine(backend="gpu")
    engine.load_model(sim_cls())
    return engine._backend_impl._shader_code  # adjust attribute name if different in your Engine


def _compile_sim_helper(sim_instance, method_name: str) -> str:
    """Compile a single sim-level helper and return the WGSL function string."""
    val_ctx = ValidatorEngine().run(sim_instance)
    method  = getattr(sim_instance, method_name)
    wgsl, _ = _compile_helper_fn(
        method,
        uniforms_map={},
        val_ctx=val_ctx,
        wgsl_fn_name=f"sim_{method_name}",
    )
    return wgsl


# =============================================================================
# Helper function signatures
# =============================================================================

class TestHelperSignatures:
    """The WGSL signature of a compiled helper must use ptr<function, T>
    for mutating domain parameters and plain T for read-only ones."""

    def test_mutating_cell_param_becomes_ptr(self):
        """Helper that mutates its Cell arg → ptr<function, Cell> in signature."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def buff(self, c: MyCell, amount: ol.f32):
                c.energy += amount

            @ol.cell_rule
            def rule(self, cell: MyCell):
                tmp = cell.copy()
                self.buff(tmp, 5.0)

        wgsl = _compile_sim_helper(S(), "buff")
        assert "c: ptr<function, Cell>" in wgsl, (
            f"Expected 'c: ptr<function, Cell>' in signature, got:\n{wgsl}"
        )

    def test_mutating_tissue_param_becomes_ptr(self):
        """Helper that mutates its Tissue arg → ptr<function, Tissue>."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def heal(self, t: MyTissue, amount: ol.f32):
                t.oxygen += amount

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass  # validator requires a rule

        wgsl = _compile_sim_helper(S(), "heal")
        assert "t: ptr<function, Tissue>" in wgsl, (
            f"Expected 't: ptr<function, Tissue>' in signature, got:\n{wgsl}"
        )

    def test_readonly_cell_param_is_plain_value(self):
        """Helper that only reads its Cell arg → plain Cell in signature."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def score(self, c: MyCell) -> ol.f32:
                return c.health * c.energy

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass

        wgsl = _compile_sim_helper(S(), "score")
        # Must have "c: Cell" but NOT "ptr<function, Cell>"
        assert re.search(r"\bc: Cell\b", wgsl), (
            f"Expected 'c: Cell' (plain) in signature, got:\n{wgsl}"
        )
        assert "ptr<function, Cell>" not in wgsl, (
            f"Unexpected ptr in read-only param signature:\n{wgsl}"
        )

    def test_mutating_and_readonly_params_mixed(self):
        """Helper with one mutating and one read-only domain param — only mutating gets ptr."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def transfer(self, src: MyCell, dst: MyCell):
                # src is read-only, dst is mutated
                dst.energy += src.energy * 0.5

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass

        wgsl = _compile_sim_helper(S(), "transfer")
        assert re.search(r"\bsrc: Cell\b", wgsl), f"Expected src as plain Cell:\n{wgsl}"
        assert "dst: ptr<function, Cell>" in wgsl, f"Expected dst as ptr:\n{wgsl}"


    def test_rng_state_always_last_param(self):
        """Every helper always ends with _rng_state: ptr<function, u32>."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def noop(self, c: MyCell) -> ol.f32:
                return c.health

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass

        wgsl = _compile_sim_helper(S(), "noop")
        assert "_rng_state: ptr<function, u32>" in wgsl, (
            f"Expected rng_state param:\n{wgsl}"
        )


# =============================================================================
# Call-site pointer passing
# =============================================================================

class TestCallSitePointerPassing:
    """At the call site self.helper(arg), mutating domain args must be &arg."""

    def test_local_var_passed_as_ampersand(self):
        """tmp assigned from copy(), then passed to mutating helper → &tmp."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def buff(self, c: MyCell, amount: ol.f32):
                c.energy += amount

            @ol.cell_rule
            def rule(self, cell: MyCell):
                tmp = cell.copy()
                self.buff(tmp, 5.0)

        wgsl = _get_wgsl(S)
        assert "sim_buff(&tmp," in wgsl or "sim_buff(&tmp ," in wgsl, (
            f"Expected '&tmp' at call site:\n{wgsl}"
        )

    def test_pointer_arg_forwarded_bare(self):
        """Inside a helper that receives c as ptr, forwarding c to another
        mutating helper must pass c bare (already a pointer), not &c."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def inner(self, c: MyCell):
                c.health -= 0.1

            def outer(self, c: MyCell):
                self.inner(c)   # c is already ptr here — forward bare

            @ol.cell_rule
            def rule(self, cell: MyCell):
                tmp = cell.copy()
                self.outer(tmp)

        _get_wgsl(S)
        wgsl_outer = _compile_sim_helper(S(), "outer")
        # c is a ptr-param inside outer; forwarding to inner must be bare "c"
        assert "sim_inner(c," in wgsl_outer, (
            f"Expected bare 'c' forwarded (not &c) in outer body:\n{wgsl_outer}"
        )

    def test_readonly_domain_arg_has_no_ampersand(self):
        """Arg passed to a read-only domain parameter must NOT get &."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def score(self, c: MyCell) -> ol.f32:
                return c.health

            @ol.cell_rule
            def rule(self, cell: MyCell):
                x = self.score(cell)
                cell.energy = x

        wgsl = _get_wgsl(S)
        # score(cell, ...) — no &cell
        assert re.search(r"sim_score\(cell[^&]", wgsl) or "sim_score(cell," in wgsl, (
            f"Expected cell without & in read-only call:\n{wgsl}"
        )
        assert "sim_score(&cell," not in wgsl, (
            f"Unexpected & on read-only arg:\n{wgsl}"
        )

    def test_main_rule_arg_passed_without_ampersand_to_mutating(self):
        """The rule's main arg is a plain var; passing it to a mutating helper → &cell."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def grow(self, c: MyCell, rate: ol.f32):
                c.energy += rate

            @ol.cell_rule
            def rule(self, cell: MyCell):
                self.grow(cell, 2.0)

        wgsl = _get_wgsl(S)
        assert "sim_grow(&cell," in wgsl, (
            f"Expected '&cell' when passing rule main arg to mutating helper:\n{wgsl}"
        )


# =============================================================================
# Pointer dereference inside helper body
# =============================================================================

class TestPtrParamDereferenceInBody:
    """Inside helper bodies, ptr-params must be dereferenced as (*param).field."""

    def test_field_read_on_ptr_param(self):
        """Reading c.energy inside a mutating helper → (*c).energy in WGSL."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def read_and_mutate(self, c: MyCell) -> ol.f32:
                result = c.energy       # read
                c.energy = result * 2.0  # mutate
                return result

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass

        wgsl = _compile_sim_helper(S(), "read_and_mutate")
        assert "(*c).energy" in wgsl, (
            f"Expected (*c).energy dereference in helper body:\n{wgsl}"
        )

    def test_field_write_on_ptr_param(self):
        """Writing c.health inside a mutating helper → (*c).health = ... in WGSL."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def reset_health(self, c: MyCell):
                c.health = 1.0

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass

        wgsl = _compile_sim_helper(S(), "reset_health")
        assert "(*c).health = " in wgsl, (
            f"Expected (*c).health write in helper body:\n{wgsl}"
        )

    def test_plain_param_no_dereference(self):
        """Read-only param is accessed directly — no (*param) wrapping."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            def get_health(self, c: MyCell) -> ol.f32:
                return c.health

            @ol.cell_rule
            def rule(self, cell: MyCell):
                pass

        wgsl = _compile_sim_helper(S(), "get_health")
        assert "(*c)" not in wgsl, (
            f"Unexpected dereference on read-only param:\n{wgsl}"
        )
        assert "c.health" in wgsl, (
            f"Expected direct c.health access:\n{wgsl}"
        )

    def test_rule_main_arg_not_dereferenced(self):
        """In a rule kernel the main agent is a plain var — no (*cell) anywhere."""

        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.energy = cell.health * 2.0

        wgsl = _get_wgsl(S)
        assert "(*cell)" not in wgsl, (
            f"Rule main arg must not be dereferenced:\n{wgsl}"
        )


def test_vec3_assign_without_copy():
    wgpu = pytest.importorskip("wgpu")

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            a: ol.ivec3
            b: ol.ivec3 = ol.ivec3()
            a = b

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    with pytest.raises(Exception) as excinfo:
        engine.load_model(BadSim())


def test_vec3_assign_with_copy():
    wgpu = pytest.importorskip("wgpu")

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            a: ol.ivec3
            b: ol.ivec3 = ol.ivec3()
            a = b.copy()

    engine = ol.Engine(backend="gpu")
    engine.cells.add(MyCell())

    try:
        engine.load_model(GoodSim())
    except Exception as e:
        pytest.fail(f"GPU execution failed for GoodSim: {e}")