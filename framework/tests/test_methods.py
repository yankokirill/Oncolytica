"""
Tests for domain-method compilation:
  Step 3 — 'self' becomes explicit '_self' parameter (ptr or value)
  Step 4 — method calls translated to function calls:
              c.find_closest()  →  cell_find_closest(c, _rng_state)
              c.find_closest()  →  cell_find_closest(&c, _rng_state)  (if mutating)

Coverage:
  _self as value        — non-mutating domain method
  _self as ptr          — mutating domain method (writes to self field)
  self.field access     — self.x → (*_self).x or _self.x
  obj.method() in rule  — c.helper() → cell_helper(c, &_rng)
  obj.method() in sim helper — c.helper() → cell_helper(c, _rng_state)
  mutating obj.method() — receiver passed as &c
  chained: self.method() inside domain method — prefix_method(_self/_self&, ...)
  sim helper calling domain method — correct mangled name + receiver
"""
import re
import pytest
import oncolytica as ol
from oncolytica.core.validation._validator import ValidatorEngine
from oncolytica.gpu.compiler._compiler import _compile_helper_fn, WGSLCompiler


# ── Shared domain fixtures ────────────────────────────────────────────────────

class MyTissue(ol.Tissue):
    oxygen: ol.f32 = 1.0

class MyChem(ol.Chemistry):
    drug: ol.f32 = 0.0

class MyCell(ol.Cell):
    pos:    ol.vec3
    energy: ol.f32 = 1.0
    health: ol.f32 = 1.0
    age:    ol.i32 = 0

class MyMetrics(ol.Metrics):
    total: ol.u32 = 0

class MyParams(ol.Params):
    rate: ol.f32 = 1.0


def _compile_sim_helper(sim_instance, method_name: str) -> str:
    val_ctx = ValidatorEngine().run(sim_instance)
    method  = getattr(sim_instance, method_name)
    wgsl, _ = _compile_helper_fn(
        method,
        uniforms_map={},
        val_ctx=val_ctx,
        wgsl_fn_name=f"sim_{method_name}",
    )
    return wgsl


def _compile_domain_method(cls, method_name: str, sim_instance) -> str:
    """Compile a single domain-class method and return the WGSL function string."""
    from oncolytica.gpu.compiler._type_system import domain_base_of
    val_ctx = ValidatorEngine().run(sim_instance)
    method  = getattr(cls, method_name)
    base    = domain_base_of(cls)
    prefix  = base.__name__.lower()
    mangled = f"{prefix}_{method_name}"
    wgsl, _ = _compile_helper_fn(
        method,
        uniforms_map={},
        val_ctx=val_ctx,
        wgsl_fn_name=mangled,
        main_param="_self",
        main_class=cls,
    )
    return wgsl


def _get_wgsl(sim_cls) -> str:
    engine = ol.Engine(backend="gpu")
    engine.load_model(sim_cls())
    return engine._backend_impl._shader_code


# =============================================================================
# Step 3 — _self parameter: value vs pointer
# =============================================================================

class TestSelfParameter:
    """Domain methods must receive _self as ptr when they mutate it, else by value."""

    def test_non_mutating_method_gets_self_by_value(self):
        """Method that only reads self → _self: Cell (not ptr)."""

        class ReadOnlyCell(MyCell):
            def get_energy(self) -> ol.f32:
                return self.energy

        class S(ol.Simulation[MyTissue, MyChem, ReadOnlyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ReadOnlyCell):
                cell.health = self.read(cell)

            def read(self, c: ReadOnlyCell) -> ol.f32:
                return c.get_energy()

        _get_wgsl(S)
        wgsl = _compile_domain_method(ReadOnlyCell, "get_energy", S())
        assert "_self: Cell" in wgsl, (
            f"Non-mutating method should receive _self by value.\nGot:\n{wgsl}"
        )
        assert "ptr<function, Cell>" not in wgsl, (
            f"Non-mutating method must NOT use ptr.\nGot:\n{wgsl}"
        )

    def test_mutating_method_gets_self_as_ptr(self):
        """Method that writes to self field → _self: ptr<function, Cell>."""

        class MutCell(MyCell):
            def age_up(self):
                self.age += 1

        class S(ol.Simulation[MyTissue, MyChem, MutCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MutCell):
                cell.age_up()

        _get_wgsl(S)
        wgsl = _compile_domain_method(MutCell, "age_up", S())
        assert "_self: ptr<function, Cell>" in wgsl, (
            f"Mutating method should receive _self as ptr.\nGot:\n{wgsl}"
        )

    def test_self_field_read_in_value_method(self):
        """self.energy in non-mutating method → _self.energy (no dereference)."""

        class ReadCell(MyCell):
            def get_energy(self) -> ol.f32:
                return self.energy

        class S(ol.Simulation[MyTissue, MyChem, ReadCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ReadCell):
                cell.health = cell.get_energy()

        wgsl = _compile_domain_method(ReadCell, "get_energy", S())
        assert "_self.energy" in wgsl, (
            f"Value receiver: self.energy should → _self.energy.\nGot:\n{wgsl}"
        )

    def test_self_field_write_in_ptr_method(self):
        """self.age += 1 in mutating method → (*_self).age += 1."""

        class MutCell(MyCell):
            def age_up(self):
                self.age += 1

        class S(ol.Simulation[MyTissue, MyChem, MutCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MutCell):
                cell.age_up()

        wgsl = _compile_domain_method(MutCell, "age_up", S())
        assert "(*_self).age" in wgsl, (
            f"Ptr receiver: self.age should → (*_self).age.\nGot:\n{wgsl}"
        )

    def test_self_appears_first_in_signature(self):
        """_self must be the first parameter in the WGSL function signature."""

        class AnyCell(MyCell):
            def scale(self, factor: ol.f32):
                self.energy = self.energy * factor

        class S(ol.Simulation[MyTissue, MyChem, AnyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: AnyCell):
                cell.scale(2.0)

        wgsl = _compile_domain_method(AnyCell, "scale", S())
        # signature: fn cell_scale(_self: ..., factor: f32, _rng_state: ...)
        m = re.search(r"fn cell_scale\(([^)]+)\)", wgsl)
        assert m, f"Could not find fn cell_scale signature.\nGot:\n{wgsl}"
        first_param = m.group(1).split(",")[0].strip()
        assert first_param.startswith("_self"), (
            f"_self must be first param, got: '{first_param}'"
        )


# =============================================================================
# Step 4 — obj.method() → prefix_method(receiver, ..., _rng_state)
# =============================================================================

class TestMethodToFunctionTranslation:
    """obj.method(args) in rule/helper bodies must become prefix_method(obj, args, rng)."""

    def test_non_mutating_call_passes_receiver_by_value(self):
        """c.get_energy() in rule → cell_get_energy(c, &_rng)."""

        class ReadCell(MyCell):
            def get_energy(self) -> ol.f32:
                return self.energy

        class S(ol.Simulation[MyTissue, MyChem, ReadCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ReadCell):
                cell.health = cell.get_energy()

        wgsl = _get_wgsl(S)
        # receiver passed by value (no &)
        assert re.search(r"cell_get_energy\(cell,", wgsl), (
            f"Expected cell_get_energy(cell, ...) (no &).\nGot:\n{wgsl[:3000]}"
        )

    def test_mutating_call_passes_receiver_by_ptr(self):
        """cell.age_up() in rule → cell_age_up(&cell, &_rng)."""

        class MutCell(MyCell):
            def age_up(self):
                self.age += 1

        class S(ol.Simulation[MyTissue, MyChem, MutCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MutCell):
                cell.age_up()

        wgsl = _get_wgsl(S)
        assert re.search(r"cell_age_up\(&cell,", wgsl), (
            f"Expected cell_age_up(&cell, ...) (with &).\nGot:\n{wgsl[:3000]}"
        )

    def test_method_call_with_extra_args(self):
        """cell.scale(2.0) in rule → cell_scale(&cell, 2.0, &_rng)."""

        class ScaleCell(MyCell):
            def scale(self, factor: ol.f32):
                self.energy = self.energy * factor

        class S(ol.Simulation[MyTissue, MyChem, ScaleCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ScaleCell):
                cell.scale(2.0)

        wgsl = _get_wgsl(S)
        assert re.search(r"cell_scale\(&cell,\s*2\.0", wgsl), (
            f"Expected cell_scale(&cell, 2.0, ...).\nGot:\n{wgsl[:3000]}"
        )

    def test_method_call_on_local_copy(self):
        """tmp = cell.copy(); tmp.age_up() → cell_age_up(&tmp, &_rng)."""

        class MutCell(MyCell):
            def age_up(self):
                self.age += 1

        class S(ol.Simulation[MyTissue, MyChem, MutCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MutCell):
                tmp = cell.copy()
                tmp.age_up()

        wgsl = _get_wgsl(S)
        assert re.search(r"cell_age_up\(&tmp,", wgsl), (
            f"Expected cell_age_up(&tmp, ...).\nGot:\n{wgsl[:3000]}"
        )

    def test_rng_state_passed_in_helper_context(self):
        """In a sim helper, domain method call uses _rng_state (not &_rng)."""

        class ReadCell(MyCell):
            def get_energy(self) -> ol.f32:
                return self.energy

        class S(ol.Simulation[MyTissue, MyChem, ReadCell, MyMetrics, MyParams]):
            def helper(self, c: ReadCell) -> ol.f32:
                return c.get_energy()

            @ol.cell_rule
            def rule(self, cell: ReadCell):
                cell.health = self.helper(cell)

        wgsl = _compile_sim_helper(S(), "helper")
        assert re.search(r"fn sim_helper\(c: Cell, _rng_state: ptr<function, u32>\) -> f32", wgsl), (
            f"Expected cell_get_energy(c, _rng_state) in helper context.\nGot:\n{wgsl}"
        )

    def test_method_call_in_rule_uses_rng_ref(self):
        """In a rule kernel, domain method call uses &_rng."""

        class ReadCell(MyCell):
            def get_energy(self) -> ol.f32:
                return self.energy

        class S(ol.Simulation[MyTissue, MyChem, ReadCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ReadCell):
                cell.health = cell.get_energy()

        wgsl = _get_wgsl(S)
        assert re.search(r"cell_get_energy\(cell,\s*&_rng\)", wgsl), (
            f"Expected cell_get_energy(cell, &_rng) in rule context.\nGot:\n{wgsl[:3000]}"
        )

    def test_domain_method_function_emitted_in_shader(self):
        """The compiled domain method must appear as a top-level fn in the shader."""

        class AgeCell(MyCell):
            def tick(self):
                self.age += 1

        class S(ol.Simulation[MyTissue, MyChem, AgeCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: AgeCell):
                cell.tick()

        wgsl = _get_wgsl(S)
        assert "fn cell_tick(" in wgsl, (
            f"Expected 'fn cell_tick(' in shader.\nGot:\n{wgsl[:3000]}"
        )

    def test_self_method_call_inside_domain_method(self):
        """self.helper() inside a domain method → cell_helper(_self/_self&, _rng_state)."""

        class ChainCell(MyCell):
            def boost(self):
                self.energy += 1.0

            def double_boost(self):
                self.boost()
                self.boost()

        class S(ol.Simulation[MyTissue, MyChem, ChainCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ChainCell):
                cell.double_boost()

        wgsl = _get_wgsl(S)
        # Inside cell_double_boost, self.boost() should become cell_boost(&_self, ...)
        assert re.search(r"cell_boost\(_self,", wgsl), (
            f"Expected cell_boost(&_self, ...) inside domain method.\nGot:\n{wgsl[:3000]}"
        )

    def test_copy_and_die_not_translated_as_function_call(self):
        """copy() and die() must NOT be translated as cell_copy() / cell_die()."""

        class DieCell(MyCell):
            pass

        class S(ol.Simulation[MyTissue, MyChem, DieCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: DieCell):
                cell.die()

        wgsl = _get_wgsl(S)
        assert "cell_die(" not in wgsl, (
            f"die() must be inlined, not translated as cell_die().\nGot:\n{wgsl[:3000]}"
        )
        assert "cell_copy(" not in wgsl, (
            f"copy() must not appear as cell_copy().\nGot:\n{wgsl[:3000]}"
        )


def _validate(sim_cls):
    """Run the full validation pipeline and return the context (raises on error)."""
    return ValidatorEngine().run(sim_cls())

# =============================================================================
# Rule 2.3 — loop iterator is read-only inside a domain method
# =============================================================================

class TestLoopIteratorMutationInDomainMethod:

    def test_mutate_loop_var_in_domain_method_raises(self):
        class NeighborCell(ol.Cell):
            energy: ol.f32 = 1.0
            pos: ol.vec3

            def drain_neighbors(self):
                for n in self.neighbors:
                    n.energy = 0.0

        class Sim(ol.Simulation[MyTissue, MyChem, NeighborCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: NeighborCell):
                cell.drain_neighbors()

        with pytest.raises(CompilationError, match="readonly|iterator|modify"):
            _validate(Sim)
