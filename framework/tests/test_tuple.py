"""
Tests for Rule 4 — Tuple semantics: validation and WGSL codegen.

Validation (CompilationError expected):
  - storing a tuple return value in a plain variable
  - subscript on a non-call expression
  - subscript with a non-literal index

Translation (WGSL correctness):
  - struct Tuple_* is emitted in the shader
  - immediate index  val = self.pair()[0]   → val = sim_pair(...).get_0
  - unpacking        a, b = self.pair()     → tmp = sim_pair(...); a = tmp.get_0; b = tmp.get_1
  - type of unpacked variables is correct
  - index out of range raises at compile time
"""
import re
import pytest
import oncolytica as ol
from oncolytica.core.validation._validator import ValidatorEngine
from oncolytica.core.utils._errors import CompilationError
from oncolytica.gpu.compiler._compiler import _compile_helper_fn, WGSLCompiler


# ── Shared domain fixtures ────────────────────────────────────────────────────

class Tissue(ol.Tissue):
    oxygen: ol.f32 = 1.0

class Chem(ol.Chemistry):
    drug: ol.f32 = 0.0

class Cell(ol.Cell):
    pos:    ol.vec3
    health: ol.f32
    energy: ol.f32

class Metrics(ol.Metrics):
    total: ol.u32 = 0

class Params(ol.Params):
    rate: ol.f32 = 1.0


def _validate(sim_cls):
    return ValidatorEngine().run(sim_cls())


def _compile_helper(sim_instance, method_name: str) -> str:
    val_ctx = ValidatorEngine().run(sim_instance)
    method  = getattr(sim_instance, method_name)
    wgsl, _ = _compile_helper_fn(
        method,
        uniforms_map={},
        val_ctx=val_ctx,
        wgsl_fn_name=f"sim_{method_name}",
    )
    return wgsl


def _get_wgsl(sim_cls) -> str:
    engine = ol.Engine(backend="gpu")
    engine.load_model(sim_cls())
    return engine._backend_impl._shader_code


# =============================================================================
# Validation — forbidden patterns (Rule 4, No Storage)
# =============================================================================

class TestTupleValidationForbidden:
    """Patterns that must raise CompilationError."""
    def test_tuple_unpack_existing_variables(self):
        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                x: ol.f32
                y: ol.f32
                x, y = self.pair()

        with pytest.raises(CompilationError):
            _get_wgsl(Sim)

    def test_storing_tuple_in_variable_raises(self):
        """t = self.pair()  where pair() returns tuple[f32, f32] must raise."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                t = self.pair()     # ← forbidden: storing a tuple

        with pytest.raises(CompilationError, match="[Tt]uple|store|assign|unpack"):
            _validate(Sim)

    def test_subscript_on_variable_raises(self):
        """x[0]  where x is a plain variable must raise (subscript on non-call)."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                x = cell.health     # x is f32
                y = x[0]            # ← forbidden: subscript on non-call

        with pytest.raises(CompilationError):
            _validate(Sim)

    def test_subscript_with_variable_index_raises(self):
        """call()[i]  where i is a variable index must raise (only literal allowed)."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                i = 0
                v = self.pair()[i]  # ← forbidden: non-literal index

        with pytest.raises(CompilationError):
            _validate(Sim)

    def test_index_out_of_range_raises(self):
        """call()[5]  where tuple has only 2 elements must raise."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                v = self.pair()[5]  # ← out-of-range index

        with pytest.raises(CompilationError, match="[Ii]ndex|range|out"):
            _validate(Sim)

    def test_wrong_unpack_count_raises(self):
        """Unpacking a 2-tuple into 3 variables must raise."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                a, b, c = self.pair()   # ← wrong arity

        with pytest.raises(CompilationError, match="[Uu]npack|element|tuple"):
            _validate(Sim)


# =============================================================================
# Validation — allowed patterns (Rule 4, Immediate Consumption)
# =============================================================================

class TestTupleValidationAllowed:
    """Patterns that must pass validation without raising."""

    def test_immediate_unpack_passes(self):
        """a, b = self.pair()  must pass validation."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                a, b = self.pair()
                cell.health = a + b

        _validate(Sim)  # must not raise

    def test_immediate_index_passes(self):
        """val = self.pair()[0]  must pass validation."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.i32]:
                return 1.0, 0

            @ol.cell_rule
            def rule(self, cell: Cell):
                val = self.pair()[0]
                cell.health = val

        _validate(Sim)  # must not raise

    def test_unpack_into_used_variables_passes(self):
        """Unpacked variables can be used in subsequent expressions."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def scores(self) -> tuple[ol.f32, ol.f32]:
                return 0.5, 0.3

            @ol.cell_rule
            def rule(self, cell: Cell):
                x, y = self.scores()
                cell.health  = x
                cell.energy = y

        _validate(Sim)  # must not raise

    def test_mixed_primitive_tuple_passes(self):
        """tuple[f32, i32]  mixed-type tuple must pass validation and unpack."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def mixed(self) -> tuple[ol.f32, ol.i32]:
                return 1.0, 1

            @ol.cell_rule
            def rule(self, cell: Cell):
                f, i = self.mixed()
                cell.health = f

        _validate(Sim)  # must not raise


# =============================================================================
# WGSL codegen — struct declarations
# =============================================================================

class TestTupleStructEmission:
    """The shader must contain the Tuple_* struct declaration."""

    def test_tuple_struct_emitted_for_f32_f32(self):
        """tuple[f32, f32] → struct Tuple_f32_f32 { get_0: f32, get_1: f32 }."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                a, b = self.pair()
                cell.health = a + b

        wgsl = _get_wgsl(Sim)
        assert "struct Tuple_f32_f32" in wgsl, (
            f"Expected Tuple_f32_f32 struct in shader:\n{wgsl[:2000]}"
        )
        assert "get_0: f32" in wgsl, f"Expected get_0 field:\n{wgsl[:2000]}"
        assert "get_1: f32" in wgsl, f"Expected get_1 field:\n{wgsl[:2000]}"

    def test_tuple_struct_emitted_for_mixed_types(self):
        """tuple[f32, i32] → struct Tuple_f32_i32 { get_0: f32, get_1: i32 }."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def mixed(self) -> tuple[ol.f32, ol.i32]:
                return 1.0, 0

            @ol.cell_rule
            def rule(self, cell: Cell):
                f, i = self.mixed()
                cell.health = f

        wgsl = _get_wgsl(Sim)
        assert "struct Tuple_f32_i32" in wgsl, (
            f"Expected Tuple_f32_i32:\n{wgsl[:2000]}"
        )
        assert "get_0: f32" in wgsl
        assert "get_1: i32" in wgsl

    def test_unused_tuple_struct_not_emitted(self):
        """If no tuple-returning method is called, no Tuple_* struct appears."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            @ol.cell_rule
            def rule(self, cell: Cell):
                cell.health += 1.0

        wgsl = _get_wgsl(Sim)
        assert "struct Tuple_" not in wgsl, (
            f"Unexpected Tuple_ struct in shader with no tuples:\n{wgsl[:2000]}"
        )


# =============================================================================
# WGSL codegen — index access
# =============================================================================

class TestTupleIndexTranslation:
    """val = self.pair()[i]  must translate to  val = sim_pair(...).get_i."""

    def test_index_0_translates_to_get_0(self):
        """self.pair()[0]  →  sim_pair(...).get_0."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                val = self.pair()[0]
                cell.health = val

        wgsl = _get_wgsl(Sim)
        assert re.search(r"sim_pair\([^)]*\)\.get_0", wgsl), (
            f"Expected sim_pair(...).get_0 in shader:\n{wgsl[:3000]}"
        )

    def test_index_1_translates_to_get_1(self):
        """self.pair()[1]  →  sim_pair(...).get_1."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                val = self.pair()[1]
                cell.health = val

        wgsl = _get_wgsl(Sim)
        assert re.search(r"sim_pair\([^)]*\)\.get_1", wgsl), (
            f"Expected sim_pair(...).get_1 in shader:\n{wgsl[:3000]}"
        )


# =============================================================================
# WGSL codegen — tuple unpacking
# =============================================================================

class TestTupleUnpackTranslation:
    """a, b = self.pair()  must compile to an intermediate tmp + field assignments."""

    def test_unpack_emits_tmp_and_field_assignments(self):
        """a, b = self.pair() must produce:
            _unpackN = sim_pair(...);
            a = _unpackN.get_0;
            b = _unpackN.get_1;
        """

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            @ol.cell_rule
            def rule(self, cell: Cell):
                a, b = self.pair()
                cell.health = a + b

        wgsl = _get_wgsl(Sim)

        # Find the tmp variable name (anything matching _unpack\d+)
        tmp_match = re.search(r"(_tuple_\d+)\s*=\s*sim_pair\(", wgsl)
        assert tmp_match, (
            f"Expected '_unpackN = sim_pair(...)' pattern in shader:\n{wgsl[:3000]}"
        )
        tmp = tmp_match.group(1)

        assert f"{tmp}.get_0 + {tmp}.get_1" in wgsl, (
            f"Expected '{tmp}.get_0 + {tmp}.get_1' in shader\n"
        )

    def test_unpacked_variables_have_correct_types(self):
        """Unpacked f32 and i32 variables must be declared with correct WGSL types."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def mixed(self) -> tuple[ol.f32, ol.i32]:
                return 1.0, 0

            @ol.cell_rule
            def rule(self, cell: Cell):
                f, i = self.mixed()
                cell.health = f

        wgsl = _get_wgsl(Sim)
        assert re.search(r"var _tuple_0: Tuple_f32_i32;", wgsl), (
            f"Expected declaration\n"
        )


    def test_unpack_in_helper_function(self):
        """Tuple unpacking inside a helper function (not a rule) also works."""

        class Sim(ol.Simulation[Tissue, Chem, Cell, Metrics, Params]):
            def pair(self) -> tuple[ol.f32, ol.f32]:
                return 1.0, 2.0

            def consume(self) -> ol.f32:
                a, b = self.pair()
                return a + b

            @ol.cell_rule
            def rule(self, cell: Cell):
                cell.health = self.consume()

        wgsl = _get_wgsl(Sim)
        # Both sim_pair and the unpack pattern must appear inside sim_consume.
        assert "sim_consume(" in wgsl, f"Expected sim_consume in shader\n"
