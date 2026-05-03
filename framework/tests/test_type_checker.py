"""
test_type_checker_intrinsics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 3 (TypeChecker) — tests for strict argument checking of intrinsic functions
(_check_intrinsic_args / _check_clamp_args) and type inference edge cases not covered by test_validator.py.

Structure
---------
  group A  — unary float functions (sin, cos, sqrt, …)
  group B  — binary float functions (pow, atan2, step)
  group C  — ternary float functions (lerp, smoothstep)
  group D  — vector functions (length, normalize, dot, cross, reflect, lerp_vec)
  group E  — clamp (overloads: f32×3 or vec3×3)
  group F  — mixed scalar functions (abs, fabs, sign, min, max)
  group G  — arg count mismatches
  group H  — TypeChecker edge cases (bool→i32, ivec3, range, IfExp)
"""

import pytest
import oncolytica as ol
from oncolytica import CompilationError


# ── Common stubs ────────────────────────────────────────────────────────────

class MyCell(ol.Cell):
    pos:    ol.vec3
    health: ol.f32
    flags:  ol.i32
    age:    ol.u32
    active: ol.bool


class MyTissue(ol.Tissue):
    oxygen: ol.f32 = 1.0


class MyChem(ol.Chemistry):
    drug: ol.f32 = 0.0


class MyMetrics(ol.Metrics):
    count: ol.u32 = 0


class MyParams(ol.Params):
    pass


def _load(sim_cls):
    """Shortcut: create engine, call load_model, return ctx (no GPU needed)."""
    engine = ol.Engine(backend="cpu")
    engine.load_model(sim_cls())


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A — unary float-only functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnaryFloatFunctions:
    """sin, cos, tan, asin, acos, atan, sqrt, exp, log, log2, fract, floor, ceil, round, trunc"""

    # ── sin ────────────────────────────────────────────────────────────────────

    def test_sin_float_literal_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sin(5.0)
        _load(S)

    def test_sin_f32_var_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sin(cell.health)
        _load(S)

    def test_sin_explicit_cast_passes(self):
        """sin(f32(5)) — explicit cast should satisfy the constraint."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                n: ol.i32 = 5
                cell.health = ol.math.sin(ol.f32(n))
        _load(S)

    def test_sin_int_literal_fails(self):
        """sin(5) — int literal must be rejected; WGSL requires f32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sin(5)
        with pytest.raises(CompilationError, match="sin"):
            _load(S)

    def test_sin_i32_var_fails(self):
        """sin(i32_var) — typed i32 variable must be rejected."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                n: ol.i32 = 3
                cell.health = ol.math.sin(n)
        with pytest.raises(CompilationError, match="sin"):
            _load(S)

    def test_sin_u32_var_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sin(cell.age)  # age is u32
        with pytest.raises(CompilationError, match="sin"):
            _load(S)

    # ── cos, sqrt, exp, log — spot checks ──────────────────────────────────────

    def test_cos_int_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.cos(0)
        with pytest.raises(CompilationError, match="cos"):
            _load(S)

    def test_sqrt_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sqrt(cell.health)
        _load(S)

    def test_sqrt_int_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sqrt(4)
        with pytest.raises(CompilationError, match="sqrt"):
            _load(S)

    def test_exp_i32_field_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.exp(cell.flags)  # flags is i32
        with pytest.raises(CompilationError, match="exp"):
            _load(S)

    def test_log_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.log(cell.health)
        _load(S)

    def test_floor_int_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.floor(cell.flags)
        with pytest.raises(CompilationError, match="floor"):
            _load(S)

    def test_fract_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.fract(cell.health)
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B — binary float functions: pow(f32, f32), atan2(f32, f32), step(f32, f32)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryFloatFunctions:

    def test_pow_float_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.pow(2.0, 3.0)
        _load(S)

    def test_pow_int_int_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.pow(2, 3)
        with pytest.raises(CompilationError, match="pow"):
            _load(S)

    def test_pow_float_int_fails(self):
        """Second argument is int — still invalid."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.pow(2.0, 3)
        with pytest.raises(CompilationError, match="pow"):
            _load(S)

    def test_pow_int_float_fails(self):
        """First argument is int — invalid."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.pow(2, 3.0)
        with pytest.raises(CompilationError, match="pow"):
            _load(S)

    def test_atan2_float_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.atan2(cell.health, 1.0)
        _load(S)

    def test_atan2_int_first_arg_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.atan2(0, cell.health)
        with pytest.raises(CompilationError, match="atan2"):
            _load(S)

    def test_step_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.step(0.5, cell.health)
        _load(S)

    def test_step_int_edge_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.step(0, cell.health)
        with pytest.raises(CompilationError, match="step"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C — ternary float: lerp(f32,f32,f32), smoothstep(f32,f32,f32)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTernaryFloatFunctions:

    def test_lerp_all_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.lerp(0.0, 100.0, cell.health)
        _load(S)

    def test_lerp_int_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.lerp(0, 100, cell.health)
        with pytest.raises(CompilationError, match="lerp"):
            _load(S)

    def test_lerp_mixed_int_float_fails(self):
        """Third argument (t) is i32 — should fail."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                n: ol.i32 = 1
                cell.health = ol.math.lerp(0.0, 1.0, n)
        with pytest.raises(CompilationError, match="lerp"):
            _load(S)

    def test_smoothstep_all_float_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.smoothstep(0.0, 1.0, cell.health)
        _load(S)

    def test_smoothstep_int_edges_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.smoothstep(0, 1, cell.health)
        with pytest.raises(CompilationError, match="smoothstep"):
            _load(S)

    def test_smoothstep_last_int_fails(self):
        """Edge values are float, x is i32 — all three must be f32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.smoothstep(0.0, 1.0, cell.flags)
        with pytest.raises(CompilationError, match="smoothstep"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D — vector functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorFunctions:

    def test_length_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.length(cell.pos)
        _load(S)

    def test_length_scalar_fails(self):
        """length() requires vec3, not f32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.length(cell.health)
        with pytest.raises(CompilationError, match="length"):
            _load(S)

    def test_length_sq_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.length_sq(cell.pos)
        _load(S)

    def test_normalize_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = ol.math.normalize(cell.pos)
        _load(S)

    def test_normalize_float_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = ol.math.normalize(cell.health)
        with pytest.raises(CompilationError, match="normalize"):
            _load(S)

    def test_dot_two_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.dot(cell.pos, cell.pos)
        _load(S)

    def test_dot_vec3_scalar_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.dot(cell.pos, cell.health)
        with pytest.raises(CompilationError, match="dot"):
            _load(S)

    def test_cross_two_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = ol.math.cross(cell.pos, ol.vec3(0.0, 1.0, 0.0))
        _load(S)

    def test_reflect_two_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                n = ol.math.normalize(cell.pos)
                cell.pos = ol.math.reflect(cell.pos, n)
        _load(S)

    def test_lerp_vec_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                target = ol.vec3(0.0, 0.0, 0.0)
                cell.pos = ol.math.lerp_vec(cell.pos, target, 0.1)
        _load(S)

    def test_lerp_vec_int_t_fails(self):
        """Third arg (t) must be f32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = ol.math.lerp_vec(cell.pos, cell.pos, 1)
        with pytest.raises(CompilationError, match="lerp_vec"):
            _load(S)

    def test_distance_two_vec3_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.distance(cell.pos, ol.vec3(0.0, 0.0, 0.0))
        _load(S)

    def test_distance_scalar_arg_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.distance(cell.health, 0.0)
        with pytest.raises(CompilationError, match="distance"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP E — clamp (overloads: f32×3 / vec3×3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClamp:

    def test_clamp_f32_overload_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.clamp(cell.health, 0.0, 100.0)
        _load(S)

    def test_clamp_vec3_overload_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                lo = ol.vec3(0.0, 0.0, 0.0)
                hi = ol.vec3(100.0, 100.0, 100.0)
                cell.pos = ol.math.clamp(cell.pos, lo, hi)
        _load(S)

    def test_clamp_int_first_arg_fails(self):
        """int first arg — neither f32 nor vec3."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.clamp(5, 0.0, 1.0)
        with pytest.raises(CompilationError, match="clamp"):
            _load(S)

    def test_clamp_vec3_mixed_with_float_fails(self):
        """vec3 first arg but f32 bounds — type mismatch."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = ol.math.clamp(cell.pos, 0.0, 100.0)
        with pytest.raises(CompilationError, match="clamp"):
            _load(S)

    def test_clamp_float_mixed_with_vec3_fails(self):
        """f32 first arg but vec3 bounds."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                lo = ol.vec3(0.0, 0.0, 0.0)
                hi = ol.vec3(1.0, 1.0, 1.0)
                cell.health = ol.math.clamp(cell.health, lo, hi)
        with pytest.raises(CompilationError, match="clamp"):
            _load(S)

    def test_clamp_i32_field_fails(self):
        """i32 field as first arg to clamp."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.clamp(cell.flags, 0.0, 1.0)
        with pytest.raises(CompilationError, match="clamp"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP F — mixed scalar functions: abs (scalar), fabs (f32 only), sign, min, max
# ═══════════════════════════════════════════════════════════════════════════════

class TestScalarFunctions:

    def test_abs_i32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.flags = ol.math.abs(cell.flags)
        _load(S)

    def test_abs_f32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.abs(cell.health)
        _load(S)

    def test_fabs_f32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.fabs(cell.health)
        _load(S)

    def test_fabs_i32_fails(self):
        """fabs is float-only, unlike abs."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.fabs(cell.flags)
        with pytest.raises(CompilationError, match="fabs"):
            _load(S)

    def test_sign_f32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sign(cell.health)
        _load(S)

    def test_sign_i32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.flags = ol.math.sign(cell.flags)
        _load(S)

    def test_min_f32_f32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.min(cell.health, 100.0)
        _load(S)

    def test_min_i32_i32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.flags = ol.math.min(cell.flags, 255)
        _load(S)

    def test_min_f32_i32_fails(self):
        """Mixed scalar types: f32 + i32 is forbidden for min."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.min(cell.health, cell.flags)
        with pytest.raises(CompilationError, match="min"):
            _load(S)

    def test_max_i32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.flags = ol.math.max(cell.flags, 0)
        _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP G — incorrect number of arguments
# ═══════════════════════════════════════════════════════════════════════════════

class TestArgCountMismatches:

    def test_sin_zero_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sin()
        with pytest.raises(CompilationError, match="sin"):
            _load(S)

    def test_sin_two_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.sin(1.0, 2.0)
        with pytest.raises(CompilationError, match="sin"):
            _load(S)

    def test_dot_one_arg_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.dot(cell.pos)
        with pytest.raises(CompilationError, match="dot"):
            _load(S)

    def test_dot_three_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.dot(cell.pos, cell.pos, cell.pos)
        with pytest.raises(CompilationError, match="dot"):
            _load(S)

    def test_pow_one_arg_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.pow(2.0)
        with pytest.raises(CompilationError, match="pow"):
            _load(S)

    def test_lerp_two_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.lerp(0.0, 1.0)
        with pytest.raises(CompilationError, match="lerp"):
            _load(S)

    def test_smoothstep_two_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.smoothstep(0.0, 1.0)
        with pytest.raises(CompilationError, match="smoothstep"):
            _load(S)

    def test_clamp_two_args_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ol.math.clamp(cell.health, 0.0)
        with pytest.raises(CompilationError, match="clamp"):
            _load(S)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP H — TypeChecker edge cases (not directly related to intrinsics)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypecheckerEdgeCases:

    def test_bool_field_access_returns_i32(self):
        """ol.bool-declared field at TypeChecker level returns i32 (not bool).
        Therefore assignment to i32 variable should pass."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                x: ol.i32 = cell.active  # active declared as ol.bool → physically i32
        _load(S)

    def test_bool_field_cannot_assign_to_u32(self):
        """bool field yields i32, which is incompatible with u32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                x: ol.u32 = cell.active  # i32 → u32 is forbidden
        with pytest.raises(CompilationError, match="Type mismatch"):
            _load(S)

    def test_range_with_float_arg_fails(self):
        """range() only accepts integer arguments."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for i in range(cell.health):  # f32 — forbidden
                    cell.flags = cell.flags + 1
        with pytest.raises(CompilationError, match="range"):
            _load(S)

    def test_range_with_i32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for i in range(10):
                    cell.flags = cell.flags + i
        _load(S)

    def test_range_loop_var_is_i32(self):
        """range() loop variable gets type i32 and can be used in arithmetic with i32 fields."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for i in range(5):
                    cell.flags = cell.flags + i  # i — i32, flags — i32 ✓
        _load(S)

    def test_range_loop_var_cannot_assign_to_f32(self):
        """range() loop variable is i32, cannot assign directly to f32 field."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                for i in range(5):
                    cell.health = i  # i32 to f32 — forbidden
        with pytest.raises(CompilationError, match="Type mismatch"):
            _load(S)

    def test_ifexp_type_from_body(self):
        """Ternary operator: result type is the type of the body branch."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                # Both branches are f32 → result is f32 → can assign to f32 field.
                cell.health = 1.0 if cell.health > 0.0 else 0.0
        _load(S)

    def test_bool_arithmetic_promotes_to_i32_not_f32(self):
        """True + True → i32; cannot be directly used as f32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                result = True + True      # → i32
                cell.health = result      # i32 → f32: forbidden
        with pytest.raises(CompilationError, match="Type mismatch"):
            _load(S)

    def test_bool_and_i32_compat_in_binop(self):
        """bool + i32 are compatible in arithmetic and result in i32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.flags = cell.active + 1   # bool(i32) + int(i32) → i32 ✓
        _load(S)

    def test_unary_neg_on_f32_is_f32(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = -cell.health
        _load(S)

    def test_unary_neg_on_i32_cannot_assign_to_f32(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = -cell.flags   # i32 → f32 forbidden
        with pytest.raises(CompilationError, match="Type mismatch"):
            _load(S)

    def test_vec3_mult_scalar_result_is_vec3(self):
        """vec3 * f32 → vec3."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = cell.pos * 2.0
        _load(S)

    def test_vec3_div_scalar_result_is_vec3(self):
        """vec3 / f32 → vec3."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.pos = cell.pos / 2.0
        _load(S)

    def test_mixing_vec3_types_fails(self):
        class IvecCell(ol.Cell):
            pos: ol.vec3
            coord: ol.ivec3

        class S(ol.Simulation[MyTissue, MyChem, IvecCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: IvecCell):
                cell.pos = cell.pos + cell.coord  # vec3 + ivec3 — запрещено

        with pytest.raises(CompilationError):
            ol.Engine(backend="cpu").load_model(S())

    def test_vec3_component_access_gives_f32(self):
        """vec3.x → f32."""
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = cell.pos.x  # f32

        _load(S)

    def test_undeclared_variable_raises(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = ghost_var  # noqa: F821

        with pytest.raises(CompilationError, match="no known type"):
            _load(S)

    def test_augassign_f32_plus_f32_passes(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health += 1.0

        _load(S)

    def test_augassign_f32_plus_int_fails(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health += 1  # int → type mismatch

        with pytest.raises(CompilationError, match="Type mismatch"):
            _load(S)

    def test_bool_not_always_returns_bool(self):
        class S(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.age = not cell.health  # u32 ← bool: forbidden

        with pytest.raises(CompilationError, match="Type mismatch"):
            _load(S)
