"""
test_validator.py
~~~~~~~~~~~~~~~~~
Comprehensive test suite for the Oncolytica 5-phase compiler frontend.

Test organisation mirrors the pipeline phases:

    Phase 0  ContextBuilder          – test_context_*
    Phase 1  NamingValidator         – test_phase1_naming_*
             SyntaxValidator         – test_phase1_syntax_*
    Phase 2  ScopeBuilder            – test_phase2_scope_*
    Phase 3  TypeChecker             – test_phase3_types_*
    Phase 4  CallGraphValidator      – test_phase4_callgraph_*
    Phase 5  DomainValidator         – test_phase5_domain_*

Positive tests (should compile without error)  – test_valid_*
"""
import oncolytica as ol
from oncolytica import CompilationError
import pytest


# ---------------------------------------------------------------------------
# Shared memory-layout classes used across tests
# ---------------------------------------------------------------------------

class MyCell(ol.CellData):
    pos:     ol.vec3 = ol.vec3(50.0, 50.0, 50.0)
    health:  ol.f32
    mutated: ol.bool


class MyTissue(ol.TissueData):
    oxygen: ol.f32 = 1.0


class MyChem(ol.ChemistryData):
    drug_conc: ol.f32 = 0.0


class MyMetrics(ol.MetricsData):
    alive_cells: ol.u32 = 0


# ===========================================================================
# POSITIVE TESTS – valid simulations that must compile without error
# ===========================================================================

def test_valid_minimal_simulation():
    """A minimal but fully valid Simulation should pass all phases."""

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            if cell.health < 0.0:
                cell.health = 0.0

    engine = ol.Engine(backend="cpu")
    engine.load_model(GoodSim())   # must NOT raise


def test_valid_helper_method():
    """A simulation with a typed helper called from a rule must pass."""

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def clamp_health(self, value: ol.f32) -> ol.f32:
            return value

        @ol.cell_rule
        def rule(self, cell: MyCell):
            cell.health = self.clamp_health(cell.health)

    engine = ol.Engine(backend="cpu")
    engine.load_model(GoodSim())


MAX_HEALTH: ol.f32 = 100.0
def test_valid_constant_read():
    """A rule that reads (but does not write) an UPPER_CASE constant passes."""

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):

        @ol.cell_rule
        def rule(self, cell: MyCell):
            if cell.health > MAX_HEALTH:
                cell.health = MAX_HEALTH

    engine = ol.Engine(backend="cpu")
    engine.load_model(GoodSim())


def test_valid_explicit_cast():
    """Explicit type cast  f32(5)  in a binary expression must pass."""

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            cell.health = cell.health + ol.f32(1)

    engine = ol.Engine(backend="cpu")
    engine.load_model(GoodSim())


# ===========================================================================
# PHASE 0 – CONTEXT BUILDER
# ===========================================================================

def test_context_multiple_inheritance_forbidden():
    """Memory classes cannot inherit from multiple framework base classes."""

    class MutantData(ol.CellData, ol.TissueData):
        pos: ol.vec3 = ol.vec3(50.0, 50.0, 50.0)
        pass

    class BadSim(ol.Simulation[MyTissue, MyChem, MutantData, MyMetrics]):
        pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="inherits from multiple framework base classes"):
        engine.load_model(BadSim())


# ===========================================================================
# PHASE 1-A – NAMING VALIDATOR
# ===========================================================================

def test_phase1_naming_camel_case_method_forbidden():
    """CamelCase method names must be rejected by NamingValidator."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def badMethodName(self, cell: MyCell):  # CamelCase – forbidden
            pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="must be snake_case"):
        engine.load_model(BadSim())


def test_phase1_naming_camel_case_local_var_forbidden():
    """CamelCase local variables must be rejected by NamingValidator."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            myVar = 5.0

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="must be snake_case"):
        engine.load_model(BadSim())


def test_phase1_naming_constant_store_forbidden():
    """Assigning to an UPPER_CASE name inside a rule must be rejected."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        THRESHOLD: ol.f32 = 10.0

        @ol.cell_rule
        def rule(self, cell: MyCell):
            THRESHOLD = 5.0  # shadows a constant

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError,
                       match="Variable 'THRESHOLD' shadows a constant|Assignment to UPPER_CASE"):
        engine.load_model(BadSim())


# ===========================================================================
# PHASE 1-B – SYNTAX VALIDATOR
# ===========================================================================

def test_phase1_syntax_list_comprehension_forbidden():
    """List comprehensions cannot be translated to WGSL."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            a = [x for x in range(5)]

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="List comprehensions are forbidden"):
        engine.load_model(BadSim())


def test_phase1_syntax_cascading_assignment_forbidden():
    """Cascading assignment  a = b = c  is forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            a = b = 5.0  # Cascading – forbidden

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cascading assignment"):
        engine.load_model(BadSim())


def test_phase1_syntax_tuple_unpacking_forbidden():
    """Tuple unpacking  a, b = func()  is forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def get_pair(self) -> ol.vec3:
            return ol.vec3(1.0, 2.0, 3.0)

        @ol.cell_rule
        def rule(self, cell: MyCell):
            x, y, z = self.get_pair()  # Tuple unpacking – forbidden

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Tuple|unpacking"):
        engine.load_model(BadSim())


def test_phase1_syntax_chained_comparison_forbidden():
    """WGSL does not support ``0 < x < 10``; use two separate conditions."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            if 0.0 < cell.health < 100.0:
                cell.health = 100.0

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Chained comparisons"):
        engine.load_model(BadSim())


def test_phase1_syntax_while_loop_forbidden():
    """while loops are forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            i = ol.i32(0)
            while i < 10:
                i = i + ol.i32(1)

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="'while' loops are forbidden"):
        engine.load_model(BadSim())


def test_phase1_syntax_lambda_forbidden():
    """Lambda expressions are forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            fn = lambda x: x + 1.0

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Lambda"):
        engine.load_model(BadSim())


# ===========================================================================
# PHASE 2 – SCOPE BUILDER  (signature validation + symbol table)
# ===========================================================================

def test_phase2_scope_missing_type_annotation():
    """All rule parameters (except self) must have type annotations."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell):  # Missing ': MyCell'
            pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="lacks a type annotation"):
        engine.load_model(BadSim())


def test_phase2_scope_wrong_param_count_cell_rule():
    """@cell_rule strictly requires (self, cell); extra params are forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell, extra: ol.f32):
            pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match=r"needs 2 parameter\(s\) \(self, cell\); got 3"):
        engine.load_model(BadSim())


def test_phase2_scope_multiple_rule_decorators_forbidden():
    """A method cannot carry more than one @ol.*_rule decorator."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        @ol.cell_rule
        def rule(self, cell: MyCell):
            pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="multiple @ol\.\*_rule decorators"):
        engine.load_model(BadSim())


def test_phase2_scope_wrong_param_type_for_rule():
    """The cell parameter of @cell_rule must be a CellData subclass."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyTissue):  # Wrong base class
            pass

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="must be a subclass of CellData"):
        engine.load_model(BadSim())


# ===========================================================================
# PHASE 3 – TYPE CHECKER
# ===========================================================================

def test_phase3_types_strict_int_float_mix():
    """
    Mixing f32 and int without an explicit cast must raise a Type mismatch error.
    WGSL requires: f32(5) + cell.health, NOT 5 + cell.health.
    """

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            dummy = cell.health + 5  # f32 + int – implicit cast forbidden

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Type mismatch"):
        engine.load_model(BadSim())


def test_phase3_types_return_type_mismatch():
    """Helper method return type must match the declared return annotation."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def get_value(self) -> ol.i32:
            return 5.5  # float literal, but declared return is i32

        @ol.cell_rule
        def rule(self, cell: MyCell):
            a = self.get_value()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Return type mismatch"):
        engine.load_model(BadSim())


def test_phase3_types_augassign_type_mismatch():
    """``cell.health += 1`` mixes f32 and int – forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            cell.health += 1  # f32 += int

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Type mismatch"):
        engine.load_model(BadSim())


def test_phase3_types_vec3_add_scalar_forbidden():
    """vec3 + scalar is not valid WGSL (only vec3 * scalar and vec3 / scalar)."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            cell.pos = cell.pos + 1.0  # vec3 + scalar – forbidden

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="vec3"):
        engine.load_model(BadSim())


def test_phase3_types_explicit_cast_passes():
    """Explicit f32(n) cast must allow the operation to pass type checking."""

    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            cell.health = cell.health + ol.f32(1)

    engine = ol.Engine(backend="cpu")
    engine.load_model(GoodSim())  # must NOT raise


# ===========================================================================
# PHASE 4 – CALL GRAPH VALIDATOR
# ===========================================================================

def test_phase4_callgraph_direct_recursion():
    """Direct recursion is forbidden (WGSL has no call stack)."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def recursive_helper(self) -> ol.i32:
            return self.recursive_helper()

        @ol.cell_rule
        def rule(self, cell: MyCell):
            a = self.recursive_helper()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Recursive call detected"):
        engine.load_model(BadSim())


def test_phase4_callgraph_indirect_recursion():
    """Indirect recursion  A → B → A  is also forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def ping(self) -> ol.i32:
            return self.pong()

        def pong(self) -> ol.i32:
            return self.ping()

        @ol.cell_rule
        def rule(self, cell: MyCell):
            a = self.ping()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Recursive call detected"):
        engine.load_model(BadSim())


def test_phase4_callgraph_undefined_method():
    """Calling a self.method that does not exist must be rejected."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            self.ghost_method()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="is not defined in the Simulation class"):
        engine.load_model(BadSim())


# ===========================================================================
# PHASE 5 – DOMAIN VALIDATOR
# ===========================================================================

def test_phase5_domain_undefined_attribute():
    """Accessing an attribute not declared in the Data class is forbidden."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            cell.magic_power = 100.0  # 'magic_power' not in MyCell

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="'magic_power' is not a valid field"):
        engine.load_model(BadSim())


def test_phase5_domain_self_mutation_in_rule_forbidden():
    """Mutating ``self.<attr>`` is only permitted inside ``__init__``."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        my_counter: ol.i32 = 0

        @ol.cell_rule
        def rule(self, cell: MyCell):
            self.my_counter += 1  # cannot mutate simulation state from a rule

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Assignment to 'self' is forbidden outside __init__"):
        engine.load_model(BadSim())


def test_phase5_domain_neighbor_attribute_mutation_forbidden():
    """Only the current agent pointer may have its attributes mutated."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            for neighbor in cell.neighbors:
                neighbor.health = 0.0  # cannot mutate a neighbor

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cannot modify attribute"):
        engine.load_model(BadSim())


def test_phase5_domain_die_on_neighbor_forbidden():
    """``neighbor.die()`` is forbidden; only ``cell.die()`` is allowed."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            for neighbor in cell.neighbors:
                neighbor.die()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cannot call 'die\(\)'"):
        engine.load_model(BadSim())


def test_phase5_domain_invalid_vec3_component():
    """Accessing ``cell.pos.w`` must be rejected (only .x .y .z are valid)."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            bad = cell.pos.w  # 'w' is not a valid vec3 component

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="not a valid vec3 component"):
        engine.load_model(BadSim())


# ===========================================================================
# LEGACY COMPATIBILITY  (original test names kept for CI stability)
# ===========================================================================

# Phase 0
def test_validator_multiple_inheritance_forbidden():
    test_context_multiple_inheritance_forbidden()

# Phase 1 naming
def test_validator_naming_camel_case_forbidden():
    test_phase1_naming_camel_case_method_forbidden()

# Phase 1 syntax
def test_validator_syntax_comprehensions_forbidden():
    test_phase1_syntax_list_comprehension_forbidden()

def test_validator_syntax_cascading_and_unpacking():
    test_phase1_syntax_cascading_assignment_forbidden()

def test_validator_syntax_chained_comparisons():
    test_phase1_syntax_chained_comparison_forbidden()

# Phase 4
def test_validator_callgraph_direct_recursion():
    test_phase4_callgraph_direct_recursion()

def test_validator_callgraph_indirect_recursion():
    test_phase4_callgraph_indirect_recursion()

def test_validator_callgraph_undefined_method():
    test_phase4_callgraph_undefined_method()

# Phase 2
def test_validator_signature_missing_type_hints():
    test_phase2_scope_missing_type_annotation()

def test_validator_signature_wrong_arg_count():
    test_phase2_scope_wrong_param_count_cell_rule()

# Phase 5
def test_validator_semantics_undefined_attribute():
    test_phase5_domain_undefined_attribute()

def test_validator_semantics_self_mutation_forbidden():
    test_phase5_domain_self_mutation_in_rule_forbidden()

# Phase 3
def test_validator_semantics_return_type_mismatch():
    test_phase3_types_return_type_mismatch()

def test_validator_semantics_shadowing_constants():
    test_phase1_naming_constant_store_forbidden()

def test_dsl_rule2_strict_typing_mixed_math():
    test_phase3_types_strict_int_float_mix()

def test_dsl_rule4_multiple_assignments():
    """a = b = c  must raise a Cascading assignment error."""

    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            a = b = 10.0

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cascading assignment"):
        engine.load_model(BadSim())

def test_dsl_rule4_tuple_unpacking():
    test_phase1_syntax_tuple_unpacking_forbidden()


def test_rule6_helper_method_complex_expression_forbidden():
    """
    Test that passing agent pointers to helper methods inside complex expressions
    is forbidden (Rule 6).
    """
    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def get_energy(self, cell: MyCell) -> ol.f32:
            return cell.health * 2.0

        @ol.cell_rule
        def rule(self, cell: MyCell):
            # Forbidden: helper call with a pointer argument nested inside a condition
            if self.get_energy(cell) < 5.0:
                cell.health = 0.0

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="cannot be nested"):
        engine.load_model(BadSim())


def test_rule6_helper_method_assignment_allowed():
    """
    Test that assigning a helper method result to a variable is allowed (Rule 6).
    """
    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        def get_energy(self, cell: MyCell) -> ol.f32:
            return cell.health * 2.0

        @ol.cell_rule
        def rule(self, cell: MyCell):
            # Allowed: Extracting the helper method call to a separate local variable
            energy = self.get_energy(cell)
            if energy < 5.0:
                cell.health = 0.0

    engine = ol.Engine(backend="cpu")
    try:
        engine.load_model(GoodSim())
    except CompilationError as e:
        pytest.fail(f"Valid helper method usage raised CompilationError: {e}")


def test_rule7_neighbor_mutation_attribute_forbidden():
    """
    Test that modifying a neighbor's attribute is forbidden (Rule 7).
    """
    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            for nb in cell.neighbors:
                # Forbidden: Mutating another agent's state
                nb.health -= 1.0

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cannot modify attribute of 'nb'"):
        engine.load_model(BadSim())


def test_rule7_neighbor_mutation_method_forbidden():
    """
    Test that calling mutating methods (like die) on a neighbor is forbidden (Rule 7).
    """
    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            for nb in cell.neighbors:
                # Forbidden: Killing another agent directly
                nb.die()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cannot call 'die\\(\\)' on 'nb'"):
        engine.load_model(BadSim())


def test_rule7_cells_in_voxel_mutation_method_forbidden():
    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.tissue_rule
        def rule(self, t: MyTissue):
            for c in t.cells:
                c.die()

    engine = ol.Engine(backend="cpu")
    with pytest.raises(CompilationError, match="Cannot call"):
        engine.load_model(BadSim())


def test_rule7_wrong_die_method():
    class BadSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.tissue_rule
        def rule(self, tissue: MyTissue):
            tissue.die()

    engine = ol.Engine(backend="gpu")
    with pytest.raises(CompilationError, match="is not a valid field"):
        engine.load_model(BadSim())


def test_rule7_self_mutation_in_loop_allowed():
    """
    Test that modifying the main agent based on neighbor data is perfectly legal.
    """
    class GoodSim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics]):
        @ol.cell_rule
        def rule(self, cell: MyCell):
            for nb in cell.neighbors:
                # Reading from neighbor is allowed, modifying 'cell' is allowed
                if nb.health > 50.0:
                    cell.health += 1.0

    engine = ol.Engine(backend="cpu")
    try:
        engine.load_model(GoodSim())
    except CompilationError as e:
        pytest.fail(f"Valid self-mutation raised CompilationError: {e}")

