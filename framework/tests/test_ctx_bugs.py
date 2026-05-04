"""
Tests targeting the 7 bugs found in context building and validation pipeline.

Bug map:
  BUG-1 & BUG-2 — DomainValidator._analyze_method_mutations() writes domain-method
                   AST nodes into ctx.method_nodes, which should contain only sim_*
                   methods. After DomainValidator runs, method_nodes has 27+ entries
                   instead of the 3 (rule) methods the sim class actually defines.

  BUG-3  — method_return_hints['cell_copy'] == typing.Any, not the concrete user class.
            Dangerous if the special-case branch in TypeChecker is ever reordered.

  BUG-4  — method_return_hints for methods with explicit '-> None' annotation stores
            <class 'NoneType'> (the *class*) rather than None the singleton.
            Some callers test `expected is None` and miss type(None).

  BUG-5  — class_fields / class_field_types for a domain class may omit fields that
            are present in the class body but absent from __annotations__ at the
            level _collect_fields inspects (e.g. hidden by gpu-incompatible filter
            or by a missing annotation in the concrete class).

  BUG-6  — (confirmed non-issue) GBMMetrics having no methods is expected.

  BUG-7  — TypeChecker never processes domain-method bodies, so ctx.type_map
            contains no entries for AST nodes inside domain methods. This means
            CallGraphValidator cannot resolve obj.method() calls made *inside*
            a domain method, so call_graph edges for domain→domain calls are
            missing and ordered_methods topological order is incomplete.
"""

import ast
import textwrap
import pytest
import oncolytica as ol
from oncolytica.core.validation._validator import ValidatorEngine
from oncolytica.core.validation._context import ContextBuilder, mangle_sim


# ─────────────────────────────────────────────────────────────────────────────
# Shared minimal fixtures
# ─────────────────────────────────────────────────────────────────────────────

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
    total: ol.i32 = 0


class MyParams(ol.Params):
    rate: ol.f32 = 1.0


def _run(sim_cls):
    """Run the full validation pipeline and return (ctx, sim_instance)."""
    sim = sim_cls()
    ctx = ValidatorEngine().run(sim)
    return ctx, sim


# ─────────────────────────────────────────────────────────────────────────────
# BUG-1 & BUG-2  method_nodes must contain ONLY sim_* keys
# ─────────────────────────────────────────────────────────────────────────────

class TestMethodNodesPurity:
    """
    BUG-1 / BUG-2: DomainValidator._analyze_method_mutations() injects domain
    method AST nodes (cell_*, tissue_*, …) into ctx.method_nodes.  After the
    pipeline finishes, every key in method_nodes MUST start with 'sim_'.
    """

    def test_method_nodes_keys_all_start_with_sim(self):
        """All keys in ctx.method_nodes must be sim_* after the full pipeline."""

        class MutatingCell(MyCell):
            def age_up(self):
                self.age += 1

        class Sim(ol.Simulation[MyTissue, MyChem, MutatingCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MutatingCell):
                cell.age_up()

        ctx, _ = _run(Sim)
        non_sim = [k for k in ctx.method_nodes if not k.startswith("sim_")]
        assert non_sim == [], (
            f"ctx.method_nodes contains non-sim_ keys after pipeline: {non_sim}\n"
            f"BUG-1/BUG-2: DomainValidator._analyze_method_mutations() is "
            f"injecting domain AST nodes into method_nodes."
        )

    def test_method_nodes_count_equals_sim_method_count(self):
        """Number of entries in method_nodes must equal the number of methods
        defined directly on the Simulation subclass."""

        class Sim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = 1.0

            def helper(self, c: MyCell) -> ol.f32:
                return c.energy

        # Sim defines exactly 2 methods: rule + helper
        ctx, _ = _run(Sim)
        assert len(ctx.method_nodes) == 2, (
            f"Expected 2 entries in method_nodes, got {len(ctx.method_nodes)}.\n"
            f"Keys: {list(ctx.method_nodes.keys())}\n"
            f"BUG-1/BUG-2: extra domain nodes are leaking into method_nodes."
        )

    def test_domain_method_ast_nodes_absent_from_method_nodes_after_pipeline(self):
        """Concretely: cell_age_up must NOT appear in method_nodes."""

        class AgingCell(MyCell):
            def age_up(self):
                self.age += 1

        class Sim(ol.Simulation[MyTissue, MyChem, AgingCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: AgingCell):
                cell.age_up()

        ctx, _ = _run(Sim)
        assert "cell_age_up" not in ctx.method_nodes, (
            "BUG-1: 'cell_age_up' was injected into ctx.method_nodes by "
            "DomainValidator._analyze_method_mutations(). It belongs only in "
            "ctx.domain_method_nodes."
        )

    def test_domain_nodes_without_parent_do_not_reach_error_formatter(self):
        """
        When the error formatter in ValidatorEngine.run() walks e.node.parent,
        any AST node injected by DomainValidator lacks the .parent attribute set
        by ContextBuilder._build_ast().  Triggering a CompilationError on such
        a node must NOT raise AttributeError.

        This test confirms the formatter survives even if injected nodes exist.
        We cannot directly trigger a compile error on an injected node without
        the bug being present, so instead we verify that every node in
        method_nodes carries a .parent attribute (or is None for the top-level
        FunctionDef whose parent is the ClassDef).
        """

        class MutCell(MyCell):
            def boost(self):
                self.energy += 1.0

        class Sim(ol.Simulation[MyTissue, MyChem, MutCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MutCell):
                cell.boost()

        ctx, _ = _run(Sim)
        for key, node in ctx.method_nodes.items():
            # Every injected domain node is parsed fresh with ast.parse() and
            # never gets the parent-annotation walk that ContextBuilder does.
            # So `hasattr(node, 'parent')` is False for such nodes.
            assert hasattr(node, "parent"), (
                f"AST node for '{key}' in method_nodes lacks .parent attribute. "
                f"It was likely injected by DomainValidator without the parent-"
                f"annotation walk done by ContextBuilder._build_ast(). "
                f"BUG-1: domain nodes must not live in method_nodes."
            )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3  copy() return hint is typing.Any, not the concrete domain class
# ─────────────────────────────────────────────────────────────────────────────

class TestCopyReturnHint:
    """
    BUG-3: method_return_hints for cell_copy / tissue_copy / chemistry_copy
    is typing.Any (inherited from the base class annotation '-> Any').
    It should be the concrete user class so the type_map records a real type.
    """

    def test_copy_return_hint_is_concrete_class_not_any(self):
        """method_return_hints['cell_copy'] must be the concrete Cell subclass."""
        import typing

        class ConcreteCell(MyCell):
            pass

        class Sim(ol.Simulation[MyTissue, MyChem, ConcreteCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ConcreteCell):
                cell.health = 1.0

        ctx, _ = _run(Sim)
        hint = ctx.method_return_hints.get("cell_copy")
        assert hint is not typing.Any, (
            f"BUG-3: method_return_hints['cell_copy'] == typing.Any. "
            f"It should be {ConcreteCell} so type_map gets a concrete type "
            f"when the copy() special-case branch is skipped or reordered."
        )
        # Positive assertion: must be the concrete subclass
        assert hint is ConcreteCell, (
            f"Expected method_return_hints['cell_copy'] == {ConcreteCell}, "
            f"got {hint}."
        )

    def test_type_map_records_concrete_type_for_copy_call(self):
        """
        After TypeChecker runs, every copy() call in a sim method body must
        have a concrete (non-Any) type in ctx.type_map.
        """
        import typing

        class CellWithCopy(MyCell):
            pass

        class Sim(ol.Simulation[MyTissue, MyChem, CellWithCopy, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: CellWithCopy):
                clone = cell.copy()
                clone.health = cell.health

        ctx, _ = _run(Sim)

        # Find all Call nodes in the rule body where the callee is .copy()
        copy_call_types = []
        for mangled, node in ctx.method_nodes.items():
            for child in ast.walk(node):
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "copy"
                        and not child.args):
                    t = ctx.type_map.get(id(child))
                    copy_call_types.append(t)

        assert copy_call_types, "No copy() calls found — test fixture is broken."
        for t in copy_call_types:
            assert t is not typing.Any and t is not None, (
                f"BUG-3: copy() call has type {t!r} in type_map instead of "
                f"the concrete class {CellWithCopy}."
            )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-4  method_return_hints: explicit '-> None' stores NoneType, not None
# ─────────────────────────────────────────────────────────────────────────────

class TestNoneReturnHint:
    """
    BUG-4: _resolve_hints() returns type(None) (the NoneType class) for methods
    annotated with '-> None'.  Code that checks `expected is None` silently
    misclassifies void methods as having an unknown return type.
    """

    def test_explicit_none_return_stored_as_none_singleton(self):
        """method_return_hints for '-> None' methods must be None, not NoneType."""

        class Sim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = 1.0

            def reset(self, c: MyCell) -> None:
                c.health = 1.0

        ctx, _ = _run(Sim)
        hint = ctx.method_return_hints.get("sim_reset")
        assert hint is None, (
            f"BUG-4: method_return_hints['sim_reset'] == {hint!r} "
            f"(expected None the singleton, got NoneType or something else). "
            f"Methods annotated '-> None' must store None, not type(None)."
        )

    def test_domain_method_none_return_stored_as_none_singleton(self):
        """Same contract for domain-class methods that annotate '-> None'."""

        class ExplicitCell(MyCell):
            def reset(self) -> None:
                self.health = 1.0

        class Sim(ol.Simulation[MyTissue, MyChem, ExplicitCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ExplicitCell):
                cell.reset()

        ctx, _ = _run(Sim)
        hint = ctx.method_return_hints.get("cell_reset")
        assert hint is None, (
            f"BUG-4: method_return_hints['cell_reset'] == {hint!r} "
            f"(expected None the singleton). Domain method '-> None' must "
            f"store None, not type(None)."
        )

    def test_type_checker_void_return_does_not_false_positive(self):
        """
        TypeChecker._on_return uses `is_void = expected is None or expected is type(None)`.
        If BUG-4 is present but TypeChecker guards correctly, this test still
        validates the *contract* — if the guard is ever simplified to just
        `expected is None`, the pipeline must not raise a false CompilationError
        on a bare `return` in a void method.
        """
        # This must compile without error even if BUG-4 is present,
        # because TypeChecker has the `type(None)` guard. The test documents
        # the fragility: removing that guard would break this.
        class Sim(ol.Simulation[MyTissue, MyChem, MyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: MyCell):
                cell.health = self.clamp_val(cell.health)

            def clamp_val(self, v: ol.f32) -> ol.f32:
                return v

            def void_reset(self, c: MyCell) -> None:
                c.health = 1.0
                return

        # Must not raise
        ctx, _ = _run(Sim)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-5  class_fields / class_field_types completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestClassFieldsCompleteness:
    """
    BUG-5: Fields declared on a domain class might be absent from class_fields
    or class_field_types if they are filtered as non-GPU-compatible, missing
    from __annotations__, or shadowed by a base-class annotation.
    """

    def test_all_declared_fields_appear_in_class_fields(self):
        """Every non-underscore annotated field on a domain class must appear
        in ctx.class_fields[cls]."""

        class RichCell(MyCell):
            score:   ol.f32 = 0.0
            level:   ol.i32 = 0
            active:  ol.bool = True
            tag:     ol.i32 = 0

        class Sim(ol.Simulation[MyTissue, MyChem, RichCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: RichCell):
                cell.health = 1.0

        ctx, _ = _run(Sim)
        declared = {"score", "level", "active", "tag"}
        recorded = ctx.class_fields.get(RichCell, set())
        missing = declared - recorded
        assert not missing, (
            f"BUG-5: Fields {missing} declared on RichCell are absent from "
            f"ctx.class_fields[RichCell].\nRecorded: {recorded}"
        )

    def test_all_declared_fields_appear_in_class_field_types(self):
        """Every GPU-compatible field must also have its type in class_field_types."""

        class TypedCell(MyCell):
            rate:  ol.f32 = 0.0
            count: ol.i32 = 0

        class Sim(ol.Simulation[MyTissue, MyChem, TypedCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: TypedCell):
                cell.health = 1.0

        ctx, _ = _run(Sim)
        field_types = ctx.class_field_types.get(TypedCell, {})
        for fname, expected_type in [("rate", ol.f32), ("count", ol.i32)]:
            assert fname in field_types, (
                f"BUG-5: field '{fname}' missing from class_field_types[TypedCell]."
            )
            assert field_types[fname] is expected_type, (
                f"BUG-5: class_field_types[TypedCell]['{fname}'] == "
                f"{field_types[fname]}, expected {expected_type}."
            )

    def test_metrics_class_all_fields_present(self):
        """Regression for the GBM model: all Metrics fields must be captured."""

        class FullMetrics(ol.Metrics):
            infected_count:  ol.i32 = 0
            recruited_count: ol.i32 = 0
            inactive_count:  ol.i32 = 0
            total_cells:     ol.i32 = 0
            total_alive:     ol.i32 = 0   # the field missing from the GBM dump
            total_pdgf:      ol.f32 = 0.0

        class Sim(ol.Simulation[MyTissue, MyChem, MyCell, FullMetrics, MyParams]):
            @ol.metric_rule
            def collect(self, cell: MyCell, metrics: FullMetrics):
                metrics.total_cells += 1

        ctx, _ = _run(Sim)
        recorded = ctx.class_fields.get(FullMetrics, set())
        expected = {"infected_count", "recruited_count", "inactive_count",
                    "total_cells", "total_alive", "total_pdgf"}
        missing = expected - recorded
        assert not missing, (
            f"BUG-5: Fields {missing} missing from class_fields[FullMetrics].\n"
            f"Recorded: {recorded}\n"
            f"This matches the GBM ctx dump where 'total_alive' was absent."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-7  TypeChecker does not process domain-method bodies
#         → type_map empty for domain nodes
#         → call_graph missing domain→domain edges
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainMethodTypeMapAndCallGraph:
    """
    BUG-7: TypeChecker.validate() only iterates ctx.method_nodes (sim_* methods).
    Domain-method bodies are never type-checked, so:
      (a) ctx.type_map has no entries for AST nodes inside domain methods.
      (b) CallGraphValidator._process_call() uses type_map to resolve the
          receiver type of obj.method() calls; without entries it skips the
          edge → call_graph[cell_X] stays empty even when cell_X calls cell_Y.
      (c) ctx.ordered_methods lacks a guaranteed callee-before-caller ordering
          for domain→domain chains.
    """

    def test_type_map_populated_for_domain_method_nodes(self):
        """
        After the pipeline, ctx.type_map must contain at least one entry whose
        id corresponds to an AST node that lives *inside* a domain method body.

        We pick a simple read: `return self.energy` in get_energy().
        The Name node for 'self.energy' (or the Attribute node) should be typed.
        """

        class EnergyCell(MyCell):
            def get_energy(self) -> ol.f32:
                return self.energy

        class Sim(ol.Simulation[MyTissue, MyChem, EnergyCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: EnergyCell):
                cell.health = cell.get_energy()

        ctx, _ = _run(Sim)

        domain_node = ctx.domain_method_nodes.get((EnergyCell, "get_energy"))
        assert domain_node is not None, (
            "domain_method_nodes[(EnergyCell, 'get_energy')] is None — "
            "the domain method AST was not captured."
        )

        # Collect all expression-node ids inside the domain method body.
        domain_ids = {id(n) for n in ast.walk(domain_node)}
        typed_domain_ids = domain_ids & set(ctx.type_map.keys())

        assert typed_domain_ids, (
            "BUG-7: ctx.type_map has NO entries for AST nodes inside the "
            "domain method 'get_energy'. TypeChecker never processes domain "
            "method bodies, so the type_map is empty for them."
        )

    def test_call_graph_has_domain_to_domain_edge(self):
        """
        cell_double_boost calls cell_boost.  After the pipeline,
        ctx.call_graph['cell_double_boost'] must contain 'cell_boost'.
        """

        class ChainCell(MyCell):
            def boost(self):
                self.energy += 1.0

            def double_boost(self):
                self.boost()
                self.boost()

        class Sim(ol.Simulation[MyTissue, MyChem, ChainCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ChainCell):
                cell.double_boost()

        ctx, _ = _run(Sim)

        edge = ctx.call_graph.get("cell_double_boost", set())
        assert "cell_boost" in edge, (
            f"BUG-7: call_graph['cell_double_boost'] == {edge!r}. "
            f"Expected 'cell_boost' to be in the set because double_boost() "
            f"calls self.boost() twice. TypeChecker never types domain method "
            f"bodies, so type_map[id(self_name_node)] is None and the edge "
            f"is never added by CallGraphValidator."
        )

    def test_ordered_methods_callee_before_caller_for_domain_chain(self):
        """
        cell_boost must appear before cell_double_boost in ordered_methods
        (so the WGSL emitter can forward-reference-free emit the callee first).
        """

        class ChainCell(MyCell):
            def boost(self):
                self.energy += 1.0

            def double_boost(self):
                self.boost()
                self.boost()

        class Sim(ol.Simulation[MyTissue, MyChem, ChainCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: ChainCell):
                cell.double_boost()

        ctx, _ = _run(Sim)

        ordered = ctx.ordered_methods
        assert "cell_boost" in ordered, (
            "BUG-7: 'cell_boost' absent from ordered_methods entirely."
        )
        assert "cell_double_boost" in ordered, (
            "BUG-7: 'cell_double_boost' absent from ordered_methods entirely."
        )
        idx_boost  = ordered.index("cell_boost")
        idx_double = ordered.index("cell_double_boost")
        assert idx_boost < idx_double, (
            f"BUG-7: ordered_methods has cell_boost at {idx_boost} and "
            f"cell_double_boost at {idx_double}. Callee must precede caller. "
            f"Full order: {ordered}"
        )

    def test_three_level_domain_chain_ordered_correctly(self):
        """
        c → b → a: ordered_methods must have a < b < c.
        Tests that the topo-sort extends to multi-level domain chains.
        """

        class DeepCell(MyCell):
            def a(self) -> ol.f32:
                return self.energy

            def b(self) -> ol.f32:
                return self.a()

            def c(self) -> ol.f32:
                return self.b()

        class Sim(ol.Simulation[MyTissue, MyChem, DeepCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: DeepCell):
                cell.health = cell.c()

        ctx, _ = _run(Sim)
        ordered = ctx.ordered_methods
        for name in ("cell_a", "cell_b", "cell_c"):
            assert name in ordered, f"BUG-7: '{name}' missing from ordered_methods."

        assert ordered.index("cell_a") < ordered.index("cell_b"), (
            "BUG-7: cell_a must precede cell_b in ordered_methods."
        )
        assert ordered.index("cell_b") < ordered.index("cell_c"), (
            "BUG-7: cell_b must precede cell_c in ordered_methods."
        )

    def test_symbol_table_populated_for_domain_method(self):
        """
        ctx.symbol_table must contain entries for domain-method mangled names,
        not only for sim_* methods.

        Currently symbol_table only has 3 entries (sim_*) because TypeChecker
        only calls _check_function() for nodes in ctx.method_nodes (sim only).
        """

        class AnnotatedCell(MyCell):
            def compute(self, factor: ol.f32) -> ol.f32:
                result: ol.f32 = self.energy * factor
                return result

        class Sim(ol.Simulation[MyTissue, MyChem, AnnotatedCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: AnnotatedCell):
                cell.health = cell.compute(2.0)

        ctx, _ = _run(Sim)

        # cell_compute must appear in symbol_table with 'result' typed as f32
        assert "cell_compute" in ctx.symbol_table, (
            "BUG-7: 'cell_compute' absent from ctx.symbol_table. "
            "TypeChecker does not process domain method bodies."
        )
        local_env = ctx.symbol_table.get("cell_compute", {})
        assert "result" in local_env, (
            f"BUG-7: local variable 'result' not typed in symbol_table['cell_compute']. "
            f"Got: {local_env}"
        )
        assert local_env["result"] is ol.f32, (
            f"BUG-7: symbol_table['cell_compute']['result'] == {local_env['result']!r}, "
            f"expected ol.f32."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-cutting: call_graph edges for domain→domain calls must NOT be empty
# when the sim-method call_graph correctly has the sim→domain edges.
# ─────────────────────────────────────────────────────────────────────────────

class TestCallGraphCompleteness:
    """Integration tests ensuring call_graph reflects reality for all call kinds."""

    def test_sim_to_domain_edge_present(self):
        """sim_rule → cell_age_up must appear (this was already working)."""

        class AgeCell(MyCell):
            def age_up(self):
                self.age += 1

        class Sim(ol.Simulation[MyTissue, MyChem, AgeCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: AgeCell):
                cell.age_up()

        ctx, _ = _run(Sim)
        assert "cell_age_up" in ctx.call_graph.get("sim_rule", set()), (
            "sim_rule → cell_age_up edge missing from call_graph."
        )

    def test_domain_to_domain_edge_not_empty(self):
        """cell_outer → cell_inner edge must appear (BUG-7 breaks this)."""

        class EdgeCell(MyCell):
            def inner(self) -> ol.f32:
                return self.energy

            def outer(self) -> ol.f32:
                return self.inner()

        class Sim(ol.Simulation[MyTissue, MyChem, EdgeCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: EdgeCell):
                cell.health = cell.outer()

        ctx, _ = _run(Sim)
        outer_edges = ctx.call_graph.get("cell_outer", set())
        assert outer_edges != set(), (
            f"BUG-7: call_graph['cell_outer'] == {{}} (empty). "
            f"The edge to 'cell_inner' was not built because TypeChecker "
            f"never typed the body of the domain method, leaving "
            f"type_map[id(self_name_node)] == None."
        )
        assert "cell_inner" in outer_edges, (
            f"BUG-7: 'cell_inner' not in call_graph['cell_outer']. "
            f"Got: {outer_edges}"
        )

    def test_cross_domain_type_edge(self):
        """
        cell_behavior calls tissue.is_overcrowded().
        call_graph['cell_behavior'] (or the sim rule that calls it) must
        eventually reach 'tissue_is_overcrowded'.
        """

        class CrowdTissue(MyTissue):
            cap: ol.i32 = 5

            def is_full(self, n: ol.i32) -> ol.bool:
                return n >= self.cap

        class CrowdCell(MyCell):
            def check(self, t: CrowdTissue) -> ol.bool:
                return t.is_full(3)

        class Sim(ol.Simulation[CrowdTissue, MyChem, CrowdCell, MyMetrics, MyParams]):
            @ol.cell_rule
            def rule(self, cell: CrowdCell):
                tissue: CrowdTissue = self.tissue_at(cell.pos)
                _ = cell.check(tissue)

        ctx, _ = _run(Sim)
        # The sim rule calls cell.check, which calls tissue.is_full.
        # Both edges must be in the graph.
        sim_edges = ctx.call_graph.get("sim_rule", set())
        assert "cell_check" in sim_edges, (
            f"sim_rule → cell_check edge missing. Got sim_rule edges: {sim_edges}"
        )
        cell_check_edges = ctx.call_graph.get("cell_check", set())
        assert "tissue_is_full" in cell_check_edges, (
            f"BUG-7: cell_check → tissue_is_full edge missing. "
            f"Got cell_check edges: {cell_check_edges}"
        )
