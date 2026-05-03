"""
Static-analysis pipeline (compiler frontend) for user Simulation classes.

Pipeline  (5 Phases, Dragon-Book style)
----------------------------------------
  Phase 0   ContextBuilder
              Parse source → AST; extract signatures, fields, constants.
              Zero validation here – pure data extraction.

  Phase 1   Syntactic & Lexical  (fail fast on surface errors)
              NamingValidator  – snake_case / UPPER_SNAKE_CASE enforcement
              SyntaxValidator  – WGSL-incompatible construct rejection
              SignatureValidator     – validates signatures & decorators,

  Phase 2   Type Inference & Type Checking  (all expressions typed)
              TypeChecker      – bottom-up type inference, writes TypeMap
                                 (ctx.type_map), enforces strict WGSL
                                 type compatibility and return-type contracts

  Phase 3   Control Flow & Call-Graph  (structural correctness)
              CallGraphValidator – cycle / recursion detection,
                                   topological sort for WGSL emission

  Phase 4   Domain-Specific Semantics  (Oncolytica business rules)
              DomainValidator  – mutability constraints, valid field access,
                                 action-call ownership rules

Blackboard
----------
All validators communicate exclusively through ``ValidationContext``.
Phases may *add* derived data to it but must never mutate data that an
earlier phase already produced.  On the first error the pipeline raises
``CompilationError`` (Fail-Fast).
"""

from __future__ import annotations

from inspect import Signature
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from oncolytica.core.utils._errors import CompilationError

from ._base import BaseValidator
from ._context import ValidationContext, ContextBuilder
from ._syntax import NamingValidator, SyntaxValidator, SignatureValidator
from ._type_checker import TypeChecker
from ._call_graph import CallGraphValidator
from ._domain import DomainValidator

import ast

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ValidatorEngine:
    """
    Runs all five phases in order against a ``ValidationContext``.

    Each phase validator is a ``BaseValidator`` sub-class; they communicate
    exclusively through the ``ValidationContext`` blackboard.
    On the first ``CompilationError`` the pipeline stops (Fail-Fast).
    """

    def __init__(self) -> None:
        self.pipeline: List[BaseValidator] = [
            # Phase 1 – Syntactic & Lexical
            NamingValidator(),
            SyntaxValidator(),
            SignatureValidator(),
            # Phase 2 – Type Inference & Checking
            TypeChecker(),
            # Phase 3 – Control Flow & Call Graph
            CallGraphValidator(),
            # Phase 4 – Domain-Specific Semantics
            DomainValidator(),
        ]

    def run(
            self,
            sim_instance: Any,
    ) -> ValidationContext:
        ctx = ContextBuilder.build(sim_instance)

        try:
            for validator in self.pipeline:
                validator.validate(ctx)
        except CompilationError as e:
            if e.node and ctx.source_code and e.lineno:
                func_node = e.node
                while func_node and not isinstance(func_node, ast.FunctionDef):
                    func_node = getattr(func_node, "parent", None)

                abs_file = "<unknown>"
                abs_lineno = e.lineno

                if func_node and isinstance(func_node, ast.FunctionDef):
                    method_name = func_node.name
                    loc = ctx.method_locations.get(method_name)
                    if loc:
                        abs_file, method_real_start_line = loc
                        offset_within_method = e.lineno - func_node.lineno
                        abs_lineno = method_real_start_line + offset_within_method

                lines = ctx.source_code.splitlines()
                if 0 <= e.lineno - 1 < len(lines):
                    line_text = lines[e.lineno - 1]
                    col = getattr(e.node, "col_offset", 0)
                    end_col = getattr(e.node, "end_col_offset", None)

                    prefix = "".join("\t" if c == "\t" else " " for c in line_text[:col])
                    pointer = "^" + "~" * (end_col - col - 1) if (end_col and end_col > col) else "^"

                    file_link = f'File "{abs_file}", line {abs_lineno + 1}'

                    formatted_msg = (
                        f"\n  {file_link}\n"
                        f"    {line_text.rstrip()}\n"
                        f"    {prefix}{pointer}\n"
                        f"CompilationError: {e.raw_message}"
                    )
                    e.args = (formatted_msg,)
            raise

        return ctx
