from __future__ import annotations

import ast
import re
from typing import Optional

from oncolytica.core.utils._errors import CompilationError
from ._context import ValidationContext

class BaseValidator(ast.NodeVisitor):
    """
    Common base for all pipeline validators.

    Each concrete validator is an ``ast.NodeVisitor`` whose ``visit_*``
    methods raise ``CompilationError`` on violation.  After the AST walk,
    ``post_validate()`` is called for checks that require the full tree
    to have been traversed first (e.g., graph cycle detection).
    """

    def __init__(self) -> None:
        self.ctx: Optional[ValidationContext] = None

    def validate(self, ctx: ValidationContext) -> None:
        self.ctx = ctx
        if ctx.tree:
            self.visit(ctx.tree)
        self.post_validate()

    def post_validate(self) -> None:
        """Override for checks that need the complete traversal result."""
