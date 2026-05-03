"""
_errors.py — Compilation error type for the Oncolytica DSL validator.
"""

from __future__ import annotations
import ast


class CompilationError(Exception):
    """
    Raised when user simulation code violates Oncolytica DSL rules.
    """

    def __init__(self, message: str, node: ast.AST | None = None) -> None:
        self.raw_message = message
        self.node = node
        self.lineno = getattr(node, "lineno", None) if node else None

        super().__init__(f"Line {self.lineno or '?'}: {message}")
