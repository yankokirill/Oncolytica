"""
_errors.py — Compilation error type for the Oncolytica DSL validator.
"""

from __future__ import annotations


class CompilationError(Exception):
    """
    Raised when user simulation code violates Oncolytica DSL rules.

    Attributes:
        lineno:     Source line number where the violation was detected (1-based),
                    or None if the location is unknown.
        col_offset: Column offset within that line, or None.
    """

    def __init__(
        self,
        message: str,
        lineno: int | None = None,
        col_offset: int | None = None,
    ) -> None:
        self.lineno = lineno
        self.col_offset = col_offset

        super().__init__(f"{message}")
