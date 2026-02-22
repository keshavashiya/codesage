"""Static code analysis checks for the review pipeline.

All checkers return List[ReviewFinding] for unified pipeline integration.
Python checkers use AST; TreeSitterReviewChecker uses tree-sitter for
Rust/Go/JS/TS; GenericFileChecker uses regex as a fallback.
"""

from codesage.review.checks.python_checks import PythonBadPracticeChecker
from codesage.review.checks.complexity import ComplexityChecker
from codesage.review.checks.structure import StructureChecker
from codesage.review.checks.naming import NamingChecker
from codesage.review.checks.generic_checks import GenericFileChecker
from codesage.review.checks.treesitter_checks import (
    TreeSitterReviewChecker,
    TREESITTER_AVAILABLE,
)

__all__ = [
    "PythonBadPracticeChecker",
    "ComplexityChecker",
    "StructureChecker",
    "NamingChecker",
    "GenericFileChecker",
    "TreeSitterReviewChecker",
    "TREESITTER_AVAILABLE",
]
