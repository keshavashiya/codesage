"""Language-agnostic static checks using text/regex analysis.

These checks run on any source file regardless of language and do NOT
require an AST parser.  They produce a reasonable baseline of findings
for Rust, Go, JavaScript, TypeScript, and any other text-based language.

Rules:
- GEN-LONG-FUNCTION : Function body exceeds MAX_FUNCTION_LINES lines
- GEN-TODO          : Unresolved TODO / FIXME / HACK / XXX comment
- GEN-LONG-LINE     : Line exceeds MAX_LINE_LENGTH characters
- GEN-HIGH-NESTING  : Indentation depth exceeds MAX_NESTING_DEPTH levels
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from codesage.review.models import ReviewFinding
from codesage.utils.language_detector import EXTENSION_TO_LANGUAGE

# ---------------------------------------------------------------------------
# Thresholds (mirrored from python structure.py for consistency)
# ---------------------------------------------------------------------------
MAX_FUNCTION_LINES = 60   # slightly more lenient for non-Python
MAX_LINE_LENGTH = 120
MAX_NESTING_DEPTH = 5     # slightly more lenient (Rust/Go routinely nest more)
INDENT_UNIT = 4           # spaces per level; tabs count as 1 unit

# ---------------------------------------------------------------------------
# Language → function-definition regex
# Each pattern must have a group(1) that captures the function name.
# ---------------------------------------------------------------------------
_FN_PATTERNS: Dict[str, re.Pattern] = {
    "rust": re.compile(
        r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    ),
    "go": re.compile(
        r"^\s*func\s+(?:\([^)]+\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    ),
    "javascript": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)"
        r"|^\s*(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*="
        r"\s*(?:async\s+)?(?:function|\()"
    ),
    "typescript": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)"
        r"|^\s*(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*="
        r"\s*(?:async\s+)?(?:function|\()"
    ),
}

# Also detect method-style defs (class methods in Rust/Go structs, JS classes)
_METHOD_PATTERNS: Dict[str, re.Pattern] = {
    "rust": re.compile(
        r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    ),
    "go": re.compile(
        r"^\s*func\s+\([^)]+\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    ),
    "javascript": re.compile(
        r"^\s+(?:async\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{"
    ),
    "typescript": re.compile(
        r"^\s+(?:async\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*(?::\s*\S+\s*)?\{"
    ),
}

# Non-Python extension → language (derived from the canonical EXTENSION_TO_LANGUAGE)
_EXT_TO_LANG: Dict[str, str] = {
    ext: lang
    for ext, lang in EXTENSION_TO_LANGUAGE.items()
    if lang != "python"
}

_TODO_RE = re.compile(
    r"(?://|#|/\*)\s*(TODO|FIXME|HACK|XXX)\b[:\s]*(.*)",
    re.IGNORECASE,
)


def _detect_language(file_path: Path) -> Optional[str]:
    return _EXT_TO_LANG.get(file_path.suffix.lower())


def _nesting_depth(line: str) -> int:
    """Estimate nesting depth from leading whitespace."""
    stripped = line.lstrip()
    if not stripped:
        return 0
    indent = len(line) - len(stripped)
    # Tabs treated as 1 indent unit each
    tabs = line[:len(line) - len(stripped)].count("\t")
    spaces = indent - tabs
    return tabs + spaces // INDENT_UNIT


def _extract_function_ranges(
    lines: List[str], language: str
) -> List[Tuple[int, str]]:
    """Return list of (start_line_1indexed, fn_name) for detected functions."""
    pattern = _FN_PATTERNS.get(language)
    mpattern = _METHOD_PATTERNS.get(language)
    ranges: List[Tuple[int, str]] = []
    for lineno, line in enumerate(lines, 1):
        name: Optional[str] = None
        if pattern:
            m = pattern.match(line)
            if m:
                # JS/TS pattern has two capture groups
                name = m.group(1) or (m.lastindex >= 2 and m.group(2)) or "?"
        if name is None and mpattern:
            m = mpattern.match(line)
            if m:
                name = m.group(1) or "?"
        if name and name not in ("if", "else", "for", "while", "match",
                                  "loop", "return", "let", "use", "mod"):
            ranges.append((lineno, name))
    return ranges


class GenericFileChecker:
    """Language-agnostic static analysis for non-Python source files."""

    def check(self, file_path: Path, content: str) -> List[ReviewFinding]:
        """Run all generic checks on the given file content.

        Args:
            file_path: Path used for language detection and finding location.
            content:   Full source text of the file.

        Returns:
            List of ReviewFinding (may be empty).
        """
        language = _detect_language(file_path)
        if language is None:
            return []  # Unsupported extension

        lines = content.splitlines()
        findings: List[ReviewFinding] = []

        findings.extend(self._check_todos(file_path, lines))
        findings.extend(self._check_long_lines(file_path, lines))
        findings.extend(self._check_high_nesting(file_path, lines))
        findings.extend(self._check_long_functions(file_path, lines, language))

        return findings

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_todos(
        self, file_path: Path, lines: List[str]
    ) -> List[ReviewFinding]:
        findings: List[ReviewFinding] = []
        for lineno, line in enumerate(lines, 1):
            m = _TODO_RE.search(line)
            if m:
                marker = m.group(1).upper()
                msg = m.group(2).strip()
                findings.append(ReviewFinding(
                    rule_id="GEN-TODO",
                    file=file_path,
                    line=lineno,
                    message=(
                        f"{marker}: {msg[:100]}" if msg else f"Unresolved {marker} marker"
                    ),
                    severity="warning" if marker == "FIXME" else "suggestion",
                    category="maintenance",
                    source="generic",
                ))
        return findings

    def _check_long_lines(
        self, file_path: Path, lines: List[str]
    ) -> List[ReviewFinding]:
        findings: List[ReviewFinding] = []
        for lineno, line in enumerate(lines, 1):
            # Skip comment-only lines (often long URLs/docs)
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            if len(line.rstrip("\n")) > MAX_LINE_LENGTH:
                findings.append(ReviewFinding(
                    rule_id="GEN-LONG-LINE",
                    file=file_path,
                    line=lineno,
                    message=(
                        f"Line exceeds {MAX_LINE_LENGTH} characters "
                        f"({len(line.rstrip())} chars)"
                    ),
                    severity="suggestion",
                    category="style",
                    source="generic",
                ))
        return findings

    def _check_high_nesting(
        self, file_path: Path, lines: List[str]
    ) -> List[ReviewFinding]:
        """Flag lines where indentation exceeds MAX_NESTING_DEPTH."""
        findings: List[ReviewFinding] = []
        flagged_blocks: set = set()  # (depth, approx_block_start) to avoid spam
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue
            depth = _nesting_depth(line)
            if depth > MAX_NESTING_DEPTH:
                block_key = (depth, lineno // 5)  # Group nearby lines
                if block_key not in flagged_blocks:
                    flagged_blocks.add(block_key)
                    findings.append(ReviewFinding(
                        rule_id="GEN-HIGH-NESTING",
                        file=file_path,
                        line=lineno,
                        message=(
                            f"Nesting depth {depth} exceeds limit of "
                            f"{MAX_NESTING_DEPTH}; consider extracting a function"
                        ),
                        severity="warning",
                        category="complexity",
                        source="generic",
                    ))
        return findings

    def _check_long_functions(
        self, file_path: Path, lines: List[str], language: str
    ) -> List[ReviewFinding]:
        """Detect functions that exceed MAX_FUNCTION_LINES lines."""
        findings: List[ReviewFinding] = []
        fn_starts = _extract_function_ranges(lines, language)
        if not fn_starts:
            return findings

        # Approximate end of each function = start of next function (or EOF)
        for i, (start_line, fn_name) in enumerate(fn_starts):
            end_line = fn_starts[i + 1][0] - 1 if i + 1 < len(fn_starts) else len(lines)
            fn_len = end_line - start_line + 1
            if fn_len > MAX_FUNCTION_LINES:
                findings.append(ReviewFinding(
                    rule_id="GEN-LONG-FUNCTION",
                    file=file_path,
                    line=start_line,
                    message=(
                        f"Function '{fn_name}' is {fn_len} lines long "
                        f"(limit: {MAX_FUNCTION_LINES}); consider splitting"
                    ),
                    severity="warning",
                    category="complexity",
                    source="generic",
                ))
        return findings
