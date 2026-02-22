"""Unit tests for the unified review pipeline and related modules."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codesage.review.models import (
    FileChange,
    IssueSeverity,
    ReviewFinding,
    UnifiedReviewResult,
)


# ---------------------------------------------------------------------------
# IssueSeverity comparison
# ---------------------------------------------------------------------------


class TestIssueSeverityOrder:
    def test_critical_greater_than_warning(self):
        assert IssueSeverity.CRITICAL > IssueSeverity.WARNING

    def test_high_between_critical_and_warning(self):
        assert IssueSeverity.HIGH > IssueSeverity.WARNING
        assert IssueSeverity.HIGH < IssueSeverity.CRITICAL

    def test_suggestion_less_than_warning(self):
        assert IssueSeverity.SUGGESTION < IssueSeverity.WARNING

    def test_praise_less_than_all(self):
        assert IssueSeverity.PRAISE < IssueSeverity.SUGGESTION

    def test_equal(self):
        assert IssueSeverity.HIGH == IssueSeverity.HIGH
        assert IssueSeverity.HIGH <= IssueSeverity.HIGH
        assert IssueSeverity.HIGH >= IssueSeverity.HIGH


# ---------------------------------------------------------------------------
# UnifiedReviewResult
# ---------------------------------------------------------------------------


def _make_result(severities):
    findings = [
        ReviewFinding(
            severity=s,
            category="test",
            file=Path("test.py"),
            message=f"msg {s}",
        )
        for s in severities
    ]
    return UnifiedReviewResult(findings=findings)


class TestUnifiedReviewResult:
    def test_counts(self):
        result = _make_result(["critical", "high", "warning", "suggestion", "praise"])
        assert result.critical_count == 1
        assert result.high_count == 1
        assert result.warning_count == 1
        assert result.suggestion_count == 1
        assert result.praise_count == 1

    def test_active_findings_excludes_suppressed(self):
        result = _make_result(["critical", "warning"])
        result.findings[1].suppressed = True
        assert len(result.active_findings) == 1
        assert result.active_findings[0].severity == "critical"
        assert result.suppressed_count == 1

    def test_has_blocking_issues_critical_threshold(self):
        result = _make_result(["warning"])
        assert not result.has_blocking_issues("critical")

    def test_has_blocking_issues_high_threshold(self):
        result = _make_result(["high"])
        assert result.has_blocking_issues("high")
        assert not result.has_blocking_issues("critical")

    def test_exit_code_clean(self):
        result = _make_result(["warning"])
        assert result.exit_code == 0

    def test_exit_code_blocked(self):
        result = _make_result(["critical"])
        assert result.exit_code == 1

    def test_to_dict_keys(self):
        result = _make_result(["warning", "suggestion"])
        d = result.to_dict()
        assert "summary" in d
        assert "findings" in d
        assert d["summary"]["warnings"] == 1
        assert d["summary"]["suggestions"] == 1

    def test_to_sarif_structure(self):
        result = _make_result(["critical"])
        result.findings[0].rule_id = "TEST-001"
        sarif = result.to_sarif()
        assert sarif["version"] == "2.1.0"
        run = sarif["runs"][0]
        assert run["results"][0]["ruleId"] == "TEST-001"
        assert run["results"][0]["level"] == "error"


# ---------------------------------------------------------------------------
# ReviewFinding.to_sarif_result
# ---------------------------------------------------------------------------


class TestReviewFindingToSarif:
    def test_critical_maps_to_error(self):
        f = ReviewFinding(
            severity="critical",
            category="security",
            file=Path("app.py"),
            line=10,
            rule_id="SEC-001",
            message="SQL injection",
        )
        sarif = f.to_sarif_result()
        assert sarif["level"] == "error"
        assert sarif["ruleId"] == "SEC-001"
        region = sarif["locations"][0]["physicalLocation"]["region"]
        assert region["startLine"] == 10

    def test_suggestion_maps_to_note(self):
        f = ReviewFinding(
            severity="suggestion",
            category="naming",
            file=Path("utils.py"),
            message="Use snake_case",
        )
        sarif = f.to_sarif_result()
        assert sarif["level"] == "note"
        # No region when no line
        assert "region" not in sarif["locations"][0]["physicalLocation"]


# ---------------------------------------------------------------------------
# PythonBadPracticeChecker
# ---------------------------------------------------------------------------


class TestPythonBadPracticeChecker:
    def setup_method(self):
        from codesage.review.checks.python_checks import PythonBadPracticeChecker

        self.checker = PythonBadPracticeChecker()

    def _check(self, code):
        return self.checker.check(Path("test.py"), textwrap.dedent(code))

    def test_magic_number(self):
        findings = self._check("""
            x = 42
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-MAGIC-NUM" in rule_ids

    def test_no_magic_number_for_zero_one(self):
        findings = self._check("""
            x = 0
            y = 1
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-MAGIC-NUM" not in rule_ids

    def test_bare_except(self):
        findings = self._check("""
            try:
                pass
            except:
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-BARE-EXCEPT" in rule_ids

    def test_mutable_default(self):
        findings = self._check("""
            def foo(x=[]):
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-MUTABLE-DEFAULT" in rule_ids

    def test_missing_return_type(self):
        findings = self._check("""
            def my_function(x):
                return x
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-MISSING-RETURN-TYPE" in rule_ids

    def test_has_return_type_no_finding(self):
        findings = self._check("""
            def my_function(x) -> int:
                return x
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-MISSING-RETURN-TYPE" not in rule_ids

    def test_syntax_error_returns_empty(self):
        findings = self.checker.check(Path("bad.py"), "def (:")
        assert findings == []


# ---------------------------------------------------------------------------
# ComplexityChecker
# ---------------------------------------------------------------------------


class TestComplexityChecker:
    def setup_method(self):
        from codesage.review.checks.complexity import ComplexityChecker

        self.checker = ComplexityChecker(threshold=3)

    def _check(self, code):
        return self.checker.check(Path("test.py"), textwrap.dedent(code))

    def test_simple_function_no_finding(self):
        findings = self._check("""
            def simple():
                return 1
        """)
        assert findings == []

    def test_complex_function_flagged(self):
        findings = self._check("""
            def complex(a, b, c, d):
                if a:
                    pass
                if b:
                    pass
                if c:
                    pass
                if d:
                    pass
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "PY-HIGH-COMPLEXITY"


# ---------------------------------------------------------------------------
# StructureChecker
# ---------------------------------------------------------------------------


class TestStructureChecker:
    def setup_method(self):
        from codesage.review.checks.structure import StructureChecker

        self.checker = StructureChecker(
            max_function_lines=5,
            max_nesting_depth=2,
            max_parameters=2,
        )

    def _check(self, code):
        return self.checker.check(Path("test.py"), textwrap.dedent(code))

    def test_long_function(self):
        body = "    pass\n" * 10
        code = f"def long_func():\n{body}"
        findings = self._check(code)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-LONG-FUNCTION" in rule_ids

    def test_too_many_params(self):
        findings = self._check("""
            def foo(a, b, c):
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-TOO-MANY-PARAMS" in rule_ids


# ---------------------------------------------------------------------------
# NamingChecker
# ---------------------------------------------------------------------------


class TestNamingChecker:
    def setup_method(self):
        from codesage.review.checks.naming import NamingChecker

        self.checker = NamingChecker()

    def _check(self, code):
        return self.checker.check(Path("test.py"), textwrap.dedent(code))

    def test_bad_function_name(self):
        findings = self._check("""
            def MyFunction():
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-FUNC-NAMING" in rule_ids

    def test_good_function_name(self):
        findings = self._check("""
            def my_function():
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-FUNC-NAMING" not in rule_ids

    def test_bad_class_name(self):
        findings = self._check("""
            class my_class:
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-CLASS-NAMING" in rule_ids

    def test_good_class_name(self):
        findings = self._check("""
            class MyClass:
                pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-CLASS-NAMING" not in rule_ids

    def test_dunder_methods_exempt(self):
        findings = self._check("""
            class Foo:
                def __init__(self):
                    pass
                def __str__(self):
                    pass
        """)
        rule_ids = [f.rule_id for f in findings]
        assert "PY-FUNC-NAMING" not in rule_ids


# ---------------------------------------------------------------------------
# SuppressionParser
# ---------------------------------------------------------------------------


class TestSuppressionParser:
    def setup_method(self):
        from codesage.review.suppression import SuppressionParser

        self.parser = SuppressionParser()

    def test_inline_ignore_all(self):
        content = "x = 42  # codesage:ignore\n"
        result = self.parser.parse_content(content)
        # Line 1 should have empty set (ignore all)
        assert 1 in result
        assert len(result[1]) == 0  # empty = ignore all

    def test_inline_ignore_specific_rule(self):
        content = "x = 42  # codesage:ignore PY-MAGIC-NUM\n"
        result = self.parser.parse_content(content)
        assert 1 in result
        assert "PY-MAGIC-NUM" in result[1]

    def test_ignore_next_line(self):
        content = "# codesage:ignore-next-line\nx = 42\n"
        result = self.parser.parse_content(content)
        assert 2 in result

    def test_no_suppression(self):
        content = "x = 42\ny = 1\n"
        result = self.parser.parse_content(content)
        assert result == {}


class TestApplySuppressions:
    def test_suppresses_matching_finding(self):
        from codesage.review.suppression import apply_suppressions

        findings = [
            ReviewFinding(
                severity="suggestion",
                category="practice",
                file=Path("test.py"),
                line=5,
                rule_id="PY-MAGIC-NUM",
                message="Magic number",
            )
        ]
        file_suppressions = {Path("test.py"): {5: {"PY-MAGIC-NUM"}}}
        apply_suppressions(findings, file_suppressions)
        assert findings[0].suppressed is True

    def test_does_not_suppress_different_rule(self):
        from codesage.review.suppression import apply_suppressions

        findings = [
            ReviewFinding(
                severity="warning",
                category="complexity",
                file=Path("test.py"),
                line=5,
                rule_id="PY-HIGH-COMPLEXITY",
                message="Complex",
            )
        ]
        file_suppressions = {Path("test.py"): {5: {"PY-MAGIC-NUM"}}}
        apply_suppressions(findings, file_suppressions)
        assert findings[0].suppressed is False

    def test_empty_set_suppresses_all_on_line(self):
        from codesage.review.suppression import apply_suppressions

        findings = [
            ReviewFinding(
                severity="warning",
                category="complexity",
                file=Path("test.py"),
                line=3,
                rule_id="PY-HIGH-COMPLEXITY",
                message="Complex",
            )
        ]
        file_suppressions = {Path("test.py"): {3: set()}}  # ignore all
        apply_suppressions(findings, file_suppressions)
        assert findings[0].suppressed is True


# ---------------------------------------------------------------------------
# RichReviewOutput (smoke test)
# ---------------------------------------------------------------------------


class TestRichReviewOutput:
    def test_renders_without_error(self):
        from io import StringIO

        from rich.console import Console

        from codesage.review.output import RichReviewOutput

        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        output = RichReviewOutput(console=console)
        result = UnifiedReviewResult(
            findings=[
                ReviewFinding(
                    severity="critical",
                    category="security",
                    file=Path("app.py"),
                    line=10,
                    rule_id="SEC-001",
                    message="SQL injection",
                    suggestion="Use parameterized queries",
                )
            ],
            files_changed=[
                FileChange(path=Path("app.py"), status="M", additions=20, deletions=5)
            ],
            mode="fast",
            duration_ms=250.0,
        )
        output.print_result(result, severity_threshold="high")
        text = buf.getvalue()
        assert "BLOCKED" in text
        assert "SQL injection" in text
        assert "SEC-001" in text

    def test_passes_when_no_blocking_issues(self):
        from io import StringIO

        from rich.console import Console

        from codesage.review.output import RichReviewOutput

        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        output = RichReviewOutput(console=console)
        result = UnifiedReviewResult(
            findings=[
                ReviewFinding(
                    severity="suggestion",
                    category="naming",
                    file=Path("utils.py"),
                    message="Use snake_case",
                )
            ],
            mode="fast",
            duration_ms=100.0,
        )
        output.print_result(result, severity_threshold="high")
        text = buf.getvalue()
        assert "Passed" in text
