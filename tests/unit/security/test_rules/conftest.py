"""Shared fixtures for security rule tests."""

import pytest
from pathlib import Path
from typing import List, Optional

from codesage.security.models import SecurityFinding, SecurityRule
from codesage.security.scanner import SecurityScanner


@pytest.fixture
def scanner() -> SecurityScanner:
    """Create a security scanner with all rules enabled."""
    return SecurityScanner()


def assert_rule_matches(
    rule: SecurityRule,
    code: str,
    expected_matches: int = 1,
    should_match: bool = True,
) -> List:
    """Helper to test if a rule matches code.

    Args:
        rule: The security rule to test.
        code: Source code to scan.
        expected_matches: Number of expected matches.
        should_match: Whether the rule should match.

    Returns:
        List of matches found.
    """
    matches = list(rule.matches(code))

    if should_match:
        assert len(matches) >= expected_matches, (
            f"Rule {rule.id} ({rule.name}) should match but didn't.\n"
            f"Code: {code[:200]}..."
        )
    else:
        assert len(matches) == 0, (
            f"Rule {rule.id} ({rule.name}) should NOT match but did.\n"
            f"Code: {code[:200]}...\n"
            f"Matches: {[m.group(0) for m in matches]}"
        )

    return matches


def scan_code_string(
    scanner: SecurityScanner,
    code: str,
    tmp_path: Path,
    filename: str = "test.py",
) -> List[SecurityFinding]:
    """Scan a code string and return findings.

    Args:
        scanner: Security scanner instance.
        code: Source code to scan.
        tmp_path: Temporary directory.
        filename: Name for the test file.

    Returns:
        List of security findings.
    """
    test_file = tmp_path / filename
    test_file.write_text(code)
    return scanner.scan_file(test_file)
