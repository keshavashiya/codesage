"""Tests for XSS and path traversal rules (SEC030-SEC040)."""

import pytest

from codesage.security.rules.xss import XSS_RULES
from codesage.security.rules import get_rule_by_id

from .conftest import assert_rule_matches


class TestXSSUnescapedOutput:
    """Tests for SEC030 - XSS Unescaped Output."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC030")

    def test_detects_unescaped_jinja(self, rule):
        """Detect unescaped Jinja2 template variable."""
        code = '{{ user_input }}'
        assert_rule_matches(rule, code)

    def test_detects_variable_in_template(self, rule):
        """Detect variable without escape filter."""
        code = '<div>{{ message }}</div>'
        assert_rule_matches(rule, code)


class TestXSSInnerHTML:
    """Tests for SEC031 - XSS innerHTML Assignment."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC031")

    def test_detects_innerhtml_variable(self, rule):
        """Detect innerHTML assignment with variable."""
        code = 'element.innerHTML = userContent;'
        assert_rule_matches(rule, code)

    def test_detects_innerhtml_concatenation(self, rule):
        """Detect innerHTML with string concatenation."""
        code = 'div.innerHTML = "<p>" + message + "</p>";'
        assert_rule_matches(rule, code)

    def test_ignores_textcontent(self, rule):
        """Should not match textContent."""
        code = 'element.textContent = userInput;'
        assert_rule_matches(rule, code, should_match=False)


class TestXSSDocumentWrite:
    """Tests for SEC032 - XSS document.write."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC032")

    def test_detects_document_write(self, rule):
        """Detect document.write call."""
        code = 'document.write(userInput);'
        assert_rule_matches(rule, code)

    def test_detects_document_write_string(self, rule):
        """Detect document.write with string."""
        code = 'document.write("<script>" + code + "</script>");'
        assert_rule_matches(rule, code)


class TestPathTraversal:
    """Tests for SEC040 - Path Traversal."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC040")

    def test_detects_open_with_concat(self, rule):
        """Detect open() with concatenated path."""
        code = 'open(base_path + filename)'
        assert_rule_matches(rule, code)

    def test_detects_read_with_concat(self, rule):
        """Detect read operation with concatenated path."""
        code = 'file.read(directory + user_file)'
        assert_rule_matches(rule, code)

    def test_ignores_static_path(self, rule):
        """Should not match static paths."""
        code = 'open("/etc/config.ini")'
        assert_rule_matches(rule, code, should_match=False)


class TestAllXSSRules:
    """Integration tests for all XSS rules."""

    def test_all_xss_rules_exist(self):
        """Verify all expected XSS rules are defined."""
        expected_ids = ["SEC030", "SEC031", "SEC032", "SEC040"]
        actual_ids = [rule.id for rule in XSS_RULES]

        for expected in expected_ids:
            assert expected in actual_ids, f"Missing rule: {expected}"

    def test_all_rules_have_cwe_ids(self):
        """All XSS rules should have CWE IDs."""
        for rule in XSS_RULES:
            assert rule.cwe_id, f"Rule {rule.id} missing CWE ID"

    def test_xss_rules_have_correct_cwe(self):
        """XSS rules should have CWE-79."""
        xss_rule_ids = ["SEC030", "SEC031", "SEC032"]
        for rule_id in xss_rule_ids:
            rule = get_rule_by_id(rule_id)
            assert rule.cwe_id == "CWE-79", f"Rule {rule_id} should have CWE-79"

    def test_path_traversal_has_correct_cwe(self):
        """Path traversal rule should have CWE-22."""
        rule = get_rule_by_id("SEC040")
        assert rule.cwe_id == "CWE-22"
