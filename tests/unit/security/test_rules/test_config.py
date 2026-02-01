"""Tests for configuration security rules (SEC050-SEC071)."""

import pytest

from codesage.security.rules.config import CONFIG_RULES
from codesage.security.rules import get_rule_by_id

from .conftest import assert_rule_matches


class TestDebugModeEnabled:
    """Tests for SEC050 - Debug Mode Enabled."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC050")

    def test_detects_debug_true_uppercase(self, rule):
        """Detect DEBUG = True."""
        code = 'DEBUG = True'
        assert_rule_matches(rule, code)

    def test_detects_debug_true_lowercase(self, rule):
        """Detect debug = True."""
        code = 'debug = True'
        assert_rule_matches(rule, code)

    def test_detects_flask_run_debug(self, rule):
        """Detect app.run(debug=True)."""
        code = 'app.run(host="0.0.0.0", debug=True)'
        assert_rule_matches(rule, code)

    def test_ignores_debug_false(self, rule):
        """Should not match DEBUG = False."""
        code = 'DEBUG = False'
        assert_rule_matches(rule, code, should_match=False)


class TestInsecureSSL:
    """Tests for SEC051 - Insecure SSL/TLS."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC051")

    def test_detects_verify_false(self, rule):
        """Detect verify=False in requests."""
        code = 'requests.get(url, verify=False)'
        assert_rule_matches(rule, code)

    def test_detects_ssl_verify_false(self, rule):
        """Detect ssl_verify=False."""
        code = 'client = Client(ssl_verify=False)'
        assert_rule_matches(rule, code)

    def test_detects_ssl_dot_verify_false(self, rule):
        """Detect ssl.verify=False."""
        code = 'ssl.verify = False'
        assert_rule_matches(rule, code)

    def test_ignores_verify_true(self, rule):
        """Should not match verify=True."""
        code = 'requests.get(url, verify=True)'
        assert_rule_matches(rule, code, should_match=False)


class TestStackTraceExposure:
    """Tests for SEC070 - Stack Trace Exposure."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC070")

    def test_detects_traceback_print_exc(self, rule):
        """Detect traceback.print_exc()."""
        code = 'traceback.print_exc()'
        assert_rule_matches(rule, code)

    def test_detects_print_exception(self, rule):
        """Detect print(e) in exception handler."""
        code = '''
try:
    risky_operation()
except Exception as e:
    print(e)
'''
        assert_rule_matches(rule, code)

    def test_detects_print_error(self, rule):
        """Detect print(error)."""
        code = '''
except ValueError as error:
    print(error)
'''
        assert_rule_matches(rule, code)


class TestVerboseErrorMessages:
    """Tests for SEC071 - Verbose Error Messages."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC071")

    def test_detects_exception_with_password(self, rule):
        """Detect exception message containing interpolated password."""
        code = 'raise Exception(f"Invalid password: {password}")'
        assert_rule_matches(rule, code)

    def test_detects_error_with_secret(self, rule):
        """Detect error message containing interpolated secret."""
        code = 'return Error(f"Failed with secret: {secret}")'
        assert_rule_matches(rule, code)

    def test_detects_exception_with_token(self, rule):
        """Detect exception message containing interpolated token."""
        code = 'raise ValueError(f"Token error: {token}")'
        assert_rule_matches(rule, code)

    def test_ignores_env_var_name_mentions(self, rule):
        """Should not flag mentions of env var names (not actual values)."""
        code = 'raise ValueError("API_KEY environment variable required")'
        assert_rule_matches(rule, code, should_match=False)


class TestAllConfigRules:
    """Integration tests for all configuration rules."""

    def test_all_config_rules_exist(self):
        """Verify all expected config rules are defined."""
        expected_ids = ["SEC050", "SEC051", "SEC070", "SEC071"]
        actual_ids = [rule.id for rule in CONFIG_RULES]

        for expected in expected_ids:
            assert expected in actual_ids, f"Missing rule: {expected}"

    def test_all_rules_have_cwe_ids(self):
        """All config rules should have CWE IDs."""
        for rule in CONFIG_RULES:
            assert rule.cwe_id, f"Rule {rule.id} missing CWE ID"

    def test_debug_mode_has_correct_cwe(self):
        """Debug mode rule should have CWE-489."""
        rule = get_rule_by_id("SEC050")
        assert rule.cwe_id == "CWE-489"

    def test_insecure_ssl_has_correct_cwe(self):
        """Insecure SSL rule should have CWE-295."""
        rule = get_rule_by_id("SEC051")
        assert rule.cwe_id == "CWE-295"

    def test_info_disclosure_rules_have_correct_cwe(self):
        """Information disclosure rules should have CWE-209."""
        info_rules = ["SEC070", "SEC071"]
        for rule_id in info_rules:
            rule = get_rule_by_id(rule_id)
            assert rule.cwe_id == "CWE-209", f"Rule {rule_id} should have CWE-209"

    def test_all_rules_have_fix_suggestions(self):
        """All config rules should have fix suggestions."""
        for rule in CONFIG_RULES:
            assert rule.fix_suggestion, f"Rule {rule.id} missing fix suggestion"
