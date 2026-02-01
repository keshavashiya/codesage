"""Tests for injection vulnerability rules (SEC010-SEC023)."""

import pytest

from codesage.security.rules.injection import INJECTION_RULES
from codesage.security.rules import get_rule_by_id

from .conftest import assert_rule_matches


class TestSQLInjectionStringFormatting:
    """Tests for SEC010 - SQL Injection via String Formatting."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC010")

    def test_detects_percent_formatting_select(self, rule):
        """Detect SELECT with %s formatting."""
        code = 'cursor.execute("SELECT * FROM users WHERE id = %s" % user_id)'
        assert_rule_matches(rule, code)

    def test_detects_percent_formatting_delete(self, rule):
        """Detect DELETE with %s formatting."""
        code = 'cursor.execute("DELETE FROM users WHERE name = %s" % name)'
        assert_rule_matches(rule, code)


class TestSQLInjectionFString:
    """Tests for SEC011 - SQL Injection via F-string."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC011")

    def test_detects_fstring_select(self, rule):
        """Detect SELECT with f-string interpolation."""
        code = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
        assert_rule_matches(rule, code)

    def test_detects_fstring_insert(self, rule):
        """Detect INSERT with f-string interpolation."""
        code = 'db.execute(f"INSERT INTO logs VALUES ({message})")'
        assert_rule_matches(rule, code)

    def test_detects_fstring_update(self, rule):
        """Detect UPDATE with f-string interpolation."""
        code = 'conn.execute(f"UPDATE users SET name = {name} WHERE id = {id}")'
        assert_rule_matches(rule, code)

    def test_ignores_parameterized_query(self, rule):
        """Should not match parameterized queries."""
        code = 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
        assert_rule_matches(rule, code, should_match=False)


class TestSQLInjectionConcatenation:
    """Tests for SEC012 - SQL Injection via Concatenation."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC012")

    def test_detects_string_concat(self, rule):
        """Detect query built with string concatenation."""
        code = 'cursor.execute("SELECT * FROM users WHERE id = " + user_id)'
        assert_rule_matches(rule, code)

    def test_detects_str_conversion(self, rule):
        """Detect query with str() conversion."""
        code = 'cursor.execute("SELECT * FROM users WHERE id = " + str(user_id))'
        assert_rule_matches(rule, code)


class TestCommandInjectionOsSystem:
    """Tests for SEC020 - Command Injection via os.system."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC020")

    def test_detects_os_system_fstring(self, rule):
        """Detect os.system with f-string."""
        code = 'os.system(f"rm -rf {user_input}")'
        assert_rule_matches(rule, code)

    def test_detects_os_system_concat(self, rule):
        """Detect os.system with concatenation."""
        code = 'os.system("ls " + directory)'
        assert_rule_matches(rule, code)

    def test_ignores_static_command(self, rule):
        """Should not match static commands."""
        code = 'os.system("ls -la")'
        assert_rule_matches(rule, code, should_match=False)


class TestCommandInjectionSubprocess:
    """Tests for SEC021 - Command Injection via subprocess shell."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC021")

    def test_detects_subprocess_run_shell_true(self, rule):
        """Detect subprocess.run with shell=True."""
        code = 'subprocess.run(cmd, shell=True)'
        assert_rule_matches(rule, code)

    def test_detects_subprocess_call_shell_true(self, rule):
        """Detect subprocess.call with shell=True."""
        code = 'subprocess.call(command, shell=True)'
        assert_rule_matches(rule, code)

    def test_detects_subprocess_popen_shell_true(self, rule):
        """Detect subprocess.Popen with shell=True."""
        code = 'subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)'
        assert_rule_matches(rule, code)

    def test_ignores_shell_false(self, rule):
        """Should not match shell=False."""
        code = 'subprocess.run(["ls", "-la"], shell=False)'
        assert_rule_matches(rule, code, should_match=False)


class TestCodeInjectionEval:
    """Tests for SEC022 - Code Injection via eval."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC022")

    def test_detects_eval_variable(self, rule):
        """Detect eval with variable."""
        code = 'result = eval(user_input)'
        assert_rule_matches(rule, code)

    def test_detects_eval_fstring(self, rule):
        """Detect eval with f-string."""
        code = 'eval(f"{expression}")'
        assert_rule_matches(rule, code)

    def test_ignores_literal_eval(self, rule):
        """Should not match ast.literal_eval."""
        code = 'import ast; ast.literal_eval(data)'
        # This tests the pattern - literal_eval is a different function
        assert_rule_matches(rule, code, should_match=False)


class TestCodeInjectionExec:
    """Tests for SEC023 - Code Injection via exec."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC023")

    def test_detects_exec_variable(self, rule):
        """Detect exec with variable."""
        code = 'exec(code_string)'
        assert_rule_matches(rule, code)

    def test_detects_exec_fstring(self, rule):
        """Detect exec with f-string."""
        code = 'exec(f"print({value})")'
        assert_rule_matches(rule, code)


class TestAllInjectionRules:
    """Integration tests for all injection rules."""

    def test_all_injection_rules_exist(self):
        """Verify all expected injection rules are defined."""
        expected_ids = ["SEC010", "SEC011", "SEC012", "SEC020", "SEC021", "SEC022", "SEC023"]
        actual_ids = [rule.id for rule in INJECTION_RULES]

        for expected in expected_ids:
            assert expected in actual_ids, f"Missing rule: {expected}"

    def test_all_rules_have_cwe_ids(self):
        """All injection rules should have CWE IDs."""
        for rule in INJECTION_RULES:
            assert rule.cwe_id, f"Rule {rule.id} missing CWE ID"

    def test_sql_injection_rules_have_correct_cwe(self):
        """SQL injection rules should have CWE-89."""
        sql_rules = ["SEC010", "SEC011", "SEC012"]
        for rule_id in sql_rules:
            rule = get_rule_by_id(rule_id)
            assert rule.cwe_id == "CWE-89", f"Rule {rule_id} should have CWE-89"

    def test_command_injection_rules_have_correct_cwe(self):
        """Command injection rules should have CWE-78."""
        cmd_rules = ["SEC020", "SEC021"]
        for rule_id in cmd_rules:
            rule = get_rule_by_id(rule_id)
            assert rule.cwe_id == "CWE-78", f"Rule {rule_id} should have CWE-78"
