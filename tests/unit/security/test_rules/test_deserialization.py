"""Tests for deserialization security rules (SEC060-SEC061)."""

import pytest

from codesage.security.rules.deserialization import DESERIALIZATION_RULES
from codesage.security.rules import get_rule_by_id

from .conftest import assert_rule_matches


class TestUnsafePickle:
    """Tests for SEC060 - Unsafe Deserialization Pickle."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC060")

    def test_detects_pickle_load(self, rule):
        """Detect pickle.load() call."""
        code = 'data = pickle.load(file)'
        assert_rule_matches(rule, code)

    def test_detects_pickle_loads(self, rule):
        """Detect pickle.loads() call."""
        code = 'obj = pickle.loads(raw_data)'
        assert_rule_matches(rule, code)

    def test_detects_import_and_use(self, rule):
        """Detect pickle use after import."""
        code = '''
import pickle
user_obj = pickle.loads(request.data)
'''
        assert_rule_matches(rule, code)

    def test_ignores_pickle_dump(self, rule):
        """Should not match pickle.dump (serialization)."""
        code = 'pickle.dump(data, file)'
        assert_rule_matches(rule, code, should_match=False)


class TestUnsafeYAML:
    """Tests for SEC061 - Unsafe YAML Load."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC061")

    def test_detects_yaml_load_unsafe(self, rule):
        """Detect yaml.load() without SafeLoader."""
        code = 'config = yaml.load(file)'
        assert_rule_matches(rule, code)

    def test_detects_yaml_load_with_loader_kwarg(self, rule):
        """Detect yaml.load with non-safe Loader."""
        code = 'data = yaml.load(content, Loader=yaml.FullLoader)'
        assert_rule_matches(rule, code)

    def test_ignores_yaml_safe_load(self, rule):
        """Should not match yaml.safe_load()."""
        code = 'config = yaml.safe_load(file)'
        assert_rule_matches(rule, code, should_match=False)

    def test_ignores_yaml_load_with_safeloader(self, rule):
        """Should not match yaml.load with SafeLoader."""
        code = 'data = yaml.load(content, Loader=SafeLoader)'
        assert_rule_matches(rule, code, should_match=False)


class TestAllDeserializationRules:
    """Integration tests for all deserialization rules."""

    def test_all_deserialization_rules_exist(self):
        """Verify all expected deserialization rules are defined."""
        expected_ids = ["SEC060", "SEC061"]
        actual_ids = [rule.id for rule in DESERIALIZATION_RULES]

        for expected in expected_ids:
            assert expected in actual_ids, f"Missing rule: {expected}"

    def test_all_rules_have_cwe_502(self):
        """All deserialization rules should have CWE-502."""
        for rule in DESERIALIZATION_RULES:
            assert rule.cwe_id == "CWE-502", f"Rule {rule.id} should have CWE-502"

    def test_all_rules_have_fix_suggestions(self):
        """All deserialization rules should have fix suggestions."""
        for rule in DESERIALIZATION_RULES:
            assert rule.fix_suggestion, f"Rule {rule.id} missing fix suggestion"

    def test_all_rules_are_high_severity(self):
        """Deserialization rules should be high severity."""
        from codesage.security.models import Severity

        for rule in DESERIALIZATION_RULES:
            assert rule.severity == Severity.HIGH, (
                f"Rule {rule.id} should be HIGH severity"
            )
