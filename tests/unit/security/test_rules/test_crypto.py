"""Tests for cryptography rules (SEC052-SEC054)."""

import pytest

from codesage.security.rules.crypto import CRYPTO_RULES
from codesage.security.rules import get_rule_by_id

from .conftest import assert_rule_matches


class TestWeakCryptoMD5:
    """Tests for SEC052 - Weak Cryptography MD5."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC052")

    def test_detects_hashlib_md5(self, rule):
        """Detect hashlib.md5() call."""
        code = 'hash = hashlib.md5(data)'
        assert_rule_matches(rule, code)

    def test_detects_md5_direct(self, rule):
        """Detect md5() direct call."""
        code = 'digest = md5(password.encode())'
        assert_rule_matches(rule, code)

    def test_detects_md5_with_update(self, rule):
        """Detect md5() with subsequent calls."""
        code = 'h = hashlib.md5()\nh.update(data)'
        assert_rule_matches(rule, code)


class TestWeakCryptoSHA1:
    """Tests for SEC053 - Weak Cryptography SHA1."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC053")

    def test_detects_hashlib_sha1(self, rule):
        """Detect hashlib.sha1() call."""
        code = 'hash = hashlib.sha1(data)'
        assert_rule_matches(rule, code)

    def test_detects_sha1_direct(self, rule):
        """Detect sha1() direct call."""
        code = 'digest = sha1(content.encode())'
        assert_rule_matches(rule, code)

    def test_ignores_sha256(self, rule):
        """Should not match sha256."""
        code = 'hash = hashlib.sha256(data)'
        assert_rule_matches(rule, code, should_match=False)


class TestHardcodedIVNonce:
    """Tests for SEC054 - Hardcoded IV/Nonce."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC054")

    def test_detects_hardcoded_iv(self, rule):
        """Detect hardcoded IV assignment."""
        code = 'iv = "1234567890123456"'
        assert_rule_matches(rule, code)

    def test_detects_hardcoded_nonce(self, rule):
        """Detect hardcoded nonce assignment."""
        code = "nonce = b'random_nonce_value'"
        assert_rule_matches(rule, code)

    def test_detects_iv_in_bytes(self, rule):
        """Detect IV as bytes literal."""
        code = 'iv = b"0000000000000000"'
        assert_rule_matches(rule, code)

    def test_ignores_random_iv(self, rule):
        """Should not match randomly generated IV."""
        code = 'iv = os.urandom(16)'
        assert_rule_matches(rule, code, should_match=False)


class TestAllCryptoRules:
    """Integration tests for all cryptography rules."""

    def test_all_crypto_rules_exist(self):
        """Verify all expected crypto rules are defined."""
        expected_ids = ["SEC052", "SEC053", "SEC054"]
        actual_ids = [rule.id for rule in CRYPTO_RULES]

        for expected in expected_ids:
            assert expected in actual_ids, f"Missing rule: {expected}"

    def test_all_rules_have_cwe_ids(self):
        """All crypto rules should have CWE IDs."""
        for rule in CRYPTO_RULES:
            assert rule.cwe_id, f"Rule {rule.id} missing CWE ID"

    def test_weak_hash_rules_have_correct_cwe(self):
        """Weak hash rules should have CWE-328."""
        weak_hash_rules = ["SEC052", "SEC053"]
        for rule_id in weak_hash_rules:
            rule = get_rule_by_id(rule_id)
            assert rule.cwe_id == "CWE-328", f"Rule {rule_id} should have CWE-328"

    def test_hardcoded_iv_has_correct_cwe(self):
        """Hardcoded IV rule should have CWE-329."""
        rule = get_rule_by_id("SEC054")
        assert rule.cwe_id == "CWE-329"

    def test_all_rules_have_fix_suggestions(self):
        """All crypto rules should have fix suggestions."""
        for rule in CRYPTO_RULES:
            assert rule.fix_suggestion, f"Rule {rule.id} missing fix suggestion"
