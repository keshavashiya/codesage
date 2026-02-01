"""Tests for secret detection rules (SEC001-SEC008)."""

import pytest

from codesage.security.rules.secrets import SECRETS_RULES
from codesage.security.rules import get_rule_by_id

from .conftest import assert_rule_matches


class TestHardcodedPassword:
    """Tests for SEC001 - Hardcoded Password."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC001")

    def test_detects_password_equals_string(self, rule):
        """Detect password = 'secret123'."""
        code = 'password = "secret123"'
        assert_rule_matches(rule, code)

    def test_detects_passwd_equals_string(self, rule):
        """Detect passwd = 'mysecret'."""
        code = "passwd = 'mysecretpwd'"
        assert_rule_matches(rule, code)

    def test_detects_pwd_colon_string(self, rule):
        """Detect pwd: 'password123'."""
        code = "pwd: 'password123'"
        assert_rule_matches(rule, code)

    def test_ignores_env_variable(self, rule):
        """Should not match environment variable access."""
        code = 'password = os.getenv("PASSWORD")'
        assert_rule_matches(rule, code, should_match=False)

    def test_ignores_short_values(self, rule):
        """Should not match short values (likely placeholders)."""
        code = 'password = "abc"'  # Only 3 chars
        assert_rule_matches(rule, code, should_match=False)


class TestHardcodedAPIKey:
    """Tests for SEC002 - Hardcoded API Key."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC002")

    def test_detects_api_key_equals(self, rule):
        """Detect api_key = 'sk-abc123xyz...'."""
        code = 'api_key = "sk-abc123xyz789def456"'
        assert_rule_matches(rule, code)

    def test_detects_apikey_no_underscore(self, rule):
        """Detect apikey = 'value'."""
        code = "apikey = 'my-secret-api-key-123'"
        assert_rule_matches(rule, code)

    def test_detects_api_secret(self, rule):
        """Detect api_secret = 'value'."""
        code = 'api_secret = "super_secret_key_123"'
        assert_rule_matches(rule, code)

    def test_ignores_env_variable(self, rule):
        """Should not match environment variable access."""
        code = 'api_key = os.environ["API_KEY"]'
        assert_rule_matches(rule, code, should_match=False)


class TestAWSAccessKey:
    """Tests for SEC003 - AWS Access Key."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC003")

    def test_detects_aws_access_key_id(self, rule):
        """Detect AKIA... pattern."""
        code = 'aws_key = "AKIAIOSFODNN7EXAMPLE"'
        assert_rule_matches(rule, code)

    def test_detects_embedded_key(self, rule):
        """Detect key embedded in larger string."""
        code = 'config = {"key": "AKIAIOSFODNN7EXAMPLE"}'
        assert_rule_matches(rule, code)

    def test_ignores_non_akia_prefix(self, rule):
        """Should not match non-AKIA strings."""
        code = 'key = "NOTAKIAIOSFODNN7EXA"'
        assert_rule_matches(rule, code, should_match=False)


class TestAWSSecretKey:
    """Tests for SEC004 - AWS Secret Key."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC004")

    def test_detects_aws_secret_access_key(self, rule):
        """Detect aws_secret_access_key = 'value'."""
        # 40-char base64-ish secret
        code = 'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        assert_rule_matches(rule, code)

    def test_detects_secret_key_variant(self, rule):
        """Detect secret_key = 'value'."""
        code = 'secret_key = "abcdefghijklmnopqrstuvwxyz1234567890ABCD"'
        assert_rule_matches(rule, code)


class TestPrivateKey:
    """Tests for SEC005 - Private Key."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC005")

    def test_detects_rsa_private_key(self, rule):
        """Detect RSA private key header."""
        code = '-----BEGIN RSA PRIVATE KEY-----'
        assert_rule_matches(rule, code)

    def test_detects_generic_private_key(self, rule):
        """Detect generic private key header."""
        code = '-----BEGIN PRIVATE KEY-----'
        assert_rule_matches(rule, code)

    def test_detects_ec_private_key(self, rule):
        """Detect EC private key header."""
        code = '-----BEGIN EC PRIVATE KEY-----'
        assert_rule_matches(rule, code)

    def test_detects_openssh_private_key(self, rule):
        """Detect OpenSSH private key header."""
        code = '-----BEGIN OPENSSH PRIVATE KEY-----'
        assert_rule_matches(rule, code)

    def test_ignores_public_key(self, rule):
        """Should not match public key."""
        code = '-----BEGIN PUBLIC KEY-----'
        assert_rule_matches(rule, code, should_match=False)


class TestJWTToken:
    """Tests for SEC006 - JWT Token."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC006")

    def test_detects_jwt_token(self, rule):
        """Detect JWT token pattern."""
        # A typical JWT structure: header.payload.signature
        code = 'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"'
        assert_rule_matches(rule, code)

    def test_ignores_partial_jwt(self, rule):
        """Should not match incomplete JWT."""
        code = 'token = "eyJhbGciOiJIUzI1"'  # Too short
        assert_rule_matches(rule, code, should_match=False)


class TestGenericSecret:
    """Tests for SEC007 - Generic Secret Assignment."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC007")

    def test_detects_secret_assignment(self, rule):
        """Detect secret = 'value'."""
        code = 'secret = "my_very_secret_value"'
        assert_rule_matches(rule, code)

    def test_detects_token_assignment(self, rule):
        """Detect token = 'value'."""
        code = 'token = "abc123xyz789"'
        assert_rule_matches(rule, code)

    def test_detects_auth_key(self, rule):
        """Detect auth_key = 'value'."""
        code = 'auth_key = "authentication_key_123"'
        assert_rule_matches(rule, code)


class TestDatabaseConnectionString:
    """Tests for SEC008 - Database Connection String."""

    @pytest.fixture
    def rule(self):
        return get_rule_by_id("SEC008")

    def test_detects_postgres_connection(self, rule):
        """Detect PostgreSQL connection string."""
        code = 'db_url = "postgres://user:password@localhost:5432/mydb"'
        assert_rule_matches(rule, code)

    def test_detects_mysql_connection(self, rule):
        """Detect MySQL connection string."""
        code = 'db_url = "mysql://root:secret@db.example.com/production"'
        assert_rule_matches(rule, code)

    def test_detects_mongodb_connection(self, rule):
        """Detect MongoDB connection string."""
        code = 'mongo_url = "mongodb://admin:pass123@mongo.server.com/data"'
        assert_rule_matches(rule, code)

    def test_detects_redis_connection(self, rule):
        """Detect Redis connection string."""
        code = 'redis_url = "redis://default:mypassword@cache.server.com:6379"'
        assert_rule_matches(rule, code)

    def test_ignores_url_without_credentials(self, rule):
        """Should not match URL without embedded credentials."""
        code = 'db_url = "postgres://localhost:5432/mydb"'
        assert_rule_matches(rule, code, should_match=False)


class TestAllSecretsRules:
    """Integration tests for all secrets rules."""

    def test_all_secrets_rules_exist(self):
        """Verify all expected secrets rules are defined."""
        expected_ids = [f"SEC00{i}" for i in range(1, 9)]
        actual_ids = [rule.id for rule in SECRETS_RULES]

        for expected in expected_ids:
            assert expected in actual_ids, f"Missing rule: {expected}"

    def test_all_rules_have_cwe_ids(self):
        """All secrets rules should have CWE IDs."""
        for rule in SECRETS_RULES:
            assert rule.cwe_id, f"Rule {rule.id} missing CWE ID"

    def test_all_rules_have_fix_suggestions(self):
        """All secrets rules should have fix suggestions."""
        for rule in SECRETS_RULES:
            assert rule.fix_suggestion, f"Rule {rule.id} missing fix suggestion"
