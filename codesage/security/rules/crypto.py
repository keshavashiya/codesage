"""Cryptography rules.

Patterns for detecting weak or insecure cryptographic practices.
"""

from codesage.security.models import SecurityRule, Severity

CRYPTO_RULES = [
    SecurityRule(
        id="SEC052",
        name="Weak Cryptography - MD5",
        pattern=r"""(?:hashlib\.)?md5\s*\(""",
        severity=Severity.MEDIUM,
        message="Use of MD5 hash detected",
        description="MD5 is cryptographically weak and should not be used for security",
        category="cryptography",
        cwe_id="CWE-328",
        fix_suggestion="Use SHA-256 or stronger: hashlib.sha256()",
    ),
    SecurityRule(
        id="SEC053",
        name="Weak Cryptography - SHA1",
        pattern=r"""(?:hashlib\.)?sha1\s*\(""",
        severity=Severity.LOW,
        message="Use of SHA1 hash detected",
        description="SHA1 is deprecated for security purposes",
        category="cryptography",
        cwe_id="CWE-328",
        fix_suggestion="Use SHA-256 or stronger",
    ),
    SecurityRule(
        id="SEC054",
        name="Hardcoded IV/Nonce",
        pattern=r"""(?:iv|nonce)\s*=\s*[b]?['\"][^'\"]+['\"]""",
        severity=Severity.HIGH,
        message="Hardcoded initialization vector or nonce",
        description="IVs and nonces should be randomly generated",
        category="cryptography",
        cwe_id="CWE-329",
        fix_suggestion="Generate random IVs: os.urandom(16)",
    ),
]
