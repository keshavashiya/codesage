"""Configuration security rules.

Patterns for detecting insecure configurations.
"""

from codesage.security.models import SecurityRule, Severity

CONFIG_RULES = [
    SecurityRule(
        id="SEC050",
        name="Debug Mode Enabled",
        pattern=r"""DEBUG\s*=\s*True|debug\s*=\s*True|\.run\s*\([^)]*debug\s*=\s*True""",
        severity=Severity.MEDIUM,
        message="Debug mode appears to be enabled",
        description="Debug mode should be disabled in production",
        category="configuration",
        cwe_id="CWE-489",
        fix_suggestion="Use environment-based configuration for debug settings",
    ),
    SecurityRule(
        id="SEC051",
        name="Insecure SSL/TLS",
        pattern=r"""verify\s*=\s*False|ssl[_.]verify\s*=\s*False""",
        severity=Severity.HIGH,
        message="SSL/TLS verification disabled",
        description="Disabling SSL verification allows MITM attacks",
        category="configuration",
        cwe_id="CWE-295",
        fix_suggestion="Enable SSL verification or use proper certificates",
    ),
    SecurityRule(
        id="SEC070",
        name="Stack Trace Exposure",
        pattern=r"""traceback\.print_exc|print\s*\(\s*(?:e|err|error|exception)\s*\)""",
        severity=Severity.LOW,
        message="Potential stack trace exposure",
        description="Stack traces can reveal sensitive information",
        category="information_disclosure",
        cwe_id="CWE-209",
        fix_suggestion="Log errors securely without exposing to users",
    ),
    SecurityRule(
        id="SEC071",
        name="Verbose Error Messages",
        # Match raise/return with Error types that interpolate sensitive variables
        # Look for {password}, {secret}, {key}, {token} patterns (f-string interpolation)
        pattern=r"""(?:raise|return)\s+[A-Za-z]*(?:Error|Exception)\s*\([^)]*\{(?:password|secret|api_key|token|credential)[^}]*\}""",
        severity=Severity.MEDIUM,
        message="Sensitive data in error message",
        description="Error messages should not contain sensitive data",
        category="information_disclosure",
        cwe_id="CWE-209",
        fix_suggestion="Use generic error messages for users",
    ),
]
