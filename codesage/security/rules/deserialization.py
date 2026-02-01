"""Deserialization security rules.

Patterns for detecting unsafe deserialization practices.
"""

from codesage.security.models import SecurityRule, Severity

DESERIALIZATION_RULES = [
    SecurityRule(
        id="SEC060",
        name="Unsafe Deserialization - Pickle",
        pattern=r"""pickle\.loads?\s*\(""",
        severity=Severity.HIGH,
        message="Use of pickle for deserialization",
        description="pickle can execute arbitrary code during deserialization",
        category="deserialization",
        cwe_id="CWE-502",
        fix_suggestion="Use JSON or other safe serialization formats",
    ),
    SecurityRule(
        id="SEC061",
        name="Unsafe YAML Load",
        # Match yaml.load() that does NOT use SafeLoader
        # Only SafeLoader is truly safe - FullLoader and BaseLoader are not
        pattern=r"""yaml\.load\s*\((?![^)]*SafeLoader)""",
        severity=Severity.HIGH,
        message="Unsafe YAML loading detected",
        description="yaml.load without SafeLoader can execute arbitrary code",
        category="deserialization",
        cwe_id="CWE-502",
        fix_suggestion="Use yaml.safe_load() or yaml.load(data, Loader=SafeLoader)",
    ),
]
