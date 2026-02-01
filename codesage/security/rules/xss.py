"""Cross-Site Scripting (XSS) rules.

Patterns for detecting potential XSS vulnerabilities.
"""

from codesage.security.models import SecurityRule, Severity

XSS_RULES = [
    SecurityRule(
        id="SEC030",
        name="XSS - Unescaped Output",
        # Match Jinja2 template patterns but exclude Cypher query syntax ({{id: $var}})
        pattern=r"""\{\{\s*[a-zA-Z_][a-zA-Z0-9_.]*\s*\}\}(?!\s*\|.*?escape)""",
        severity=Severity.MEDIUM,
        message="Potentially unescaped template output",
        description="Template variables should be escaped to prevent XSS",
        category="xss",
        cwe_id="CWE-79",
        fix_suggestion="Use appropriate escaping filters in templates",
        # Only check template/web files
        file_patterns=["*.html", "*.jinja", "*.jinja2", "*.j2", "*.htm"],
    ),
    SecurityRule(
        id="SEC031",
        name="XSS - innerHTML Assignment",
        pattern=r"""\.innerHTML\s*=\s*[^'\"]+""",
        severity=Severity.MEDIUM,
        message="Direct innerHTML assignment detected",
        description="innerHTML with user input can lead to XSS",
        category="xss",
        cwe_id="CWE-79",
        fix_suggestion="Use textContent or proper sanitization",
        file_patterns=["*.js", "*.ts", "*.jsx", "*.tsx", "*.html"],
    ),
    SecurityRule(
        id="SEC032",
        name="XSS - document.write",
        pattern=r"""document\.write\s*\(""",
        severity=Severity.MEDIUM,
        message="Use of document.write() detected",
        description="document.write can be used for XSS attacks",
        category="xss",
        cwe_id="CWE-79",
        fix_suggestion="Use DOM manipulation methods instead",
        file_patterns=["*.js", "*.ts", "*.jsx", "*.tsx", "*.html"],
    ),
    SecurityRule(
        id="SEC040",
        name="Path Traversal",
        pattern=r"""(?:open|read|write)\s*\([^)]*\+\s*[a-zA-Z_][a-zA-Z0-9_]*""",
        severity=Severity.MEDIUM,
        message="Potential path traversal vulnerability",
        description="File paths built with user input may allow path traversal",
        category="path_traversal",
        cwe_id="CWE-22",
        fix_suggestion="Validate and sanitize file paths, use os.path.basename()",
    ),
]
