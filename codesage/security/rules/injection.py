"""Injection vulnerability rules.

Patterns for detecting SQL, command, and code injection vulnerabilities.
"""

from codesage.security.models import SecurityRule, Severity

INJECTION_RULES = [
    # SQL Injection
    SecurityRule(
        id="SEC010",
        name="SQL Injection - String Formatting",
        pattern=r"""(?:execute|executemany|raw)\s*\(\s*[f]?['\"].*?(?:SELECT|INSERT|UPDATE|DELETE|DROP).*?%[sd]""",
        severity=Severity.HIGH,
        message="Potential SQL injection via string formatting",
        description="SQL queries should use parameterized queries, not string formatting",
        category="injection",
        cwe_id="CWE-89",
        fix_suggestion="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
    ),
    SecurityRule(
        id="SEC011",
        name="SQL Injection - F-string",
        pattern=r"""(?:execute|executemany|raw)\s*\(\s*f['\"].*?(?:SELECT|INSERT|UPDATE|DELETE|DROP).*?\{""",
        severity=Severity.HIGH,
        message="SQL injection via f-string interpolation",
        description="Never use f-strings for SQL queries",
        category="injection",
        cwe_id="CWE-89",
        fix_suggestion="Use parameterized queries instead of f-strings",
    ),
    SecurityRule(
        id="SEC012",
        name="SQL Injection - Concatenation",
        pattern=r"""(?:execute|executemany)\s*\([^)]*\+\s*(?:str\()?[a-zA-Z_][a-zA-Z0-9_]*""",
        severity=Severity.HIGH,
        message="Potential SQL injection via string concatenation",
        description="SQL queries should not be built with string concatenation",
        category="injection",
        cwe_id="CWE-89",
        fix_suggestion="Use parameterized queries",
    ),
    # Command Injection
    SecurityRule(
        id="SEC020",
        name="Command Injection - os.system",
        pattern=r"""os\.system\s*\(\s*[f]?['\"].*?\{|os\.system\s*\([^)]*\+""",
        severity=Severity.HIGH,
        message="Potential command injection via os.system",
        description="os.system with user input can lead to command injection",
        category="injection",
        cwe_id="CWE-78",
        fix_suggestion="Use subprocess with shell=False and list arguments",
    ),
    SecurityRule(
        id="SEC021",
        name="Command Injection - subprocess shell",
        pattern=r"""subprocess\.(?:run|call|Popen)\s*\([^)]*shell\s*=\s*True""",
        severity=Severity.MEDIUM,
        message="Subprocess with shell=True detected",
        description="Using shell=True can be dangerous with untrusted input",
        category="injection",
        cwe_id="CWE-78",
        fix_suggestion="Use shell=False and pass arguments as a list",
    ),
    # Code Injection
    SecurityRule(
        id="SEC022",
        name="Code Injection - eval",
        # Use negative lookbehind to exclude ast.literal_eval
        pattern=r"""(?<!literal_)eval\s*\(\s*(?:[a-zA-Z_][a-zA-Z0-9_]*|[f]?['\"].*?\{)""",
        severity=Severity.HIGH,
        message="Use of eval() with dynamic content",
        description="eval() can execute arbitrary code and should be avoided",
        category="injection",
        cwe_id="CWE-95",
        fix_suggestion="Use ast.literal_eval() for safe evaluation of literals",
    ),
    SecurityRule(
        id="SEC023",
        name="Code Injection - exec",
        pattern=r"""exec\s*\(\s*(?:[a-zA-Z_][a-zA-Z0-9_]*|[f]?['\"].*?\{)""",
        severity=Severity.HIGH,
        message="Use of exec() with dynamic content",
        description="exec() can execute arbitrary code",
        category="injection",
        cwe_id="CWE-95",
        fix_suggestion="Avoid exec() with dynamic content",
    ),
]
