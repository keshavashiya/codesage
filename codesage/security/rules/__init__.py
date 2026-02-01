"""Security rules registry.

Aggregates all security rules from category modules.
"""

from typing import List, Optional

from codesage.security.models import SecurityRule, Severity
from codesage.security.rules.secrets import SECRETS_RULES
from codesage.security.rules.injection import INJECTION_RULES
from codesage.security.rules.xss import XSS_RULES
from codesage.security.rules.crypto import CRYPTO_RULES
from codesage.security.rules.config import CONFIG_RULES
from codesage.security.rules.deserialization import DESERIALIZATION_RULES

# Aggregate all rules
ALL_RULES: List[SecurityRule] = (
    SECRETS_RULES
    + INJECTION_RULES
    + XSS_RULES
    + CRYPTO_RULES
    + CONFIG_RULES
    + DESERIALIZATION_RULES
)


def get_rules_by_category(category: str) -> List[SecurityRule]:
    """Get all rules for a specific category."""
    return [rule for rule in ALL_RULES if rule.category == category]


def get_rules_by_severity(min_severity: Severity) -> List[SecurityRule]:
    """Get all rules with severity >= min_severity."""
    return [rule for rule in ALL_RULES if rule.severity >= min_severity]


def get_enabled_rules() -> List[SecurityRule]:
    """Get all enabled rules."""
    return [rule for rule in ALL_RULES if rule.enabled]


def get_rule_by_id(rule_id: str) -> Optional[SecurityRule]:
    """Get a rule by its ID."""
    for rule in ALL_RULES:
        if rule.id == rule_id:
            return rule
    return None
