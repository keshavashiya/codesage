"""Feature flag management utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from codesage.utils.config import Config, FeaturesConfig


class FeatureFlags:
    """Simple feature flag manager backed by Config."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def list(self) -> Dict[str, bool]:
        """List all feature flags and their values."""
        return {k: bool(v) for k, v in asdict(self._config.features).items()}

    def is_enabled(self, feature: str) -> bool:
        """Check whether a feature is enabled."""
        return bool(getattr(self._config.features, feature, False))

    def enable(self, feature: str) -> None:
        """Enable a feature flag and persist config."""
        if hasattr(self._config.features, feature):
            setattr(self._config.features, feature, True)
            self._config.save()

    def disable(self, feature: str) -> None:
        """Disable a feature flag and persist config."""
        if hasattr(self._config.features, feature):
            setattr(self._config.features, feature, False)
            self._config.save()

    def reset(self) -> None:
        """Reset feature flags to defaults and persist config."""
        self._config.features = FeaturesConfig()
        self._config.save()
