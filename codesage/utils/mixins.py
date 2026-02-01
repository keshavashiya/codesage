"""Common mixins and interfaces for CodeSage components.

Provides reusable base classes for common patterns:
- Serializable: to_dict/from_dict serialization
- Clearable: clear() method for resettable stores
- MetricsProvider: get_metrics() for statistics
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar

T = TypeVar("T")


class Serializable(ABC):
    """Mixin for objects that can be serialized to/from dictionaries."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize object to dictionary.

        Returns:
            Dictionary representation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """Deserialize object from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Instance of the class.
        """
        pass


class Clearable(ABC):
    """Interface for stores that can be cleared."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        pass


class MetricsProvider(ABC):
    """Interface for components that provide metrics."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics/statistics from the component.

        Returns:
            Dictionary with component metrics.
        """
        pass


class Closeable(ABC):
    """Interface for components with resources that need to be closed."""

    @abstractmethod
    def close(self) -> None:
        """Close resources held by the component."""
        pass

    def __enter__(self) -> "Closeable":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
