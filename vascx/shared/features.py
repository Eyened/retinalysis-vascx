from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, List

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class FeatureSet:
    _registry = {}

    def __init__(self, name: str, features: Iterable["Feature"]):
        self.name = name
        self.features = list(features)
        self.__class__._register_instance(self)

    def __iter__(self):
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    @classmethod
    def _register_instance(cls, instance: FeatureSet):
        if instance.name in cls._registry:
            raise ValueError(
                f"Attempt to register more than one FeatureSet instance with name '{instance.name}'. FeatureSet names must be unique."
            )
        cls._registry[instance.name] = instance

    @classmethod
    def get_by_name(cls, name):
        return cls._registry.get(name, None)


class Feature(ABC):
    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the feature value for provided domain arguments."""
        pass

    @abstractmethod
    def display_name(self, **kwargs) -> str:
        """Return the display name for the feature."""
        pass

    def name_tokens(self, **kwargs: Any) -> List[str]:
        """Return semantic tokens used to build the canonical machine name."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement canonical naming tokens"
        )

    def canonical_name(self, **kwargs: Any) -> str:
        """Return the canonical machine-readable feature name."""
        return "_".join(token for token in self.name_tokens(**kwargs) if token)

    @abstractmethod
    def _plot(self, ax: 'Axes', layer: Any, **kwargs: Any) -> 'Axes':
        """Subclass draws onto ax for the given layer and returns ax."""

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "N/A"
        if isinstance(value, np.generic):
            display_value = value.item()
        else:
            display_value = value
            
        if isinstance(display_value, (int, float)) and not isinstance(display_value, bool):
            return f"{display_value:.3g}"
        else:
            return str(display_value)

    def plot(self, ax: 'Axes', layer: Any, **kwargs: Any) -> 'Axes':
        """Compute value, delegate drawing to _plot, annotate value at upper-left, return ax."""
        value = self.compute(layer, **kwargs)
        ax = self._plot(ax, layer, **kwargs)

        # Display values starting from top-left, going down
        y_start = 0.99
        formatted_value = self._format_value(value)
        ax.text(0.01, y_start, formatted_value, transform=ax.transAxes, ha='left', va='top', color='white', fontsize=8)
            
        return ax
