from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from vascx.layer import VesselLayer


class FeatureSet:
    def __init__(self, features):
        self.features = features

    def items(self):
        return self.features.items()


class Feature(ABC):
    @abstractmethod
    def compute(self, parameters: List[Tuple[Any]]):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, parameters: List[Tuple[Any]]):
        """Generate plots and/or written explanation about the computation of these features."""
        pass


class LayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: VesselLayer):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, layer: VesselLayer, **kwargs):
        """Generate plots and/or written explanation about the computation of these features."""
        pass
