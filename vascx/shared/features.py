from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    pass


class FeatureSet:
    _registry = {}

    def __init__(self, name: str, features: Dict[str, Feature]):
        self.name = name
        self.features = features
        self.__class__._register_instance(self)

    def items(self):
        return self.features.items()

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
    def compute(self, parameters: List[Tuple[Any]]):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, parameters: List[Tuple[Any]]):
        """Generate plots and/or written explanation about the computation of these features."""
        pass
