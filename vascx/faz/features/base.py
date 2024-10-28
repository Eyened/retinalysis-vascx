from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from vascx.shared.features import Feature

if TYPE_CHECKING:
    from vascx.faz.layer import FAZLayer


class FAZLayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: FAZLayer):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, layer: FAZLayer, **kwargs):
        """Generate plots and/or written explanation about the computation of these features."""
        pass
