from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from vascx.shared.features import Feature

if TYPE_CHECKING:
    from vascx.faz.layer import FAZLayer
    from matplotlib.axes import Axes


class FAZLayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: FAZLayer):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def _plot(self, ax: 'Axes', layer: FAZLayer, **kwargs) -> 'Axes':
        """Subclass draws onto ax for the given VesselTreeLayer and returns ax."""

    def plot(self,  ax: 'Axes', layer: FAZLayer, **kwargs):
        return super().plot(ax=ax, layer=layer, **kwargs)
