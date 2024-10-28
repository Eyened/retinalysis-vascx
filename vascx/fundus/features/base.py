from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from vascx.shared.features import Feature

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.fundus.vessels_layer import FundusVesselsLayer


class RetinaFeature(Feature):
    @abstractmethod
    def compute(self, retina: Retina):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, retina: Retina, **kwargs):
        """Generate plots and/or written explanation about the computation of these features."""
        pass

class LayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: VesselTreeLayer):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, layer: VesselTreeLayer, **kwargs):
        """Generate plots and/or written explanation about the computation of these features."""
        pass


class VesselsLayerFeature(Feature):
    @abstractmethod
    def compute(self, layer: FundusVesselsLayer):
        """Compute the feature given its parameters.
        Subclasses are free to define the type and semantics of the parameters.
        """
        pass

    @abstractmethod
    def plot(self, layer: FundusVesselsLayer, **kwargs):
        """Generate plots and/or written explanation about the computation of these features."""
        pass
