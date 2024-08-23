from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class BifurcationAngles(LayerFeature):
    def compute(self, layer: VesselTreeLayer):
        bifurcations = self.get_bifurcation_points(layer)
        return len(bifurcations)

    def plot(self, layer: VesselTreeLayer, **kwargs):
        fig, ax = layer.plot(
            segments=True,
        )

        bifurcations = layer.bifurcations

        bifurcations = self.get_bifurcation_points(layer)
        for p in bifurcations:
            ax.scatter(*p.tuple_xy, s=1, color="g")
