from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rtnls_enface.base import Point

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.retina import VesselLayer


@dataclass
class BifurcationCount(LayerFeature):
    def get_bifurcation_points(self, layer: VesselLayer):
        bifurcations = [
            Point(*layer.graph.nodes[n]["o"])
            for n in layer.graph.nodes()
            if layer.graph.degree(n) > 1
        ]
        return bifurcations

    def compute(self, layer: VesselLayer):
        bifurcations = self.get_bifurcation_points(layer)
        return len(bifurcations)

    def plot(self, layer: VesselLayer, **kwargs):
        fig, ax = layer.vessels.plot(
            cmap="tab20",
            min_numpoints=0,
            **kwargs,
        )

        bifurcations = self.get_bifurcation_points(layer)
        for p in bifurcations:
            ax.scatter(*p.tuple_xy, s=1)
