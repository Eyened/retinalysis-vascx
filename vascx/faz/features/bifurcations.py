from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vascx.faz.features.base import FazLayerFeature

from rtnls_enface.base import Point

if TYPE_CHECKING:
    from vascx.faz.layer import FazLayer


@dataclass
class BifurcationCount(FazLayerFeature):
    def _get_bifurcation_points(self, layer: FazLayer):
        bifurcations = []

        digraph = layer.digraph
        for node in digraph.nodes():
            if digraph.degree(node) >= 3:
                bifurcations.append(Point(*digraph.nodes[node]["o"]))

        return bifurcations

    def compute(self, layer: FazLayer):
        return len(self._get_bifurcation_points(layer))

    def plot(self, layer: FazLayer, **kwargs):
        fig, ax = layer.plot(mask=True)

        bifurcations = self._get_bifurcation_points(layer)
        for p in bifurcations:
            ax.scatter(*p.tuple_xy, s=2, color="g", marker="x")
