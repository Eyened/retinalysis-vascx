from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vascx.faz.features.base import FAZLayerFeature

from rtnls_enface.base import Point

if TYPE_CHECKING:
    from vascx.faz.layer import FAZLayer


class BifurcationCount(FAZLayerFeature):
    def _get_bifurcation_points(self, layer: FAZLayer):
        bifurcations = []

        digraph = layer.digraph
        for node in digraph.nodes():
            if digraph.degree(node) >= 3:
                bifurcations.append(Point(*digraph.nodes[node]["o"]))

        return bifurcations

    def compute(self, layer: FAZLayer):
        return len(self._get_bifurcation_points(layer))

    def _plot(self, ax, layer: FAZLayer, **kwargs):
        _, ax = layer.plot(ax=ax, mask=True)

        bifurcations = self._get_bifurcation_points(layer)
        for p in bifurcations:
            ax.scatter(*p.tuple_xy, s=2, color="g", marker="x")
