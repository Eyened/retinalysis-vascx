from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class BifurcationCount(LayerFeature):
    def __init__(self, grid_field: GridField = None):
        """
        Calculation of the number of bifurcation points.

        """
        self.grid_field = grid_field

    def _get_bifurcation_points(self, layer: VesselTreeLayer):
        return layer.filter_bifurcations(self.grid_field)

    def compute(self, layer: VesselTreeLayer):
        bifurcations = self._get_bifurcation_points(layer)
        return len(bifurcations)

    def plot(self, ax, layer: VesselTreeLayer, **kwargs):
        ax = layer.plot(
            ax=ax,
            segments=True,
        )

        bifurcations = self._get_bifurcation_points(layer)

        # plot ETDRS region
        if self.grid_field is not None:
            layer.retina.grids[self.grid_field.grid()].plot(ax, self.grid_field)
        
        for bif in bifurcations:
            ax.scatter(*bif.position.tuple_xy, s=1, color="g")
        return ax
