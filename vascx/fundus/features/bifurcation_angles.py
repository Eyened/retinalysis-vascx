from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib import patches

from rtnls_enface.base import Line, Point
from vascx.shared.aggregators import mean_median

from .base import LayerFeature

if TYPE_CHECKING:
    from rtnls_enface.grids.base import GridField
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class BifurcationAngles(LayerFeature):
    def __init__(self, delta: int = 20, grid_field: GridField = None, aggregator=mean_median,):
        """
        Calculation of bifurcation angles.

        Args:
            delta (int): The distance from the bifurcation point to calculate the angle.
            max_angle (str): Angles larger than this value will be ignored.

        """
        self.delta = delta
        self.max_angle = 160
        self.grid_field = grid_field
        self.aggregator = aggregator

    def _get_bifurcation_points(self, layer: VesselTreeLayer):
        bifurcations = layer.filter_bifurcations(self.grid_field)
        bifurcations = [bif for bif in bifurcations if bif.outgoing_min_length > self.delta]
        return bifurcations

    def compute(self, layer: VesselTreeLayer):
        bifurcations = self._get_bifurcation_points(layer)
        return self.aggregator([bif.angle(self.delta) for bif in bifurcations])

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
            line1, line2 = bif.lines(self.delta)
            angle = bif.angle(self.delta)

            if angle > self.max_angle:
                continue

            radius = line1.length

            arc = patches.Arc(
                bif.position.tuple_xy,
                2 * radius,
                2 * radius,
                angle=0,
                theta1=line2.orientation(),
                theta2=line1.orientation(),
                color="white",
                linewidth=0.5,
            )

            line1.plot(ax, color="white", linewidth=0.5)
            line2.plot(ax, color="white", linewidth=0.5)
            ax.add_patch(arc)

            ax.text(
                bif.position.x + 5,
                bif.position.y,
                # f"{line1.orientation():.1f} - {line2.orientation():.1f} [{line1.counterclockwise_angle_to(line2):.1f}]",
                f"{angle:.1f}",
                fontsize=3.6,
                color="white",
            )
        return ax
