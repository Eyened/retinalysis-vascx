from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib import patches

from rtnls_enface.base import Line, Point

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class BifurcationAngles(LayerFeature):
    def __init__(self, delta: int = 20):
        """
        Calculation of bifurcation angles.

        Args:
            delta (int): The distance from the bifurcation point to calculate the angle.
            max_angle (str): Angles larger than this value will be ignored.

        """
        self.delta = delta
        self.max_angle = 160

    def compute(self, layer: VesselTreeLayer):
        bifurcations = self.get_bifurcation_points(layer)
        return len(bifurcations)

    def plot(self, ax, layer: VesselTreeLayer, **kwargs):
        ax = layer.plot(
            ax=ax,
            segments=True,
        )

        bifurcations = layer.bifurcations

        for bif in bifurcations:
            if bif.outgoing[0].length < self.delta:
                continue
            if bif.outgoing[1].length < self.delta:
                continue

            to1 = Point(*bif.outgoing[0].spline.get_point_pixels(self.delta))
            to2 = Point(*bif.outgoing[1].spline.get_point_pixels(self.delta))

            line1 = Line(bif.position, to1)
            line2 = Line(bif.position, to2)

            angle = line1.counterclockwise_angle_to(line2)
            if angle > 180:
                line1, line2 = line2, line1

            angle = line1.angle_to(line2)
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
