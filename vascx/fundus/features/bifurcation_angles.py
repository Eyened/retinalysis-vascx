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

    # def plot(self, ax, layer: VesselTreeLayer, **kwargs):
    #     ax = layer.plot(
    #         ax=ax,
    #         segments=True,
    #     )

    #     bifurcations = layer.bifurcations

    #     for bif in bifurcations:
    #         if bif.outgoing[0].length < self.delta:
    #             continue
    #         if bif.outgoing[1].length < self.delta:
    #             continue

    #         to1 = Point(*bif.outgoing[0].spline.get_point_pixels(self.delta))
    #         to2 = Point(*bif.outgoing[1].spline.get_point_pixels(self.delta))

    #         line1 = Line(bif.position, to1)
    #         line2 = Line(bif.position, to2)

    #         angle = line1.counterclockwise_angle_to(line2)
    #         if angle > 180:
    #             line1, line2 = line2, line1

    #         angle = line1.angle_to(line2)
    #         if angle > self.max_angle:
    #             continue

    #         radius = line1.length

    #         arc = patches.Arc(
    #             bif.position.tuple_xy,
    #             2 * radius,
    #             2 * radius,
    #             angle=0,
    #             theta1=line2.orientation(),
    #             theta2=line1.orientation(),
    #             color="white",
    #             linewidth=0.5,
    #         )

    #         line1.plot(ax, color="white", linewidth=0.5)
    #         line2.plot(ax, color="white", linewidth=0.5)
    #         ax.add_patch(arc)

    #         ax.text(
    #             bif.position.x + 5,
    #             bif.position.y,
    #             # f"{line1.orientation():.1f} - {line2.orientation():.1f} [{line1.counterclockwise_angle_to(line2):.1f}]",
    #             f"{angle:.1f}",
    #             fontsize=3.6,
    #             color="white",
    #         )
    #     return ax



    def plot(self, ax, layer: VesselTreeLayer, all_three_angles=False, **kwargs):
        # Plot the vessel tree layer
        ax = layer.plot(
            ax=ax,
            segments=True,
        )

        # Retrieve bifurcations from the layer
        bifurcations = layer.bifurcations

        for bif in bifurcations:
            # Skip bifurcations with short outgoing segments
            if bif.outgoing[0].length < self.delta:
                continue
            if bif.outgoing[1].length < self.delta:
                continue

            # Calculate points and lines for outgoing segments
            to1 = Point(*bif.outgoing[0].spline.get_point_pixels(self.delta))
            to2 = Point(*bif.outgoing[1].spline.get_point_pixels(self.delta))

            line1 = Line(bif.position, to1)
            line2 = Line(bif.position, to2)

            # Calculate and adjust the angle between outgoing segments
            angle = line1.counterclockwise_angle_to(line2)
            if angle > 180:
                line1, line2 = line2, line1

            angle = line1.angle_to(line2)
            if angle > self.max_angle:
                continue

            # Draw the arc and lines for outgoing segments
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

            # Add text annotation for the angle between outgoing segments
            ax.text(
                bif.position.x + 5,
                bif.position.y,
                f"{angle:.1f}",
                fontsize=3.6,
                color="white",
            )

            # If all_angles is True, compute and display angles with the incoming vessel
            if all_three_angles:
                from_incoming = Point(*bif.incoming.spline.get_point_pixels(bif.incoming.spline.total_distance-self.delta))
                incoming_line = Line(bif.position, from_incoming)

                # Calculate angles between incoming and outgoing segments
                angle_incoming_to1 = incoming_line.angle_to(line1)
                angle_incoming_to2 = incoming_line.angle_to(line2)

                # Draw lines for incoming segment
                incoming_line.plot(ax, color="yellow", linewidth=0.5)

                # Add text annotations for the angles between incoming and outgoing segments
                ax.text(
                    bif.position.x - 5,
                    bif.position.y - 5,
                    f"{angle_incoming_to1:.1f}",
                    fontsize=3.6,
                    color="yellow",
                )
                ax.text(
                    bif.position.x - 5,
                    bif.position.y + 5,
                    f"{angle_incoming_to2:.1f}",
                    fontsize=3.6,
                    color="yellow",
                )

        return ax