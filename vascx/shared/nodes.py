from __future__ import annotations

from typing import TYPE_CHECKING, List

from rtnls_enface.base import Line, Point

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer
    from vascx.shared.segment import Segment


class Node:
    def __init__(self, position: Point, node: int = None):
        self.position: Point = position
        self.node = node  # networkx node id
        self.layer: VesselTreeLayer = None


class Endpoint(Node):
    pass


class Bifurcation(Node):
    def __init__(
        self,
        position: Point,
        incoming: Segment,
        outgoing: List[Segment],
        node: int = None,
    ):
        super().__init__(position, node=node)
        self.incoming = incoming
        self.outgoing = outgoing

    @property
    def outgoing_min_length(self):
        return min(self.outgoing[0].length, self.outgoing[1].length)
    
    def lines(self, delta: float, spline_error_fraction: float = 0.05):
        to1 = Point(
            *self.outgoing[0]
            .get_spline(error_fraction=spline_error_fraction)
            .get_point_pixels(delta)
        )
        to2 = Point(
            *self.outgoing[1]
            .get_spline(error_fraction=spline_error_fraction)
            .get_point_pixels(delta)
        )

        line1 = Line(self.position, to1)
        line2 = Line(self.position, to2)

        angle = line1.counterclockwise_angle_to(line2)

        if angle > 180:
            line1, line2 = line2, line1

        return line1, line2
    
    def angle(self, delta: float, spline_error_fraction: float = 0.05):
        line1, line2 = self.lines(delta, spline_error_fraction=spline_error_fraction)
        return line1.angle_to(line2)



class Crossing(Node):
    def __init(self, position: Point, over: Segment, under: Segment):
        super().__init__(position)
        self.over = over
        self.under = under
