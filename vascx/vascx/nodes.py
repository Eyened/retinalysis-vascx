from __future__ import annotations

from typing import TYPE_CHECKING, List

from rtnls_enface.base import Point

from vascx.segment import Segment

if TYPE_CHECKING:
    from vascx.layer import VesselLayer


class Node:
    def __init__(self, position: Point):
        self.position: Point = position
        self.layer: VesselLayer = None


class Endpoint(Node):
    pass


class Bifurcation(Node):
    def __init__(self, position: Point, incoming: Segment, outgoing: List[Segment]):
        super().__init__(position)
        self.incoming = incoming
        self.outgoing = outgoing


class Crossing(Node):
    def __init(self, position: Point, over: Segment, under: Segment):
        super().__init__(position)
        self.over = over
        self.under = under
