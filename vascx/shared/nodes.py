from __future__ import annotations

from typing import TYPE_CHECKING, List

from rtnls_enface.base import Point

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


class Crossing(Node):
    def __init(self, position: Point, over: Segment, under: Segment):
        super().__init__(position)
        self.over = over
        self.under = under
