from __future__ import annotations

from typing import TYPE_CHECKING

from rtnls_enface.types import TuplePoint

if TYPE_CHECKING:
    from vascx.layer import VesselLayer


class Node:
    def __init__(self, position: TuplePoint):
        self.position: TuplePoint = position
        self.layer: VesselLayer = None


class Endpoint(Node):
    pass


class Bifurcation(Node):
    pass


class Crossing(Node):
    pass
