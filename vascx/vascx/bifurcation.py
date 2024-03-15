from __future__ import annotations


from typing import List

from vascx.segment import Segment


class Bifurcation:
    def __init__(self, incoming: List[Segment], outgoing: List[Segment]):
        self.incoming = incoming
        self.outgoing = outgoing
